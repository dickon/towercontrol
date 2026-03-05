"""
VideoRecorder — continuously encode game captures into rolling H.264 MP4 segments.

Keeps up to ``max_total_bytes`` of history by deleting the oldest segments.
Each captured frame becomes one second of 1-fps video; the real wall-clock
timestamp for every frame is stored in ``video/segments.json`` so the browser
can seek to an exact moment in history regardless of tick-rate variations.

Public API
----------
    recorder = VideoRecorder(Path("video"))
    recorder.start(width, height)          # begin encoding
    recorder.push_frame(img, timestamp)    # queue a PIL Image
    recorder.resize(width, height)         # call if capture dims change
    recorder.stop()                        # flush & finalize
"""

import json
import logging
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from PIL import Image

log = logging.getLogger(__name__)

_SEGMENTS_JSON   = "segments.json"
_FPS             = 5               # frames per second in the recorded video
_DEFAULT_CRF     = 28
_DEFAULT_PRESET  = "fast"
_DEFAULT_THREADS = 2
_DEFAULT_MAX_BYTES     = 50 * 1024 ** 3   # 50 GB
_DEFAULT_SEGMENT_SEC   = 3600             # 1-hour segments
_DEFAULT_FFMPEG        = "ffmpeg.exe"
_FRAME_QUEUE_DEPTH     = 20


@dataclass
class SegmentMeta:
    """Metadata for one recorded video segment."""
    filename:    str
    start_ts:    float          # Unix epoch of first frame
    end_ts:      float          # Unix epoch of last frame
    frame_count: int
    size_bytes:  int
    wall_times:  List[float] = field(default_factory=list)  # real ts per frame


class VideoRecorder:
    """
    Thread-safe video recorder that pipes raw RGB frames into FFmpeg.

    One background thread owns the FFmpeg subprocess and all file I/O.
    The main thread (and automation loop) only calls ``push_frame()``.
    """

    def __init__(
        self,
        output_dir: Path,
        max_total_bytes: int  = _DEFAULT_MAX_BYTES,
        segment_duration_sec: float = _DEFAULT_SEGMENT_SEC,
        ffmpeg_path: str      = _DEFAULT_FFMPEG,
        crf: int              = _DEFAULT_CRF,
        preset: str           = _DEFAULT_PRESET,
        threads: int          = _DEFAULT_THREADS,
    ):
        self._dir        = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_bytes  = max_total_bytes
        self._seg_dur    = segment_duration_sec
        self._ffmpeg     = str(ffmpeg_path)
        self._crf        = crf
        self._preset     = preset
        self._threads    = threads

        # Current dimensions (set by start() / resize())
        self._width  = 0
        self._height = 0

        # FFmpeg subprocess
        self._proc: Optional[subprocess.Popen] = None

        # Current segment state
        self._seg_path:       Optional[Path]  = None
        self._seg_start_ts:   float           = 0.0
        self._seg_wall_times: List[float]     = []

        # All completed segments (loaded from disk + recorded this session)
        self._segments: List[SegmentMeta] = []

        # Worker thread communication
        # Items are (PIL.Image, float) | ("RESIZE", (w, h)) | None (poison-pill)
        self._queue:   queue.Queue          = queue.Queue(maxsize=_FRAME_QUEUE_DEPTH)
        self._worker:  Optional[threading.Thread] = None
        self._running: bool                 = False
        self._lock                          = threading.Lock()

        self._load_segments()

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self, width: int, height: int) -> None:
        """Begin encoding.  Safe to call multiple times (e.g., after resize)."""
        with self._lock:
            self._width  = width
            self._height = height
            if not self._running:
                self._running = True
                self._worker = threading.Thread(
                    target=self._encode_loop, daemon=True, name="video-encoder"
                )
                self._worker.start()
                log.info("VideoRecorder started (%dx%d)", width, height)

    def push_frame(self, img: Image.Image, timestamp: float) -> None:
        """Queue a frame for encoding.  Silently drops the frame if queue is full."""
        if not self._running:
            return
        try:
            self._queue.put_nowait((img, timestamp))
        except queue.Full:
            log.debug("VideoRecorder: frame queue full — dropping frame")

    def resize(self, width: int, height: int) -> None:
        """Handle a capture-dimension change: restart FFmpeg with the new size."""
        with self._lock:
            if width == self._width and height == self._height:
                return
            log.info(
                "VideoRecorder: resize %dx%d → %dx%d",
                self._width, self._height, width, height,
            )
            self._width  = width
            self._height = height
        # Signal worker via queue (bypass the Full guard with block=True + short timeout)
        try:
            self._queue.put(("RESIZE", (width, height)), timeout=2)
        except queue.Full:
            log.warning("VideoRecorder: resize signal dropped (queue full)")

    def stop(self) -> None:
        """Gracefully stop encoding and finalize the current segment."""
        self._running = False
        try:
            self._queue.put(None, timeout=5)   # poison pill
        except queue.Full:
            pass
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=30)
        log.info("VideoRecorder stopped")

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def segments(self) -> List[SegmentMeta]:
        """Completed segment metadata (snapshot; does not include active segment)."""
        return list(self._segments)

    @property
    def active_segment(self) -> Optional[SegmentMeta]:
        """
        Partial metadata for the currently-recording segment, or None.
        The wall_times list is a snapshot; size_bytes reflects the file on disk.
        """
        seg_path = self._seg_path
        wall_times = list(self._seg_wall_times)
        if seg_path and seg_path.exists() and wall_times:
            try:
                size = seg_path.stat().st_size
            except OSError:
                size = 0
            return SegmentMeta(
                filename    = seg_path.name,
                start_ts    = self._seg_start_ts,
                end_ts      = wall_times[-1],
                frame_count = len(wall_times),
                size_bytes  = size,
                wall_times  = wall_times,
            )
        return None

    # ── Worker thread ──────────────────────────────────────────────────────

    def _encode_loop(self) -> None:
        """Background thread: drain the frame queue into the active FFmpeg process."""
        while True:
            item = self._queue.get()

            # Poison pill → shut down
            if item is None:
                break

            img, ts = item

            # Resize signal: finalize current segment, restart with new dims
            if img == "RESIZE":
                new_w, new_h = ts
                self._finalize_segment()
                self._proc = None
                with self._lock:
                    self._width  = new_w
                    self._height = new_h
                continue

            # Lazily start the first segment
            if not self._proc_alive():
                self._start_segment(img.width, img.height)
                if self._proc is None:
                    continue  # FFmpeg failed to start; skip frame

            # Rotate segment when duration exceeded
            if self._seg_start_ts and (ts - self._seg_start_ts) >= self._seg_dur:
                self._finalize_segment()
                self._start_segment(img.width, img.height)
                if self._proc is None:
                    continue

            # Drop frame if image size changed unexpectedly (avoid corrupt stream)
            if img.width != self._width or img.height != self._height:
                self.resize(img.width, img.height)
                continue

            # Write raw RGB frame bytes to FFmpeg stdin
            try:
                raw = img.convert("RGB").tobytes()
                self._proc.stdin.write(raw)
                self._proc.stdin.flush()
                self._seg_wall_times.append(ts)
            except (BrokenPipeError, OSError) as exc:
                log.warning("VideoRecorder: pipe error (%s) — restarting FFmpeg", exc)
                self._proc = None
                self._start_segment(img.width, img.height)
                if self._proc is None:
                    continue
                try:
                    self._proc.stdin.write(img.convert("RGB").tobytes())
                    self._proc.stdin.flush()
                    self._seg_wall_times.append(ts)
                except Exception as exc2:
                    log.error("VideoRecorder: retry failed: %s", exc2)

    def _proc_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ── FFmpeg subprocess management ────────────────────────────────────────

    def _start_segment(self, width: int, height: int) -> None:
        """Spawn a new FFmpeg process writing to a fresh segment file."""
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        self._seg_path       = self._dir / f"seg_{ts_str}.mp4"
        self._seg_start_ts   = time.time()
        self._seg_wall_times = []

        cmd = [
            self._ffmpeg,
            "-loglevel", "warning",
            # Input: raw RGB frames from stdin
            "-f",          "rawvideo",
            "-pix_fmt",    "rgb24",
            "-video_size", f"{width}x{height}",
            "-r",          str(_FPS),
            "-i",          "pipe:0",
            # Output: H.264 baseline-compatible (yuv420p), fragmented MP4
            "-c:v",        "libx264",
            "-pix_fmt",    "yuv420p",           # required for browser/VLC compat
            "-vf",         "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # ensure even dims
            "-g",          str(_FPS * 30),      # keyframe every 30 s → fragment flush
            "-crf",        str(self._crf),
            "-preset",     self._preset,
            "-threads",    str(self._threads),
            "-movflags",   "+frag_keyframe+empty_moov+default_base_moof",
            "-f",          "mp4",
            str(self._seg_path),
        ]

        log.info(
            "VideoRecorder: starting segment %s (%dx%d)",
            self._seg_path.name, width, height,
        )
        try:
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self._proc = subprocess.Popen(
                cmd,
                stdin  = subprocess.PIPE,
                stdout = subprocess.DEVNULL,
                stderr = subprocess.PIPE,
                creationflags = creationflags,
            )
        except FileNotFoundError:
            log.error("VideoRecorder: ffmpeg not found at '%s'", self._ffmpeg)
            self._proc = None
        except Exception as exc:
            log.error("VideoRecorder: failed to start FFmpeg: %s", exc)
            self._proc = None

    def _finalize_segment(self) -> None:
        """Close FFmpeg stdin, wait for it to exit, then record metadata."""
        proc = self._proc
        self._proc = None
        if proc is None:
            return

        # Close stdin to signal EOF
        try:
            proc.stdin.close()
        except Exception:
            pass

        try:
            _, stderr = proc.communicate(timeout=30)
            if proc.returncode not in (None, 0):
                log.warning(
                    "VideoRecorder: FFmpeg exited with code %d: %s",
                    proc.returncode,
                    stderr.decode(errors="replace")[:500],
                )
        except subprocess.TimeoutExpired:
            proc.kill()
            log.warning("VideoRecorder: FFmpeg did not exit in 30 s — killed")
        except Exception as exc:
            log.warning("VideoRecorder: finalize error: %s", exc)

        seg_path   = self._seg_path
        wall_times = self._seg_wall_times
        self._seg_path       = None
        self._seg_start_ts   = 0.0
        self._seg_wall_times = []

        if seg_path and seg_path.exists() and wall_times:
            try:
                size = seg_path.stat().st_size
            except OSError:
                size = 0
            meta = SegmentMeta(
                filename    = seg_path.name,
                start_ts    = wall_times[0],
                end_ts      = wall_times[-1],
                frame_count = len(wall_times),
                size_bytes  = size,
                wall_times  = wall_times,
            )
            self._segments.append(meta)
            self._save_segments()
            self._enforce_quota()
            log.info(
                "VideoRecorder: segment %s finalized — %d frames, %.1f MB",
                meta.filename, meta.frame_count, size / 1e6,
            )

    # ── Metadata persistence ────────────────────────────────────────────────

    def _load_segments(self) -> None:
        meta_path = self._dir / _SEGMENTS_JSON
        if not meta_path.exists():
            self._segments = []
            self._scan_existing()
            return
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            self._segments = [
                SegmentMeta(
                    filename    = s["filename"],
                    start_ts    = s["start_ts"],
                    end_ts      = s["end_ts"],
                    frame_count = s["frame_count"],
                    size_bytes  = s.get("size_bytes", 0),
                    wall_times  = s.get("wall_times", []),
                )
                for s in data
                if (self._dir / s["filename"]).exists()
            ]
            log.info(
                "VideoRecorder: loaded %d completed segments", len(self._segments)
            )
        except Exception as exc:
            log.warning("VideoRecorder: failed to load segments.json: %s", exc)
            self._segments = []
            self._scan_existing()

    def _scan_existing(self) -> None:
        """Build minimal metadata from existing *.mp4 files (no wall_times)."""
        for p in sorted(self._dir.glob("seg_*.mp4")):
            stat = p.stat()
            self._segments.append(SegmentMeta(
                filename    = p.name,
                start_ts    = stat.st_mtime,
                end_ts      = stat.st_mtime,
                frame_count = 0,
                size_bytes  = stat.st_size,
                wall_times  = [],
            ))
        if self._segments:
            self._save_segments()

    def _save_segments(self) -> None:
        meta_path = self._dir / _SEGMENTS_JSON
        try:
            data = [
                {
                    "filename":    s.filename,
                    "start_ts":    s.start_ts,
                    "end_ts":      s.end_ts,
                    "frame_count": s.frame_count,
                    "size_bytes":  s.size_bytes,
                    "wall_times":  s.wall_times,
                }
                for s in self._segments
            ]
            meta_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            log.warning("VideoRecorder: failed to save segments.json: %s", exc)

    def _enforce_quota(self) -> None:
        """Delete oldest segments until total on-disk size ≤ max_total_bytes."""
        total = sum(s.size_bytes for s in self._segments)
        seg = self._seg_path
        if seg and seg.exists():
            try:
                total += seg.stat().st_size
            except OSError:
                pass

        while total > self._max_bytes and len(self._segments) > 1:
            oldest = self._segments.pop(0)
            p = self._dir / oldest.filename
            if p.exists():
                try:
                    p.unlink()
                    log.info(
                        "VideoRecorder: deleted %s (quota enforcement)", oldest.filename
                    )
                except Exception as exc:
                    log.warning(
                        "VideoRecorder: could not delete %s: %s", oldest.filename, exc
                    )
            total -= oldest.size_bytes

        self._save_segments()
