/*  TowerControl – dashboard client */

const WS_URL = `ws://${location.host}/ws`;
const API    = (path) => `${location.origin}/api/${path}`;

let ws      = null;
let canvas, ctx, overlay, octx;
let imgNatW = 0, imgNatH = 0;   // natural size of the last received image
let _clickCrosshair = null;     // {fx, fy} of pinned action crosshair, or null
let _timelineChart  = null;     // Chart.js instance for the timeline widget
let _liveImageSrc      = null;     // data-URL of the last live frame (for hover restore)
let _hoverFile         = null;     // filename currently previewed on timeline hover
let _hoverTimeMs       = null;     // chart time (ms) at current timeline hover position
let _timelineHovering  = false;    // true while mouse is over the timeline canvas
let _timelineActions   = [];       // [{t, fx, fy, reason}] from last timeline fetch
let _timelineSeq       = null;     // last timeline_seq seen via WebSocket
let _clickInjectionEnabled = false; // toggled by chkClickInject; off by default

// ── Timeline overview (Ableton-style range navigator) ────────────────────────
let _overviewDataMinT   = 0;      // ms – earliest point in full dataset
let _overviewDataMaxT   = 0;      // ms – latest point in full dataset
let _viewMinT           = 0;      // ms – left edge of zoom window (0 = uninitialised)
let _viewMaxT           = 0;      // ms – right edge of zoom window
let _overviewWavePts    = [];     // wave series for minimap drawing
let _overviewActTs      = [];     // action timestamps (ms) for tick marks
let _ovDrag             = null;   // active drag state
let _overviewInitialized = false; // true after first data paint
const _OV_EDGE_PX = 20;          // px from window edge that counts as resize handle (wider for touch)

// ── Bootstrap ───────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  canvas  = document.getElementById("gameCanvas");
  ctx     = canvas.getContext("2d");
  overlay = document.getElementById("overlayCanvas");
  octx    = overlay.getContext("2d");

  canvas.addEventListener("click",      onCanvasClick);
  canvas.addEventListener("mousemove",  onCanvasMouseMove);
  canvas.addEventListener("mouseleave", onCanvasMouseLeave);

  connectWS();
  connectLog();
  loadParamSchema();
  fetchTimeline();
  setInterval(fetchTimeline, 30000); // fallback poll; real-time updates come via WebSocket
});

// ── WebSocket ───────────────────────────────────────────────────────────

function connectWS() {
  ws = new WebSocket(WS_URL);
  ws.onmessage = (evt) => {
    try { handleState(JSON.parse(evt.data)); }
    catch (e) { console.error("ws parse error", e); }
  };
  ws.onclose = () => {
    updateBadge("offline");
    setTimeout(connectWS, 3000);
  };
  ws.onerror = () => ws.close();
}

// ── SSE log stream ──────────────────────────────────────────────────────

const LOG_MAX_LINES = 2000;

function connectLog() {
  const statusEl = document.getElementById("logStatus");
  if (!statusEl) return;

  const es = new EventSource("/log/stream");

  es.onopen = () => {
    statusEl.textContent = "connected";
    statusEl.className   = "badge bg-success";
  };

  es.onmessage = (evt) => {
    try {
      const line = JSON.parse(evt.data);
      appendLogLine(line);
    } catch (_) { /* keepalive comment lines are not valid JSON */ }
  };

  es.onerror = () => {
    statusEl.textContent = "disconnected";
    statusEl.className   = "badge bg-danger";
    es.close();
    setTimeout(connectLog, 5000);
  };
}

function appendLogLine(line) {
  const pre  = document.getElementById("logContent");
  const pane = document.getElementById("logPane");
  if (!pre || !pane) return;

  pre.insertAdjacentHTML("beforeend", colourLogLine(line) + "\n");

  // Trim DOM to avoid unbounded growth
  while (pre.childNodes.length > LOG_MAX_LINES * 2) pre.removeChild(pre.firstChild);

  const autoScroll = document.getElementById("logAutoScroll");
  if (autoScroll && autoScroll.checked) pane.scrollTop = pane.scrollHeight;
}

function colourLogLine(line) {
  const safe = esc(line);
  if (/\bERROR\b|\bCRITICAL\b/i.test(line)) return `<span style="color:#f66">${safe}</span>`;
  if (/\bWARNING\b/i.test(line))             return `<span style="color:#fc3">${safe}</span>`;
  if (/\bINFO\b/i.test(line))               return `<span style="color:#8cf">${safe}</span>`;
  return `<span style="color:#ccc">${safe}</span>`;
}

function clearLog() {
  const pre = document.getElementById("logContent");
  if (pre) pre.innerHTML = "";
}

// ── Canvas coordinate inspector ─────────────────────────────────────────

function canvasFrac(evt) {
  const rect = canvas.getBoundingClientRect();
  return {
    fx: (evt.clientX - rect.left) / rect.width,
    fy: (evt.clientY - rect.top)  / rect.height,
  };
}

function onCanvasMouseMove(evt) {
  if (!imgNatW) return;
  const { fx, fy } = canvasFrac(evt);
  const el = document.getElementById("coordDisplay");
  if (el) el.textContent = `fx=${fx.toFixed(4)}  fy=${fy.toFixed(4)}`;
}

function onCanvasMouseLeave() {
  const el = document.getElementById("coordDisplay");
  if (el) el.textContent = "";
}

function onCanvasClick(evt) {
  if (!imgNatW) return;
  if (!_clickInjectionEnabled) return;  // toggle must be on
  const { fx, fy } = canvasFrac(evt);
  console.log("Injecting click at", fx.toFixed(4), fy.toFixed(4));
  api("click", { fx, fy });
}

function toggleClickInjection(chk) {
  _clickInjectionEnabled = chk.checked;
  const hint = document.getElementById("clickInjectHint");
  if (hint) hint.textContent = _clickInjectionEnabled
    ? "Click injection ON — click image to send a click."
    : "Click injection OFF. Hover to inspect coordinates.";
}

// ── State update ────────────────────────────────────────────────────────

function handleState(s) {
  updateBadge(s.bot_status || "offline");

  // Image
  if (s.image) {
    const src = "data:image/jpeg;base64," + s.image;
    _liveImageSrc = src;          // keep a copy for timeline-hover restore
    // Only update the canvas when not previewing a capture hover
    if (!_hoverFile) {
      const img = new Image();
      img.onload = () => {
        imgNatW       = img.naturalWidth;
        imgNatH       = img.naturalHeight;
        canvas.width  = imgNatW;
        canvas.height = imgNatH;
        overlay.width  = imgNatW;
        overlay.height = imgNatH;
        ctx.drawImage(img, 0, 0);
        document.getElementById("noImage").style.display = "none";
      };
      img.src = src;
    }
  }

  // Window info
  if (s.window_rect) {
    const r = s.window_rect;
    document.getElementById("windowInfo").textContent =
      `${r.width}×${r.height} @ (${r.left},${r.top})`;
  }

  renderCtxTable(s.ctx_full || {});
  updateWatchdogUI(s.watchdog || null);

  // Resources
  const tbody = document.getElementById("resourcesTable");
  tbody.innerHTML = "";
  for (const [k, v] of Object.entries(s.resources || {})) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="text-muted">${esc(k)}</td><td>${esc(v)}</td>`;
    tbody.appendChild(tr);
  }

  renderOcrList(s.ocr_results || []);
  renderUpgradePriority(s.upgrade_priority || [], s.upgrade_state ?? 0);
  renderUpgradesSeen(s.upgrade_seen || {});

  // Action log
  const logEl = document.getElementById("actionLog");
  logEl.innerHTML = "";
  for (const a of (s.recent_actions || []).slice(-15).reverse()) {
    const li = document.createElement("li");
    li.className = "d-flex align-items-baseline gap-1 py-0";
    const t = new Date(a.time * 1000).toLocaleTimeString();
    const hasCoords = a.fx != null && a.fy != null;

    // Crosshair button (only for clicks with coords)
    if (hasCoords) {
      const btn = document.createElement("button");
      btn.className = "action-xhair-btn btn btn-sm p-0";
      btn.style.cssText = "font-size:0.7rem;line-height:1;min-width:1.3rem;color:#555;background:none;border:1px solid #333;border-radius:3px;flex-shrink:0";
      btn.title = `Show crosshair at (${a.fx}, ${a.fy})`;
      btn.textContent = "⊕";
      const fx = a.fx, fy = a.fy;
      btn.addEventListener("click", (e) => { e.stopPropagation(); toggleActionCrosshair(btn, fx, fy); });
      li.appendChild(btn);
    }

    const info = document.createElement("span");
    info.className = "flex-grow-1";
    const coordStr = hasCoords
      ? `<span class="text-muted" style="font-size:0.68rem"> (${a.fx},${a.fy}) px(${a.ax},${a.ay})</span>`
      : "";
    info.innerHTML =
      `<span class="text-muted">${t}</span> ` +
      `<span class="text-warning">${esc(a.type)}</span> ` +
      `<span>${esc(a.reason || "")}</span>` +
      coordStr;
    li.appendChild(info);
    logEl.appendChild(li);
  }

  updateParamValues(s.strategy_params || {});
  updatePauseCountdown((s.strategy_params || {}).input_pause_remaining ?? 0);
  renderPerkHistory(s.perk_selection_history || []);
  renderUpgradePurchaseHistory(s.upgrade_purchase_history || []);
  renderUpgradeAdvanceHistory(s.upgrade_advance_history || []);

  // Refresh timeline whenever the server reports new data (new wave, action, or capture file)
  if (s.timeline_seq != null && s.timeline_seq !== _timelineSeq) {
    _timelineSeq = s.timeline_seq;
    fetchTimeline();
  }
}

// ── Context table ────────────────────────────────────────────────────────

const CTX_LABELS = {
  status:                  "Status",
  running:                 "Running",
  input_enabled:           "Input enabled",
  ocr_time_s:              "OCR time (s)",
  screen:                  "Screen",
  tier:                    "Tier",
  wave:                    "Wave",
  error_count:             "Errors",
  battle_start_ago:        "Battle started (s ago)",
  wave_history_len:        "Wave history",
  action_history_len:      "Action history",
  upgrade_state:           "Upgrade state",
  upgrade_mode_seen:       "Upgrade mode",
  upgrade_scroll_dir:      "Scroll dir",
  upgrade_scroll_start:    "Scrolling (s)",
  recover_stage:           "Recover stage",
  no_perk_until:           "No-perk for (s)",
  upgrades_finished:       "Upgrades finished",
  last_upgrade_action_ago: "Last upgrade (s ago)",
  last_seen_upgrades_ago:  "Last saw upgrades (s ago)",
  window:                  "Window",
  ocr_engine:              "OCR engine",
  loop_tick:               "Loop tick (s)",
};

function renderCtxTable(ctx) {
  const tbody = document.getElementById("ctxTable");
  if (!tbody) return;
  tbody.innerHTML = "";
  for (const [k, label] of Object.entries(CTX_LABELS)) {
    if (!(k in ctx)) continue;
    const v   = ctx[k];
    const tr  = document.createElement("tr");
    const val = v === null || v === undefined ? "—" : String(v);
    let colour = "";
    if (k === "running")       colour = v ? "color:#6f6" : "color:#f66";
    if (k === "input_enabled") colour = v ? "color:#6f6" : "color:#fc3";
    if (k === "error_count" && v > 0) colour = "color:#f66";
    tr.innerHTML =
      `<td class="text-muted ps-2" style="width:55%">${esc(label)}</td>` +
      `<td class="pe-2" style="${colour}">${esc(val)}</td>`;
    tbody.appendChild(tr);
  }
}

// ── Pause input ──────────────────────────────────────────────────────────

function pauseInput() {
  const btn = document.getElementById("btnPauseInput");
  if (btn && btn.dataset.paused === "1") {
    // Cancel: re-enable input immediately
    api("resume").then(r => {
      if (r?.ok) updatePauseCountdown(0);
    });
  } else {
    api("pause_input").then(r => {
      if (r?.ok) updatePauseCountdown(r.remaining_s ?? 1200);
    });
  }
}

function updatePauseCountdown(remainingSecs) {
  const btn      = document.getElementById("btnPauseInput");
  const countEl  = document.getElementById("pauseCountdown");
  if (!btn) return;

  const secs = Math.max(0, Math.round(remainingSecs));
  if (secs > 0) {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    const display = `${m}:${String(s).padStart(2, "0")}`;
    if (countEl) countEl.textContent = `resuming in ${display}`;
    btn.textContent  = "\u23F9 Cancel Pause";
    btn.dataset.paused = "1";
    btn.classList.replace("btn-outline-warning", "btn-warning");
  } else {
    if (countEl) countEl.textContent = "";
    btn.textContent  = "\u23F8 Pause Input 20m";
    btn.dataset.paused = "0";
    btn.classList.replace("btn-warning", "btn-outline-warning");
  }
}

// ── Restart ──────────────────────────────────────────────────────────────

function confirmRestart() {
  if (!confirm("Reset all game state (wave, tier, upgrade progress) and continue running?")) return;
  api("restart").then(r => {
    if (!r?.ok) console.error("Restart failed", r);
  });
}

// ── Watchdog ─────────────────────────────────────────────────────────────

let _watchdogState = null;

function updateWatchdogUI(w) {
  if (!w) return;
  _watchdogState = w;

  const chkWd = document.getElementById("chkWatchdog");
  const chkGl = document.getElementById("chkGameLaunch");
  const statusEl = document.getElementById("watchdogStatus");
  if (chkWd) {
    chkWd.checked = !!w.enabled;
  }
  if (chkGl) {
    chkGl.checked = !!w.game_launch_enabled;
  }
  if (statusEl) {
    const parts = [];
    if (w.last_bs_restart_ago != null)
      parts.push("BS restart " + w.last_bs_restart_ago + "s ago");
    if (w.last_game_launch_ago != null)
      parts.push("game launch " + w.last_game_launch_ago + "s ago");
    if (w.last_game_ui_seen_ago != null)
      parts.push("game UI seen " + w.last_game_ui_seen_ago + "s ago");
    statusEl.textContent = parts.join(" · ");
  }
}

function toggleWatchdog() {
  const newVal = _watchdogState ? !_watchdogState.enabled : true;
  api("watchdog", { watchdog: newVal });
}

function toggleGameLaunch() {
  const newVal = _watchdogState ? !_watchdogState.game_launch_enabled : true;
  api("watchdog", { game_launch: newVal });
}

// ── Perk history ─────────────────────────────────────────────────────────

function renderPerkHistory(history) {
  const tbody = document.getElementById("perkHistoryTable");
  if (!tbody) return;
  tbody.innerHTML = "";
  for (const p of [...history].reverse()) {
    const tr = document.createElement("tr");
    const t  = new Date(p.timestamp * 1000).toLocaleTimeString();
    const label = p.text && p.text !== p.selected ? esc(p.text) : esc(p.selected);
    tr.innerHTML =
      `<td class="text-muted ps-2">${t}</td>` +
      `<td class="text-info">${esc(p.wave ?? "?")}</td>` +
      `<td>${label}</td>`;
    tbody.appendChild(tr);
  }
}

// ── Upgrade purchase history ──────────────────────────────────────────────

function fmtVal(v) {
  if (v == null) return "—";
  if (v >= 1e12) return (v / 1e12).toPrecision(4) + "T";
  if (v >= 1e9)  return (v / 1e9).toPrecision(4)  + "B";
  if (v >= 1e6)  return (v / 1e6).toPrecision(4)  + "M";
  if (v >= 1e3)  return (v / 1e3).toPrecision(4)  + "K";
  return String(+v.toPrecision(4));
}

function renderUpgradePurchaseHistory(history) {
  const tbody = document.getElementById("upgradePurchaseTable");
  if (!tbody) return;
  tbody.innerHTML = "";
  for (const p of [...history].reverse()) {
    const tr = document.createElement("tr");
    const t  = new Date(p.timestamp * 1000).toLocaleTimeString();
    const cost = p.cost != null ? p.cost.toLocaleString() : "—";
    const val  = fmtVal(p.current_value);
    let upgradeCell = esc(p.upgrade_name);
    if (p.button_img) {
      upgradeCell = `<img src="data:image/jpeg;base64,${p.button_img}" style="height:28px;vertical-align:middle;margin-right:4px;border-radius:3px;image-rendering:pixelated" title="Upgrade button image"> ` + upgradeCell;
    }
    tr.innerHTML =
      `<td class="text-muted ps-2">${t}</td>` +
      `<td class="text-info">${esc(p.wave ?? "?")}</td>` +
      `<td>${upgradeCell}</td>` +
      `<td class="text-end text-secondary">${esc(val)}</td>` +
      `<td class="text-end pe-2 text-warning">${esc(cost)}</td>`;
    tbody.appendChild(tr);
  }
}

// ── Upgrade advance history ───────────────────────────────────────────────

function renderUpgradeAdvanceHistory(history) {
  const tbody = document.getElementById("upgradeAdvanceTable");
  if (!tbody) return;
  tbody.innerHTML = "";
  for (const p of [...history].reverse()) {
    const tr = document.createElement("tr");
    const t  = new Date(p.timestamp * 1000).toLocaleTimeString();
    let fromCell = esc(p.from_upgrade);
    if (p.from_button_img) {
      fromCell = `<img src="data:image/jpeg;base64,${p.from_button_img}" style="height:28px;vertical-align:middle;margin-right:4px;border-radius:3px;image-rendering:pixelated" title="From upgrade button image"> ` + fromCell;
    }
    tr.innerHTML =
      `<td class="text-muted ps-2">${t}</td>` +
      `<td class="text-info">${esc(p.wave ?? "?")}</td>` +
      `<td class="text-secondary">${fromCell}</td>` +
      `<td class="text-success">${esc(p.to_upgrade)}</td>` +
      `<td class="text-muted">${esc(p.reason)}</td>`;
    tbody.appendChild(tr);
  }
}

// ── Upgrade priority ─────────────────────────────────────────────────────

function fmtCap(v) {
  if (v == null) return "";
  if (v >= 1e9) return (v / 1e9).toFixed(v % 1e9 === 0 ? 0 : 1) + "B";
  if (v >= 1e6) return (v / 1e6).toFixed(v % 1e6 === 0 ? 0 : 1) + "M";
  if (v >= 1e3) return (v / 1e3).toFixed(v % 1e3 === 0 ? 0 : 1) + "K";
  return String(v);
}

const TAB_COLOUR = { ATTACK: "#f77", DEFENSE: "#6cf", UTILITY: "#8f8" };

function renderUpgradePriority(list, currentState) {
  const tbody = document.getElementById("upgradePriorityTable");
  if (!tbody) return;
  tbody.innerHTML = "";
  list.forEach((item, i) => {
    const isActive = i === currentState;
    const isDone   = i < currentState;
    const tr = document.createElement("tr");
    tr.style.cursor = "pointer";
    if (isActive) tr.style.background = "#3a3000";
    else if (isDone) tr.style.opacity = "0.45";
    const tabCol = TAB_COLOUR[item.page] || "#aaa";
    tr.innerHTML =
      `<td class="ps-2 text-muted">${isActive ? "▶" : i}</td>` +
      `<td style="color:${tabCol}">${esc(item.page)}</td>` +
      `<td>${esc(item.label)}${item.needs_scroll ? ' <span class="text-muted" title="needs scroll">↕</span>' : ""}</td>` +
      `<td class="text-end pe-2 text-muted">${esc(fmtCap(item.cost_threshold))}</td>`;
    tr.addEventListener("click", () => {
      api("upgrade_state", { index: i }).catch(e => console.error("upgrade_state error", e));
    });
    tbody.appendChild(tr);
  });
}

// ── Detected upgrades ────────────────────────────────────────────────────

function fmtNum(v) {
  if (v == null) return "—";
  const abs = Math.abs(v);
  if (abs >= 1e12) return (v / 1e12).toFixed(abs % 1e12 === 0 ? 0 : 2) + "T";
  if (abs >= 1e9)  return (v / 1e9 ).toFixed(abs % 1e9  === 0 ? 0 : 2) + "B";
  if (abs >= 1e6)  return (v / 1e6 ).toFixed(abs % 1e6  === 0 ? 0 : 2) + "M";
  if (abs >= 1e3)  return (v / 1e3 ).toFixed(abs % 1e3  === 0 ? 0 : 1) + "K";
  return String(v);
}

function fmtAgo(sec) {
  if (sec == null) return "—";
  if (sec < 60)  return `${sec}s ago`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  return `${Math.floor(sec / 3600)}h ago`;
}

// Expanded crop modal (shared singleton)
let _cropModal = null;
function _showCropModal(label, src) {
  if (!_cropModal) {
    _cropModal = document.createElement("div");
    _cropModal.style.cssText =
      "position:fixed;inset:0;background:rgba(0,0,0,.75);display:flex;align-items:center;" +
      "justify-content:center;z-index:9999;cursor:zoom-out";
    _cropModal.addEventListener("click", () => (_cropModal.style.display = "none"));
    const img = document.createElement("img");
    img.id = "cropModalImg";
    img.style.cssText = "max-width:90vw;max-height:90vh;image-rendering:pixelated;border:2px solid #555";
    const cap = document.createElement("div");
    cap.id = "cropModalCap";
    cap.style.cssText = "position:absolute;bottom:8%;color:#eee;font-size:0.9rem;text-shadow:0 0 4px #000";
    _cropModal.appendChild(img);
    _cropModal.appendChild(cap);
    document.body.appendChild(_cropModal);
  }
  document.getElementById("cropModalImg").src = src;
  document.getElementById("cropModalCap").textContent = label;
  _cropModal.style.display = "flex";
}

function renderUpgradesSeen(seen) {
  const tbody = document.getElementById("upgradeSeenTable");
  if (!tbody) return;
  tbody.innerHTML = "";
  const entries = Object.entries(seen).sort((a, b) => a[0].localeCompare(b[0]));
  if (!entries.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="5" class="text-muted ps-2">No upgrades detected yet</td>`;
    tbody.appendChild(tr);
    return;
  }
  for (const [label, d] of entries) {
    const tr = document.createElement("tr");
    const stale = d.seen_ago > 30;
    if (stale) tr.style.opacity = "0.5";

    let costCell;
    if (d.is_max) {
      costCell = `<span style="color:#f77">MAX</span>`;
    } else if (d.is_affordable === false && d.cost != null) {
      costCell = `<span style="color:#888">${esc(fmtNum(d.cost))} 🔒</span>`;
    } else if (d.cost != null) {
      const indicator = d.is_affordable === true ? ' ✓' : '';
      costCell = `<span style="color:#fc3">${esc(fmtNum(d.cost))}${indicator}</span>`;
    } else {
      costCell = `<span class="text-muted">—</span>`;
    }

    const xCount = d.upgrades_to_purchase > 1
      ? ` <span class="text-muted">×${d.upgrades_to_purchase}</span>` : "";

    const thumbCell = d.crop_b64
      ? `<img src="data:image/jpeg;base64,${d.crop_b64}"
              style="height:32px;cursor:zoom-in;image-rendering:pixelated"
              title="Click to enlarge">`
      : `<span class="text-muted" style="font-size:0.65rem">no img</span>`;

    tr.innerHTML =
      `<td class="ps-2">${esc(label)}${xCount}</td>` +
      `<td style="color:#8cf">${esc(fmtNum(d.current_value))}</td>` +
      `<td>${costCell}</td>` +
      `<td class="text-end pe-2 text-muted">${esc(fmtAgo(d.seen_ago))}</td>` +
      `<td class="pe-1">${thumbCell}</td>`;

    if (d.crop_b64) {
      const imgEl = tr.querySelector("img");
      const src = `data:image/jpeg;base64,${d.crop_b64}`;
      imgEl.addEventListener("click", (e) => { e.stopPropagation(); _showCropModal(label, src); });
    }
    tbody.appendChild(tr);
  }
}

// ── OCR list + bbox highlight ────────────────────────────────────────────

const _pinnedBoxes = new Set();   // indices of pinned OCR results
let   _ocrResults  = [];          // latest result list

function renderOcrList(results) {
  _ocrResults = results;
  _pinnedBoxes.clear();
  const ul = document.getElementById("ocrList");
  if (!ul) return;
  setText("ocrCount", results.length ? `${results.length} elements` : "");
  ul.innerHTML = "";
  results.forEach((r, i) => {
    const li  = document.createElement("li");
    li.id     = `ocr-item-${i}`;
    li.className  = "ocr-item px-2 py-1 d-flex align-items-baseline gap-2";
    li.style.cssText = "cursor:default;border-bottom:1px solid #1a1a1a;font-size:0.75rem";
    const conf = r.conf >= 80 ? "#6f6" : r.conf >= 50 ? "#fc3" : "#f66";
    const btn  = document.createElement("button");
    btn.title     = "Pin bounding box";
    btn.className = "btn btn-sm p-0 ocr-pin-btn";
    btn.style.cssText = "font-size:0.7rem;line-height:1;min-width:1.4rem;color:#555;background:none;border:1px solid #333;border-radius:3px";
    btn.textContent = "⬜";
    btn.addEventListener("click", (e) => { e.stopPropagation(); togglePin(i, btn); });
    li.appendChild(btn);
    const info = document.createElement("span");
    info.className = "d-flex justify-content-between align-items-baseline flex-grow-1";
    info.innerHTML =
      `<span class="font-monospace text-info">${esc(r.text)}</span>` +
      `<span class="text-muted ms-2" style="white-space:nowrap;font-size:0.68rem">` +
        `(${r.x0.toFixed(3)}, ${r.y0.toFixed(3)}) ` +
        `${r.w.toFixed(3)}×${r.h.toFixed(3)} ` +
        `<span style="color:${conf}">${r.conf}%</span>` +
      `</span>`;
    li.appendChild(info);
    li.addEventListener("mouseenter", () => drawOverlay(r));
    li.addEventListener("mouseleave", () => drawOverlay(null));
    ul.appendChild(li);
  });
}

function togglePin(i, btn) {
  if (_pinnedBoxes.has(i)) {
    _pinnedBoxes.delete(i);
    btn.textContent = "⬜";
    btn.style.color = "#555";
  } else {
    _pinnedBoxes.add(i);
    btn.textContent = "🔲";
    btn.style.color = "#0cf";
  }
  drawOverlay(null);
}

function drawOverlay(hoverResult) {
  if (!octx) return;
  octx.clearRect(0, 0, overlay.width, overlay.height);
  // Draw all pinned boxes first
  for (const i of _pinnedBoxes) {
    if (_ocrResults[i]) _drawBox(_ocrResults[i], /*pinned=*/true);
  }
  // Draw hover box on top (if any)
  if (hoverResult) _drawBox(hoverResult, /*pinned=*/false);
  // Draw pinned click crosshair (if any)
  if (_clickCrosshair) _drawCrosshair(_clickCrosshair.fx, _clickCrosshair.fy);
}

function _drawCrosshair(fx, fy) {
  if (!octx || !overlay.width || !overlay.height) return;
  const cx = fx * overlay.width;
  const cy = fy * overlay.height;
  const arm = Math.max(40, overlay.width * 0.04);
  octx.save();
  octx.strokeStyle = "#ff0";
  octx.lineWidth   = 2.5;
  // horizontal arm
  octx.beginPath(); octx.moveTo(cx - arm, cy); octx.lineTo(cx + arm, cy); octx.stroke();
  // vertical arm
  octx.beginPath(); octx.moveTo(cx, cy - arm); octx.lineTo(cx, cy + arm); octx.stroke();
  // centre circle
  octx.beginPath(); octx.arc(cx, cy, 7, 0, Math.PI * 2); octx.stroke();
  octx.restore();
}

function toggleActionCrosshair(btn, fx, fy) {
  if (_clickCrosshair && _clickCrosshair.fx === fx && _clickCrosshair.fy === fy) {
    _clickCrosshair = null;
    btn.style.color = "#555";
  } else {
    // Deactivate previous button if any
    const prev = document.querySelector(".action-xhair-btn.active-xhair");
    if (prev) { prev.style.color = "#555"; prev.classList.remove("active-xhair"); }
    _clickCrosshair = { fx, fy };
    btn.style.color = "#ff0";
    btn.classList.add("active-xhair");
  }
  drawOverlay(null);
}

function _drawBox(r, pinned) {
  const x = r.x0 * overlay.width;
  const y = r.y0 * overlay.height;
  const w = r.w  * overlay.width;
  const h = r.h  * overlay.height;
  const colour = r.conf >= 80 ? (pinned ? "#0cf" : "#0f0")
               : r.conf >= 50 ? "#fc3"
               : "#f00";
  octx.strokeStyle = colour;
  octx.lineWidth   = pinned ? 2 : 1.5;
  octx.strokeRect(x, y, w, h);
  octx.fillStyle = pinned ? "rgba(0,200,255,0.10)" : "rgba(0,255,128,0.10)";
  octx.fillRect(x, y, w, h);
}

function highlightBox(r) { drawOverlay(r); }   // kept for compat
function clearOverlay()   { drawOverlay(null); }

// ── Controls ────────────────────────────────────────────────────────────

async function api(action, body) {
  const opts = { method: "POST" };
  if (body) {
    opts.headers = { "Content-Type": "application/json" };
    opts.body    = JSON.stringify(body);
  }
  try {
    const r = await fetch(API(action), opts);
    return await r.json();
  } catch (e) {
    console.error("API error", e);
  }
}

// ── Strategy parameter UI ───────────────────────────────────────────────

async function loadParamSchema() {
  try {
    const r      = await fetch(API("params/schema"));
    const schema = await r.json();
    buildParamUI(schema);
  } catch (e) {
    console.error("Failed to load param schema", e);
  }
}

function buildParamUI(schema) {
  const container = document.getElementById("paramsContainer");
  if (!container) return;
  container.innerHTML = "";

  for (const [key, def] of Object.entries(schema)) {
    const div = document.createElement("div");
    div.className = "mb-2 d-flex align-items-center justify-content-between";

    const label = document.createElement("label");
    label.className   = "form-label mb-0 small me-2";
    label.textContent = def.label || key;
    div.appendChild(label);

    let input;
    if (def.type === "bool") {
      input           = document.createElement("input");
      input.type      = "checkbox";
      input.className = "form-check-input";
      input.dataset.paramKey = key;
      input.addEventListener("change", () => api("params", { [key]: input.checked }));
    } else if (def.type === "choice") {
      input           = document.createElement("select");
      input.className = "form-select form-select-sm";
      input.style.width = "130px";
      for (const c of (def.choices || [])) {
        const opt = document.createElement("option");
        opt.value = opt.textContent = c;
        input.appendChild(opt);
      }
      input.dataset.paramKey = key;
      input.addEventListener("change", () => api("params", { [key]: input.value }));
    } else {
      input           = document.createElement("input");
      input.type      = "number";
      input.className = "form-control form-control-sm";
      input.style.width = "90px";
      if (def.min  !== undefined) input.min  = def.min;
      if (def.max  !== undefined) input.max  = def.max;
      if (def.step !== undefined) input.step = def.step;
      input.dataset.paramKey = key;
      input.addEventListener("change", () => api("params", { [key]: Number(input.value) }));
    }
    div.appendChild(input);
    container.appendChild(div);
  }
}

function updateParamValues(params) {
  for (const [key, val] of Object.entries(params)) {
    const el = document.querySelector(`[data-param-key="${key}"]`);
    if (!el) continue;
    if (el.type === "checkbox") el.checked = !!val;
    else el.value = val;
  }
}

// ── Helpers ─────────────────────────────────────────────────────────────

function updateBadge(status) {
  const el = document.getElementById("statusBadge");
  el.textContent = status;
  el.className   = "badge badge-" + status;
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = String(val);
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

// ── Timeline chart ─────────────────────────────────────────────────────────────────

async function fetchTimeline() {
  const statusEl = document.getElementById("timelineStatus");
  try {
    const r = await fetch("/api/timeline");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    renderTimeline(d);
    if (statusEl) statusEl.textContent = `Updated ${new Date().toLocaleTimeString()}`;
  } catch (e) {
    console.warn("Timeline fetch failed", e);
    if (statusEl) statusEl.textContent = `Fetch failed: ${e.message}`;
  }
}

function _fmtRate(n) {
  if (n == null) return "—";
  if (n >= 1e12) return (n / 1e12).toPrecision(4) + "T";
  if (n >= 1e9)  return (n / 1e9 ).toPrecision(4) + "B";
  if (n >= 1e6)  return (n / 1e6 ).toPrecision(4) + "M";
  if (n >= 1e3)  return (n / 1e3 ).toPrecision(4) + "K";
  return String(Math.round(n));
}

// Utility: Remove outliers using IQR method
function filterOutliersIQR(points, valueKey = 'y') {
  if (!points.length) return points;
  const values = points.map(p => p[valueKey]).filter(v => typeof v === 'number');
  if (values.length < 4) return points; // Not enough data for IQR
  const sorted = [...values].sort((a, b) => a - b);
  const q1 = sorted[Math.floor(sorted.length * 0.25)];
  const q3 = sorted[Math.floor(sorted.length * 0.75)];
  const iqr = q3 - q1;
  const min = q1 - 1.5 * iqr;
  const max = q3 + 1.5 * iqr;
  return points.filter(p => p[valueKey] >= min && p[valueKey] <= max);
}

function renderTimeline(data) {
  const cvs = document.getElementById("timelineChart");
  if (!cvs) return;

  // Build dataset arrays
  const wavePts = (data.wave_history  || []).map(p => ({ x: p.t * 1000, y: p.wave }));

  const tickPts = (data.capture_ticks || [])
    .filter(p => p.wave != null)
    .map(p => ({
      x: p.t * 1000, y: p.wave,
      _file: p.file, _tier: p.tier, _tab: p.tab,
      _png: p.png, _png_anno: p.png_anno,
    }));

  // Store action history for crosshair overlay during hover
  _timelineActions = (data.action_history || []);

  // Build hue map: unique reasons → evenly-spaced rainbow hues
  const _uniqueReasons = [...new Set(_timelineActions.map(a => a.reason || "click"))];
  _uniqueReasons.sort();
  const _reasonHue = {};
  _uniqueReasons.forEach((r, i) => {
    _reasonHue[r] = Math.round((i / Math.max(_uniqueReasons.length, 1)) * 360);
  });

  // Inline Chart.js plugin: draw a vertical line for each click action
  const _clickLinesPlugin = {
    id: "clickLines",
    afterDraw(chart) {
      if (!_timelineActions.length) return;
      const xScale = chart.scales.x;
      const { ctx: c2, chartArea: ca } = chart;
      c2.save();
      c2.lineWidth = 1;
      c2.globalAlpha = 0.65;
      for (const a of _timelineActions) {
        const xPx = xScale.getPixelForValue(a.t * 1000);
        if (xPx < ca.left || xPx > ca.right) continue;
        const hue = _reasonHue[a.reason || "click"] ?? 0;
        c2.strokeStyle = `hsl(${hue},100%,55%)`;
        c2.beginPath();
        c2.moveTo(xPx, ca.top);
        c2.lineTo(xPx, ca.bottom);
        c2.stroke();
      }
      c2.restore();
    },
  };


  let cashPts = (data.rate_history || [])
    .filter(p => p.cash_pm != null)
    .map(p => ({ x: p.t * 1000, y: p.cash_pm }));
  cashPts = filterOutliersIQR(cashPts, 'y');

  let coinPts = (data.rate_history || [])
    .filter(p => p.coin_pm != null)
    .map(p => ({ x: p.t * 1000, y: p.coin_pm }));
  coinPts = filterOutliersIQR(coinPts, 'y');

  let spendPts = (data.spend_rate_history || [])
    .filter(p => p.spend_pm != null)
    .map(p => ({ x: p.t * 1000, y: p.spend_pm }));
  // No IQR filter on spend — each point is already an aggregated rate

  let waveRatePts = (data.wave_rate_history || [])
    .map(p => ({ x: p.t * 1000, y: p.waves_ph }));
  waveRatePts = filterOutliersIQR(waveRatePts, 'y');

  const datasets = [
    {
      label: "Wave",
      data: wavePts,
      borderColor: "#6cf",
      backgroundColor: "transparent",
      borderWidth: 1.5,
      pointRadius: 2,
      tension: 0.15,
      yAxisID: "yWave",
    },
    {
      label: "Captures",
      data: tickPts,
      borderColor: "#fc3",
      backgroundColor: "#fc3",
      borderWidth: 0,
      pointRadius: 6,
      pointStyle: "triangle",
      showLine: false,
      yAxisID: "yWave",
    },
    {
      label: "Cash /min",
      data: cashPts,
      borderColor: "#8f8",
      backgroundColor: "transparent",
      borderWidth: 1.5,
      borderDash: [5, 3],
      pointRadius: 1.5,
      tension: 0.2,
      yAxisID: "yRate",
    },
    {
      label: "Coin /min",
      data: coinPts,
      borderColor: "#f9a",
      backgroundColor: "transparent",
      borderWidth: 1.5,
      borderDash: [5, 3],
      pointRadius: 1.5,
      tension: 0.2,
      yAxisID: "yCoinRate",
    },
    {
      label: "Spend /min",
      data: spendPts,
      borderColor: "#fa0",
      backgroundColor: "transparent",
      borderWidth: 1.5,
      borderDash: [4, 2],
      pointRadius: 2,
      tension: 0.2,
      yAxisID: "yRate",
    },
    {
      label: "Wave /h",
      data: waveRatePts,
      borderColor: "#c8f",
      backgroundColor: "transparent",
      borderWidth: 1.5,
      borderDash: [3, 3],
      pointRadius: 1.5,
      tension: 0.3,
      yAxisID: "yWaveRate",
    },
  ];

  // Determine if session spans multiple calendar days
  const allTs = [
    ...wavePts.map(p => p.x),
    ...tickPts.map(p => p.x),
    ...cashPts.map(p => p.x),
    ...coinPts.map(p => p.x),
    ...spendPts.map(p => p.x),
    ...waveRatePts.map(p => p.x),
  ];

  // ── Store overview data ────────────────────────────────────────────────────
  _overviewWavePts = wavePts;
  _overviewActTs   = _timelineActions.map(a => a.t * 1000);
  if (allTs.length > 0) {
    const prevDataMaxT    = _overviewDataMaxT;
    _overviewDataMinT     = Math.min(...allTs);
    const newDataMaxT     = Math.max(...allTs);

    if (!_overviewInitialized) {
      // First paint: show last hour if possible, else full data extent
      _overviewDataMaxT = newDataMaxT;
      const ONE_HOUR_MS = 60 * 60 * 1000;
      if (_overviewDataMaxT - _overviewDataMinT > ONE_HOUR_MS) {
        _viewMaxT = _overviewDataMaxT;
        _viewMinT = _overviewDataMaxT - ONE_HOUR_MS;
      } else {
        _viewMinT = _overviewDataMinT;
        _viewMaxT = _overviewDataMaxT;
      }
      _overviewInitialized = true;
    } else {
      // Auto-follow: if the right edge of the window was at (or within 2 s of)
      // the right edge of the data, slide the window forward with the new data.
      const _AT_RIGHT_EDGE = prevDataMaxT > 0 && (_viewMaxT >= prevDataMaxT - 2000);
      _overviewDataMaxT = newDataMaxT;
      if (_AT_RIGHT_EDGE && newDataMaxT > prevDataMaxT) {
        const winSpan = _viewMaxT - _viewMinT;
        _viewMaxT = _overviewDataMaxT;
        _viewMinT = _viewMaxT - winSpan;
      }
    }
  }

  const multiDay = allTs.length >= 2 &&
    (new Date(Math.max(...allTs)).toLocaleDateString() !==
     new Date(Math.min(...allTs)).toLocaleDateString());

  const timeUnit      = multiDay ? "hour"   : "minute";
  const displayFormat = multiDay ? "MM/dd HH:mm" : "HH:mm";

  const chartConfig = {
    type: "line",
    data: { datasets },
    plugins: [_clickLinesPlugin],
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          labels: { color: "#aaa", boxWidth: 12, font: { size: 11 } },
        },
        tooltip: {
          callbacks: {
            title(items) {
              if (!items.length) return "";
              return new Date(items[0].parsed.x).toLocaleTimeString();
            },
            label(ci) {
              const raw = ci.raw;
              // Capture tick: show filename + tab
              if (ci.datasetIndex === 1) {
                const tier = raw._tier ? ` t${raw._tier}` : "";
                const tab  = raw._tab  ? ` [${raw._tab}]` : "";
                return `\u{1F4F8}${tier}${tab}  w${raw.y}  ${raw._file}`;
              }
              const v = raw.y;
              if (v == null) return null;
              return `${ci.dataset.label}: ${_fmtRate(v)}`;
            },
          },
        },
      },
      scales: {
        x: {
          type: "time",
          min: _viewMinT || undefined,
          max: _viewMaxT || undefined,
          time: {
            unit: timeUnit,
            tooltipFormat: "HH:mm:ss dd/MM",
            displayFormats: { minute: displayFormat, hour: displayFormat },
          },
          ticks: { color: "#999", maxTicksLimit: 12, font: { size: 10 } },
          grid:  { color: "#1a1a1a" },
        },
        yWave: {
          position: "left",
          title:    { display: true, text: "Wave", color: "#6cf", font: { size: 10 } },
          ticks:    { color: "#6cf", font: { size: 10 } },
          grid:     { color: "#222" },
        },
        yRate: {
          position: "right",
          title:    { display: true, text: "Cash /min", color: "#8f8", font: { size: 10 } },
          min: 0,
          ticks: {
            color: "#8f8",
            font:  { size: 10 },
            callback(v) { return _fmtRate(v); },
          },
          grid: { drawOnChartArea: false },
        },
        yCoinRate: {
          position: "right",
          offset: true,
          title:    { display: true, text: "Coin /min", color: "#f9a", font: { size: 10 } },
          min: 0,
          ticks: {
            color: "#f9a",
            font:  { size: 10 },
            callback(v) { return _fmtRate(v); },
          },
          grid: { drawOnChartArea: false },
        },
        yWaveRate: {
          position: "right",
          offset: true,
          title:    { display: true, text: "w/h", color: "#c8f", font: { size: 10 } },
          min: 0,
          ticks:    { color: "#c8f", font: { size: 10 } },
          grid:     { drawOnChartArea: false },
        },
      },
    },
  };

  if (_timelineChart) {
    // Patch datasets in-place and redraw without animation
    _timelineChart.data.datasets[0].data = wavePts;
    _timelineChart.data.datasets[1].data = tickPts;
    _timelineChart.data.datasets[2].data = cashPts;
    _timelineChart.data.datasets[3].data = coinPts;
    _timelineChart.data.datasets[4].data = spendPts;
    _timelineChart.data.datasets[5].data = waveRatePts;
    _timelineChart.options.scales.x.time.unit           = timeUnit;
    _timelineChart.options.scales.x.time.displayFormats = { minute: displayFormat, hour: displayFormat };
    _timelineChart.options.scales.x.min = _viewMinT || undefined;
    _timelineChart.options.scales.x.max = _viewMaxT || undefined;
    // Swap in the freshly-built plugin (carries updated _reasonHue closure)
    _timelineChart.config.plugins = [_clickLinesPlugin];
    _timelineChart.update("none");
  } else {
    _timelineChart = new Chart(cvs, chartConfig);
    _attachTimelineHover(cvs);
    const ovCvs = document.getElementById("timelineOverview");
    if (ovCvs) _attachOverviewInteraction(ovCvs);
  }
  // Repaint the overview strip
  _renderOverview();
}

function _attachTimelineHover(cvs) {
  // Shared scrub logic used by both mouse and touch
  function _scrubAt(clientX) {
    if (!_timelineChart) return;
    const ticks  = _timelineChart.data.datasets[1].data;
    const rect   = cvs.getBoundingClientRect();
    const xPixel = clientX - rect.left;
    const xScale = _timelineChart.scales.x;
    const tMs    = xScale.getValueForPixel(xPixel);
    _timelineHovering = true;
    _hoverTimeMs = tMs;
    let best = null, bestDist = Infinity;
    for (const tick of ticks) {
      const d = Math.abs(tick.x - tMs);
      if (d < bestDist) { bestDist = d; best = tick; }
    }
    const file = best ? (best._png_anno || best._png) : null;
    if (file && file !== _hoverFile) {
      _loadCapturePreview(file);
    } else {
      _drawTimelineCrosshairs(tMs);
    }
  }

  cvs.addEventListener("mousemove", (e) => _scrubAt(e.clientX));
  cvs.addEventListener("mouseleave", () => _clearTimelineHover());

  // Touch scrubbing (iPad / mobile)
  cvs.addEventListener("touchmove", (e) => {
    e.preventDefault();
    _scrubAt(e.touches[0].clientX);
  }, { passive: false });
  cvs.addEventListener("touchend", () => _clearTimelineHover());
}

function _loadCapturePreview(filename) {
  _hoverFile = filename;
  const img = new Image();
  img.onload = () => {
    canvas.width  = img.naturalWidth;
    canvas.height = img.naturalHeight;
    overlay.width  = img.naturalWidth;
    overlay.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);
    document.getElementById("noImage").style.display = "none";
    _drawTimelineCrosshairs(_hoverTimeMs);
  };
  img.src = "/debug/" + filename;
}

function _drawTimelineCrosshairs(tMs) {
  if (!octx || !overlay.width || !overlay.height) return;
  octx.clearRect(0, 0, overlay.width, overlay.height);

  if (tMs != null && _timelineActions.length > 0) {
    const tSec = tMs / 1000;

    // Find the single nearest action by time
    let nearest = null, nearestDist = Infinity;
    for (const a of _timelineActions) {
      const d = Math.abs(a.t - tSec);
      if (d < nearestDist) { nearestDist = d; nearest = a; }
    }

    if (nearest) {
      const cx  = nearest.fx * overlay.width;
      const cy  = nearest.fy * overlay.height;
      const arm = Math.max(30, overlay.width * 0.03);
      octx.save();
      octx.strokeStyle = "#ff0";
      octx.lineWidth   = 2.5;
      octx.beginPath(); octx.moveTo(cx - arm, cy); octx.lineTo(cx + arm, cy); octx.stroke();
      octx.beginPath(); octx.moveTo(cx, cy - arm); octx.lineTo(cx, cy + arm); octx.stroke();
      octx.beginPath(); octx.arc(cx, cy, 7, 0, Math.PI * 2); octx.stroke();
      octx.restore();

      // Tooltip: reason + time delta
      const dt   = Math.round(nearest.t - tSec);
      const sign = dt >= 0 ? "+" : "";
      const tip  = `${nearest.reason || "click"}  (${sign}${dt}s)`;
      octx.save();
      octx.font      = "12px monospace";
      octx.fillStyle = "rgba(0,0,0,0.6)";
      const tw = octx.measureText(tip).width;
      const tx = Math.min(cx + 10, overlay.width - tw - 6);
      const ty = Math.max(cy - 10, 16);
      octx.fillRect(tx - 2, ty - 13, tw + 4, 16);
      octx.fillStyle = "#ff0";
      octx.fillText(tip, tx, ty);
      octx.restore();
    }
  }

  // Bottom filename label
  if (_hoverFile) {
    octx.save();
    octx.font        = "bold 13px monospace";
    octx.globalAlpha = 0.9;
    const label = "\u23F8 " + _hoverFile +
                  (_timelineActions.length === 0 ? "  [no action data]" : "");
    octx.fillStyle = "rgba(0,0,0,0.55)";
    const tw = octx.measureText(label).width;
    octx.fillRect(4, overlay.height - 22, tw + 8, 18);
    octx.fillStyle = "#fc3";
    octx.fillText(label, 8, overlay.height - 8);
    octx.restore();
  }
}

function _clearTimelineHover() {
  if (!_timelineHovering) return;
  _timelineHovering = false;
  _hoverFile   = null;
  _hoverTimeMs = null;
  octx.clearRect(0, 0, overlay.width, overlay.height);
  // Restore the live image if available
  if (_liveImageSrc) {
    const img = new Image();
    img.onload = () => {
      imgNatW       = img.naturalWidth;
      imgNatH       = img.naturalHeight;
      canvas.width  = imgNatW;
      canvas.height = imgNatH;
      overlay.width  = imgNatW;
      overlay.height = imgNatH;
      ctx.drawImage(img, 0, 0);
    };
    img.src = _liveImageSrc;
  }
}

// ── Overview strip helpers ─────────────────────────────────────────────────────

/**
 * Clamp and apply a new zoom window, then refresh both the main chart and
 * the overview minimap.
 */
function _setViewRange(minT, maxT) {
  const MIN_SPAN = 60_000; // 1 minute minimum window
  // Enforce minimum span
  if (maxT - minT < MIN_SPAN) {
    const mid = (minT + maxT) / 2;
    minT = mid - MIN_SPAN / 2;
    maxT = mid + MIN_SPAN / 2;
  }
  // Clamp to data extent
  if (_overviewDataMaxT > _overviewDataMinT) {
    if (minT < _overviewDataMinT) { maxT += _overviewDataMinT - minT; minT = _overviewDataMinT; }
    if (maxT > _overviewDataMaxT) { minT -= maxT - _overviewDataMaxT; maxT = _overviewDataMaxT; }
    minT = Math.max(minT, _overviewDataMinT);
    maxT = Math.min(maxT, _overviewDataMaxT);
  }
  _viewMinT = minT;
  _viewMaxT = maxT;
  if (_timelineChart) {
    _timelineChart.options.scales.x.min = minT;
    _timelineChart.options.scales.x.max = maxT;
    _timelineChart.update("none");
  }
  _renderOverview();
}

/** Draw the full-extent minimap with a highlighted window rect. */
function _renderOverview() {
  const cvs = document.getElementById("timelineOverview");
  if (!cvs) return;
  const cssRect = cvs.getBoundingClientRect();
  const W = cssRect.width, H = cssRect.height;
  if (W <= 0 || H <= 0) return;

  const dpr = window.devicePixelRatio || 1;
  // Only resize backing store when CSS size changes (resizing clears the canvas)
  const needW = Math.round(W * dpr), needH = Math.round(H * dpr);
  if (cvs.width !== needW || cvs.height !== needH) {
    cvs.width  = needW;
    cvs.height = needH;
  }
  const c = cvs.getContext("2d");
  c.setTransform(dpr, 0, 0, dpr, 0, 0); // work in CSS pixels

  const dMin = _overviewDataMinT, dMax = _overviewDataMaxT;
  const span = dMax - dMin || 1;
  const toPx = ms => ((ms - dMin) / span) * W;

  // Background
  c.fillStyle = "#080808";
  c.fillRect(0, 0, W, H);

  // Mini wave line
  if (_overviewWavePts.length >= 2) {
    const ys    = _overviewWavePts.map(p => p.y);
    const wMin  = Math.min(...ys), wMax = Math.max(...ys);
    const wSpan = wMax - wMin || 1;
    c.beginPath();
    c.strokeStyle = "rgba(102,204,255,0.55)";
    c.lineWidth   = 1;
    for (let i = 0; i < _overviewWavePts.length; i++) {
      const p  = _overviewWavePts[i];
      const px = toPx(p.x);
      const py = H - 4 - ((p.y - wMin) / wSpan) * (H - 10);
      i === 0 ? c.moveTo(px, py) : c.lineTo(px, py);
    }
    c.stroke();
  }

  // Action tick marks at the bottom edge
  c.strokeStyle = "rgba(255,195,50,0.45)";
  c.lineWidth   = 1;
  for (const tMs of _overviewActTs) {
    const px = toPx(tMs);
    c.beginPath(); c.moveTo(px, H - 5); c.lineTo(px, H); c.stroke();
  }

  // Dim regions outside the current view window
  const vL = Math.max(0, toPx(_viewMinT));
  const vR = Math.min(W, toPx(_viewMaxT));
  c.fillStyle = "rgba(0,0,0,0.52)";
  c.fillRect(0, 0, vL, H);
  c.fillRect(vR, 0, W - vR, H);

  // Window border
  c.strokeStyle = "rgba(255,255,255,0.80)";
  c.lineWidth   = 1.5;
  const winW    = Math.max(vR - vL - 1.5, 0);
  c.strokeRect(vL + 0.75, 0.75, winW, H - 1.5);

  // Resize handle ticks on left/right window edges
  const hH = Math.round(H * 0.45), hY = Math.round((H - hH) / 2);
  c.fillStyle = "rgba(255,255,255,0.65)";
  if (vL + 5 < vR - 7) {               // only draw if window is wide enough
    c.fillRect(vL + 3, hY, 2, hH);
    c.fillRect(vR - 5, hY, 2, hH);
  }

  // Time labels at window edges
  c.save();
  c.font         = "9px monospace";
  c.fillStyle    = "rgba(255,255,255,0.55)";
  const fmtT     = ms => new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const leftLbl  = fmtT(_viewMinT);
  const rightLbl = fmtT(_viewMaxT);
  const lw       = c.measureText(leftLbl).width;
  const rw       = c.measureText(rightLbl).width;
  // Left label: just to the right of the left edge (flip inside if too close to left border)
  const lx = vL + 4 + lw < vR - rw - 8 ? vL + 4 : vL + 4;
  c.textAlign  = "left";
  c.fillText(leftLbl, Math.max(2, vL + 4), H - 3);
  // Right label: just to the left of the right edge
  c.textAlign  = "right";
  c.fillText(rightLbl, Math.min(W - 2, vR - 4), H - 3);
  c.restore();
}

/** Convert a clientX coordinate to a millisecond timestamp on the overview canvas. */
function _ovToMs(clientX) {
  const cvs  = document.getElementById("timelineOverview");
  if (!cvs) return 0;
  const rect = cvs.getBoundingClientRect();
  const span = _overviewDataMaxT - _overviewDataMinT || 1;
  return _overviewDataMinT + ((clientX - rect.left) / rect.width) * span;
}

/** Determine the drag mode given a clientX on the overview canvas. */
function _ovGetMode(clientX) {
  const cvs  = document.getElementById("timelineOverview");
  if (!cvs) return "jump";
  const rect = cvs.getBoundingClientRect();
  const W    = rect.width;
  const span = _overviewDataMaxT - _overviewDataMinT || 1;
  const px   = clientX - rect.left;
  const vL   = ((_viewMinT - _overviewDataMinT) / span) * W;
  const vR   = ((_viewMaxT - _overviewDataMinT) / span) * W;
  if (Math.abs(px - vL) <= _OV_EDGE_PX) return "resizeL";
  if (Math.abs(px - vR) <= _OV_EDGE_PX) return "resizeR";
  if (px >= vL && px <= vR)              return "pan";
  return "jump";
}

/** Attach all mouse / wheel / touch interaction to the overview canvas. */
function _attachOverviewInteraction(cvs) {
  // ── Shared drag logic ─────────────────────────────────────────────────────
  function _applyDrag(clientX) {
    if (!_ovDrag) return;
    const ms   = _ovToMs(clientX);
    const dxMs = ms - _ovToMs(_ovDrag.startClientX);
    if (_ovDrag.mode === "pan") {
      _setViewRange(_ovDrag.startVMin + dxMs, _ovDrag.startVMax + dxMs);
    } else if (_ovDrag.mode === "resizeL") {
      _setViewRange(ms, _viewMaxT);
    } else if (_ovDrag.mode === "resizeR") {
      _setViewRange(_viewMinT, ms);
    } else {
      // jump-drag: keep window width, slide centre to cursor
      const halfSpan = (_viewMaxT - _viewMinT) / 2;
      _setViewRange(ms - halfSpan, ms + halfSpan);
    }
  }

  // ── Mouse ─────────────────────────────────────────────────────────────────
  cvs.addEventListener("mousemove", (e) => {
    if (_ovDrag) {
      _applyDrag(e.clientX);
    } else {
      const mode = _ovGetMode(e.clientX);
      cvs.style.cursor = (mode === "resizeL" || mode === "resizeR") ? "ew-resize"
                        : mode === "pan"                            ? "grab"
                        :                                            "crosshair";
    }
  });

  cvs.addEventListener("mousedown", (e) => {
    e.preventDefault();
    const mode = _ovGetMode(e.clientX);
    _ovDrag = { mode, startClientX: e.clientX, startVMin: _viewMinT, startVMax: _viewMaxT };
    cvs.style.cursor = "grabbing";
    if (mode === "jump") {
      const ms = _ovToMs(e.clientX);
      const halfSpan = (_viewMaxT - _viewMinT) / 2;
      _setViewRange(ms - halfSpan, ms + halfSpan);
    }
  });

  // Release anywhere in the page
  window.addEventListener("mouseup", () => { if (_ovDrag) _ovDrag = null; });

  // Scroll wheel: zoom window in/out around cursor position
  cvs.addEventListener("wheel", (e) => {
    e.preventDefault();
    const factor   = e.deltaY > 0 ? 1.3 : (1 / 1.3);
    const cursorMs = _ovToMs(e.clientX);
    const lFrac    = (_viewMaxT - _viewMinT) > 0
                     ? (cursorMs - _viewMinT) / (_viewMaxT - _viewMinT)
                     : 0.5;
    const newSpan  = (_viewMaxT - _viewMinT) * factor;
    _setViewRange(cursorMs - lFrac * newSpan, cursorMs + (1 - lFrac) * newSpan);
  }, { passive: false });

  // Double-click: reset to full data extent
  cvs.addEventListener("dblclick", () => {
    if (_overviewDataMaxT > _overviewDataMinT)
      _setViewRange(_overviewDataMinT, _overviewDataMaxT);
  });

  // ── Touch (iPad / mobile) ─────────────────────────────────────────────────
  let _lastTapTime    = 0;
  let _pinchStartSnap = null; // { dist, vMin, vMax, midMs }

  function _pinchDist(touches) {
    const dx = touches[0].clientX - touches[1].clientX;
    const dy = touches[0].clientY - touches[1].clientY;
    return Math.hypot(dx, dy);
  }

  cvs.addEventListener("touchstart", (e) => {
    e.preventDefault();
    if (e.touches.length === 2) {
      // Begin pinch-to-zoom
      _ovDrag = null;
      _pinchStartSnap = {
        dist  : _pinchDist(e.touches),
        vMin  : _viewMinT,
        vMax  : _viewMaxT,
        midMs : _ovToMs((e.touches[0].clientX + e.touches[1].clientX) / 2),
      };
      return;
    }
    _pinchStartSnap = null;
    const cx   = e.touches[0].clientX;
    const mode = _ovGetMode(cx);
    _ovDrag = { mode, startClientX: cx, startVMin: _viewMinT, startVMax: _viewMaxT };
    if (mode === "jump") {
      const ms = _ovToMs(cx);
      const halfSpan = (_viewMaxT - _viewMinT) / 2;
      _setViewRange(ms - halfSpan, ms + halfSpan);
    }
    // Double-tap → reset
    const now = Date.now();
    if (now - _lastTapTime < 350) {
      if (_overviewDataMaxT > _overviewDataMinT)
        _setViewRange(_overviewDataMinT, _overviewDataMaxT);
    }
    _lastTapTime = now;
  }, { passive: false });

  cvs.addEventListener("touchmove", (e) => {
    e.preventDefault();
    if (e.touches.length === 2 && _pinchStartSnap) {
      // Pinch zoom around the original midpoint
      const newDist  = _pinchDist(e.touches);
      const factor   = _pinchStartSnap.dist / newDist; // <1 = zoom in, >1 = zoom out
      const midMs    = _pinchStartSnap.midMs;
      const origSpan = _pinchStartSnap.vMax - _pinchStartSnap.vMin;
      const lFrac    = origSpan > 0 ? (midMs - _pinchStartSnap.vMin) / origSpan : 0.5;
      const newSpan  = origSpan * factor;
      _setViewRange(midMs - lFrac * newSpan, midMs + (1 - lFrac) * newSpan);
      return;
    }
    _applyDrag(e.touches[0].clientX);
  }, { passive: false });

  cvs.addEventListener("touchend",   () => { _ovDrag = null; _pinchStartSnap = null; });
  cvs.addEventListener("touchcancel",() => { _ovDrag = null; _pinchStartSnap = null; });
}
