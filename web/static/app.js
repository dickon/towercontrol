/*  TowerControl – dashboard client */

const WS_URL = `ws://${location.host}/ws`;
const API    = (path) => `${location.origin}/api/${path}`;

let ws      = null;
let canvas, ctx, overlay, octx;
let imgNatW = 0, imgNatH = 0;   // natural size of the last received image

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
  const { fx, fy } = canvasFrac(evt);
  console.log("Injecting click at", fx.toFixed(4), fy.toFixed(4));
  api("click", { fx, fy });
}

// ── State update ────────────────────────────────────────────────────────

function handleState(s) {
  updateBadge(s.bot_status || "offline");

  // Image
  if (s.image) {
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
    img.src = "data:image/jpeg;base64," + s.image;
  }

  // Window info
  if (s.window_rect) {
    const r = s.window_rect;
    document.getElementById("windowInfo").textContent =
      `${r.width}×${r.height} @ (${r.left},${r.top})`;
  }

  renderCtxTable(s.ctx_full || {});

  // Resources
  const tbody = document.getElementById("resourcesTable");
  tbody.innerHTML = "";
  for (const [k, v] of Object.entries(s.resources || {})) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="text-muted">${esc(k)}</td><td>${esc(v)}</td>`;
    tbody.appendChild(tr);
  }

  renderOcrList(s.ocr_results || []);

  // Action log
  const logEl = document.getElementById("actionLog");
  logEl.innerHTML = "";
  for (const a of (s.recent_actions || []).slice(-15).reverse()) {
    const li = document.createElement("li");
    const t  = new Date(a.time * 1000).toLocaleTimeString();
    li.innerHTML =
      `<span class="text-muted">${t}</span> ` +
      `<span class="text-warning">${esc(a.type)}</span> ` +
      `<span>${esc(a.reason || "")}</span>`;
    logEl.appendChild(li);
  }

  updateParamValues(s.strategy_params || {});
  renderPerkHistory(s.perk_selection_history || []);
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

// ── Restart ──────────────────────────────────────────────────────────────

function confirmRestart() {
  if (!confirm("Reset all game state (wave, tier, upgrade progress) and continue running?")) return;
  api("restart").then(r => {
    if (!r?.ok) console.error("Restart failed", r);
  });
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
      if (def.min !== undefined) input.min = def.min;
      if (def.max !== undefined) input.max = def.max;
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
