/*  TowerControl – dashboard client */

const WS_URL = `ws://${location.host}/ws`;
const API    = (path) => `${location.origin}/api/${path}`;

let ws = null;
let canvas, ctx;
let imgNatW = 0, imgNatH = 0;  // natural size of the last received image

// ── Bootstrap ───────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  canvas = document.getElementById("gameCanvas");
  ctx    = canvas.getContext("2d");

  // Click-to-inject on the canvas
  canvas.addEventListener("click", onCanvasClick);

  connectWS();
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

// ── State update ────────────────────────────────────────────────────────

function handleState(s) {
  // Status badge
  updateBadge(s.bot_status || "offline");

  // Image
  if (s.image) {
    const img = new Image();
    img.onload = () => {
      imgNatW = img.naturalWidth;
      imgNatH = img.naturalHeight;
      canvas.width  = imgNatW;
      canvas.height = imgNatH;
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

  // Screen & element info
  setText("currentScreen", s.current_screen || "—");
  setText("elementCount",  s.element_count ?? "—");
  setText("tabsScanned",   (s.tabs_scanned || []).join(", ") || "—");
  setText("errorCount",    s.error_count ?? 0);

  // Resources
  const tbody = document.getElementById("resourcesTable");
  tbody.innerHTML = "";
  for (const [k, v] of Object.entries(s.resources || {})) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="text-muted">${esc(k)}</td><td>${esc(v)}</td>`;
    tbody.appendChild(tr);
  }

  // OCR text
  document.getElementById("ocrText").textContent =
    (s.raw_texts || []).join("  ·  ");

  // Action log
  const logEl = document.getElementById("actionLog");
  logEl.innerHTML = "";
  for (const a of (s.recent_actions || []).slice(-15).reverse()) {
    const li = document.createElement("li");
    const t = new Date(a.time * 1000).toLocaleTimeString();
    li.innerHTML = `<span class="text-muted">${t}</span> `
      + `<span class="text-warning">${esc(a.type)}</span> `
      + `<span>${esc(a.reason || "")}</span>`;
    logEl.appendChild(li);
  }

  // Update param values live
  updateParamValues(s.strategy_params || {});
}

// ── Controls ────────────────────────────────────────────────────────────

async function api(action, body) {
  const opts = { method: "POST" };
  if (body) {
    opts.headers = { "Content-Type": "application/json" };
    opts.body = JSON.stringify(body);
  }
  try {
    const r = await fetch(API(action), opts);
    return await r.json();
  } catch (e) {
    console.error("API error", e);
  }
}

function onCanvasClick(evt) {
  if (!imgNatW) return;
  const rect = canvas.getBoundingClientRect();
  const scaleX = imgNatW / rect.width;
  const scaleY = imgNatH / rect.height;
  const x = Math.round((evt.clientX - rect.left) * scaleX);
  const y = Math.round((evt.clientY - rect.top)  * scaleY);
  console.log("Injecting click at", x, y);
  api("click", { x, y });
}

// ── Strategy parameter UI ───────────────────────────────────────────────

async function loadParamSchema() {
  try {
    const r = await fetch(API("params/schema"));
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
    label.className = "form-label mb-0 small me-2";
    label.textContent = def.label || key;
    div.appendChild(label);

    let input;
    if (def.type === "bool") {
      input = document.createElement("input");
      input.type = "checkbox";
      input.className = "form-check-input";
      input.dataset.paramKey = key;
      input.addEventListener("change", () =>
        api("params", { [key]: input.checked }));
    } else if (def.type === "choice") {
      input = document.createElement("select");
      input.className = "form-select form-select-sm";
      input.style.width = "130px";
      for (const c of (def.choices || [])) {
        const opt = document.createElement("option");
        opt.value = c;
        opt.textContent = c;
        input.appendChild(opt);
      }
      input.dataset.paramKey = key;
      input.addEventListener("change", () =>
        api("params", { [key]: input.value }));
    } else {
      input = document.createElement("input");
      input.type = "number";
      input.className = "form-control form-control-sm";
      input.style.width = "90px";
      if (def.min !== undefined) input.min = def.min;
      if (def.max !== undefined) input.max = def.max;
      input.dataset.paramKey = key;
      input.addEventListener("change", () =>
        api("params", { [key]: Number(input.value) }));
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
  el.className = "badge badge-" + status;
}

function setText(id, val) {
  document.getElementById(id).textContent = String(val);
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}
