"use strict";

// Approximate place names for the 24 half-degree areas (centroid-nearest town or landmark).
const NAMES = {
  r0c0: "Healdsburg", r0c1: "Lake Berryessa", r0c2: "Woodland", r0c3: "Roseville",
  r1c0: "Clear Lake", r1c1: "Williams", r1c2: "Yuba City", r1c3: "Grass Valley",
  r2c0: "Mendocino NF", r2c1: "Orland", r2c2: "Chico", r2c3: "Feather River",
  r3c0: "Yolla Bolly", r3c1: "Red Bluff", r3c2: "Lassen foothills", r3c3: "Lake Almanor",
  r4c0: "Weaverville", r4c1: "Redding", r4c2: "Burney", r4c3: "Hat Creek",
  r5c0: "Trinity Alps", r5c1: "Mount Shasta", r5c2: "McCloud River", r5c3: "Fall River Valley",
};
const RISK = [0.05, 0.12, 0.2463, 0.4]; // ramp breaks; 0.2463 is the model's alert threshold
const RISK_COLORS = ["--risk-0", "--risk-1", "--risk-2", "--risk-3", "--risk-4"].map((v) =>
  getComputedStyle(document.documentElement).getPropertyValue(v).trim());

const $ = (s) => document.querySelector(s);
const nf = new Intl.NumberFormat("en-US");
const pct = (v, d = 0) => `${(v * 100).toFixed(d)}%`;
const name = (id) => `${NAMES[id] ?? id}`;
const parseDay = (s) => new Date(`${s}T12:00:00`);
const fmtDay = (s, opts) => parseDay(s).toLocaleDateString("en-US", opts);
const riskColor = (s) => RISK_COLORS[RISK.findIndex((b) => s < b) === -1 ? 4 : RISK.findIndex((b) => s < b)];

let ov, map, fireLayer, badgeLayer;
const rects = {};
const cache = new Map();
let state = { date: null, data: null, selected: null, stripYear: null, timer: null, token: 0 };

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error ?? res.statusText);
  return res.json();
}

function setText(key, text) {
  document.querySelectorAll(`[data-k="${key}"]`).forEach((el) => (el.textContent = text));
}

/* ---------- Map ---------- */

// Heat colour ramp: [score, r, g, b, alpha]. Low risk stays nearly clear so the map shows through.
const HEAT = [
  [0.0, 255, 239, 153, 0.0],
  [0.06, 255, 239, 153, 0.28],
  [0.14, 255, 179, 92, 0.45],
  [0.25, 232, 64, 13, 0.55],
  [0.45, 175, 30, 0, 0.66],
  [0.75, 120, 20, 0, 0.74],
];
const VIEW_CELLS = 8; // the view spans 8 x 8 cells with the modelled 4 x 6 area in the middle
const CANVAS_PX = 256;

let grid, heatOverlay, heatCanvas, shown = null, anim = 0;

function heatColor(v) {
  if (v <= HEAT[0][0]) return HEAT[0].slice(1);
  for (let i = 1; i < HEAT.length; i++) {
    if (v <= HEAT[i][0]) {
      const [a, b] = [HEAT[i - 1], HEAT[i]];
      const t = (v - a[0]) / (b[0] - a[0]);
      return [1, 2, 3, 4].map((k) => a[k] + (b[k] - a[k]) * t);
    }
  }
  return HEAT[HEAT.length - 1].slice(1);
}

function initMap() {
  map = L.map("map", { zoomSnap: 0.25, scrollWheelZoom: false, attributionControl: true, maxZoom: 10, maxBoundsViscosity: 1 });
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 10, className: "basemap",
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map);

  const south = Math.min(...ov.cells.map((c) => c.bounds[0][0]));
  const west = Math.min(...ov.cells.map((c) => c.bounds[0][1]));
  const north = Math.max(...ov.cells.map((c) => c.bounds[1][0]));
  const east = Math.max(...ov.cells.map((c) => c.bounds[1][1]));
  const step = ov.cells[0].bounds[1][0] - ov.cells[0].bounds[0][0];
  const rows = Math.round((north - south) / step), cols = Math.round((east - west) / step);
  const s0 = south - Math.floor((VIEW_CELLS - rows) / 2) * step;
  const w0 = west - Math.floor((VIEW_CELLS - cols) / 2) * step;
  const span = VIEW_CELLS * step;
  const outer = L.latLngBounds([s0, w0], [s0 + span, w0 + span]);
  grid = { south, west, north, east, step, rows, cols, s0, w0, span };

  heatCanvas = document.createElement("canvas");
  heatCanvas.width = heatCanvas.height = CANVAS_PX;
  heatOverlay = L.imageOverlay(heatCanvas.toDataURL(), outer, { interactive: false, className: "heat" }).addTo(map);

  // Invisible hit areas so each modelled area can still be hovered and clicked.
  for (const cell of ov.cells) {
    rects[cell.id] = L.rectangle(cell.bounds, { stroke: false, fillOpacity: 0, fillColor: "#111111" })
      .on("click", () => selectCell(cell.id))
      .on("mouseover", () => rects[cell.id].setStyle({ stroke: true, color: "#111111", weight: 1, opacity: 0.5, dashArray: "4 4" }))
      .on("mouseout", () => outline(cell.id))
      .bindTooltip("", { sticky: true })
      .addTo(map);
  }
  badgeLayer = L.layerGroup().addTo(map);
  fireLayer = L.layerGroup().addTo(map);

  // Lock the view: can't zoom out past the region or pan away from Northern California.
  const lock = () => {
    map.invalidateSize();
    const z = map.getBoundsZoom(outer, false);
    map.setMinZoom(z);
    map.setMaxBounds(outer.pad(0.02));
    if (map.getZoom() <= z || !map.getBounds().intersects(outer)) map.fitBounds(outer);
  };
  map.fitBounds(outer);
  lock();
  new ResizeObserver(lock).observe($("#map"));
}

function outline(id) {
  const on = state.selected === id;
  rects[id].setStyle(on ? { stroke: true, color: "#111111", weight: 2, opacity: 0.9, dashArray: "5 4" } : { stroke: false });
}

// Smooth only supported scores. Outside coverage and missing cells remain fully transparent.
function drawHeat(scores) {
  const g = grid, N = CANVAS_PX;
  const ctx = heatCanvas.getContext("2d");
  const img = ctx.createImageData(N, N);
  const values = Array.from({ length: g.rows }, (_, r) =>
    Array.from({ length: g.cols }, (_, c) => scores[`r${r}c${c}`]));
  for (let y = 0; y < N; y++) {
    const lat = g.s0 + g.span - ((y + 0.5) / N) * g.span;
    if (lat < g.south || lat >= g.north) continue;
    const row = Math.floor((lat - g.south) / g.step);
    const fr = Math.min(g.rows - 1, Math.max(0, (lat - g.south) / g.step - 0.5));
    const r0 = Math.min(g.rows - 2, Math.floor(fr)), tr = fr - r0;
    for (let x = 0; x < N; x++) {
      const lon = g.w0 + ((x + 0.5) / N) * g.span;
      if (lon < g.west || lon >= g.east) continue;
      const col = Math.floor((lon - g.west) / g.step);
      if (!Number.isFinite(values[row][col])) continue;
      const fc = Math.min(g.cols - 1, Math.max(0, (lon - g.west) / g.step - 0.5));
      const c0 = Math.min(g.cols - 2, Math.floor(fc)), tc = fc - c0;
      // Normalize over finite neighbors so a missing score is never treated as zero risk.
      let total = 0, weight = 0;
      for (let dr = 0; dr <= 1; dr++) {
        for (let dc = 0; dc <= 1; dc++) {
          const value = values[r0 + dr][c0 + dc];
          const w = (dr ? tr : 1 - tr) * (dc ? tc : 1 - tc);
          if (Number.isFinite(value)) { total += value * w; weight += w; }
        }
      }
      if (!weight) continue;
      const [hr, hg, hb, a] = heatColor(total / weight);
      const i = (y * N + x) * 4;
      if (a > 0) {
        img.data[i] = hr;
        img.data[i + 1] = hg;
        img.data[i + 2] = hb;
        img.data[i + 3] = a * 255;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  heatOverlay.setUrl(heatCanvas.toDataURL());
}

// Ease from the scores on screen to the new day's scores.
function animateHeat(target) {
  cancelAnimationFrame(anim);
  target = Object.fromEntries(Object.entries(target).filter(([, score]) => Number.isFinite(score)));
  const from = shown ?? {};
  const dur = state.timer ? Math.min(220, Number($("#speed").value) * 0.7) : 380;
  const t0 = performance.now();
  const frame = (now) => {
    const t = Math.min(1, (now - t0) / dur), e = t * (2 - t);
    shown = Object.fromEntries(Object.keys(target).map((k) => {
      const start = Number.isFinite(from[k]) ? from[k] : 0;
      return [k, start + (target[k] - start) * e];
    }));
    drawHeat(shown);
    if (t < 1) anim = requestAnimationFrame(frame);
  };
  anim = requestAnimationFrame(frame);
}

function paintMap(day) {
  badgeLayer.clearLayers();
  fireLayer.clearLayers();
  animateHeat(Object.fromEntries(day.cells.map((c) => [c.id, c.score])));
  for (const c of day.cells) {
    outline(c.id);
    rects[c.id].setTooltipContent(`<b>${name(c.id)}</b><br>Rank ${c.rank} of 24 · score ${c.score.toFixed(2)}${c.fires ? `<br>${c.fires} fire${c.fires > 1 ? "s" : ""} discovered` : ""}`);
    if (c.rank <= ov.metrics.review_k) {
      L.marker(rects[c.id].getBounds().getCenter(), {
        interactive: false,
        icon: L.divIcon({ className: "", html: `<span class="rank-badge">${c.rank}</span>`, iconSize: [26, 26], iconAnchor: [13, 13] }),
      }).addTo(badgeLayer);
    }
  }
  for (const [lat, lon] of day.fires) {
    L.circleMarker([lat, lon], { radius: 4.5, color: "#111111", weight: 2, fillColor: "#ffffff", fillOpacity: 1, interactive: false }).addTo(fireLayer);
  }
}

/* ---------- Day panel ---------- */

function renderPanel(day) {
  $("#day-issued").textContent = `Outlook issued evening of ${fmtDay(day.issued, { weekday: "short", month: "short", day: "numeric" })}`;
  $("#day-title").textContent = fmtDay(day.date, { weekday: "long", month: "long", day: "numeric", year: "numeric" });
  const s = day.summary;
  const cls = s.fire_cells === 0 ? "" : s.caught > 0 ? "good" : "bad";
  $("#verdict").innerHTML = `
    <div><b>${s.fires}</b><span>fires discovered</span></div>
    <div><b>${s.fire_cells}</b><span>areas with fire</span></div>
    <div class="${cls}"><b>${s.caught}/${ov.metrics.review_k}</b><span>top picks with fire</span></div>`;
  $("#live-note").textContent = `Scored live on IBM LinuxONE (${day.machine}) in ${day.live_ms.toFixed(1)} ms.`;

  const top = day.cells.slice(0, ov.metrics.review_k);
  $("#watch").innerHTML = top.map((c) => `
    <button class="watch-item${state.selected === c.id ? " active" : ""}" data-cell="${c.id}">
      <span class="rank-badge">${c.rank}</span>
      <span class="name">${name(c.id)}</span>
      <span class="score" style="background:${riskColor(c.score)};color:${c.score >= RISK[2] ? "#fff" : "#111"}">${c.score.toFixed(2)}</span>
      <ul>${(c.reasons.length ? c.reasons : ["Ranked high mostly on its fire history for this month"]).map((r) => `<li>${r}</li>`).join("")}</ul>
      <span class="result ${c.fires ? "hit" : "miss"}">${c.fires ? `${c.fires} fire${c.fires > 1 ? "s" : ""} discovered here` : "No fire recorded here"}</span>
    </button>`).join("");
  document.querySelectorAll(".watch-item").forEach((el) => el.addEventListener("click", () => selectCell(el.dataset.cell)));
  renderCell();
}

function anomalyTag(z, invert = false) {
  if (z == null || Math.abs(z) < 1) return "";
  const up = z > 0;
  const bad = invert ? !up : up;
  return `<em class="${bad ? "up" : "down"}">${Math.abs(z).toFixed(1)}σ ${up ? "above" : "below"} normal</em>`;
}

function renderCell() {
  const card = $("#cell-card");
  const c = state.data?.cells.find((x) => x.id === state.selected);
  if (!c) { card.hidden = true; return; }
  const w = c.weather, a = c.anomaly;
  card.hidden = false;
  card.innerHTML = `
    <p class="eyebrow">Area ${c.id.toUpperCase()}</p>
    <h4>${name(c.id)}</h4>
    <p class="small muted">Rank ${c.rank} of 24 · score ${c.score.toFixed(3)}${c.alert ? " · above alert line" : ""}</p>
    <div class="wx">
      <div><span>Yesterday's high</span><b>${w.tmax.toFixed(1)} °C</b>${anomalyTag(a.heat)}</div>
      <div><span>Lowest humidity</span><b>${w.rh.toFixed(0)}%</b>${anomalyTag(a.humidity, true)}</div>
      <div><span>Max wind</span><b>${w.wind.toFixed(1)} m/s</b>${anomalyTag(a.wind)}</div>
      <div><span>Rain, last 30 days</span><b>${w.rain30.toFixed(1)} mm</b>${anomalyTag(a.rain30, true)}</div>
      <div><span>Dry days in a row</span><b>${w.dry_days >= 30 ? "30+" : w.dry_days}</b></div>
      <div><span>Past fire rate (this month)</span><b>${pct(c.past_rate)}</b></div>
    </div>
    <p class="small ${c.fires ? "hit" : "miss"}" style="margin-top:12px">${c.fires ? `${c.fires} fire${c.fires > 1 ? "s" : ""} discovered here that day` : "No fire recorded here that day"}</p>`;
  card.querySelectorAll("em").forEach((e) => (e.style.cssText = "display:block;font-style:normal;font-size:11px"));
}

function selectCell(id) {
  state.selected = state.selected === id ? null : id;
  if (state.data) { paintMap(state.data); renderPanel(state.data); }
}

/* ---------- Season strip ---------- */

function renderStrip(year) {
  const t = ov.timeline;
  const idx = t.dates.map((d, i) => (d.startsWith(String(year)) ? i : -1)).filter((i) => i >= 0);
  const max = Math.max(...idx.map((i) => t.fires[i]), 1);
  const svg = $("#strip");
  svg.setAttribute("viewBox", `0 0 ${idx.length} 100`);
  svg.innerHTML = idx.map((i, x) => {
    const h = t.fires[i] ? Math.max(6, (t.fires[i] / max) * 92) : 0;
    const fill = t.caught[i] > 0 ? "var(--phoenix)" : "var(--stone)";
    return h ? `<rect x="${x + 0.12}" y="${100 - h}" width="0.76" height="${h}" fill="${fill}"><title>${t.dates[i]}: ${t.fires[i]} fires</title></rect>` : "";
  }).join("") + `<rect id="cursor" x="0" y="0" width="1" height="100" fill="rgba(17,17,17,0.18)"/>`;
  svg.dataset.start = idx[0];
  svg.dataset.len = idx.length;
  state.stripYear = year;
  $("#strip-year").textContent = year;
}

function moveCursor(date) {
  const year = Number(date.slice(0, 4));
  if (year !== state.stripYear) renderStrip(year);
  const svg = $("#strip");
  const i = ov.timeline.dates.indexOf(date) - Number(svg.dataset.start);
  svg.querySelector("#cursor").setAttribute("x", i);
}

/* ---------- Navigation ---------- */

async function loadDay(date) {
  const token = ++state.token;
  state.date = date;
  $("#date").value = date;
  moveCursor(date);
  let day = cache.get(date);
  if (!day) {
    day = await getJSON(`/api/day?date=${date}`);
    cache.set(date, day);
    if (cache.size > 400) cache.delete(cache.keys().next().value);
  }
  if (token !== state.token) return;
  state.data = day;
  paintMap(day);
  renderPanel(day);
  history.replaceState(null, "", `#d=${date}`);
}

function step(delta) {
  const dates = ov.timeline.dates;
  const i = dates.indexOf(state.date) + delta;
  if (i < 0 || i >= dates.length) { stopPlay(); return false; }
  loadDay(dates[i]);
  return true;
}

function stopPlay() {
  clearInterval(state.timer);
  state.timer = null;
  $("#play").textContent = "Play season";
  $("#play").setAttribute("aria-pressed", "false");
}

function togglePlay() {
  if (state.timer) return stopPlay();
  $("#play").textContent = "Pause";
  $("#play").setAttribute("aria-pressed", "true");
  state.timer = setInterval(() => step(1), Number($("#speed").value));
}

function initControls() {
  const input = $("#date");
  input.min = ov.dates.first;
  input.max = ov.dates.last;
  input.addEventListener("change", () => input.value && loadDay(input.value));
  $("#prev").addEventListener("click", () => step(-1));
  $("#next").addEventListener("click", () => step(1));
  $("#play").addEventListener("click", togglePlay);
  $("#speed").addEventListener("change", () => { if (state.timer) { stopPlay(); togglePlay(); } });
  $("#strip").addEventListener("click", (e) => {
    const svg = e.currentTarget;
    const r = svg.getBoundingClientRect();
    const i = Math.floor(((e.clientX - r.left) / r.width) * Number(svg.dataset.len));
    const d = ov.timeline.dates[Number(svg.dataset.start) + Math.max(0, i)];
    if (d) loadDay(d);
  });

  // Busiest days (by fires discovered) as quick jumps.
  const t = ov.timeline;
  const busiest = t.dates.map((d, i) => ({ d, f: t.fires[i] })).sort((a, b) => b.f - a.f).slice(0, 6).sort((a, b) => (a.d < b.d ? -1 : 1));
  $("#jumps").innerHTML = busiest.map((b) => `<button data-d="${b.d}">${fmtDay(b.d, { month: "short", day: "numeric", year: "numeric" })} · ${b.f} fires</button>`).join(" ");
  $("#jumps").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { stopPlay(); loadDay(b.dataset.d); }));

  return busiest;
}

/* ---------- Numbers and charts ---------- */

function fillNumbers() {
  const m = ov.metrics, r = ov.runtime;
  setText("fires", nf.format(ov.data.fire_records_total));
  setText("recall", pct(m.review_recall));
  setText("ms", `${r.score_day_ms.toFixed(1)} ms`);
  setText("trees", ov.model.trees);
  setText("features", ov.model.features);
  setText("prev", pct(m.prevalence));
  setText("noacc", pct(m.always_no_accuracy));
  setText("acc", pct(m.accuracy));

  const chip = $("#runtime-chip");
  chip.classList.add("ok");
  chip.innerHTML = `<span class="dot"></span>Live on IBM LinuxONE · ${r.machine}`;

  $("#results-intro").textContent = `${nf.format(m.rows)} area-days (24 areas × 3 years), ${nf.format(m.positives)} of them with at least one fire. Recomputed live from the model each time the server starts.`;

  const tiles = [
    { tone: "tone-petal", label: "Average precision", value: m.average_precision.toFixed(3), vs: `Random guessing: ${m.prevalence.toFixed(3)} · past fire rate alone: ${m.baseline_average_precision.toFixed(3)}` },
    { tone: "tone-canary", label: "ROC AUC", value: m.roc_auc.toFixed(3), vs: "Random guessing: 0.500" },
    { tone: "tone-mint", label: `Fire days caught checking ${m.review_k} of 24 areas`, value: pct(m.review_recall), vs: `Random picks: ${pct(m.review_random, 1)} · ${(m.review_recall / m.review_random).toFixed(1)}× better` },
    { tone: "tone-violet", label: "Fires in the test period", value: nf.format(ov.data.replay_fires), vs: `${nf.format(m.positives)} area-days with fire, none seen in training` },
  ];
  $("#metric-grid").innerHTML = tiles.map((t) => `<div class="metric ${t.tone}"><span class="label">${t.label}</span><span class="value">${t.value}</span><span class="vs">${t.vs}</span></div>`).join("");

  const c = m.confusion;
  $("#confusion").innerHTML = `
    <div class="hd"></div><div class="hd">Fire happened</div><div class="hd">No fire</div>
    <div class="hd">Alert raised</div><div class="good"><b>${nf.format(c.tp)}</b>caught</div><div class="bad"><b>${nf.format(c.fp)}</b>false alarms</div>
    <div class="hd">No alert</div><div class="bad"><b>${nf.format(c.fn)}</b>missed</div><div class="good"><b>${nf.format(c.tn)}</b>correctly quiet</div>`;

  const mo = ov.model;
  const runs = [
    ["Architecture", r.machine, r.system],
    ["CPUs", r.cpus, "LinuxONE Community Cloud"],
    ["Python / scikit-learn", `${r.python} / ${r.sklearn}`, `numpy ${r.numpy} · pandas ${r.pandas}`],
    ["Model size", `${(mo.size_bytes / 1024).toFixed(0)} KB`, `${mo.trees} trees · ${mo.features} inputs`],
    ["Load model + data", `${nf.format(Math.round(r.load_ms))} ms`, "at server start"],
    ["Score all 3 years", `${nf.format(Math.round(r.score_all_ms))} ms`, `${nf.format(m.rows)} area-days`],
    ["Score one day", `${r.score_day_ms.toFixed(1)} ms`, "24 areas, median of 20 runs"],
    ["Model fingerprint", (mo.sha256 ?? "").slice(0, 12), "SHA-256, matches manifest"],
  ];
  $("#run-grid").innerHTML = runs.map(([k, v, s]) => `<div><span>${k}</span><b>${v}</b><small>${s}</small></div>`).join("");
}

function drawCharts() {
  if (!window.Chart) return;
  Chart.defaults.font.family = "Inter, system-ui, sans-serif";
  Chart.defaults.color = "#6d6c6b";
  Chart.defaults.plugins.legend.display = false;
  Chart.defaults.maintainAspectRatio = false;
  const grid = { color: "rgba(17,17,17,0.06)" };
  const m = ov.metrics;

  new Chart($("#importance-chart"), {
    type: "bar",
    data: { labels: ov.importance.slice(0, 8).map((d) => d.feature), datasets: [{ data: ov.importance.slice(0, 8).map((d) => d.importance), backgroundColor: "#e8400d", borderRadius: 4 }] },
    options: { indexAxis: "y", scales: { x: { grid, title: { display: true, text: "Drop in average precision" } }, y: { grid: { display: false }, ticks: { color: "#111" } } } },
  });

  new Chart($("#pr-chart"), {
    type: "line",
    data: {
      datasets: [
        { label: "Model", data: m.pr_curve.map(([x, y]) => ({ x, y })), borderColor: "#e8400d", backgroundColor: "rgba(232,64,13,0.12)", fill: true, pointRadius: 0, tension: 0.15 },
        { label: "Random", data: [{ x: 0, y: m.prevalence }, { x: 1, y: m.prevalence }], borderColor: "#b1b1af", borderDash: [6, 6], pointRadius: 0 },
      ],
    },
    options: {
      plugins: { legend: { display: true, labels: { boxWidth: 12 } } },
      scales: { x: { type: "linear", min: 0, max: 1, grid, title: { display: true, text: "Recall (share of fire days caught)" } }, y: { min: 0, max: 1, grid, title: { display: true, text: "Precision" } } },
    },
  });

  new Chart($("#year-chart"), {
    type: "bar",
    data: { labels: m.years.map((y) => y.year), datasets: [{ data: m.years.map((y) => y.ap), backgroundColor: ["#ffd7f0", "#ffef99", "#b7efb2"], borderRadius: 6 }] },
    options: { scales: { y: { min: 0, max: 0.6, grid }, x: { grid: { display: false }, ticks: { color: "#111" } } } },
  });

  new Chart($("#review-chart"), {
    type: "bar",
    data: { labels: ["Random 3 areas", "Model's top 3"], datasets: [{ data: [m.review_random, m.review_recall], backgroundColor: ["#ecebea", "#e8400d"], borderRadius: 6 }] },
    options: { indexAxis: "y", scales: { x: { min: 0, max: 0.5, grid, ticks: { callback: (v) => pct(v) } }, y: { grid: { display: false }, ticks: { color: "#111" } } } },
  });
}

/* ---------- Boot ---------- */

async function boot() {
  try {
    ov = await getJSON("/api/overview");
  } catch (e) {
    $("#runtime-chip").textContent = "Server offline";
    $("#day-title").textContent = "Couldn't reach the model server.";
    return;
  }
  fillNumbers();
  initMap();
  const busiest = initControls();
  drawCharts();
  const fromHash = location.hash.match(/d=(\d{4}-\d{2}-\d{2})/)?.[1];
  const start = fromHash && ov.timeline.dates.includes(fromHash) ? fromHash : busiest[busiest.length - 1].d;
  loadDay(start);
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, select")) return;
    if (e.key === "ArrowLeft") step(-1);
    if (e.key === "ArrowRight") step(1);
  });
}

boot();
