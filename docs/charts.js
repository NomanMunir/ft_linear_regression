/* Tiny SVG chart helpers for the study docs. No libraries, works offline.
   DATA is a copy of ../data.csv so the pages work when opened from disk. */

const DATA = {
  km: [240000, 139800, 150500, 185530, 176000, 114800, 166800, 89000, 144500, 84000,
       82029, 63060, 74000, 97500, 67000, 76025, 48235, 93000, 60949, 65674, 54000,
       68500, 22899, 61789],
  price: [3650, 3800, 4400, 4450, 5250, 5350, 5800, 5990, 5999, 6200, 6390, 6390,
          6600, 6800, 6800, 6900, 6900, 6990, 7490, 7555, 7990, 7990, 7990, 8290],
};

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs, parent) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
}

function niceTicks(min, max, count) {
  const span = max - min || 1;
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(f => f * mag).find(s => span / s <= count) || 10 * mag;
  const ticks = [];
  for (let t = Math.ceil(min / step) * step; t <= max + 1e-9; t += step) ticks.push(+t.toFixed(10));
  return ticks;
}

function fmt(n) {
  const a = Math.abs(n);
  if (a >= 1e6 || (a > 0 && a < 1e-3)) return n.toExponential(1);
  if (a >= 1000) return (n / 1000).toFixed(a >= 10000 ? 0 : 1) + "k";
  return String(+n.toFixed(3));
}

/* Base frame: axes, grid, ticks. Returns {svg, sx, sy}. */
function frame(container, opts) {
  const W = opts.width || 720, H = opts.height || 400;
  const m = { l: 62, r: 16, t: 14, b: 46 };
  container.innerHTML = "";
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.label || "chart" }, container);
  const [x0, x1] = opts.xDomain, [y0, y1] = opts.yDomain;
  const sx = x => m.l + (x - x0) / (x1 - x0) * (W - m.l - m.r);
  const sy = y => H - m.b - (y - y0) / (y1 - y0) * (H - m.t - m.b);
  for (const t of niceTicks(x0, x1, 6)) {
    svgEl("line", { class: "gridline", x1: sx(t), x2: sx(t), y1: m.t, y2: H - m.b }, svg);
    svgEl("text", { x: sx(t), y: H - m.b + 18, "text-anchor": "middle" }, svg).textContent = fmt(t);
  }
  for (const t of niceTicks(y0, y1, 5)) {
    svgEl("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t) }, svg);
    svgEl("text", { x: m.l - 8, y: sy(t) + 4, "text-anchor": "end" }, svg).textContent = fmt(t);
  }
  svgEl("line", { class: "axis", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }, svg);
  svgEl("line", { class: "axis", x1: m.l, x2: m.l, y1: m.t, y2: H - m.b }, svg);
  if (opts.xLabel) svgEl("text", { x: (m.l + W - m.r) / 2, y: H - 8, "text-anchor": "middle" }, svg).textContent = opts.xLabel;
  if (opts.yLabel) svgEl("text", { x: 14, y: (m.t + H - m.b) / 2, "text-anchor": "middle",
    transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})` }, svg).textContent = opts.yLabel;
  const clip = svgEl("clipPath", { id: "clip-" + Math.random().toString(36).slice(2) }, svg);
  svgEl("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b }, clip);
  const plot = svgEl("g", { "clip-path": `url(#${clip.id})` }, svg);
  return { svg, plot, sx, sy };
}

/* Scatter of the dataset + the line theta0 + theta1 * km (+ optional residuals). */
function drawScatter(container, opts) {
  const xDomain = opts.xDomain || [0, 250000];
  const yDomain = opts.yDomain || [0, 10000];
  const { plot, sx, sy } = frame(container, { ...opts, xDomain, yDomain,
    xLabel: opts.xLabel || "mileage (km)", yLabel: opts.yLabel || "price",
    label: "car price against mileage" });
  const f = x => opts.theta0 + opts.theta1 * x;
  if (opts.residuals) {
    DATA.km.forEach((x, i) => svgEl("line", { class: "residual",
      x1: sx(x), x2: sx(x), y1: sy(DATA.price[i]), y2: sy(f(x)) }, plot));
  }
  if (opts.theta0 !== undefined) {
    svgEl("line", { class: "line", x1: sx(xDomain[0]), x2: sx(xDomain[1]),
      y1: sy(f(xDomain[0])), y2: sy(f(xDomain[1])) }, plot);
  }
  DATA.km.forEach((x, i) => svgEl("circle", { class: "point", cx: sx(x), cy: sy(DATA.price[i]), r: 4.5 }, plot));
}

/* Generic curve, e.g. cost against iteration or cost against theta1. */
function drawCurve(container, xs, ys, opts) {
  const finite = ys.filter(Number.isFinite);
  const yMin = opts.yMin !== undefined ? opts.yMin : Math.min(...finite);
  const yMax = opts.yMax !== undefined ? opts.yMax : Math.max(...finite);
  const { plot, sx, sy } = frame(container, { ...opts,
    xDomain: [xs[0], xs[xs.length - 1] === xs[0] ? xs[0] + 1 : xs[xs.length - 1]],
    yDomain: [yMin, yMax === yMin ? yMin + 1 : yMax] });
  const pts = xs.map((x, i) => [x, ys[i]]).filter(p => Number.isFinite(p[1]));
  svgEl("path", { class: "curve", d: pts.map((p, i) => (i ? "L" : "M") + sx(p[0]).toFixed(1) + "," + sy(p[1]).toFixed(1)).join("") }, plot);
  (opts.markers || []).forEach(mk => {
    svgEl("circle", { cx: sx(mk.x), cy: sy(mk.y), r: 6, fill: mk.color || "var(--blue)" }, plot);
    if (mk.label) svgEl("text", { x: sx(mk.x) + 9, y: sy(mk.y) - 8 }, plot).textContent = mk.label;
  });
}

/* Same maths as train.py, in JavaScript, for the interactive pages. */
function meanStd(values) {
  const m = values.length;
  const mean = values.reduce((a, b) => a + b, 0) / m;
  const variance = values.reduce((a, v) => a + (v - mean) * (v - mean), 0) / m;
  return { mean, std: Math.sqrt(variance) };
}

function costOf(xs, ys, t0, t1) {
  let total = 0;
  for (let i = 0; i < xs.length; i++) { const e = t0 + t1 * xs[i] - ys[i]; total += e * e; }
  return total / (2 * xs.length);
}

function gradientStep(xs, ys, t0, t1, lr) {
  const m = xs.length;
  let s0 = 0, s1 = 0;
  for (let i = 0; i < m; i++) { const e = (t0 + t1 * xs[i]) - ys[i]; s0 += e; s1 += e * xs[i]; }
  const tmp0 = lr * (1 / m) * s0, tmp1 = lr * (1 / m) * s1;
  return [t0 - tmp0, t1 - tmp1];   // simultaneous update
}
