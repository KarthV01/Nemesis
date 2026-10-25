/* Minimal dependency-free SVG line charts for the Nemesis dashboard.
 *
 * One reusable renderer: each chart is a small multiple of the same form —
 * a faint per-episode "raw" line plus a bold trailing-mean "rolling" line,
 * one y-axis only, recessive gridlines, and a shared crosshair + tooltip on
 * hover. Series identity is carried by a legend/direct label and a semantic
 * color, never by color alone. No build step, no external library. */
(function () {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  const W = 620, H = 150;
  const M = { l: 36, r: 10, t: 10, b: 18 };
  const PLOT_W = W - M.l - M.r;
  const PLOT_H = H - M.t - M.b;

  // One tooltip element shared by every chart on the page.
  let tip = null;
  function tooltip() {
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "tooltip";
      document.body.appendChild(tip);
    }
    return tip;
  }

  function el(name, attrs) {
    const node = document.createElementNS(NS, name);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    return node;
  }

  function extent(points) {
    let lo = Infinity, hi = -Infinity;
    for (const p of points) { if (p.y < lo) lo = p.y; if (p.y > hi) hi = p.y; }
    return [lo, hi];
  }

  function pathFor(points, sx, sy) {
    return points.map((p, i) => (i ? "L" : "M") + sx(p.x).toFixed(1) + " " + sy(p.y).toFixed(1)).join(" ");
  }

  /* cfg = {
   *   series: [{ key, label, color, raw:[{x,y}], roll:[{x,y}], dots?:bool }],
   *   yDomain: [min,max] | null (auto),
   *   fmt: (y)=>string
   * } */
  function renderLineChart(svgHost, cfg) {
    const fmt = cfg.fmt || ((y) => (Math.round(y * 1000) / 1000).toString());

    // Collect all points to size the axes.
    const all = [];
    for (const s of cfg.series) {
      for (const p of (s.raw || [])) all.push(p);
      for (const p of (s.roll || [])) all.push(p);
    }
    svgHost.innerHTML = "";
    if (!all.length) {
      svgHost.appendChild(Object.assign(document.createElement("div"),
        { className: "empty", textContent: "waiting for episodes…" }));
      return;
    }

    let xmin = Infinity, xmax = -Infinity;
    for (const p of all) { if (p.x < xmin) xmin = p.x; if (p.x > xmax) xmax = p.x; }
    let [ymin, ymax] = cfg.yDomain ? cfg.yDomain : extent(all);
    if (cfg.yDomain == null) {
      if (ymin === ymax) { ymin -= 1; ymax += 1; }
      const pad = (ymax - ymin) * 0.08;
      ymin -= pad; ymax += pad;
      if (cfg.yMin0 && ymin < 0) ymin = 0;  // counts never go below zero
    }
    const xspan = xmax - xmin || 1;
    const yspan = ymax - ymin || 1;
    const sx = (x) => M.l + ((x - xmin) / xspan) * PLOT_W;
    const sy = (y) => M.t + (1 - (y - ymin) / yspan) * PLOT_H;

    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none",
      role: "img", "aria-label": cfg.ariaLabel || "" });

    // Horizontal gridlines + y tick labels (min, mid, max).
    for (const frac of [0, 0.5, 1]) {
      const yVal = ymin + frac * yspan;
      const y = sy(yVal);
      svg.appendChild(el("line", { class: frac === 0 ? "axis-line" : "grid-line",
        x1: M.l, x2: W - M.r, y1: y, y2: y }));
      const label = el("text", { class: "axis-label", x: M.l - 5, y: y + 3, "text-anchor": "end" });
      label.textContent = fmt(yVal);
      svg.appendChild(label);
    }
    // x endpoints.
    for (const [xv, anchor, xpix] of [[xmin, "start", M.l], [xmax, "end", W - M.r]]) {
      const t = el("text", { class: "axis-label", x: xpix, y: H - 5, "text-anchor": anchor });
      t.textContent = "ep " + xv;
      svg.appendChild(t);
    }

    // Series: faint raw line, bold rolling line, optional raw dots.
    for (const s of cfg.series) {
      if (s.raw && s.raw.length) {
        svg.appendChild(el("path", { class: "series-raw", stroke: s.color, d: pathFor(s.raw, sx, sy) }));
        if (s.dots) {
          for (const p of s.raw) svg.appendChild(el("circle",
            { class: "dot", cx: sx(p.x), cy: sy(p.y), r: 2.2, fill: s.color }));
        }
      }
      if (s.roll && s.roll.length) {
        svg.appendChild(el("path", { class: "series-roll", stroke: s.color, d: pathFor(s.roll, sx, sy) }));
      }
    }

    // Hover crosshair + markers (built once, shown on move).
    const cross = el("line", { class: "crosshair", y1: M.t, y2: M.t + PLOT_H, x1: -10, x2: -10,
      style: "display:none" });
    svg.appendChild(cross);
    const markers = cfg.series.map((s) => {
      const c = el("circle", { r: 3.5, fill: s.color, stroke: "#0b0d10", "stroke-width": 1.5,
        style: "display:none" });
      svg.appendChild(c);
      return c;
    });

    // Lookup tables keyed by x for the hover readout (prefer rolling value).
    const lookups = cfg.series.map((s) => {
      const m = new Map();
      for (const p of (s.raw || [])) m.set(p.x, p.y);
      for (const p of (s.roll || [])) m.set(p.x, p.y); // rolling wins if both
      const xs = [...m.keys()].sort((a, b) => a - b);
      return { map: m, xs };
    });
    const allXs = [...new Set(all.map((p) => p.x))].sort((a, b) => a - b);

    const overlay = el("rect", { x: M.l, y: M.t, width: PLOT_W, height: PLOT_H,
      fill: "transparent", style: "cursor:crosshair" });
    svg.appendChild(overlay);

    function nearestX(px) {
      const xVal = xmin + ((px - M.l) / PLOT_W) * xspan;
      let best = allXs[0], bestD = Infinity;
      for (const x of allXs) { const d = Math.abs(x - xVal); if (d < bestD) { bestD = d; best = x; } }
      return best;
    }

    function onMove(ev) {
      const rect = svg.getBoundingClientRect();
      const px = ((ev.clientX - rect.left) / rect.width) * W;
      const x = nearestX(px);
      cross.setAttribute("x1", sx(x)); cross.setAttribute("x2", sx(x));
      cross.style.display = "";
      const t = tooltip();
      let rows = `<div class="tt-ep">episode ${x}</div>`;
      cfg.series.forEach((s, i) => {
        const y = lookups[i].map.get(x);
        if (y == null) { markers[i].style.display = "none"; return; }
        markers[i].setAttribute("cx", sx(x));
        markers[i].setAttribute("cy", sy(y));
        markers[i].style.display = "";
        rows += `<div class="tt-row"><span class="swatch" style="background:${s.color}"></span>` +
          `<span>${s.label}</span><span class="v">${fmt(y)}</span></div>`;
      });
      t.innerHTML = rows;
      t.style.display = "block";
      t.style.left = Math.min(ev.clientX + 14, window.innerWidth - t.offsetWidth - 8) + "px";
      t.style.top = Math.min(ev.clientY + 14, window.innerHeight - t.offsetHeight - 8) + "px";
    }
    function onLeave() {
      cross.style.display = "none";
      markers.forEach((m) => (m.style.display = "none"));
      if (tip) tip.style.display = "none";
    }
    overlay.addEventListener("pointermove", onMove);
    overlay.addEventListener("pointerleave", onLeave);

    svgHost.appendChild(svg);
  }

  window.NemesisCharts = { renderLineChart };
})();
