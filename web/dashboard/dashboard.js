/* Nemesis dashboard controller.
 *
 * Left panel: the live commentator — Server-Sent Events from /events, styled by
 * EventKind (and nemesis.live's attack/defend aliases), exactly the dict shape
 * nemesis.live sends. Right panel: evaluation metric time-series polled from
 * /metrics, drawn by charts.js. Win rate alone misleads, so the metric vector
 * (reward per side, attack success, defender availability, false positives) is
 * shown alongside it. */
(function () {
  "use strict";

  const MAX_LINES = 1200;
  const POLL_MS = 3000;
  const el = (id) => document.getElementById(id);

  // ---- styling of the feed by kind ---------------------------------------- //
  const TAGS = {
    system: "SYSTEM", red: "RED", blue: "BLUE", attack: "ATTACK", defend: "DEFEND",
    reward: "REWARD", train: "TRAIN", metric: "METRIC",
  };
  const LEGEND = [
    ["system", "var(--system)"], ["red", "var(--red)"], ["blue", "var(--blue)"],
    ["reward", "var(--reward-up)"], ["train", "var(--train)"], ["metric", "var(--metric)"],
  ];

  const log = el("log");
  const prompt = el("prompt");
  let paused = false;
  let held = [];
  let total = 0;

  (function buildFeedLegend() {
    const box = el("feed-legend");
    for (const [kind, color] of LEGEND) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.innerHTML = `<span class="swatch" style="background:${color}"></span>${kind}`;
      box.appendChild(chip);
    }
  })();

  function span(cls, text) {
    const s = document.createElement("span");
    s.className = cls;
    s.textContent = text;
    return s;
  }

  function render(e) {
    const line = document.createElement("div");
    line.className = "line " + e.kind;
    if (e.kind === "reward" && e.reward < 0) line.classList.add("down");

    const ts = new Date((e.t || Date.now() / 1000) * 1000).toLocaleTimeString([], { hour12: false });
    line.append(span("ts", ts), span("tag", TAGS[e.kind] || String(e.kind).toUpperCase()));

    const msg = span("msg", "");
    if (e.agent) msg.append(span("who", `${e.agent}${e.role ? " [" + e.role + "]" : ""}  `));
    msg.append(span(e.success === "FALSE" ? "fail" : "", e.text));
    if (e.success === "FALSE") msg.append(span("who", "  (failed)"));
    line.append(msg);

    const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
    log.insertBefore(line, prompt);
    while (log.children.length > MAX_LINES) log.removeChild(log.firstChild);
    if (atBottom) log.scrollTop = log.scrollHeight;
  }

  function update(e) {
    total += 1;
    el("count").textContent = total;
    if (e.episode !== undefined) el("episode").textContent = e.episode;
    if (e.step !== undefined) el("step").textContent = e.step;
    if (e.kind === "system" && /started/i.test(e.text || "")) el("step").textContent = "0";
    if (paused) held.push(e); else render(e);
  }

  function setStatus(state) {
    const s = el("status");
    s.className = "stat " + (state === "live" ? "live" : state === "down" ? "down" : "");
    s.querySelector("b").textContent = state;
  }

  el("pause").onclick = () => {
    paused = !paused;
    el("pause").textContent = paused ? "[resume]" : "[pause]";
    if (!paused) { held.forEach(render); held = []; }
  };
  el("clear").onclick = () => { while (log.firstChild !== prompt) log.removeChild(log.firstChild); };

  const source = new EventSource("/events");
  source.onopen = () => setStatus("live");
  source.onerror = () => setStatus(source.readyState === EventSource.CLOSED ? "down" : "reconnecting");
  source.onmessage = (msg) => update(JSON.parse(msg.data));

  // ---- metrics: KPIs + charts --------------------------------------------- //
  const pct = (v) => (v == null ? "—" : Math.round(v * 100) + "%");
  const signed = (v) => (v == null ? "—" : (v >= 0 ? "+" : "") + v.toFixed(1));
  const num1 = (v) => (v == null ? "—" : v.toFixed(1));

  const KPI_DEFS = [
    { key: "episodes", cls: "", label: "episodes", fmt: (v) => (v == null ? "0" : String(v)) },
    { key: "blue_win_rate", cls: "win", label: "blue win rate", fmt: pct },
    { key: "attack_success_rate", cls: "attack", label: "attack success", fmt: pct },
    { key: "defender_availability", cls: "avail", label: "defender avail.", fmt: pct },
    { key: "false_positives", cls: "fp", label: "false positives", fmt: num1, sub: "mean / episode" },
    { key: "blue_reward", cls: "blue", label: "blue reward", fmt: signed, sub: "mean / episode" },
    { key: "red_reward", cls: "red", label: "red reward", fmt: signed, sub: "mean / episode" },
  ];

  function renderKpis(summary) {
    const box = el("kpis");
    box.innerHTML = "";
    if (!summary) return;
    for (const def of KPI_DEFS) {
      const v = summary[def.key];
      if (v == null && def.key !== "episodes") continue;
      const tile = document.createElement("div");
      tile.className = "kpi " + def.cls;
      tile.innerHTML =
        `<div class="k-label">${def.label}</div>` +
        `<div class="k-value">${def.fmt(v)}</div>` +
        (def.sub ? `<div class="k-sub">${def.sub}</div>` : "");
      box.appendChild(tile);
    }
  }

  const CSS = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function chartPanel(title, note, legendSeries) {
    const wrap = document.createElement("div");
    wrap.className = "chart";
    const legend = legendSeries
      ? `<span class="legend">${legendSeries.map((s) =>
          `<span class="chip"><span class="swatch" style="background:${s.color}"></span>${s.label}</span>`).join("")}</span>`
      : "";
    wrap.innerHTML =
      `<div class="chart-head"><span class="chart-title">${title}</span>` +
      `<span class="chart-note">${note}</span></div>` +
      `<div class="svg-host"></div>${legend}`;
    return wrap;
  }

  function drawCharts(payload) {
    const host = el("charts");
    host.innerHTML = "";
    if (!payload || !payload.count) {
      host.innerHTML = `<div class="empty">No evaluated episodes yet. Metric charts appear once the run writes EpisodeMetrics rows.</div>`;
      return;
    }
    const raw = payload.raw, roll = payload.rolling, cum = payload.cumulative, win = payload.window;
    const C = {
      blue: CSS("--s-blue"), red: CSS("--s-red"), winc: CSS("--s-win"),
      attack: CSS("--s-attack"), avail: CSS("--s-avail"), fp: CSS("--s-fp"),
    };

    function add(def) {
      if (!def.hasData) return;
      const panel = chartPanel(def.title, def.note, def.legend);
      host.appendChild(panel);
      window.NemesisCharts.renderLineChart(panel.querySelector(".svg-host"), def.cfg);
    }

    // 1) Reward per side — the honest signal while both policies move.
    const rewardLegend = [
      { label: "blue", color: C.blue }, { label: "red", color: C.red },
    ];
    add({
      title: "reward per side",
      note: `rolling mean (window ${win})`,
      legend: rewardLegend,
      hasData: (raw.blue_reward.length || raw.red_reward.length) > 0,
      cfg: {
        ariaLabel: "mean reward per episode for blue and red",
        fmt: signed, yDomain: null,
        series: [
          { key: "blue_reward", label: "blue", color: C.blue, raw: raw.blue_reward, roll: roll.blue_reward },
          { key: "red_reward", label: "red", color: C.red, raw: raw.red_reward, roll: roll.red_reward },
        ],
      },
    });

    // 2) Blue win rate — per-episode wins (dots) vs cumulative rate (line).
    add({
      title: "blue win rate",
      note: "dots = win/loss · line = cumulative",
      hasData: raw.blue_win.length > 0,
      cfg: {
        ariaLabel: "blue win rate over episodes", fmt: pct, yDomain: [0, 1],
        series: [
          { key: "blue_win", label: "cumulative win rate", color: C.winc,
            raw: raw.blue_win, roll: cum.blue_win_rate, dots: true },
        ],
      },
    });

    // 3) Attack success rate.
    add({
      title: "attack success rate",
      note: `rolling mean (window ${win})`,
      hasData: raw.attack_success_rate.length > 0,
      cfg: {
        ariaLabel: "fraction of red objectives compromised", fmt: pct, yDomain: [0, 1],
        series: [{ key: "attack_success_rate", label: "attack success", color: C.attack,
          raw: raw.attack_success_rate, roll: roll.attack_success_rate }],
      },
    });

    // 4) Defender availability.
    add({
      title: "defender availability",
      note: `rolling mean (window ${win})`,
      hasData: raw.defender_availability.length > 0,
      cfg: {
        ariaLabel: "fraction of steps green services stayed up", fmt: pct, yDomain: [0, 1],
        series: [{ key: "defender_availability", label: "availability", color: C.avail,
          raw: raw.defender_availability, roll: roll.defender_availability }],
      },
    });

    // 5) False positives.
    add({
      title: "false positives",
      note: `rolling mean (window ${win})`,
      hasData: raw.false_positives.length > 0,
      cfg: {
        ariaLabel: "blue actions that disrupted a clean host", fmt: num1, yDomain: null, yMin0: true,
        series: [{ key: "false_positives", label: "false positives", color: C.fp,
          raw: raw.false_positives, roll: roll.false_positives }],
      },
    });
  }

  async function pollMetrics() {
    try {
      const res = await fetch("/metrics", { cache: "no-store" });
      if (!res.ok) throw new Error("metrics " + res.status);
      const payload = await res.json();
      el("metrics-sub").textContent = payload.count
        ? `${payload.count} episodes · window ${payload.window}`
        : "no metrics yet";
      renderKpis(payload.summary);
      drawCharts(payload);
    } catch (err) {
      el("metrics-sub").textContent = "metrics unavailable";
    }
  }

  pollMetrics();
  setInterval(pollMetrics, POLL_MS);
})();
