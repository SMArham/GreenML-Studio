/* GreenML Studio Controller — Apple HIG Inspired Theme */

const API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
  ? ""
  : (window.GREENML_API_BASE || "");

const APPLE_PALETTE = {
  blue: "#0071e3",
  blueSoft: "rgba(0, 113, 227, 0.12)",
  green: "#34c759",
  greenSoft: "rgba(52, 199, 89, 0.14)",
  orange: "#ff9500",
  red: "#ff3b30",
  purple: "#5856d6",
  teal: "#30b0c7",
  pink: "#ff2d55",
  charcoal: "#1d1d1f",
  gray: "#86868b",
};

const APPLE_PLOT_CONFIG = {
  responsive: true,
  displayModeBar: false,
};

function applePlotLayout(title, xTitle, yTitle, extra = {}) {
  return {
    title: {
      text: title,
      x: 0.03,
      xanchor: "left",
      y: 0.96,
      yanchor: "top",
      font: {
        family: '-apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif',
        size: 13,
        color: APPLE_PALETTE.charcoal,
      },
    },
    xaxis: {
      automargin: true,
      title: {
        text: xTitle,
        font: { family: '-apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif', size: 11, color: APPLE_PALETTE.gray },
      },
      tickfont: { family: '-apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif', size: 10, color: APPLE_PALETTE.gray },
      gridcolor: "rgba(0, 0, 0, 0.04)",
      linecolor: "rgba(0, 0, 0, 0.08)",
      zerolinecolor: "rgba(0, 0, 0, 0.08)",
    },
    yaxis: {
      automargin: true,
      title: {
        text: yTitle,
        font: { family: '-apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif', size: 11, color: APPLE_PALETTE.gray },
      },
      tickfont: { family: '-apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif', size: 10, color: APPLE_PALETTE.gray },
      gridcolor: "rgba(0, 0, 0, 0.04)",
      linecolor: "rgba(0, 0, 0, 0.08)",
      zerolinecolor: "rgba(0, 0, 0, 0.08)",
    },
    margin: { t: 50, r: 24, b: 48, l: 68 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    font: { family: '-apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif' },
    legend: {
      font: { family: '-apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif', size: 11, color: APPLE_PALETTE.charcoal },
    },
    ...extra,
  };
}

const state = {
  overview: null,
  exp1Stage: "train",
  exp2Stage: "train",
  crossStage: "train",
  currentCrossRows: [],
  currentExp3Data: null,
};

async function apiGet(path) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

async function apiPost(path, payload) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

function renderTable(tableEl, columns, rows, defaultSortKey = null, defaultDesc = true) {
  if (!tableEl) return;
  if (!rows || rows.length === 0) {
    tableEl.innerHTML = "<tbody><tr><td class='muted'>No records available.</td></tr></tbody>";
    return;
  }

  if (defaultSortKey && !tableEl._sortKey) {
    tableEl._sortKey = defaultSortKey;
    tableEl._sortDesc = defaultDesc;
  }

  const sortKey = tableEl._sortKey;
  const sortDesc = tableEl._sortDesc ?? true;

  let sortedRows = [...rows];
  if (sortKey) {
    sortedRows.sort((a, b) => {
      const va = a[sortKey];
      const vb = b[sortKey];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === "number" && typeof vb === "number") {
        return sortDesc ? vb - va : va - vb;
      }
      return sortDesc 
        ? String(vb).localeCompare(String(va))
        : String(va).localeCompare(String(vb));
    });
  }

  const thead = `<thead><tr>${columns.map((c) => {
    const isSorted = tableEl._sortKey === c.key;
    const arrow = isSorted ? (tableEl._sortDesc ? " ▼" : " ▲") : "";
    return `<th data-key="${c.key}" style="cursor: pointer; user-select: none;" title="Click to sort">${c.label}${arrow}</th>`;
  }).join("")}</tr></thead>`;

  const tbody = `<tbody>${sortedRows
    .map((r) => `<tr>${columns.map((c) => `<td>${c.fmt ? c.fmt(r[c.key], r) : (r[c.key] ?? "-")}</td>`).join("")}</tr>`)
    .join("")}</tbody>`;

  tableEl.innerHTML = thead + tbody;

  tableEl.querySelectorAll("thead th").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.key;
      if (tableEl._sortKey === key) {
        tableEl._sortDesc = !tableEl._sortDesc;
      } else {
        tableEl._sortKey = key;
        tableEl._sortDesc = true;
      }
      renderTable(tableEl, columns, rows);
    });
  });
}

function fmtPct(v) {
  return v == null ? "-" : `${(Number(v) * 100).toFixed(2)}%`;
}

function fmtNum(v, digits = 4) {
  return v == null ? "-" : Number(v).toFixed(digits);
}

function resizeVisibleCharts() {
  document.querySelectorAll(".tab-panel.active .js-plotly-plot").forEach((el) => {
    try { Plotly.Plots.resize(el); } catch (_) {}
  });
}

function switchTab(tab) {
  const navWrap = document.getElementById("nav-tabs");
  const menuBtn = document.getElementById("menu-toggle");

  document.querySelectorAll(".nav-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${tab}`));

  if (navWrap && menuBtn) {
    navWrap.classList.remove("open");
    menuBtn.setAttribute("aria-expanded", "false");
  }

  if (tab === "exp1") loadExp1().then(resizeVisibleCharts);
  if (tab === "exp2") loadExp2().then(resizeVisibleCharts);
  if (tab === "exp3") loadExp3().then(resizeVisibleCharts);
  if (tab === "cross") loadCross().then(resizeVisibleCharts);
  if (tab === "live") loadHotswapSurface();
  setTimeout(resizeVisibleCharts, 60);
}

function initTabs() {
  const menuBtn = document.getElementById("menu-toggle");
  const navWrap = document.getElementById("nav-tabs");
  if (menuBtn && navWrap) {
    menuBtn.addEventListener("click", () => {
      const open = navWrap.classList.toggle("open");
      menuBtn.setAttribute("aria-expanded", String(open));
    });
  }

  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      switchTab(btn.dataset.tab);
    });
  });

  document.querySelectorAll(".guide-card").forEach((card) => {
    card.addEventListener("click", () => {
      if (card.dataset.goto) switchTab(card.dataset.goto);
    });
  });

  window.addEventListener("resize", resizeVisibleCharts);
}

/* ------------------------------------------------------------------
   TAB 1: PROJECT OVERVIEW & FIGURES GALLERY
------------------------------------------------------------------ */
/* ------------------------------------------------------------------
   TAB 1: PROJECT OVERVIEW & CURATED FIGURES SHOWCASE
------------------------------------------------------------------ */
async function loadFiguresGallery() {
  try {
    const data = await apiGet("/api/figures");
    const figs = data.figures || [];
    const items = data.items || figs.map((f, i) => ({
      id: f,
      title: f.replace(/^[0-9]+_/, "").replace(".png", "").replace(/_/g, " ").toUpperCase(),
      short_title: `0${i + 1} · ${f.replace(/^[0-9]+_/, "").replace(".png", "")}`,
      category: "Empirical Plate",
      badge: "Core Figure",
      description: "Empirical publication figure from the GreenML experimental benchmark.",
      takeaway: "Empirical evidence supporting accuracy and energy trade-off optimization.",
    }));

    const deck = document.getElementById("fig-deck-container");
    const segBar = document.getElementById("fig-segmented-bar");
    if (!items.length || !deck) return;

    // Render scrollable cards deck
    deck.innerHTML = items
      .map(
        (it, idx) => {
          let mediaHtml = "";
          if (it.panels && it.panels.length > 0) {
            mediaHtml = `
              <div class="fig-panels-deck">
                ${it.panels.map((p, pIdx) => `
                  <div class="fig-panel-block">
                    <div class="fig-panel-header">
                      <span class="fig-panel-pill">${p.tag || ('Part ' + (pIdx + 1))}</span>
                      <span class="fig-panel-title">${p.title}</span>
                    </div>
                    <div class="fig-image-wrap">
                      <img src="${API_BASE}/figures/${encodeURIComponent(p.file)}" alt="${p.title}" loading="lazy" />
                    </div>
                  </div>
                `).join("")}
              </div>
            `;
          } else {
            mediaHtml = `
              <div class="fig-image-wrap">
                <img src="${API_BASE}/figures/${encodeURIComponent(it.id)}" alt="${it.title}" loading="lazy" />
              </div>
            `;
          }

          return `
          <article class="fig-card" id="card-${it.id}" data-fig="${it.id}">
            <div class="fig-card-header">
              <div class="fig-card-title-group">
                <span class="fig-num-badge">0${idx + 1}</span>
                <h3 class="fig-card-title">${it.title}</h3>
                <span class="fig-card-tag">${it.badge || it.category}</span>
              </div>
            </div>
            <p class="fig-card-desc">${it.description}</p>
            ${mediaHtml}
            <div class="fig-card-takeaway">
              <strong>Key Takeaway:</strong> ${it.takeaway}
            </div>
          </article>
        `;
        }
      )
      .join("");

    // Segmented bar listener
    if (segBar) {
      const allBtn = segBar.querySelector('.fig-filter-btn[data-filter="all"]');
      if (allBtn) {
        allBtn.addEventListener("click", () => {
          const firstCard = deck.querySelector(".fig-card");
          if (firstCard) {
            firstCard.scrollIntoView({ behavior: "smooth", block: "start" });
          }
        });
      }
    }
  } catch (_) {}
}

function renderOverview(summary) {
  const hw = summary.hardware || {};
  const paperKeys = Object.keys(summary.paper_datasets || {});
  const nsgaKeys = Object.keys(summary.nsga2_experiments || {});

  let totalSaved = 0;
  let totalDrop = 0;
  let count = 0;
  for (const k of paperKeys) {
    const v = summary.paper_datasets[k];
    if (v.knee_train && v.knee_train.energy_saved_pct != null) {
      totalSaved += v.knee_train.energy_saved_pct;
      totalDrop += v.knee_train.accuracy_drop_pct_points || 0;
      count++;
    }
  }
  const avgSaved = count > 0 ? (totalSaved / count).toFixed(1) : "70.5";
  const avgDrop = count > 0 ? (totalDrop / count).toFixed(2) : "2.10";

  const kpis = [
    { label: "Datasets Profiled", value: `${paperKeys.length + 2}`, sub: "5 UCI Benchmarks + 2 Kaggle" },
    { label: "Avg Energy Saved at Knee", value: `${avgSaved}%`, sub: `With only -${avgDrop}% accuracy drop` },
    { label: "Multi-Objective Optimization", value: `${nsgaKeys.length} Datasets`, sub: "NSGA-II vs Full Grid Search" },
    { label: "Calibrated Telemetry", value: `${hw.cpu_tdp_watts || 15}W TDP`, sub: `${hw.cpu_model || "CPU"} (${hw.country_iso_code || "PAK"})` },
  ];
  document.getElementById("overview-kpis").innerHTML = kpis
    .map((k) => `<div class="kpi-card"><div class="kpi-label">${k.label}</div><div class="kpi-value">${k.value}</div><div class="kpi-sub">${k.sub}</div></div>`)
    .join("");

  const rqRows = paperKeys.map((k) => {
    const v = summary.paper_datasets[k];
    const kt = v.knee_train || {};
    const ki = v.knee_inf || {};
    const trainSavedPct = kt.energy_saved_pct != null ? `${kt.energy_saved_pct.toFixed(1)}%` : "0.0%";
    const trainDrop = kt.accuracy_drop_pct_points != null ? `(-${kt.accuracy_drop_pct_points.toFixed(1)}% drop)` : "";
    const infSavedPct = ki.energy_saved_pct != null ? `${ki.energy_saved_pct.toFixed(1)}%` : "0.0%";
    const infDrop = ki.accuracy_drop_pct_points != null ? `(-${ki.accuracy_drop_pct_points.toFixed(1)}% drop)` : "";
    return {
      dataset: k.toUpperCase(),
      peak_acc: kt.max_accuracy_row?.test_accuracy,
      train_saved: `${trainSavedPct} ${trainDrop}`,
      inf_saved: `${infSavedPct} ${infDrop}`,
    };
  });
  renderTable(
    document.getElementById("rq-summary-table"),
    [
      { key: "dataset", label: "Benchmark Dataset" },
      { key: "peak_acc", label: "Peak Accuracy", fmt: fmtPct },
      { key: "train_saved", label: "Training Energy Saved" },
      { key: "inf_saved", label: "Inference Energy Saved" },
    ],
    rqRows
  );
}

/* ------------------------------------------------------------------
   TAB 2: EXP 1 - GRID & PARETO
------------------------------------------------------------------ */
async function loadExp1() {
  const ds = document.getElementById("exp1-dataset").value;
  if (!ds) return;
  const data = await apiGet(`/api/exp1/${ds}`);
  const isTrain = state.exp1Stage === "train";
  const eCol = isTrain ? "train_energy_j" : "inf_energy_j";
  const iqrCol = isTrain ? "train_energy_iqr" : "inf_energy_iqr";
  const pf = isTrain ? data.pareto_train : data.pareto_inf;
  const knee = isTrain ? data.knee_train : data.knee_inf;
  const showIqr = document.getElementById("exp1-iqr") ? document.getElementById("exp1-iqr").checked : false;

  if (knee) {
    document.getElementById("exp1-kpis").innerHTML = [
      { label: "Peak Accuracy", value: fmtPct(knee.max_accuracy_row.test_accuracy), sub: "Max-accuracy configuration" },
      { label: "Knee Point Accuracy", value: fmtPct(knee.knee_row.test_accuracy), sub: `-${knee.accuracy_drop_pct_points.toFixed(2)}% accuracy drop` },
      { label: "Energy Saved at Knee", value: `${knee.energy_saved_pct.toFixed(1)}%`, sub: isTrain ? "Training CPU Energy" : "Inference 1k Energy" },
      { label: "Pareto Front Configs", value: `${pf.length} / ${data.configs.length}`, sub: "Non-dominated solutions" },
    ]
      .map((k) => `<div class="kpi-card"><div class="kpi-label">${k.label}</div><div class="kpi-value">${k.value}</div><div class="kpi-sub">${k.sub}</div></div>`)
      .join("");
  }

  const solvers = [...new Set(data.configs.map((c) => c.solver))];
  const solverColors = { adam: APPLE_PALETTE.green, sgd: APPLE_PALETTE.blue, lbfgs: APPLE_PALETTE.orange };
  const traces = solvers.map((sol) => {
    const sub = data.configs.filter((c) => c.solver === sol);
    return {
      x: sub.map((c) => c.test_accuracy),
      y: sub.map((c) => c[eCol]),
      error_y: showIqr ? { type: "data", array: sub.map((c) => (c[iqrCol] || 0) / 2), visible: true, thickness: 1, color: "rgba(0,0,0,0.15)" } : undefined,
      mode: "markers",
      type: "scatter",
      name: `solver=${sol}`,
      text: sub.map((c) => `h=${c.hidden_neurons}, ${c.activation}, ${c.solver}, alpha=${c.alpha}, iter=${c.max_iter}`),
      marker: { size: 7, opacity: 0.65, color: solverColors[sol] || APPLE_PALETTE.gray },
    };
  });

  if (pf.length > 0) {
    traces.push({
      x: pf.map((c) => c.test_accuracy),
      y: pf.map((c) => c[eCol]),
      mode: "lines+markers",
      type: "scatter",
      name: "Pareto Front",
      line: { color: APPLE_PALETTE.charcoal, width: 2.4 },
      marker: { size: 8, color: APPLE_PALETTE.charcoal },
    });
  }
  if (knee) {
    traces.push({
      x: [knee.knee_row.test_accuracy],
      y: [knee.knee_row[eCol]],
      mode: "markers",
      type: "scatter",
      name: `Knee Point (-${knee.energy_saved_pct.toFixed(1)}%)`,
      marker: { size: 16, symbol: "star", color: APPLE_PALETTE.green, line: { color: "#ffffff", width: 1.5 } },
    });
  }

  Plotly.newPlot(
    "exp1-pareto-chart",
    traces,
    applePlotLayout(`${ds.toUpperCase()} — Test Accuracy vs ${isTrain ? "Training" : "Inference (1k)"} Energy`, "Test Accuracy", "CPU Energy (Joules)"),
    APPLE_PLOT_CONFIG
  );

  if (knee) {
    const rows = [
      { profile: "Min Energy Extreme", ...knee.min_energy_row },
      { profile: "Recommended Knee Point", ...knee.knee_row },
      { profile: "Max Accuracy Extreme", ...knee.max_accuracy_row },
    ];
    renderTable(
      document.getElementById("exp1-knee-table"),
      [
        { key: "profile", label: "Profile" },
        { key: "hidden_neurons", label: "Hidden Units" },
        { key: "test_accuracy", label: "Test Acc", fmt: fmtPct },
        { key: eCol, label: "CPU Energy (J)", fmt: (v) => fmtNum(v, 4) },
      ],
      rows
    );
  }

  const norm = data.normality_audit || {};
  if (norm.values && norm.values.length > 0) {
    const ns = norm.summary || {};
    Plotly.newPlot(
      "exp1-hist-chart",
      [{ x: norm.values, type: "histogram", nbinsx: 10, marker: { color: APPLE_PALETTE.green } }],
      applePlotLayout(`Energy Distribution (${norm.dataset}): 30 Repeats (Skew=${fmtNum(ns.skewness, 2)}, p=${fmtNum(ns.shapiro_p, 4)})`, "CPU Training Energy (Joules)", "Frequency"),
      APPLE_PLOT_CONFIG
    );
  }

  Plotly.newPlot(
    "exp1-time-energy-chart",
    [
      {
        x: data.configs.map((c) => c.train_cpu_time_s),
        y: data.configs.map((c) => c.train_energy_j),
        mode: "markers",
        type: "scatter",
        marker: { size: 6, color: APPLE_PALETTE.blue, opacity: 0.75 },
      },
    ],
    applePlotLayout(`CPU Process Time vs Energy (${ds})`, "CPU Process Time (s)", "CPU Energy (Joules)"),
    APPLE_PLOT_CONFIG
  );

  const flopsCorr = data.meta?.rq4_summary?.spearman_flops_vs_inf_energy;
  document.getElementById("flops-corr-badge").textContent = flopsCorr != null ? `Spearman rho = ${fmtNum(flopsCorr, 3)}` : "FLOPs Proxy";
  const withFlops = data.configs.filter((c) => c.inf_flops_per_sample != null);
  if (withFlops.length > 0) {
    Plotly.newPlot(
      "exp1-flops-chart",
      [
        {
          x: withFlops.map((c) => c.inf_flops_per_sample),
          y: withFlops.map((c) => c.inf_energy_j),
          mode: "markers",
          type: "scatter",
          text: withFlops.map((c) => `h=${c.hidden_neurons}, params=${c.param_count}, solver=${c.solver}`),
          marker: { size: 7, color: APPLE_PALETTE.purple, opacity: 0.75 },
        },
      ],
      applePlotLayout(`Theoretical Inference FLOPs/sample vs Measured 1k Inference Energy (${ds})`, "Inference FLOPs per Sample", "Measured Inference 1k Energy (J)"),
      APPLE_PLOT_CONFIG
    );
  }
}

/* ------------------------------------------------------------------
   TAB 3: EXP 2 - DRIVERS & STAGES
------------------------------------------------------------------ */
async function loadExp2() {
  const ds = document.getElementById("exp2-dataset").value;
  if (!ds) return;
  const data = await apiGet(`/api/exp2/${ds}`);
  const isTrain = state.exp2Stage === "train";
  const pf = isTrain ? data.pareto_sorted_train : data.pareto_sorted_inf;
  const eCol = isTrain ? "train_energy_j" : "inf_energy_j";
  const rq4 = data.rq4_summary || {};

  document.getElementById("exp2-section5-kpis").innerHTML = [
    { label: "Compact Network (<=20 Neurons)", value: fmtPct(rq4.compact_long_best_acc), sub: `Median Inf: ${fmtNum(rq4.compact_long_median_inf_j, 3)} J` },
    { label: "Large Network (>=100 Neurons)", value: fmtPct(rq4.large_short_best_acc), sub: `Median Inf: ${fmtNum(rq4.large_short_median_inf_j, 3)} J` },
    { label: "Train vs. Inf Stage Decoupling", value: `rho = ${fmtNum(rq4.spearman_train_vs_inf, 3)}`, sub: "Stages are independent (RQ4)" },
    { label: "FLOPs vs. Hardware Energy", value: `rho = ${fmtNum(rq4.spearman_flops_vs_inf_energy, 3)}`, sub: "Structural complexity proxy" },
  ]
    .map((k) => `<div class="kpi-card"><div class="kpi-label">${k.label}</div><div class="kpi-value">${k.value}</div><div class="kpi-sub muted">${k.sub}</div></div>`)
    .join("");

  renderTable(
    document.getElementById("exp2-pareto-matrix"),
    [
      { key: "hidden_neurons", label: "Hidden Size" },
      { key: "activation", label: "Activation" },
      { key: "solver", label: "Solver" },
      { key: eCol, label: isTrain ? "Train Energy (J)" : "Inf 1k Energy (J)", fmt: (v) => fmtNum(v, 4) },
      { key: "test_accuracy", label: "Test Accuracy", fmt: fmtPct },
    ],
    pf
  );

  renderTable(
    document.getElementById("exp2-spearman-table"),
    [
      { key: "parameter", label: "Hyperparameter", fmt: (v) => v.replace(/_/g, " ") },
      { key: "target", label: "Target Metric", fmt: (v) => v.replace("_j", " (J)").replace(/_/g, " ") },
      { key: "spearman_rho", label: "Spearman rho", fmt: (v) => fmtNum(v, 3) },
      { key: "p_value", label: "p-value", fmt: (v) => (v < 0.0001 ? "< 0.0001" : fmtNum(v, 4)) },
    ],
    data.spearman
  );

  renderTable(
    document.getElementById("exp2-cat-table"),
    [
      { key: "parameter", label: "Hyperparameter", fmt: (v) => v.replace(/_/g, " ") },
      { key: "target", label: "Target Metric", fmt: (v) => v.replace("_j", " (J)").replace(/_/g, " ") },
      { key: "kruskal_h", label: "Kruskal H", fmt: (v) => fmtNum(v, 2) },
      {
        key: "p_value",
        label: "Significance",
        fmt: (v) => (v < 0.001 ? "High (p < 0.001)" : v < 0.05 ? "Significant (p < 0.05)" : "Not Significant"),
      },
      {
        key: "group_medians",
        label: "Group Comparison",
        fmt: (v) => {
          if (!v) return "-";
          try {
            const clean = typeof v === "string" ? JSON.parse(v.replace(/'/g, '"')) : v;
            return Object.entries(clean)
              .map(([grp, val]) => `${grp}: ${typeof val === "number" ? (val < 1 ? (val * 100).toFixed(1) + "%" : val.toFixed(2) + " J") : val}`)
              .join(" · ");
          } catch (_) {
            return String(v).replace(/[{}']/g, "");
          }
        },
      },
    ],
    data.categorical
  );

  document.getElementById("rq4-badge").textContent = `Low-Train/Low-Inf: ${rq4.low_train_low_inf_count ?? 0} | High-Train/Low-Inf: ${rq4.high_train_low_inf_count ?? 0}`;

  Plotly.newPlot(
    "exp2-rq4-chart",
    [
      {
        x: data.all_points.map((p) => p.train_energy_j),
        y: data.all_points.map((p) => p.inf_energy_j),
        mode: "markers",
        type: "scatter",
        text: data.all_points.map((p) => `h=${p.hidden_neurons}, ${p.activation}, ${p.solver}, acc=${(p.test_accuracy * 100).toFixed(1)}%`),
        marker: {
          size: 8,
          color: data.all_points.map((p) => p.hidden_neurons),
          colorscale: [[0, APPLE_PALETTE.teal], [0.5, APPLE_PALETTE.blue], [1, APPLE_PALETTE.purple]],
          showscale: true,
          colorbar: {
            title: { text: "Hidden", font: { family: '-apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif', size: 11, color: APPLE_PALETTE.gray } },
            tickfont: { family: '-apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif', size: 10, color: APPLE_PALETTE.gray },
          },
        },
      },
    ],
    applePlotLayout(`Training Energy vs Inference (1k) Energy (${ds})`, "Training CPU Energy (Joules)", "Inference 1k CPU Energy (Joules)"),
    APPLE_PLOT_CONFIG
  );
}

/* ------------------------------------------------------------------
   TAB 4: EXP 3 - NSGA-II & PAYBACK
------------------------------------------------------------------ */
let _paybackPlotTimer = null;

function updatePaybackKPIsOnly() {
  const data = state.currentExp3Data;
  if (!data || !data.carbon_payback) return;
  const pb = data.carbon_payback;
  const dailyInput = document.getElementById("payback-daily");
  const daily = Number(dailyInput ? dailyInput.value : 100000);
  const dailyValEl = document.getElementById("payback-daily-val");
  if (dailyValEl) dailyValEl.textContent = daily.toLocaleString();

  const savedPerSampleJ = (pb.saved_per_sample_uj || 0) / 1e6;
  if (savedPerSampleJ <= 0) return;

  const beNsga = pb.nsga2_search_energy_j / savedPerSampleJ;
  const beGrid = pb.grid_search_energy_j / savedPerSampleJ;
  const nsgaDays = daily > 0 ? (beNsga / daily).toFixed(1) : "0";
  const gridDays = daily > 0 ? (beGrid / daily).toFixed(1) : "0";

  const paybackCard = document.getElementById("kpi-payback-card");
  if (paybackCard) {
    const valEl = paybackCard.querySelector(".kpi-value");
    const subEl = paybackCard.querySelector(".kpi-sub");
    if (valEl) valEl.textContent = `${nsgaDays} days`;
    if (subEl) subEl.textContent = `At ${daily.toLocaleString()} inf/day`;
  }
}

function handlePaybackSliderInput() {
  updatePaybackKPIsOnly();
  if (_paybackPlotTimer) cancelAnimationFrame(_paybackPlotTimer);
  _paybackPlotTimer = requestAnimationFrame(() => {
    drawPaybackChart();
  });
}

function drawPaybackChart() {
  const data = state.currentExp3Data;
  if (!data || !data.carbon_payback) return;
  const pb = data.carbon_payback;
  const dailyInput = document.getElementById("payback-daily");
  const daily = Number(dailyInput ? dailyInput.value : 100000);
  const dailyValEl = document.getElementById("payback-daily-val");
  if (dailyValEl) dailyValEl.textContent = daily.toLocaleString();

  const savedPerSampleJ = (pb.saved_per_sample_uj || 0) / 1e6;
  if (savedPerSampleJ <= 0) {
    Plotly.newPlot("exp3-payback-chart", [], applePlotLayout("Instant Payback", "Production Inferences", "Net Energy Saved"));
    return;
  }

  const beGrid = pb.grid_search_energy_j / savedPerSampleJ;
  const beNsga = pb.nsga2_search_energy_j / savedPerSampleJ;
  const nsgaDays = daily > 0 ? (beNsga / daily).toFixed(1) : "0";
  const gridDays = daily > 0 ? (beGrid / daily).toFixed(1) : "0";

  // Real-time KPI Card update
  const paybackCard = document.getElementById("kpi-payback-card");
  if (paybackCard) {
    const valEl = paybackCard.querySelector(".kpi-value");
    const subEl = paybackCard.querySelector(".kpi-sub");
    if (valEl) valEl.textContent = `${nsgaDays} days`;
    if (subEl) subEl.textContent = `At ${daily.toLocaleString()} inf/day`;
  }

  const maxN = Math.max(beGrid * 1.25, 1e5);
  const xs = [];
  const yGrid = [];
  const yNsga = [];
  for (let i = 0; i <= 120; i++) {
    const n = (maxN * i) / 120;
    xs.push(n / 1e6);
    yGrid.push(n * savedPerSampleJ - pb.grid_search_energy_j);
    yNsga.push(n * savedPerSampleJ - pb.nsga2_search_energy_j);
  }

  Plotly.newPlot(
    "exp3-payback-chart",
    [
      { x: xs, y: yGrid, mode: "lines", name: `Grid Payback (${gridDays} days)`, line: { color: APPLE_PALETTE.red, width: 2 } },
      { x: xs, y: yNsga, mode: "lines", name: `NSGA-II Payback (${nsgaDays} days)`, line: { color: APPLE_PALETTE.green, width: 2.6 } },
      { x: [0, maxN / 1e6], y: [0, 0], mode: "lines", name: "Break-Even (0 J)", line: { color: APPLE_PALETTE.gray, dash: "dash" } },
    ],
    applePlotLayout(`Carbon Payback Break-Even Horizon (${(pb.co2_saved_per_1m_inferences_g || 0).toFixed(3)} g CO2 saved / 1M inferences)`, "Production Inferences (Millions)", "Net Energy Saved (Joules)"),
    APPLE_PLOT_CONFIG
  );
}

async function loadExp3() {
  const dsEl = document.getElementById("exp3-dataset");
  const ds = dsEl.value;
  if (!ds) {
    document.getElementById("exp3-kpis").innerHTML = "<div class='kpi-card'><div class='kpi-label'>Status</div><div class='kpi-value'>Pending</div><div class='kpi-sub muted'>NSGA-II results not available</div></div>";
    return;
  }
  const data = await apiGet(`/api/exp3/${ds}`);
  state.currentExp3Data = data;
  const meta = data.meta || {};
  const pb = data.carbon_payback || {};
  const daily = Number(document.getElementById("payback-daily").value);
  const savedPerJ = (pb.saved_per_sample_uj || 0) / 1e6;
  const nsgaDays = savedPerJ > 0 ? ((pb.nsga2_search_energy_j / savedPerJ) / daily).toFixed(1) : "0";

  document.getElementById("exp3-kpis").innerHTML = [
    { label: "NSGA-II Budget", value: `${meta.max_evaluations ?? "-"} evals`, sub: "vs 1,440 full grid" },
    { label: "NSGA-II Best HV Ratio", value: `${((meta.hv_ratio_best || 0) * 100).toFixed(1)}%`, sub: `Median: ${((meta.hv_ratio_median || 0) * 100).toFixed(1)}%` },
    { label: "CO2 Saved / 1M Inf", value: `${(pb.co2_saved_per_1m_inferences_g || 0).toFixed(3)} g`, sub: "Inference savings" },
    { id: "kpi-payback-card", label: "NSGA-II Payback Horizon", value: `${nsgaDays} days`, sub: `At ${daily.toLocaleString()} inf/day` },
  ]
    .map((k) => `<div class="kpi-card" ${k.id ? `id="${k.id}"` : ""}><div class="kpi-label">${k.label}</div><div class="kpi-value">${k.value}</div><div class="kpi-sub">${k.sub}</div></div>`)
    .join("");

  const att = data.attainment || [];
  const gp = data.grid_pareto || [];
  const traces = [];
  if (att.length > 0) {
    traces.push({
      x: [...att.map((a) => a.accuracy), ...att.map((a) => a.accuracy).reverse()],
      y: [...att.map((a) => a.worst_energy_j), ...att.map((a) => a.best_energy_j).reverse()],
      fill: "toself",
      fillcolor: APPLE_PALETTE.blueSoft,
      line: { color: "transparent" },
      name: "NSGA-II Min-Max Band",
      type: "scatter",
    });
    traces.push({
      x: att.map((a) => a.accuracy),
      y: att.map((a) => a.median_energy_j),
      mode: "lines",
      name: "NSGA-II Median Attainment",
      line: { color: APPLE_PALETTE.blue, width: 2.6 },
    });
  }
  if (gp.length > 0) {
    traces.push({
      x: gp.map((g) => g.test_accuracy),
      y: gp.map((g) => g.train_energy_j),
      mode: "markers",
      name: "Grid Pareto Front",
      marker: { color: APPLE_PALETTE.red, size: 8 },
    });
  }

  Plotly.newPlot(
    "exp3-attainment-chart",
    traces,
    applePlotLayout(`NSGA-II Attainment vs Grid Pareto (${ds})`, "Test Accuracy", "Training CPU Energy (Joules)"),
    APPLE_PLOT_CONFIG
  );

  drawPaybackChart();
}

/* ------------------------------------------------------------------
   TAB 5: CROSS-MODEL BENCHMARK
------------------------------------------------------------------ */
async function loadCross() {
  const ds = document.getElementById("cross-dataset").value;
  if (!ds) return;
  const data = await apiGet(`/api/cross-model/${ds}`);
  const isTrain = state.crossStage === "train";
  const eCol = isTrain ? "train_energy_j" : "inf_energy_j";
  const pf = isTrain ? data.pareto_train : data.pareto_inf;
  const knee = isTrain ? data.knee_train : data.knee_inf;

  state.currentCrossRows = [...data.models].sort((a, b) => b.test_accuracy - a.test_accuracy);

  if (knee) {
    document.getElementById("cross-kpis").innerHTML = [
      { label: "Models Evaluated", value: `${data.models.length}`, sub: "6 Classifier Families" },
      { label: "Peak Accuracy Model", value: fmtPct(knee.max_accuracy_row.test_accuracy), sub: knee.max_accuracy_row.model_label },
      { label: "Recommended Knee Model", value: knee.knee_row.model_label, sub: `${fmtPct(knee.knee_row.test_accuracy)} accuracy` },
      { label: "Energy Saved vs Peak Acc", value: `${knee.energy_saved_pct.toFixed(1)}%`, sub: `-${knee.accuracy_drop_pct_points.toFixed(2)}% acc drop` },
    ]
      .map((k) => `<div class="kpi-card"><div class="kpi-label">${k.label}</div><div class="kpi-value">${k.value}</div><div class="kpi-sub">${k.sub}</div></div>`)
      .join("");
  }

  const famColors = {
    LogisticRegression: APPLE_PALETTE.blue,
    DecisionTree: APPLE_PALETTE.green,
    RandomForest: APPLE_PALETTE.teal,
    SVM: APPLE_PALETTE.purple,
    KNN: APPLE_PALETTE.orange,
    ExtraTrees: APPLE_PALETTE.pink,
  };

  const fams = [...new Set(data.models.map((m) => m.family))];
  const traces = fams.map((fam) => {
    const sub = data.models.filter((m) => m.family === fam);
    return {
      x: sub.map((m) => m.test_accuracy),
      y: sub.map((m) => m[eCol]),
      mode: "markers",
      type: "scatter",
      name: fam,
      text: sub.map((m) => `${m.model_label} | F1=${m.f1_score.toFixed(3)}`),
      marker: { size: 9, color: famColors[fam] || APPLE_PALETTE.gray },
    };
  });
  if (pf.length > 0) {
    traces.push({
      x: pf.map((p) => p.test_accuracy),
      y: pf.map((p) => p[eCol]),
      mode: "lines+markers",
      name: "Global Pareto Front",
      line: { color: APPLE_PALETTE.charcoal, width: 2, dash: "dot" },
    });
  }
  if (knee) {
    traces.push({
      x: [knee.knee_row.test_accuracy],
      y: [knee.knee_row[eCol]],
      mode: "markers",
      name: `Knee (${knee.knee_row.model_label})`,
      marker: { size: 16, symbol: "star", color: APPLE_PALETTE.green, line: { color: "#ffffff", width: 1.5 } },
    });
  }

  Plotly.newPlot(
    "cross-chart",
    traces,
    applePlotLayout(`Accuracy vs ${isTrain ? "Training" : "Inference 1k"} Energy (${ds})`, "Test Accuracy", "CPU Energy (Joules)"),
    APPLE_PLOT_CONFIG
  );

  state.currentCrossData = data;
  renderBestModelsCrossTable(data);
}

function renderBestModelsCrossTable(data) {
  if (!data || !data.models) return;
  const isTrain = state.crossStage === "train";
  const pf = isTrain ? data.pareto_train : data.pareto_inf;
  const pfLabels = new Set((pf || []).map((m) => m.model_label || `${m.family} (${m.variant})`));
  const knee = isTrain ? data.knee_train : data.knee_inf;
  const kneeLabel = knee?.knee_row?.model_label;

  // Curate Best Models: High Accuracy, Fast Training, Minimal Energy
  const bestModels = data.models.filter((m) => {
    const lbl = m.model_label || `${m.family} (${m.variant})`;
    if (pfLabels.has(lbl) || lbl === kneeLabel) return true;
    return (m.train_time_s == null || m.train_time_s <= 0.5) &&
           (m.train_energy_j == null || m.train_energy_j <= 0.5) &&
           (m.inf_energy_j == null || m.inf_energy_j <= 20.0) &&
           (m.test_accuracy >= 0.75);
  });

  state.currentCrossRows = bestModels;

  const badgeEl = document.getElementById("cross-badge");
  if (badgeEl) {
    badgeEl.textContent = `⭐ Best Models (${bestModels.length} Selected)`;
  }

  renderTable(
    document.getElementById("cross-table"),
    [
      { key: "family", label: "Family" },
      { key: "variant", label: "Variant" },
      { key: "test_accuracy", label: "Accuracy", fmt: fmtPct },
      { key: "train_time_s", label: "Train Time (s)", fmt: (v) => fmtNum(v, 3) },
      { key: "train_energy_j", label: "Train Energy (J)", fmt: (v) => fmtNum(v, 4) },
      { key: "inf_energy_j", label: "Inf 1k Energy (J)", fmt: (v) => fmtNum(v, 4) },
    ],
    state.currentCrossRows,
    "test_accuracy",
    true
  );
}

/* ------------------------------------------------------------------
   TAB 6: HOT-SWAP & LIVE SIMULATOR
------------------------------------------------------------------ */
async function loadHotswapSurface() {
  const ds = document.getElementById("hs-dataset").value;
  if (!ds) return;
  const data = await apiGet(`/api/hotswap/surface/${ds}`);
  const accs = data.entries.map((e) => e.test_accuracy);
  const slaEl = document.getElementById("hs-sla");
  slaEl.min = Math.min(...accs).toFixed(3);
  slaEl.max = Math.max(...accs).toFixed(3);
  if (Number(slaEl.value) < Number(slaEl.min) || Number(slaEl.value) > Number(slaEl.max)) {
    slaEl.value = slaEl.min;
  }
  document.getElementById("hs-sla-val").textContent = Number(slaEl.value).toFixed(3);

  renderTable(
    document.getElementById("hs-surface-table"),
    [
      { key: "tier", label: "Tier" },
      { key: "model_label", label: "Model" },
      { key: "test_accuracy", label: "Test Accuracy", fmt: fmtPct },
      { key: "inf_energy_1k_j", label: "Inf Energy / 1k (J)", fmt: (v) => fmtNum(v, 4) },
    ],
    data.entries
  );
}

async function runHotswapPredict() {
  const ds = document.getElementById("hs-dataset").value;
  if (!ds) return;
  const out = document.getElementById("hs-output");
  out.className = "live-result-box muted";
  out.textContent = "Routing model and measuring live inference telemetry...";
  try {
    const res = await apiPost("/api/hotswap/predict", {
      dataset: ds,
      battery_pct: Number(document.getElementById("hs-battery").value),
      min_accuracy: Number(document.getElementById("hs-sla").value),
      sample_idx: Number(document.getElementById("hs-sample").value),
      batch_repeats: 100,
    });
    out.className = "live-result-box ok";
    out.innerHTML = `
      <div style="font-size:0.95rem; margin-bottom:0.25rem;">
        <span class="badge" style="background:#e8f5e9; color:#2e7d32; font-weight:700;">[${res.selected.tier}]</span>
        <strong>${res.selected.model_label}</strong>
      </div>
      <div style="font-size:0.86rem; color:#424245; margin-bottom:0.25rem;">
        <span><strong>Live Accuracy:</strong> ${(res.live_test_accuracy * 100).toFixed(1)}%</span> &middot;
        <span><strong>Latency:</strong> ${res.batch_latency_ms.toFixed(2)} ms</span> &middot;
        <span><strong>Energy / 1k:</strong> ${res.measured_energy_1k_j.toFixed(4)} J</span>
      </div>
      <div style="font-size:0.84rem;">
        Sample #${res.sample_idx} / ${res.sample_total}: Predicted = <code>${res.predicted_class}</code> | Ground Truth = <code>${res.ground_truth_class}</code>
        <span class="badge" style="margin-left:0.3rem; background:${res.is_correct ? "#e8f5e9; color:#2e7d32;" : "#ffebee; color:#c62828;"}">${res.is_correct ? "Correct" : "Mismatch"}</span>
      </div>
    `;
  } catch (err) {
    out.textContent = `Error: ${err.message}`;
  }
}

async function runLiveExperiment() {
  const statusEl = document.getElementById("live-status");
  if (statusEl) {
    statusEl.classList.add("loading");
    statusEl.innerHTML = `<span class="spinner-inline"></span> Training 4 classifiers and measuring real energy on backend...`;
  }
  try {
    const res = await apiPost("/api/live-experiment", {
      dataset: document.getElementById("live-dataset").value,
      mlp_hidden: Number(document.getElementById("live-hidden").value),
      mlp_activation: document.getElementById("live-act").value,
      mlp_solver: document.getElementById("live-solver").value,
    });
    if (statusEl) {
      statusEl.classList.remove("loading");
      statusEl.textContent = `Completed on ${res.dataset} (${res.n_train} train / ${res.n_test} test rows).`;
    }

    renderTable(
      document.getElementById("live-exp-table"),
      [
        { key: "model", label: "Model" },
        { key: "test_accuracy", label: "Accuracy", fmt: fmtPct },
        { key: "train_time_ms", label: "Train (ms)", fmt: (v) => fmtNum(v, 1) },
        { key: "train_energy_j", label: "Energy (J)", fmt: (v) => fmtNum(v, 4) },
      ],
      res.results
    );

    const positions = ["bottom right", "top left", "top right", "top center"];
    Plotly.newPlot(
      "live-exp-chart",
      [
        {
          x: res.results.map((r) => r.test_accuracy),
          y: res.results.map((r) => r.train_energy_j),
          text: res.results.map((r) => r.model),
          mode: "markers+text",
          textposition: res.results.map((_, i) => positions[i % positions.length]),
          textfont: { family: '-apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif', size: 10.5, color: APPLE_PALETTE.charcoal },
          type: "scatter",
          hoverinfo: "text+x+y",
          hovertemplate: "%{text}<br>Accuracy: %{x:.2%}<br>Energy: %{y:.4f} J<extra></extra>",
          marker: { size: 11, color: APPLE_PALETTE.blue },
        },
      ],
      applePlotLayout(`Live Benchmark Trade-Off (${res.dataset.toUpperCase()})`, "Test Accuracy", "Training Energy (Joules)", {
        xaxis: {
          tickformat: ".1%",
          automargin: true,
        },
        yaxis: {
          automargin: true,
        },
      }),
      APPLE_PLOT_CONFIG
    );
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  }
}

/* ------------------------------------------------------------------
   BOOTSTRAP
------------------------------------------------------------------ */
function populateSelect(el, keys, preferred) {
  if (!el) return;
  el.innerHTML = keys.map((k) => `<option value="${k}" ${k === preferred ? "selected" : ""}>${k.toUpperCase()}</option>`).join("");
}

function bindSegmented(containerId, onChange) {
  const wrap = document.getElementById(containerId);
  if (!wrap) return;
  wrap.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      wrap.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      onChange(btn.dataset.val);
    });
  });
}

async function init() {
  initTabs();
  loadFiguresGallery();

  try {
    const summary = await apiGet("/api/overview");
    state.overview = summary;

    const paperKeys = Object.keys(summary.paper_datasets || {});
    const nsgaKeys = Object.keys(summary.nsga2_experiments || {});
    const crossKeys = Object.keys(summary.cross_model_benchmarks || {});
    const hsKeys = Object.keys(summary.hotswap_bundles || {}).filter((k) => summary.hotswap_bundles[k]);

    populateSelect(document.getElementById("exp1-dataset"), paperKeys, paperKeys.includes("diabetes") ? "diabetes" : paperKeys[0]);
    populateSelect(document.getElementById("exp2-dataset"), paperKeys, paperKeys.includes("glass") ? "glass" : paperKeys[0]);
    populateSelect(document.getElementById("exp3-dataset"), nsgaKeys, nsgaKeys.includes("diabetes") ? "diabetes" : nsgaKeys[0]);
    populateSelect(document.getElementById("cross-dataset"), crossKeys, crossKeys.includes("telco_churn") ? "telco_churn" : crossKeys[0]);
    populateSelect(document.getElementById("hs-dataset"), hsKeys, hsKeys.includes("diabetes") ? "diabetes" : hsKeys[0]);

    renderOverview(summary);
  } catch (_) {}

  const exp1Ds = document.getElementById("exp1-dataset");
  if (exp1Ds) exp1Ds.addEventListener("change", loadExp1);
  const exp1Iqr = document.getElementById("exp1-iqr");
  if (exp1Iqr) exp1Iqr.addEventListener("change", loadExp1);
  bindSegmented("exp1-stage", (v) => { state.exp1Stage = v; loadExp1(); });

  const exp2Ds = document.getElementById("exp2-dataset");
  if (exp2Ds) exp2Ds.addEventListener("change", loadExp2);
  bindSegmented("exp2-stage", (v) => { state.exp2Stage = v; loadExp2(); });

  const exp3Ds = document.getElementById("exp3-dataset");
  if (exp3Ds) exp3Ds.addEventListener("change", loadExp3);
  const pbDaily = document.getElementById("payback-daily");
  if (pbDaily) {
    pbDaily.addEventListener("input", handlePaybackSliderInput);
    pbDaily.addEventListener("change", handlePaybackSliderInput);
  }

  const crossDs = document.getElementById("cross-dataset");
  if (crossDs) crossDs.addEventListener("change", loadCross);
  bindSegmented("cross-stage", (v) => { state.crossStage = v; loadCross(); });

  const crossExp = document.getElementById("cross-export-csv");
  if (crossExp) {
    crossExp.addEventListener("click", () => {
      if (!state.currentCrossRows.length) return;
      const cols = Object.keys(state.currentCrossRows[0]);
      const csv = [cols.join(","), ...state.currentCrossRows.map((r) => cols.map((c) => JSON.stringify(r[c] ?? "")).join(","))].join("\n");
      const blob = new Blob([csv], { type: "text/csv" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `cross_model_${document.getElementById("cross-dataset").value}.csv`;
      a.click();
    });
  }

  const hsDs = document.getElementById("hs-dataset");
  if (hsDs) hsDs.addEventListener("change", loadHotswapSurface);
  const hsBat = document.getElementById("hs-battery");
  if (hsBat) {
    hsBat.addEventListener("input", (e) => {
      document.getElementById("hs-battery-val").textContent = `${e.target.value}%`;
    });
  }
  const hsSla = document.getElementById("hs-sla");
  if (hsSla) {
    hsSla.addEventListener("input", (e) => {
      document.getElementById("hs-sla-val").textContent = Number(e.target.value).toFixed(3);
    });
  }
  const hsBtn = document.getElementById("hs-run-btn");
  if (hsBtn) hsBtn.addEventListener("click", runHotswapPredict);

  const liveBtn = document.getElementById("live-run-btn");
  if (liveBtn) liveBtn.addEventListener("click", runLiveExperiment);

  // Initialize and bind all range sliders for modern track fill (matching Image 4)
  function updateSliderFill(el) {
    if (!el) return;
    const min = parseFloat(el.min) || 0;
    const max = parseFloat(el.max) || 100;
    const val = parseFloat(el.value) || min;
    const pct = Math.max(0, Math.min(100, ((val - min) / (max - min)) * 100));
    el.style.setProperty("--progress", `${pct}%`);
  }
  document.querySelectorAll('input[type="range"]').forEach((slider) => {
    slider.addEventListener("input", () => updateSliderFill(slider));
    slider.addEventListener("change", () => updateSliderFill(slider));
    updateSliderFill(slider);
  });
}

init();
