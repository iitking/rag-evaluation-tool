/**
 * RAG Evaluation & Optimization Tool - Frontend Application Logic
 * Pure modern JavaScript, connects directly to FastAPI backend
 */

let appConfig = null;
let currentStrategies = [];
let currentFile = null;
let sampleData = null;
let benchmarkPayload = null;
let elapsedTimer = null;
let elapsedSeconds = 0;

// Initialize on page load
document.addEventListener("DOMContentLoaded", async () => {
  if (window.lucide) {
    lucide.createIcons();
  }
  setupEventHandlers();
  await loadAppConfig();
  await loadSampleDataset();
});

// ----------------------------- Config Loading -----------------------------
async function loadAppConfig() {
  try {
    const res = await fetch("/config");
    if (!res.ok) throw new Error("Failed to fetch configuration");
    appConfig = await res.json();

    // Populate Embedders
    renderEmbedders(appConfig.allowed_embedders || []);

    // Populate Presets & Strategies
    currentStrategies = [...(appConfig.default_strategies || [])];
    renderStrategies();

    // Populate Generator Models
    const genSelect = document.getElementById("select-generator");
    genSelect.innerHTML = "";
    (appConfig.supported_generator_models || [appConfig.generator_model]).forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      if (m === appConfig.generator_model) opt.selected = true;
      genSelect.appendChild(opt);
    });

    // Populate Judge Models
    const jdgSelect = document.getElementById("select-judge");
    jdgSelect.innerHTML = "";
    (appConfig.supported_judge_models || [appConfig.judge_model]).forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      if (m === appConfig.judge_model) opt.selected = true;
      jdgSelect.appendChild(opt);
    });

    // Top-K slider
    const topKInput = document.getElementById("input-top-k");
    const topKDisplay = document.getElementById("top-k-display");
    topKInput.value = appConfig.top_k || 4;
    topKDisplay.textContent = topKInput.value;
    topKInput.addEventListener("input", (e) => {
      topKDisplay.textContent = e.target.value;
    });

    updatePlannedRunsBadge();
  } catch (err) {
    console.error("Config load error:", err);
    const statusEl = document.getElementById("backend-status");
    if (statusEl) {
      statusEl.className = "flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-red-500/10 text-red-400 border border-red-500/20";
      statusEl.innerHTML = '<span class="w-2 h-2 rounded-full bg-red-400"></span><span>Backend Error</span>';
    }
  }
}

// ----------------------------- Sample Data --------------------------------
async function loadSampleDataset() {
  try {
    const res = await fetch("/sample");
    if (res.ok) {
      sampleData = await res.json();
    }
  } catch (err) {
    console.warn("Could not fetch sample dataset:", err);
  }
}

function applySampleDataset() {
  if (!sampleData) return;
  const questionsInput = document.getElementById("questions-input");
  questionsInput.value = sampleData.questions || "";
  updateQuestionCountBadge();

  // Set virtual file
  const blob = new Blob([sampleData.content], { type: "text/markdown" });
  currentFile = new File([blob], sampleData.filename || "rag_guide.md", { type: "text/markdown" });
  showSelectedFile(currentFile.name, currentFile.size);

  if (sampleData.strategies && sampleData.strategies.length > 0) {
    currentStrategies = [...sampleData.strategies];
    renderStrategies();
  }

  updatePlannedRunsBadge();
}

// ----------------------------- UI Handlers --------------------------------
function setupEventHandlers() {
  // Preset selector
  const presetSelect = document.getElementById("preset-select");
  presetSelect.addEventListener("change", (e) => {
    if (appConfig && appConfig.strategy_presets && appConfig.strategy_presets[e.target.value]) {
      currentStrategies = [...appConfig.strategy_presets[e.target.value]];
      renderStrategies();
      updatePlannedRunsBadge();
    }
  });

  // Add Strategy button
  document.getElementById("btn-add-strategy").addEventListener("click", () => {
    const nextIdx = currentStrategies.length + 1;
    currentStrategies.push({
      name: `custom_${nextIdx}`,
      chunk: 512,
      overlap: 50,
    });
    renderStrategies();
    updatePlannedRunsBadge();
  });

  // 1-Click Sample Buttons
  document.getElementById("btn-load-sample").addEventListener("click", applySampleDataset);
  document.getElementById("btn-quick-sample-center").addEventListener("click", applySampleDataset);

  // File Dropzone
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");

  dropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      currentFile = e.target.files[0];
      showSelectedFile(currentFile.name, currentFile.size);
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("border-indigo-500", "bg-indigo-950/20");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("border-indigo-500", "bg-indigo-950/20");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("border-indigo-500", "bg-indigo-950/20");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      currentFile = e.dataTransfer.files[0];
      showSelectedFile(currentFile.name, currentFile.size);
    }
  });

  document.getElementById("btn-remove-file").addEventListener("click", (e) => {
    e.stopPropagation();
    currentFile = null;
    document.getElementById("dropzone-selected").classList.add("hidden");
    document.getElementById("dropzone-idle").classList.remove("hidden");
    fileInput.value = "";
  });

  // Questions input counter
  const questionsInput = document.getElementById("questions-input");
  questionsInput.addEventListener("input", updateQuestionCountBadge);

  // Advanced accordion toggle
  const toggleAdv = document.getElementById("toggle-advanced");
  const advBody = document.getElementById("advanced-settings-body");
  const advChevron = document.getElementById("advanced-chevron");
  toggleAdv.addEventListener("click", () => {
    const isHidden = advBody.classList.contains("hidden");
    if (isHidden) {
      advBody.classList.remove("hidden");
      advChevron.style.transform = "rotate(180deg)";
    } else {
      advBody.classList.add("hidden");
      advChevron.style.transform = "rotate(0deg)";
    }
  });

  // Run Benchmark Button
  document.getElementById("btn-run-benchmark").addEventListener("click", runBenchmark);

  // Tab switching
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((p) => p.classList.add("hidden"));
      btn.classList.add("active");
      const targetPane = document.getElementById(btn.dataset.tab);
      if (targetPane) targetPane.classList.remove("hidden");

      // Trigger Plotly relayout on tab switch to fix sizing
      if (btn.dataset.tab === "tab-analytics") {
        window.dispatchEvent(new Event("resize"));
      }
    });
  });

  // Copy Markdown Report Button
  document.getElementById("btn-copy-md").addEventListener("click", () => {
    const mdText = document.getElementById("markdown-preview-box").textContent;
    navigator.clipboard.writeText(mdText).then(() => {
      const btn = document.getElementById("btn-copy-md");
      btn.innerHTML = '<i data-lucide="check" class="w-3.5 h-3.5 text-emerald-400"></i><span class="text-emerald-400">Copied!</span>';
      if (window.lucide) lucide.createIcons();
      setTimeout(() => {
        btn.innerHTML = '<i data-lucide="copy" class="w-3.5 h-3.5"></i><span>Copy Markdown</span>';
        if (window.lucide) lucide.createIcons();
      }, 2000);
    });
  });

  // Exports
  document.getElementById("btn-export-csv").addEventListener("click", downloadCSV);
  document.getElementById("btn-export-markdown").addEventListener("click", downloadMarkdown);
  document.getElementById("btn-export-json").addEventListener("click", downloadJSON);
}

function showSelectedFile(name, size) {
  document.getElementById("dropzone-idle").classList.add("hidden");
  const selectedEl = document.getElementById("dropzone-selected");
  selectedEl.classList.remove("hidden");
  document.getElementById("selected-file-name").textContent = name;
  const kb = (size / 1024).toFixed(1);
  document.getElementById("selected-file-size").textContent = `(${kb} KB)`;
  if (window.lucide) lucide.createIcons();
}

function updateQuestionCountBadge() {
  const val = document.getElementById("questions-input").value;
  const lines = val.split("\n").map((l) => l.trim()).filter((l) => l.length > 0);
  const badge = document.getElementById("question-count-badge");
  badge.textContent = `${lines.length} question${lines.length === 1 ? "" : "s"}`;
}

// ----------------------------- Strategies & Embedders ---------------------
function renderStrategies() {
  const container = document.getElementById("strategies-list");
  container.innerHTML = "";

  currentStrategies.forEach((strat, idx) => {
    const row = document.createElement("div");
    row.className = "strategy-row flex items-center gap-2 p-2 rounded-lg bg-[#0f172a] border border-slate-800 text-xs";
    row.innerHTML = `
      <input type="text" value="${strat.name}" class="w-2/5 bg-transparent border-b border-slate-700 px-1.5 py-1 text-slate-200 focus:outline-none focus:border-indigo-500 strat-name" data-idx="${idx}" placeholder="Name" />
      <div class="flex items-center gap-1 w-1/4">
        <span class="text-[10px] text-slate-500 font-mono">c:</span>
        <input type="number" value="${strat.chunk}" min="100" max="4000" step="64" class="w-full bg-[#131b2e] border border-slate-700 rounded px-1.5 py-1 text-slate-200 text-center strat-chunk" data-idx="${idx}" />
      </div>
      <div class="flex items-center gap-1 w-1/4">
        <span class="text-[10px] text-slate-500 font-mono">o:</span>
        <input type="number" value="${strat.overlap}" min="0" max="1000" step="10" class="w-full bg-[#131b2e] border border-slate-700 rounded px-1.5 py-1 text-slate-200 text-center strat-overlap" data-idx="${idx}" />
      </div>
      <button type="button" class="text-slate-500 hover:text-red-400 p-1 btn-delete-strat" data-idx="${idx}">
        <i data-lucide="trash-2" class="w-3.5 h-3.5"></i>
      </button>
    `;

    // Row input listeners
    row.querySelector(".strat-name").addEventListener("input", (e) => {
      currentStrategies[idx].name = e.target.value;
    });
    row.querySelector(".strat-chunk").addEventListener("input", (e) => {
      currentStrategies[idx].chunk = parseInt(e.target.value) || 256;
    });
    row.querySelector(".strat-overlap").addEventListener("input", (e) => {
      currentStrategies[idx].overlap = parseInt(e.target.value) || 0;
    });
    row.querySelector(".btn-delete-strat").addEventListener("click", () => {
      currentStrategies.splice(idx, 1);
      renderStrategies();
      updatePlannedRunsBadge();
    });

    container.appendChild(row);
  });

  if (window.lucide) lucide.createIcons();
}

function renderEmbedders(embedders) {
  const container = document.getElementById("embedders-container");
  container.innerHTML = "";

  embedders.forEach((name, idx) => {
    const isChecked = idx < 2; // Default select first two
    const label = document.createElement("label");
    label.className = "flex items-start gap-2 p-2.5 rounded-lg bg-[#0f172a] border border-slate-800 hover:border-slate-700 cursor-pointer text-xs";
    label.innerHTML = `
      <input type="checkbox" value="${name}" class="mt-0.5 rounded border-slate-700 text-indigo-500 focus:ring-0 embedder-checkbox" ${isChecked ? "checked" : ""} />
      <div class="overflow-hidden">
        <div class="font-medium text-slate-200 text-[11px] truncate" title="${name}">${name}</div>
        <div class="text-[10px] text-slate-500 font-mono">local CPU</div>
      </div>
    `;

    label.querySelector(".embedder-checkbox").addEventListener("change", updatePlannedRunsBadge);
    container.appendChild(label);
  });
}

function getSelectedEmbedders() {
  const boxes = document.querySelectorAll(".embedder-checkbox:checked");
  return Array.from(boxes).map((b) => b.value);
}

function updatePlannedRunsBadge() {
  const selectedEmb = getSelectedEmbedders();
  const total = currentStrategies.length * selectedEmb.length;
  const badge = document.getElementById("planned-runs-badge");
  if (badge) {
    badge.textContent = `${total} Run${total === 1 ? "" : "s"}`;
  }
}

// ----------------------------- Benchmark Execution ------------------------
async function runBenchmark() {
  if (!currentFile) {
    alert("Please select or upload a document first, or click '1-Click Sample Demo'.");
    return;
  }
  const questionsVal = document.getElementById("questions-input").value.trim();
  if (!questionsVal) {
    alert("Please enter at least one question.");
    return;
  }
  const selectedEmb = getSelectedEmbedders();
  if (selectedEmb.length === 0) {
    alert("Please select at least one embedding model.");
    return;
  }
  if (currentStrategies.length === 0) {
    alert("Please configure at least one chunking strategy.");
    return;
  }

  // Show running overlay
  const runningState = document.getElementById("running-state");
  const emptyState = document.getElementById("empty-state");
  const resultsView = document.getElementById("results-view");
  const btnRun = document.getElementById("btn-run-benchmark");

  emptyState.classList.add("hidden");
  resultsView.classList.add("hidden");
  runningState.classList.remove("hidden");
  btnRun.disabled = true;
  btnRun.classList.add("opacity-50", "cursor-not-allowed");

  // Timer & ticker
  elapsedSeconds = 0;
  document.getElementById("elapsed-counter").textContent = "0s";
  const tickerEl = document.getElementById("running-status-ticker");
  const tickerSteps = [
    "Loading & validating document text...",
    "Chunking text across specified strategies...",
    "Generating local vector embeddings in isolated ChromaDB collections...",
    "Retrieving Top-K contexts & generating answers via Groq LLM...",
    "Executing LLM-as-a-judge faithfulness & relevance evaluations...",
    "Aggregating benchmark metrics and building reports...",
  ];
  let tickerIdx = 0;
  tickerEl.textContent = tickerSteps[0];

  clearInterval(elapsedTimer);
  elapsedTimer = setInterval(() => {
    elapsedSeconds++;
    document.getElementById("elapsed-counter").textContent = `${elapsedSeconds}s`;
    if (elapsedSeconds % 4 === 0 && tickerIdx < tickerSteps.length - 1) {
      tickerIdx++;
      tickerEl.textContent = tickerSteps[tickerIdx];
    }
  }, 1000);

  // Build FormData
  const formData = new FormData();
  formData.append("file", currentFile);
  formData.append("questions", questionsVal);
  formData.append("strategies", JSON.stringify(currentStrategies));
  formData.append("embedders", selectedEmb.join(","));
  formData.append("top_k", document.getElementById("input-top-k").value);
  formData.append("generator_model", document.getElementById("select-generator").value);
  formData.append("judge_model", document.getElementById("select-judge").value);

  try {
    const res = await fetch("/optimize", {
      method: "POST",
      body: formData,
    });

    clearInterval(elapsedTimer);
    runningState.classList.add("hidden");
    btnRun.disabled = false;
    btnRun.classList.remove("opacity-50", "cursor-not-allowed");

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Benchmark failed with status ${res.status}`);
    }

    benchmarkPayload = await res.json();
    renderResults(benchmarkPayload);
  } catch (err) {
    clearInterval(elapsedTimer);
    runningState.classList.add("hidden");
    btnRun.disabled = false;
    btnRun.classList.remove("opacity-50", "cursor-not-allowed");
    emptyState.classList.remove("hidden");
    alert(`Benchmark Error: ${err.message}`);
  }
}

// ----------------------------- Results Presentation -----------------------
function renderResults(payload) {
  const results = payload.results || [];
  if (results.length === 0) {
    alert("No successful experiment results were returned.");
    return;
  }

  document.getElementById("empty-state").classList.add("hidden");
  const resultsView = document.getElementById("results-view");
  resultsView.classList.remove("hidden");
  resultsView.classList.add("flex");

  // Calculate Best overall, Fastest, Most faithful, and Top hit-rate
  const sortedByOverall = [...results].sort((a, b) => (b.summary.overall || 0) - (a.summary.overall || 0));
  const best = sortedByOverall[0];

  const sortedByLatency = [...results].sort((a, b) => (a.summary.avg_latency_s || 999) - (b.summary.avg_latency_s || 999));
  const fastest = sortedByLatency[0];

  const sortedByFaith = [...results].sort((a, b) => (b.summary.avg_faithfulness || 0) - (a.summary.avg_faithfulness || 0));
  const mostFaithful = sortedByFaith[0];

  const hitCandidates = results.filter((r) => r.summary.retrieval_hit_rate !== null && r.summary.retrieval_hit_rate !== undefined);
  const bestHit = hitCandidates.length > 0 ? hitCandidates.sort((a, b) => b.summary.retrieval_hit_rate - a.summary.retrieval_hit_rate)[0] : null;

  // KPI Cards
  document.getElementById("kpi-best-score").textContent = `${best.summary.overall || 0}/10`;
  document.getElementById("kpi-best-strategy").textContent = `${best.summary.strategy} (${best.summary.embedder.split("/").pop()})`;

  document.getElementById("kpi-fastest-latency").textContent = `${fastest.summary.avg_latency_s || 0}s`;
  document.getElementById("kpi-fastest-strategy").textContent = `${fastest.summary.strategy}`;

  document.getElementById("kpi-faith-score").textContent = `${mostFaithful.summary.avg_faithfulness || 0}/10`;
  document.getElementById("kpi-faith-strategy").textContent = `${mostFaithful.summary.strategy}`;

  if (bestHit && bestHit.summary.retrieval_hit_rate !== null) {
    document.getElementById("kpi-hit-rate").textContent = `${Math.round(bestHit.summary.retrieval_hit_rate * 100)}%`;
    document.getElementById("kpi-hit-strategy").textContent = `${bestHit.summary.strategy}`;
  } else {
    document.getElementById("kpi-hit-rate").textContent = "N/A";
    document.getElementById("kpi-hit-strategy").textContent = "No expected phrase";
  }

  // Render Charts
  renderVisualCharts(results);

  // Render Leaderboard
  renderLeaderboardTable(sortedByOverall);

  // Render Inspector
  setupQuestionInspector(results);

  // Render Details Accordion
  renderDetailsAccordion(results);

  // Render Markdown Preview
  document.getElementById("markdown-preview-box").textContent = payload.markdown_report || "Report not available.";

  if (window.lucide) lucide.createIcons();
}

// ----------------------------- Plotly Charts ------------------------------
function renderVisualCharts(results) {
  const darkLayout = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { color: "#94a3b8", size: 10, family: "monospace" },
    margin: { t: 30, b: 40, l: 40, r: 20 },
  };

  const labels = results.map((r) => `${r.summary.strategy} | ${r.summary.embedder.split("/").pop()}`);

  // 1. Quality Chart (Relevance vs Faithfulness)
  const traceRel = {
    x: labels,
    y: results.map((r) => r.summary.avg_relevance || 0),
    name: "Relevance",
    type: "bar",
    marker: { color: "#6366f1" },
  };
  const traceFaith = {
    x: labels,
    y: results.map((r) => r.summary.avg_faithfulness || 0),
    name: "Faithfulness",
    type: "bar",
    marker: { color: "#10b981" },
  };
  Plotly.newPlot(
    "chart-quality",
    [traceRel, traceFaith],
    {
      ...darkLayout,
      barmode: "group",
      yaxis: { range: [0, 10], gridcolor: "#1e293b" },
      xaxis: { gridcolor: "#1e293b", tickangle: -20 },
    },
    { responsive: true, displayModeBar: false }
  );

  // 2. Radar Chart
  const radarCategories = ["Relevance", "Faithfulness", "Hit Rate", "Speed Score"];
  const maxLat = Math.max(...results.map((r) => r.summary.avg_latency_s || 1.0)) || 1.0;

  const radarTraces = results.slice(0, 4).map((r, i) => {
    const latScore = Math.max(1.0, 10.0 - ((r.summary.avg_latency_s || 1.0) / maxLat) * 8.0);
    const hitScore = r.summary.retrieval_hit_rate !== null ? r.summary.retrieval_hit_rate * 10.0 : 8.0;
    const vals = [
      r.summary.avg_relevance || 5.0,
      r.summary.avg_faithfulness || 5.0,
      hitScore,
      latScore,
      r.summary.avg_relevance || 5.0, // close loop
    ];
    return {
      type: "scatterpolar",
      r: vals,
      theta: [...radarCategories, radarCategories[0]],
      fill: "toself",
      name: `${r.summary.strategy}`,
    };
  });

  Plotly.newPlot(
    "chart-radar",
    radarTraces,
    {
      ...darkLayout,
      polar: {
        radialaxis: { visible: true, range: [0, 10], gridcolor: "#1e293b" },
        angularaxis: { gridcolor: "#1e293b" },
        bgcolor: "transparent",
      },
    },
    { responsive: true, displayModeBar: false }
  );

  // 3. Pareto Frontier (Latency vs Quality)
  const tracePareto = {
    x: results.map((r) => r.summary.avg_latency_s || 0),
    y: results.map((r) => r.summary.overall || 0),
    text: labels,
    mode: "markers+text",
    textposition: "top center",
    marker: {
      size: results.map((r) => Math.max(10, Math.min(30, (r.summary.num_chunks || 5) * 2))),
      color: results.map((r) => r.summary.overall || 0),
      colorscale: "Viridis",
      showscale: false,
    },
    type: "scatter",
  };
  Plotly.newPlot(
    "chart-pareto",
    [tracePareto],
    {
      ...darkLayout,
      xaxis: { title: "Latency (s)", gridcolor: "#1e293b" },
      yaxis: { title: "Overall Score (1-10)", range: [0, 10], gridcolor: "#1e293b" },
    },
    { responsive: true, displayModeBar: false }
  );

  // 4. Token Consumption
  const traceInTokens = {
    x: labels,
    y: results.map((r) => r.summary.total_input_tokens || 0),
    name: "Input Tokens",
    type: "bar",
    marker: { color: "#f59e0b" },
  };
  const traceOutTokens = {
    x: labels,
    y: results.map((r) => r.summary.total_output_tokens || 0),
    name: "Output Tokens",
    type: "bar",
    marker: { color: "#ef4444" },
  };
  Plotly.newPlot(
    "chart-tokens",
    [traceInTokens, traceOutTokens],
    {
      ...darkLayout,
      barmode: "stack",
      yaxis: { gridcolor: "#1e293b" },
      xaxis: { gridcolor: "#1e293b", tickangle: -20 },
    },
    { responsive: true, displayModeBar: false }
  );
}

// ----------------------------- Leaderboard --------------------------------
function renderLeaderboardTable(sortedResults) {
  const tbody = document.getElementById("leaderboard-tbody");
  tbody.innerHTML = "";

  sortedResults.forEach((r, idx) => {
    const s = r.summary;
    const rankMedal = idx === 0 ? "🥇 #1" : idx === 1 ? "🥈 #2" : idx === 2 ? "🥉 #3" : `#${idx + 1}`;
    const hitRateText = s.retrieval_hit_rate !== null ? `${Math.round(s.retrieval_hit_rate * 100)}%` : "N/A";
    const totalTokens = (s.total_input_tokens || 0) + (s.total_output_tokens || 0);

    const tr = document.createElement("tr");
    tr.className = "hover:bg-slate-800/40 transition-colors";
    tr.innerHTML = `
      <td class="py-2.5 px-3 font-bold ${idx === 0 ? "text-amber-400" : "text-slate-300"}">${rankMedal}</td>
      <td class="py-2.5 px-3 font-semibold text-white">${s.strategy}</td>
      <td class="py-2.5 px-3 text-slate-400 text-[11px]">${s.embedder}</td>
      <td class="py-2.5 px-3 font-bold text-emerald-400">${s.overall || 0}/10</td>
      <td class="py-2.5 px-3 text-indigo-300">${s.avg_relevance || 0}</td>
      <td class="py-2.5 px-3 text-purple-300">${s.avg_faithfulness || 0}</td>
      <td class="py-2.5 px-3 ${s.retrieval_hit_rate === 1.0 ? "text-emerald-400" : "text-slate-300"}">${hitRateText}</td>
      <td class="py-2.5 px-3 text-sky-400">${s.avg_latency_s || 0}s</td>
      <td class="py-2.5 px-3 text-slate-400">${s.num_chunks || 0}</td>
      <td class="py-2.5 px-3 text-amber-300">${totalTokens.toLocaleString()}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ----------------------------- Question Inspector -------------------------
function setupQuestionInspector(results) {
  const qSelect = document.getElementById("inspector-question-select");
  qSelect.innerHTML = "";

  const firstDetails = results[0].details || [];
  firstDetails.forEach((d, idx) => {
    const opt = document.createElement("option");
    opt.value = d.question;
    opt.textContent = `Q${idx + 1}: ${d.question}`;
    qSelect.appendChild(opt);
  });

  const updateColumns = () => {
    const chosenQ = qSelect.value;
    const container = document.getElementById("inspector-columns");
    container.innerHTML = "";

    // Show top 2-3 experiments
    const topExp = results.slice(0, 3);
    document.getElementById("inspector-count").textContent = topExp.length;

    topExp.forEach((r) => {
      const match = r.details.find((d) => d.question === chosenQ);
      if (!match) return;

      const hitBadge = match.retrieval_hit === true
        ? '<span class="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">✅ Hit</span>'
        : match.retrieval_hit === false
        ? '<span class="text-[10px] px-1.5 py-0.5 rounded bg-red-500/10 text-red-400 border border-red-500/20">❌ Missed</span>'
        : '';

      const col = document.createElement("div");
      col.className = "bg-[#0f172a] border border-slate-800 rounded-xl p-4 flex flex-col gap-3";
      col.innerHTML = `
        <div class="border-b border-slate-800 pb-2">
          <div class="text-xs font-bold text-white">${r.summary.strategy}</div>
          <div class="text-[10px] text-slate-500 font-mono">${r.summary.embedder}</div>
          <div class="flex items-center gap-2 mt-1.5 text-[11px]">
            <span class="text-indigo-400 font-bold">Rel: ${match.relevance_score || 'N/A'}/10</span>
            <span class="text-emerald-400 font-bold">Faith: ${match.faithfulness_score || 'N/A'}/10</span>
            ${hitBadge}
          </div>
        </div>

        <div>
          <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1">Generated Answer:</span>
          <div class="p-2.5 rounded bg-[#131b2e] text-xs text-slate-200 border border-slate-800">${match.answer}</div>
        </div>

        ${match.explanation ? `
        <div class="text-[11px] text-slate-400">
          <span class="font-bold text-slate-300">Judge Rationale:</span> ${match.explanation}
        </div>` : ''}

        <details class="text-xs text-slate-400">
          <summary class="cursor-pointer text-[11px] text-indigo-400 hover:text-indigo-300 font-medium">View Retrieved Chunks (${(match.retrieved_chunks || []).length})</summary>
          <div class="mt-2 space-y-1.5 max-h-48 overflow-y-auto">
            ${(match.retrieved_chunks || [match.context]).map((c, cIdx) => `
              <div class="p-2 bg-[#0b0f17] border border-slate-800 rounded text-[10px] font-mono text-slate-300">
                <span class="text-indigo-400 font-bold block mb-0.5">Chunk #${cIdx + 1}</span>
                ${c.slice(0, 300)}${c.length > 300 ? '...' : ''}
              </div>
            `).join('')}
          </div>
        </details>
      `;
      container.appendChild(col);
    });
  };

  qSelect.addEventListener("change", updateColumns);
  updateColumns();
}

// ----------------------------- Details Accordion --------------------------
function renderDetailsAccordion(results) {
  const container = document.getElementById("experiments-accordion");
  container.innerHTML = "";

  results.forEach((r, expIdx) => {
    const s = r.summary;
    const details = document.createElement("details");
    details.className = "bg-[#0f172a] border border-slate-800 rounded-xl overflow-hidden group";
    details.innerHTML = `
      <summary class="p-4 cursor-pointer flex items-center justify-between text-xs font-semibold text-slate-200 hover:text-white bg-[#131b2e]/60 transition-colors">
        <div class="flex items-center gap-2">
          <span class="w-6 h-6 rounded-full bg-indigo-500/20 text-indigo-400 flex items-center justify-center text-[10px] font-bold">#${expIdx + 1}</span>
          <span>${s.strategy}</span>
          <span class="text-slate-500">|</span>
          <span class="text-slate-400 font-mono text-[11px]">${s.embedder}</span>
        </div>
        <div class="flex items-center gap-3">
          <span class="text-emerald-400 font-bold">${s.overall || 0}/10</span>
          <span class="text-sky-400 font-mono text-[11px]">${s.avg_latency_s}s</span>
          <i data-lucide="chevron-down" class="w-4 h-4 text-slate-400 group-open:rotate-180 transition-transform"></i>
        </div>
      </summary>

      <div class="p-4 border-t border-slate-800 space-y-4">
        ${r.details.map((d, dIdx) => `
          <div class="p-3 bg-[#0b0f17] border border-slate-800/80 rounded-lg text-xs space-y-2">
            <div class="font-bold text-white">Q${dIdx + 1}: ${d.question}</div>
            <div class="text-slate-300 bg-[#131b2e] p-2 rounded border border-slate-800">${d.answer}</div>
            <div class="flex flex-wrap items-center gap-3 text-[11px] text-slate-400">
              <span>Relevance: <b class="text-indigo-400">${d.relevance_score || 'N/A'}/10</b></span>
              <span>Faithfulness: <b class="text-emerald-400">${d.faithfulness_score || 'N/A'}/10</b></span>
              <span>Hit: <b class="${d.retrieval_hit ? 'text-emerald-400' : 'text-slate-400'}">${d.retrieval_hit === true ? 'Yes' : d.retrieval_hit === false ? 'No' : 'N/A'}</b></span>
              <span>Latency: <b>${d.latency_s}s</b></span>
              <span>Tokens: in ${d.input_tokens}, out ${d.output_tokens}</span>
            </div>
            ${d.explanation ? `<div class="text-[11px] text-slate-400 italic">Judge: ${d.explanation}</div>` : ''}
          </div>
        `).join('')}
      </div>
    `;
    container.appendChild(details);
  });
}

// ----------------------------- Downloads ----------------------------------
function downloadCSV() {
  if (!benchmarkPayload || !benchmarkPayload.results) return;
  const rows = [];
  benchmarkPayload.results.forEach((r) => {
    const s = r.summary;
    r.details.forEach((d) => {
      rows.push({
        strategy: s.strategy,
        embedder: s.embedder,
        chunk_size: s.chunk_size,
        overlap: s.overlap,
        overall_score: s.overall,
        question: d.question,
        expected: d.expected || "",
        answer: d.answer,
        relevance_score: d.relevance_score,
        faithfulness_score: d.faithfulness_score,
        retrieval_hit: d.retrieval_hit,
        latency_s: d.latency_s,
        input_tokens: d.input_tokens,
        output_tokens: d.output_tokens,
        judge_explanation: d.explanation || "",
      });
    });
  });

  if (rows.length === 0) return;
  const keys = Object.keys(rows[0]);
  const csvContent = [
    keys.join(","),
    ...rows.map((row) => keys.map((k) => `"${String(row[k] ?? "").replace(/"/g, '""')}"`).join(",")),
  ].join("\n");

  downloadFile("rag_benchmark_results.csv", "text/csv", csvContent);
}

function downloadMarkdown() {
  if (!benchmarkPayload || !benchmarkPayload.markdown_report) return;
  downloadFile("rag_benchmark_report.md", "text/markdown", benchmarkPayload.markdown_report);
}

function downloadJSON() {
  if (!benchmarkPayload) return;
  const jsonContent = JSON.stringify(benchmarkPayload, null, 2);
  downloadFile("rag_benchmark_data.json", "application/json", jsonContent);
}

function downloadFile(filename, mimeType, content) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
