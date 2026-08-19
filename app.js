const ALPHA_SWEEP = [1.0, 0.5, 0.25].map((alpha) => ({ alpha, beta: 0.0 }));
const BETA_SWEEP = [0.0, 0.4, 0.8].map((beta) => ({ alpha: 1.0, beta }));
const PARAMS = uniqueParams([...ALPHA_SWEEP, ...BETA_SWEEP]);
const MODEL_COLORS = {
  RotatE: "#85BAEA",
  ComplEx: "#14293A",
  HousE: "#2B8D8F",
  TuckER: "#eeb72bce",
  pLogicNet: "#FF8F5C",
  RNNLogic: "#99251F",
};
const USER_MODEL_COLORS = ["#111827", "#dc2626", "#2563eb", "#16a34a", "#9333ea", "#f97316", "#0891b2", "#be123c"];

const state = {
  manifest: null,
  lastResult: null,
  analysis: null,
  userModelRowCount: 0,
};

const el = {
  datasetName: document.getElementById("datasetName"),
  datasetEntitiesFile: document.getElementById("datasetEntitiesFile"),
  datasetRelationsFile: document.getElementById("datasetRelationsFile"),
  datasetTrainFile: document.getElementById("datasetTrainFile"),
  userModels: document.getElementById("userModels"),
  addModelButton: document.getElementById("addModelButton"),
  useBaselines: document.getElementById("useBaselines"),
  baselineOnly: document.getElementById("baselineOnly"),
  runButton: document.getElementById("runButton"),
  status: document.getElementById("status"),
  results: document.getElementById("results"),
  summary: document.getElementById("summary"),
  plots: document.getElementById("plots"),
  downloadJson: document.getElementById("downloadJson"),
  downloadCsv: document.getElementById("downloadCsv"),
  downloadSvgs: document.getElementById("downloadSvgs"),
  printPdf: document.getElementById("printPdf"),
};

el.addModelButton.addEventListener("click", () => addUserModelRow());
el.runButton.addEventListener("click", runComparison);
el.baselineOnly.addEventListener("change", () => {
  const enabled = el.baselineOnly.checked;
  setUserModelsDisabled(enabled);
  el.datasetEntitiesFile.disabled = enabled;
  el.datasetRelationsFile.disabled = enabled;
  el.datasetTrainFile.disabled = enabled;
  if (enabled) el.useBaselines.checked = true;
});
el.downloadJson.addEventListener("click", () => {
  if (state.lastResult) downloadText("probe-results.json", JSON.stringify(state.lastResult, null, 2), "application/json");
});
el.downloadCsv.addEventListener("click", () => {
  if (state.lastResult) downloadText("probe-results.csv", resultToCsv(state.lastResult), "text/csv");
});
el.downloadSvgs.addEventListener("click", downloadPlotSvgs);
el.printPdf.addEventListener("click", printResultsPdf);

loadManifest();
addUserModelRow();

function addUserModelRow(initial = {}) {
  state.userModelRowCount += 1;
  const rowIndex = state.userModelRowCount;
  const color = initial.color || USER_MODEL_COLORS[(rowIndex - 1) % USER_MODEL_COLORS.length];
  const row = document.createElement("div");
  row.className = "userModelRow";
  row.dataset.userModelRow = String(rowIndex);
  row.innerHTML = `
    <label>
      Model name
      <input type="text" data-user-model-name placeholder="MyModel ${rowIndex}" value="${escapeHtml(initial.name || "")}" />
    </label>
    <label class="colorField">
      Color
      <input type="color" data-user-model-color value="${escapeHtml(color)}" />
    </label>
    <label class="fileDrop userModelFiles">
      Result JSON files
      <input type="file" data-user-model-files multiple accept=".json,application/json" />
      <span>List of [[h,r,t], mode, rank]. Triple elements must be id, mode is either "h" or "t"</span>
    </label>
    <button class="secondary userModelRemove" type="button">Remove</button>
  `;

  row.querySelector(".userModelRemove").addEventListener("click", () => {
    row.remove();
    if (!el.userModels.querySelector("[data-user-model-row]")) addUserModelRow();
    updateUserModelControls();
  });

  el.userModels.appendChild(row);
  updateUserModelControls();
}

function getUserModelInputs() {
  return [...el.userModels.querySelectorAll("[data-user-model-row]")].map((row, index) => {
    const name = row.querySelector("[data-user-model-name]").value.trim() || `Uploaded model ${index + 1}`;
    const color = row.querySelector("[data-user-model-color]").value || USER_MODEL_COLORS[index % USER_MODEL_COLORS.length];
    const files = [...row.querySelector("[data-user-model-files]").files];
    return { name, color, files };
  });
}

function setUserModelsDisabled(disabled) {
  el.addModelButton.disabled = disabled;
  for (const row of el.userModels.querySelectorAll("[data-user-model-row]")) {
    for (const input of row.querySelectorAll("input")) input.disabled = disabled;
  }
  updateUserModelControls();
}

function updateUserModelControls() {
  const rows = [...el.userModels.querySelectorAll("[data-user-model-row]")];
  const disabled = el.baselineOnly.checked;
  el.addModelButton.disabled = disabled;
  rows.forEach((row) => {
    for (const input of row.querySelectorAll("input")) input.disabled = disabled;
    const removeButton = row.querySelector(".userModelRemove");
    removeButton.disabled = disabled || rows.length === 1;
  });
}

async function loadManifest() {
  try {
    const response = await fetch("baseline-manifest.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.manifest = await response.json();
    const count = Object.values(state.manifest.datasets || {}).reduce(
      (sum, models) => sum + Object.values(models).reduce((inner, files) => inner + files.length, 0),
      0
    );
    setStatus(`Baseline manifest loaded (${count} files). Waiting for uploads.`);
  } catch (error) {
    setStatus(`Baseline manifest was not loaded. You can still score uploaded model files.\n${error.message}`);
  }
}

async function runComparison() {
  el.runButton.disabled = true;
  el.results.hidden = true;
  try {
    let datasetName = el.datasetName.value.trim();
    const entityFile = el.datasetEntitiesFile.files[0];
    const relationFile = el.datasetRelationsFile.files[0];
    const trainFile = el.datasetTrainFile.files[0];
    const baselineOnly = el.baselineOnly.checked;
    const userModels = getUserModelInputs().filter((model) => model.files.length);

    if (!datasetName && baselineOnly) datasetName = firstManifestDataset();
    if (!datasetName) throw new Error("Enter a dataset name.");
    if (!baselineOnly && !entityFile) throw new Error("Upload entities.dict or use built-in baselines only.");
    if (!baselineOnly && !trainFile) throw new Error("Upload train.txt or use built-in baselines only.");
    if (!baselineOnly && !userModels.length) throw new Error("Upload at least one user model result JSON file, or use baselines only.");

    setStatus(baselineOnly ? "Loading built-in dataset..." : "Parsing uploaded dataset...");
    const dataset = baselineOnly ? await fetchDatasetByName(datasetName) : await parseDataset({ entityFile, relationFile, trainFile, datasetName });

    const modelGroups = [];
    if (!baselineOnly && userModels.length) {
      setStatus(`Parsing ${userModels.length} uploaded model${userModels.length > 1 ? "s" : ""}...`);
      for (const [modelIndex, userModel] of userModels.entries()) {
        const userInfos = await readJsonFiles(userModel.files);
        modelGroups.push({
          model: userModel.name || `Uploaded model ${modelIndex + 1}`,
          source: "user",
          color: userModel.color,
          files: userInfos.map((info, index) => ({
            name: userModel.files[index].name,
            info,
          })),
        });
      }
    }

    if (el.useBaselines.checked || baselineOnly) {
      const baselineModels = state.manifest?.datasets?.[datasetName] || {};
      const names = Object.keys(baselineModels);
      setStatus(names.length ? `Loading baselines for ${datasetName}: ${names.join(", ")}...` : `No baseline files found for ${datasetName}.`);

      for (const [baselineModel, paths] of Object.entries(baselineModels)) {
        const files = [];
        for (const path of paths) {
          const info = await fetchJson(path);
          files.push({ name: basename(path), info });
        }
        modelGroups.push({ model: baselineModel, source: "baseline", files });
      }
    }

    if (!modelGroups.length) throw new Error("No models to compare. Upload model files or enable baselines.");

    assignModelIds(modelGroups);
    state.analysis = { dataset, modelGroups };

    setStatus("Calculating PROBE metrics...");
    const result = calculateResults({
      datasetName,
      dataset,
      params: PARAMS,
      modelGroups,
    });

    state.lastResult = result;
    renderResults(result);
    setStatus(
      [
        "Done.",
        `Dataset: ${datasetName}`,
        `Entities: ${dataset.nentity}`,
        `Training triples: ${dataset.trainTripleCount}`,
        `Models compared: ${result.models.length}`,
      ].join("\n")
    );
  } catch (error) {
    setStatus(`Error: ${error.message}`);
  } finally {
    el.runButton.disabled = false;
  }
}

async function parseDataset({ entityFile, relationFile, trainFile, datasetName = "" }) {
  if (!entityFile) throw new Error("Dataset upload is missing entities.dict.");
  if (!trainFile) throw new Error("Dataset upload is missing train.txt.");

  return parseDatasetTexts({
    entitiesText: await entityFile.text(),
    relationsText: relationFile ? await relationFile.text() : "",
    trainText: await trainFile.text(),
    datasetName,
  });
}

async function fetchDatasetByName(datasetName) {
  const [entitiesResponse, relationsResponse, trainResponse] = await Promise.all([
    fetch(encodePath(`data/${datasetName}/entities.dict`)),
    fetch(encodePath(`data/${datasetName}/relations.dict`)),
    fetch(encodePath(`data/${datasetName}/train.txt`)),
  ]);

  if (!entitiesResponse.ok || !trainResponse.ok) {
    throw new Error(`Built-in dataset files were not found for ${datasetName}.`);
  }

  return parseDatasetTexts({
    entitiesText: await entitiesResponse.text(),
    relationsText: relationsResponse.ok ? await relationsResponse.text() : "",
    trainText: await trainResponse.text(),
    datasetName,
  });
}

async function parseDatasetTexts({ entitiesText, relationsText, trainText, datasetName }) {
  const entity2id = parseEntityDict(entitiesText);
  const id2entity = invertMap(entity2id);
  const relation2id = relationsText ? parseEntityDict(relationsText) : {};
  const id2relation = invertMap(relation2id);
  const counts = Object.create(null);
  for (const id of Object.values(entity2id)) counts[id] = 0;

  let trainTripleCount = 0;
  const trainTriples = [];
  for (const line of trainText.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const [head, relation, tail] = line.split("\t");
    const headId = entity2id[head];
    const tailId = entity2id[tail];
    const relationId = relation2id[relation] ?? relation;
    if (headId === undefined || tailId === undefined) {
      throw new Error(`train.txt contains an entity that is not in entities.dict: ${line}`);
    }
    trainTriples.push([headId, relationId, tailId]);
    counts[headId] += 1;
    counts[tailId] += 1;
    trainTripleCount += 1;
  }
  const popularity = buildPopularityInfo(counts);

  return {
    nentity: Object.keys(entity2id).length,
    trainTripleCount,
    trainTriples,
    counts,
    popularityPercentiles: popularity.percentiles,
    popularityPositions: popularity.positions,
    popularityOrder: popularity.order,
    decodeMap: await fetchDecodeMap(datasetName),
    id2entity,
    id2relation,
  };
}

function parseEntityDict(text) {
  const entity2id = Object.create(null);
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const [id, entity] = line.split("\t");
    if (id === undefined || entity === undefined) throw new Error(`Invalid entities.dict row: ${line}`);
    entity2id[entity] = Number(id);
  }
  return entity2id;
}

function invertMap(map) {
  return Object.fromEntries(Object.entries(map).map(([key, value]) => [value, key]));
}

async function fetchDecodeMap(datasetName) {
  const normalized = String(datasetName || "").toLowerCase();
  const decodePath =
    normalized === "fb15k237"
      ? "data/FB15k237/FB_decoded.json"
      : normalized === "wn18rr"
        ? "data/wn18rr/WN_decoded.json"
        : "";
  if (!decodePath) return {};
  try {
    const response = await fetch(encodePath(decodePath));
    return response.ok ? await response.json() : {};
  } catch {
    return {};
  }
}

function buildPopularityInfo(counts) {
  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1] || Number(a[0]) - Number(b[0]));
  const total = Math.max(1, sorted.length);
  const percentiles = Object.create(null);
  const positions = Object.create(null);
  sorted.forEach(([id], index) => {
    percentiles[id] = ((index + 1) / total) * 100;
    positions[id] = index + 1;
  });
  return {
    percentiles,
    positions,
    order: sorted.map(([id, degree], index) => ({
      id: Number(id),
      degree,
      position: index + 1,
    })),
  };
}

async function readJsonFiles(files) {
  const parsed = [];
  for (const file of files) {
    try {
      parsed.push(JSON.parse(await file.text()));
    } catch (error) {
      throw new Error(`${file.name} is not valid JSON: ${error.message}`);
    }
  }
  return parsed;
}

async function fetchJson(path) {
  const response = await fetch(encodePath(path));
  if (!response.ok) throw new Error(`Could not load ${path}: HTTP ${response.status}`);
  return response.json();
}

function calculateResults({ datasetName, dataset, params, modelGroups }) {
  const models = modelGroups.map((group) => {
    const runs = group.files.map((file) => {
      validateInfoDump(file.info, file.name);
      return {
        file: file.name,
        scores: Object.fromEntries(
          params.map(({ alpha, beta }) => [paramKey(alpha, beta), calculateProbeMetric(file.info, dataset, alpha, beta)])
        ),
      };
    });

    return {
      id: group.id,
      model: group.model,
      source: group.source,
      color: group.color,
      runCount: runs.length,
      runs,
      scores: Object.fromEntries(
        params.map(({ alpha, beta }) => {
          const values = runs.map((run) => run.scores[paramKey(alpha, beta)]);
          return [paramKey(alpha, beta), summarize(values)];
        })
      ),
    };
  });

  return {
    dataset: {
      name: datasetName,
      nentity: dataset.nentity,
      trainTripleCount: dataset.trainTripleCount,
    },
    params,
    models,
  };
}

function calculateProbeMetric(info, dataset, alpha, beta) {
  const nCoeff = Math.pow(1 / dataset.nentity, alpha);
  let weightSum = 0;
  let weightedScore = 0;

  for (const row of info) {
    const [query, mode, rawRank] = row;
    const rank = Number(rawRank);
    if (!Number.isFinite(rank) || rank <= 0) throw new Error(`Invalid rank value: ${rawRank}`);

    const targetEntity = mode === "h" ? Number(query[0]) : Number(query[2]);
    const entityCount = dataset.counts[targetEntity] ?? 0;
    const transformedRank = (Math.pow(1 / rank, alpha) - nCoeff) / (1 - nCoeff);
    const rawWeight = 1 / Math.pow(1 + entityCount, beta);
    weightSum += rawWeight;
    weightedScore += transformedRank * rawWeight;
  }

  if (!weightSum) throw new Error("Info dump has no rows.");
  return weightedScore / weightSum;
}

function validateInfoDump(info, name) {
  if (!Array.isArray(info) || !info.length) throw new Error(`${name} must be a non-empty JSON array.`);
  for (let index = 0; index < info.length; index += 1) {
    const row = info[index];
    if (!Array.isArray(row) || row.length < 3) throw new Error(`${name} row ${index} must be [query, mode, rank].`);
    if (!Array.isArray(row[0]) || row[0].length < 3) throw new Error(`${name} row ${index} query must be [h, r, t].`);
    if (row[1] !== "h" && row[1] !== "t") throw new Error(`${name} row ${index} mode must be "h" or "t".`);
  }
}

function summarize(values) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / values.length;
  return {
    mean,
    std: Math.sqrt(variance),
    min: Math.min(...values),
    max: Math.max(...values),
  };
}

function renderResults(result) {
  el.results.hidden = false;
  el.summary.innerHTML = "";
  el.plots.innerHTML = "";

  const baselineCount = result.models.filter((model) => model.source === "baseline").length;
  const stats = [
    ["Dataset", result.dataset.name],
    ["Entities", result.dataset.nentity.toLocaleString()],
    ["Train triples", result.dataset.trainTripleCount.toLocaleString()],
    ["Baselines", baselineCount.toLocaleString()],
  ];

  for (const [label, value] of stats) {
    const node = document.createElement("div");
    node.className = "stat";
    node.innerHTML = `<small>${escapeHtml(label)}</small><strong>${escapeHtml(String(value))}</strong>`;
    el.summary.appendChild(node);
  }

  renderStandardMetrics(result);
  renderAlphaBetaSliderSweep(result);
  renderAlphaShowcase(result);
  renderBetaShowcase(result);
  // renderHyperPlaneSection();
  render3DHyperPlaneSection();
}

function renderSweep(result, title, varyingKey, params) {
  const section = document.createElement("section");
  section.className = "plotSweep";
  section.innerHTML = `<h3>${escapeHtml(title)}</h3>`;

  const row = document.createElement("div");
  row.className = "plotRow";

  for (const param of params) {
    const chart = document.createElement("article");
    chart.className = "chartPanel";
    const label =
      varyingKey === "alpha"
        ? `alpha=${param.alpha}, beta=${param.beta}`
        : `beta=${param.beta}, alpha=${param.alpha}`;
    chart.innerHTML = `
      <div class="chartTitle">${escapeHtml(label)}</div>
      ${renderBarChartSvg(getRowsForParam(result, param), label)}
    `;
    row.appendChild(chart);
  }

  section.appendChild(row);
  el.plots.appendChild(section);
}

function renderStandardMetrics(result) {
  const metrics = [
    { key: "mrr", label: "MRR" },
    { key: "hits1", label: "Hits@1" },
    { key: "hits3", label: "Hits@3" },
  ];
  const section = document.createElement("section");
  section.className = "plotSweep standardMetrics";
  section.innerHTML = `<h3>Standard metrics</h3>`;

  const row = document.createElement("div");
  row.className = "plotRow";

  for (const metric of metrics) {
    const rows = result.models
      .map((model) => {
        const group = state.analysis.modelGroups.find((item) => item.id === model.id);
        const values = group.files.map((file) => calculateStandardMetric(file.info, metric.key));
        const stat = summarize(values);
        return {
          model: model.model,
          source: model.source,
          runCount: model.runCount,
          score: stat.mean,
          std: stat.std,
          color: modelColor(model),
        };
      })
      .sort((a, b) => b.score - a.score);

    const chart = document.createElement("article");
    chart.className = "chartPanel";
    chart.innerHTML = `
      <div class="chartTitle">${escapeHtml(metric.label)}</div>
      ${renderBarChartSvg(rows, metric.label)}
    `;
    row.appendChild(chart);
  }

  section.appendChild(row);
  el.plots.appendChild(section);
}

function calculateStandardMetric(info, metric) {
  if (!info.length) return 0;
  const values = info.map((row) => Number(row[2])).filter((rank) => Number.isFinite(rank) && rank > 0);
  if (!values.length) return 0;
  if (metric === "mrr") return values.reduce((sum, rank) => sum + 1 / rank, 0) / values.length;
  if (metric === "hits1") return values.filter((rank) => rank <= 1).length / values.length;
  if (metric === "hits3") return values.filter((rank) => rank <= 3).length / values.length;
  return 0;
}

function renderAlphaBetaSliderSweep(result) {
  const section = document.createElement("section");
  section.className = "plotSweep sliderSweep alphaBetaSweep";
  section.innerHTML = `
    <div class="sliderSweepLayout">
      <article class="chartPanel sliderSweepPanel alphaBetaSweepPanel">
        <div class="chartTitle">Predictive sharpness & popularity-bias robustness slider evaluation</div>
        <div class="alphaBetaSweepBody">
          <div class="combinedDynamicChart"></div>
          <div class="verticalSweepControls">
            <label class="verticalSweepControl">
              <span class="verticalControlHead">alpha <strong data-alpha-value>1.00</strong></span>
              <span class="verticalSliderStack">
                <small>high predictive<br>sharpness</small>
                <span class="verticalRangeFrame">
                  <input class="verticalRange" type="range" min="0.01" max="2" step="0.01" value="1" data-alpha-slider />
                </span>
                <small>less predictive<br>sharpness</small>
              </span>
            </label>
            <label class="verticalSweepControl">
              <span class="verticalControlHead">beta <strong data-beta-value>0.00</strong></span>
              <span class="verticalSliderStack">
                <small>high popularity<br>bias</small>
                <span class="verticalRangeFrame">
                  <input class="verticalRange" type="range" min="0" max="1" step="0.01" value="0" data-beta-slider />
                </span>
                <small>less popularity<br>bias</small>
              </span>
            </label>
          </div>
        </div>
      </article>
    </div>
  `;

  el.plots.appendChild(section);
  const chartHost = section.querySelector(".combinedDynamicChart");
  const alphaSlider = section.querySelector("[data-alpha-slider]");
  const betaSlider = section.querySelector("[data-beta-slider]");
  const alphaValue = section.querySelector("[data-alpha-value]");
  const betaValue = section.querySelector("[data-beta-value]");
  const chart = createSliderChart(chartHost, result, "Interactive alpha-beta sweep");

  const update = () => {
    const alpha = Number(alphaSlider.value);
    const beta = Number(betaSlider.value);
    alphaValue.textContent = alpha.toFixed(2);
    betaValue.textContent = beta.toFixed(2);
    chart.update(calculateProbeRows(result, alpha, beta));
  };

  alphaSlider.addEventListener("input", update);
  betaSlider.addEventListener("input", update);
  update();
}

function renderAlphaSliderSweep(result) {
  const section = document.createElement("section");
  section.className = "plotSweep sliderSweep";
  section.innerHTML = `
    <div class="sliderSweepLayout">
      <article class="chartPanel sliderSweepPanel">
        <div class="chartTitle">Alpha-wise <span data-alpha-caption>alpha=1.00, beta=0</span></div>
        <div class="alphaDynamicChart"></div>
        <label class="sweepSliderControl">
          <span class="sliderEnds">
            <small>less predictive sharpness</small>
            <strong data-alpha-value>1.00</strong>
            <small>high predictive sharpness</small>
          </span>
          <input type="range" min="0.01" max="2" step="0.01" value="1" data-alpha-slider />
        </label>
      </article>
    </div>
  `;

  el.plots.appendChild(section);
  const chartHost = section.querySelector(".alphaDynamicChart");
  const slider = section.querySelector("[data-alpha-slider]");
  const value = section.querySelector("[data-alpha-value]");
  const caption = section.querySelector("[data-alpha-caption]");
  const chart = createAlphaSliderChart(chartHost, result);

  const update = () => {
    const alpha = Number(slider.value);
    value.textContent = alpha.toFixed(2);
    caption.textContent = `alpha=${alpha.toFixed(2)}, beta=0`;
    chart.update(alpha);
  };

  slider.addEventListener("input", update);
  update();
}

function renderBetaSliderSweep(result) {
  const section = document.createElement("section");
  section.className = "plotSweep sliderSweep";
  section.innerHTML = `
    <div class="sliderSweepLayout">
      <article class="chartPanel sliderSweepPanel">
        <div class="chartTitle">Beta-wise <span data-beta-caption>beta=0.00, alpha=1</span></div>
        <div class="betaDynamicChart"></div>
        <label class="sweepSliderControl">
          <span class="sliderEnds">
            <small>less popularity bias</small>
            <strong data-beta-value>0.00</strong>
            <small>high popularity bias</small>
          </span>
          <input type="range" min="0" max="1" step="0.01" value="0" data-beta-slider />
        </label>
      </article>
    </div>
  `;

  el.plots.appendChild(section);
  const chartHost = section.querySelector(".betaDynamicChart");
  const slider = section.querySelector("[data-beta-slider]");
  const value = section.querySelector("[data-beta-value]");
  const caption = section.querySelector("[data-beta-caption]");
  const chart = createSliderChart(chartHost, result, "Interactive beta sweep");

  const update = () => {
    const beta = Number(slider.value);
    value.textContent = beta.toFixed(2);
    caption.textContent = `beta=${beta.toFixed(2)}, alpha=1`;
    chart.update(calculateBetaRows(result, beta));
  };

  slider.addEventListener("input", update);
  update();
}

function createAlphaSliderChart(host, result) {
  const chart = createSliderChart(host, result, "Interactive alpha sweep");
  return {
    update(alpha) {
      chart.update(calculateAlphaRows(result, alpha));
    },
  };
}

function createSliderChart(host, result, ariaLabel) {
  const margin = { top: 26, right: 12, bottom: 58, left: 42 };
  const plotHeight = 170;
  const modelSlot = 54;
  const rows = result.models.map((model) => ({ ...model, color: modelColor(model) }));
  const width = Math.max(420, margin.left + margin.right + rows.length * modelSlot);
  const height = margin.top + plotHeight + margin.bottom;
  const plotWidth = width - margin.left - margin.right;
  const barWidth = Math.min(28, Math.max(18, modelSlot * 0.46));
  const baseY = margin.top + plotHeight;

  host.innerHTML = `
    <svg class="chartSvg sliderSweepSvg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(ariaLabel)}">
      <g data-grid></g>
      <line class="axisLine" x1="${margin.left}" y1="${baseY}" x2="${width - margin.right}" y2="${baseY}"></line>
      <line class="axisLine" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${baseY}"></line>
      <g data-bars></g>
    </svg>
  `;

  const svg = host.querySelector("svg");
  const grid = svg.querySelector("[data-grid]");
  const bars = svg.querySelector("[data-bars]");
  const groups = new Map();

  for (const row of rows) {
    const dynamicClass = ariaLabel.toLowerCase().includes("beta") ? "betaBarGroup" : "alphaBarGroup";
    const group = createSvgElement("g", { class: `barGroup ${dynamicClass}` });
    group.innerHTML = `
      <title></title>
      <rect class="chartBar ${row.source === "user" ? "userBar" : ""}" x="${-barWidth / 2}" y="${baseY}" width="${barWidth}" height="0" rx="5" ry="5" fill="${row.color}"></rect>
      <line class="errorLine sweepErrorLine" data-error-main x1="0" y1="${baseY}" x2="0" y2="${baseY}"></line>
      <line class="errorLine sweepErrorLine" data-error-top x1="-7" y1="${baseY}" x2="7" y2="${baseY}"></line>
      <line class="errorLine sweepErrorLine" data-error-bottom x1="-7" y1="${baseY}" x2="7" y2="${baseY}"></line>
      <text class="valueLabel" x="0" y="${margin.top}" text-anchor="middle">
        <tspan x="0" dy="0"></tspan>
        <tspan x="0" dy="11"></tspan>
      </text>
      <text class="xTick" x="0" y="${baseY + 16}" text-anchor="end" transform="rotate(-34 0 ${baseY + 16})">${escapeHtml(row.model)}</text>
    `;
    bars.appendChild(group);
    groups.set(row.id, group);
  }

  return {
    update(dynamicRows) {
      const maxValue = Math.max(...dynamicRows.map((row) => row.score + row.std), 1e-12);
      const minValue = Math.min(...dynamicRows.map((row) => Math.max(0, row.score - row.std)));
      const spread = Math.max(maxValue - minValue, maxValue * 0.08, 1e-6);
      let yMin = Math.max(0, minValue - spread * 0.35);
      if (yMin === 0 && minValue > 0) yMin = minValue * 0.65;
      const yMax = maxValue + spread * 0.2;
      const yRange = Math.max(yMax - yMin, 1e-9);
      const yScale = (score) => baseY - ((score - yMin) / yRange) * plotHeight;
      const ticks = makeTicks(yMin, yMax, 5);

      grid.innerHTML = ticks
        .map((tick) => {
          const y = yScale(tick);
          return `
            <line class="gridLine" x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}"></line>
            <text class="yTick" x="${margin.left - 8}" y="${y + 4}" text-anchor="end">${formatTick(tick)}</text>
          `;
        })
        .join("");

      dynamicRows.forEach((row, index) => {
        const centerX = margin.left + index * (plotWidth / dynamicRows.length) + plotWidth / dynamicRows.length / 2;
        const barHeight = Math.max(2, ((row.score - yMin) / yRange) * plotHeight);
        const y = baseY - barHeight;
        const group = groups.get(row.id);
        const scoreLabel = formatCompactScore(row.score);
        const stdLabel = formatCompactScore(row.std);
        const errTop = yScale(Math.min(yMax, row.score + row.std));
        const errBottom = yScale(Math.max(yMin, row.score - row.std));
        const showError = row.runCount > 1 && row.std > 0;

        group.style.transform = `translate(${centerX}px, 0px)`;
        group.querySelector("title").textContent = `${row.model}: ${scoreLabel} +/- ${stdLabel}`;
        group.querySelector("rect").setAttribute("y", y);
        group.querySelector("rect").setAttribute("height", barHeight);
        group.querySelector("[data-error-main]").setAttribute("y1", errTop);
        group.querySelector("[data-error-main]").setAttribute("y2", errBottom);
        group.querySelector("[data-error-top]").setAttribute("y1", errTop);
        group.querySelector("[data-error-top]").setAttribute("y2", errTop);
        group.querySelector("[data-error-bottom]").setAttribute("y1", errBottom);
        group.querySelector("[data-error-bottom]").setAttribute("y2", errBottom);
        for (const line of group.querySelectorAll(".sweepErrorLine")) {
          line.style.opacity = showError ? "1" : "0";
        }
        group.querySelector(".valueLabel").setAttribute("y", Math.max(12, yScale(Math.min(yMax, row.score + row.std)) - 20));
        group.querySelector(".valueLabel tspan:first-child").textContent = `${scoreLabel}\u00b1`;
        group.querySelector(".valueLabel tspan:last-child").textContent = stdLabel;
      });
    },
  };
}

function calculateAlphaRows(result, alpha) {
  return calculateProbeRows(result, alpha, 0);
}

function calculateBetaRows(result, beta) {
  return calculateProbeRows(result, 1, beta);
}

function calculateProbeRows(result, alpha, beta) {
  return result.models
    .map((model) => {
      const group = state.analysis.modelGroups.find((item) => item.id === model.id);
      const values = group.files.map((file) => calculateProbeMetric(file.info, state.analysis.dataset, alpha, beta));
      const stat = summarize(values);
      return {
        id: model.id,
        model: model.model,
        source: model.source,
        runCount: model.runCount,
        score: stat.mean,
        std: stat.std,
        color: modelColor(model),
      };
    })
    .sort((a, b) => b.score - a.score);
}

function renderAlphaShowcase(result) {
  const section = document.createElement("section");
  section.className = "showcase";
  section.innerHTML = `
    <div class="showcaseHead">
      <div>
        <h3>Predictive sharpness: rank histogram</h3>
      </div>
      <div class="modelPicker" data-picker="alpha"></div>
    </div>
    <div class="showcaseBody" data-body="alpha"></div>
  `;

  el.plots.appendChild(section);
  const picker = section.querySelector('[data-picker="alpha"]');
  const body = section.querySelector('[data-body="alpha"]');
  let histogramChart;
  renderModelPicker(picker, result.models, getDefaultModelIds(result.models), () => {
    const ids = getSelectedModelIds(picker);
    histogramChart.update(ids);
  });
  histogramChart = createRankClusterHistogram(body, result.models.map((model) => model.id));
  histogramChart.update(getSelectedModelIds(picker));
}

function renderBetaShowcase(result) {
  const section = document.createElement("section");
  section.className = "showcase";
  section.innerHTML = `
    <div class="showcaseHead">
      <div>
        <h3>Popularity-bias robustness: query examples</h3>
      </div>
      <div class="modelPicker" data-picker="beta"></div>
    </div>
    <div class="showcaseBody" data-body="beta"></div>
  `;

  el.plots.appendChild(section);
  const picker = section.querySelector('[data-picker="beta"]');
  const body = section.querySelector('[data-body="beta"]');
  let currentPayload = null;
  let page = 0;

  const renderPage = () => {
    body.innerHTML = renderBetaCaseExplorer(currentPayload, page);
    const nextButton = body.querySelector("[data-next-case]");
    if (nextButton) {
      nextButton.addEventListener("click", () => {
        page = (page + 1) % currentPayload.cases.length;
        renderPage();
      });
    }
    const popularitySvg = body.querySelector("[data-popularity-svg]");
    if (popularitySvg) {
      popularitySvg.addEventListener("click", (event) => {
        const nextPage = findClosestBetaCaseIndex(event, popularitySvg, currentPayload);
        if (nextPage !== -1) {
          page = nextPage;
          renderPage();
        }
      });
    }
    enableCaseGraphDrag(body);
  };

  const rebuild = () => {
    const ids = getSelectedModelIds(picker);
    if (ids.length < 2) {
      currentPayload = null;
      body.innerHTML = `<p class="emptyState">Select at least two models.</p>`;
      return;
    }
    currentPayload = buildBetaCasePayload(result, ids);
    page = 0;
    renderPage();
  };

  renderModelPicker(picker, result.models, getDefaultModelIds(result.models), () => {
    rebuild();
  });
  rebuild();
}

function renderModelPicker(container, models, selectedIds, onChange) {
  container.innerHTML = models
    .map((model) => {
      const checked = selectedIds.includes(model.id) ? "checked" : "";
      const label = `${model.model}${model.source === "user" ? " (user)" : ""}`;
      return `
        <label class="modelChoice">
          <input type="checkbox" value="${escapeHtml(model.id)}" ${checked} />
          <span class="colorDot" style="background:${modelColor(model)}"></span>
          ${escapeHtml(label)}
        </label>
      `;
    })
    .join("");

  for (const input of container.querySelectorAll("input")) {
    input.addEventListener("change", onChange);
  }
}

function getSelectedModelIds(container) {
  return [...container.querySelectorAll("input:checked")].map((input) => input.value);
}

function getDefaultModelIds(models) {
  return models.map((model) => model.id);
}

function renderRankClusterHistogram(selectedIds) {
  const bins = getRankClusterBins();
  const stats = getRankClusterStats(selectedIds, bins);
  if (!stats.length) return `<p class="emptyState">Select at least one model.</p>`;

  return renderRankClusterSvg(stats, bins);
}

function getRankClusterBins() {
  return [
    { lo: 1, hi: 1, label: "1" },
    { lo: 2, hi: 5, label: "2-5" },
    { lo: 6, hi: 20, label: "6-20" },
    { lo: 21, hi: 50, label: "21-50" },
    { lo: 51, hi: 100, label: "51-100" },
  ];
}

function getRankClusterStats(selectedIds, bins) {
  return getSelectedGroups(selectedIds).map((group) => {
    const runCounts = group.files.map((file) => countRanksByBin(file.info, bins));
    return {
      id: group.id,
      model: group.model,
      source: group.source,
      color: modelColor(group),
      bins: bins.map((_, index) => summarize(runCounts.map((counts) => counts[index]))),
    };
  });
}

function createRankClusterHistogram(host, modelIds) {
  const bins = getRankClusterBins();
  const allStats = getRankClusterStats(modelIds, bins);
  const statsById = new Map(allStats.map((model) => [model.id, model]));
  const width = 1120;
  const height = 360;
  const margin = { top: 22, right: 16, bottom: 80, left: 58 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const baseY = margin.top + plotHeight;

  host.innerHTML = `
    <div class="rankClusterFixedBox">
      <svg
        class="rankClusterFixedSvg"
        width="${width}"
        height="${height}"
        viewBox="0 0 ${width} ${height}"
        role="img"
        aria-label="Rank cluster histogram"
      >
        <line class="axisLine" x1="${margin.left}" y1="${baseY}" x2="${width - margin.right}" y2="${baseY}"></line>
        <line class="axisLine" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${baseY}"></line>
        <g data-rank-grid></g>
        <g data-rank-bars></g>
        <g data-rank-x-ticks></g>
        <text class="axisTitle" x="${-(margin.top + plotHeight / 2)}" y="14" text-anchor="middle" transform="rotate(-90)"># prediction</text>
        <text class="axisTitle" x="${margin.left + plotWidth / 2}" y="${baseY + 42}" text-anchor="middle">Rank</text>
        <g data-rank-legend></g>
        <text class="emptyStateSvg" data-rank-empty x="${margin.left + plotWidth / 2}" y="${margin.top + plotHeight / 2}" text-anchor="middle">Select at least one model.</text>
      </svg>
    </div>
  `;

  const svg = host.querySelector("svg");
  const grid = svg.querySelector("[data-rank-grid]");
  const bars = svg.querySelector("[data-rank-bars]");
  const xTicks = svg.querySelector("[data-rank-x-ticks]");
  const legend = svg.querySelector("[data-rank-legend]");
  const emptyText = svg.querySelector("[data-rank-empty]");
  const barGroups = new Map();
  const tickGroups = [];

  for (const [binIndex, bin] of bins.entries()) {
    const tick = createSvgElement("text", {
      class: "xTick rankXTick",
      x: 0,
      y: baseY + 20,
      "text-anchor": "middle",
    });
    tick.textContent = bin.label;
    xTicks.appendChild(tick);
    tickGroups.push(tick);

    for (const model of allStats) {
      const group = createSvgElement("g", { class: "clusterGroup rankBarGroup" });
      group.style.opacity = "0";
      group.innerHTML = `
        <title></title>
        <rect class="clusterBar" x="-4" y="${baseY}" width="8" height="0" rx="4" ry="4" fill="${model.color}"></rect>
        <line class="rankErrorLine" data-error-main x1="0" y1="${baseY}" x2="0" y2="${baseY}"></line>
        <line class="rankErrorLine" data-error-top x1="-4" y1="${baseY}" x2="4" y2="${baseY}"></line>
        <line class="rankErrorLine" data-error-bottom x1="-4" y1="${baseY}" x2="4" y2="${baseY}"></line>
        <text class="hoverLabel" x="0" y="${baseY}" text-anchor="middle"></text>
      `;
      bars.appendChild(group);
      barGroups.set(`${model.id}:${binIndex}`, group);
    }
  }

  legend.innerHTML = allStats
    .map((model, index) => {
      const legendY = height - 20;
      const legendX = margin.left + index * 115;
      return `
        <rect x="${legendX}" y="${legendY - 10}" width="10" height="10" rx="2" fill="${model.color}"></rect>
        <text class="legendText" x="${legendX + 14}" y="${legendY - 1}">${escapeHtml(model.model)}</text>
      `;
    })
    .join("");

  const hideGroup = (group) => {
    group.style.opacity = "0";
    group.style.pointerEvents = "none";
    group.querySelector("rect").setAttribute("height", 0);
    group.querySelector("rect").setAttribute("y", baseY);
    for (const line of group.querySelectorAll(".rankErrorLine")) {
      line.style.opacity = "0";
    }
  };

  return {
    update(selectedIds) {
      const selected = selectedIds.map((id) => statsById.get(id)).filter(Boolean);
      emptyText.style.opacity = selected.length ? "0" : "1";
      if (!selected.length) {
        for (const group of barGroups.values()) hideGroup(group);
        return;
      }

      const visibleValues = selected.flatMap((model) =>
        model.bins.flatMap((bin) => [
          bin.mean,
          bin.mean + bin.std,
          Math.max(1, bin.mean - bin.std),
        ])
      );
      const positiveValues = visibleValues.filter((value) => value > 0);
      const minValue = Math.max(1, Math.min(...(positiveValues.length ? positiveValues : [1])));
      const maxValue = Math.max(...visibleValues, 1);
      const logPad = Math.max(
        0.08,
        (Math.log10(maxValue) - Math.log10(minValue)) * 0.12
      );
      const yMin = Math.pow(10, Math.max(0, Math.log10(minValue) - logPad));
      const yMax = Math.pow(10, Math.log10(maxValue) + logPad);
      const logRange = Math.max(1e-9, Math.log10(yMax) - Math.log10(yMin));
      const yScale = (value) => {
        const safe = Math.max(yMin, value);
        return (
          margin.top +
          plotHeight -
          ((Math.log10(safe) - Math.log10(yMin)) / logRange) * plotHeight
        );
      };
      const ticks = logTicks(yMin, yMax);
      grid.innerHTML = ticks
        .map((tick) => {
          const y = yScale(tick);
          return `
            <line class="gridLine" x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}"></line>
            <text class="yTick" x="${margin.left - 8}" y="${y + 4}" text-anchor="end">${formatCountTick(tick)}</text>
          `;
        })
        .join("");

      const modelCount = Math.max(1, selected.length);
      const barGap = modelCount > 3 ? 5 : 7;
      const barWidth = Math.max(8, Math.min(modelCount <= 2 ? 28 : 24, 34 - modelCount * 2));
      const totalBarWidth = modelCount * barWidth + Math.max(0, modelCount - 1) * barGap;
      const desiredBinStep = Math.max(92, totalBarWidth + Math.max(38, 76 - modelCount * 3));
      const activePlotWidth = Math.min(plotWidth, desiredBinStep * bins.length);
      const binStep = activePlotWidth / bins.length;
      const activeStartX = margin.left + (plotWidth - activePlotWidth) / 2;
      const errorCapWidth = Math.max(8, Math.min(16, barWidth * 0.82));
      const activeKeys = new Set();

      bins.forEach((bin, binIndex) => {
        const center = activeStartX + binIndex * binStep + binStep / 2;
        tickGroups[binIndex].style.transform = `translate(${center}px, 0px)`;
        const startX = center - totalBarWidth / 2;

        selected.forEach((model, modelIndex) => {
          const group = barGroups.get(`${model.id}:${binIndex}`);
          const binStat = model.bins[binIndex];
          const value = Math.max(yMin, binStat.mean);
          const errorTopValue = Math.max(yMin, binStat.mean + binStat.std);
          const errorBottomValue = Math.max(yMin, binStat.mean - binStat.std);
          const topY = yScale(value);
          const errorTopY = yScale(errorTopValue);
          const errorBottomY = yScale(errorBottomValue);
          const centerX = startX + modelIndex * (barWidth + barGap) + barWidth / 2;
          const predictionLabel = `# ${formatCount(binStat.mean)} \u00b1 ${formatCount(binStat.std)}`;
          const showError = binStat.std > 0;
          activeKeys.add(`${model.id}:${binIndex}`);

          group.style.opacity = "1";
          group.style.pointerEvents = "auto";
          group.style.transform = `translate(${centerX}px, 0px)`;
          group.querySelector("title").textContent = `${model.model}: ${predictionLabel}`;
          group.querySelector("rect").setAttribute("x", -barWidth / 2);
          group.querySelector("rect").setAttribute("y", topY);
          group.querySelector("rect").setAttribute("width", barWidth);
          group.querySelector("rect").setAttribute("height", baseY - topY);
          group.querySelector("[data-error-main]").setAttribute("y1", errorTopY);
          group.querySelector("[data-error-main]").setAttribute("y2", errorBottomY);
          group.querySelector("[data-error-top]").setAttribute("x1", -errorCapWidth / 2);
          group.querySelector("[data-error-top]").setAttribute("x2", errorCapWidth / 2);
          group.querySelector("[data-error-top]").setAttribute("y1", errorTopY);
          group.querySelector("[data-error-top]").setAttribute("y2", errorTopY);
          group.querySelector("[data-error-bottom]").setAttribute("x1", -errorCapWidth / 2);
          group.querySelector("[data-error-bottom]").setAttribute("x2", errorCapWidth / 2);
          group.querySelector("[data-error-bottom]").setAttribute("y1", errorBottomY);
          group.querySelector("[data-error-bottom]").setAttribute("y2", errorBottomY);
          for (const line of group.querySelectorAll(".rankErrorLine")) {
            line.style.opacity = showError ? "1" : "0";
          }
          const label = group.querySelector(".hoverLabel");
          label.setAttribute("y", Math.min(topY, errorTopY) - 8);
          label.textContent = predictionLabel;
        });
      });

      for (const [key, group] of barGroups.entries()) {
        if (!activeKeys.has(key)) hideGroup(group);
      }
    },
  };
}

function countRanksByBin(info, bins) {
  const counts = Array.from({ length: bins.length }, () => 0);
  for (const row of info) {
    const rank = Number(row[2]);
    for (let index = 0; index < bins.length; index += 1) {
      if (rank >= bins[index].lo && rank <= bins[index].hi) {
        counts[index] += 1;
        break;
      }
    }
  }
  return counts;
}

function renderRankClusterSvg(stats, bins) {
  const FIGURE_WIDTH = 1120;
  const FIGURE_HEIGHT = 360;
  const margin = { top: 22, right: 16, bottom: 80, left: 58 };
  const modelCount = Math.max(1, stats.length);
  const barGap = modelCount > 3 ? 5 : 7;
  const barWidth = Math.max(8, Math.min(modelCount <= 2 ? 28 : 24, 34 - modelCount * 2));
  const totalBarWidth = modelCount * barWidth + Math.max(0, modelCount - 1) * barGap;

  const width = FIGURE_WIDTH;
  const height = FIGURE_HEIGHT;

  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const baseY = margin.top + plotHeight;
  const desiredBinStep = Math.max(92, totalBarWidth + Math.max(38, 76 - modelCount * 3));
  const activePlotWidth = Math.min(plotWidth, desiredBinStep * bins.length);
  const binStep = activePlotWidth / bins.length;
  const activeStartX = margin.left + (plotWidth - activePlotWidth) / 2;

  const visibleValues = stats.flatMap((model) =>
    model.bins.flatMap((bin) => [
      bin.mean,
      bin.mean + bin.std,
      Math.max(1, bin.mean - bin.std),
    ])
  );

  const positiveValues = visibleValues.filter((value) => value > 0);
  const minValue = Math.max(1, Math.min(...(positiveValues.length ? positiveValues : [1])));
  const maxValue = Math.max(...visibleValues, 1);

  const logPad = Math.max(
    0.08,
    (Math.log10(maxValue) - Math.log10(minValue)) * 0.12
  );

  const yMin = Math.pow(10, Math.max(0, Math.log10(minValue) - logPad));
  const yMax = Math.pow(10, Math.log10(maxValue) + logPad);
  const logRange = Math.max(1e-9, Math.log10(yMax) - Math.log10(yMin));

  const ticks = logTicks(yMin, yMax);

  const yScale = (value) => {
    const safe = Math.max(yMin, value);
    return (
      margin.top +
      plotHeight -
      ((Math.log10(safe) - Math.log10(yMin)) / logRange) * plotHeight
    );
  };

  const errorCapWidth = Math.max(8, Math.min(16, barWidth * 0.82));

  const grid = ticks
    .map((tick) => {
      const y = yScale(tick);
      return `
        <line class="gridLine" x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}"></line>
        <text class="yTick" x="${margin.left - 8}" y="${y + 4}" text-anchor="end">${formatCountTick(tick)}</text>
      `;
    })
    .join("");

  const bars = bins
    .map((bin, binIndex) => {
      const groupX = activeStartX + binIndex * binStep;
      const center = groupX + binStep / 2;

      const startX = center - totalBarWidth / 2;

      const binBars = stats
        .map((model, modelIndex) => {
          const binStat = model.bins[binIndex];
          const value = Math.max(yMin, binStat.mean);
          const errorTopValue = Math.max(yMin, binStat.mean + binStat.std);
          const errorBottomValue = Math.max(yMin, binStat.mean - binStat.std);
          const topY = yScale(value);
          const errorTopY = yScale(errorTopValue);
          const errorBottomY = yScale(errorBottomValue);
          const x = startX + modelIndex * (barWidth + barGap);
          const midX = x + barWidth / 2;
          const predictionLabel = `# ${formatCount(binStat.mean)} \u00b1 ${formatCount(binStat.std)}`;
          const errorLines = binStat.std > 0
            ? `
              <line class="rankErrorLine" x1="${midX}" y1="${errorTopY}" x2="${midX}" y2="${errorBottomY}"></line>
              <line class="rankErrorLine" x1="${midX - errorCapWidth / 2}" y1="${errorTopY}" x2="${midX + errorCapWidth / 2}" y2="${errorTopY}"></line>
              <line class="rankErrorLine" x1="${midX - errorCapWidth / 2}" y1="${errorBottomY}" x2="${midX + errorCapWidth / 2}" y2="${errorBottomY}"></line>
            `
            : "";

          return `
            <g class="clusterGroup">
              <title>${escapeHtml(predictionLabel)}</title>
              <rect
                class="clusterBar"
                x="${x}"
                y="${topY}"
                width="${barWidth}"
                height="${baseY - topY}"
                rx="4"
                ry="4"
                fill="${model.color}"
              ></rect>
              ${errorLines}
              <text 
                class="hoverLabel" 
                x="${midX}" 
                y="${Math.min(topY, errorTopY) - 8}" 
                text-anchor="middle"
              >${escapeHtml(predictionLabel)}</text>
            </g>
          `;
        })
        .join("");

      return `
        ${binBars}
        <text class="xTick" x="${center}" y="${baseY + 20}" text-anchor="middle">${escapeHtml(bin.label)}</text>
      `;
    })
    .join("");

  const legend = stats
    .map((model, index) => {
      const legendY = height - 20;
      const legendX = margin.left + index * 115;

      return `
        <rect x="${legendX}" y="${legendY - 10}" width="10" height="10" rx="2" fill="${model.color}"></rect>
        <text class="legendText" x="${legendX + 14}" y="${legendY - 1}">${escapeHtml(model.model)}</text>
      `;
    })
    .join("");

  return `
    <div class="rankClusterFixedBox">
      <svg
        class="rankClusterFixedSvg"
        width="${width}"
        height="${height}"
        viewBox="0 0 ${width} ${height}"
        role="img"
        aria-label="Rank cluster histogram"
      >
        <line class="axisLine" x1="${margin.left}" y1="${baseY}" x2="${width - margin.right}" y2="${baseY}"></line>
        <line class="axisLine" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${baseY}"></line>

        ${grid}
        ${bars}

        <text class="axisTitle" x="${-(margin.top + plotHeight / 2)}" y="14" text-anchor="middle" transform="rotate(-90)"># prediction</text>
        <text class="axisTitle" x="${margin.left + plotWidth / 2}" y="${baseY + 42}" text-anchor="middle">Rank</text>

        ${legend}
      </svg>
    </div>
  `;
}

function buildBetaCasePayload(result, selectedIds) {
  const selectedModels = result.models.filter((model) => selectedIds.includes(model.id));
  if (selectedModels.length < 2) {
    return { error: "Select at least two models.", cases: [] };
  }

  const groups = getSelectedGroups(selectedIds);
  const rankMaps = Object.fromEntries(groups.map((group) => [group.id, buildMeanRankMap(group)]));
  const referenceRanks = rankMaps[groups[0].id];
  const cases = [];

  for (const [key, row] of referenceRanks.entries()) {
    if (!groups.every((group) => rankMaps[group.id].has(key))) continue;
    const topPopularity = state.analysis.dataset.popularityPercentiles[row.target] ?? 100;
    cases.push({
      key,
      row,
      topPopularity,
      ranks: Object.fromEntries(groups.map((group) => [group.id, rankMaps[group.id].get(key)?.rank ?? null])),
    });
  }

  cases.sort((a, b) => a.row.degree - b.row.degree || (state.analysis.dataset.popularityPositions?.[a.row.target] ?? 0) - (state.analysis.dataset.popularityPositions?.[b.row.target] ?? 0));
  if (!cases.length) {
    return {
      error: "No aligned test cases were found for the selected models.",
      cases: [],
    };
  }

  return {
    groups,
    cases,
  };
}

function renderBetaCaseExplorer(payload, page) {
  if (!payload || payload.error) {
    return `<p class="emptyState">${escapeHtml(payload?.error || "Select at least two models.")}</p>`;
  }
  const item = payload.cases[page % payload.cases.length];
  const dataset = state.analysis.dataset;
  const target = item.row.target;
  const query = item.row.query;
  const targetName = formatEntity(target, dataset);
  const relationName = formatRelation(query[1], dataset);

  return `
    <div class="caseIntro">
      Prediction case ${page + 1} / ${payload.cases.length}. Click the popularity plot to select the query to inspect.
    </div>
    <div class="betaCaseExplorer">
      <section class="popularityPanel">
        ${renderPopularityCurveSvg(target, dataset)}
        <button class="caseNavButton" type="button" data-next-case>Next case</button>
      </section>
      <section class="caseGraphPanel">
        ${renderCaseSubgraphSvg(item, dataset)}
      </section>
      <section class="rankStackPanel">
        <div class="caseMeta">
          <strong>${escapeHtml(targetName)}</strong>
          <span>degree ${item.row.degree} · top ${formatCompactScore(item.topPopularity)}%</span>
        </div>
        ${renderRankStack(payload.groups, item)}
      </section>
    </div>
  `;
}

function renderPopularityCurveSvg(targetId, dataset) {
  const width = 280;
  const height = 360;
  const margin = { top: 18, right: 12, bottom: 44, left: 58 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const order = dataset.popularityOrder || [];
  const targetPosition = dataset.popularityPositions?.[targetId] || 1;
  const targetDegree = dataset.counts[targetId] ?? 0;
  const maxDegree = Math.max(...order.map((item) => Number(item.degree)), 1);
  const xLogMax = Math.max(1e-9, Math.log10(maxDegree));
  const xScale = (degree) => margin.left + (Math.log10(Math.max(1, Number(degree))) / xLogMax) * plotWidth;
  const yScale = (position) => margin.top + ((position - 1) / Math.max(1, order.length - 1)) * plotHeight;
  const sampleStep = Math.max(1, Math.ceil(order.length / 500));
  const points = order
    .filter((_, index) => index % sampleStep === 0)
    .map((item) => `${xScale(item.degree)},${yScale(item.position)}`)
    .join(" ");
  const targetX = xScale(targetDegree);
  const targetY = yScale(targetPosition);
  const degreeGuides = powerOfTenTicks(maxDegree)
    .map((degree) => {
      const x = xScale(degree);
      return `
        <line class="degreeGuideLine" x1="${x}" y1="${margin.top}" x2="${x}" y2="${margin.top + plotHeight}"></line>
        <text class="degreeGuideLabel" x="${x}" y="${height - 24}" text-anchor="middle">${formatCountTick(degree)}</text>
      `;
    })
    .join("");

  return `
    <svg class="popularitySvg" data-popularity-svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Popularity sorted degree curve">
      <line class="axisLine" x1="${margin.left}" y1="${margin.top + plotHeight}" x2="${width - margin.right}" y2="${margin.top + plotHeight}"></line>
      <line class="axisLine" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotHeight}"></line>
      ${degreeGuides}
      <polyline class="popularityLine" points="${points}"></polyline>
      <circle class="targetGlowPoint" cx="${targetX}" cy="${targetY}" r="8"></circle>
      <circle class="targetPoint" cx="${targetX}" cy="${targetY}" r="4"></circle>
      <text class="axisTitle" x="${margin.left + plotWidth / 2}" y="${height - 6}" text-anchor="middle">degree</text>
      <text class="axisTitle" x="${-(margin.top + plotHeight / 2)}" y="14" text-anchor="middle" transform="rotate(-90)">sorted entity position</text>
    </svg>
  `;
}

function findClosestBetaCaseIndex(event, svg, payload) {
  if (!payload?.cases?.length) return -1;
  const dataset = state.analysis.dataset;
  const order = dataset.popularityOrder || [];
  if (!order.length) return -1;

  const width = 280;
  const height = 360;
  const margin = { top: 18, right: 12, bottom: 44, left: 58 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const rect = svg.getBoundingClientRect();
  const svgX = ((event.clientX - rect.left) / Math.max(1, rect.width)) * width;
  const svgY = ((event.clientY - rect.top) / Math.max(1, rect.height)) * height;
  const xRatio = clamp((svgX - margin.left) / plotWidth, 0, 1);
  const yRatio = clamp((svgY - margin.top) / plotHeight, 0, 1);
  const clickedPosition = 1 + yRatio * Math.max(1, order.length - 1);
  const maxDegree = Math.max(...order.map((item) => Number(item.degree)), 1);
  const clickedLogDegree = xRatio * Math.max(1e-9, Math.log10(maxDegree));

  let bestIndex = -1;
  let bestDistance = Infinity;
  payload.cases.forEach((item, index) => {
    const target = item.row.target;
    const position = dataset.popularityPositions?.[target] || 1;
    const degree = dataset.counts[target] ?? 0;
    const positionDistance = Math.abs(position - clickedPosition) / Math.max(1, order.length);
    const degreeDistance = Math.abs(Math.log10(Math.max(1, degree)) - clickedLogDegree) / Math.max(1e-9, Math.log10(maxDegree));
    const distance = degreeDistance * 0.75 + positionDistance * 0.25;
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });

  return bestIndex;
}

function renderCaseSubgraphSvg(item, dataset) {
  const [head, relation, tail] = item.row.query;
  const target = item.row.target;
  const rawEdges = getOneHopEdges(item.row.query, dataset);
  const testEdge = { h: head, r: relation, t: tail, isTest: true };
  const displayEdges = [testEdge, ...rawEdges.slice(0, 20)];
  const nodes = [...new Set(displayEdges.flatMap((edge) => [edge.h, edge.t]))];
  if (!nodes.includes(head)) nodes.push(head);
  if (!nodes.includes(tail)) nodes.push(tail);

  const width = 520;
  const height = 320;
  const center = { x: width / 2, y: height / 2 };
  const neighborCount = Math.max(1, nodes.length - 2);
  const edgeOpacity = clamp(22 / Math.sqrt(Math.max(1, rawEdges.length)), 0.045, 0.34);
  const nodeOpacity = clamp(18 / Math.sqrt(neighborCount), 0.08, 0.72);
  const neighborRadius = clamp(42 / Math.sqrt(neighborCount), 2.2, 7);
  const positions = new Map();
  positions.set(head, { x: width * 0.34, y: height * 0.5 });
  positions.set(tail, { x: width * 0.66, y: height * 0.5 });
  const others = nodes.filter((node) => node !== head && node !== tail);
  others.forEach((node, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(1, others.length) - Math.PI / 2;
    const ring = Math.floor(index / 80);
    const radius = Math.max(76, 126 - ring * 12);
    const radiusX = radius * 1.14;
    const radiusY = radius;
    positions.set(node, {
      x: center.x + Math.cos(angle) * radiusX,
      y: center.y + Math.sin(angle) * radiusY,
    });
  });

  const edgeMarkup = displayEdges
    .map((edge) => {
      const from = positions.get(edge.h);
      const to = positions.get(edge.t);
      const isTest = edge.isTest || (edge.h === head && edge.t === tail && String(edge.r) === String(relation));
      return `
        <line class="graphEdge ${isTest ? "testEdge" : ""}" data-h="${edge.h}" data-t="${edge.t}" x1="${from.x}" y1="${from.y}" x2="${to.x}" y2="${to.y}" style="opacity:${isTest ? 1 : edgeOpacity}"></line>
      `;
    })
    .join("");

  const nodeMarkup = nodes
    .map((node) => {
      const pos = positions.get(node);
      const isTarget = node === target;
      const isNamed = node === head || node === tail || node === target;
      const isQueryNode = node === head || node === tail || node === target;
      const label = isNamed ? formatEntity(node, dataset) : "";
      return `
        <g class="graphNodeGroup ${isQueryNode ? "draggableQueryNode" : ""}" data-node="${node}" data-x="${pos.x}" data-y="${pos.y}" transform="translate(${pos.x} ${pos.y})">
          ${isTarget ? `<circle class="targetNodeGlow" cx="0" cy="0" r="20"></circle>` : ""}
          <circle class="graphNode ${isTarget ? "targetNode" : ""}" cx="0" cy="0" r="${isTarget ? 12 : neighborRadius}" style="opacity:${isQueryNode ? 1 : nodeOpacity}"></circle>
          ${label ? `<text class="graphLabel" x="0" y="-18" text-anchor="middle">${escapeHtml(truncateText(label, 24))}</text>` : ""}
        </g>
      `;
    })
    .join("");
  const hiddenMarkup = renderHiddenEdgeMarkers({
    hiddenEdges: rawEdges.slice(20),
    displayEdges,
    positions,
    anchors: [head, tail],
    width,
    height,
  });

  return `
    <svg class="caseGraphSvg" viewBox="0 0 ${width} ${height}" role="img" aria-label="One hop subgraph">
      ${edgeMarkup}
      ${hiddenMarkup}
      ${nodeMarkup}
      <text class="graphRelation" data-relation-label data-h="${head}" data-t="${tail}" x="${(positions.get(head).x + positions.get(tail).x) / 2}" y="${(positions.get(head).y + positions.get(tail).y) / 2 - 14}" text-anchor="middle">${escapeHtml(truncateText(formatRelation(relation, dataset), 30))}</text>
    </svg>
  `;
}

function renderHiddenEdgeMarkers({ hiddenEdges, displayEdges, positions, anchors, width, height }) {
  if (!hiddenEdges.length) return "";

  return anchors
    .map((anchor) => {
      const hiddenCount = hiddenEdges.filter((edge) => edge.h === anchor || edge.t === anchor).length;
      if (!hiddenCount) return "";

      const anchorPos = positions.get(anchor);
      if (!anchorPos) return "";

      const incidentAngles = displayEdges
        .filter((edge) => edge.h === anchor || edge.t === anchor)
        .map((edge) => {
          const other = edge.h === anchor ? edge.t : edge.h;
          const otherPos = positions.get(other);
          return otherPos ? Math.atan2(otherPos.y - anchorPos.y, otherPos.x - anchorPos.x) : null;
        })
        .filter((angle) => angle != null)
        .sort((a, b) => a - b);

      const angle = midpointOfLargestAngleGap(incidentAngles, anchorPos.x < width / 2 ? Math.PI : 0);
      const distance = 48;
      const x = clamp(anchorPos.x + Math.cos(angle) * distance, 28, width - 28);
      const y = clamp(anchorPos.y + Math.sin(angle) * distance, 28, height - 28);

      return `
        <text class="graphMoreNode" x="${x}" y="${y}" text-anchor="middle">
          <title>${hiddenCount} more 1-hop edges</title>
          ...
        </text>
      `;
    })
    .join("");
}

function midpointOfLargestAngleGap(angles, fallback) {
  if (!angles.length) return fallback;
  if (angles.length === 1) return angles[0] + Math.PI / 2;

  const full = Math.PI * 2;
  let bestGap = -Infinity;
  let bestStart = angles[0];

  for (let index = 0; index < angles.length; index += 1) {
    const current = angles[index];
    const next = index === angles.length - 1 ? angles[0] + full : angles[index + 1];
    const gap = next - current;
    if (gap > bestGap) {
      bestGap = gap;
      bestStart = current;
    }
  }

  return bestStart + bestGap / 2;
}

function getOneHopEdges(query, dataset) {
  const [head, relation, tail] = query;
  const related = [];
  for (const [h, r, t] of dataset.trainTriples || []) {
    if (h === head || t === head || h === tail || t === tail) {
      related.push({ h, r, t });
    }
  }
  related.unshift({ h: head, r: relation, t: tail });
  return related;
}

function enableCaseGraphDrag(root) {
  const svg = root.querySelector(".caseGraphSvg");
  if (!svg) return;

  const nodeGroups = [...svg.querySelectorAll(".draggableQueryNode")];
  const lines = [...svg.querySelectorAll(".graphEdge")];
  const relationLabel = svg.querySelector("[data-relation-label]");
  let active = null;

  const svgPoint = (event) => {
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    return point.matrixTransform(svg.getScreenCTM().inverse());
  };

  const setNodePosition = (group, x, y) => {
    group.dataset.x = x;
    group.dataset.y = y;
    group.setAttribute("transform", `translate(${x} ${y})`);
    const nodeId = group.dataset.node;
    for (const line of lines) {
      if (line.dataset.h === nodeId) {
        line.setAttribute("x1", x);
        line.setAttribute("y1", y);
      }
      if (line.dataset.t === nodeId) {
        line.setAttribute("x2", x);
        line.setAttribute("y2", y);
      }
    }
    updateRelationLabel();
  };

  const updateRelationLabel = () => {
    if (!relationLabel) return;
    const headGroup = svg.querySelector(`.draggableQueryNode[data-node="${relationLabel.dataset.h}"]`);
    const tailGroup = svg.querySelector(`.draggableQueryNode[data-node="${relationLabel.dataset.t}"]`);
    if (!headGroup || !tailGroup) return;
    const hx = Number(headGroup.dataset.x);
    const hy = Number(headGroup.dataset.y);
    const tx = Number(tailGroup.dataset.x);
    const ty = Number(tailGroup.dataset.y);
    relationLabel.setAttribute("x", (hx + tx) / 2);
    relationLabel.setAttribute("y", (hy + ty) / 2 - 14);
  };

  for (const group of nodeGroups) {
    group.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      const point = svgPoint(event);
      active = {
        group,
        offsetX: point.x - Number(group.dataset.x),
        offsetY: point.y - Number(group.dataset.y),
      };
      group.classList.add("dragging");
      group.setPointerCapture(event.pointerId);
    });

    group.addEventListener("pointermove", (event) => {
      if (!active || active.group !== group) return;
      const point = svgPoint(event);
      setNodePosition(group, clamp(point.x - active.offsetX, 18, 502), clamp(point.y - active.offsetY, 22, 298));
    });

    const stop = (event) => {
      if (!active || active.group !== group) return;
      group.classList.remove("dragging");
      try {
        group.releasePointerCapture(event.pointerId);
      } catch {
        // The pointer may already be released by the browser.
      }
      active = null;
    };

    group.addEventListener("pointerup", stop);
    group.addEventListener("pointercancel", stop);
  }
}

function renderRankStack(groups, item) {
  const rows = groups
    .map((group) => ({
      group,
      rank: item.ranks[group.id],
    }))
    .sort((a, b) => (a.rank ?? Infinity) - (b.rank ?? Infinity));

  return `
    <div class="rankStack">
      ${rows
        .map(({ group, rank }, index) => `
          <div class="rankItem ${index === 0 ? "bestRankItem" : ""}">
            <span class="colorDot" style="background:${modelColor(group)}"></span>
            <span>${escapeHtml(group.model)}</span>
            <strong>${rank == null ? "-" : formatRank(rank)}</strong>
          </div>
        `)
        .join("")}
    </div>
  `;
}

function renderHyperPlaneSection() {
  const section = document.createElement("section");
  section.className = "showcase heatSection";
  section.innerHTML = `
    <div class="showcaseHead">
      <div>
        <h3>Dual perspective landscape</h3>
      </div>
      <div class="heatControls">
        <label>
          Alphas
          <input type="text" data-heat-alphas value="1, 0.5, 0.25" />
        </label>
        <label>
          Betas
          <input type="text" data-heat-betas value="0, 0.4, 0.8" />
        </label>
        <button type="button" data-heat-draw>Draw grid</button>
      </div>
    </div>
    <div class="showcaseBody" data-heat-body></div>
  `;

  el.plots.appendChild(section);
  const alphaInput = section.querySelector("[data-heat-alphas]");
  const betaInput = section.querySelector("[data-heat-betas]");
  const drawButton = section.querySelector("[data-heat-draw]");
  const body = section.querySelector("[data-heat-body]");

  const draw = () => {
    try {
      const alphas = parseNumberList(alphaInput.value, "alphas");
      const betas = parseNumberList(betaInput.value, "betas");
      if (alphas.some((alpha) => alpha <= 0)) throw new Error("alpha values must be greater than zero.");
      body.innerHTML = renderHyperPlaneGrid(alphas, betas);
    } catch (error) {
      body.innerHTML = `<p class="emptyState">${escapeHtml(error.message)}</p>`;
    }
  };

  drawButton.addEventListener("click", draw);
  draw();
}

function render3DHyperPlaneSection() {
  const section = document.createElement("section");
  section.className = "showcase hyper3dSection";
  section.innerHTML = `
    <div class="showcaseHead">
      <div>
        <h3>Dual perspective evaluation landscape</h3>
      </div>
      <div class="heatControls">
        <label>
          Mesh
          <input type="number" data-3d-resolution min="7" max="21" step="2" value="13" />
        </label>
        <button type="button" data-3d-draw>Draw 3D</button>
      </div>
    </div>
    <div class="showcaseBody" data-3d-body>
      <p class="emptyState">Preparing 3D view...</p>
    </div>
  `;

  el.plots.appendChild(section);
  const resolutionInput = section.querySelector("[data-3d-resolution]");
  const drawButton = section.querySelector("[data-3d-draw]");
  const body = section.querySelector("[data-3d-body]");

  const draw = () => {
    const resolution = clamp(Math.round(Number(resolutionInput.value) || 13), 7, 21);
    body.innerHTML = `<p class="emptyState">Computing dense normalized surfaces...</p>`;
    setTimeout(() => {
      try {
        const alphas = linspace(0.01, 2, resolution);
        const betas = linspace(0, 1, resolution);
        const hyperData = computeHyperPlaneData(alphas, betas);
        if (hyperData.error) throw new Error(hyperData.error);
        body.innerHTML = render3DHyperPlaneShell(hyperData);
        const canvas = body.querySelector("[data-3d-canvas]");
        init3DHyperPlaneViewer(canvas, hyperData, alphas, betas);
      } catch (error) {
        body.innerHTML = `<p class="emptyState">${escapeHtml(error.message)}</p>`;
      }
    }, 20);
  };

  drawButton.addEventListener("click", draw);
  draw();
}

function render3DHyperPlaneShell(hyperData) {
  const legend = hyperData.groups
    .map((group) => `
      <label class="hyper3dLegendItem">
        <input type="checkbox" data-surface-toggle="${escapeHtml(group.id)}" checked />
        <span class="colorDot" style="background:${modelColor(group)}"></span>
        ${escapeHtml(group.model)}
      </label>
    `)
    .join("");
  const floorControls = get3DFloorSections()
    .map((section) => `
      <label class="floorSectionChoice">
        <input type="checkbox" data-floor-section="${escapeHtml(section.id)}" checked />
        <span>${section.shortLabel.map((line) => escapeHtml(line)).join("<br>")}</span>
      </label>
    `)
    .join("");

  return `
    <div class="hyper3dPanel">
      <div class="hyper3dStage">
        <div class="hyper3dControlGroup hyper3dFloorControls">
          <span>Evaluation focus</span>
          <div class="floorSectionPicker">${floorControls}</div>
        </div>
        <canvas class="hyper3dCanvas" data-3d-canvas width="960" height="560"></canvas>
        <div class="hyper3dControlGroup hyper3dModelControls">
          <span>Models</span>
          <div class="hyper3dLegend">${legend}</div>
        </div>
      </div>
      <div class="hyper3dHint">Drag to rotate · wheel to zoom</div>
    </div>
  `;
}

function init3DHyperPlaneViewer(canvas, hyperData, alphas, betas) {
  const ctx = canvas.getContext("2d");
  const surfaces = hyperData.groups.map((group) => ({
    id: group.id,
    model: group.model,
    color: modelColor(group),
    z: hyperData.normalizedScores.get(group.id),
    visible: true,
  }));
  const state3d = {
    rotX: 0.72,
    rotZ: -0.7,
    zoom: 1,
    dragging: false,
    lastX: 0,
    lastY: 0,
    floorSections: new Set(get3DFloorSections().map((section) => section.id)),
  };

  const draw = () => draw3DHyperPlaneCanvas(ctx, canvas, surfaces, alphas, betas, state3d);
  const panel = canvas.closest(".hyper3dPanel");
  const updateFloorSections = () => {
    const selected = [...panel.querySelectorAll("[data-floor-section]:checked")].map((input) => input.dataset.floorSection);
    state3d.floorSections = new Set(selected);
  };
  for (const checkbox of panel.querySelectorAll("[data-floor-section]")) {
    checkbox.addEventListener("change", () => {
      updateFloorSections();
      draw();
    });
  }
  for (const toggle of panel.querySelectorAll("[data-surface-toggle]")) {
    toggle.addEventListener("change", () => {
      const surface = surfaces.find((item) => item.id === toggle.dataset.surfaceToggle);
      if (surface) surface.visible = toggle.checked;
      draw();
    });
  }

  canvas.addEventListener("pointerdown", (event) => {
    state3d.dragging = true;
    state3d.lastX = event.clientX;
    state3d.lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!state3d.dragging) return;
    const dx = event.clientX - state3d.lastX;
    const dy = event.clientY - state3d.lastY;
    state3d.rotZ += dx * 0.01;
    state3d.rotX = clamp(state3d.rotX + dy * 0.008, 0.18, 1.35);
    state3d.lastX = event.clientX;
    state3d.lastY = event.clientY;
    draw();
  });

  const stop = (event) => {
    state3d.dragging = false;
    try {
      canvas.releasePointerCapture(event.pointerId);
    } catch {
      // Pointer may already be released.
    }
  };
  canvas.addEventListener("pointerup", stop);
  canvas.addEventListener("pointercancel", stop);
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    state3d.zoom = clamp(state3d.zoom * (event.deltaY > 0 ? 0.92 : 1.08), 0.62, 2.2);
    draw();
  }, { passive: false });

  draw();
}

function draw3DHyperPlaneCanvas(ctx, canvas, surfaces, alphas, betas, state3d) {
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const alphaMin = Math.min(...alphas);
  const alphaMax = Math.max(...alphas);
  const betaMin = Math.min(...betas);
  const betaMax = Math.max(...betas);
  const baseZ = -0.7;
  const floorBounds = {
    alphaMin,
    alphaMax,
    betaMin,
    betaMax,
    baseZ,
  };
  draw3DFloorSections(ctx, state3d, width, height, floorBounds, "fill");
  const projectedAxes = [
    { from: value3DPoint(alphaMin, betaMin, baseZ, alphaMin, alphaMax, betaMin, betaMax), to: value3DPoint(alphaMax, betaMin, baseZ, alphaMin, alphaMax, betaMin, betaMax), label: "alpha" },
    { from: value3DPoint(alphaMin, betaMin, baseZ, alphaMin, alphaMax, betaMin, betaMax), to: value3DPoint(alphaMin, betaMax, baseZ, alphaMin, alphaMax, betaMin, betaMax), label: "beta" },
    { from: value3DPoint(alphaMin, betaMin, baseZ, alphaMin, alphaMax, betaMin, betaMax), to: value3DPoint(alphaMin, betaMin, 0.7, alphaMin, alphaMax, betaMin, betaMax), label: "" },
  ].map((axis) => ({
    ...axis,
    from2d: project3D(axis.from, state3d, width, height),
    to2d: project3D(axis.to, state3d, width, height),
  }));

  ctx.strokeStyle = "#667085";
  ctx.lineWidth = 1.1;
  ctx.fillStyle = "#344054";
  ctx.font = "23px Arial";
  for (const axis of projectedAxes) {
    ctx.beginPath();
    ctx.moveTo(axis.from2d.x, axis.from2d.y);
    ctx.lineTo(axis.to2d.x, axis.to2d.y);
    ctx.stroke();
    ctx.fillText(axis.label, axis.to2d.x + 10, axis.to2d.y - 7);
  }
  draw3DTicks(ctx, state3d, width, height, {
    alphaMin,
    alphaMax,
    betaMin,
    betaMax,
    baseZ,
  });

  const polygons = [];
  for (const surface of surfaces) {
    if (!surface.visible) continue;
    for (let b = 0; b < betas.length - 1; b += 1) {
      for (let a = 0; a < alphas.length - 1; a += 1) {
        const verts = [
          make3DPoint(a, b, surface.z[b][a], alphas.length, betas.length),
          make3DPoint(a + 1, b, surface.z[b][a + 1], alphas.length, betas.length),
          make3DPoint(a + 1, b + 1, surface.z[b + 1][a + 1], alphas.length, betas.length),
          make3DPoint(a, b + 1, surface.z[b + 1][a], alphas.length, betas.length),
        ];
        const projected = verts.map((point) => project3D(point, state3d, width, height));
        polygons.push({
          projected,
          depth: projected.reduce((sum, point) => sum + point.depth, 0) / projected.length,
          color: surface.color,
          active: is3DFloorSectionActive(state3d, (alphas[a] + alphas[a + 1]) / 2, (betas[b] + betas[b + 1]) / 2),
        });
      }
    }
  }

  polygons.sort((a, b) => a.depth - b.depth);
  for (const poly of polygons) {
    ctx.beginPath();
    ctx.moveTo(poly.projected[0].x, poly.projected[0].y);
    for (let index = 1; index < poly.projected.length; index += 1) {
      ctx.lineTo(poly.projected[index].x, poly.projected[index].y);
    }
    ctx.closePath();
    ctx.fillStyle = withAlpha(poly.color, poly.active ? 0.28 : 0.035);
    ctx.strokeStyle = withAlpha(poly.color, poly.active ? 0.5 : 0.08);
    ctx.lineWidth = poly.active ? 0.7 : 0.4;
    ctx.fill();
    ctx.stroke();
  }

  draw3DFloorSections(ctx, state3d, width, height, floorBounds, "label");
}

function make3DPoint(aIndex, bIndex, z, alphaCount, betaCount) {
  const x = alphaCount === 1 ? 0 : (aIndex / (alphaCount - 1) - 0.5) * 2.35;
  const y = betaCount === 1 ? 0 : (bIndex / (betaCount - 1) - 0.5) * 2.05;
  const zz = (z - 0.5) * 1.4;
  return [x, y, zz];
}

function value3DPoint(alpha, beta, z, alphaMin, alphaMax, betaMin, betaMax) {
  const x = ((alpha - alphaMin) / Math.max(1e-9, alphaMax - alphaMin) - 0.5) * 2.35;
  const y = ((beta - betaMin) / Math.max(1e-9, betaMax - betaMin) - 0.5) * 2.05;
  return [x, y, z];
}

function get3DFloorSections() {
  return [
    {
      id: "low-alpha-low-beta",
      shortLabel: ["Less sharp", "low robust"],
      alphaLo: 0.01,
      alphaHi: 1.0,
      betaLo: 0.0,
      betaHi: 0.5,
      fill: "20, 184, 166",
      label: ["Less sharp", "low robust"],
    },
    {
      id: "high-alpha-low-beta",
      shortLabel: ["High sharp", "low robust"],
      alphaLo: 1.0,
      alphaHi: 2.0,
      betaLo: 0.0,
      betaHi: 0.5,
      fill: "31, 111, 235",
      label: ["High sharp", "low robust"],
    },
    {
      id: "low-alpha-high-beta",
      shortLabel: ["Less sharp", "high robust"],
      alphaLo: 0.01,
      alphaHi: 1.0,
      betaLo: 0.5,
      betaHi: 1.0,
      fill: "245, 158, 11",
      label: ["Less sharp", "high robust"],
    },
    {
      id: "high-alpha-high-beta",
      shortLabel: ["High sharp", "high robust"],
      alphaLo: 1.0,
      alphaHi: 2.0,
      betaLo: 0.5,
      betaHi: 1.0,
      fill: "239, 68, 68",
      label: ["High sharp", "high robust"],
    },
  ];
}

function get3DFloorSectionId(alpha, beta) {
  const section = get3DFloorSections().find((item) =>
    alpha >= item.alphaLo &&
    alpha <= item.alphaHi &&
    beta >= item.betaLo &&
    beta <= item.betaHi
  );
  return section?.id || "";
}

function is3DFloorSectionActive(state3d, alpha, beta) {
  return state3d.floorSections?.has(get3DFloorSectionId(alpha, beta));
}

function draw3DFloorSections(ctx, state3d, width, height, bounds, mode) {
  const { alphaMin, alphaMax, betaMin, betaMax, baseZ } = bounds;
  const sections = get3DFloorSections();

  ctx.save();

  if (mode === "fill") {
    for (const section of sections) {
      const alphaLo = clamp(section.alphaLo, alphaMin, alphaMax);
      const alphaHi = clamp(section.alphaHi, alphaMin, alphaMax);
      const betaLo = clamp(section.betaLo, betaMin, betaMax);
      const betaHi = clamp(section.betaHi, betaMin, betaMax);
      if (alphaHi <= alphaLo || betaHi <= betaLo) continue;

      const projected = [
        value3DPoint(alphaLo, betaLo, baseZ, alphaMin, alphaMax, betaMin, betaMax),
        value3DPoint(alphaHi, betaLo, baseZ, alphaMin, alphaMax, betaMin, betaMax),
        value3DPoint(alphaHi, betaHi, baseZ, alphaMin, alphaMax, betaMin, betaMax),
        value3DPoint(alphaLo, betaHi, baseZ, alphaMin, alphaMax, betaMin, betaMax),
      ].map((point) => project3D(point, state3d, width, height));

      ctx.beginPath();
      ctx.moveTo(projected[0].x, projected[0].y);
      for (let index = 1; index < projected.length; index += 1) {
        ctx.lineTo(projected[index].x, projected[index].y);
      }
      ctx.closePath();
      ctx.fillStyle = `rgba(${section.fill}, 0.1)`;
      ctx.strokeStyle = "rgba(100, 116, 139, 0.18)";
      ctx.lineWidth = 0.8;
      ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
    return;
  }

  ctx.strokeStyle = "rgba(71, 85, 105, 0.46)";
  ctx.lineWidth = 1.1;
  const dividerLines = [
    [
      value3DPoint(1.0, betaMin, baseZ, alphaMin, alphaMax, betaMin, betaMax),
      value3DPoint(1.0, betaMax, baseZ, alphaMin, alphaMax, betaMin, betaMax),
    ],
    [
      value3DPoint(alphaMin, 0.5, baseZ, alphaMin, alphaMax, betaMin, betaMax),
      value3DPoint(alphaMax, 0.5, baseZ, alphaMin, alphaMax, betaMin, betaMax),
    ],
  ];
  for (const [from, to] of dividerLines) {
    drawProjectedLine(ctx, project3D(from, state3d, width, height), project3D(to, state3d, width, height));
  }

  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = "700 12px Arial";
  for (const section of sections) {
    if (!state3d.floorSections?.has(section.id)) continue;
    const alphaLo = clamp(section.alphaLo, alphaMin, alphaMax);
    const alphaHi = clamp(section.alphaHi, alphaMin, alphaMax);
    const betaLo = clamp(section.betaLo, betaMin, betaMax);
    const betaHi = clamp(section.betaHi, betaMin, betaMax);
    if (alphaHi <= alphaLo || betaHi <= betaLo) continue;

    const center = project3D(
      value3DPoint((alphaLo + alphaHi) / 2, (betaLo + betaHi) / 2, baseZ, alphaMin, alphaMax, betaMin, betaMax),
      state3d,
      width,
      height
    );
    draw3DFloorLabel(ctx, center.x, center.y, section.label);
  }

  ctx.restore();
}

function draw3DFloorLabel(ctx, x, y, lines) {
  const lineHeight = 14;
  const paddingX = 8;
  const paddingY = 5;
  const widths = lines.map((line) => ctx.measureText(line).width);
  const boxWidth = Math.max(...widths) + paddingX * 2;
  const boxHeight = lines.length * lineHeight + paddingY * 2;
  const boxX = x - boxWidth / 2;
  const boxY = y - boxHeight / 2;

  ctx.fillStyle = "rgba(255, 255, 255, 0.76)";
  ctx.strokeStyle = "rgba(148, 163, 184, 0.42)";
  ctx.lineWidth = 0.8;
  ctx.beginPath();
  if (typeof ctx.roundRect === "function") {
    ctx.roundRect(boxX, boxY, boxWidth, boxHeight, 6);
  } else {
    ctx.rect(boxX, boxY, boxWidth, boxHeight);
  }
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = "#172033";
  lines.forEach((line, index) => {
    ctx.fillText(line, x, boxY + paddingY + lineHeight * index + lineHeight / 2);
  });
}

function draw3DTicks(ctx, state3d, width, height, bounds) {
  const { alphaMin, alphaMax, betaMin, betaMax, baseZ } = bounds;
  const alphaTicks = [0.01, ...rangeTicks(0.5, 2.0, 0.5)].filter((value) => value >= alphaMin - 1e-9 && value <= alphaMax + 1e-9);
  const betaTicks = rangeTicks(0, 1, 0.2).filter((value) => value >= betaMin - 1e-9 && value <= betaMax + 1e-9);
  const betaTickLen = (betaMax - betaMin) * 0.035;
  const alphaTickLen = (alphaMax - alphaMin) * 0.022;

  ctx.save();
  ctx.strokeStyle = "#98a2b3";
  ctx.fillStyle = "#667085";
  ctx.lineWidth = 0.9;
  ctx.font = "18px Arial";

  for (const alpha of alphaTicks) {
    const p1 = project3D(value3DPoint(alpha, betaMin, baseZ, alphaMin, alphaMax, betaMin, betaMax), state3d, width, height);
    const p2 = project3D(value3DPoint(alpha, betaMin - betaTickLen, baseZ, alphaMin, alphaMax, betaMin, betaMax), state3d, width, height);
    drawProjectedLine(ctx, p1, p2);
    ctx.fillText(formatAxisTick(alpha), p2.x - 14, p2.y + 23);
  }

  for (const beta of betaTicks) {
    const p1 = project3D(value3DPoint(alphaMin, beta, baseZ, alphaMin, alphaMax, betaMin, betaMax), state3d, width, height);
    const p2 = project3D(value3DPoint(alphaMin - alphaTickLen, beta, baseZ, alphaMin, alphaMax, betaMin, betaMax), state3d, width, height);
    drawProjectedLine(ctx, p1, p2);
    ctx.fillText(formatAxisTick(beta), p2.x - 38, p2.y + 6);
  }

  for (const zValue of [0.5, 1]) {
    const actualZ = (zValue - 0.5) * 1.4;
    const p = project3D(value3DPoint(alphaMin, betaMin, actualZ, alphaMin, alphaMax, betaMin, betaMax), state3d, width, height);
    const p2 = project3D(value3DPoint(alphaMin - alphaTickLen, betaMin, actualZ, alphaMin, alphaMax, betaMin, betaMax), state3d, width, height);
    drawProjectedLine(ctx, p, p2);
    ctx.fillText(formatAxisTick(zValue), p2.x - 34, p2.y + 6);
  }

  ctx.restore();
}

function drawProjectedLine(ctx, a, b) {
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
}

function rangeTicks(start, end, step) {
  const ticks = [];
  for (let value = start; value <= end + step / 2; value += step) {
    ticks.push(Number(value.toFixed(10)));
  }
  return ticks;
}

function formatAxisTick(value) {
  return Number(value).toFixed(value < 0.1 && value > 0 ? 2 : 1).replace(/\.0$/, "");
}

function project3D([x, y, z], state3d, width, height) {
  const cosZ = Math.cos(state3d.rotZ);
  const sinZ = Math.sin(state3d.rotZ);
  const xz = x * cosZ - y * sinZ;
  const yz = x * sinZ + y * cosZ;
  const cosX = Math.cos(state3d.rotX);
  const sinX = Math.sin(state3d.rotX);
  const yy = yz * cosX - z * sinX;
  const zz = yz * sinX + z * cosX;
  const scale = 165 * state3d.zoom;
  return {
    x: width / 2 + xz * scale,
    y: height / 2 + yy * scale,
    depth: zz,
  };
}

function parseNumberList(value, label) {
  const numbers = value
    .split(",")
    .map((item) => Number(item.trim()))
    .filter((item) => Number.isFinite(item));
  if (!numbers.length) throw new Error(`Enter at least one ${label} value.`);
  if (numbers.some((item) => item < 0)) throw new Error(`${label} values must be non-negative.`);
  return numbers;
}

function renderHyperPlaneGrid(alphas, betas) {
  const hyperData = computeHyperPlaneData(alphas, betas);
  if (hyperData.error) return `<p class="emptyState">${escapeHtml(hyperData.error)}</p>`;
  const { groups, rawScores, normalizedScores } = hyperData;

  const panels = groups
    .map((group) => {
      const label = `${group.model}${group.source === "user" ? " (user)" : ""}`;
      return `
        <article class="heatPanel">
          <div class="chartTitle">
            <span class="colorDot" style="background:${modelColor(group)}"></span>
            ${escapeHtml(label)}
          </div>
          ${renderHyperPlaneSvg({
            alphas,
            betas,
            raw: rawScores.get(group.id),
            normalized: normalizedScores.get(group.id),
            color: modelColor(group),
          })}
        </article>
      `;
    })
    .join("");

  return `
    <div class="heatNote">Color uses normalized model scores at each alpha-beta point on an RdYlBu_r scale; text shows the raw mean metric.</div>
    <div class="heatGrid">${panels}</div>
  `;
}

function computeHyperPlaneData(alphas, betas) {
  const groups = state.analysis?.modelGroups || [];
  const dataset = state.analysis?.dataset;
  if (!groups.length || !dataset) return { error: "Run a comparison first." };

  const rawScores = new Map(groups.map((group) => [group.id, makeMatrix(betas.length, alphas.length, 0)]));
  const normalizedScores = new Map(groups.map((group) => [group.id, makeMatrix(betas.length, alphas.length, 0)]));

  for (let aIndex = 0; aIndex < alphas.length; aIndex += 1) {
    for (let bIndex = 0; bIndex < betas.length; bIndex += 1) {
      const pointScores = groups.map((group) => {
        const values = group.files.map((file) => calculateProbeMetric(file.info, dataset, alphas[aIndex], betas[bIndex]));
        const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
        rawScores.get(group.id)[bIndex][aIndex] = mean;
        return mean;
      });

      const min = Math.min(...pointScores);
      const max = Math.max(...pointScores);
      const denom = max - min + 1e-12;
      groups.forEach((group, modelIndex) => {
        normalizedScores.get(group.id)[bIndex][aIndex] = (pointScores[modelIndex] - min) / denom;
      });
    }
  }

  return { groups, rawScores, normalizedScores };
}

function renderHyperPlaneSvg({ alphas, betas, raw, normalized }) {
  const margin = { top: 16, right: 14, bottom: 42, left: 48 };
  const cellSize = 58;
  const width = margin.left + margin.right + alphas.length * cellSize;
  const height = margin.top + margin.bottom + betas.length * cellSize;
  const fieldWidth = alphas.length * cellSize;
  const fieldHeight = betas.length * cellSize;
  const samplesX = Math.min(110, Math.max(48, alphas.length * 28));
  const samplesY = Math.min(110, Math.max(48, betas.length * 28));
  const sampleWidth = fieldWidth / samplesX;
  const sampleHeight = fieldHeight / samplesY;

  const field = Array.from({ length: samplesY }, (_, yIndex) =>
    Array.from({ length: samplesX }, (_, xIndex) => {
      const x = margin.left + xIndex * sampleWidth;
      const y = margin.top + yIndex * sampleHeight;
      const gridX = alphas.length === 1 ? 0 : (xIndex / Math.max(1, samplesX - 1)) * (alphas.length - 1);
      const gridY = betas.length === 1 ? 0 : (yIndex / Math.max(1, samplesY - 1)) * (betas.length - 1);
      const value = bilinearValue(normalized, gridX, gridY);
      return `<rect class="heatSample" x="${x}" y="${y}" width="${sampleWidth + 0.2}" height="${sampleHeight + 0.2}" fill="${rdYlBuR(value)}"></rect>`;
    }).join("")
  ).join("");

  const pointLabels = betas
    .map((beta, bIndex) =>
      alphas
        .map((alpha, aIndex) => {
          const x = margin.left + aIndex * cellSize + cellSize / 2;
          const y = margin.top + bIndex * cellSize + cellSize / 2;
          return `
            <circle class="heatPoint" cx="${x}" cy="${y}" r="14"></circle>
            <text class="heatCellText" x="${x}" y="${y + 4}" text-anchor="middle">${formatCompactScore(raw[bIndex][aIndex])}</text>
          `;
        })
        .join("")
    )
    .join("");

  const xTicks = alphas
    .map((alpha, index) => {
      const x = margin.left + index * cellSize + cellSize / 2;
      return `<text class="xTick" x="${x}" y="${height - 18}" text-anchor="middle">${escapeHtml(String(alpha))}</text>`;
    })
    .join("");

  const yTicks = betas
    .map((beta, index) => {
      const y = margin.top + index * cellSize + cellSize / 2 + 4;
      return `<text class="yTick" x="${margin.left - 8}" y="${y}" text-anchor="end">${escapeHtml(String(beta))}</text>`;
    })
    .join("");

  return `
    <svg class="heatSvg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Alpha beta heat grid">
      <g>
        ${field}
      </g>
      <rect class="heatFrame" x="${margin.left}" y="${margin.top}" width="${fieldWidth}" height="${fieldHeight}" rx="8" ry="8"></rect>
      ${pointLabels}
      ${xTicks}
      ${yTicks}
      <text class="axisTitle" x="${margin.left + (alphas.length * cellSize) / 2}" y="${height - 2}" text-anchor="middle">alpha</text>
      <text class="axisTitle" x="${-(margin.top + (betas.length * cellSize) / 2)}" y="13" text-anchor="middle" transform="rotate(-90)">beta</text>
    </svg>
  `;
}

function makeMatrix(rows, cols, value) {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => value));
}

function linspace(start, end, count) {
  if (count <= 1) return [start];
  return Array.from({ length: count }, (_, index) => start + ((end - start) * index) / (count - 1));
}

function buildMeanRankMap(group) {
  const aggregate = new Map();
  for (const file of group.files) {
    for (const row of file.info) {
      const [query, mode, rawRank] = row;
      const key = infoKey(query, mode);
      const rank = Number(rawRank);
      const target = mode === "h" ? Number(query[0]) : Number(query[2]);
      const current = aggregate.get(key) || {
        query,
        mode,
        target,
        degree: state.analysis.dataset.counts[target] ?? 0,
        sum: 0,
        count: 0,
      };
      current.sum += rank;
      current.count += 1;
      aggregate.set(key, current);
    }
  }

  return new Map(
    [...aggregate.entries()].map(([key, value]) => [
      key,
      {
        query: value.query,
        mode: value.mode,
        target: value.target,
        degree: value.degree,
        rank: value.sum / value.count,
      },
    ])
  );
}

function getRowsForParam(result, { alpha, beta }) {
  const key = paramKey(alpha, beta);
  return result.models
    .map((model) => ({
      model: model.model,
      source: model.source,
      runCount: model.runCount,
      score: model.scores[key].mean,
      std: model.scores[key].std,
      color: modelColor(model),
    }))
    .sort((a, b) => b.score - a.score);
}

function renderBarChartSvg(rows, ariaLabel) {
  const margin = { top: 30, right: 14, bottom: 66, left: 44 };
  const plotHeight = 220;
  const modelSlot = 62;
  const width = Math.max(400, margin.left + margin.right + rows.length * modelSlot);
  const height = margin.top + plotHeight + margin.bottom;
  const plotWidth = width - margin.left - margin.right;
  const maxValue = Math.max(...rows.map((row) => row.score + row.std), 1e-12);
  const minValue = Math.min(...rows.map((row) => Math.max(0, row.score - row.std)));
  const spread = Math.max(maxValue - minValue, maxValue * 0.08, 1e-6);
  let yMin = Math.max(0, minValue - spread * 0.35);
  if (yMin === 0 && minValue > 0) yMin = minValue * 0.65;
  const yMax = maxValue + spread * 0.2;
  const yRange = Math.max(yMax - yMin, 1e-9);
  const ticks = makeTicks(yMin, yMax, 5);
  const barWidth = Math.min(32, Math.max(22, modelSlot * 0.48));
  const baseY = margin.top + plotHeight;
  const yScale = (value) => baseY - ((value - yMin) / yRange) * plotHeight;

  const tickMarkup = ticks
    .map((tick) => {
      const y = yScale(tick);
      return `
        <line class="gridLine" x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}"></line>
        <text class="yTick" x="${margin.left - 8}" y="${y + 4}" text-anchor="end">${formatTick(tick)}</text>
      `;
    })
    .join("");

  const barMarkup = rows
    .map((row, index) => {
      const slotX = margin.left + index * (plotWidth / rows.length);
      const centerX = slotX + plotWidth / rows.length / 2;
      const barHeight = Math.max(2, ((row.score - yMin) / yRange) * plotHeight);
      const x = centerX - barWidth / 2;
      const y = baseY - barHeight;
      const errTop = yScale(Math.min(yMax, row.score + row.std));
      const errBottom = yScale(Math.max(yMin, row.score - row.std));
      const labelY = Math.max(12, errTop - 20);
      const modelLabel = escapeHtml(row.model);
      const scoreLabel = formatCompactScore(row.score);
      const stdLabel = formatCompactScore(row.std);

      return `
        <g class="barGroup">
          <title>${escapeHtml(row.model)}: ${scoreLabel} +/- ${stdLabel}</title>
          <rect class="chartBar ${row.source === "user" ? "userBar" : ""}" x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" rx="5" ry="5" fill="${row.color}"></rect>
          ${
            row.runCount > 1
              ? `<line class="errorLine" x1="${centerX}" y1="${errTop}" x2="${centerX}" y2="${errBottom}"></line>
                 <line class="errorLine" x1="${centerX - 7}" y1="${errTop}" x2="${centerX + 7}" y2="${errTop}"></line>
                 <line class="errorLine" x1="${centerX - 7}" y1="${errBottom}" x2="${centerX + 7}" y2="${errBottom}"></line>`
              : ""
          }
          <text class="valueLabel" x="${centerX}" y="${labelY}" text-anchor="middle">
            <tspan x="${centerX}" dy="0">${escapeHtml(scoreLabel)}&#177;</tspan>
            <tspan x="${centerX}" dy="11">${escapeHtml(stdLabel)}</tspan>
          </text>
        </g>
        <text class="xTick" x="${centerX}" y="${baseY + 16}" text-anchor="end" transform="rotate(-34 ${centerX} ${baseY + 16})">${modelLabel}</text>
      `;
    })
    .join("");

  return `
    <svg class="chartSvg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(ariaLabel)}">
      <line class="axisLine" x1="${margin.left}" y1="${baseY}" x2="${width - margin.right}" y2="${baseY}"></line>
      <line class="axisLine" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${baseY}"></line>
      ${tickMarkup}
      ${barMarkup}
    </svg>
  `;
}

function resultToCsv(result) {
  const lines = ["dataset,model,source,runs,alpha,beta,mean,std,min,max"];
  for (const model of result.models) {
    for (const { alpha, beta } of result.params) {
      const score = model.scores[paramKey(alpha, beta)];
      lines.push(
        [
          result.dataset.name,
          model.model,
          model.source,
          model.runCount,
          alpha,
          beta,
          score.mean,
          score.std,
          score.min,
          score.max,
        ]
          .map(csvCell)
          .join(",")
      );
    }
  }
  return `${lines.join("\n")}\n`;
}

function setStatus(message) {
  el.status.textContent = message;
}

function assignModelIds(modelGroups) {
  for (const [index, group] of modelGroups.entries()) {
    group.id = `${group.source}:${group.model}:${index}`;
  }
}

function firstManifestDataset() {
  return Object.keys(state.manifest?.datasets || {})[0] || "";
}

function paramKey(alpha, beta) {
  return `a=${alpha}|b=${beta}`;
}

function uniqueParams(params) {
  const seen = new Set();
  return params.filter(({ alpha, beta }) => {
    const key = paramKey(alpha, beta);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function encodePath(path) {
  return path.split("/").map(encodeURIComponent).join("/");
}

function basename(path) {
  return path.split("/").pop();
}

function formatScore(value) {
  return Number(value).toFixed(6);
}

function formatCompactScore(value) {
  return Number(value).toFixed(3);
}

function formatTick(value) {
  return Number(value).toFixed(value >= 1 ? 0 : 3).replace(/0+$/, "").replace(/\.$/, "");
}

function makeTicks(min, max, count) {
  return Array.from({ length: count }, (_, index) => min + ((max - min) / (count - 1)) * index);
}

function logTicks(min, max) {
  const ticks = [];
  const start = Math.floor(Math.log10(min));
  const end = Math.ceil(Math.log10(max));
  for (let power = start; power <= end; power += 1) {
    const value = Math.pow(10, power);
    if (value >= min && value <= max) ticks.push(value);
  }
  return ticks.length > 1 ? ticks : [min, max];
}

function powerOfTenTicks(maxValue) {
  const ticks = [];
  for (let value = 1; value <= maxValue; value *= 10) {
    ticks.push(value);
  }
  return ticks;
}

function formatCountTick(value) {
  if (value >= 1000) return `${Number(value / 1000).toFixed(value >= 10000 ? 0 : 1).replace(/\.0$/, "")}k`;
  return String(Math.round(value));
}

function formatCount(value) {
  return Number(value).toFixed(value >= 10 ? 1 : 2).replace(/\.0+$/, "");
}

function formatRank(value) {
  return Number(value).toFixed(value >= 10 ? 1 : 2).replace(/\.0+$/, "");
}

function bilinearValue(matrix, x, y) {
  const rows = matrix.length;
  const cols = matrix[0]?.length || 0;
  if (!rows || !cols) return 0;

  const x0 = Math.floor(clamp(x, 0, cols - 1));
  const y0 = Math.floor(clamp(y, 0, rows - 1));
  const x1 = Math.min(cols - 1, x0 + 1);
  const y1 = Math.min(rows - 1, y0 + 1);
  const tx = clamp(x - x0, 0, 1);
  const ty = clamp(y - y0, 0, 1);

  const top = matrix[y0][x0] * (1 - tx) + matrix[y0][x1] * tx;
  const bottom = matrix[y1][x0] * (1 - tx) + matrix[y1][x1] * tx;
  return top * (1 - ty) + bottom * ty;
}

function rdYlBuR(value) {
  const palette = [
    "#313695",
    "#4575b4",
    "#74add1",
    "#abd9e9",
    "#e0f3f8",
    "#ffffbf",
    "#fee090",
    "#fdae61",
    "#f46d43",
    "#d73027",
    "#a50026",
  ];
  const t = clamp(value, 0, 1) * (palette.length - 1);
  const index = Math.floor(t);
  const nextIndex = Math.min(palette.length - 1, index + 1);
  return mixColor(palette[index], palette[nextIndex], t - index);
}

function withAlpha(color, alpha) {
  const rgb = hexToRgb(color || "#8a95a6");
  return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
}

function mixColor(start, end, amount) {
  const a = hexToRgb(start);
  const b = hexToRgb(end);
  const t = clamp(amount, 0, 1);
  const mixed = {
    r: Math.round(a.r + (b.r - a.r) * t),
    g: Math.round(a.g + (b.g - a.g) * t),
    b: Math.round(a.b + (b.b - a.b) * t),
  };
  return `rgb(${mixed.r}, ${mixed.g}, ${mixed.b})`;
}

function hexToRgb(hex) {
  const normalized = String(hex || "").replace("#", "");
  const expanded = normalized.length === 3 || normalized.length === 4
    ? normalized.split("").map((char) => `${char}${char}`).join("")
    : normalized;
  const full = expanded.length === 8 ? expanded.slice(0, 6) : expanded;
  const value = Number.parseInt(full || "8a95a6", 16);
  return {
    r: (value >> 16) & 255,
    g: (value >> 8) & 255,
    b: value & 255,
  };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function createSvgElement(tag, attributes = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, value);
  }
  return element;
}

function infoKey(query, mode) {
  return `${query[0]}|${query[1]}|${query[2]}|${mode}`;
}

function getSelectedGroups(selectedIds) {
  const selected = new Set(selectedIds);
  return (state.analysis?.modelGroups || []).filter((group) => selected.has(group.id));
}

function formatEntity(id, dataset) {
  const rawName = dataset.id2entity?.[id];
  if (!rawName) return String(id);
  return dataset.decodeMap?.[rawName] || prettifyEntityName(rawName);
}

function formatRelation(id, dataset) {
  return prettifyEntityName(dataset.id2relation?.[id] || String(id));
}

function formatQuery(query, dataset) {
  const [head, relation, tail] = query;
  return `[${formatEntity(head, dataset)}, ${formatRelation(relation, dataset)}, ${formatEntity(tail, dataset)}]`;
}

function prettifyEntityName(value) {
  return String(value)
    .replace(/^\/m\//, "")
    .replace(/^\/g\//, "")
    .replace(/_/g, " ");
}

function truncateText(value, maxLength) {
  const text = String(value);
  return text.length > maxLength ? `${text.slice(0, Math.max(0, maxLength - 1))}...` : text;
}

function modelColor(model) {
  return model.color || MODEL_COLORS[model.model] || "#8a95a6";
}

function csvCell(value) {
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function downloadText(filename, text, mimeType) {
  const blob = new Blob([text], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function downloadPlotSvgs() {
  const svgs = [...el.plots.querySelectorAll("svg")];
  if (!svgs.length) {
    setStatus("No SVG plots are available to download yet.");
    return;
  }

  const cssText = collectSameOriginCss();
  svgs.forEach((svg, index) => {
    const title = findPlotTitle(svg) || `plot-${index + 1}`;
    const filename = `${slugify(`${state.lastResult?.dataset?.name || "probe"}-${title}`)}.svg`;
    downloadText(filename, serializeStandaloneSvg(svg, cssText), "image/svg+xml;charset=utf-8");
  });
  setStatus(`Downloaded ${svgs.length} SVG plot${svgs.length === 1 ? "" : "s"}. Canvas-only views such as the 3D plot should be saved through PDF.`);
}

function printResultsPdf() {
  if (!state.lastResult) {
    setStatus("Run a comparison before saving a PDF.");
    return;
  }
  window.print();
}

function serializeStandaloneSvg(svg, cssText) {
  const clone = svg.cloneNode(true);
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("version", "1.1");
  if (!clone.getAttribute("width")) clone.setAttribute("width", svg.viewBox.baseVal.width || svg.clientWidth);
  if (!clone.getAttribute("height")) clone.setAttribute("height", svg.viewBox.baseVal.height || svg.clientHeight);

  const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
  style.textContent = cssText;
  clone.insertBefore(style, clone.firstChild);

  return `<?xml version="1.0" encoding="UTF-8"?>\n${new XMLSerializer().serializeToString(clone)}`;
}

function collectSameOriginCss() {
  const chunks = [];
  for (const sheet of document.styleSheets) {
    try {
      const rules = [...sheet.cssRules].map((rule) => rule.cssText).join("\n");
      if (rules) chunks.push(rules);
    } catch {
      // Cross-origin stylesheets, such as web fonts, cannot be read by the browser.
    }
  }
  return chunks.join("\n");
}

function findPlotTitle(node) {
  const section = node.closest("section, article, .chartPanel, .showcase");
  const titleNode = section?.querySelector("h3, .chartTitle");
  return titleNode?.textContent?.trim() || node.getAttribute("aria-label") || "";
}

function slugify(value) {
  return String(value || "plot")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80) || "plot";
}
