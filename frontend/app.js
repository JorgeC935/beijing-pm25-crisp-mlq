"use strict";

const $ = (id) => document.getElementById(id);
const API = () => `http://127.0.0.1:${$("port").value}`;

const fields = [
  ["pm25", "PM2.5 (µg/m³)", 75],
  ["pm10", "PM10 (µg/m³)", 100],
  ["so2", "SO₂ (µg/m³)", 10],
  ["no2", "NO₂ (µg/m³)", 35],
  ["co", "CO (µg/m³)", 600],
  ["o3", "O₃ (µg/m³)", 45],
  ["temp", "Temperatura (°C)", 8],
  ["pres", "Presión (hPa)", 1015],
  ["dewp", "Punto de rocío (°C)", -5],
  ["rain", "Lluvia (mm)", 0],
  ["wspm", "Viento (m/s)", 2],
  ["wd", "Dirección del viento", "NW"],
];

const MIN_RETRAIN_MEASUREMENTS = 1;

let state = null;
let busy = false;
let modelList = [];
let metricData = null;
let historyList = [];

const fmt = (value, digits = 2) => {
  if (value === null || value === undefined || value === "") return "—";
  return Number(value).toLocaleString("es-BO", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
};
const count = (value) => value === null || value === undefined
  ? "—"
  : Number(value).toLocaleString("es-BO");
const date = (value) => value
  ? String(value).slice(0, 16).replace("T", " ")
  : "—";

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = text;
  if (className) node.className = className;
  return node;
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(API() + path, options);
  } catch {
    throw new Error("No se pudo conectar con la API. Revisa Uvicorn y el puerto seleccionado.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload.detail;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail ?? payload));
  }
  return payload;
}

function notify(message, type = "") {
  const notice = $("notice");
  notice.textContent = message;
  notice.className = "notice" + (type ? ` ${type}` : "");
}

function switchTab(id) {
  document.querySelectorAll(".section").forEach((section) => {
    section.hidden = section.id !== id;
    section.classList.toggle("active", section.id === id);
  });
  document.querySelectorAll("[data-tab]").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === id);
  });
}

function controls() {
  const trained = Boolean(state?.trained);
  const historical = Number(state?.historical ?? 0);
  const simulated = Number(state?.simulated ?? 0);
  const hasFile = Boolean($("historical-file").files.length);

  $("initial-btn").disabled = busy || !state || trained || (historical === 0 && !hasFile);
  $("retrain-btn").disabled = busy || !trained || simulated < MIN_RETRAIN_MEASUREMENTS;
  $("obs-fields").disabled = busy || !trained;
  $("port").disabled = busy;

  $("historical-file").disabled = busy || trained || historical > 0;
  $("historical-file").hidden = historical > 0 || trained;
  $("initial-help").textContent = trained
    ? "El modelo de seis horas ya está entrenado."
    : historical > 0
      ? "El histórico ya está guardado en PostgreSQL. Se entrenará con esos registros sin volver a importarlos."
      : "Selecciona el CSV UCI original de Wanliu (35.064 registros).";
  $("initial-btn").textContent = historical > 0
    ? "Entrenar con histórico disponible"
    : "Importar CSV y entrenar";
  $("retrain-btn").textContent = simulated < MIN_RETRAIN_MEASUREMENTS
    ? "Registra una medición nueva"
    : "Reentrenar y comparar modelos";
}

function renderTable(bodyId, rows, emptyMessage, columnCount) {
  const body = $(bodyId);
  body.replaceChildren();
  if (!rows.length) {
    const tr = document.createElement("tr");
    const td = element("td", emptyMessage);
    td.colSpan = columnCount;
    tr.append(td);
    body.append(tr);
    return;
  }
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const value of row) {
      const td = document.createElement("td");
      if (value instanceof Node) td.append(value);
      else td.textContent = value === null || value === undefined ? "—" : String(value);
      tr.append(td);
    }
    body.append(tr);
  }
}

function updateState(nextState) {
  state = nextState;
  $("connection").textContent = "API + PostgreSQL";
  $("connection").classList.add("connected");
  $("historical-count").textContent = count(state.historical);
  $("simulated-count").textContent = count(state.simulated);
  $("active-version").textContent = state.version ?? "Sin entrenar";
  $("next-date").textContent = date(state.next_expected_observed_at);
  controls();
}

async function refreshState() {
  updateState(await request("/state"));
}

function clearMetrics() {
  metricData = null;
  $("metric-set").replaceChildren();
  $("metric-model").replaceChildren();
  for (const id of ["global-mae", "global-rmse", "global-r2"]) $(id).textContent = "—";
  $("metric-caption").textContent = "Entrena un modelo para consultar sus métricas.";
  $("metric-improvement").textContent = "";
  $("evaluation-details").replaceChildren();
  renderTable("metric-comparison", [], "Sin métricas disponibles.", 5);
}

async function refreshModels() {
  modelList = await request("/models");
  renderTable(
    "versions",
    modelList.map((model) => [
      model.version,
      model.kind === "initial" ? "Inicial" : "Reentrenamiento",
      model.is_active ? "Activo" : "Inactivo",
      count(model.observation_count),
      model.is_demo ? "Demostración" : "Histórico",
    ]),
    "Sin modelos de seis horas registrados.",
    5,
  );

  const select = $("metric-model");
  const previous = select.value;
  select.replaceChildren();
  for (const model of modelList) {
    const option = element(
      "option",
      `${model.version}${model.is_active ? " · activo" : ""}${model.is_demo ? " · demo" : ""}`,
    );
    option.value = model.id;
    select.append(option);
  }
  const active = modelList.find((model) => model.is_active);
  select.value = active?.id ?? (modelList.some((model) => String(model.id) === previous) ? previous : "");
  if (select.value) await loadMetrics();
  else clearMetrics();
}

async function loadMetrics() {
  const modelId = $("metric-model").value;
  if (!modelId) {
    clearMetrics();
    return;
  }

  metricData = await request(`/metrics?model_id=${encodeURIComponent(modelId)}`);
  const sets = [...new Set(metricData.metrics.map((metric) => metric.evaluation_set))];
  const labels = {
    validation: "Validación histórica",
    test: "Prueba histórica",
    retrain_validation: metricData.split?.protocol === "fixed_historical_benchmark_v1"
      ? "Referencia histórica común (retrospectiva)" : "Evaluación del reentrenamiento",
  };

  const select = $("metric-set");
  const previous = select.value;
  select.replaceChildren();
  for (const setName of sets) {
    const option = element("option", labels[setName] ?? setName);
    option.value = setName;
    select.append(option);
  }
  select.value = sets.includes(previous)
    ? previous
    : sets.includes("test")
      ? "test"
      : sets[0] ?? "";
  renderMetrics();
}

function renderMetrics() {
  if (!metricData) return;
  const evaluationSet = $("metric-set").value;
  const model = metricData.metrics.find(
    (metric) => metric.evaluation_set === evaluationSet && metric.scope === "global",
  );
  const baseline = metricData.baseline_metrics?.[evaluationSet];

  $("global-mae").textContent = fmt(model?.mae);
  $("global-rmse").textContent = fmt(model?.rmse);
  $("global-r2").textContent = fmt(model?.r2, 3);

  const caption = [
    metricData.is_demo ? "Versión de demostración con mediciones simuladas." : "Evaluación histórica.",
    ...(metricData.split?.protocol === "fixed_historical_benchmark_v1"
      ? ["Comparación retrospectiva; no es una prueba de rendimiento futuro."] : []),
    `${count(model?.sample_count)} ejemplos evaluados.`,
    `${metricData.feature_count} características.`,
    "Objetivo: PM2.5 a +6 h.",
  ];
  $("metric-caption").textContent = caption.join(" ");

  const rows = [];
  if (model) rows.push(["Modelo IA", fmt(model.mae), fmt(model.rmse), fmt(model.r2, 3), count(model.sample_count)]);
  if (baseline) rows.push(["Persistencia", fmt(baseline.mae), fmt(baseline.rmse), fmt(baseline.r2, 3), count(baseline.sample_count)]);

  const comparison = metricData.comparison;
  if (evaluationSet === "retrain_validation" && comparison?.incumbent) {
    rows.push([
      "Modelo anterior",
      fmt(comparison.incumbent.mae),
      fmt(comparison.incumbent.rmse),
      fmt(comparison.incumbent.r2, 3),
      count(comparison.incumbent.sample_count),
    ]);
  }
  renderTable("metric-comparison", rows, "Sin métricas disponibles.", 5);

  if (model && baseline && baseline.mae > 0) {
    const improvement = ((baseline.mae - model.mae) / baseline.mae) * 100;
    $("metric-improvement").textContent = improvement >= 0
      ? `El modelo reduce el MAE ${fmt(improvement, 2)}% frente a persistencia.`
      : `El MAE del modelo es ${fmt(Math.abs(improvement), 2)}% mayor que el de persistencia.`;
    $("metric-improvement").className = "comparison-note " + (improvement >= 0 ? "positive" : "caution");
  } else {
    $("metric-improvement").textContent = "";
  }

  const split = metricData.split ?? {};
  const detailItems = [
    ["Protocolo", split.protocol ?? "—"],
    ["Orígenes de entrenamiento", count(split.train_origins)],
    ["Ejemplos nuevos usados en el ajuste", count(split.new_training_examples)],
    ["Orígenes de validación", count(split.validation_origins)],
    ["Orígenes de prueba", count(split.test_origins)],
    ["Fin del período usado para ajustar", date(split.training_target_end)],
    ["Horizonte", "6 horas"],
  ];
  const details = $("evaluation-details");
  details.replaceChildren();
  for (const [label, value] of detailItems) {
    const item = element("div");
    item.append(element("small", label), element("strong", value));
    details.append(item);
  }
}

function showForecast(run) {
  if (!run) return;
  const prediction = run.prediction;
  $("current-pm").textContent = fmt(run.input?.pm25);
  $("forecast-value").textContent = fmt(prediction?.pm25_predicted, 1);
  $("origin").textContent = `Medición registrada: ${date(run.observed_at)}`;
  $("forecast-target").textContent = `Fecha objetivo: ${date(prediction?.target_at)}`;
  $("forecast-version").textContent = run.model_version ?? "—";

  const current = Number(run.input?.pm25);
  const future = Number(prediction?.pm25_predicted);
  const delta = future - current;
  const change = $("forecast-change");
  if (Number.isFinite(delta)) {
    const prefix = delta > 0 ? "+" : "";
    change.textContent = `Variación estimada: ${prefix}${fmt(delta)} µg/m³`;
    change.className = "change-pill " + (delta > 0 ? "up" : delta < 0 ? "down" : "neutral");
  } else {
    change.textContent = "Variación: —";
    change.className = "change-pill";
  }
}

function showHistoryDetail(run) {
  const detail = $("history-detail");
  detail.hidden = false;
  detail.replaceChildren(
    element("h2", `Pronóstico originado el ${date(run.observed_at)}`),
    element("p", run.model_version, "help-text"),
  );

  const grid = element("div", undefined, "detail-grid");
  for (const [key, label] of fields) {
    const item = element("div");
    item.append(element("small", label), element("strong", String(run.input?.[key] ?? "—")));
    grid.append(item);
  }
  const prediction = element("div", undefined, "history-prediction");
  prediction.append(
    element("span", "PM2.5 estimado a +6 h"),
    element("strong", `${fmt(run.prediction?.pm25_predicted)} µg/m³`),
    element("small", `Fecha objetivo: ${date(run.prediction?.target_at)}`),
  );
  detail.append(grid, prediction);

  if (run.observed_pm25 !== null && run.observed_pm25 !== undefined) {
    detail.append(element(
      "p",
      `Observación posterior: ${fmt(run.observed_pm25)} µg/m³ · Error absoluto: ${fmt(run.absolute_error)} µg/m³${run.observed_is_simulated ? " · dato simulado" : " · dato histórico"}`,
      "help-text",
    ));
  } else {
    detail.append(element(
      "p",
      "Todavía no existe una observación registrada para la fecha objetivo.",
      "help-text",
    ));
  }

  const button = element("button", "Ver en Pronóstico", "button-secondary");
  button.addEventListener("click", () => {
    showForecast(run);
    switchTab("forecast");
  });
  detail.append(button);
}

async function refreshHistory() {
  historyList = await request("/predictions");
  const rows = historyList.map((run) => {
    const button = element("button", "Detalle", "table-button");
    button.addEventListener("click", () => showHistoryDetail(run));
    return [
      date(run.observed_at),
      fmt(run.input?.pm25),
      fmt(run.prediction?.pm25_predicted),
      date(run.prediction?.target_at),
      run.observed_pm25 === null ? "—" : fmt(run.observed_pm25),
      run.absolute_error === null ? "—" : fmt(run.absolute_error),
      button,
    ];
  });
  renderTable("history-list", rows, "Sin predicciones registradas.", 7);
  if (historyList.length) showForecast(historyList[0]);
}

async function refreshAll() {
  await refreshState();
  await refreshModels();
  await refreshHistory();
  controls();
}

async function action(message, work) {
  if (busy) return;
  busy = true;
  controls();
  notify(message);
  try {
    await work();
  } catch (error) {
    notify(error.message, "error");
  } finally {
    busy = false;
    controls();
  }
}

function renderTraining(result) {
  const card = $("training-result");
  card.hidden = false;
  card.replaceChildren();

  const title = result.status === "trained"
    ? "Entrenamiento inicial completado"
    : result.status === "promoted"
      ? "Nuevo modelo promovido"
      : "Candidato evaluado; se conserva el modelo activo";
  card.append(element("h2", title), element("p", result.note));

  const split = result.split ?? {};
  const items = [
    ["Versión", result.model_version],
    ["Horizonte", "6 horas"],
    ["Observaciones disponibles", count(result.observations)],
    ["Mediciones simuladas disponibles", count(result.simulated_observations)],
    ["Ejemplos de entrenamiento", count(split.train_examples)],
    ["Ejemplos nuevos usados en el ajuste", count(split.new_training_examples)],
    ["Ejemplos de validación", count(split.validation_examples)],
    ["Ejemplos de prueba", count(split.test_examples)],
  ];
  const grid = element("div", undefined, "detail-grid");
  for (const [label, value] of items) {
    const item = element("div");
    item.append(element("small", label), element("strong", value ?? "—"));
    grid.append(item);
  }
  card.append(grid);

  if (result.metrics?.length) {
    const list = element("div", undefined, "training-metrics");
    for (const metric of result.metrics) {
      list.append(element(
        "p",
        `${metric.evaluation_set === "retrain_validation" && split.protocol === "fixed_historical_benchmark_v1" ? "Referencia histórica común" : metric.evaluation_set}: MAE ${fmt(metric.mae)} · RMSE ${fmt(metric.rmse)} · R² ${fmt(metric.r2, 3)} · ${count(metric.sample_count)} ejemplos`,
      ));
    }
    card.append(list);
  }

  if (result.comparison) {
    const comparison = result.comparison;
    card.append(element(
      "p",
      `Comparación del candidato: MAE ${fmt(comparison.candidate?.mae)}; modelo anterior ${fmt(comparison.incumbent?.mae)}; persistencia ${fmt(comparison.persistence?.mae)}.`,
      "comparison-note",
    ));
    if (comparison.promotion_reason) card.append(element("p", comparison.promotion_reason, "help-text"));
  }
}

function buildObservationForm() {
  for (const [key, label, defaultValue] of fields) {
    const wrapper = element("div", undefined, "field");
    const inputLabel = element("label", label);
    const input = element("input");
    input.id = `field-${key}`;
    input.name = key;
    inputLabel.htmlFor = input.id;
    input.type = key === "wd" ? "text" : "number";
    if (key !== "wd") input.step = "any";
    if (key === "pm25") {
      input.required = true;
      input.min = "0";
    }
    input.value = defaultValue;
    wrapper.append(inputLabel, input);
    $("fields").append(wrapper);
  }
}

document.querySelectorAll("[data-tab]").forEach((button) => {
  button.addEventListener("click", () => switchTab(button.dataset.tab));
});

$("port").value = ["8000", "8001"].includes(localStorage.getItem("wanliu-6h-port"))
  ? localStorage.getItem("wanliu-6h-port")
  : "8000";
$("port").addEventListener("change", () => {
  localStorage.setItem("wanliu-6h-port", $("port").value);
  state = null;
  controls();
  action("Conectando con el puerto seleccionado…", async () => {
    await refreshAll();
    notify("Conexión correcta.", "ok");
  });
});

buildObservationForm();
$("historical-file").addEventListener("change", controls);

$("initial-btn").addEventListener("click", () => action(
  "Preparando el entrenamiento inicial… No cierres esta página.",
  async () => {
    const body = new FormData();
    if ($("historical-file").files.length) {
      body.append("file", $("historical-file").files[0]);
    }
    const result = await request("/train-initial", { method: "POST", body });
    renderTraining(result);
    await refreshAll();
    notify("Entrenamiento inicial completado.", "ok");
  },
));

$("retrain-btn").addEventListener("click", () => action(
  "Entrenando el candidato y comparándolo con el modelo activo…",
  async () => {
    const result = await request("/retrain", { method: "POST" });
    renderTraining(result);
    await refreshAll();
    $("metric-model").value = String(result.model_id);
    await loadMetrics();
    notify(result.status === "promoted"
      ? "El nuevo modelo fue promovido."
      : "El modelo anterior se mantiene activo.", "ok");
  },
));

$("observation-form").addEventListener("submit", (event) => {
  event.preventDefault();
  action("Registrando la medición y calculando el pronóstico a seis horas…", async () => {
    const observation = {};
    for (const [key] of fields) {
      const value = $(`field-${key}`).value.trim();
      observation[key] = key === "wd" ? (value || "Desconocido") : (value === "" ? null : Number(value));
    }
    const result = await request("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(observation),
    });
    showForecast(result);
    await refreshAll();
    switchTab("forecast");
    notify("Medición registrada y pronóstico a +6 horas generado.", "ok");
  });
});

$("metric-model").addEventListener("change", () => action(
  "Cargando métricas…",
  async () => loadMetrics(),
));
$("metric-set").addEventListener("change", renderMetrics);
$("refresh-models").addEventListener("click", () => action(
  "Actualizando versiones…",
  async () => {
    await refreshModels();
    notify("Versiones actualizadas.", "ok");
  },
));
$("refresh-history").addEventListener("click", () => action(
  "Actualizando historial…",
  async () => {
    await refreshHistory();
    notify("Historial actualizado.", "ok");
  },
));

refreshAll()
  .then(() => notify("Sistema conectado. Consulta el estado y registra una medición.", "ok"))
  .catch((error) => notify(error.message, "error"));
