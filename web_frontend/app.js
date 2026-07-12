const state = { factors: [], industry: null };
const palette = ["#1f4e79", "#5f6f52", "#8a6d3b", "#7b4f46", "#4f6d7a", "#765e8b", "#92724a", "#4c6772", "#87605b", "#5d6872", "#53665b", "#755b65", "#76834e", "#6c6c57", "#496077", "#845e4a", "#5d6f91", "#76604f", "#4c7471", "#7d657a"];
const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.remove("hidden");
  window.setTimeout(() => toast.classList.add("hidden"), 5200);
}

async function api(path) {
  const response = await fetch(path);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function fmt(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  if (typeof value === "number") return Math.abs(value) >= 1000 ? value.toLocaleString("en-US", { maximumFractionDigits: digits }) : value.toFixed(digits);
  return String(value);
}

function renderTable(el, rows, columns) {
  if (!rows || !rows.length) {
    el.innerHTML = `<tbody><tr><td colspan="${columns.length}">暂无数据</td></tr></tbody>`;
    return;
  }
  const heads = columns.map((column) => `<th>${escapeHtml(column.label)}</th>`).join("");
  const body = rows.map((row) => `<tr>${columns.map((column) => {
    const raw = row[column.key];
    const value = column.format ? column.format(raw, row) : fmt(raw, column.digits ?? 4);
    return `<td>${column.html ? value : escapeHtml(value)}</td>`;
  }).join("")}</tr>`).join("");
  el.innerHTML = `<thead><tr>${heads}</tr></thead><tbody>${body}</tbody>`;
}

function metricCards(el, metrics) {
  el.innerHTML = metrics.map((item) => `
    <div class="metric"><strong>${escapeHtml(fmt(item.value, item.digits ?? 4))}</strong><span>${escapeHtml(item.label)}</span></div>
  `).join("");
}

function bounds(series, keys) {
  const values = [];
  series.forEach((row) => keys.forEach((key) => {
    const value = Number(row[key]);
    if (Number.isFinite(value)) values.push(value);
  }));
  if (!values.length) return { min: 0, max: 1 };
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) { min -= 1; max += 1; }
  const padding = (max - min) * 0.08;
  return { min: min - padding, max: max + padding };
}

function linePath(series, key, x, y) {
  let active = false;
  return series.map((row, index) => {
    const value = Number(row[key]);
    if (!Number.isFinite(value)) { active = false; return ""; }
    const command = active ? "L" : "M";
    active = true;
    return `${command}${x(index).toFixed(2)},${y(value).toFixed(2)}`;
  }).join(" ");
}

function tickIndexes(count, desired = 6) {
  if (count <= desired) return Array.from({ length: count }, (_, index) => index);
  const ticks = new Set();
  for (let index = 0; index < desired; index += 1) ticks.add(Math.round(index * (count - 1) / (desired - 1)));
  return [...ticks].sort((left, right) => left - right);
}

function lineChart(el, series, configs, options = {}) {
  const available = configs.filter((config) => series.some((row) => Number.isFinite(Number(row[config.key]))));
  if (!series?.length || !available.length) {
    el.innerHTML = "<div class='empty'>暂无图表数据</div>";
    return;
  }
  const width = 920;
  const height = 320;
  const pad = { left: 70, right: 18, top: 18, bottom: 42 };
  const range = bounds(series, available.map((config) => config.key));
  const x = (index) => pad.left + (index / Math.max(series.length - 1, 1)) * (width - pad.left - pad.right);
  const y = (value) => pad.top + (1 - (value - range.min) / (range.max - range.min)) * (height - pad.top - pad.bottom);
  const xTicks = tickIndexes(series.length);
  const yTicks = Array.from({ length: 5 }, (_, index) => range.max - index * (range.max - range.min) / 4);
  const grid = yTicks.map((value) => `<line class="grid-line" x1="${pad.left}" x2="${width - pad.right}" y1="${y(value)}" y2="${y(value)}" /><text class="label" text-anchor="end" x="${pad.left - 8}" y="${y(value) + 4}">${fmt(value, 3)}</text>`).join("");
  const dates = xTicks.map((index) => `<line class="grid-line vertical" x1="${x(index)}" x2="${x(index)}" y1="${pad.top}" y2="${height - pad.bottom}" /><text class="label" text-anchor="middle" x="${x(index)}" y="${height - 14}">${escapeHtml(series[index].trade_date || "")}</text>`).join("");
  const zero = range.min <= 0 && range.max >= 0 ? `<line class="zero-axis" x1="${pad.left}" x2="${width - pad.right}" y1="${y(0)}" y2="${y(0)}" />` : "";
  const paths = available.map((config, index) => `<path d="${linePath(series, config.key, x, y)}" fill="none" stroke="${config.color || palette[index % palette.length]}" stroke-width="${config.emphasis ? 2.7 : 1.45}" ${config.dash ? 'stroke-dasharray="5 3"' : ""} />`).join("");
  const legend = available.map((config, index) => `<span><i style="background:${config.color || palette[index % palette.length]}"></i>${escapeHtml(config.label)}</span>`).join("");
  let drawdownAnnotation = "";
  const drawdown = options.drawdown;
  if (drawdown) {
    const dateIndex = (date) => {
      if (!date) return -1;
      const normalized = String(date).replaceAll("-", "");
      return series.findIndex((row) => String(row.trade_date || "").replaceAll("-", "") === normalized);
    };
    const peakIndex = dateIndex(drawdown.peakDate);
    const troughIndex = dateIndex(drawdown.troughDate);
    const recoveryIndex = dateIndex(drawdown.recoveryDate);
    if (peakIndex >= 0 && troughIndex >= peakIndex) {
      const left = x(peakIndex);
      const right = x(troughIndex);
      const marker = (index, color, label) => index < 0 ? "" : `<line class="drawdown-marker" x1="${x(index)}" x2="${x(index)}" y1="${pad.top}" y2="${height - pad.bottom}" stroke="${color}" /><circle cx="${x(index)}" cy="${y(Number(series[index].Cumulative_IC))}" r="4" fill="${color}" /><text class="drawdown-label" x="${x(index)}" y="${Math.max(pad.top + 12, y(Number(series[index].Cumulative_IC)) - 9)}" text-anchor="middle">${label}</text>`;
      drawdownAnnotation = `<rect class="drawdown-band" x="${left}" y="${pad.top}" width="${Math.max(right - left, 1)}" height="${height - pad.top - pad.bottom}" />${marker(peakIndex, "#58748a", "峰值")}${marker(troughIndex, "#a15d54", "谷底")}${marker(recoveryIndex, "#5f7659", "恢复")}`;
    }
  }  el.innerHTML = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
    ${grid}${dates}<line class="axis" x1="${pad.left}" x2="${pad.left}" y1="${pad.top}" y2="${height - pad.bottom}" />
    <line class="axis" x1="${pad.left}" x2="${width - pad.right}" y1="${height - pad.bottom}" y2="${height - pad.bottom}" />
    ${zero}${drawdownAnnotation}${paths}<line class="chart-guide hidden" x1="0" x2="0" y1="${pad.top}" y2="${height - pad.bottom}" />
    <rect class="chart-hit" x="${pad.left}" y="${pad.top}" width="${width - pad.left - pad.right}" height="${height - pad.top - pad.bottom}" fill="transparent" />
  </svg><div class="chart-tooltip hidden"></div><div class="legend">${legend}</div>`;

  const hit = el.querySelector(".chart-hit");
  const guide = el.querySelector(".chart-guide");
  const tooltip = el.querySelector(".chart-tooltip");
  const showPoint = (event) => {
    const rect = hit.getBoundingClientRect();
    const fraction = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    const index = Math.round(fraction * Math.max(series.length - 1, 0));
    const previous = series[Math.max(0, index - 1)];
    const values = available.map((config) => {
      const value = Number(series[index][config.key]);
      const prior = Number(previous[config.key]);
      const delta = index && Number.isFinite(value) && Number.isFinite(prior) ? value - prior : null;
      return `<tr><td><i style="background:${config.color}"></i>${escapeHtml(config.label)}</td><td>${escapeHtml(fmt(value, 5))}</td><td>${delta === null ? "--" : escapeHtml(fmt(delta, 5))}</td></tr>`;
    }).join("");
    guide.setAttribute("x1", x(index));
    guide.setAttribute("x2", x(index));
    guide.classList.remove("hidden");
    tooltip.innerHTML = `<strong>${escapeHtml(series[index].trade_date || "")}</strong><table><thead><tr><th>序列</th><th>数值</th><th>较前期</th></tr></thead><tbody>${values}</tbody></table>`;
    tooltip.style.left = `${Math.min(Math.max(event.clientX - rect.left + 12, 8), rect.width - 230)}px`;
    tooltip.style.top = "12px";
    tooltip.classList.remove("hidden");
  };
  hit.addEventListener("mousemove", showPoint);
  hit.addEventListener("mouseleave", () => { guide.classList.add("hidden"); tooltip.classList.add("hidden"); });
}
function barChart(el, series, key, labelKey = "period") {
  if (!series?.length || !key) {
    el.innerHTML = "<div class='empty'>暂无图表数据</div>";
    return;
  }
  const width = 760;
  const height = 220;
  const pad = { left: 62, right: 16, top: 18, bottom: 42 };
  const values = series.map((row) => Number(row[key])).filter(Number.isFinite);
  if (!values.length) { el.innerHTML = "<div class='empty'>暂无图表数据</div>"; return; }
  const min = Math.min(0, ...values);
  const max = Math.max(0, ...values);
  const y = (value) => pad.top + (1 - (value - min) / Math.max(max - min, 1e-9)) * (height - pad.top - pad.bottom);
  const zeroY = y(0);
  const xWidth = (width - pad.left - pad.right) / series.length;
  const yTicks = Array.from({ length: 5 }, (_, index) => max - index * (max - min) / 4);
  const grid = yTicks.map((value) => `<line class="grid-line" x1="${pad.left}" x2="${width - pad.right}" y1="${y(value)}" y2="${y(value)}" /><text class="label" text-anchor="end" x="${pad.left - 8}" y="${y(value) + 4}">${fmt(value, 3)}</text>`).join("");
  const labels = tickIndexes(series.length).map((index) => `<text class="label" text-anchor="middle" x="${pad.left + (index + .5) * xWidth}" y="${height - 14}">${escapeHtml(series[index][labelKey] ?? "")}</text>`).join("");
  const bars = series.map((row, index) => {
    const value = Number(row[key]) || 0;
    const top = Math.min(y(value), zeroY);
    const heightValue = Math.max(Math.abs(zeroY - y(value)), 1);
    const color = value >= 0 ? "#365b75" : "#8a5c56";
    return `<rect x="${pad.left + index * xWidth + 3}" y="${top}" width="${Math.max(xWidth - 6, 2)}" height="${heightValue}" fill="${color}"><title>${escapeHtml(row[labelKey] ?? "")}：${fmt(value, 5)}</title></rect>`;
  }).join("");
  el.innerHTML = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
    ${grid}<line class="zero-axis" x1="${pad.left}" x2="${width - pad.right}" y1="${zeroY}" y2="${zeroY}" />${bars}${labels}
  </svg>`;
}

function heatColor(value, maximum) {
  if (!Number.isFinite(value)) return "#f2f4f5";
  const strength = Math.min(Math.abs(value) / Math.max(maximum, 1e-12), 1);
  if (value >= 0) return `rgb(${Math.round(242 - 104 * strength)}, ${Math.round(247 - 63 * strength)}, ${Math.round(243 - 98 * strength)})`;
  return `rgb(${Math.round(249 - 76 * strength)}, ${Math.round(242 - 105 * strength)}, ${Math.round(240 - 111 * strength)})`;
}

function matrixHeatmap(title, rowLabels, columnLabels, matrix) {
  const maximum = Math.max(1e-12, ...matrix.flat().filter(Number.isFinite).map(Math.abs));
  const header = columnLabels.map((label) => `<th>${escapeHtml(label)}</th>`).join("");
  const rows = matrix.map((values, rowIndex) => `<tr><th>${escapeHtml(rowLabels[rowIndex])}</th>${values.map((value) => `<td title="${escapeHtml(fmt(value, 6))}" style="background:${heatColor(value, maximum)}">${escapeHtml(fmt(value, 4))}</td>`).join("")}</tr>`).join("");
  return `<section class="heatmap-block"><h4>${escapeHtml(title)}</h4><div class="table-wrap"><table class="heatmap-table"><thead><tr><th></th>${header}</tr></thead><tbody>${rows}</tbody></table></div></section>`;
}

function renderIcHeatmaps(monthly, yearly) {
  const years = [...new Set(monthly.map((row) => String(row.year_month).slice(0, 4)))].sort();
  const months = Array.from({ length: 12 }, (_, index) => String(index + 1).padStart(2, "0"));
  const monthlyValues = (key) => years.map((year) => months.map((month) => monthly.find((row) => String(row.year_month) === `${year}-${month}`)?.[key] ?? null));
  $("monthlyIcHeatmap").innerHTML = matrixHeatmap("Rank IC 均值", years, months, monthlyValues("rank_ic_mean")) + matrixHeatmap("ICIR", years, months, monthlyValues("icir"));
  const yearColumns = yearly.map((row) => String(row.year));
  $("yearlyIcHeatmap").innerHTML = matrixHeatmap("年度指标", ["Rank IC 均值", "ICIR"], yearColumns, [yearly.map((row) => row.rank_ic_mean), yearly.map((row) => row.icir)]);
}
function integerInput(id, minimum, maximum) {
  const value = Number($(id).value);
  if (!Number.isInteger(value) || value < minimum || value > maximum) throw new Error(`${$(id).previousSibling?.textContent || id} 必须在 ${minimum} 到 ${maximum} 之间`);
  return value;
}

function analysisParams() {
  const start = $("startInput").value.trim();
  const end = $("endInput").value.trim();
  if (!/^\d{8}$/.test(start) || !/^\d{8}$/.test(end) || start > end) throw new Error("日期需为 YYYYMMDD，且开始日期不得晚于结束日期");
  return {
    factor: $("factorSelect").value,
    version: $("versionInput").value.trim() || "v1",
    start,
    end,
    icHolding: integerInput("icHoldingInput", 1, 60),
    icMa: integerInput("icMaInput", 1, 252),
    layers: integerInput("layersInput", 2, 20),
    holding: integerInput("holdingInput", 1, 60),
    industry: $("industryToggle").checked,
  };
}

async function loadHealth() {
  try {
    await api("/api/health");
    $("healthDot").className = "dot ok";
    $("healthText").textContent = "数据库已连接";
  } catch (error) {
    $("healthDot").className = "dot bad";
    $("healthText").textContent = "数据库连接失败";
    showToast(error.message);
  }
}

async function loadOverview() {
  const tables = await api("/api/tables/status");
  renderTable($("tableStatus"), tables.tables, [
    { key: "table", label: "表名", digits: 0 }, { key: "rows", label: "行数", digits: 0 },
    { key: "start_date", label: "开始", digits: 0 }, { key: "end_date", label: "结束", digits: 0 },
  ]);
}
async function loadFactors() {
  const data = await api("/api/factors");
  state.factors = data.factors;
  renderTable($("factorTable"), state.factors, [
    { key: "factor_name", label: "因子" }, { key: "factor_version", label: "版本" },
    { key: "rows", label: "行数", digits: 0 },
    { key: "coverage", label: "覆盖区间", format: (_, row) => `${row.start_date || "--"} ~ ${row.end_date || "--"}` },
    { key: "formula_latex", label: "公式", html: true, format: (raw) => `\\(${escapeHtml(raw)}\\)` },
    { key: "last_created_at", label: "最后写入" },
  ]);
  if (window.MathJax?.typesetPromise) window.MathJax.typesetPromise([$("factorTable")]);
  window.setTimeout(() => { if (window.MathJax?.typesetPromise) window.MathJax.typesetPromise([$("factorTable")]); }, 800);
  const select = $("factorSelect");
  select.innerHTML = state.factors.map((factor) => `<option value="${escapeHtml(factor.factor_name)}" data-version="${escapeHtml(factor.factor_version)}">${escapeHtml(factor.factor_name)} / ${escapeHtml(factor.factor_version)}</option>`).join("");
  const alpha60 = [...select.options].find((option) => option.value === "alpha_60");
  if (alpha60) select.value = "alpha_60";
  $("versionInput").value = select.selectedOptions[0]?.dataset.version || "v1";
}

function renderIc(data) {
  const summary = data.summary;
  const recoveryStatus = { recovered: "已恢复", unrecovered: "未恢复", none: "无回撤" }[summary.max_drawdown_recovery_status] || (summary.max_drawdown_recovery_status || "未恢复");
  $("icWindow").textContent = `${data.effective_window.start_date} - ${data.effective_window.end_date}`;
  metricCards($("icMetrics"), [
    { label: "Rank IC 均值", value: summary.rank_ic_mean }, { label: "Rank IC 标准差", value: summary.rank_ic_std },
    { label: "ICIR", value: summary.icir }, { label: "IC 胜率", value: summary.ic_win_rate },
    { label: "期末累计 IC", value: summary.cumulative_ic_final }, { label: "累计 IC 最大回撤", value: summary.cumulative_ic_max_drawdown },
    { label: "最大回撤比例", value: summary.cumulative_ic_max_drawdown_ratio }, { label: "最大回撤日期", value: summary.max_drawdown_date, digits: 0 },
    { label: `回撤恢复（${recoveryStatus}）`, value: summary.max_drawdown_recovery_days, digits: 0 }, { label: "持有期 / 均线", value: `${summary.holding_period} / ${summary.ic_ma_window}`, digits: 0 },
  ]);
  lineChart($("icSeriesChart"), data.series, [
    { key: "Rank_IC", label: "Rank IC", color: "#365b75" }, { key: "IC_MA", label: "IC 均线", color: "#7a6240", emphasis: true },
  ]);
  lineChart($("cumulativeIcChart"), data.series, [{ key: "Cumulative_IC", label: "累计 IC", color: "#4e6958", emphasis: true }], { drawdown: { peakDate: summary.max_drawdown_peak_date, troughDate: summary.max_drawdown_date, recoveryDate: summary.max_drawdown_recovery_date } });
  barChart($("icDecayChart"), data.ic_decay, "rank_ic_mean", "period");
  renderTable($("icNwTable"), [data.newey_west], [
    { key: "t_stat", label: "NW t 值" }, { key: "p_value", label: "双尾 p 值" }, { key: "lags", label: "滞后阶数", digits: 0 },
    { key: "mean", label: "IC 均值" }, { key: "se", label: "NW 标准误" }, { key: "status", label: "状态", digits: 0 }, { key: "conclusion", label: "结论", digits: 0 },
  ]);
  renderIcHeatmaps(data.monthly, data.yearly);
}

function groupLabel(index, layers) {
  return index === 0 ? "G1 (高因子)" : index === layers - 1 ? `G${layers} (低因子)` : `G${index + 1}`;
}

function quantileConfigs(layers, suffix) {
  const configs = Array.from({ length: layers }, (_, index) => ({
    key: `sum_ret_${index}_${suffix}`, label: groupLabel(index, layers), color: palette[index % palette.length],
  }));
  configs.push({ key: `sum_ret_L-S_${suffix}`, label: "L-S", color: "#1d2935", emphasis: true, dash: true });
  return configs;
}

function renderQuantile(data) {
  const summary = data.summary;
  const layers = Number(summary.layers);
  $("quantileWindow").textContent = `${data.effective_window.start_date} - ${data.effective_window.end_date} / ${layers} 组 / ${summary.holding_period} 日`;
  metricCards($("quantileMetrics"), [
    { label: "等权 L-S 期末累计", value: summary.ls_ew_final }, { label: "市值加权 L-S 期末累计", value: summary.ls_vw_final },
    { label: "等权平均换手", value: summary.avg_turnover_ew }, { label: "市值加权平均换手", value: summary.avg_turnover_vw },
    { label: "等权年化换手", value: summary.ann_turnover_ew }, { label: "市值加权年化换手", value: summary.ann_turnover_vw },
    { label: "预处理后行数", value: summary.rows_after_preprocess, digits: 0 }, { label: "分层后行数", value: summary.rows_after_layering, digits: 0 },
  ]);
  lineChart($("quantileEwChart"), data.equal_weight, quantileConfigs(layers, "ew"));
  lineChart($("quantileVwChart"), data.value_weight, quantileConfigs(layers, "vw"));
  const finalRows = Array.from({ length: layers }, (_, index) => ({
    group: groupLabel(index, layers), equal_weight: data.final_cumulative.equal_weight[String(index)],
    value_weight: data.final_cumulative.value_weight[String(index)],
    turnover: data.layer_turnover.find((row) => Number(row.layer) === index)?.turnover,
  }));
  finalRows.push({ group: "L-S", equal_weight: data.final_cumulative.equal_weight["L-S"], value_weight: data.final_cumulative.value_weight["L-S"], turnover: null });
  renderTable($("quantileFinalTable"), finalRows, [
    { key: "group", label: "分组", digits: 0 }, { key: "equal_weight", label: "等权期末累计" }, { key: "value_weight", label: "市值加权期末累计" }, { key: "turnover", label: "等权单层换手" },
  ]);
  const nwRows = [
    { weighting: "等权 L-S", ...data.newey_west.equal_weight }, { weighting: "市值加权 L-S", ...data.newey_west.value_weight },
  ];
  renderTable($("quantileNwTable"), nwRows, [
    { key: "weighting", label: "组合", digits: 0 }, { key: "t_stat", label: "NW t 值" }, { key: "p_value", label: "单尾 p 值" }, { key: "conclusion", label: "结论", digits: 0 },
  ]);
  lineChart($("turnoverChart"), data.turnover, [
    { key: "turnover_ls_ew", label: "等权 L-S", color: "#365b75", emphasis: true }, { key: "turnover_ls_vw", label: "市值加权 L-S", color: "#7a6240", emphasis: true },
  ]);
  barChart($("layerTurnoverChart"), data.layer_turnover, "turnover", "layer");
  state.industry = data.industry;
  renderIndustry();
}

function renderIndustry() {
  const panel = $("industryPanel");
  if (!state.industry?.enabled) { panel.classList.add("hidden"); return; }
  panel.classList.remove("hidden");
  const select = $("industrySelect");
  const selected = select.value;
  select.innerHTML = state.industry.summary.map((row) => `<option value="${escapeHtml(row.industry_name)}">${escapeHtml(row.industry_name)}</option>`).join("");
  if ([...select.options].some((option) => option.value === selected)) select.value = selected;
  const name = select.value;
  const series = state.industry.series.filter((row) => row.industry_name === name);
  const layers = Number($("layersInput").value);
  lineChart($("industryChart"), series, quantileConfigs(layers, "ew"));
  renderTable($("industrySummaryTable"), state.industry.summary, [
    { key: "industry_name", label: "行业", digits: 0 }, { key: "observations", label: "样本行数", digits: 0 }, { key: "ls_final", label: "等权 L-S 期末累计" },
  ]);
}

async function runAnalysis() {
  const params = analysisParams();
  if (!params.factor) throw new Error("未找到可分析的因子");
  const button = $("runButton");
  button.disabled = true;
  button.textContent = "计算中";
  $("runStatus").textContent = "计算中";
  try {
    await loadOverview(params.start, params.end);
    const common = { factor_version: params.version, start_date: params.start, end_date: params.end };
    const [ic, quantile] = await Promise.all([
      api(`/api/factors/${encodeURIComponent(params.factor)}/ic?${new URLSearchParams({ ...common, holding_period: params.icHolding, ic_ma_window: params.icMa })}`),
      api(`/api/factors/${encodeURIComponent(params.factor)}/quantile?${new URLSearchParams({ ...common, layers: params.layers, holding_period: params.holding, industry_grouping: params.industry })}`),
    ]);
    renderIc(ic);
    renderQuantile(quantile);
    $("runStatus").textContent = "已完成";
  } catch (error) {
    $("runStatus").textContent = "运行失败";
    showToast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "运行分析";
  }
}

$("factorSelect").addEventListener("change", (event) => { $("versionInput").value = event.target.selectedOptions[0]?.dataset.version || "v1"; });
$("industrySelect").addEventListener("change", renderIndustry);
$("runButton").addEventListener("click", runAnalysis);

(async function boot() {
  try {
    await loadHealth();
    await loadFactors();
    await runAnalysis();
  } catch (error) {
    showToast(error.message);
  }
})();