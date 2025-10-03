// app/static/js/actions.js

import { setStatus, refreshDatasetStatus, renderPage, addBuilderRow, buildQueryFromFilters } from "./ui.js";
import { setTableData, setDA, getState, setSelectedCenter, setLastRadius } from "./state.js";
import { API, normalizeSQL } from "./api.js";
import { VirtualTable } from "./virtual_table.js";

/** Convert array rows to objects keyed by column name (map-friendly). */
function rowsToObjects(columns, rows) {
  if (!rows?.length) return [];
  const arrayRows = Array.isArray(rows[0]);
  if (!arrayRows) return rows;
  return rows.map(r => {
    const o = {};
    for (let i = 0; i < columns.length; i++) o[columns[i]] = r[i];
    return o;
  });
}

/* ============================================================
 * Dataset loading
 * ============================================================ */

export async function loadAvailableDatasets() {
  try {
    const select = document.getElementById("dataset-select");
    if (!select) return;

    setStatus("Loading dataset list...");
    const json = await API.listDatasets();
    select.innerHTML = '<option value="">-- Select a dataset --</option>';
    if (json.datasets?.length) {
      json.datasets.forEach((name) => {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name.replace(/_/g, " ").replace(/\w/g, (l) => l.toUpperCase());
        if (name === json.current_default) opt.textContent += " (default)";
        select.appendChild(opt);
      });
    }
    await refreshDatasetStatus();
    setStatus("Dataset list ready.", "success");
  } catch (e) {
    setStatus(e.message || "Failed to load dataset list.", "error");
  }
}

export async function loadPresetDataset() {
  const select = document.getElementById("dataset-select");
  const datasetName = select?.value || "";
  if (!datasetName) {
    setStatus("Choose a dataset from the list.", "warning");
    return;
  }
  setStatus(`Loading preset: ${datasetName}...`);
  try {
    const json = await API.loadPreset(datasetName);
    setTableData(json.columns || [], []);
    renderPage();
    await refreshDatasetStatus();
    setStatus(json.message || `Loaded preset: ${datasetName}`, "success");
  } catch (err) {
    setStatus(err.message || "Preset loading is not configured on the server.", "error");
  }
}

export async function loadCsvById() {
  const inp = document.getElementById("drive-file-id");
  const fid = (inp?.value || "").trim();
  if (!fid) return setStatus("Please paste a Google Drive File ID.", "warning");
  setStatus("Loading CSV from Drive (public)...");
  try {
    const json = await API.loadCsvPublic(fid);

    // Remember for the status bar
    window.localStorage.setItem("last_drive_file_id", fid);

    setTableData(json.columns || [], []);
    renderPage();
    await refreshDatasetStatus();
    setStatus(json.message || "CSV loaded.", "success");
  } catch (e) {
    setStatus(e.message || "Failed to load CSV from Drive.", "error");
  }
}

/* ============================================================
 * SQL querying
 * ============================================================ */

export async function loadSchemaAndInitBuilder() {
  try {
    const cols = await API.columns().catch(() => ({ columns: [] }));
    const columns = cols?.columns || [];
    if (columns.length) {
      setTableData(columns, []);
      renderPage();
      const toggle = document.getElementById("toggle-sql");
      if (toggle && !toggle.checked) addBuilderRow();
    }
  } catch {}
}

export async function runSQL() {
  const ta = document.getElementById("sql-input");
  let sql = ta ? ta.value : "";
  sql = normalizeSQL(sql);

  if (!sql) {
    setStatus("Enter a SELECT query to run.", "warning");
    return;
  }
  if (!/^\s*select\b/i.test(sql)) {
    setStatus("Only SELECT queries are allowed.", "error");
    return;
  }

  setStatus("Running query...");
  try {
    const json = await API.query(sql, { soft_limit: 0 });
    setTableData(json.columns || [], json.rows || []);
    renderPage();
    setStatus(`Fetched ${json.rows?.length ?? 0} rows.`, "success");
  } catch (e) {
    setStatus(e.message || "Query failed.", "error");
  }
}

export async function runSQLPaged() {
  const ta = document.getElementById("sql-input");
  let sql = ta ? ta.value : "";
  sql = normalizeSQL(sql);

  if (!sql) return setStatus("Enter a SELECT query to run.", "warning");
  if (!/^\s*select\b/i.test(sql)) return setStatus("Only SELECT queries are allowed.", "error");

  const resultsEl = document.getElementById("results");
  if (!resultsEl) return;

  resultsEl.innerHTML = "";
  setStatus("Running paged query...");

  try {
    const first = await API.queryPaginated(sql, { offset: 0, limit: 200, want_total: false });
    const columns = first.columns || [];
    if (!columns.length) {
      setTableData([], []);
      renderPage();
      setStatus("No columns returned.", "warning");
      return;
    }

    const fetchPage = async (offset, limit) => {
      const resp = await API.queryPaginated(sql, { offset, limit, want_total: false });
      return { rows: resp.rows || [], hasMore: (resp.rows || []).length === limit };
    };

    new VirtualTable(resultsEl, columns, fetchPage, { rowHeight: 32, pageSize: 200, buffer: 6 });
    setTableData(columns, first.rows || []);
    renderPage();
    setStatus("Paged query ready. Scroll to load more.", "success");
  } catch (e) {
    setStatus(e.message || "Paged query failed.", "error");
  }
}

export async function buildFromFiltersAndRun() {
  try {
    const sql = normalizeSQL(buildQueryFromFilters());
    if (!sql) {
      setStatus("No filters selected. Nothing to run.", "warning");
      return;
    }
    const ta = document.getElementById("sql-input");
    if (ta) ta.value = sql;
    await runSQL();
  } catch (e) {
    setStatus(e.message || "Failed to build query from filters.", "error");
  }
}

/* ============================================================
 * CSV export
 * ============================================================ */

export async function downloadCSV() {
  const ta = document.getElementById("sql-input");
  let sql = ta ? ta.value : "";
  sql = normalizeSQL(sql);

  if (!sql || !/^\s*select\b/i.test(sql)) {
    setStatus("Enter a SELECT to export.", "warning");
    return;
  }
  try {
    setStatus("Exporting CSV...");
    await API.exportCSV(sql, { filename: "query_result.csv", autoDownload: true });
    setStatus("CSV download started.", "success");
  } catch (e) {
    setStatus(e.message || "Export failed.", "error");
  }
}

/* ============================================================
 * Map + radius tools
 * ============================================================ */

function hasCol(columns, name) {
  if (!columns) return false;
  const target = String(name).toLowerCase();
  return columns.some(c => String(c).toLowerCase() === target);
}

function detectLatLonColumns(columns = []) {
  const cf = (s) => (s ? String(s).toLowerCase() : "");
  const candLat = ["latitude", "lat", "y", "lat_deg"];
  const candLon = ["longitude", "lon", "lng", "x", "lon_deg"];
  let lat = null, lon = null;
  for (const c of columns) {
    const k = cf(c);
    if (!lat && candLat.includes(k)) lat = c;
    if (!lon && candLon.includes(k)) lon = c;
  }
  return {
    lat_col: lat || "LATITUDE",
    lon_col: lon || "LONGITUDE",
  };
}

function buildProps(obj, columns, latCol, lonCol) {
  // Filter out lat/lon columns
  const nonLL = columns.filter(c => {
    const k = String(c).toLowerCase();
    return k !== String(latCol).toLowerCase() && k !== String(lonCol).toLowerCase();
  });

  let useCols;

  if (nonLL.length <= 10) {
    useCols = nonLL;
  } else {
    const PRESET = [
      "DAUID", "cluster_id", "DAUIDs", "population_2021",
      "median_age_of_the_population", "employment_rate",
      "average_after_tax_income_in_2020", "average_household_size",
      "total_households", "dwelling_count"
    ];

    const cfMap = Object.keys(obj).reduce((m, k) => (m[k.toLowerCase()] = k, m), {});
    useCols = PRESET
      .map(c => cfMap[String(c).toLowerCase()] || c)
      .filter(c => Object.prototype.hasOwnProperty.call(obj, c));

    if (useCols.length < 3) {
      useCols = nonLL.slice(0, 10);
    }
  }

  const out = {};
  for (const k of useCols) {
    if (Object.prototype.hasOwnProperty.call(obj, k)) {
      out[k] = obj[k];
    }
  }
  return out;
}

export async function showMap() {
  const { currentColumns, fullResult, lastRadiusRows, lastRadiusMeta } = getState() || {};

  const rawRows = Array.isArray(lastRadiusRows) && lastRadiusRows.length
    ? lastRadiusRows
    : (Array.isArray(fullResult) ? fullResult : []);

  const columns = (lastRadiusMeta?.columns && lastRadiusMeta.columns.length)
    ? lastRadiusMeta.columns
    : (currentColumns || []);

  if (!rawRows.length || !columns.length) {
    alert("Run a query first, then click Show Map.");
    return;
  }

  // Convert array-rows to object-rows so the map can find LATITUDE/LONGITUDE by key
  const arrayRows = Array.isArray(rawRows[0]);
  const rows = arrayRows ? rowsToObjects(columns, rawRows) : rawRows;

  const cf = (s) => (s ? String(s).toLowerCase() : "");
  const candLat = ["latitude", "lat", "y", "lat_deg"];
  const candLon = ["longitude", "lon", "lng", "x", "lon_deg"];

  let lat_col = "LATITUDE";
  let lon_col = "LONGITUDE";
  for (const c of columns) {
    const k = cf(c);
    if (candLat.includes(k)) lat_col = c;
    if (candLon.includes(k)) lon_col = c;
  }

  const payload = {
    rows,
    columns,
    lat_col,
    lon_col,
    title: "Result Map"
  };

  if (lastRadiusMeta?.center &&
      Number.isFinite(lastRadiusMeta.center.lat) &&
      Number.isFinite(lastRadiusMeta.center.lon)) {
    payload.center_lat = lastRadiusMeta.center.lat;
    payload.center_lon = lastRadiusMeta.center.lon;
  }
  if (Number.isFinite(lastRadiusMeta?.radius_km)) {
    payload.radius_km = lastRadiusMeta.radius_km;
  }

  const html = await API.mapHTML(payload);
  const w = window.open("", "_blank");
  w.document.open();
  w.document.write(html);
  w.document.close();
}

/* ============================================================
 * Address autocomplete
 * ============================================================ */

// Replace the wireAutocomplete function in your actions.js with this improved version:

// Replace the wireAutocomplete function in your actions.js (around line 213):

// Replace your wireAutocomplete function with this speed-optimized version:

// Replace your wireAutocomplete function with this debug version:

export function wireAutocomplete() {
  const input = document.getElementById("address");
  if (!input) {
    console.warn("Address input not found - skipping autocomplete setup");
    return;
  }

  console.log("🔧 Debug autocomplete initialized for:", input);

  let inFlight = 0;
  let searchTimeout = null;
  const dropdownId = "address-suggest-dropdown";

  const ensureDropdown = () => {
    let dd = document.getElementById(dropdownId);
    if (!dd) {
      dd = document.createElement("div");
      dd.id = dropdownId;
      dd.className = "dropdown";
      dd.style.cssText = `
        position: absolute;
        top: 100%;
        left: 0;
        right: 0;
        background: #2a2a2a;
        color: white;
        border: 1px solid #333;
        border-top: none;
        max-height: 300px;
        overflow-y: auto;
        z-index: 1000;
        box-shadow: 0 6px 20px rgba(0,0,0,0.35);
        border-radius: 0 0 10px 10px;
        display: none;
      `;

      // Ensure parent container is positioned
      if (input.parentElement) {
        input.parentElement.style.position = "relative";
        input.parentElement.appendChild(dd);
        console.log("🔧 Dropdown created and added to:", input.parentElement);
      }
    }
    console.log("🔧 Dropdown element:", dd);
    return dd;
  };

  const render = (items) => {
    console.log("🔧 Render called with items:", items);
    const dd = ensureDropdown();
    dd.innerHTML = "";

    if (!items || items.length === 0) {
      console.log("🔧 No items to render, hiding dropdown");
      dd.style.display = "none";
      return;
    }

    console.log(`🔧 Rendering ${items.length} items`);
    dd.style.display = "block";

    items.slice(0, 6).forEach((item, index) => {
      console.log(`🔧 Rendering item ${index}:`, item);

      const row = document.createElement("div");
      row.className = "dropdown-item";
      row.style.cssText = `
        padding: 12px 16px;
        cursor: pointer;
        border-bottom: 1px solid #333;
        color: white;
        background: transparent;
        transition: background-color 0.15s ease;
        font-size: 14px;
        line-height: 1.4;
      `;

      const displayText = item.label || item.display_name || `${item.lat}, ${item.lon}`;
      row.textContent = displayText;
      console.log(`🔧 Item ${index} text: "${displayText}"`);

      row.addEventListener("mouseenter", () => {
        row.style.backgroundColor = "#0066cc";
        row.style.color = "#fff";
      });

      row.addEventListener("mouseleave", () => {
        row.style.backgroundColor = "transparent";
        row.style.color = "white";
      });

      row.addEventListener("click", () => {
        console.log("🔧 Item clicked:", displayText);
        const shortLabel = item.label || displayText.split(',').slice(0, 2).join(', ');
        input.value = shortLabel;

        if (Number.isFinite(item.lat) && Number.isFinite(item.lon)) {
          setSelectedCenter({
            lat: item.lat,
            lon: item.lon,
            label: shortLabel,
            fullAddress: item.display_name || displayText
          });
        }

        dd.style.display = "none";
      });

      dd.appendChild(row);
    });

    console.log("🔧 Dropdown after rendering:", dd.innerHTML);
    console.log("🔧 Dropdown display style:", dd.style.display);
  };

  // Debug input handler
  input.addEventListener("input", async () => {
    const q = (input.value || "").trim();
    console.log(`🔧 Input event: "${q}"`);

    // Clear previous timeout
    if (searchTimeout) {
      clearTimeout(searchTimeout);
      searchTimeout = null;
    }

    if (!q || q.length < 2) {
      console.log("🔧 Query too short, hiding dropdown");
      ensureDropdown().style.display = "none";
      return;
    }

    searchTimeout = setTimeout(async () => {
      const ticket = ++inFlight;
      console.log(`🔧 [${ticket}] Starting search for: "${q}"`);

      try {
        console.log(`🔧 [${ticket}] Making API call...`);
        const response = await fetch(`/geocode_suggest_ca?q=${encodeURIComponent(q)}&limit=6`);
        console.log(`🔧 [${ticket}] Response status:`, response.status);

        const resp = await response.json();
        console.log(`🔧 [${ticket}] Response data:`, resp);

        if (ticket !== inFlight) {
          console.log(`🔧 [${ticket}] Ignoring old request`);
          return;
        }

        const items = resp?.suggestions || [];
        console.log(`🔧 [${ticket}] Extracted ${items.length} suggestions:`, items);

        if (items.length > 0) {
          console.log(`🔧 [${ticket}] First item:`, items[0]);
        }

        render(items);

      } catch (error) {
        console.error(`🔧 [${ticket}] Search error:`, error);
        if (ticket !== inFlight) return;

        const dd = ensureDropdown();
        dd.innerHTML = `
          <div style="padding: 12px 16px; color: #888; font-style: italic;">
            Error: ${error.message}
          </div>
        `;
        dd.style.display = "block";
      }
    }, 200);
  });

  // Simple focus handler
  input.addEventListener("focus", () => {
    console.log("🔧 Input focused");
    if (input.value.trim().length >= 2) {
      input.dispatchEvent(new Event('input'));
    }
  });

  console.log("🔧 All event listeners attached");
}

/* ============================================================
 * Clustering
 * ============================================================ */

function hasLatLonColumns(columns) {
  if (!columns) return { hasLat: false, hasLon: false };

  const cf = (s) => (s ? String(s).toLowerCase() : "");
  const candLat = ["latitude", "lat", "y", "lat_deg"];
  const candLon = ["longitude", "lon", "lng", "x", "lon_deg"];

  let latCol = null, lonCol = null;

  for (const c of columns) {
    const k = cf(c);
    if (!latCol && candLat.includes(k)) latCol = c;
    if (!lonCol && candLon.includes(k)) lonCol = c;
  }

  return {
    hasLat: !!latCol,
    hasLon: !!lonCol,
    latCol: latCol || "LATITUDE",
    lonCol: lonCol || "LONGITUDE"
  };
}

function getClusteringParams(method) {
  if (method === "population") {
    const targetPop = parseInt(document.getElementById("target-pop")?.value || "1000");
    const tolerance = parseFloat(document.getElementById("tolerance")?.value || "10");

    if (targetPop < 100) {
      setStatus("Target population must be at least 100.", "error");
      return null;
    }

    if (tolerance < 5 || tolerance > 50) {
      setStatus("Tolerance must be between 5% and 50%.", "error");
      return null;
    }

    return {
      target_pop: targetPop,
      tolerance: tolerance / 100
    };

  } else if (method === "kmeans") {
    const numClusters = parseInt(document.getElementById("num-clusters")?.value || "10");

    if (numClusters < 2) {
      setStatus("Number of clusters must be at least 2.", "error");
      return null;
    }

    return {
      n_clusters: numClusters
    };
  }

  setStatus(`Unknown clustering method: ${method}`, "error");
  return null;
}

export function syncClusterPanels() {
  const method = document.getElementById("cluster-method")?.value;
  const populationParams = document.getElementById("population-params");
  const kmeansParams = document.getElementById("kmeans-params");
  const outlierParams = document.getElementById("outlier-params");
  const allowOutliers = document.getElementById("allow-outliers")?.checked;

  if (!populationParams || !kmeansParams) return;

  if (method === "population") {
    populationParams.style.display = "block";
    kmeansParams.style.display = "none";

    if (outlierParams) {
      outlierParams.style.display = allowOutliers ? "block" : "none";
    }
  } else if (method === "kmeans") {
    populationParams.style.display = "none";
    kmeansParams.style.display = "block";
  }
}

export function updateMinClusterPlaceholder() {
  const targetPop = parseInt(document.getElementById("target-pop")?.value || "1000");
  const tolerance = parseInt(document.getElementById("tolerance")?.value || "10");
  const minClusterInput = document.getElementById("min-cluster-size");

  if (minClusterInput && targetPop > 0 && tolerance > 0) {
    const minSuggested = Math.floor(targetPop * (100 - tolerance) / 100);
    minClusterInput.placeholder = `Auto (~${minSuggested})`;
  }
}

function validateClusteringParams(method, rows, columns) {
  const targetPop = parseInt(document.getElementById("target-pop")?.value || "1000");
  const tolerance = parseFloat(document.getElementById("tolerance")?.value || "10") / 100;
  const numClusters = parseInt(document.getElementById("num-clusters")?.value || "10");

  const issues = [];
  const suggestions = [];

  if (method === "population") {
    let popCol = null;
    for (const col of columns) {
      if (col.toLowerCase().includes('pop')) {
        popCol = col;
        break;
      }
    }

    if (!popCol) {
      issues.push("Cannot find population column in data for validation");
      return { issues, suggestions };
    }

    let totalPop = 0;
    let minPop = Infinity;
    let maxPop = -Infinity;
    let validRows = 0;

    const arrayRows = Array.isArray(rows[0]);
    const popIndex = arrayRows ? columns.indexOf(popCol) : null;

    for (const row of rows) {
      const pop = arrayRows ? row[popIndex] : row[popCol];
      const popNum = parseFloat(pop);

      if (!isNaN(popNum) && popNum > 0) {
        totalPop += popNum;
        minPop = Math.min(minPop, popNum);
        maxPop = Math.max(maxPop, popNum);
        validRows++;
      }
    }

    const avgPop = totalPop / validRows;
    const expectedClusters = totalPop / targetPop;

    console.log(`Population analysis: Total=${totalPop.toLocaleString()}, Avg=${avgPop.toFixed(0)}, Min=${minPop}, Max=${maxPop}`);
    console.log(`Target=${targetPop.toLocaleString()}, Expected clusters=${expectedClusters.toFixed(1)}`);

    if (targetPop > totalPop) {
      issues.push(`Target population (${targetPop.toLocaleString()}) exceeds total population (${totalPop.toLocaleString()})`);
      suggestions.push(`Maximum possible target: ${Math.floor(totalPop).toLocaleString()}`);
    }

    if (targetPop > totalPop * 0.9) {
      issues.push(`Target population (${targetPop.toLocaleString()}) is ${(targetPop/totalPop*100).toFixed(1)}% of total - would create only 1 cluster`);
      suggestions.push(`For multiple clusters, try target around ${Math.floor(totalPop/10).toLocaleString()} (creates ~10 clusters)`);
    }

    if (expectedClusters < 2) {
      issues.push(`Would create only ${expectedClusters.toFixed(1)} clusters`);
      suggestions.push(`Reduce target to ${Math.floor(totalPop/5).toLocaleString()} for ~5 clusters, or ${Math.floor(totalPop/10).toLocaleString()} for ~10 clusters`);
    }

    if (expectedClusters > 500) {
      issues.push(`Would create ${Math.floor(expectedClusters)} clusters (too many to manage)`);
      suggestions.push(`Increase target to at least ${Math.floor(totalPop/100).toLocaleString()} for ~100 clusters`);
    }

    if (targetPop < avgPop) {
      issues.push(`Target (${targetPop.toLocaleString()}) is smaller than average area population (${avgPop.toFixed(0)})`);
      suggestions.push(`Consider target of at least ${Math.floor(avgPop * 2).toLocaleString()} to group multiple areas`);
    }

    if (issues.length === 0) {
      console.log(`Parameters look good: ${expectedClusters.toFixed(1)} expected clusters with ${targetPop.toLocaleString()} target population`);
    }

  } else if (method === "kmeans") {
    if (numClusters >= rows.length) {
      issues.push(`Number of clusters (${numClusters}) must be less than number of areas (${rows.length})`);
      suggestions.push(`Use at most ${Math.floor(rows.length/2)} clusters`);
    }

    if (numClusters < 2) {
      issues.push("Need at least 2 clusters for K-means");
      suggestions.push("Use between 5 and 50 clusters");
    }
  }

  return { issues, suggestions };
}

export async function runClustering() {
  const { currentColumns, fullResult } = getState();

  if (!fullResult.length || !currentColumns.length) {
    setStatus("Run a query first to get data for clustering.", "warning");
    return;
  }

  const hasLatLon = hasLatLonColumns(currentColumns);
  if (!hasLatLon.hasLat || !hasLatLon.hasLon) {
    setStatus("Data must have latitude and longitude columns for clustering.", "error");
    return;
  }

  const method = document.getElementById("cluster-method")?.value || "population";

  const validation = validateClusteringParams(method, fullResult, currentColumns);

  if (validation.issues.length > 0) {
    let errorMsg = "Parameter Issues:\n";
    errorMsg += validation.issues.join("\n");

    if (validation.suggestions.length > 0) {
      errorMsg += "\n\nSuggestions:\n";
      errorMsg += validation.suggestions.join("\n");
    }

    setStatus(errorMsg, "error");
    return;
  }

  const params = getClusteringParams(method);
  if (!params) return;

  // Warning for large datasets
  if (fullResult.length > 50000) {
    setStatus("Warning: Large dataset detected. This may take several minutes or cause browser issues.", "warning");
    await new Promise(resolve => setTimeout(resolve, 2000)); // Give user time to read warning
  }

  setStatus("Running clustering algorithm...");

  try {
    const arrayRows = Array.isArray(fullResult[0]);
    const dataRows = arrayRows ? rowsToObjects(currentColumns, fullResult) : fullResult;

    console.log(`Starting clustering: ${method} on ${dataRows.length} areas`);
    console.log("Parameters:", params);

    let result;

    if (dataRows.length > 5000) {
      setStatus(`Clustering ${dataRows.length} areas using chunked approach...`);
      result = await API.clusterChunked(dataRows, currentColumns, method, params);
    } else {
      const clusterData = {
        method: method,
        rows: dataRows,
        columns: currentColumns,
        ...params
      };

      result = await API.cluster(clusterData);
    }

    console.log("=== CLUSTERING RESULT ===");
    console.log("Full result object:", result);
    console.log("result.rows length:", result.rows?.length);
    console.log("result.columns:", result.columns);

    if (result.rows && result.columns && result.rows.length > 0) {
      setDA(result.rows);
      setTableData(result.columns, result.rows);
      renderPage();

      let statusMsg = result.message || `Clustering completed: ${result.rows.length} clusters formed.`;
      let statusType = "success";

      if (result.coverage_analysis) {
        const coverage = result.coverage_analysis;
        const coveragePercent = coverage.area_coverage_percent;

        if (coveragePercent < 50) {
          statusType = "error";
        } else if (coveragePercent < 70) {
          statusType = "warning";
        }

        statusMsg = `${statusMsg}\n\nCLUSTERING COVERAGE ANALYSIS:`;
        statusMsg += `\n• Area Coverage: ${coveragePercent}% (${coverage.clustered_areas.toLocaleString()}/${coverage.total_areas.toLocaleString()} areas)`;
        statusMsg += `\n• Population Coverage: ${coverage.population_coverage_percent}% of total population`;
        statusMsg += `\n• Coverage Quality: ${coverage.coverage_quality}`;
        statusMsg += `\n• Outliers: ${coverage.outlier_areas.toLocaleString()} areas`;

        if (coverage.warnings && coverage.warnings.length > 0) {
          statusMsg += `\n\nWARNINGS:`;
          coverage.warnings.forEach(warning => {
            statusMsg += `\n• ${warning}`;
          });
        }

        if (coverage.suggestions && coverage.suggestions.length > 0 && coveragePercent < 70) {
          statusMsg += `\n\nSUGGESTIONS TO IMPROVE COVERAGE:`;
          coverage.suggestions.slice(0, 3).forEach(suggestion => {
            statusMsg += `\n• ${suggestion}`;
          });
        }

        if (result.performance) {
          const perf = result.performance;
          statusMsg += `\n\nPerformance: ${perf.elapsed_seconds?.toFixed(1)}s`;
        }
      }

      setStatus(statusMsg, statusType);

    } else {
      let errorMsg = "Clustering completed but no clusters were formed.";

      if (result.coverage_analysis) {
        const coverage = result.coverage_analysis;
        errorMsg += `\n\nCoverage: ${coverage.area_coverage_percent}% of areas clustered`;

        if (coverage.warnings && coverage.warnings.length > 0) {
          errorMsg += `\n\nIssues:`;
          coverage.warnings.forEach(warning => {
            errorMsg += `\n• ${warning}`;
          });
        }

        if (coverage.suggestions && coverage.suggestions.length > 0) {
          errorMsg += `\n\nTry these fixes:`;
          coverage.suggestions.slice(0, 2).forEach(suggestion => {
            errorMsg += `\n• ${suggestion}`;
          });
        }
      } else {
        errorMsg += " Try different parameters.";
      }

      setStatus(errorMsg, "warning");
      console.log("Clustering result:", result);

      if (result.summaries) {
        console.log("Cluster summaries:", result.summaries);
      }
      if (result.parameters_used) {
        console.log("Parameters used:", result.parameters_used);
      }
    }
  } catch (error) {
    console.error("Clustering error:", error);
    setStatus(error.message || "Clustering failed.", "error");
  }
}

/* ============================================================
 * Address-based radius filtering
 * ============================================================ */

export async function runAddressRadius() {
  const addrInput = document.getElementById("address");
  const radiusInput = document.getElementById("radius-km");
  const address = addrInput?.value?.trim();
  const km = parseFloat(radiusInput?.value || "0");

  if (!address) {
    setStatus("Please enter an address.", "error");
    return;
  }
  if (!(km > 0)) {
    setStatus("Please enter a positive radius (km).", "error");
    return;
  }

  const { currentColumns, fullResult } = getState();
  if (!fullResult || fullResult.length === 0) {
    setStatus("Run a query first to get data to filter.", "warning");
    return;
  }

  setStatus("Setting up spatial filter...");

  try {
    let center = getState().selectedCenter;

    if (!center) {
      setStatus("Geocoding address...");
      const resp = await API.suggest(address, 1);
      const suggestions = resp?.suggestions || resp?.items || resp?.results || [];
      const best = suggestions[0];

      if (!best) {
        setStatus("No geocoding results found for that address.", "error");
        return;
      }

      const lat = parseFloat(best.lat ?? best.latitude);
      const lon = parseFloat(best.lon ?? best.longitude ?? best.lng);
      if (Number.isNaN(lat) || Number.isNaN(lon)) {
        setStatus("Geocoder returned invalid coordinates.", "error");
        return;
      }
      center = { lat, lon, label: best.display_name || best.label || address };
      setSelectedCenter(center);
    }

    setLastRadius(km);
    setStatus(`Applying spatial filter: ${km} km around ${center.label || (center.lat + "," + center.lon)}...`);

    const hasLatLon = hasLatLonColumns(currentColumns);
    if (!hasLatLon.hasLat || !hasLatLon.hasLon) {
      setStatus("Current data doesn't have latitude/longitude columns for spatial filtering.", "error");
      return;
    }

    try {
      const result = await API.filterRadiusSQL(
        center.lat,
        center.lon,
        km,
        hasLatLon.latCol,
        hasLatLon.lonCol
      );

      if (result.rows && result.columns) {
        setTableData(result.columns, result.rows);

        window.lastRadiusRows = result.rows;
        window.lastRadiusMeta = {
          columns: result.columns,
          center: center,
          radius_km: km
        };

        renderPage();
        setStatus(result.message || `Filtered to ${result.rows.length} rows within ${km} km.`, "success");

        const autoShowMap = true;
        if (autoShowMap) {
          setTimeout(() => showMap(), 500);
        }
      } else {
        setStatus("Spatial filter returned no data.", "warning");
      }
    } catch (filterError) {
      console.error("Spatial filter error:", filterError);
      setStatus("Spatial filtering failed: " + filterError.message, "error");
    }

  } catch (err) {
    console.error("Address geocoding error:", err);
    setStatus("Failed to geocode address: " + err.message, "error");
  }
}