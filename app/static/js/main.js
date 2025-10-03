// app/static/js/main.js
import {
  loadAvailableDatasets, loadPresetDataset, loadCsvById,
  runSQL, buildFromFiltersAndRun,
  runClustering, syncClusterPanels, updateMinClusterPlaceholder,
  showMap, downloadCSV,
  runAddressRadius,
  wireAutocomplete, loadSchemaAndInitBuilder
} from "./actions.js";
import { refreshDatasetStatus, addBuilderRow, renderPage, enableActionButtons, setStatus, wirePaging } from "./ui.js";
import { getState } from "./state.js";

// Event listeners (guarded)
const toggle = document.getElementById("toggle-sql");
if (toggle) {
  toggle.addEventListener("change", (e) => {
    const on = e.target.checked;
    document.getElementById("sql-mode").style.display = on ? "" : "none";
    document.getElementById("query-builder").style.display = on ? "none" : "";
    if (!on && getState().currentColumns.length) {
      document.getElementById("builder-filters").innerHTML = "";
      addBuilderRow();
    }
  });
}

const addFilterBtn = document.getElementById("add-filter");
if (addFilterBtn) addFilterBtn.addEventListener("click", addBuilderRow);

const runBuilderBtn = document.getElementById("run-builder");
if (runBuilderBtn) runBuilderBtn.addEventListener("click", buildFromFiltersAndRun);

const runBtn = document.getElementById("run-btn");
if (runBtn) runBtn.addEventListener("click", runSQL);

const clusterBtn = document.getElementById("cluster-btn");
if (clusterBtn) clusterBtn.addEventListener("click", runClustering);

const mapBtn = document.getElementById("map-btn");
if (mapBtn) mapBtn.addEventListener("click", showMap);

const dlBtn = document.getElementById("download-btn");
if (dlBtn) dlBtn.addEventListener("click", downloadCSV);

const addrRadiusBtn = document.getElementById("radius-btn");
if (addrRadiusBtn) addrRadiusBtn.addEventListener("click", runAddressRadius);

const loadById = document.getElementById("load-by-id");
if (loadById) loadById.addEventListener("click", loadCsvById);

const loadPresetBtn = document.getElementById("load-preset-dataset");
if (loadPresetBtn) loadPresetBtn.addEventListener("click", loadPresetDataset);

const refBtn = document.getElementById("refresh-dataset-info");
if (refBtn) refBtn.addEventListener("click", refreshDatasetStatus);

// Clustering parameter listeners
const methodSelect = document.getElementById("cluster-method");
if (methodSelect) methodSelect.addEventListener("change", syncClusterPanels);

const allowOutliers = document.getElementById("allow-outliers");
if (allowOutliers) allowOutliers.addEventListener("change", syncClusterPanels);

const targetPopInput = document.getElementById("target-pop");
if (targetPopInput) targetPopInput.addEventListener("input", updateMinClusterPlaceholder);

// Error handling for development
window.addEventListener('error', function(e) {
  console.error('JavaScript Error:', e.error);
  console.error('Stack:', e.error?.stack);
  console.error('Line:', e.lineno, 'Column:', e.colno);
});

window.addEventListener('unhandledrejection', function(e) {
  console.error('Unhandled Promise Rejection:', e.reason);
});

// Initial boot
function init() {
  try {
    refreshDatasetStatus();
    loadAvailableDatasets();
    loadSchemaAndInitBuilder();
    syncClusterPanels();
    updateMinClusterPlaceholder();
    enableActionButtons();
    wireAutocomplete();
    wirePaging();
    setStatus("Ready.", "success");
    console.log("UI initialized successfully");
  } catch (error) {
    console.error("Initialization error:", error);
    setStatus("Initialization failed: " + error.message, "error");
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}