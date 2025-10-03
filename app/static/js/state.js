// app/static/js/state.js
export let fullResult = [];
export let currentColumns = [];
export let currentPage = 1;
export let pageSize = 25;

export let selectedCenter = null;
export let daRows = null;
export let lastRadiusRows = null;
export let lastRadiusMeta = null;
export let lastRadius = null;

export function setTableData(columns, rows) {
  currentColumns = Array.isArray(columns) ? columns : [];
  fullResult = Array.isArray(rows) ? rows : [];
  currentPage = 1;
}

export function setDA(rows) {
  daRows = rows || null;
}

export function setSelectedCenter(obj) {
  selectedCenter = obj;
}

export function setPageSize(n) {
  pageSize = n;
  currentPage = 1;
}

export function setLastRadius(km) {
  lastRadius = km;
}

export function setCurrentPage(page) {
  currentPage = page;
}

export function nextPage() {
  currentPage++;
}

export function prevPage() {
  if (currentPage > 1) currentPage--;
}

export function getState() {
  return {
    fullResult,
    currentColumns,
    currentPage,
    pageSize,
    selectedCenter,
    daRows,
    lastRadiusRows,
    lastRadiusMeta,
    lastRadius
  };
}