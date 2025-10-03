// app/static/js/ui.js

import { getState, currentColumns, setPageSize, setTableData, setCurrentPage, nextPage, prevPage } from "./state.js";
import { API } from "./api.js";

/* ---------- Status + utilities ---------- */

export function setStatus(msg, type = "info") {
  const el = document.getElementById("status");
  if (!el) return;
  el.textContent = msg;
  el.className = "status " + type;
}

export function toCSV(columns, rows) {
  const arrayRows = rows.length && Array.isArray(rows[0]);
  const header = columns.join(",");
  const body = rows
    .map((r) =>
      columns
        .map((c, i) => {
          const v = arrayRows ? r[i] : r[c];
          if (v == null) return "";
          const s = String(v).replaceAll('"', '""');
          return /[",\n]/.test(s) ? `"${s}"` : s;
        })
        .join(",")
    )
    .join("\n");
  return header + "\n" + body;
}

export function enableActionButtons() {
  const { fullResult, currentColumns } = getState();
  const hasRows = fullResult.length > 0 && currentColumns.length > 0;
  const dl = document.getElementById("download-btn");
  const map = document.getElementById("map-btn");
  if (dl) dl.disabled = !hasRows;
  if (map) map.disabled = !hasRows;
}

/* ---------- Dataset banner ---------- */

export async function refreshDatasetStatus() {
  try {
    const res = await fetch("/dataset_status");
    const json = await res.json();

    // Elements
    const bar   = document.getElementById("dataset-status");
    const name  = document.querySelector("#ds-label .ds-name");
    const idEl  = document.querySelector("#ds-label .ds-id");
    const srcEl = document.querySelector("#ds-label .ds-src");
    const rows  = document.getElementById("ds-rows");

    // Derive a friendly dataset label
    const src = json?.source || "";
    const fromServerName =
      json?.dataset_name || json?.file_name ||
      (src === "drive" ? "Drive CSV" :
       src === "local_csv" ? "Local CSV" :
       src || "—");

    const fileIdFromServer = json?.file_id || "";
    const fileId = fileIdFromServer || window.localStorage.getItem("last_drive_file_id") || "";

    if (name)  name.textContent  = `Dataset: ${fromServerName}`;
    if (idEl)  idEl.textContent  = fileId ? ` (${fileId})` : "";
    if (srcEl) srcEl.textContent = src ? ` [${src}]` : "";
    if (rows)  rows.textContent  = `Rows: ${json?.row_count != null ? Number(json.row_count).toLocaleString() : "—"}`;

    if (bar) bar.dataset.ok = json?.ok ? "true" : "false";
  } catch (e) {
    const bar = document.getElementById("dataset-status");
    if (bar) bar.dataset.ok = "false";
  }
}

/* ---------- Query Builder ---------- */

export function addBuilderRow() {
  const wrap = document.getElementById("builder-filters");
  if (!wrap) return;

  const row = document.createElement("div");
  row.className = "builder-row";

  const colSelect = document.createElement("select");
  colSelect.className = "builder-col";
  colSelect.name = "col";
  for (const col of currentColumns) {
    const opt = document.createElement("option");
    opt.value = opt.textContent = col;
    colSelect.appendChild(opt);
  }

  const opSelect = document.createElement("select");
  opSelect.className = "builder-op";
  opSelect.name = "op";
  ["=", "!=", ">", "<", ">=", "<=", "LIKE"].forEach((op) => {
    const opt = document.createElement("option");
    opt.value = opt.textContent = op;
    opSelect.appendChild(opt);
  });

  const input = document.createElement("input");
  input.className = "builder-val";
  input.name = "val";
  input.type = "text";

  const removeBtn = document.createElement("button");
  removeBtn.className = "remove-filter";
  removeBtn.type = "button";
  removeBtn.textContent = "Remove";
  removeBtn.onclick = () => row.remove();

  row.append(colSelect, opSelect, input, removeBtn);
  wrap.appendChild(row);
}

export function buildQueryFromFilters() {
  const filtersHost = document.getElementById("builder-filters");
  if (!filtersHost) {
    return "SELECT * FROM data";
  }

  const sqlLit = (v) => {
    if (v === null || v === undefined) return "NULL";
    const num = typeof v === "number" ? v : Number(v);
    const isNum = Number.isFinite(num) && String(v).trim() !== "";
    if (isNum) return String(num);
    return `'${String(v).replace(/'/g, "''")}'`;
  };

  const rows = Array.from(filtersHost.querySelectorAll(".builder-row"));
  const clauses = [];

  for (const row of rows) {
    const colEl = row.querySelector('[name="col"]');
    const opEl  = row.querySelector('[name="op"]');
    const valEl = row.querySelector('[name="val"]');

    const col = colEl?.value?.trim();
    let op    = opEl?.value?.trim();
    let val   = valEl?.value ?? "";

    if (!col || !op) continue;

    const opMap = {
      equals: "=", eq: "=", "!=": "!=", neq: "!=",
      ">": ">", gt: ">", ">=": ">=", gte: ">=",
      "<": "<", lt: "<", "<=": "<=", lte: "<=",
      contains: "LIKE", notcontains: "NOT LIKE",
      like: "LIKE", notlike: "NOT LIKE",
      "is null": "IS NULL", "is not null": "IS NOT NULL",
      in: "IN", notin: "NOT IN"
    };
    op = opMap[op.toLowerCase?.() || op] || op;

    if (op === "IS NULL" || op === "IS NOT NULL") {
      clauses.push(`${col} ${op}`);
      continue;
    }

    if (op === "IN" || op === "NOT IN") {
      const parts = String(val).split(",").map(s => s.trim()).filter(Boolean);
      if (parts.length === 0) continue;
      const list = parts.map(sqlLit).join(", ");
      clauses.push(`${col} ${op} (${list})`);
      continue;
    }

    if (op === "LIKE" || op === "NOT LIKE") {
      const pattern = `%${String(val).trim()}%`;
      clauses.push(`${col} ${op} ${sqlLit(pattern)}`);
      continue;
    }

    clauses.push(`${col} ${op} ${sqlLit(val)}`);
  }

  const whereSql = clauses.length ? " WHERE " + clauses.join(" AND ") : "";

  const limitEl = document.getElementById("builder-limit");
  let limitSql = "";
  if (limitEl && limitEl.value !== "") {
    const n = parseInt(limitEl.value, 10);
    if (Number.isFinite(n) && n > 0) limitSql = ` LIMIT ${n}`;
  }

  return `SELECT * FROM data${whereSql}${limitSql}`;
}

/* ---------- Table rendering & paging ---------- */

export function renderPage() {
  const { fullResult, currentColumns, currentPage, pageSize } = getState();
  const results = document.getElementById("results");

  if (!currentColumns.length) {
    if (results) results.innerHTML = "<p>No results.</p>";
    const rc = document.getElementById("row-count"); if (rc) rc.textContent = "0";
    const tp = document.getElementById("total-pages"); if (tp) tp.textContent = "1";
    const pi = document.getElementById("page-input"); if (pi) pi.value = 1;
    enableActionButtons();
    return;
  }

  const start = (currentPage - 1) * pageSize;
  const slice = fullResult.slice(start, start + pageSize);
  const arrayRows = slice.length && Array.isArray(slice[0]);

  let html = "<table><thead><tr>";
  for (const c of currentColumns) html += `<th>${c}</th>`;
  html += "</tr></thead><tbody>";

  for (const row of slice) {
    html += "<tr>";
    for (let i = 0; i < currentColumns.length; i++) {
      const c = currentColumns[i];
      const v = arrayRows ? row[i] : row[c];
      html += `<td>${v == null ? "" : v}</td>`;
    }
    html += "</tr>";
  }
  html += "</tbody></table>";
  if (results) results.innerHTML = html;

  const totalPages = Math.max(1, Math.ceil(fullResult.length / pageSize));
  const tp = document.getElementById("total-pages"); if (tp) tp.textContent = totalPages;
  const pi = document.getElementById("page-input"); if (pi) pi.value = currentPage;
  const prev = document.getElementById("prev-page"); if (prev) prev.disabled = currentPage <= 1;
  const next = document.getElementById("next-page"); if (next) next.disabled = currentPage >= totalPages;
  const rc = document.getElementById("row-count"); if (rc) rc.textContent = fullResult.length;

  enableActionButtons();
}

export function wirePaging() {
  const ps = document.getElementById("page-size");
  if (ps) ps.addEventListener("change", (e) => {
    setPageSize(parseInt(e.target.value, 10));
    renderPage();
  });

  const prev = document.getElementById("prev-page");
  if (prev) prev.addEventListener("click", () => {
    prevPage();
    renderPage();
  });

  const next = document.getElementById("next-page");
  if (next) next.addEventListener("click", () => {
    const { fullResult, pageSize, currentPage } = getState();
    const totalPages = Math.max(1, Math.ceil(fullResult.length / pageSize));
    if (currentPage < totalPages) {
      nextPage();
      renderPage();
    }
  });

  const pi = document.getElementById("page-input");
  if (pi) pi.addEventListener("change", (e) => {
    const { fullResult, pageSize } = getState();
    const totalPages = Math.max(1, Math.ceil(fullResult.length / pageSize));
    let val = parseInt(e.target.value, 10);
    if (!Number.isInteger(val) || val < 1) val = 1;
    if (val > totalPages) val = totalPages;
    setCurrentPage(val);
    renderPage();
  });
}