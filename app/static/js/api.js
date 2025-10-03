// app/static/js/api.js

const DEFAULT_TIMEOUT_MS = 300_000; // 5 minutes for large datasets
const BASE_URL = ""; // same-origin

function withTimeout(timeoutMs = DEFAULT_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error("Request timed out")), timeoutMs);
  return { signal: controller.signal, done: () => clearTimeout(timer) };
}

async function parseMaybeJSON(res) {
  const text = await res.text();
  try { return JSON.parse(text); } catch { return null; }
}

function errorFromResponse(res, json) {
  const msg = (json && (json.error || json.message)) || ("Request failed (" + res.status + " " + res.statusText + ")");
  const e = new Error(msg);
  e.status = res.status;
  e.payload = json;
  return e;
}

async function postJSON(path, body, opts = {}) {
  const url = BASE_URL + path;
  const timeoutMs = opts.timeoutMs || DEFAULT_TIMEOUT_MS;
  const headers = opts.headers || {};
  const tt = withTimeout(timeoutMs);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: Object.assign({ "Content-Type": "application/json", "Accept": "application/json" }, headers),
      body: JSON.stringify(body || {}),
      signal: tt.signal,
    });
    const json = await parseMaybeJSON(res);
    if (!res.ok || (json && (json.error || json.status === "error"))) throw errorFromResponse(res, json);
    return json || {};
  } finally { tt.done(); }
}

async function getJSON(path, opts = {}) {
  const url = BASE_URL + path;
  const timeoutMs = opts.timeoutMs || DEFAULT_TIMEOUT_MS;
  const headers = opts.headers || {};
  const tt = withTimeout(timeoutMs);
  try {
    const res = await fetch(url, { method: "GET", headers: Object.assign({ "Accept": "application/json" }, headers), signal: tt.signal });
    const json = await parseMaybeJSON(res);
    if (!res.ok || (json && (json.error || json.status === "error"))) throw errorFromResponse(res, json);
    return json || {};
  } finally { tt.done(); }
}

function triggerBlobDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename || "download.bin"; document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

export function normalizeSQL(s) {
  if (!s) return "";
  s = s.replace(/\u00A0|\u2007|\u202F/g, " "); // normalize unicode spaces
  s = s.replace(/[\u2018\u2019]/g, "'").replace(/[\u201C\u201D]/g, '"'); // smart quotes -> ascii
  return s.replace(/\s+/g, " ").trim();
}

export const API = {
  // --- health + metadata ---
  async datasetStatus() { return getJSON("/dataset_status"); },
  async columns() { return getJSON("/columns"); },
  async health() {
    try { return await getJSON("/health"); } catch (e) { return { ok: false, error: e && e.message ? e.message : "unreachable" }; }
  },

  // --- datasets / loading ---
  async listDatasets() {
    const status = await this.datasetStatus().catch(() => ({}));
    return Object.assign({ datasets: [], current_default: null }, status || {});
  },
  async loadCsvPublic(file_id, opts = {}) {
    if (!file_id) throw new Error("Missing file_id");
    return postJSON("/load_csv", { file_id }, opts);
  },
  async loadCsvApi(file_id, opts = {}) {
    if (!file_id) throw new Error("Missing file_id");
    return postJSON("/load_csv_api", { file_id }, opts);
  },
  async loadPreset() {
    throw new Error("Preset datasets are not configured on the server.");
  },

  // --- querying ---
  async query(sql, opts = {}) {
    const payload = { sql: sql };
    if (typeof opts.soft_limit === "number") payload.soft_limit = opts.soft_limit;
    if (opts.timeoutMs) return postJSON("/query", payload, { timeoutMs: opts.timeoutMs });
    return postJSON("/query", payload);
  },
  async queryPaginated(sql, p = {}) {
    const payload = { sql: sql, limit: p.limit || 100, offset: p.offset || 0, want_total: !!p.want_total };
    return postJSON("/query_paginated", payload);
  },
  async queryCount(sql) { return postJSON("/query_count", { sql }); },

  // --- clustering ---
  async cluster(body) {
    return postJSON("/cluster", body, { timeoutMs: 300_000 }); // 5 minute timeout
  },

  // Chunked clustering for large datasets
  async clusterChunked(rows, columns, method, params) {
    const CHUNK_SIZE = 1000; // Reduced chunk size for better memory management
    const session_id = Date.now().toString();

    console.log(`Processing ${rows.length} rows in chunks of ${CHUNK_SIZE}`);

    // Process data in smaller chunks
    for (let i = 0; i < rows.length; i += CHUNK_SIZE) {
      const chunk = rows.slice(i, i + CHUNK_SIZE);
      const is_final = (i + CHUNK_SIZE) >= rows.length;
      const progress = Math.round((i / rows.length) * 100);

      console.log(`Processing chunk ${Math.floor(i / CHUNK_SIZE) + 1}/${Math.ceil(rows.length / CHUNK_SIZE)} (${progress}%)`);

      const chunkData = {
        method: method,
        chunk: chunk,
        is_final: is_final,
        chunk_id: Math.floor(i / CHUNK_SIZE),
        session_id: session_id,
        total_chunks: Math.ceil(rows.length / CHUNK_SIZE),
        ...params
      };

      try {
        const result = await postJSON("/cluster_chunked", chunkData, {
          timeoutMs: 600_000 // 10 minute timeout per chunk
        });

        if (is_final) {
          return result;
        }

        // Small delay to prevent overwhelming the server and browser
        await new Promise(resolve => setTimeout(resolve, 100));

      } catch (error) {
        console.error(`Chunk ${Math.floor(i / CHUNK_SIZE)} failed:`, error);
        throw new Error(`Clustering failed at chunk ${Math.floor(i / CHUNK_SIZE)}: ${error.message}`);
      }
    }
  },

  // --- downloads ---
  async downloadCSV(body = {}, filename = "export.csv") {
    const r = await fetch("/download_csv", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error("Download failed (" + r.status + ")");
    const blob = await r.blob();
    triggerBlobDownload(blob, filename);
  },
  async exportCSV(sql, opts = {}) {
    const filename = opts.filename || "export.csv";
    const body = { sql };
    if (opts.autoDownload) {
      return this.downloadCSV(body, filename);
    }
    return postJSON("/export_csv", body);
  },

  // --- map / spatial ---
  async mapHTML(body = {}) {
    const r = await fetch("/map", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const html = await r.text();
    if (!r.ok) throw new Error("Map failed (" + r.status + ") " + html);
    return html;
  },
  async filterRadius(center_lat, center_lon, radius_km, lat_col = "LATITUDE", lon_col = "LONGITUDE") {
    return postJSON("/filter_radius", { center_lat, center_lon, radius_km, lat_col, lon_col });
  },
  async filterRadiusSQL(center_lat, center_lon, radius_km, lat_col = "LATITUDE", lon_col = "LONGITUDE") {
    return postJSON("/filter_radius_sql", { center_lat, center_lon, radius_km, lat_col, lon_col });
  },

  // --- geocoding ---
  async geocode(q) { return postJSON("/geocode", { q }); },
  async suggest(q, limit = 6) {
    const params = new URLSearchParams({ q: q, limit: String(limit) });
    try {
      // Try Canadian-specific endpoint first
      return await getJSON("/geocode_suggest_ca?" + params.toString());
    } catch (e1) {
      console.warn("Canadian geocode endpoint failed, trying fallback:", e1);
      try {
        // Fallback to original endpoint
        return await getJSON("/geocode_suggest?" + params.toString());
      } catch (e2) {
        console.warn("All geocode endpoints failed:", e2);
        // Return empty suggestions instead of throwing
        return { suggestions: [] };
      }
    }
  },

  // --- generic passthrough ---
  async post(path, body, opts = {}) { return postJSON(path, body, opts); },
};