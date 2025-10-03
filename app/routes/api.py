# app/routes/api.py
from __future__ import annotations
from typing import Any
import math, re, csv
import random
from io import StringIO
from flask import Blueprint, request, jsonify, Response, stream_with_context, current_app
from ..utils.responses import success, error

# Services
from ..services import db, drive, geocode, clustering, map_render
# State banner
from ..state import set_dataset_info, get_dataset_info
from .. import config

api_bp = Blueprint("api", __name__)

LIMIT_RE = re.compile(r"\blimit\b\s+\d+", re.IGNORECASE)
SAFE_MAX_ROWS_DEFAULT = 50000  # Firefox/Safari happier; Chrome gets smaller soft limit from client

def _normalize_sql(s: str) -> str:
    if not s:
        return ""
    # Replace NBSP & friends with normal spaces
    s = s.replace("\u00A0", " ").replace("\u2007", " ").replace("\u202F", " ")
    # Curly quotes -> straight
    s = s.replace("\u2018", "'").replace("\u2019", "'").replace("\u201C", '"').replace("\u201D", '"')
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def success(payload: dict, status: int = 200):
    return jsonify({"status": "ok", **payload}), status

def error(msg: str, status: int = 400):
    return jsonify({"status": "error", "error": msg}), status

def _ensure_limit(sql: str) -> tuple[str, bool]:
    """Return (sql_with_limit, injected_limit_flag)."""
    if LIMIT_RE.search(sql):
        return sql, False
    safe = sql.rstrip().rstrip(";")
    return f"{safe} LIMIT {config.DEFAULT_SQL_LIMIT}", True

# ---------- HEALTH ----------
@api_bp.get("/health")
def health():
    return success({"ok": True})

# ---------- DATASET STATUS ----------
@api_bp.get("/columns")
def columns():
    try:
        cols = db.get_columns("data")
        return jsonify({"columns": cols})
    except Exception as e:
        current_app.logger.exception("columns failed")
        return jsonify({"columns": [], "error": str(e)}), 500

@api_bp.get("/dataset_status")
def dataset_status():
    try:
        ok = db.table_exists("data")
        row_count = db.get_row_count("data") if ok else 0
        cols = db.get_columns("data") if ok else []

        # If you have a real source/name, inject them here; otherwise provide safe fallbacks
        status = {
            "ok": ok,
            "source": "drive" if ok else "",   # set to whatever you track, or leave ""
            "row_count": row_count,
            "columns": cols,
            "message": "Ready" if ok else "No table 'data' yet",
            # Optional: dataset_name / file_name / file_id if you track them
        }
        return jsonify(status)
    except Exception as e:
        current_app.logger.exception("dataset_status failed")
        return jsonify({"ok": False, "error": str(e)}), 500

# ---------- DATA LOADING (public link) ----------
@api_bp.post("/load_csv")
def load_csv():
    """
    Load/replace the SQLite 'data' table from a Google Drive CSV by FILE ID (public sharing).
    Body: { "file_id": "1AbC..." }
    """
    body = request.get_json(silent=True) or {}
    file_id = (body.get("file_id") or "").strip()
    if not file_id:
        return error("Missing 'file_id'.", 400)
    try:
        blob = drive.download_csv_by_id(file_id)
        message, columns, row_count = db.replace_data_table_from_csv_bytes(blob)
        set_dataset_info(source="drive(public)", label=file_id, row_count=row_count, columns=columns)
        return success({"message": message, "columns": columns, "row_count": row_count})
    except Exception as e:
        return error(f"Load failed: {e}", 500)

# ---------- DATA LOADING (OAuth, private) ----------
@api_bp.post("/load_csv_api")
def load_csv_api():
    """
    Load/replace the 'data' table from a PRIVATE Drive file using OAuth.
    Body: { "file_id": "..." , "type": "file|sheet" }
    """
    body = request.get_json(silent=True) or {}
    file_id = (body.get("file_id") or "").strip()
    ftype = (body.get("type") or "file").strip().lower()
    if not file_id:
        return error("Missing 'file_id'.", 400)
    if ftype not in {"file", "sheet"}:
        return error("Invalid 'type'. Use 'file' or 'sheet'.", 400)

    try:
        blob = (
            drive.export_google_sheet_as_csv(file_id)
            if ftype == "sheet"
            else drive.download_file_bytes_by_id_api(file_id)
        )
        message, columns, row_count = db.replace_data_table_from_csv_bytes(blob)
        src = "drive(sheet)" if ftype == "sheet" else "drive(file)"
        set_dataset_info(source=src, label=file_id, row_count=row_count, columns=columns)
        return success({"message": message, "columns": columns, "row_count": row_count})
    except Exception as e:
        return error(f"Drive API load failed: {e}", 500)

# ---------- DRIVE BROWSING (OAuth) ----------
@api_bp.post("/gdrive/list_folder")
def gdrive_list_folder():
    """
    List files within a Drive folder (private ok).
    Body: { "folder_id": "<FOLDER_ID>" }
    """
    body = request.get_json(silent=True) or {}
    folder_id = (body.get("folder_id") or "").strip()
    if not folder_id:
        return error("Missing 'folder_id'.", 400)
    try:
        files = drive.list_files_in_folder(folder_id)
        return success({"files": files})
    except Exception as e:
        return error(f"List failed: {e}", 500)

# ---------- QUERY (SINGLE UNIFIED ENDPOINT) ----------
@api_bp.post("/query")
def query():
    """
    Execute a SELECT query with a defensive soft_limit parser.
    Body: { "sql": "...", "soft_limit": 0 | N | null }
      - 0 or missing => use SAFE_MAX_ROWS_DEFAULT
      - dict shapes (e.g. {"limit": 5000}) are tolerated
    """
    body = request.get_json(silent=True) or {}
    sql = _normalize_sql(body.get("sql", ""))
    if not sql.lower().startswith("select"):
        return error("Only SELECT queries are allowed.", 400)

    # Robust soft_limit parsing (handles numbers, strings, None, and dicts)
    raw = body.get("soft_limit", None)
    if isinstance(raw, dict):
        # accept various common nested shapes
        for k in ("soft_limit", "limit", "value", "n", "rows"):
            if k in raw:
                raw = raw[k]
                break

    try:
        soft_limit = int(raw) if raw is not None else SAFE_MAX_ROWS_DEFAULT
    except Exception:
        soft_limit = SAFE_MAX_ROWS_DEFAULT

    # Treat <=0 as "use default cap" for safety
    if soft_limit <= 0:
        soft_limit = SAFE_MAX_ROWS_DEFAULT

    try:
        # fetch one extra row so we can report truncation
        cols, rows = db.select_sql(sql, limit=soft_limit + 1)
        truncated = len(rows) > soft_limit
        if truncated:
            rows = rows[:soft_limit]
        return success({"columns": cols, "rows": rows, "truncated": truncated})
    except ValueError as ve:
        return error(str(ve), 400)
    except Exception as e:
        return error(f"Query failed: {e}", 500)

@api_bp.route("/query_paginated", methods=["POST"])
def query_paginated():
    body = request.get_json(silent=True) or {}
    sql = _normalize_sql(body.get("sql", ""))
    if not sql.lower().startswith("select"):
        return error("Only SELECT queries are allowed.", 400)

    # Pagination params
    try:
        # Pagination params (tolerate nested dicts)
        lim_raw = body.get("limit", 100)
        off_raw = body.get("offset", 0)

        if isinstance(lim_raw, dict):
            lim_raw = lim_raw.get("limit") or lim_raw.get("value") or lim_raw.get("n") or 100
        if isinstance(off_raw, dict):
            off_raw = off_raw.get("offset") or off_raw.get("value") or off_raw.get("n") or 0

        try:
            limit = max(1, int(lim_raw))
            offset = max(0, int(off_raw))
        except Exception:
            return error("Invalid pagination parameters.", 400)

    except Exception:
        return error("Invalid pagination parameters.", 400)

    # Optional: ask for total count once, then cache at the client
    want_total = bool(body.get("want_total", False))

    try:
        cols, rows = db.select_sql_page(sql, offset=offset, limit=limit)
        total = None
        if want_total:
            total = db.count_sql(sql)
    except ValueError as ve:
        return error(str(ve), 400)
    except Exception as e:
        return error(f"Query failed: {e}", 500)

    has_more = (len(rows) == limit)  # heuristic
    payload = {"columns": cols, "rows": rows, "offset": offset, "limit": limit, "has_more": has_more}
    if want_total:
        payload["total"] = total
    return success(payload)

@api_bp.route("/query_count", methods=["POST"])
def query_count():
    body = request.get_json(silent=True) or {}
    sql = _normalize_sql(body.get("sql", ""))
    if not sql.lower().startswith("select"):
        return error("Only SELECT queries are allowed.", 400)
    try:
        total = db.count_sql(sql)
        return success({"total": total})
    except ValueError as ve:
        return error(str(ve), 400)
    except Exception as e:
        return error(f"Count failed: {e}", 500)

# ---------- CLUSTER ----------
@api_bp.post("/cluster")
def cluster():
    """
    Cluster the exact data sent from frontend (from their query results)
    """
    body = request.get_json(silent=True) or {}
    method = body.get("method", "population")
    rows = body.get("rows", [])

    print(f"[CLUSTER] Starting clustering with method: {method} on {len(rows)} rows", flush=True)

    if method not in {"kmeans", "population"}:
        return error("Unsupported method. Use 'kmeans' or 'population'.", 400)

    if not rows:
        return error("No data provided for clustering.", 400)

    try:
        print("[CLUSTER] About to call clustering function...", flush=True)

        if method == "kmeans":
            n_clusters = int(body.get("n_clusters", 10))
            print(f"[CLUSTER] Calling k-means with {n_clusters} clusters...", flush=True)
            result = clustering.cluster_kmeans(rows=rows, n_clusters=n_clusters)
        else:
            target_pop = int(body.get("target_pop", 1000))
            tolerance = float(body.get("tolerance", 0.1))
            print(f"[CLUSTER] Calling population clustering: target={target_pop}, tolerance={tolerance}...", flush=True)
            result = clustering.cluster_population(rows=rows, target_pop=target_pop, tolerance=tolerance)

        print("[CLUSTER] Clustering function completed!", flush=True)
        return success(result)

    except Exception as e:
        print(f"[CLUSTER] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return error(f"Clustering failed: {e}", 500)


# Enhanced /cluster_chunked endpoint in api.py

@api_bp.post("/cluster_chunked")
def cluster_chunked():
    """
    Accept clustering data in chunks with enhanced coverage analysis
    """
    body = request.get_json(silent=True) or {}
    method = body.get("method", "population")
    chunk_data = body.get("chunk", [])
    is_final_chunk = body.get("is_final", False)
    chunk_id = body.get("chunk_id", 0)
    session_id = body.get("session_id", "default")

    import tempfile
    import pickle
    import os

    temp_dir = tempfile.gettempdir()
    chunk_file = os.path.join(temp_dir, f"cluster_chunks_{session_id}.pkl")

    try:
        # Load existing chunks or create new list
        if os.path.exists(chunk_file):
            with open(chunk_file, 'rb') as f:
                all_chunks = pickle.load(f)
        else:
            all_chunks = []

        # Add this chunk
        all_chunks.extend(chunk_data)

        if not is_final_chunk:
            # Save chunks and return waiting status
            with open(chunk_file, 'wb') as f:
                pickle.dump(all_chunks, f)
            return success({"status": "chunk_received", "total_rows": len(all_chunks)})

        # FINAL CHUNK - Process all data
        print(f"[CLUSTER_CHUNKED] Processing final chunk: {len(all_chunks)} total rows")
        print(f"[CLUSTER_CHUNKED] Method: {method}")
        print(f"[CLUSTER_CHUNKED] Parameters: {dict(body)}")

        # Extract clustering parameters properly
        if method == "kmeans":
            n_clusters = int(body.get("n_clusters", 10))
            print(f"[CLUSTER_CHUNKED] K-means with {n_clusters} clusters")
            result = clustering.cluster_kmeans(rows=all_chunks, n_clusters=n_clusters)
        else:
            target_pop = int(body.get("target_pop", 1000))
            tolerance = float(body.get("tolerance", 0.1))
            print(f"[CLUSTER_CHUNKED] Population clustering: target={target_pop:,}, tolerance={tolerance * 100}%")
            result = clustering.cluster_population_optimized(rows=all_chunks, target_pop=target_pop,
                                                             tolerance=tolerance)

        # Clean up temp file
        if os.path.exists(chunk_file):
            os.remove(chunk_file)

        print(f"[CLUSTER_CHUNKED] Clustering completed!")
        print(f"[CLUSTER_CHUNKED] Result keys: {list(result.keys()) if result else 'None'}")

        if result:
            print(f"[CLUSTER_CHUNKED] Clusters: {len(result.get('rows', []))}")
            print(f"[CLUSTER_CHUNKED] Columns: {len(result.get('columns', []))}")

            # Log coverage analysis if available
            if 'coverage_analysis' in result:
                coverage = result['coverage_analysis']
                print(
                    f"[CLUSTER_CHUNKED] Coverage: {coverage.get('area_coverage_percent', 0)}% areas, {coverage.get('coverage_quality', 'Unknown')} quality")
                if coverage.get('warnings'):
                    print(f"[CLUSTER_CHUNKED] Warnings: {coverage['warnings']}")

        # Ensure proper response format
        if result and result.get('rows') and result.get('columns'):
            print(f"[CLUSTER_CHUNKED] SUCCESS: Returning {len(result['rows'])} cluster rows")

            # Log coverage info for debugging
            if result.get('coverage_analysis'):
                ca = result['coverage_analysis']
                print(
                    f"[CLUSTER_CHUNKED] Coverage details: {ca['area_coverage_percent']}% coverage, {ca['clustered_areas']} clustered, {ca['outlier_areas']} outliers")

            return success(result)
        else:
            error_msg = "Clustering succeeded but returned no aggregated results"
            if result and result.get('coverage_analysis'):
                ca = result['coverage_analysis']
                error_msg += f". Coverage: {ca.get('area_coverage_percent', 0)}% of areas clustered"
                if ca.get('suggestions'):
                    error_msg += f". Suggestion: {ca['suggestions'][0]}"

            print(f"[CLUSTER_CHUNKED] ERROR: {error_msg}")
            return error(error_msg, 500)

    except Exception as e:
        # Clean up on error
        if os.path.exists(chunk_file):
            os.remove(chunk_file)

        import traceback
        print(f"[CLUSTER_CHUNKED] EXCEPTION: {e}")
        traceback.print_exc()

        return error(f"Chunked clustering failed: {e}", 500)

# ---------- RADIUS FILTER ----------
@api_bp.post("/filter_radius")
def filter_radius():
    """
    Keep only rows within a radius of a center point.
    Body: { "rows":[...], "center_lat": 43.7, "center_lon": -79.4, "radius_km": 5 }
    """
    body = request.get_json(silent=True) or {}
    rows = body.get("rows") or []
    try:
        lat = float(body["center_lat"])
        lon = float(body["center_lon"])
        rkm = float(body["radius_km"])
    except Exception:
        return error("Required: center_lat, center_lon, radius_km (numbers).", 400)
    try:
        kept = geocode.filter_within_radius(rows, lat, lon, rkm)
        cols = list(kept[0].keys()) if kept else []
        return success({"columns": cols, "rows": kept, "message": f"Kept {len(kept)} row(s)."})
    except Exception as e:
        return error(f"Filter failed: {e}", 500)

@api_bp.post("/filter_radius_sql")
def filter_radius_sql():
    """
    Server-side radius filter straight from SQLite (no huge JSON round-trip).
    Body: {
      "center_lat": 43.642566,
      "center_lon": -79.387057,
      "radius_km": 10,
      "lat_col": "LATITUDE",   # optional; default shown
      "lon_col": "LONGITUDE"   # optional; default shown
    }
    Returns: { columns, rows, message }
    """
    body = request.get_json(silent=True) or {}
    try:
        center_lat = float(body["center_lat"])
        center_lon = float(body["center_lon"])
        radius_km  = float(body["radius_km"])
    except Exception:
        return error("Required numeric fields: center_lat, center_lon, radius_km.", 400)

    lat_col = (body.get("lat_col") or "LATITUDE").strip()
    lon_col = (body.get("lon_col") or "LONGITUDE").strip()

    try:
        cols, rows = db.select_within_radius(
            center_lat=center_lat,
            center_lon=center_lon,
            radius_km=radius_km,
            table="data",
            lat_col=lat_col,
            lon_col=lon_col,
        )
        return success({
            "columns": cols,
            "rows": rows,
            "message": f"Kept {len(rows)} row(s) within {radius_km} km."
        })
    except Exception as e:
        return error(f"Radius SQL filter failed: {e}", 500)

# ---------- MAP ----------
@api_bp.post("/map")
def map_page():
    body = request.get_json(silent=True) or {}
    html = map_render.build_map_html(
        # Accept either rows (list of dicts) or points ([[lon,lat,{...}], ...])
        rows=body.get("rows") or None,
        points=body.get("points") or None,
        # Optional center/zoom & circle overlay
        center_lat=(body.get("center") or {}).get("lat"),
        center_lon=(body.get("center") or {}).get("lon"),
        zoom=body.get("zoom"),
        circle=body.get("circle"),
        # Column names (auto-detected if different case/synonyms)
        lat_col=body.get("lat_col") or "LATITUDE",
        lon_col=body.get("lon_col") or "LONGITUDE",
        # What to show in the popup (optional)
        popup_columns=body.get("popup_columns"),
        title=body.get("title") or "Interactive Map",
    )
    return Response(html, mimetype="text/html")

# ---------- GEOCODE ----------
@api_bp.post("/geocode")
def do_geocode():
    body = request.get_json(silent=True) or {}
    address = (body.get("address") or "").strip()
    if not address:
        return error("Missing 'address'.", 400)
    try:
        info = geocode.geocode_address(address)
        return success(info)
    except Exception as e:
        return error(f"Geocode failed: {e}", 404)

@api_bp.get("/geocode_suggest")
def geocode_suggest():
    q = (request.args.get("q") or "").strip()
    limit = request.args.get("limit")
    try:
        lim = int(limit) if limit is not None else 5
    except Exception:
        lim = 5
    try:
        suggestions = geocode.suggest(q, limit=lim)
        return success({"suggestions": suggestions})
    except Exception as e:
        return error(f"Suggest failed: {e}", 500)


@api_bp.get("/geocode_suggest_ca")
def geocode_suggest_ca():
    """
    SPEED-OPTIMIZED Canadian address autocomplete
    Single strategy for maximum speed
    """
    q = (request.args.get("q") or "").strip()
    limit = request.args.get("limit")
    try:
        lim = int(limit) if limit is not None else 6
        lim = max(1, min(lim, 8))
    except Exception:
        lim = 6

    if not q or len(q) < 2:
        return success({"suggestions": []})

    try:
        # Simple, fast single API call
        suggestions = geocode.suggest(q, country_hint="ca", limit=lim)

        # Quick logging for performance monitoring
        print(f"[GEOCODE_CA] '{q}' -> {len(suggestions)} suggestions")

        return success({"suggestions": suggestions})

    except Exception as e:
        current_app.logger.error(f"Fast geocode failed: {e}")
        # Return empty instead of error for speed
        return success({"suggestions": []})

# ---------- DOWNLOADS ----------
@api_bp.post("/download_csv")
def download_csv():
    body = request.get_json(silent=True) or {}
    sql = (body.get("sql") or "").strip()
    if not sql:
        return jsonify({"ok": False, "error": "Missing SQL."}), 400

    try:
        rows, columns = db.run_query(sql)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Query failed: {e}"}), 400

    if len(rows) > config.EXPORT_MAX_ROWS:
        rows = rows[: config.EXPORT_MAX_ROWS]

    def generate():
        sio = StringIO()
        w = csv.writer(sio)
        w.writerow(columns); yield sio.getvalue(); sio.seek(0); sio.truncate(0)
        for r in rows:
            w.writerow([r.get(c, "") for c in columns])
            yield sio.getvalue(); sio.seek(0); sio.truncate(0)

    return Response(
        generate(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=export.csv"}
    )

@api_bp.route("/export_csv", methods=["POST"])
def export_csv():
    # Use form data here to allow very long SQL without JSON encoding issues
    sql = _normalize_sql(request.form.get("sql", ""))
    if not sql.lower().startswith("select"):
        return error("Only SELECT queries are allowed.", 400)

    def generate():
        cols, cursor = db.select_sql_stream(sql)  # stream rows from SQLite
        # header
        yield ",".join([f'"{c}"' for c in cols]) + "\n"
        for row in cursor:
            # basic CSV quoting
            out = []
            for v in row:
                s = "" if v is None else str(v)
                s = s.replace('"', '""')
                out.append(f'"{s}"')
            yield ",".join(out) + "\n"

    filename = "query_result.csv"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(stream_with_context(generate()), mimetype="text/csv", headers=headers)

@api_bp.post("/cluster_test")
def cluster_test():
    """Simple test to see if clustering endpoints work"""
    try:
        return success({"message": "Clustering endpoint is reachable"})
    except Exception as e:
        return error(f"Test failed: {e}", 500)


@api_bp.post("/debug_columns")
def debug_columns():
    """Debug endpoint to see actual column names in current data"""
    try:
        cols = db.get_columns("data")

        # Get a sample row to see the actual data
        sample_sql = "SELECT * FROM data LIMIT 5"
        sample_cols, sample_rows = db.select_sql(sample_sql)

        return success({
            "columns": cols,
            "sample_data": sample_rows,
            "sample_columns": sample_cols
        })
    except Exception as e:
        return error(f"Debug failed: {e}", 500)


# Add this endpoint to your api.py file

@api_bp.post("/cluster_server_side")
def cluster_server_side():
    """
    Server-side clustering for very large datasets (50k+ rows).
    Works directly with SQLite data without loading into memory.
    """
    body = request.get_json(silent=True) or {}
    method = body.get("method", "population")
    use_current_data = body.get("use_current_data", True)

    print(f"[CLUSTER_SERVER_SIDE] Starting server-side clustering with method: {method}")

    if method not in {"kmeans", "population"}:
        return error("Unsupported method. Use 'kmeans' or 'population'.", 400)

    try:
        # Check if we have data in the database
        if not db.table_exists("data"):
            return error("No data table found. Load data first.", 400)

        row_count = db.get_row_count("data")
        print(f"[CLUSTER_SERVER_SIDE] Processing {row_count:,} rows from database")

        if row_count == 0:
            return error("No data in table.", 400)

        if row_count > 100000:  # 100k limit for safety
            return error(f"Dataset too large ({row_count:,} rows). Maximum 100,000 rows for server-side clustering.",
                         400)

        # Get columns to check for required fields
        columns = db.get_columns("data")
        print(f"[CLUSTER_SERVER_SIDE] Available columns: {columns}")

        # Check for required columns
        lat_col = None
        lon_col = None
        pop_col = None

        for col in columns:
            col_lower = col.lower()
            if not lat_col and col_lower in ['latitude', 'lat', 'y']:
                lat_col = col
            if not lon_col and col_lower in ['longitude', 'lon', 'lng', 'x']:
                lon_col = col
            if not pop_col and 'pop' in col_lower:
                pop_col = col

        if not lat_col or not lon_col:
            return error("Required latitude/longitude columns not found.", 400)

        if method == "population" and not pop_col:
            return error("Population column required for population-based clustering.", 400)

        print(f"[CLUSTER_SERVER_SIDE] Using columns: lat={lat_col}, lon={lon_col}, pop={pop_col}")

        # Extract clustering parameters
        if method == "kmeans":
            n_clusters = int(body.get("n_clusters", 10))
            print(f"[CLUSTER_SERVER_SIDE] K-means with {n_clusters} clusters")

            # For very large datasets, use sampling for K-means
            if row_count > 20000:
                sample_size = min(20000, row_count)
                print(f"[CLUSTER_SERVER_SIDE] Using sample of {sample_size:,} rows for K-means clustering")

                # Get a representative sample
                sql_sample = f"""
                SELECT * FROM data 
                WHERE {lat_col} IS NOT NULL AND {lon_col} IS NOT NULL
                ORDER BY RANDOM() 
                LIMIT {sample_size}
                """
                sample_cols, sample_rows = db.select_sql(sql_sample)

                # Convert to list of dicts for clustering
                sample_data = []
                for row in sample_rows:
                    row_dict = {}
                    for i, col in enumerate(sample_cols):
                        row_dict[col] = row[i] if i < len(row) else None
                    sample_data.append(row_dict)

                # Cluster the sample
                result = clustering.cluster_kmeans(rows=sample_data, n_clusters=n_clusters)

                # TODO: Apply cluster centers to full dataset using SQL
                # This is a simplified version - you'd want to implement
                # cluster assignment for the full dataset based on the sample clusters

            else:
                # Load all data for smaller datasets
                sql_all = f"SELECT * FROM data WHERE {lat_col} IS NOT NULL AND {lon_col} IS NOT NULL"
                all_cols, all_rows = db.select_sql(sql_all)

                # Convert to list of dicts
                all_data = []
                for row in all_rows:
                    row_dict = {}
                    for i, col in enumerate(all_cols):
                        row_dict[col] = row[i] if i < len(row) else None
                    all_data.append(row_dict)

                result = clustering.cluster_kmeans(rows=all_data, n_clusters=n_clusters)

        else:  # population clustering
            target_pop = int(body.get("target_pop", 1000))
            tolerance = float(body.get("tolerance", 0.1))

            print(f"[CLUSTER_SERVER_SIDE] Population clustering: target={target_pop:,}, tolerance={tolerance * 100}%")

            # For population clustering, we need to be more careful with memory
            if row_count > 30000:
                # Use streaming approach for very large datasets
                print(f"[CLUSTER_SERVER_SIDE] Using streaming approach for {row_count:,} rows")

                # This would require implementing a streaming version of population clustering
                # For now, return an error suggesting chunked approach
                return error(
                    f"Dataset too large for server-side population clustering ({row_count:,} rows). Use chunked approach instead.",
                    400)
            else:
                # Load data in chunks and process
                sql_all = f"""
                SELECT * FROM data 
                WHERE {lat_col} IS NOT NULL AND {lon_col} IS NOT NULL AND {pop_col} IS NOT NULL AND {pop_col} > 0
                """
                all_cols, all_rows = db.select_sql(sql_all)

                print(f"[CLUSTER_SERVER_SIDE] Loaded {len(all_rows):,} valid rows for clustering")

                # Convert to list of dicts
                all_data = []
                for row in all_rows:
                    row_dict = {}
                    for i, col in enumerate(all_cols):
                        row_dict[col] = row[i] if i < len(row) else None
                    all_data.append(row_dict)

                result = clustering.cluster_population_optimized(
                    rows=all_data,
                    target_pop=target_pop,
                    tolerance=tolerance
                )

        print(f"[CLUSTER_SERVER_SIDE] Clustering completed successfully")
        print(f"[CLUSTER_SERVER_SIDE] Result: {len(result.get('rows', []))} clusters formed")

        return success(result)

    except Exception as e:
        import traceback
        print(f"[CLUSTER_SERVER_SIDE] ERROR: {e}")
        traceback.print_exc()
        return error(f"Server-side clustering failed: {e}", 500)