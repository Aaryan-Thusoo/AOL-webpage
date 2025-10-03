# app/services/db.py
from __future__ import annotations

import math
import os
import re
import sqlite3
from contextlib import contextmanager
from typing import Generator, Iterable, Iterator, List, Sequence, Tuple

# Expect DB_PATH to be configured elsewhere (e.g., app/config.py)
# and imported here. If you keep config in a different place, adjust import.
try:
    from ..config import DB_PATH  # type: ignore
except Exception as e:
    raise RuntimeError(
        "DB_PATH is not configured. Ensure app/config.py exports DB_PATH."
    ) from e


# -----------------------------
# Connection management
# -----------------------------

def _apply_pragmas(conn: sqlite3.Connection) -> None:
    """
    Apply pragmatic tuning for read-heavy analytics and stability.
    Safe for both read-only and read-write connections.
    """
    # Most of these are no-ops if already set; errors are ignored on older SQLite.
    pragmas = [
        # Write-ahead logging improves concurrency; harmless in RO mode.
        ("PRAGMA journal_mode=WAL;", None),
        # Reasonable durability without being painfully slow.
        ("PRAGMA synchronous=NORMAL;", None),
        # Keep temporary objects in RAM.
        ("PRAGMA temp_store=MEMORY;", None),
        # Grow cache in KiB; negative = size in KB of cache memory
        # Here ~200MB cache (adjust to taste).
        ("PRAGMA cache_size=-200000;", None),
        # Make LIKE case-insensitive (common expectation for user queries).
        ("PRAGMA case_sensitive_like=OFF;", None),
        # Enable foreign keys if you use them (noop otherwise).
        ("PRAGMA foreign_keys=ON;", None),
    ]
    cur = conn.cursor()
    for stmt, param in pragmas:
        try:
            cur.execute(stmt) if param is None else cur.execute(stmt, param)
        except Exception:
            # Silently ignore pragma errors on platforms that don't support them.
            pass
    cur.close()


def _make_connection(readonly: bool = False) -> sqlite3.Connection:
    """
    Create a SQLite connection with row/ text factories and pragmatic defaults.
    If readonly=True, opens the DB in read-only mode (safer for public query endpoints).
    """
    if not DB_PATH:
        raise RuntimeError("DB_PATH is empty.")

    # Support both filesystem path and URI mode.
    path = DB_PATH
    uri = False

    if readonly:
        # Use URI to force read-only mode; protects against stray writes.
        # See https://www.sqlite.org/c3ref/open.html#urifilenamesinsqlite
        path = f"file:{os.fspath(DB_PATH)}?mode=ro"
        uri = True

    conn = sqlite3.connect(
        path,
        uri=uri,
        check_same_thread=False,  # allow usage across threads in Flask
    )
    # Return rows as sqlite3.Row to preserve column order and names.
    conn.row_factory = sqlite3.Row
    # Ensure all TEXT comes back as Python str, not bytes.
    conn.text_factory = str

    _apply_pragmas(conn)
    return conn


@contextmanager
def _conn(readonly: bool = False) -> Iterator[sqlite3.Connection]:
    """
    Context-managed connection; commits only for RW connections.
    """
    conn = _make_connection(readonly=readonly)
    try:
        yield conn
        if not readonly:
            conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass


# -----------------------------
# Utilities
# -----------------------------

def _clean_val(v):
    """
    Normalize SQLite values to JSON/CSV friendly Python types.
    - bytes -> utf-8 str (replace errors)
    """
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", errors="replace")
        except Exception:
            return str(v)
    return v


def _append_limit_if_needed(sql: str, limit: int | None) -> str:
    """
    If caller provided a limit and SQL has no explicit LIMIT, append one.
    Very lightweight detector; won't modify subqueries that already have LIMIT.
    """
    if not limit:
        return sql

    # crude but effective: check if there's a top-level LIMIT already
    lowered = sql.lower()
    if " limit " in lowered:
        return sql
    return f"{sql.rstrip().rstrip(';')} LIMIT {int(limit)}"


def _is_select_only(sql: str) -> bool:
    """
    Guardrail: ensure this is a single SELECT statement.
    Not bulletproof, but avoids obvious multi-statement misuse.
    """
    s = sql.strip().strip(";")
    # must start with SELECT (optionally with WITH CTE prefix)
    return bool(re.match(r"^(with\s+.+\)\s*select|select)\b", s, flags=re.IGNORECASE | re.DOTALL))


# -----------------------------
# Public API
# -----------------------------

def select_sql(sql: str, params: Sequence | None = None, limit: int | None = None) -> Tuple[List[str], List[List]]:
    """
    Run a SELECT and materialize up to `limit` rows into memory.
    Returns (columns, rows).
    Use for UI table rendering where you want a capped preview.
    """
    if not _is_select_only(sql):
        raise ValueError("Only SELECT queries are allowed.")

    q = _append_limit_if_needed(sql, limit)
    with _conn(readonly=True) as conn:
        cur = conn.execute(q, params or [])
        cols = [c[0] for c in cur.description]
        out_rows: List[List] = [list(map(_clean_val, r)) for r in cur.fetchall()]
        return cols, out_rows


def select_sql_stream(sql: str, params: Sequence | None = None) -> Tuple[List[str], Iterable[List]]:
    """
    Run a SELECT and return (columns, iterator of rows) for streaming.
    Use this for CSV export so you don't allocate the entire result in memory.
    """
    if not _is_select_only(sql):
        raise ValueError("Only SELECT queries are allowed.")

    conn = _make_connection(readonly=True)
    # We do NOT close the connection here; the caller consumes the generator.
    # Provide a generator that will close the connection when iteration completes.

    cur = conn.execute(sql, params or [])
    cols = [c[0] for c in cur.description]

    def _row_iter() -> Generator[List, None, None]:
        try:
            for r in cur:
                yield [ _clean_val(v) for v in r ]
        finally:
            try:
                cur.close()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass

    return cols, _row_iter()

def select_sql_page(sql: str, offset: int, limit: int):
    """
    Safe pagination: wraps the user's SELECT as a subquery and applies LIMIT/OFFSET.
    Returns (columns, rows).
    """
    if not _is_select_only(sql):
        raise ValueError("Only SELECT queries are allowed.")
    wrapped = f'SELECT * FROM ({sql.rstrip().rstrip(";")}) AS t LIMIT ? OFFSET ?'
    with _conn(readonly=True) as conn:
        cur = conn.execute(wrapped, (int(limit), int(offset)))
        cols = [c[0] for c in cur.description]
        rows = [list(map(_clean_val, r)) for r in cur.fetchall()]
        return cols, rows


def count_sql(sql: str) -> int:
    """
    Return total row count for an arbitrary SELECT by wrapping it.
    """
    if not _is_select_only(sql):
        raise ValueError("Only SELECT queries are allowed.")
    wrapped = f'SELECT COUNT(*) AS n FROM ({sql.rstrip().rstrip(";")}) AS t'
    with _conn(readonly=True) as conn:
        cur = conn.execute(wrapped)
        row = cur.fetchone()
        return int(row[0]) if row else 0


# -----------------------------
# Metadata helpers (optional)
# -----------------------------

def list_tables() -> List[str]:
    """
    Return user tables (excludes sqlite internal tables).
    """
    with _conn(readonly=True) as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
        )
        return [r[0] for r in cur.fetchall()]


def table_info(table_name: str) -> List[Tuple[str, str, int, str, int]]:
    """
    PRAGMA table_info(table) -> list of tuples:
      (name, type, notnull, dflt_value, pk)
    """
    with _conn(readonly=True) as conn:
        cur = conn.execute(f'PRAGMA table_info("{table_name}");')
        # Order: cid, name, type, notnull, dflt_value, pk
        return [(r["name"], r["type"], r["notnull"], r["dflt_value"], r["pk"]) for r in cur.fetchall()]


def get_columns_for_query(sql: str) -> List[str]:
    """
    Prepare the statement and return the column names without executing fully.
    Useful for client previews.
    """
    if not _is_select_only(sql):
        raise ValueError("Only SELECT queries are allowed.")
    with _conn(readonly=True) as conn:
        cur = conn.execute(sql + " LIMIT 0")
        return [c[0] for c in cur.description]


# -----------------------------
# Maintenance helpers (optional)
# -----------------------------

def vacuum_analyze() -> None:
    """
    Reclaim space and refresh statistics (useful after large ingests).
    """
    with _conn(readonly=False) as conn:
        conn.execute("VACUUM;")
        conn.execute("ANALYZE;")
        conn.commit()


import pandas as pd
import sqlite3
from io import BytesIO

def replace_data_table_from_csv_bytes(csv_bytes: bytes):
    """
    Replace (or create) the SQLite 'data' table from CSV bytes.
    Returns: (message: str, columns: list[str], row_count: int)
    """
    # Guard against HTML masquerading as CSV (e.g., Drive virus-scan page)
    sniff = csv_bytes[:200].lstrip().lower()
    if sniff.startswith(b"<!doctype html") or sniff.startswith(b"<html"):
        raise RuntimeError("Not a CSV: received HTML content (Drive interstitial).")

    # Parse CSV
    df = pd.read_csv(BytesIO(csv_bytes))
    if df.empty:
        raise RuntimeError("CSV parsed but is empty.")

    # Require key columns before replacing
    required = {"DAUID", "LATITUDE", "LONGITUDE"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV missing required columns: {missing}")

    # Write to SQLite
    conn = sqlite3.connect(DB_PATH)
    try:
        df.to_sql("data", conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()

    msg = f"Replaced 'data' table with {len(df):,} rows and {len(df.columns)} columns."
    return msg, list(df.columns), len(df)



# app/services/db.py  (add below your existing imports)

import re
import sqlite3
from typing import List, Tuple

from ..config import DB_PATH, MAX_SQL_ROWS, DEFAULT_SQL_LIMIT

_SELECT_RE = re.compile(r"^\s*select\b", re.IGNORECASE | re.DOTALL)

def _clean_select(sql: str) -> str:
    """
    Ensure it's a single SELECT statement. Strip a trailing semicolon.
    """
    if not isinstance(sql, str):
        raise ValueError("SQL must be a string.")
    s = sql.strip().rstrip(";").strip()
    if not _SELECT_RE.match(s):
        raise ValueError("Only SELECT statements are allowed.")
    # Disallow additional semicolons (no multi-statement)
    if ";" in s:
        raise ValueError("Multiple SQL statements are not allowed.")
    return s

def _exec_fetch(conn: sqlite3.Connection, sql: str) -> Tuple[List[str], List[tuple]]:
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description] if cur.description else []
    cur.close()
    return cols, rows

# app/services/db.py
import sqlite3
from ..config import DB_PATH, MAX_SQL_ROWS  # keep your existing import

def run_select(sql: str, soft_limit: int | None = None):
    """
    Execute a SELECT. If soft_limit > 0, wrap with LIMIT. If soft_limit == 0,
    don't add any extra LIMIT. If soft_limit is None, fall back to MAX_SQL_ROWS.
    Returns (columns, rows).
    """
    s = sql.strip().rstrip(";")
    if not s.lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")

    # Decide cap
    if soft_limit is None:
        # safety fallback to MAX_SQL_ROWS (config)
        cap = int(MAX_SQL_ROWS) if MAX_SQL_ROWS else 0
    else:
        cap = int(soft_limit)
        if cap < 0:
            cap = 0

    if cap > 0:
        s = f"SELECT * FROM ({s}) AS sub LIMIT {cap}"

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(s)
        rows = cur.fetchall()
        columns = [d[0] for d in cur.description] if cur.description else []
        cur.close()
    return columns, rows


def run_select_paginated(sql: str, offset: int = 0, limit: int = 200) -> Tuple[List[str], List[tuple]]:
    """
    Execute a SELECT with LIMIT/OFFSET via wrapping subquery. Returns (columns, rows).
    """
    s = _clean_select(sql)
    off = max(0, int(offset))
    lim = int(limit) if limit and int(limit) > 0 else DEFAULT_SQL_LIMIT
    lim = min(lim, int(MAX_SQL_ROWS))
    wrapped = f"SELECT * FROM ({s}) AS sub LIMIT {lim} OFFSET {off}"
    with sqlite3.connect(DB_PATH) as conn:
        return _exec_fetch(conn, wrapped)

def run_count(sql: str) -> int:
    """
    COUNT(*) for an arbitrary SELECT: wraps as subquery.
    """
    s = _clean_select(sql)
    wrapped = f"SELECT COUNT(*) FROM ({s}) AS sub"
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(wrapped)
        val = cur.fetchone()[0]
        cur.close()
        return int(val or 0)

# app/services/db.py

import re
import sqlite3
from typing import List

from ..config import DB_PATH

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def _safe_table(name: str) -> str:
    if not isinstance(name, str) or not _IDENT.match(name):
        raise ValueError("Invalid table name.")
    return name

def table_exists(table: str = "data") -> bool:
    t = _safe_table(table)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (t,))
        return cur.fetchone() is not None

def get_columns(table: str = "data") -> List[str]:
    """
    Return column names for the table, or [] if the table doesn't exist.
    """
    t = _safe_table(table)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        # If missing, return []
        cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (t,))
        if cur.fetchone() is None:
            return []
        cur.execute(f"PRAGMA table_info({t})")
        rows = cur.fetchall()
        return [r[1] for r in rows]  # r[1] = name

def get_row_count(table: str = "data") -> int:
    """
    Return COUNT(*) for the table, or 0 if it doesn't exist.
    """
    t = _safe_table(table)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (t,))
        if cur.fetchone() is None:
            return 0
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        val = cur.fetchone()[0]
        return int(val or 0)


# Add this function to your app/services/db.py file

def select_within_radius(center_lat, center_lon, radius_km, table="data", lat_col="LATITUDE", lon_col="LONGITUDE"):
    """
    Select rows within a radius of a center point using Haversine distance formula.
    Returns (columns, rows) tuple.
    """
    # Haversine distance formula in SQL
    # R = 6371 km (Earth's radius)
    haversine_sql = f"""
    SELECT *,
           (6371 * acos(cos(radians({center_lat})) 
                      * cos(radians({lat_col})) 
                      * cos(radians({lon_col}) - radians({center_lon})) 
                      + sin(radians({center_lat})) 
                      * sin(radians({lat_col})))) AS distance_km
    FROM {table}
    WHERE {lat_col} IS NOT NULL 
      AND {lon_col} IS NOT NULL
      AND (6371 * acos(cos(radians({center_lat})) 
                     * cos(radians({lat_col})) 
                     * cos(radians({lon_col}) - radians({center_lon})) 
                     + sin(radians({center_lat})) 
                     * sin(radians({lat_col})))) <= {radius_km}
    ORDER BY distance_km
    """

    try:
        return select_sql(haversine_sql)
    except Exception as e:
        # Fallback for SQLite that might not support all math functions
        print(f"Haversine SQL failed, trying simplified approach: {e}")

        # Simplified bounding box approach (less accurate but works in basic SQLite)
        # Approximate degrees per km at given latitude
        lat_range = radius_km / 111.0  # 1 degree lat ≈ 111 km
        lon_range = radius_km / (111.0 * abs(math.cos(math.radians(center_lat))))

        bbox_sql = f"""
        SELECT *
        FROM {table}
        WHERE {lat_col} IS NOT NULL 
          AND {lon_col} IS NOT NULL
          AND {lat_col} BETWEEN {center_lat - lat_range} AND {center_lat + lat_range}
          AND {lon_col} BETWEEN {center_lon - lon_range} AND {center_lon + lon_range}
        """

        columns, rows = select_sql(bbox_sql)

        # Post-process to calculate exact distances and filter
        import math

        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate the great circle distance between two points on Earth."""
            R = 6371  # Earth's radius in kilometers

            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            c = 2 * math.asin(math.sqrt(a))
            return R * c

        # Find column indices
        lat_idx = columns.index(lat_col) if lat_col in columns else None
        lon_idx = columns.index(lon_col) if lon_col in columns else None

        if lat_idx is None or lon_idx is None:
            raise ValueError(f"Could not find columns {lat_col} or {lon_col}")

        # Filter rows by exact distance and add distance column
        filtered_rows = []
        new_columns = columns + ["distance_km"]

        for row in rows:
            try:
                lat = float(row[lat_idx])
                lon = float(row[lon_idx])
                distance = haversine_distance(center_lat, center_lon, lat, lon)

                if distance <= radius_km:
                    # Convert row to list if it's a tuple, then append distance
                    new_row = list(row) + [round(distance, 3)]
                    filtered_rows.append(new_row)

            except (ValueError, TypeError):
                # Skip rows with invalid coordinates
                continue

        # Sort by distance
        filtered_rows.sort(key=lambda x: x[-1])  # Sort by distance_km column

        return new_columns, filtered_rows