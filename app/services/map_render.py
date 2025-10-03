# app/services/map_render.py
import html
import json
from typing import Iterable, List, Optional, Mapping, Any, Tuple, Union

DEFAULT_LAT_COL = "LATITUDE"
DEFAULT_LON_COL = "LONGITUDE"

# Show these when your result has a lot of columns (case-insensitive match)
PRESET_POPUP_COLS = [
    "DAUID",
    "population_2021",
    "median_age_of_the_population",
    "employment_rate",
    "average_after_tax_income_in_2020",
    "LATITUDE", "LONGITUDE",
]

RowType = Union[Mapping[str, Any], List[Any], Tuple[Any, ...]]

# -------------------------- helpers ------------------------------------------

def _coerce_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if v == v else None  # drop NaN
    except Exception:
        return None

_LAT_SYNONYMS = {"lat", "latitude", "y", "lat_deg"}
_LON_SYNONYMS = {"lon", "lng", "long", "longitude", "x", "lon_deg"}

def _ensure_row_dict(r: RowType, columns: Optional[List[str]]) -> Mapping[str, Any]:
    """Return a dict view of row r. If r is a sequence and columns are provided, zip them."""
    if isinstance(r, Mapping):
        return r
    if isinstance(r, (list, tuple)) and columns:
        d = {}
        for i, c in enumerate(columns):
            d[c] = r[i] if i < len(r) else None
        return d
    if isinstance(r, (list, tuple)):
        return {str(i): v for i, v in enumerate(r)}  # last resort keys: "0","1",...
    return {"value": r}

def _resolve_lat_lon_keys(
    rows: Iterable[RowType],
    columns: Optional[List[str]],
    lat_col: str,
    lon_col: str,
) -> Tuple[str, str]:
    key_set = set(columns or [])
    if not key_set:
        for r in list(rows)[:5]:
            if isinstance(r, Mapping):
                key_set.update(r.keys())
    if not key_set:
        return lat_col, lon_col

    def pick(cand: Optional[str], synonyms: set) -> Optional[str]:
        if cand:
            for k in key_set:
                if k.casefold() == cand.casefold():
                    return k
        for syn in synonyms:
            for k in key_set:
                if k.casefold() == syn:
                    return k
        return None

    lat_key = pick(lat_col, _LAT_SYNONYMS) or lat_col
    lon_key = pick(lon_col, _LON_SYNONYMS) or lon_col
    return lat_key, lon_key

def _effective_popup_columns(
    columns: Optional[List[str]],
    rows: Optional[Iterable[RowType]],
    points: Optional[Iterable[Any]],
    lat_col: str,
    lon_col: str,
    requested: Optional[List[str]],
) -> Optional[List[str]]:
    """
    Popup selection:
      1) If 'requested' provided -> use it.
      2) Else if 'columns' present and (len(columns minus lat/lon) <= 10) -> use those.
      3) Else try PRESET_POPUP_COLS filtered to existing names.
      4) Else fallback to first 10 available non-lat/lon columns derived from data.
    """
    if requested:
        req = [c for c in requested if c]
        if req:
            return req

    lat_cf, lon_cf = lat_col.casefold(), lon_col.casefold()

    def collect_available() -> List[str]:
        avail: List[str] = []

        def add_key(k: Optional[str]):
            if not k:
                return
            kcf = k.casefold()
            if kcf in {lat_cf, lon_cf}:
                return
            if k not in avail:
                avail.append(k)

        if columns:
            for c in columns:
                add_key(c)

        if not avail and rows:
            for r in list(rows)[:3]:
                if isinstance(r, Mapping):
                    for k in r.keys():
                        add_key(k)

        if not avail and points:
            for p in list(points)[:3]:
                if isinstance(p, (list, tuple)) and len(p) >= 3 and isinstance(p[2], dict):
                    for k in p[2].keys():
                        add_key(k)
                elif isinstance(p, dict):
                    for k in p.keys():
                        if k and k.lower() not in {"lon", "lng", "x", "lat", "y"}:
                            add_key(k)

        return avail

    if columns:
        trimmed = [c for c in columns if c and c.casefold() not in {lat_cf, lon_cf}]
        if 0 < len(trimmed) <= 10:
            return trimmed

    available = collect_available()
    if available:
        cfmap = {k.casefold(): k for k in available}
        preset = []
        for wanted in PRESET_POPUP_COLS:
            m = cfmap.get(wanted.casefold())
            if m:
                preset.append(m)
        if preset:
            return preset[:10]
        return available[:10]

    return None

# -------------------------- feature builders ---------------------------------

def _rows_to_features(
    rows: Iterable[RowType],
    columns: Optional[List[str]],
    lat_key: str,
    lon_key: str,
    popup_cols: Optional[List[str]],
) -> List[dict]:
    out: List[dict] = []
    for r in list(rows or []):
        rd = _ensure_row_dict(r, columns)
        lat = _coerce_float(rd.get(lat_key))
        lon = _coerce_float(rd.get(lon_key))
        if lat is None or lon is None:
            continue

        # Build props by policy
        props: dict = {}
        if popup_cols:
            for k in popup_cols:
                if k in rd:
                    props[k] = rd[k]
                else:
                    for rk in rd.keys():
                        if rk.casefold() == k.casefold():
                            props[rk] = rd[rk]
                            break
        else:
            for k in rd.keys():
                if len(props) >= 8:
                    break
                if k.casefold() not in {lat_key.casefold(), lon_key.casefold()}:
                    props[k] = rd[k]

        if not props:
            props = {lat_key: rd.get(lat_key), lon_key: rd.get(lon_key)}

        out.append({"lat": float(lat), "lon": float(lon), "props": props})
    return out

def _points_to_features(points: Iterable[Any],
                        popup_cols: Optional[List[str]]) -> List[dict]:
    out: List[dict] = []
    for p in list(points or []):
        lat = lon = None
        props = {}
        if isinstance(p, (list, tuple)):
            if len(p) >= 2:
                lon = _coerce_float(p[0]); lat = _coerce_float(p[1])
            if len(p) >= 3 and isinstance(p[2], dict):
                props = dict(p[2])
        elif isinstance(p, dict):
            lon = _coerce_float(p.get("lon") or p.get("lng") or p.get("x"))
            lat = _coerce_float(p.get("lat") or p.get("y"))
            props = {k: v for k, v in p.items() if k not in {"lon", "lng", "x", "lat", "y"}}
        if lat is None or lon is None:
            continue

        if popup_cols:
            fprops = {}
            for k in popup_cols:
                if k in props:
                    fprops[k] = props[k]
                else:
                    for rk in props.keys():
                        if rk.casefold() == k.casefold():
                            fprops[rk] = props[rk]
                            break
            props = fprops

        if not props:
            props = {"LATITUDE": lat, "LONGITUDE": lon}

        out.append({"lat": float(lat), "lon": float(lon), "props": props})
    return out

# -------------------------- main API -----------------------------------------

def build_map_html(
    rows: Optional[Iterable[RowType]] = None,
    columns: Optional[List[str]] = None,
    *,
    points: Optional[List[Any]] = None,
    lat_col: str = DEFAULT_LAT_COL,
    lon_col: str = DEFAULT_LON_COL,
    popup_columns: Optional[List[str]] = None,
    title: str = "Interactive Map",
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
    zoom: Optional[int] = None,
    circle: Optional[dict] = None,     # {lat, lon, radius_km}
    radius_km: Optional[float] = None, # legacy support
    **_kwargs
) -> str:
    # Decide popup columns
    eff_popup_cols = _effective_popup_columns(columns, rows, points, lat_col, lon_col, popup_columns)

    # Decide which keys to use for coordinates
    lat_key, lon_key = _resolve_lat_lon_keys(rows or [], columns, lat_col, lon_col)

    # Build features (NO LIMIT — you want all markers)
    features_rows = _rows_to_features(rows or [], columns, lat_key, lon_key, eff_popup_cols)
    features_pts  = _points_to_features(points or [], eff_popup_cols)
    features = features_rows + features_pts

    # Circle: accept either 'circle' dict or (center_* + radius_km)
    if not circle and (radius_km is not None) and (center_lat is not None) and (center_lon is not None):
        circle = {"lat": center_lat, "lon": center_lon, "radius_km": radius_km}

    # Compute bounds
    safe_title = html.escape(title or "Interactive Map")
    bounds = None
    if features:
        lats = [f["lat"] for f in features]
        lons = [f["lon"] for f in features]
        bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]

    # Include debug meta so the page can explain itself if zero markers show
    sample = None
    if features:
        # keep sample small
        sp = dict(features[0])
        if isinstance(sp.get("props"), dict):
            sp["props"] = {k: sp["props"][k] for k in list(sp["props"].keys())[:8]}
        sample = sp

    payload = {
        "title": safe_title,
        "features": features,
        "bounds": bounds,
        "center": {"lat": center_lat, "lon": center_lon} if (center_lat is not None and center_lon is not None) else None,
        "zoom": zoom,
        "circle": circle,
        "marker_count": len(features),
        "lat_key_used": lat_key,
        "lon_key_used": lon_key,
        "feature_sample": sample,
    }

    js_data = json.dumps(payload, ensure_ascii=False)

    # HTML (Leaflet + MarkerCluster). If cluster script fails to load, we fallback to plain markers (first 5k) so you see *something*.
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{safe_title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin=""/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css"/>
  <style>
    html, body {{ height:100%; margin:0; }}
    #map {{ height:100%; width:100%; }}
    .leaflet-popup-content table {{ border-collapse: collapse; }}
    .leaflet-popup-content td {{ padding:2px 6px; border-bottom:1px solid #eee; vertical-align:top; }}
    .meta-bar {{
      position:absolute; left:10px; top:10px; z-index: 1000; background:#fff; padding:6px 10px; border-radius:8px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1); font: 14px system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    }}
    .warn {{ color:#b45309; }}
    .error {{ color:#b91c1c; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="meta-bar" id="meta">Loading map…</div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
  <script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
  <script>
    var DATA = {js_data};
    console.debug("Map payload:", DATA);

    function esc(x) {{
      if (x === null || x === undefined) return "";
      return String(x)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }}

    function popupHTML(props) {{
      if (!props) return "<i>No attributes</i>";
      var html = "<div><table>";
      for (var k in props) if (Object.prototype.hasOwnProperty.call(props,k)) {{
        html += "<tr><td><b>" + esc(k) + "</b></td><td>" + esc(props[k]) + "</td></tr>";
      }}
      html += "</table></div>";
      return html;
    }}

    var meta = document.getElementById("meta");
    function setMeta(txt, cls) {{
      meta.textContent = txt;
      meta.className = "meta-bar" + (cls ? " " + cls : "");
    }}

    var map = L.map("map", {{ zoomControl: true }});
    L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
    }}).addTo(map);

    // Prefer MarkerCluster; if unavailable, fallback to plain markers (cap at 5000 for safety)
    var clusterAvailable = typeof L.markerClusterGroup === "function";
    var cluster = clusterAvailable ? L.markerClusterGroup({{ disableClusteringAtZoom: 14 }}) : null;
    if (cluster) map.addLayer(cluster);

    var added = 0;
    var features = Array.isArray(DATA.features) ? DATA.features : [];

    if (!features.length) {{
      setMeta("No features in payload. lat_key=" + DATA.lat_key_used + ", lon_key=" + DATA.lon_key_used, "error");
    }} else {{
      var fallbackCap = 5000; // only used if cluster lib is missing
      for (var i = 0; i < features.length; i++) {{
        if (!clusterAvailable && added >= fallbackCap) break; // keep the UI responsive if no cluster

        var f = features[i];
        if (!(Number.isFinite(f.lat) && Number.isFinite(f.lon))) continue;
        var m = L.marker([f.lat, f.lon]);
        m.bindPopup(popupHTML(f.props), {{ maxWidth: 360 }});
        if (cluster) cluster.addLayer(m); else m.addTo(map);
        added++;
      }}

      if (!clusterAvailable) {{
        setMeta("Added " + added + " marker(s). MarkerCluster script not loaded — showing first " + added + " only.", "warn");
        console.warn("MarkerCluster not available; fell back to plain markers. Added:", added, "out of:", features.length);
      }} else {{
        setMeta("Added " + added + " marker(s).", "");
      }}
    }}

    if (DATA.center && Number.isFinite(DATA.center.lat) && Number.isFinite(DATA.center.lon)) {{
      var z = (Number.isFinite(DATA.zoom) ? DATA.zoom : 10);
      map.setView([DATA.center.lat, DATA.center.lon], z);
    }} else if (DATA.bounds) {{
      map.fitBounds(DATA.bounds, {{ padding: [20,20] }});
    }} else {{
      map.setView([43.7, -79.4], 5);
    }}

    if (DATA.circle && Number.isFinite(DATA.circle.lat) && Number.isFinite(DATA.circle.lon) && Number.isFinite(DATA.circle.radius_km)) {{
      L.circle([DATA.circle.lat, DATA.circle.lon], {{
        radius: DATA.circle.radius_km * 1000,
        stroke: true, color: "red", weight: 2, fill: false
      }}).addTo(map);
    }}

    // If zero actually added, make it super obvious and show a sample/keys for debugging
    if (added === 0) {{
      var msg = "No markers were added. Using lat_key='" + DATA.lat_key_used + "', lon_key='" + DATA.lon_key_used + "'.";
      if (DATA.feature_sample) {{
        msg += " Example feature: " + JSON.stringify(DATA.feature_sample);
      }}
      setMeta(msg, "error");
      console.error(msg);
    }}
  </script>
</body>
</html>
"""
