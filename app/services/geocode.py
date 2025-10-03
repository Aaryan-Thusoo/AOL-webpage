import requests
from typing import List, Dict, Any
import time
import math
from .. import config

# Multiple fallback services configuration
REQUEST_TIMEOUT = 4
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

# Cache
_suggestion_cache = {}
_cache_expiry = {}
CACHE_DURATION = 600  # 10 minutes


def _is_cache_valid(query: str) -> bool:
    return (query in _suggestion_cache and
            query in _cache_expiry and
            time.time() < _cache_expiry[query])


def _cache_result(query: str, result: List[Dict[str, Any]]):
    _suggestion_cache[query] = result
    _cache_expiry[query] = time.time() + CACHE_DURATION


def suggest(query: str, country_hint: str = "ca", limit: int = 5) -> List[Dict[str, Any]]:
    """
    LocationIQ search for Canada with detailed addresses
    """
    q = (query or "").strip()
    print(f"[GEOCODE] LocationIQ search for: '{q}', limit: {limit}")

    if not q or len(q) < 2:
        print(f"[GEOCODE] Query too short, returning empty")
        return []

    # Check cache first
    cache_key = f"{q}:{country_hint}:{limit}"
    if _is_cache_valid(cache_key):
        print(f"[GEOCODE] Cache hit")
        return _suggestion_cache[cache_key]

    try:
        # CORRECT LocationIQ Search API endpoint
        url = "https://api.locationiq.com/v1/search.php"  # Remove the us1- prefix
        params = {
            "key": config.LOCATIONIQ_API_KEY,
            "q": f"{query}, Canada",  # Add Canada to query
            "format": "json",
            "limit": min(limit, 8),
            "addressdetails": 1,  # Get detailed address components
            "countrycodes": "ca"  # Restrict to Canada
        }

        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT,
                                headers={"User-Agent": USER_AGENT})
        response.raise_for_status()

        results_data = response.json()

        results = []
        for item in results_data[:limit]:
            display_name = item.get("display_name", "")
            address = item.get("address", {})

            # Build structured address from components
            address_parts = []

            # House number and street
            if address.get("house_number") and address.get("road"):
                address_parts.append(f"{address['house_number']} {address['road']}")
            elif address.get("road"):
                address_parts.append(address["road"])

            # City/town
            city = (address.get("city") or address.get("town") or
                    address.get("village") or address.get("hamlet"))
            if city:
                address_parts.append(city)

            # Province
            if address.get("state"):
                address_parts.append(address["state"])

            # Postal code
            if address.get("postcode"):
                address_parts.append(address["postcode"])

            # Create label (shorter) and display_name (full)
            if address_parts:
                structured_address = ", ".join(address_parts)
                # Label: first 2-3 parts for input field
                label_parts = address_parts[:3] if len(address_parts) >= 3 else address_parts[:2]
                label = ", ".join(label_parts)
            else:
                # Fallback to original display_name
                structured_address = display_name.replace(", Canada", "")
                label = structured_address.split(",")[0] if "," in structured_address else structured_address

            results.append({
                "lat": float(item["lat"]),
                "lon": float(item["lon"]),
                "label": label,
                "display_name": structured_address
            })

        print(f"[GEOCODE] LocationIQ returned {len(results)} structured results")
        _cache_result(cache_key, results)
        return results

    except Exception as e:
        print(f"[GEOCODE] LocationIQ search failed: {e}")
        return _try_canadian_cities_database(query, limit)


def _try_photon_service(query: str, limit: int) -> List[Dict[str, Any]]:
    """
    Try Photon API (OpenStreetMap-based, different endpoint)
    """
    print(f"[GEOCODE] Trying Photon service")

    # Photon API with Canada bounding box
    url = "https://photon.komoot.io/api/"
    params = {
        "q": query,
        "limit": min(limit, 8),
        "osm_tag": "place:city,place:town,place:village,amenity",
        "bbox": "-141,41.6,-52.6,83.23"  # Canada bounding box
    }

    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT,
                            headers={"User-Agent": USER_AGENT})
    response.raise_for_status()

    data = response.json()
    features = data.get("features", [])

    results = []
    for feature in features[:limit]:
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})
        coords = geom.get("coordinates", [])

        if len(coords) >= 2:
            name = props.get("name", "")
            city = props.get("city", "")
            state = props.get("state", "")
            country = props.get("country", "")

            # Build Canadian address
            if city and state and country == "Canada":
                label = f"{name}, {city}, {state}" if name != city else f"{city}, {state}"
                display_name = f"{label}, Canada"
            elif name:
                label = name
                display_name = f"{name}, Canada"
            else:
                continue

            results.append({
                "lat": coords[1],
                "lon": coords[0],
                "label": label,
                "display_name": display_name
            })

    return results


def _try_geocode_ca_service(query: str, limit: int) -> List[Dict[str, Any]]:
    """
    Try geocoder.ca (Canadian-specific service)
    """
    print(f"[GEOCODE] Trying geocoder.ca service")

    url = "https://geocoder.ca/"
    params = {
        "locate": query,
        "json": "1",
        "geoit": "xml"
    }

    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT,
                            headers={"User-Agent": USER_AGENT})
    response.raise_for_status()

    data = response.json()

    # geocoder.ca returns single result
    if "latt" in data and "longt" in data:
        city = data.get("standard", {}).get("city", "")
        prov = data.get("standard", {}).get("prov", "")

        if city and prov:
            label = f"{city}, {prov}"
            display_name = f"{city}, {prov}, Canada"
        else:
            label = query
            display_name = f"{query}, Canada"

        return [{
            "lat": float(data["latt"]),
            "lon": float(data["longt"]),
            "label": label,
            "display_name": display_name
        }]

    return []


def _try_nominatim_different_endpoint(query: str, limit: int) -> List[Dict[str, Any]]:
    """
    Try different Nominatim approach with better headers and rate limiting
    """
    print(f"[GEOCODE] Trying Nominatim with better headers")

    # Wait a bit to avoid rate limiting
    time.sleep(0.5)

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": f"{query}, Canada",  # Add Canada explicitly
        "format": "json",
        "limit": min(limit, 5),
        "addressdetails": 1,
        "accept-language": "en"
    }

    # Better headers to avoid blocking
    headers = {
        "User-Agent": "DataVisualizationApp/1.0 (educational use)",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "http://localhost:5000"
    }

    response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)

    if response.status_code == 403:
        print(f"[GEOCODE] Still blocked by Nominatim")
        return []

    response.raise_for_status()
    results_data = response.json()

    results = []
    for item in results_data[:limit]:
        address = item.get("address", {})
        display_name = item.get("display_name", "")

        # Filter for Canadian results only
        if "Canada" not in display_name:
            continue

        city = address.get("city") or address.get("town") or address.get("village", "")
        province = address.get("state", "")

        if city and province:
            label = f"{city}, {province}"
        else:
            label = display_name.split(",")[0]

        results.append({
            "lat": float(item["lat"]),
            "lon": float(item["lon"]),
            "label": label,
            "display_name": display_name
        })

    return results


def _try_canadian_cities_database(query: str, limit: int) -> List[Dict[str, Any]]:
    """
    Fallback: Match against major Canadian cities/landmarks
    """
    print(f"[GEOCODE] Using Canadian cities database")

    # Comprehensive Canadian locations database
    canadian_locations = [
        # Major cities
        {"name": "Toronto", "province": "ON", "lat": 43.6532, "lon": -79.3832},
        {"name": "Vancouver", "province": "BC", "lat": 49.2827, "lon": -123.1207},
        {"name": "Montreal", "province": "QC", "lat": 45.5017, "lon": -73.5673},
        {"name": "Calgary", "province": "AB", "lat": 51.0447, "lon": -114.0719},
        {"name": "Ottawa", "province": "ON", "lat": 45.4215, "lon": -75.6972},
        {"name": "Edmonton", "province": "AB", "lat": 53.5461, "lon": -113.4938},
        {"name": "Winnipeg", "province": "MB", "lat": 49.8951, "lon": -97.1384},
        {"name": "Quebec City", "province": "QC", "lat": 46.8139, "lon": -71.2082},
        {"name": "Hamilton", "province": "ON", "lat": 43.2557, "lon": -79.8711},
        {"name": "Kitchener", "province": "ON", "lat": 43.4516, "lon": -80.4925},
        {"name": "London", "province": "ON", "lat": 42.9849, "lon": -81.2453},
        {"name": "Victoria", "province": "BC", "lat": 48.4284, "lon": -123.3656},
        {"name": "Halifax", "province": "NS", "lat": 44.6488, "lon": -63.5752},
        {"name": "Saskatoon", "province": "SK", "lat": 52.1579, "lon": -106.6702},
        {"name": "Regina", "province": "SK", "lat": 50.4452, "lon": -104.6189},

        # Landmarks and notable places
        {"name": "CN Tower", "province": "ON", "lat": 43.6426, "lon": -79.3871},
        {"name": "CN Tower Toronto", "province": "ON", "lat": 43.6426, "lon": -79.3871},
        {"name": "Parliament Hill", "province": "ON", "lat": 45.4236, "lon": -75.7005},
        {"name": "Parliament Hill Ottawa", "province": "ON", "lat": 45.4236, "lon": -75.7005},
        {"name": "Rogers Centre", "province": "ON", "lat": 43.6414, "lon": -79.3894},
        {"name": "Rogers Centre Toronto", "province": "ON", "lat": 43.6414, "lon": -79.3894},
        {"name": "Old Quebec", "province": "QC", "lat": 46.8131, "lon": -71.2075},
        {"name": "Stanley Park", "province": "BC", "lat": 49.3017, "lon": -123.1447},
        {"name": "Stanley Park Vancouver", "province": "BC", "lat": 49.3017, "lon": -123.1447},

        # More cities
        {"name": "Brampton", "province": "ON", "lat": 43.7315, "lon": -79.7624},
        {"name": "Mississauga", "province": "ON", "lat": 43.5890, "lon": -79.6441},
        {"name": "Surrey", "province": "BC", "lat": 49.1913, "lon": -122.8490},
        {"name": "Laval", "province": "QC", "lat": 45.6066, "lon": -73.7124},
        {"name": "Halifax Regional Municipality", "province": "NS", "lat": 44.6488, "lon": -63.5752},
        {"name": "Markham", "province": "ON", "lat": 43.8561, "lon": -79.3370},
        {"name": "Vaughan", "province": "ON", "lat": 43.8361, "lon": -79.4985},
        {"name": "Gatineau", "province": "QC", "lat": 45.4765, "lon": -75.7013},
        {"name": "Longueuil", "province": "QC", "lat": 45.5312, "lon": -73.5185},
        {"name": "Burnaby", "province": "BC", "lat": 49.2488, "lon": -122.9805},

        # Smaller cities and towns
        {"name": "Barrie", "province": "ON", "lat": 44.3894, "lon": -79.6903},
        {"name": "Guelph", "province": "ON", "lat": 43.5448, "lon": -80.2482},
        {"name": "Kingston", "province": "ON", "lat": 44.2312, "lon": -76.4860},
        {"name": "Oshawa", "province": "ON", "lat": 43.8971, "lon": -78.8658},
        {"name": "Windsor", "province": "ON", "lat": 42.3149, "lon": -83.0364},
        {"name": "Sherbrooke", "province": "QC", "lat": 45.4042, "lon": -71.8929},
        {"name": "Saguenay", "province": "QC", "lat": 48.3985, "lon": -71.0656},
        {"name": "Trois-Rivières", "province": "QC", "lat": 46.3432, "lon": -72.5432},

        # Provincial capitals
        {"name": "St. John's", "province": "NL", "lat": 47.5615, "lon": -52.7126},
        {"name": "Fredericton", "province": "NB", "lat": 45.9636, "lon": -66.6431},
        {"name": "Charlottetown", "province": "PE", "lat": 46.2382, "lon": -63.1311},
        {"name": "Whitehorse", "province": "YT", "lat": 60.7212, "lon": -135.0568},
        {"name": "Yellowknife", "province": "NT", "lat": 62.4540, "lon": -114.3718},
        {"name": "Iqaluit", "province": "NU", "lat": 63.7467, "lon": -68.5170}
    ]

    query_lower = query.lower()
    matches = []

    for location in canadian_locations:
        name_lower = location["name"].lower()
        # Flexible matching: contains, starts with, or word boundary matches
        if (query_lower in name_lower or
                name_lower.startswith(query_lower) or
                any(word.startswith(query_lower) for word in name_lower.split())):
            matches.append({
                "lat": location["lat"],
                "lon": location["lon"],
                "label": f"{location['name']}, {location['province']}",
                "display_name": f"{location['name']}, {location['province']}, Canada"
            })

    return matches[:limit]


def _coerce_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two WGS84 points in kilometers.
    """
    R = 6371.0088  # mean Earth radius (km)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def filter_within_radius(
        rows: List[Dict[str, Any]],
        center_lat: float,
        center_lon: float,
        radius_km: float,
        lat_keys: tuple[str, ...] = ("LATITUDE", "latitude", "lat"),
        lon_keys: tuple[str, ...] = ("LONGITUDE", "longitude", "lon"),
) -> List[Dict[str, Any]]:
    """
    Keep only rows whose (lat, lon) are within radius_km of center point.
    Adds 'distance_km' (rounded) to each kept row.
    """
    kept: List[Dict[str, Any]] = []
    for r in rows:
        lat = None
        lon = None
        for k in lat_keys:
            if k in r:
                lat = _coerce_float(r[k])
                if lat is not None:
                    break
        for k in lon_keys:
            if k in r:
                lon = _coerce_float(r[k])
                if lon is not None:
                    break
        if lat is None or lon is None:
            continue  # skip rows without coords
        d = haversine_km(center_lat, center_lon, lat, lon)
        if d <= radius_km:
            out = dict(r)
            out["distance_km"] = round(d, 3)
            kept.append(out)
    return kept


def geocode_address(address: str, country_hint: str = "ca") -> Dict[str, Any]:
    """
    Resolve a free-text address to {lat, lon, display_name}.
    Uses multiple fallback services for reliability.
    """
    results = suggest(address, country_hint, 1)
    if not results:
        raise LookupError("Address not found.")
    return results[0]