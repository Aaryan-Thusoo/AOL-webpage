# Optimized clustering.py - much faster population clustering
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Set
import random
import math
import os
import time

try:
    import pandas as pd
    from sklearn.cluster import KMeans
    import numpy as np
    from sklearn.neighbors import BallTree, NearestNeighbors
    from scipy.spatial.distance import pdist, squareform
except ImportError:
    pd = None
    np = None
    KMeans = None
    BallTree = None


def load_aggregation_config(base_path: str = ".") -> Tuple[Set[str], Set[str], Set[str]]:
    """Load aggregation configuration from text files."""
    print(f"DEBUG: Looking for config files in: {os.path.abspath(base_path)}")

    def read_column_list(filename: str) -> Set[str]:
        filepath = os.path.join(base_path, filename)
        print(f"DEBUG: Checking for file: {filepath}")

        if not os.path.exists(filepath):
            print(f"DEBUG: File {filename} not found, returning empty set")
            return set()

        print(f"DEBUG: Reading file: {filename}")
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        columns = set()
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                columns.add(line)
                print(f"DEBUG: Added column: {line}")

        print(f"DEBUG: File {filename} loaded {len(columns)} columns: {columns}")
        return columns

    average_cols = read_column_list("average_columns.txt")
    median_cols = read_column_list("median_columns.txt")
    sum_cols = read_column_list("sum_columns.txt")  # Will be empty set if file missing

    print(f"DEBUG: Final config - avg: {len(average_cols)}, median: {len(median_cols)}, sum: {len(sum_cols)}")
    return average_cols, median_cols, sum_cols


def aggregate_cluster_data(
        df: "pd.DataFrame",
        cluster_col: str = "cluster_id",
        dauid_col: str = "DAUID"
) -> "pd.DataFrame":
    """Aggregate clustered data according to configuration files."""
    print(f"\n=== AGGREGATION DEBUG ===")
    print(f"Input DataFrame shape: {df.shape}")
    print(f"Input columns: {list(df.columns)}")
    print(f"Looking for cluster column: {cluster_col}")
    print(f"Looking for DAUID column: {dauid_col}")

    if df.empty:
        print("ERROR: DataFrame is empty")
        return df

    # Check if cluster_col exists
    if cluster_col not in df.columns:
        print(f"ERROR: Cluster column '{cluster_col}' not found in data")
        print(f"Available columns: {list(df.columns)}")
        return pd.DataFrame()

    # Check cluster distribution
    cluster_counts = df[cluster_col].value_counts()
    print(f"Cluster distribution: {dict(cluster_counts.head(10))}")

    # Load aggregation configuration
    try:
        average_cols, median_cols, sum_cols = load_aggregation_config()
        print(f"Config loaded - Average: {len(average_cols)}, Median: {len(median_cols)}, Sum: {len(sum_cols)}")
        print(f"Average columns: {average_cols}")
        print(f"Median columns: {median_cols}")
    except Exception as e:
        print(f"ERROR loading config: {e}")
        print("Using empty config sets")
        average_cols, median_cols, sum_cols = set(), set(), set()

    # Find DAUID column (case insensitive)
    dauid_column = None
    for col in df.columns:
        if col.upper() == dauid_col.upper():
            dauid_column = col
            break

    print(f"DAUID column found: {dauid_column}")
    if dauid_column is None:
        print(f"WARNING: No DAUID column found. Searched for variations of '{dauid_col}'")
        print(f"Available columns (case comparison): {[col.upper() for col in df.columns]}")

    # Group by cluster
    try:
        grouped = df.groupby(cluster_col)
        print(f"Grouping successful. Found {len(grouped)} cluster groups")
    except Exception as e:
        print(f"ERROR grouping by {cluster_col}: {e}")
        return pd.DataFrame()

    result_rows = []

    for cluster_id, group in grouped:
        if cluster_id == -1:  # Skip outliers
            continue

        print(f"Processing cluster {cluster_id} with {len(group)} rows")

        agg_row = {"cluster_id": cluster_id}

        # Collect DAUIDs for this cluster
        if dauid_column and dauid_column in group.columns:
            dauids = group[dauid_column].dropna().astype(str).tolist()
            agg_row["DAUIDs"] = ",".join(dauids)
            print(f"  Added {len(dauids)} DAUIDs")
        else:
            agg_row["DAUIDs"] = ""
            print(f"  No DAUIDs added (column: {dauid_column})")

        # Process each numeric column
        numeric_cols_processed = 0
        for col in df.columns:
            if col in [cluster_col, dauid_column]:
                continue

            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(group[col]):
                continue

            # Determine aggregation method
            if col in average_cols:
                agg_row[col] = group[col].mean()
                aggregation_method = "AVERAGE"
            elif col in median_cols:
                agg_row[col] = group[col].median()
                aggregation_method = "MEDIAN"
            else:
                # Default to sum
                agg_row[col] = group[col].sum()
                aggregation_method = "SUM"

            numeric_cols_processed += 1
            if numeric_cols_processed <= 3:  # Log first few for debugging
                print(f"  {col}: {aggregation_method} = {agg_row[col]}")

        print(f"  Total numeric columns processed: {numeric_cols_processed}")
        result_rows.append(agg_row)

    result_df = pd.DataFrame(result_rows)
    print(f"Final aggregated DataFrame shape: {result_df.shape}")
    print(f"Final columns: {list(result_df.columns) if not result_df.empty else 'EMPTY'}")

    return result_df


def cluster_kmeans(rows: List[Dict[str, Any]], n_clusters: int) -> Dict[str, Any]:
    """Apply KMeans clustering with aggregation."""
    if not rows:
        return {"columns": [], "rows": [], "da_rows": [], "message": "No rows to cluster."}
    if KMeans is None or pd is None:
        raise RuntimeError("scikit-learn and pandas are required for KMeans clustering.")

    df = pd.DataFrame(rows)

    # Find lat/lon column names
    lat_col = next((c for c in df.columns if c.lower() in ["latitude", "lat"]), None)
    lon_col = next((c for c in df.columns if c.lower() in ["longitude", "lon"]), None)
    if not lat_col or not lon_col:
        raise RuntimeError("Could not detect latitude/longitude columns.")

    coords = df[[lat_col, lon_col]]
    model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = model.fit_predict(coords)

    df["cluster_id"] = labels

    # Aggregate the data
    aggregated_df = aggregate_cluster_data(df)

    # Convert back to records
    if not aggregated_df.empty:
        aggregated_rows = aggregated_df.where(pd.notnull(aggregated_df), None).to_dict(orient="records")
        aggregated_cols = aggregated_df.columns.tolist()
    else:
        aggregated_rows = []
        aggregated_cols = []

    # Also keep original rows with cluster_id for mapping
    original_rows = df.where(pd.notnull(df), None).to_dict(orient="records")

    return {
        "columns": aggregated_cols,
        "rows": aggregated_rows,
        "da_rows": original_rows,
        "message": f"K-means clustered into {n_clusters} groups and aggregated data.",
        "cluster_count": len(aggregated_rows)
    }


# app/services/clustering.py

def _find_cols(df):
    cols_lower = [c.lower() for c in df.columns]

    def pref_exact(names):
        for n in names:
            if n in cols_lower:
                return df.columns[cols_lower.index(n)]
        return None

    # Latitude / longitude
    lat = pref_exact(["latitude", "lat"])
    lon = pref_exact(["longitude", "lon", "lng"])

    # Population (prefer true totals)
    pop = pref_exact(["population_2021", "total_population", "population"])

    if pop is None:
        # Fallback: any column containing 'population' or 'pop', but avoid non-counts
        blacklist = ("density", "median_", "composition", "aged", "years",
                     "percentage", "percent", "rate", "proportion", "ratio")
        candidates = [
            c for c in df.columns
            if any(k in c.lower() for k in ("population", "pop"))
            and not any(b in c.lower() for b in blacklist)
        ]
        pop = candidates[0] if candidates else None

    if not (lat and lon and pop):
        raise RuntimeError(
            f"Could not detect lat/lon/pop columns. "
            f"lat={lat}, lon={lon}, pop={pop}"
        )

    return lat, lon, pop



def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Calculate haversine distance in kilometers."""
    R = 6371.0088
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def _weighted_centroid(lat: "np.ndarray", lon: "np.ndarray", w: "np.ndarray") -> Tuple[float, float]:
    """Calculate weighted centroid."""
    W = w.sum()
    if W <= 0:
        return float(lat.mean()), float(lon.mean())
    return float((lat * w).sum() / W), float((lon * w).sum() / W)


class FastSpatialIndex:
    """Fast spatial indexing for population clustering."""

    def __init__(self, coords: "np.ndarray"):
        self.coords = coords
        self.n = len(coords)

        # Use BallTree if available, otherwise fall back to brute force
        if BallTree is not None:
            self.tree = BallTree(np.radians(coords), metric="haversine")
            self.use_tree = True
        else:
            self.tree = None
            self.use_tree = False
            # Precompute distance matrix for small datasets
            if self.n < 1000:
                self.dist_matrix = self._compute_distance_matrix()
            else:
                self.dist_matrix = None

    def _compute_distance_matrix(self):
        """Precompute distance matrix for small datasets."""
        distances = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = _haversine_km(self.coords[i, 0], self.coords[i, 1],
                                  self.coords[j, 0], self.coords[j, 1])
                distances[i, j] = distances[j, i] = d
        return distances

    def neighbors_within_km(self, center_lat: float, center_lon: float,
                            radius_km: float, exclude_indices: set) -> List[int]:
        """Find all points within radius_km of center, excluding specified indices."""
        if self.use_tree:
            # Use BallTree
            R = 6371.0088
            rad = radius_km / R
            idx = self.tree.query_radius(np.radians([[center_lat, center_lon]]),
                                         r=rad, return_distance=False)[0]
            return [i for i in idx if i not in exclude_indices]

        elif self.dist_matrix is not None:
            # Use precomputed distance matrix
            result = []
            for i in range(self.n):
                if i in exclude_indices:
                    continue
                d = _haversine_km(center_lat, center_lon, self.coords[i, 0], self.coords[i, 1])
                if d <= radius_km:
                    result.append(i)
            return result

        else:
            # Brute force for large datasets without BallTree
            result = []
            for i in range(self.n):
                if i in exclude_indices:
                    continue
                d = _haversine_km(center_lat, center_lon, self.coords[i, 0], self.coords[i, 1])
                if d <= radius_km:
                    result.append(i)
            return result


# Enhanced clustering.py with better debugging for population clustering

# Enhanced clustering with coverage analysis and parameter suggestions

def analyze_clustering_coverage(assigned, total_pop, target_pop, tolerance, summaries):
    """
    Analyze clustering coverage and provide intelligent suggestions.
    """
    total_areas = len(assigned)
    clustered_areas = int((assigned >= 0).sum())
    outlier_areas = int((assigned == -1).sum())

    area_coverage = (clustered_areas / total_areas) * 100 if total_areas > 0 else 0

    clustered_population = sum(s['total_pop'] for s in summaries)
    pop_coverage = (clustered_population / total_pop) * 100 if total_pop > 0 else 0

    # Determine coverage quality
    if area_coverage >= 85:
        coverage_quality = "Excellent"
    elif area_coverage >= 70:
        coverage_quality = "Good"
    elif area_coverage >= 50:
        coverage_quality = "Fair"
    else:
        coverage_quality = "Poor"

    # Generate suggestions based on coverage
    suggestions = []
    warnings = []

    if area_coverage < 50:
        warnings.append(f"Low clustering coverage: Only {area_coverage:.1f}% of areas were successfully clustered")
        suggestions.extend([
            f"Try reducing target population from {target_pop:,} to {int(target_pop * 0.7):,}",
            f"Increase tolerance from {tolerance * 100:.0f}% to {min(25, tolerance * 100 + 10):.0f}%",
            "Consider using K-means clustering for better coverage"
        ])
    elif area_coverage < 70:
        warnings.append(f"Moderate clustering coverage: {area_coverage:.1f}% of areas clustered")
        suggestions.extend([
            f"For better coverage, try reducing target population to {int(target_pop * 0.8):,}",
            f"Consider increasing tolerance to {min(20, tolerance * 100 + 5):.0f}%"
        ])

    if outlier_areas > total_areas * 0.3:  # More than 30% outliers
        warnings.append(
            f"High outlier rate: {outlier_areas:,} areas ({outlier_areas / total_areas * 100:.1f}%) marked as outliers")
        suggestions.append("High outlier rate suggests spatial constraints are too tight")

    # Check if clusters are too large/small
    if summaries:
        avg_cluster_size = clustered_areas / len(summaries)
        if avg_cluster_size > 500:
            suggestions.append(
                f"Large clusters detected (avg: {avg_cluster_size:.0f} areas). Consider reducing target population")
        elif avg_cluster_size < 50:
            suggestions.append(
                f"Small clusters detected (avg: {avg_cluster_size:.0f} areas). Consider increasing target population")

    return {
        "area_coverage_percent": round(area_coverage, 1),
        "population_coverage_percent": round(pop_coverage, 1),
        "coverage_quality": coverage_quality,
        "clustered_areas": clustered_areas,
        "outlier_areas": outlier_areas,
        "total_areas": total_areas,
        "warnings": warnings,
        "suggestions": suggestions
    }


def cluster_population_optimized(
        rows: List[Dict[str, Any]],
        target_pop: int,
        tolerance: float,
) -> Dict[str, Any]:
    """
    Optimized population clustering with comprehensive coverage analysis.
    """
    start_time = time.time()

    if not rows:
        return {"columns": [], "rows": [], "da_rows": [], "message": "No rows to cluster.", "summaries": []}
    if pd is None or np is None:
        raise RuntimeError("pandas and numpy are required for population clustering.")

    print(f"\n=== POPULATION CLUSTERING DEBUG ===")
    print(f"Input: {len(rows)} areas to cluster")
    print(f"Target population: {target_pop:,}, Tolerance: {tolerance * 100:.1f}%")

    df = pd.DataFrame(rows)
    lat_col, lon_col, pop_col = _find_cols(df)
    print(f"Using columns: lat={lat_col}, lon={lon_col}, pop={pop_col}")

    # Clean + filter invalids
    df = df.copy()
    df["_lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["_lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    df["_pop"] = pd.to_numeric(df[pop_col], errors="coerce").fillna(0).clip(lower=0)

    # Debug population statistics BEFORE filtering
    total_population_raw = df['_pop'].sum()
    print(f"\nPopulation statistics (before filtering):")
    print(f"  Total population: {total_population_raw:,.0f}")
    print(f"  Average per area: {df['_pop'].mean():.0f}")
    print(f"  Min population: {df['_pop'].min():.0f}")
    print(f"  Max population: {df['_pop'].max():.0f}")
    print(f"  Areas with population > 0: {(df['_pop'] > 0).sum()}")

    df = df.dropna(subset=["_lat", "_lon"]).reset_index(drop=True)

    if df.empty:
        return {"columns": [], "rows": [], "da_rows": [], "message": "No valid coordinates.", "summaries": []}

    coords = df[["_lat", "_lon"]].to_numpy(dtype=float)
    pops = df["_pop"].to_numpy(dtype=float)
    n = len(df)
    total_population = pops.sum()

    print(f"\nAfter filtering: {n} valid areas")
    print(f"Total population: {total_population:,.0f}")

    # Tolerance bands
    lower = target_pop * (1.0 - tolerance)
    upper = target_pop * (1.0 + tolerance)
    expected_clusters = total_population / target_pop if target_pop > 0 else 0

    print(f"\nClustering parameters:")
    print(f"  Acceptable population range: {lower:,.0f} - {upper:,.0f}")
    print(f"  Expected number of clusters: {expected_clusters:.1f}")

    if expected_clusters < 1:
        coverage_analysis = {
            "area_coverage_percent": 0,
            "population_coverage_percent": 0,
            "coverage_quality": "Failed",
            "warnings": [f"Target population ({target_pop:,}) exceeds total population ({total_population:,.0f})"],
            "suggestions": [f"Reduce target population to below {total_population:,.0f}"]
        }
        return {
            "columns": [], "rows": [], "da_rows": [],
            "message": f"Target population too large. Coverage: 0% of areas clustered.",
            "summaries": [],
            "coverage_analysis": coverage_analysis
        }

    # Distance constraints (simplified from original)
    max_radius_km = min(50.0, 25.0)  # More conservative
    max_link_km = max_radius_km * 0.7

    print(f"  Max link distance: {max_link_km:.1f} km")
    print(f"  Max cluster radius: {max_radius_km:.1f} km")

    # Initialize spatial index
    spatial_index = FastSpatialIndex(coords)

    # Track assignments
    assigned = np.full(n, fill_value=-2, dtype=int)  # -2=unassigned, -1=outlier, >=0=cluster_id
    cluster_id = 0
    summaries = []
    areas_processed = 0

    print(f"\nStarting clustering process...")

    # Main clustering loop (same as before)
    while True:
        unassigned_mask = (assigned == -2)
        if not np.any(unassigned_mask):
            break

        unassigned_indices = np.where(unassigned_mask)[0]
        seed = max(unassigned_indices, key=lambda i: pops[i])

        areas_processed += 1
        if areas_processed % 100 == 0:
            print(f"  Processed {areas_processed} seed attempts...")

        # Initialize cluster
        members = {seed}
        running_pop = pops[seed]

        # Greedy growth with distance constraints
        growth_rounds = 0
        max_growth_rounds = 20

        while growth_rounds < max_growth_rounds:
            growth_rounds += 1
            initial_size = len(members)

            member_indices = list(members)
            centroid_lat, centroid_lon = _weighted_centroid(
                coords[member_indices, 0],
                coords[member_indices, 1],
                pops[member_indices]
            )

            candidates = spatial_index.neighbors_within_km(
                centroid_lat, centroid_lon, max_link_km, members
            )

            if not candidates:
                break

            candidates.sort(key=lambda i: _haversine_km(
                centroid_lat, centroid_lon, coords[i, 0], coords[i, 1]
            ))

            added_any = False
            for candidate in candidates:
                if assigned[candidate] != -2:
                    continue

                new_pop = running_pop + pops[candidate]

                if running_pop >= lower:
                    if abs(new_pop - target_pop) >= abs(running_pop - target_pop):
                        continue
                elif new_pop > upper * 1.2:
                    continue

                test_members = member_indices + [candidate]
                test_centroid_lat, test_centroid_lon = _weighted_centroid(
                    coords[test_members, 0],
                    coords[test_members, 1],
                    pops[test_members]
                )

                max_dist = max(_haversine_km(test_centroid_lat, test_centroid_lon,
                                             coords[i, 0], coords[i, 1]) for i in test_members)

                if max_dist > max_radius_km:
                    continue

                members.add(candidate)
                running_pop = new_pop
                added_any = True

                if lower <= running_pop <= upper and abs(running_pop - target_pop) <= target_pop * 0.05:
                    break

            if not added_any:
                break

        # Finalize cluster
        if running_pop >= lower:
            for member in members:
                assigned[member] = cluster_id

            member_indices = list(members)
            final_centroid_lat, final_centroid_lon = _weighted_centroid(
                coords[member_indices, 0],
                coords[member_indices, 1],
                pops[member_indices]
            )

            radius = 0.0 if len(members) == 1 else max(
                _haversine_km(final_centroid_lat, final_centroid_lon,
                              coords[i, 0], coords[i, 1]) for i in member_indices
            )

            summaries.append({
                "cluster_id": cluster_id,
                "total_pop": float(running_pop),
                "centroid_lat": float(final_centroid_lat),
                "centroid_lon": float(final_centroid_lon),
                "radius_km": float(round(radius, 3)),
                "n_points": len(members),
            })

            print(f"  Cluster {cluster_id}: {len(members)} areas, population: {running_pop:,.0f}")
            cluster_id += 1
        else:
            for member in members:
                assigned[member] = -1

    elapsed = time.time() - start_time

    # ENHANCED COVERAGE ANALYSIS
    coverage_analysis = analyze_clustering_coverage(assigned, total_population, target_pop, tolerance, summaries)

    print(f"\n=== CLUSTERING RESULTS ===")
    print(f"Runtime: {elapsed:.2f} seconds")
    print(f"Clusters formed: {len(summaries)}")
    print(
        f"Area coverage: {coverage_analysis['area_coverage_percent']}% ({coverage_analysis['clustered_areas']:,}/{coverage_analysis['total_areas']:,})")
    print(
        f"Population coverage: {coverage_analysis['population_coverage_percent']}% ({sum(s['total_pop'] for s in summaries):,.0f}/{total_population:,.0f})")
    print(f"Coverage quality: {coverage_analysis['coverage_quality']}")
    print(f"Outlier areas: {coverage_analysis['outlier_areas']:,}")

    if coverage_analysis['warnings']:
        print(f"\nWARNINGS:")
        for warning in coverage_analysis['warnings']:
            print(f"  ⚠️  {warning}")

    if coverage_analysis['suggestions']:
        print(f"\nSUGGESTIONS:")
        for suggestion in coverage_analysis['suggestions']:
            print(f"  💡 {suggestion}")

    # Build enhanced message
    coverage_msg = f"Coverage: {coverage_analysis['area_coverage_percent']}% of areas clustered ({coverage_analysis['coverage_quality'].lower()} coverage)"

    if len(summaries) == 0:
        enhanced_msg = f"No clusters formed. {coverage_msg}"
        if coverage_analysis['suggestions']:
            enhanced_msg += f" Try: {coverage_analysis['suggestions'][0]}"
    else:
        base_msg = f"Formed {len(summaries)} clusters in {elapsed:.1f}s. {coverage_msg}"
        if coverage_analysis['area_coverage_percent'] < 70:
            base_msg += f". {coverage_analysis['warnings'][0] if coverage_analysis['warnings'] else ''}"
        enhanced_msg = base_msg

    # Continue with aggregation (same as before)
    df_with_clusters = df.copy()
    df_with_clusters["cluster_id"] = assigned
    df_with_clusters = df_with_clusters.drop(columns=["_lat", "_lon", "_pop"])

    aggregated_df = aggregate_cluster_data(df_with_clusters)

    if not aggregated_df.empty:
        aggregated_rows = aggregated_df.where(pd.notnull(aggregated_df), None).to_dict(orient="records")
        aggregated_cols = aggregated_df.columns.tolist()
    else:
        aggregated_rows = []
        aggregated_cols = []

    original_rows = df_with_clusters.where(pd.notnull(df_with_clusters), None).to_dict(orient="records")

    return {
        "columns": aggregated_cols,
        "rows": aggregated_rows,
        "da_rows": original_rows,
        "message": enhanced_msg,
        "summaries": summaries,
        "cluster_count": len(aggregated_rows),
        "coverage_analysis": coverage_analysis,
        "performance": {
            "elapsed_seconds": elapsed,
            "areas_processed": n,
            "clusters_formed": len(summaries),
            "outliers": coverage_analysis['outlier_areas']
        },
        "parameters_used": {
            "target_pop": target_pop,
            "tolerance_percent": tolerance * 100,
            "acceptable_range": f"{lower:,.0f}-{upper:,.0f}",
            "max_link_km": max_link_km,
            "max_radius_km": max_radius_km,
        },
    }


# Keep the original function name for backward compatibility
def cluster_population(rows, target_pop, tolerance):
    """Wrapper for backward compatibility."""
    return cluster_population_optimized(rows, target_pop, tolerance)