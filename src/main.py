"""
End-to-end pipeline entry point.

Run with:
    python -m src.main

This will:
  1. Load a road network (real OSM if internet is available, else synthetic)
  2. Simulate/load collection points with commercial time windows
  3. Run K-Means as a zone initializer
  4. Compute baseline routes (Tier 1: naive NN, Tier 2: TSP-per-zone)
  5. Solve the full joint CVRPTW (Tier 3, final approach)
  6. Print a comparison table of all three tiers
  7. Save interactive Folium maps to outputs/
"""

import os
import pandas as pd

from . import config
from .network_utils import get_road_network, build_travel_time_matrix
from .data_prep import simulate_collection_points, add_depot
from .clustering import initial_zone_clusters
from .baselines import baseline_naive_per_zone, baseline_tsp_per_zone
from .vrp_solver import solve_cvrptw
from .metrics import summarize, pct_reduction
from .visualize import plot_zones, plot_routes


def run_pipeline():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 1. Road network
    G, is_real = get_road_network()

    # 2. Collection points + depot
    df_stops = simulate_collection_points(G)
    df_stops = add_depot(G, df_stops)

    # 3. K-Means zone initializer
    df_stops, anchors = initial_zone_clusters(df_stops)

    # 4. Travel-time matrix (minutes) over real road-network shortest paths
    node_ids = df_stops["node"].tolist()
    time_matrix = build_travel_time_matrix(G, node_ids)

    demands = df_stops["demand"].tolist()
    time_windows = list(zip(df_stops["tw_start"], df_stops["tw_end"]))
    service_times = df_stops["service_time"].tolist()
    depot_index = 0

    # Safety check: with a connected road network (see network_utils.py),
    # every stop should be reachable from every other stop. If this trips,
    # something upstream (e.g. a bad snap-to-node) produced an isolated point.
    unreachable_pairs = int((time_matrix > 1e5).sum())
    if unreachable_pairs > 0:
        print(f"[main] WARNING: {unreachable_pairs} stop-pairs are unreachable "
              f"on the road network even after restricting to the largest "
              f"connected component. Results may be unreliable. Consider "
              f"regenerating collection points or checking city name spelling.")

    plot_zones(df_stops)

    # --- Tier 1: naive nearest-neighbor per K-Means zone ---
    t1_routes, t1_time = baseline_naive_per_zone(time_matrix, df_stops, service_times, depot_index)
    t1_summary = summarize("Tier 1: Naive NN per zone", t1_time, t1_routes, depot_index,
                            df_stops, time_matrix, service_times)
    plot_routes(G, df_stops, t1_routes, "Tier 1: Naive nearest-neighbor per zone",
                f"{config.OUTPUT_DIR}/tier1_naive_routes.html", depot_index)

    # --- Tier 2: independent TSP (CVRPTW, 1 vehicle) per K-Means zone ---
    t2_routes, t2_time = baseline_tsp_per_zone(
        time_matrix, df_stops, demands, time_windows, service_times, depot_index)
    t2_summary = summarize("Tier 2: K-Means + TSP per zone", t2_time, t2_routes, depot_index,
                            df_stops, time_matrix, service_times)
    plot_routes(G, df_stops, t2_routes, "Tier 2: K-Means zones + independent TSP",
                f"{config.OUTPUT_DIR}/tier2_clustered_tsp_routes.html", depot_index)

    # --- Tier 3: full joint CVRPTW across all trucks (final approach) ---
    t3_routes, t3_time, status = solve_cvrptw(
        time_matrix, demands, time_windows,
        num_vehicles=config.NUM_TRUCKS,
        vehicle_capacity=config.TRUCK_CAPACITY,
        service_times=service_times,
        depot_index=depot_index,
        time_limit_seconds=20,
    )
    if status == "INFEASIBLE":
        print("[main] WARNING: joint CVRPTW solver could not find a feasible "
              "solution with current capacity/time-window settings. Try "
              "raising TRUCK_CAPACITY or NUM_TRUCKS in config.py.")
        t3_summary = None
    else:
        if status == "OK_MIN_STOPS_RELAXED":
            print("[main] NOTE: the minimum-stops-per-vehicle floor "
                  "(ENFORCE_MIN_STOPS_PER_VEHICLE) was infeasible given "
                  "current capacity/time-window settings, so Tier 3 fell "
                  "back to workload balancing only. Some trucks may still "
                  "have few or zero stops -- see the warning above for the "
                  "config knobs to try.")
        t3_summary = summarize("Tier 3: Joint CVRP + Time Windows (final)", t3_time, t3_routes, depot_index,
                                df_stops, time_matrix, service_times)
        plot_routes(G, df_stops, t3_routes, "Tier 3: Joint CVRP + Time Windows (optimized)",
                    f"{config.OUTPUT_DIR}/tier3_optimized_routes.html", depot_index)

    # --- Comparison table ---
    rows = [t1_summary, t2_summary]
    if t3_summary:
        rows.append(t3_summary)
    df_results = pd.DataFrame(rows)

    if t3_summary:
        improvement_vs_naive = pct_reduction(t1_summary["total_time_min"], t3_summary["total_time_min"])
        improvement_vs_clustered = pct_reduction(t2_summary["total_time_min"], t3_summary["total_time_min"])
        df_results["% reduction vs Tier 1 (naive)"] = [
            0.0,
            round(pct_reduction(t1_summary["total_time_min"], t2_summary["total_time_min"]), 1),
            round(improvement_vs_naive, 1),
        ]
    print("\n=== ROUTE OPTIMIZATION COMPARISON ===")
    print(df_results.to_string(index=False))
    df_results.to_csv(f"{config.OUTPUT_DIR}/comparison_metrics.csv", index=False)
    print(f"\n[main] Saved metrics table -> {config.OUTPUT_DIR}/comparison_metrics.csv")
    print(f"[main] Road network source: {'REAL OpenStreetMap data' if is_real else 'SYNTHETIC fallback grid (no internet access in this environment)'}")

    return df_results, {"tier1": t1_routes, "tier2": t2_routes, "tier3": t3_routes if t3_summary else None}


if __name__ == "__main__":
    run_pipeline()
