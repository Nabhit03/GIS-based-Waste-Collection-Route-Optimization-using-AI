"""
Baseline routing strategies, used to produce a defensible, three-tier
comparison rather than a single unexplained improvement number:

  1. Naive nearest-neighbor per K-Means zone   (realistic "before" state --
     not literally random order, since no real crew routes randomly)
  2. K-Means zones + independent TSP per zone  (your original design)
  3. Full CVRPTW solved jointly (vrp_solver.py) (the final, best approach)

Comparing all three lets you report a genuine ablation:
naive -> +clustering -> +joint CVRP with time windows.
"""

import numpy as np

from .vrp_solver import solve_cvrptw
from . import config


def nearest_neighbor_route(time_matrix, stop_indices, service_times, depot_index=0):
    """
    Greedy nearest-neighbor tour over a set of stops, starting/ending at depot.
    Includes per-stop service time so this is directly comparable to the
    OR-Tools tiers, which also charge service time on their arc cost.
    """
    unvisited = set(stop_indices)
    route = []
    current = depot_index
    total_time = 0.0
    while unvisited:
        nxt = min(unvisited, key=lambda j: time_matrix[current][j])
        total_time += time_matrix[current][nxt] + service_times[current]
        route.append(nxt)
        current = nxt
        unvisited.remove(nxt)
    total_time += time_matrix[current][depot_index] + service_times[current]  # return to depot
    return route, total_time


def baseline_naive_per_zone(time_matrix, df_stops, service_times, depot_index=0):
    """Tier 1: nearest-neighbor route computed independently per K-Means zone."""
    total_time = 0.0
    routes = {}
    for zone_id in sorted(df_stops.loc[df_stops["zone"] >= 0, "zone"].unique()):
        stop_indices = df_stops.index[df_stops["zone"] == zone_id].tolist()
        route, t = nearest_neighbor_route(time_matrix, stop_indices, service_times, depot_index)
        routes[zone_id] = route
        total_time += t
    return routes, total_time


def baseline_tsp_per_zone(time_matrix, df_stops, demands, time_windows,
                           service_times, depot_index=0):
    """
    Tier 2: solve an independent single-vehicle CVRPTW (i.e. a constrained TSP)
    for each K-Means zone. Uses the same OR-Tools engine as the final solver,
    but with hard zone boundaries -- so it isolates the benefit of *joint*
    optimization (Tier 3) from the benefit of using OR-Tools at all.
    """
    total_time = 0.0
    routes = {}
    for zone_id in sorted(df_stops.loc[df_stops["zone"] >= 0, "zone"].unique()):
        zone_stop_idx = df_stops.index[df_stops["zone"] == zone_id].tolist()
        sub_indices = [depot_index] + zone_stop_idx
        sub_matrix = time_matrix[np.ix_(sub_indices, sub_indices)]
        sub_demands = [0] + [demands[i] for i in zone_stop_idx]
        sub_windows = [time_windows[depot_index]] + [time_windows[i] for i in zone_stop_idx]
        sub_service = [0] + [service_times[i] for i in zone_stop_idx]

        result_routes, t, status = solve_cvrptw(
            sub_matrix, sub_demands, sub_windows,
            num_vehicles=1,
            vehicle_capacity=config.TRUCK_CAPACITY,
            service_times=sub_service,
            depot_index=0,
            time_limit_seconds=8,
            # Min-stops-per-vehicle is meaningless with a single vehicle --
            # every stop in the zone is already forced onto it -- and workload
            # balancing needs >=2 vehicles to mean anything. Explicitly
            # disabling both here avoids solving each zone's sub-problem
            # twice (once with a pointless floor, once without) inside the
            # tight per-zone time budget, which was causing spurious timeouts.
            workload_balance_weight=0,
            enforce_min_stops=False,
        )
        if status not in ("OK", "OK_MIN_STOPS_RELAXED"):
            print(f"[baselines] Tier 2 zone {zone_id} ({len(zone_stop_idx)} stops) "
                  f"could not be solved even as an unconstrained single-vehicle "
                  f"route -- skipping it. This usually means real travel times "
                  f"make the shift window (SHIFT_END_MINUTE) too tight for this "
                  f"zone's stop count; consider smaller K-Means zones (more "
                  f"trucks) or a longer SHIFT_END_MINUTE.")
            continue
        # map local sub-indices back to global stop indices
        global_route = [sub_indices[local_i] for local_i in result_routes[0]]
        routes[zone_id] = global_route
        total_time += t
    return routes, total_time
