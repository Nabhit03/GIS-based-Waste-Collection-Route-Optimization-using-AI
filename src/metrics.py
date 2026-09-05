"""
Evaluation metrics for comparing baseline vs optimized routing.
"""

import numpy as np
from . import config


def pct_reduction(baseline, optimized):
    if baseline == 0:
        return 0.0
    return 100.0 * (baseline - optimized) / baseline


def route_edges(route, depot_index=0):
    """Return the set of directed edges (as frozensets, undirected) used by a route."""
    full = [depot_index] + route + [depot_index]
    return {frozenset((full[i], full[i + 1])) for i in range(len(full) - 1)}


def overlap_edges_count(routes_dict, depot_index=0):
    """
    Count edges that are reused across more than one vehicle's route
    (excluding the depot-adjacent edges, since every truck legitimately
    starts/ends there).
    """
    edge_owner_count = {}
    for _, route in routes_dict.items():
        for e in route_edges(route, depot_index):
            if depot_index in e:
                continue  # depot connections are expected to repeat
            edge_owner_count[e] = edge_owner_count.get(e, 0) + 1
    overlapping = {e: c for e, c in edge_owner_count.items() if c > 1}
    return len(overlapping)


def workload_balance_std(routes_dict):
    """Std. deviation of number of stops assigned per vehicle (lower = fairer)."""
    counts = [len(r) for r in routes_dict.values()]
    return float(np.std(counts)) if counts else 0.0


def estimate_fuel_and_co2(distance_km):
    fuel_l = distance_km / config.TRUCK_FUEL_ECONOMY_KM_PER_L
    co2_kg = fuel_l * config.DIESEL_CO2_KG_PER_L
    return fuel_l, co2_kg


def time_to_distance_km(time_minutes, avg_speed_kmph=config.DEFAULT_SPEED_KMPH):
    """Rough distance-from-time conversion for fuel/CO2 estimates when we only
    tracked travel time in the solver (time and distance are highly correlated
    on a fixed-speed-assumption graph)."""
    hours = time_minutes / 60.0
    return hours * avg_speed_kmph


def count_time_window_violations(routes_dict, df_stops, time_matrix, service_times, depot_index=0):
    """
    Walk each route and check whether every stop is reached within its time
    window. The naive nearest-neighbor baseline doesn't enforce time windows
    at all, so this typically reveals missed commercial-deadline pickups --
    a real operational cost that a pure distance/time comparison hides.
    """
    violations = 0
    tw_start = df_stops["tw_start"].tolist()
    tw_end = df_stops["tw_end"].tolist()
    for _, route in routes_dict.items():
        clock = 0.0
        current = depot_index
        for stop in route:
            clock += time_matrix[current][stop]
            clock = max(clock, tw_start[stop])  # can't service before window opens
            if clock > tw_end[stop]:
                violations += 1
            clock += service_times[stop]
            current = stop
    return violations


def summarize(name, total_time_minutes, routes_dict, depot_index=0,
              df_stops=None, time_matrix=None, service_times=None):
    distance_km = time_to_distance_km(total_time_minutes)
    fuel_l, co2_kg = estimate_fuel_and_co2(distance_km)
    result = {
        "strategy": name,
        "total_time_min": round(total_time_minutes, 1),
        "approx_distance_km": round(distance_km, 2),
        "overlap_edges": overlap_edges_count(routes_dict, depot_index),
        "workload_std_stops": round(workload_balance_std(routes_dict), 2),
        "fuel_liters": round(fuel_l, 2),
        "co2_kg": round(co2_kg, 2),
    }
    if df_stops is not None and time_matrix is not None and service_times is not None:
        result["tw_violations"] = count_time_window_violations(
            routes_dict, df_stops, time_matrix, service_times, depot_index)
    return result
