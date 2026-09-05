"""
Capacitated VRP with Time Windows (CVRPTW), solved with Google OR-Tools.

This replaces the weaker "K-Means zone -> independent TSP per zone" design.
Here, all trucks and all stops are optimized jointly in a single model, with:
  - Vehicle capacity constraints (a truck can't exceed its waste load limit)
  - Time windows (commercial stops must be visited before 9 AM)
  - A shared depot (start/end point for every route)
  - Workload balancing across trucks (stop-count dimension, see below)

K-Means anchors (from clustering.py) are used only to give the solver's
first solution strategy a geographically sensible starting point.

--- Why the stop-count dimension exists ---
Left alone, OR-Tools' CVRP solver only uses as many vehicles as it needs to
minimize total travel cost. If 3 trucks can comfortably cover all demand
within capacity/time-window limits, it will leave the other trucks with
empty routes and/or produce very uneven stop counts -- purely because fewer
depot-to-first-stop / last-stop-to-depot legs means slightly less total
distance. That's mathematically correct given "minimize total time" as the
sole objective, but operationally undesirable: you're paying for N trucks
and drivers and want them all used, with roughly comparable shift lengths.

Two independent knobs (both driven by a single stop-count dimension) fix
this without needing a different solver or objective:
  1. A *global span cost coefficient* penalizes the gap between the busiest
     and least-busy vehicle -- this directly targets workload_std_stops.
  2. A *hard minimum stop count per vehicle* prevents any truck ending with
     zero (or near-zero) stops -- this directly fixes empty routes.

Knob 2 is a hard constraint, so it's possible (with unlucky demand/time-
window geometry) for it to make the problem infeasible even though the
unconstrained problem is feasible. solve_cvrptw() below handles that
automatically: if the solve with the minimum-stops floor fails, it retries
once with the floor removed (balancing via knob 1 still applies) and prints
a clear warning, so the pipeline never silently returns nothing or crashes
because of this setting.
"""

import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from . import config


def _build_and_solve(time_matrix, demands, time_windows, num_vehicles,
                      vehicle_capacity, service_times, depot_index,
                      time_limit_seconds, workload_balance_weight,
                      min_stops_per_vehicle):
    """
    Build one RoutingModel and attempt to solve it.

    min_stops_per_vehicle : int or None
        If set (> 0), every vehicle's stop-count dimension end-cumul is
        floored at this value, forcing the solver to assign at least that
        many stops to every vehicle. Pass None/0 to disable.

    Returns (routes, total_time_minutes, status) same as solve_cvrptw,
    where status is "OK" or "INFEASIBLE".
    """
    n = len(time_matrix)
    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    # --- Travel time callback (also used as the arc cost) ---
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(round(time_matrix[from_node][to_node] + service_times[from_node]))

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # --- Capacity constraint (dimension) ---
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # no slack
        [vehicle_capacity] * num_vehicles,
        True,  # start cumul to zero
        "Capacity",
    )

    # --- Time window constraint (dimension) ---
    horizon = config.SHIFT_END_MINUTE
    routing.AddDimension(
        transit_callback_index,
        60,       # allow up to 60 min waiting slack (e.g., early arrival)
        horizon,  # max time per vehicle route
        False,    # don't force start cumul to zero (depot has a window too)
        "Time",
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    # Time windows are enforced as a SOFT upper bound (heavy penalty for
    # lateness) rather than a hard SetRange. On small/synthetic data every
    # stop happens to fit its window, so this made no visible difference --
    # but on real road-network travel times, a handful of zones/stops can
    # genuinely be unreachable by their deadline given the depot's real
    # position. A hard bound then makes the ENTIRE joint/zone solve
    # infeasible over one or two unreachable stops, discarding an otherwise
    # good solution. Soft bounds let the solver strongly prefer on-time
    # arrivals (and it usually finds them) while still finding *a* solution
    # when a genuine deadline miss is unavoidable -- exactly the "surfaces
    # missed-SLA risk" metric the README describes, rather than a crash.
    # The lower bound (can't service before a window opens) stays hard --
    # arriving early just means waiting, which is free and never infeasible.
    late_penalty = getattr(config, "TIME_WINDOW_LATE_PENALTY_PER_MINUTE", 500)
    for node_idx, (tw_start, tw_end) in enumerate(time_windows):
        index = manager.NodeToIndex(node_idx)
        time_dimension.CumulVar(index).SetMin(int(tw_start))
        if late_penalty and late_penalty > 0:
            time_dimension.SetCumulVarSoftUpperBound(index, int(tw_end), int(late_penalty))
        else:
            time_dimension.CumulVar(index).SetMax(int(tw_end))

    for vehicle_id in range(num_vehicles):
        start_index = routing.Start(vehicle_id)
        end_index = routing.End(vehicle_id)
        time_dimension.CumulVar(start_index).SetRange(
            config.SHIFT_START_MINUTE, config.SHIFT_END_MINUTE)
        time_dimension.CumulVar(end_index).SetRange(
            config.SHIFT_START_MINUTE, config.SHIFT_END_MINUTE)

    # --- Stop-count dimension: workload balance + min-stops-per-vehicle ---
    # Only built when actually needed (skipped for e.g. single-vehicle
    # per-zone sub-problems, where balancing/floors are meaningless and
    # this dimension would only add solver overhead for no benefit).
    if workload_balance_weight or min_stops_per_vehicle:
        # Counts 1 per real stop, 0 for the depot.
        def count_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return 0 if from_node == depot_index else 1

        count_callback_index = routing.RegisterUnaryTransitCallback(count_callback)
        routing.AddDimension(
            count_callback_index,
            0,  # no slack
            n,  # effectively unconstrained upper bound (total stops incl. depot)
            True,  # start cumul to zero
            "StopCount",
        )
        count_dimension = routing.GetDimensionOrDie("StopCount")

        if workload_balance_weight:
            # Penalizes (max stop-count - min stop-count) across vehicles.
            count_dimension.SetGlobalSpanCostCoefficient(int(workload_balance_weight))

        if min_stops_per_vehicle:
            for vehicle_id in range(num_vehicles):
                end_index = routing.End(vehicle_id)
                count_dimension.CumulVar(end_index).SetMin(int(min_stops_per_vehicle))

    # --- Search strategy ---
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.FromSeconds(time_limit_seconds)

    solution = routing.SolveWithParameters(search_parameters)

    if solution is None:
        return None, None, "INFEASIBLE"

    routes = {}
    total_time = 0.0
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        route_time = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != depot_index:
                route.append(node)
            prev_index = index
            index = solution.Value(routing.NextVar(index))
            route_time += routing.GetArcCostForVehicle(prev_index, index, vehicle_id)
        routes[vehicle_id] = route
        total_time += route_time

    return routes, total_time, "OK"


def solve_cvrptw(time_matrix, demands, time_windows, num_vehicles,
                  vehicle_capacity, service_times, depot_index=0,
                  time_limit_seconds=15,
                  workload_balance_weight=None,
                  enforce_min_stops=None,
                  min_stops_fraction=None):
    """
    Solve a Capacitated VRP with Time Windows, with workload balancing and
    (optionally, with automatic fallback) a minimum-stops-per-vehicle floor.

    Parameters
    ----------
    time_matrix : 2D array (minutes), shape (n_stops+1, n_stops+1), incl. depot
    demands : list[int], demand per stop (depot demand = 0)
    time_windows : list[(start, end)] in minutes, per stop (depot spans full shift)
    num_vehicles : int, number of trucks
    vehicle_capacity : int, max demand units per truck per trip
    service_times : list[int], minutes spent servicing each stop
    depot_index : int, index of the depot node in all arrays
    time_limit_seconds : solver time budget (applied per solve attempt)
    workload_balance_weight : int or None
        Span-cost coefficient balancing stop-count across vehicles. Defaults
        to config.WORKLOAD_BALANCE_WEIGHT if None. 0 disables balancing.
    enforce_min_stops : bool or None
        Whether to try enforcing a minimum stop count per vehicle. Defaults
        to config.ENFORCE_MIN_STOPS_PER_VEHICLE if None.
    min_stops_fraction : float or None
        Fraction of the "fair share" (avg stops per vehicle) each vehicle
        must be assigned, if enforce_min_stops is True. Defaults to
        config.MIN_STOPS_FRACTION_OF_FAIR_SHARE if None.

    Returns
    -------
    routes : dict[vehicle_id] -> list of stop indices (in visiting order,
             excluding depot at start/end which is implied)
    total_time_minutes : float, sum of travel time across all vehicles
    solver_status : str
        "OK" (min-stops applied successfully, or was disabled),
        "OK_MIN_STOPS_RELAXED" (min-stops was requested but infeasible, so
        the solver fell back to balancing-only and still found a solution),
        or "INFEASIBLE" (no solution even without the min-stops floor).
    """
    if workload_balance_weight is None:
        if getattr(config, "WORKLOAD_BALANCE_AUTO_SCALE", False):
            nonzero = time_matrix[time_matrix > 0]
            avg_edge_time = float(np.mean(nonzero)) if nonzero.size else 1.0
            multiplier = getattr(config, "WORKLOAD_BALANCE_AUTO_MULTIPLIER", 5)
            workload_balance_weight = max(1, int(round(avg_edge_time * multiplier)))
            print(f"[vrp_solver] Auto-scaled workload balance weight -> "
                  f"{workload_balance_weight} (avg stop-to-stop time="
                  f"{avg_edge_time:.1f} min x multiplier={multiplier})")
        else:
            workload_balance_weight = getattr(config, "WORKLOAD_BALANCE_WEIGHT", 0)
    if enforce_min_stops is None:
        enforce_min_stops = getattr(config, "ENFORCE_MIN_STOPS_PER_VEHICLE", False)
    if min_stops_fraction is None:
        min_stops_fraction = getattr(config, "MIN_STOPS_FRACTION_OF_FAIR_SHARE", 0.5)

    n_real_stops = len(time_matrix) - 1  # exclude depot
    min_stops_per_vehicle = None
    if enforce_min_stops and num_vehicles > 0 and n_real_stops > 0:
        fair_share = n_real_stops / num_vehicles
        min_stops_per_vehicle = max(1, int(fair_share * min_stops_fraction))
        # Never require more stops per vehicle, in total, than actually exist --
        # that would be trivially infeasible and there's no point attempting it.
        if min_stops_per_vehicle * num_vehicles > n_real_stops:
            min_stops_per_vehicle = max(1, n_real_stops // num_vehicles)

    if min_stops_per_vehicle:
        routes, total_time, status = _build_and_solve(
            time_matrix, demands, time_windows, num_vehicles,
            vehicle_capacity, service_times, depot_index,
            time_limit_seconds, workload_balance_weight,
            min_stops_per_vehicle,
        )
        if status == "OK":
            return routes, total_time, "OK"

        print(f"[vrp_solver] WARNING: could not find a feasible solution with "
              f"a minimum of {min_stops_per_vehicle} stops/vehicle enforced "
              f"(ENFORCE_MIN_STOPS_PER_VEHICLE in config.py). This can happen "
              f"when capacity or time-window constraints don't allow an even "
              f"split. Falling back to balancing-only (no hard minimum) -- "
              f"some trucks may still end up with few or zero stops. Consider "
              f"lowering MIN_STOPS_FRACTION_OF_FAIR_SHARE, raising "
              f"TRUCK_CAPACITY, or reducing NUM_TRUCKS if this warning persists.")

        routes, total_time, status = _build_and_solve(
            time_matrix, demands, time_windows, num_vehicles,
            vehicle_capacity, service_times, depot_index,
            time_limit_seconds, workload_balance_weight,
            None,  # no hard minimum this time
        )
        if status == "OK":
            return routes, total_time, "OK_MIN_STOPS_RELAXED"

        print("[vrp_solver] Solver returned no solution even without the "
              "min-stops floor. Try raising TRUCK_CAPACITY or NUM_TRUCKS, or "
              "loosening COMMERCIAL_DEADLINE_MINUTE / SHIFT_END_MINUTE.")
        return None, None, "INFEASIBLE"

    # enforce_min_stops disabled (or fair_share math gave nothing to enforce) --
    # single solve attempt, balancing-only.
    routes, total_time, status = _build_and_solve(
        time_matrix, demands, time_windows, num_vehicles,
        vehicle_capacity, service_times, depot_index,
        time_limit_seconds, workload_balance_weight,
        None,
    )
    if status != "OK":
        print("[vrp_solver] Solver returned no solution.")
        return None, None, "INFEASIBLE"
    return routes, total_time, "OK"
