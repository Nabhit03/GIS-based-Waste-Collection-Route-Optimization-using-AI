# GIS-Based Waste Collection Route Optimization

Optimizes municipal waste-collection routes over a **real road network**, using
K-Means as a zone initializer and **Google OR-Tools** to solve a **Capacitated
Vehicle Routing Problem with Time Windows (CVRPTW)** — trucks have load
capacity limits and commercial stops must be collected before 9 AM.

## Why this design (not just "K-Means + TSP")

A naive "cluster with K-Means, then solve independent TSPs" pipeline has two
weaknesses:
1. K-Means clusters on straight-line lat/lon distance and ignores truck
   capacity — it can produce zones that look balanced on a map but are
   infeasible or inefficient in practice.
2. Solving each zone's route independently prevents stops from being
   reassigned to a *better* truck, even when that would be optimal.

This project instead uses K-Means only to **seed** zone assignment for speed
and interpretability, then solves a **single joint CVRPTW** across all
trucks with OR-Tools, which is free to reassign any stop based on real
travel time, capacity, and time-window constraints.

To make the final improvement number credible, the pipeline benchmarks
**three tiers** rather than reporting one number in isolation:

| Tier | Method | What it isolates |
|---|---|---|
| 1 | Naive nearest-neighbor per K-Means zone | Realistic "before" state |
| 2 | K-Means zones + independent TSP per zone (OR-Tools, 1 vehicle each) | Benefit of using a real solver, with zones still fixed |
| 3 | **Joint CVRPTW across all trucks (final)** | Benefit of joint optimization + time windows |

## Pipeline

```
Road network (OSMnx, real OSM data)
        ↓
Collection points + time windows (real or simulated)
        ↓
K-Means zone initializer
        ↓
Road-network travel-time matrix (NetworkX Dijkstra)
        ↓
Tier 1: naive NN  |  Tier 2: TSP per zone  |  Tier 3: joint CVRPTW  (OR-Tools)
        ↓
Metrics: time/distance saved, fuel, CO2, overlap, workload balance, TW violations
        ↓
Interactive Folium maps (before/after)
```

## Setup

```bash
pip install -r requirements.txt
```

Requires internet access to `nominatim.openstreetmap.org` and
`overpass-api.de` to pull a real city's road network via OSMnx. **If those
hosts aren't reachable** (e.g. a sandboxed/offline environment), the
pipeline automatically falls back to a synthetic grid road network so you
can still run and validate the full pipeline — no code changes needed.
Just run it again on a machine/network with normal internet access to get
real OSM data.

## Run

```bash
python -m src.main
```

This prints a comparison table across all three tiers and writes to
`outputs/`:
- `zones_map.html` — K-Means zone assignment
- `tier1_naive_routes.html`, `tier2_clustered_tsp_routes.html`,
  `tier3_optimized_routes.html` — route maps for each tier, drawn on the
  real road network
- `comparison_metrics.csv` — full metrics table

## Configuration

Edit `src/config.py` to change:
- `CITY_NAME` — target city (used when OSM is reachable)
- `NUM_TRUCKS`, `TRUCK_CAPACITY` — fleet size and per-truck load limit
  (keep `NUM_TRUCKS * TRUCK_CAPACITY` comfortably above total simulated
  demand or the solver will report infeasible)
- `COMMERCIAL_DEADLINE_MINUTE` — the hard morning cutoff for commercial stops
- `NUM_COLLECTION_POINTS`, `COMMERCIAL_FRACTION` — demand simulation

## Using real data instead of simulated points

Replace the call to `simulate_collection_points()` in `src/main.py` with a
loader that reads your own bin/stop CSV or shapefile into a DataFrame with
these columns: `stop_id, node, lon, lat, type, demand, tw_start, tw_end,
service_time`. Everything downstream (clustering, solving, visualization)
works unchanged.

## Metrics reported

- Total travel time & approximate distance (from real road-network shortest
  paths, not straight-line)
- Estimated fuel use and CO2 emissions (based on average garbage-truck fuel
  economy)
- Route overlap (shared street segments across trucks — a proxy for
  redundant coverage)
- Workload balance (std. dev. of stops per truck)
- **Time-window violations** — the naive baseline ignores commercial
  deadlines entirely, so this metric surfaces missed-SLA risk that a pure
  distance comparison would hide

## Known limitations / future work

- No real-time traffic (uses a fixed average speed assumption per edge)
- Single-day, single-shift model — no multi-day scheduling (e.g. some bins
  collected 2x/week)
- No IoT/dynamic bin-fill-level input — demand is static per run
- OR-Tools' `PATH_CHEAPEST_ARC` + `GUIDED_LOCAL_SEARCH` is a strong but
  heuristic solver — not a proven global optimum for large instances
