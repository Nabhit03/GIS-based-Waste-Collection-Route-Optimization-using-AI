"""
Collection-point generation and preprocessing.

In a real deployment you'd load actual bin/stop locations (from a municipal
GIS export, OSM POIs like amenity=waste_basket, or IoT smart-bin data) here
instead of simulate_collection_points(). Everything downstream only cares
about the resulting DataFrame schema, so swapping in real data requires no
other code changes.
"""

import numpy as np
import pandas as pd

from . import config
from .network_utils import nearest_node


def simulate_collection_points(G, n_points=config.NUM_COLLECTION_POINTS,
                                seed=config.RANDOM_SEED):
    """
    Simulate n_points waste-collection stops scattered across the road
    network's bounding area, each snapped to the nearest real road node,
    tagged with a stop type (commercial / residential) and demand.
    """
    rng = np.random.default_rng(seed)

    xs = [d["x"] for _, d in G.nodes(data=True)]
    ys = [d["y"] for _, d in G.nodes(data=True)]
    lon_min, lon_max = min(xs), max(xs)
    lat_min, lat_max = min(ys), max(ys)

    records = []
    for i in range(n_points):
        lon = rng.uniform(lon_min, lon_max)
        lat = rng.uniform(lat_min, lat_max)
        node = nearest_node(G, lon, lat)

        is_commercial = rng.random() < config.COMMERCIAL_FRACTION
        demand = int(rng.integers(1, 4)) if not is_commercial else int(rng.integers(2, 6))

        # Time window: commercial stops must be done before 9 AM;
        # residential stops are flexible across the whole shift.
        if is_commercial:
            tw_start = config.SHIFT_START_MINUTE
            tw_end = config.COMMERCIAL_DEADLINE_MINUTE
        else:
            tw_start = config.SHIFT_START_MINUTE
            tw_end = config.SHIFT_END_MINUTE

        records.append({
            "stop_id": i + 1,
            "node": node,
            "lon": G.nodes[node]["x"],
            "lat": G.nodes[node]["y"],
            "type": "commercial" if is_commercial else "residential",
            "demand": demand,
            "tw_start": tw_start,
            "tw_end": tw_end,
            "service_time": config.SERVICE_TIME_MINUTES,
        })

    df = pd.DataFrame.from_records(records)
    df = df.drop_duplicates(subset="node").reset_index(drop=True)
    df["stop_id"] = range(1, len(df) + 1)
    print(f"[data_prep] Generated {len(df)} unique collection stops "
          f"({(df['type'] == 'commercial').sum()} commercial, "
          f"{(df['type'] == 'residential').sum()} residential).")
    return df


def add_depot(G, df, seed=config.RANDOM_SEED):
    """
    Insert a depot row at the top of the DataFrame (index 0), placed near the
    centroid of all stops (proxy for a municipal transfer station / landfill).
    Depot has no demand and a time window spanning the whole shift.
    """
    centroid_lon = df["lon"].mean()
    centroid_lat = df["lat"].mean()
    depot_node = nearest_node(G, centroid_lon, centroid_lat)

    depot_row = {
        "stop_id": 0,
        "node": depot_node,
        "lon": G.nodes[depot_node]["x"],
        "lat": G.nodes[depot_node]["y"],
        "type": "depot",
        "demand": 0,
        "tw_start": config.SHIFT_START_MINUTE,
        "tw_end": config.SHIFT_END_MINUTE,
        "service_time": 0,
    }
    df_out = pd.concat([pd.DataFrame([depot_row]), df], ignore_index=True)
    df_out["stop_id"] = range(len(df_out))  # depot = 0, stops = 1..n
    return df_out
