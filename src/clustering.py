"""
K-Means is used here as a ZONE INITIALIZER, not as the final routing
decision. Plain Euclidean K-Means on lat/lon ignores road topology and
truck capacity, so using its output as the final answer produces zones that
look balanced on a map but can be poor in practice (e.g. split by a river
with no bridge, or capacity-imbalanced).

Instead we use it to:
  1. Give OR-Tools's VRP solver good initial vehicle "anchor" points, which
     speeds up convergence and produces geographically sensible routes.
  2. Provide a quick visual sanity check of zone balance before solving.

The OR-Tools CVRP solver in vrp_solver.py is free to reassign any stop to
any vehicle based on real travel time, capacity, and time windows -- K-Means
never has the final say on which truck visits which stop.
"""

import numpy as np
from sklearn.cluster import KMeans

from . import config


def initial_zone_clusters(df_stops, n_clusters=config.NUM_TRUCKS,
                           seed=config.RANDOM_SEED):
    """
    Run K-Means on (lon, lat) of non-depot stops to get an initial,
    geographically-balanced zone assignment and per-zone anchor points.

    Returns
    -------
    df_stops : DataFrame with an added 'zone' column (depot excluded, gets -1)
    anchors : list of (lon, lat) cluster centers, one per truck
    """
    df = df_stops.copy()
    mask = df["type"] != "depot"
    coords = df.loc[mask, ["lon", "lat"]].values

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(coords)

    df["zone"] = -1
    df.loc[mask, "zone"] = labels

    counts = df.loc[mask, "zone"].value_counts().sort_index()
    print(f"[clustering] K-Means initial zone sizes (stops per truck): "
          f"{counts.to_dict()}")

    anchors = km.cluster_centers_.tolist()
    return df, anchors
