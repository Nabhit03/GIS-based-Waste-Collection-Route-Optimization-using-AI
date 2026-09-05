"""
Road network acquisition and shortest-path utilities.

Primary path: pull a real road network from OpenStreetMap via OSMnx and
compute travel times using edge length + speed.

Fallback path: if there is no internet access to OSM (e.g. sandboxed
environments, CI, offline demos), build a synthetic but realistic grid-based
road network so the rest of the pipeline can still be run and tested end to
end. Swap USE_REAL_OSM = True (or just call get_road_network()) on a machine
with internet access to use real city data with zero other code changes.
"""

import random
import numpy as np
import networkx as nx

from . import config


def get_road_network(city_name: str = config.CITY_NAME,
                      network_type: str = config.NETWORK_TYPE):
    """
    Try to fetch a real drivable road network for `city_name` using OSMnx.
    Falls back to a synthetic grid network if OSM/internet is unavailable.

    Returns
    -------
    G : networkx.MultiDiGraph
        Road network with 'length' (meters) and 'travel_time' (minutes)
        edge attributes, and 'x','y' (lon/lat) node attributes.
    is_real : bool
        True if the graph came from OpenStreetMap, False if synthetic.
    """
    try:
        import osmnx as ox
        G = ox.graph_from_place(city_name, network_type=network_type)
        G = ox.add_edge_speeds(G, fallback=config.DEFAULT_SPEED_KMPH)
        G = ox.add_edge_travel_times(G)  # adds 'travel_time' in seconds
        # convert seconds -> minutes for consistency with the rest of the pipeline
        for u, v, k, data in G.edges(keys=True, data=True):
            data["travel_time"] = data["travel_time"] / 60.0

        # Real OSM drive networks are frequently NOT fully strongly connected
        # (one-way streets, service roads, disconnected slip lanes, etc.).
        # If two collection points land in different components, the
        # travel-time matrix silently fills in a huge penalty value for that
        # pair, which then breaks the OR-Tools model (arc cost/time-window
        # dimensions overflow -> ROUTING_INVALID) and produces absurd totals.
        # Restrict to the largest strongly connected component so every node
        # can reach every other node.
        import networkx as nx
        if not nx.is_strongly_connected(G):
            largest_cc_nodes = max(nx.strongly_connected_components(G), key=len)
            n_before = len(G.nodes)
            G = G.subgraph(largest_cc_nodes).copy()
            print(f"[network_utils] Road network was not fully connected. "
                  f"Kept largest strongly-connected component: "
                  f"{len(G.nodes)}/{n_before} nodes retained.")

        print(f"[network_utils] Loaded real OSM road network for '{city_name}' "
              f"({len(G.nodes)} nodes, {len(G.edges)} edges).")
        return G, True
    except Exception as e:
        print(f"[network_utils] Could not fetch real OSM data ({e}).\n"
              f"[network_utils] Falling back to a synthetic grid road network "
              f"for local testing.")
        return _build_synthetic_grid_network(), False


def _build_synthetic_grid_network(grid_size: int = 14, spacing_m: float = 180.0,
                                   seed: int = config.RANDOM_SEED):
    """
    Build a synthetic Manhattan-style grid network that mimics a real city's
    block structure closely enough to validate clustering + routing logic
    without needing internet access.
    """
    random.seed(seed)
    np.random.seed(seed)

    G = nx.MultiDiGraph()
    # Rough lat/lon origin (used only for realistic-looking coordinates on maps)
    origin_lat, origin_lon = 30.9010, 75.8573  # Ludhiana city center, approx

    meters_per_deg_lat = 111_000
    meters_per_deg_lon = 111_000 * np.cos(np.radians(origin_lat))

    node_id = 0
    grid_ids = {}
    for i in range(grid_size):
        for j in range(grid_size):
            x_m = i * spacing_m + np.random.uniform(-20, 20)
            y_m = j * spacing_m + np.random.uniform(-20, 20)
            lon = origin_lon + x_m / meters_per_deg_lon
            lat = origin_lat + y_m / meters_per_deg_lat
            G.add_node(node_id, x=lon, y=lat)
            grid_ids[(i, j)] = node_id
            node_id += 1

    speed_mps = config.DEFAULT_SPEED_KMPH * 1000 / 60.0  # km/h -> m/min

    def add_edge(a, b):
        ax, ay = G.nodes[a]["x"], G.nodes[a]["y"]
        bx, by = G.nodes[b]["x"], G.nodes[b]["y"]
        dx = (bx - ax) * meters_per_deg_lon
        dy = (by - ay) * meters_per_deg_lat
        length_m = float(np.hypot(dx, dy))
        travel_time_min = length_m / speed_mps
        G.add_edge(a, b, key=0, length=length_m, travel_time=travel_time_min)
        G.add_edge(b, a, key=0, length=length_m, travel_time=travel_time_min)

    # Connect grid neighbors (creates a realistic block/street pattern)
    for i in range(grid_size):
        for j in range(grid_size):
            here = grid_ids[(i, j)]
            if i + 1 < grid_size:
                add_edge(here, grid_ids[(i + 1, j)])
            if j + 1 < grid_size:
                add_edge(here, grid_ids[(i, j + 1)])

    print(f"[network_utils] Built synthetic grid network "
          f"({len(G.nodes)} nodes, {len(G.edges)} edges).")
    return G


def nearest_node(G, lon, lat):
    """Find nearest graph node to a (lon, lat) point without requiring OSMnx."""
    best_node, best_dist = None, float("inf")
    for n, data in G.nodes(data=True):
        d = (data["x"] - lon) ** 2 + (data["y"] - lat) ** 2
        if d < best_dist:
            best_dist, best_node = d, n
    return best_node


def shortest_path_time(G, source, target, weight="travel_time"):
    """Shortest travel-time (minutes) between two nodes; inf if unreachable."""
    try:
        return nx.shortest_path_length(G, source, target, weight=weight)
    except nx.NetworkXNoPath:
        return float("inf")


def build_travel_time_matrix(G, node_ids, weight="travel_time"):
    """
    Build an all-pairs travel-time matrix (minutes) for a list of graph nodes.
    Uses multi-source Dijkstra per origin node, which scales far better than
    naive O(n^2) shortest_path calls on larger graphs.
    """
    n = len(node_ids)
    matrix = np.zeros((n, n))
    for i, src in enumerate(node_ids):
        lengths = nx.single_source_dijkstra_path_length(G, src, weight=weight)
        for j, dst in enumerate(node_ids):
            matrix[i, j] = lengths.get(dst, 1e6)  # large penalty if unreachable
    return matrix
