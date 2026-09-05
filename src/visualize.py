"""
Interactive map visualization using Folium.
Produces:
  - zones_map.html      : collection points colored by K-Means zone
  - optimized_routes.html : final CVRPTW routes drawn on the road network
  - baseline_routes.html  : naive per-zone routes, for visual before/after
"""

import folium
import networkx as nx
from . import config


PALETTE = ["#e63946", "#2a9d8f", "#457b9d", "#f4a261", "#8338ec",
           "#ffbe0b", "#3a86ff", "#fb5607", "#06d6a0", "#8d5524"]


def plot_zones(df_stops, out_path=f"{config.OUTPUT_DIR}/zones_map.html"):
    depot = df_stops[df_stops["type"] == "depot"].iloc[0]
    m = folium.Map(location=[depot["lat"], depot["lon"]], zoom_start=14, tiles="OpenStreetMap")

    folium.Marker(
        [depot["lat"], depot["lon"]],
        icon=folium.Icon(color="black", icon="home"),
        tooltip="Depot",
    ).add_to(m)

    for _, row in df_stops[df_stops["type"] != "depot"].iterrows():
        color = PALETTE[int(row["zone"]) % len(PALETTE)] if row["zone"] >= 0 else "gray"
        marker = ("square" if row["type"] == "commercial" else "circle")
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6 if row["type"] == "commercial" else 4,
            color=color,
            fill=True,
            fill_opacity=0.85,
            tooltip=f"Stop {row['stop_id']} ({row['type']}, zone {row['zone']})",
        ).add_to(m)

    m.save(out_path)
    print(f"[visualize] Saved zone map -> {out_path}")
    return out_path


def _node_path_coords(G, from_node, to_node, weight="travel_time"):
    """Get the actual road-following polyline coordinates between two nodes."""
    try:
        path = nx.shortest_path(G, from_node, to_node, weight=weight)
    except nx.NetworkXNoPath:
        return []
    return [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]


def plot_routes(G, df_stops, routes_dict, title, out_path, depot_index=0):
    """
    routes_dict: {vehicle_id: [stop_index, ...]} where stop_index refers to
    df_stops.index (0 = depot).
    """
    depot = df_stops.loc[depot_index]
    m = folium.Map(location=[depot["lat"], depot["lon"]], zoom_start=14, tiles="OpenStreetMap")

    folium.Marker(
        [depot["lat"], depot["lon"]],
        icon=folium.Icon(color="black", icon="home"),
        tooltip="Depot",
    ).add_to(m)

    for vehicle_id, route in routes_dict.items():
        color = PALETTE[int(vehicle_id) % len(PALETTE)]
        full_route = [depot_index] + route + [depot_index]

        for i in range(len(full_route) - 1):
            a_node = df_stops.loc[full_route[i], "node"]
            b_node = df_stops.loc[full_route[i + 1], "node"]
            coords = _node_path_coords(G, a_node, b_node)
            if coords:
                folium.PolyLine(coords, color=color, weight=3, opacity=0.8).add_to(m)

        for stop_idx in route:
            row = df_stops.loc[stop_idx]
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.9,
                tooltip=f"Truck {vehicle_id} | Stop {row['stop_id']} ({row['type']})",
            ).add_to(m)

    m.get_root().html.add_child(folium.Element(f"<h4 style='margin:8px'>{title}</h4>"))
    m.save(out_path)
    print(f"[visualize] Saved route map -> {out_path}")
    return out_path
