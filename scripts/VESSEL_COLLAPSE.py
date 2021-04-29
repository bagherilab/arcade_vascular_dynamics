from .utilities import load_json, save_csv
import tarfile
import re
import networkx as nx
import numpy as np
import pandas as pd
from math import floor#, sqrt, log10

RADIUS = 34
MARGIN = 6
RADIUS_BOUNDS = RADIUS + MARGIN
LENGTH = 6*RADIUS_BOUNDS - 3
WIDTH = 4*RADIUS_BOUNDS - 2

class VESSEL_COLLAPSE():
    NAME = "VESSEL_COLLAPSE"

    STRUCTURES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    DEGRADATION = [f"{i:03}" for i in range(0,110,10)]

    STABILIZED = [f"{i:03}" for i in range(0,110,10)]

    SEEDS = [f"{i:02}" for i in range(0,10)]

    @staticmethod
    def analyze_degradation(output_path, input_path, name=NAME,
                            degradation=DEGRADATION, structures=STRUCTURES):
        header = ",".join(["TIMEPOINT", "FLOW_TUMOR", "FLOW_TISSUE", "N_TUMOR", "N_TISSUE"]) + "\n"

        for degrade in degradation:
            for structure in structures:
                code = f"_degradation_{degrade}_{structure}"
                infile_cell = f"{input_path}{name}/{name}_degradation_{degrade}_{structure}.tar.xz"
                infile_graph = f"{input_path}{name}.GRAPH/{name}_degradation_{degrade}_{structure}.GRAPH.tar.xz"
                print(f"{name} : {code}")

                tar_cell = tarfile.open(infile_cell)
                tar_graph = tarfile.open(infile_graph)

                for cell, graph in zip(tar_cell.getmembers(), tar_graph.getmembers()):
                    out = []
                    seed = int(re.findall(r'_([0-9]{2})\.json', cell.name)[0])
                    cell_json = load_json(cell, tar=tar_cell)
                    graph_json = load_json(graph, tar=tar_graph)

                    for timepoint in range(2,45,1):
                        graph = graph_json["timepoints"][timepoint]["graph"]
                        cells = cell_json["timepoints"][timepoint]["cells"]
                        G, G_tumor, G_tissue, N_tumor, N_tissue = get_graphs(graph, cells)
                        flow_tumor, flow_tissue = get_metrics(G_tumor, G_tissue)
                        out.append([timepoint, flow_tumor, flow_tissue, N_tumor, N_tissue])

                    save_csv(f"{output_path}{name}/{name}{code}_{seed:02}", header , zip(*out), "")

    @staticmethod
    def merge_degradation(output_path, name=NAME,
                          degradation=DEGRADATION, structures=STRUCTURES, seeds=SEEDS):
        df = pd.DataFrame()

        for structure in structures:
            for degrade in degradation:
                for seed in seeds:
                    csv_file = f"{output_path}{name}/{name}_degradation_{degrade}_{structure}_{seed}.csv"
                    df0 = pd.read_csv(csv_file)
                    df0["DEGRADATION"] = degrade
                    df0["STRUCTURE"] = structure
                    df0["SEED"] = seed
                    df = df.append(df0)
                    break
                break
            break

        df.to_csv(f"{output_path}{name}/{name}_degradation.csv", index=False)

    @staticmethod
    def analyze_stabilized(output_path, input_path, name=NAME,
                           stabilized=STABILIZED, structures=STRUCTURES):
        header = ",".join(["TIMEPOINT", "FLOW_TUMOR", "FLOW_TISSUE", "N_TUMOR", "N_TISSUE", "N_EDGES", "N_STABILIZED"]) + "\n"

        for stabilize in stabilized:
            for structure in structures:
                code = f"_stabilized_{stabilize}_{structure}"
                infile_cell = f"{input_path}{name}/{name}_stabilized_{stabilize}_{structure}.tar.xz"
                infile_graph = f"{input_path}{name}.GRAPH/{name}_stabilized_{stabilize}_{structure}.GRAPH.tar.xz"
                print(f"{name} : {code}")

                tar_cell = tarfile.open(infile_cell)
                tar_graph = tarfile.open(infile_graph)

                for cell, graph in zip(tar_cell.getmembers(), tar_graph.getmembers()):
                    out = []
                    seed = int(re.findall(r'_([0-9]{2})\.json', cell.name)[0])
                    cell_json = load_json(cell, tar=tar_cell)
                    graph_json = load_json(graph, tar=tar_graph)

                    for timepoint in range(2,45,1):
                        graph = graph_json["timepoints"][timepoint]["graph"]
                        cells = cell_json["timepoints"][timepoint]["cells"]
                        G, G_tumor, G_tissue, N_tumor, N_tissue = get_graphs(graph, cells, has_stabilized=True)
                        flow_tumor, flow_tissue = get_metrics(G_tumor, G_tissue)
                        total_edges, stabilized_edges = get_stabilized(G)
                        out.append([timepoint, flow_tumor, flow_tissue, N_tumor, N_tissue, total_edges, stabilized_edges])

                    save_csv(f"{output_path}{name}/{name}{code}_{seed:02}", header , zip(*out), "")

    @staticmethod
    def merge_stabilized(output_path, name=NAME,
                          stabilized=STABILIZED, structures=STRUCTURES, seeds=SEEDS):
        df = pd.DataFrame()

        for structure in structures:
            for stabilize in stabilized:
                for seed in seeds:
                    csv_file = f"{output_path}{name}/{name}_stabilized_{stabilize}_{structure}_{seed}.csv"
                    df0 = pd.read_csv(csv_file)
                    df0["STABILIZED"] = stabilize
                    df0["STRUCTURE"] = structure
                    df0["SEED"] = seed
                    df = df.append(df0)
                    break
                break
            break

        df.to_csv(f"{output_path}{name}/{name}_stabilized.csv", index=False)

def to_location(x, y):
    """Convert (x, y) coordinate to (u, v, w, z) coordinate."""

    # Calculate u coordinate.
    uu = (x - 2)/3.0 - RADIUS_BOUNDS
    u = int(round(uu))

    # Calculate v and w coordinates based on u.
    vw = y - 2*RADIUS_BOUNDS + 2
    v = -int(floor((vw + u)/2.0))
    w = -(u + v)

    # Check if out of bounds.
    if abs(v) >= RADIUS or abs(w) >= RADIUS:
        return None

    return (u, v, w, 0)

def check_site(s, x, y):
    """Check if site is value."""
    if x >= 0 and x < LENGTH and y >= 0 and y < WIDTH:
        s.append((x, y))

def check_vertical(s, dY, sY, x0, y0):
    """Check vertical span."""
    for d in range(0, dY, 1):
        check_site(s, x0 - 1, y0 + (-(d + 1) if sY else d))

def check_horizontal(s, dX, sX, x0, y0):
    """Check horizontal  span."""
    for d in range(0, dX, 2):
        check_site(s, x0 + (-(d + 2) if sX else d), y0)
        check_site(s, x0 + (-(d + 2) if sX else d), y0 - 1)

def check_upper_diagonals(s, dX, dY, sX, sY, x0, y0):
    """Check upper diagonal span."""
    for d in range(0, dX - 1, 3):
        check_site(s, x0 + (-(d + 2) if sX else d), y0 + (-(d/3 + 1) if sY else d/3))
        check_site(s, x0 + (-(d + 3) if sX else d + 1), y0 + (-(d/3 + 1) if sY else d/3))

def check_lower_diagonals(s, dX, dY, sX, sY, x0, y0):
    """Check lower diagonal span."""
    for d in range(0, dX, 1):
        check_site(s, x0 + (-(d + 2) if sX else d), y0 + (-(d + 1) if sY else d))
        check_site(s, x0 + (-(d + 1) if sX else d - 1), y0 + (-(d + 1) if sY else d))

def get_span(from_node, to_node):
    """Get list of spanning locations."""
    s = []

    from_x, from_y = from_node
    to_x, to_y = to_node

    x0 = from_x
    y0 = from_y
    x1 = to_x
    y1 = to_y

    # Calculate deltas.
    dX = x1 - x0
    dY = y1 - y0

    # Check direction of arrow and update deltas to absolute.
    sX = dX < 0
    sY = dY < 0

    dX = abs(dX)
    dY = abs(dY)

    # Check if line is vertical.
    if x0 == x1:
        check_vertical(s, dY, sY, x0, y0)
    # Check if line is horizontal.
    elif y0 == y1:
        check_horizontal(s, dX, sX, x0, y0)
    # Check for upper diagonals (30 degrees).
    elif (float(dX)/float(dY) == 3):
        check_upper_diagonals(s, dX, dY, sX, sY, x0, y0)
    # Check for lower diagonals (60 degrees).
    elif (dX == dY):
        check_lower_diagonals(s, dX, dY, sX, sY, x0, y0)
    # All other cases.
    else:
        # Calculate starting and ending triangles.
        startx = x0 - ((2 if sX else 0) if dY < dX else 1)
        starty = y0 - (1 if sY else 0)
        endx = x1 - ((0 if sX else 2) if dY < dX else 1)
        endy = y1 - (0 if sY else 1)

        # Calculate new deltas based on triangle.
        dx = abs(endx - startx)
        dy = abs(endy - starty)

        # Initial conditions.
        x = startx
        y = starty
        e = 0

        # Add start triangle.
        check_site(s, x, y)

        # Track if triangle is even (point down) or odd (point up).
        even = False

        # Iterate until the ending triangle is reached.
        while (x != endx or y != endy):
            even = ((x + y) & 1) == 0

            if e > 3*dx:
                if not sX and not sY:
                    if even:
                        check_site(s, x - 1, y)
                    else:
                        check_site(s, x, y + 1)
                    x = x - 1
                    y = y + 1
                elif not sX and sY:
                    if even:
                        check_site(s, x, y - 1)
                    else:
                        check_site(s, x - 1, y)
                    x = x - 1
                    y = y - 1
                elif sX and not sY:
                    if even:
                        check_site(s, x + 1, y)
                    else:
                        check_site(s, x, y + 1)
                    x = x + 1
                    y = y + 1
                elif sX and sY:
                    if even:
                        check_site(s, x, y - 1)
                    else:
                        check_site(s, x + 1, y)
                    x = x + 1
                    y = y - 1

                e -= (2*dy + 2*dx)
            elif e >= 2*dx:
                if not sY:
                    y = y + 1
                else:
                    y = y - 1
                e -= 2*dx
            else:
                e += 2*dy

                if e >= dx:
                    if not sX and not sY:
                        if even:
                            check_site(s, x + 1, y)
                        else:
                            check_site(s, x, y + 1)
                        x = x + 1
                        y = y + 1
                    elif not sX and sY:
                        if even:
                            check_site(s, x, y - 1)
                        else:
                            check_site(s, x + 1, y)
                        x = x + 1
                        y = y - 1
                    elif sX and not sY:
                        if even:
                            check_site(s, x - 1, y)
                        else:
                            check_site(s, x, y + 1)
                        x = x - 1
                        y = y + 1
                    elif sX and sY:
                        if even:
                            check_site(s, x, y - 1)
                        else:
                            check_site(s, x - 1, y)
                        x = x - 1
                        y = y - 1

                    e -= 2*dx
                else:
                    if not sX:
                        x = x + 1
                    else:
                        x = x - 1

            check_site(s, x, y)

    return s

def convert_to_graph(graph, has_stabilized=False):
    """Converts edge list to graph object."""
    G = nx.DiGraph()

    all_nodes = []
    for from_node, to_node, edge in graph:
        all_nodes.append((from_node[0], from_node[1]))
        all_nodes.append((to_node[0], to_node[1]))

    for x, y in set(all_nodes):
        G.add_node((x, y), pos=(x, y))

    for from_node, to_node, edge in graph:
        from_node_name = (from_node[0], from_node[1])
        to_node_name = (to_node[0], to_node[1])

        if has_stabilized:
            G.add_edge(from_node_name, to_node_name, flow=edge[-2], stabilized=edge[-1])
        else:
            G.add_edge(from_node_name, to_node_name, flow=edge[-1])

    return G

def select_subgraph(full_graph, all_cells):
    """Selected subset of full graph adjacent to cells."""
    sublocations = set()
    subgraph = []

    # For each edges, get the corresponding span location.
    for from_node, to_node, edge in full_graph:
        # Get span locations
        spans = get_span(from_node[0:2], to_node[0:2])
        locations = set([to_location(*span) for span in spans])

        # Get agents at locations
        cells = []
        for location in locations:
            cell_location = next((i for i in all_cells if tuple(i[0]) == location), None)
            if cell_location:
                cells = cells + cell_location[1]

        # Check if any cells are cancerous
        if any([c[0] != 0 for c in cells]):
            subgraph.append((from_node, to_node, edge))
            sublocations = sublocations.union(locations)

    return subgraph, sublocations

def get_graphs(graph_timepoint, cell_timepoint, has_stabilized=False):
    """Parse timepoints for graph objects."""
    G = convert_to_graph(graph_timepoint, has_stabilized=has_stabilized)
    subgraph, sublocations = select_subgraph(graph_timepoint, cell_timepoint)
    G_tumor = convert_to_graph(subgraph, has_stabilized=has_stabilized)

    G_tissue = G.copy()
    G_tissue.remove_edges_from(n for n in G.edges() if n in G_tumor.edges())

    N_tumor = len(sublocations)
    N_tissue = LENGTH*WIDTH - N_tumor

    return G, G_tumor, G_tissue, N_tumor, N_tissue

def get_metrics(G_tumor, G_tissue):
    """Parse graphs for flow rates."""
    flow_tumor = np.nansum([G_tumor[from_node][to_node]["flow"] for from_node, to_node in G_tumor.edges()])
    flow_tissue = np.nansum([G_tissue[from_node][to_node]["flow"] for from_node, to_node in G_tissue.edges()])
    return flow_tumor, flow_tissue

def get_stabilized(G):
    """Parse graphs for stabilization."""
    total_edges = len(G.edges())
    stabilized_edges = np.nansum([G[from_node][to_node]["stabilized"] for from_node, to_node in G.edges()])
    return total_edges, stabilized_edges
