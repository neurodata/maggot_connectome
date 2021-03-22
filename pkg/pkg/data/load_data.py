import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from ..graph import MaggotGraph
from ..utils import to_pandas_edgelist

DATA_VERSION = "2021-03-10"  # set to whatever the most recent one is

DATA_PATH = Path(__file__).parent.parent.parent.parent  # don't judge me judge judy
DATA_PATH = DATA_PATH / "data"


def _get_folder(path, version):
    if path is None:
        path = DATA_PATH
    if version is None:
        version = DATA_VERSION
    folder = path / version
    return folder


def load_node_meta(path=None, version=None):
    folder = _get_folder(path, version)
    meta = pd.read_csv(folder / "meta_data.csv", index_col=0)
    meta.sort_index(inplace=True)
    return meta


def load_edgelist(graph_type="G", path=None, version=None):
    folder = _get_folder(path, version)
    edgelist = pd.read_csv(
        folder / f"{graph_type}_edgelist.txt",
        delimiter=" ",
        header=None,
        names=["source", "target", "weight"],
    )
    return edgelist


def load_networkx(graph_type="G", node_meta=None, path=None, version=None):
    edgelist = load_edgelist(graph_type, path=path, version=version)
    g = nx.from_pandas_edgelist(edgelist, edge_attr="weight", create_using=nx.DiGraph())
    if node_meta is not None:
        meta_data_dict = node_meta.to_dict(orient="index")
        nx.set_node_attributes(g, meta_data_dict)
    return g


def load_adjacency(
    graph_type="G", nodelist=None, output="numpy", path=None, version=None
):
    g = load_networkx(graph_type=graph_type, path=path, version=version)
    if output == "numpy":
        adj = nx.to_numpy_array(g, nodelist=nodelist)
    elif output == "pandas":
        adj = nx.to_pandas_adjacency(g, nodelist=nodelist)
    return adj


def load_palette(path=None, version=None):
    folder = _get_folder(path, version)
    with open(folder / "simple_color_map.json", "r") as f:
        palette = json.load(f)
    return palette


def load_maggot_graph(path=None, version=None):
    nodes = load_node_meta()
    g = nx.MultiDiGraph()
    g.add_nodes_from(nodes.index)
    nx.set_node_attributes(g, nodes.to_dict(orient="index"))
    graph_types = ["Gaa", "Gad", "Gda", "Gdd"]
    for graph_type in graph_types:
        g_type = load_networkx(graph_type=graph_type)
        for u, v, data in g_type.edges(data=True):
            g.add_edge(u, v, key=graph_type[1:], edge_type=graph_type[1:], **data)

    g_type = load_networkx(graph_type="G")
    for u, v, data in g_type.edges(data=True):
        g.add_edge(u, v, key="sum", edge_type="sum", **data)

    edges = to_pandas_edgelist(g)

    return MaggotGraph(g, nodes, edges)


# def load_networkx(graph_type, base_path=None, version=DATA_VERSION):
#     if base_path is None:
#         base_path = DATA_PATH
#     data_path = Path(base_path)
#     data_path = data_path / version
#     file_path = data_path / (graph_type + ".graphml")
#     graph = nx.read_graphml(file_path, node_type=str, edge_key_type="str")
#     return graph


def load_data(graph_type, base_path=None, version=None):
    # TODO deprecate this and pull out of old scripts
    if base_path is None:
        base_path = DATA_PATH
    if version is None:
        version = DATA_VERSION

    data_path = Path(base_path)
    data_path = data_path / version

    edgelist_path = data_path / (graph_type + ".edgelist")
    meta_path = data_path / "meta_data.csv"

    graph = nx.read_edgelist(
        edgelist_path, create_using=nx.DiGraph, nodetype=int, data=[("weight", int)]
    )
    meta = pd.read_csv(meta_path, index_col=0)
    adj = nx.to_numpy_array(graph, nodelist=meta.index.values, dtype=float)
    missing_nodes = np.setdiff1d(meta.index, list(graph.nodes()))
    for node in missing_nodes:
        graph.add_node(node)

    return Bunch(graph=graph, adj=adj, meta=meta)
