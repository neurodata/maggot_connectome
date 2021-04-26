from pkg.data import load_maggot_graph
from graspologic.utils import largest_connected_component
import networkx as nx


def flatten_muligraph(multigraph, meta_data_dict, edge_type="sum"):
    # REF: https://stackoverflow.com/questions/15590812/networkx-convert-multigraph-into-simple-graph-with-weighted-edges
    g = nx.DiGraph()
    for node in multigraph.nodes():
        g.add_node(node)
    for i, j, data in multigraph.edges(data=True):
        if data["edge_type"] == edge_type:
            w = data["weight"] if "weight" in data else 1.0
            if g.has_edge(i, j):
                g[i][j]["weight"] += w
            else:
                g.add_edge(i, j, weight=w)
    nx.set_node_attributes(g, meta_data_dict)
    return g


mg = load_maggot_graph()
g = mg.g
g = largest_connected_component(g)
g = flatten_muligraph(g, mg.nodes.to_dict(orient="index"))
nx.write_graphml(g, "maggot_connectome/sandbox/full_maggot.graphml")
