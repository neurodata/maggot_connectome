#%%
import graph_tool as gt
from graph_tool.collection import data
from graph_tool.inference import minimize_nested_blockmodel_dl, minimize_blockmodel_dl
from graph_tool.draw import draw_hierarchy
import numpy as np
import pandas as pd
import time

# g = data["celegansneural"]
g = gt.load_graph_from_csv(
    "../data/2021-04-02/G_edgelist.txt",
    directed=True,
    csv_options=dict(delimiter=" "),
    eprop_types=["float"],
    eprop_names=["weight"],
)
# print(dir(g.vertex_properties))
# state = minimize_blockmodel_dl(g, deg_corr=True, verbose=True)
# draw_hierarchy(state, output="maggot_nested_mdl.pdf")

# did not like results of discrete-binomial as much as unweighted
# discrete poisson was similar
# discrete geometric maybe the best of the weighted ones


def run_minimize_blockmodel(g):
    total_degrees = g.get_total_degrees(g.get_vertices())
    remove_verts = np.where(total_degrees == 0)[0]
    g.remove_vertex(remove_verts)
    y = g.ep.weight.copy()
    y.a = np.log(y.a)
    min_state = minimize_blockmodel_dl(
        g,
        deg_corr=True,
        verbose=True,
        state_args=dict(recs=[y], rec_types=["real-normal"]),
    )

    blocks = list(min_state.get_blocks())
    verts = g.get_vertices()

    block_map = {}

    for v, b in zip(verts, blocks):
        cell_id = int(g.vertex_properties["name"][v])
        block_map[cell_id] = int(b)

    block_series = pd.Series(block_map)
    block_series.name = "block_label"
    return block_series


currtime = time.time()
labels = run_minimize_blockmodel(g)
print(f"{time.time() - currtime:.3f} seconds elapsed.")
labels.to_csv("labels.csv")
