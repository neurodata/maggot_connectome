import numpy as np

# CLASS_KEY = "merge_class"
# node_palette
mg = load_maggot_graph()
mg = mg[mg.nodes["paper_clustered_neurons"] | mg.nodes["accessory_neurons"]]
mg = mg[mg.nodes["hemisphere"].isin(["L", "R"])]
mg.to_largest_connected_component(verbose=True)
out_degrees = np.count_nonzero(mg.sum.adj, axis=0)
in_degrees = np.count_nonzero(mg.sum.adj, axis=1)
max_in_out_degree = np.maximum(out_degrees, in_degrees)
# TODO we could OOS these back in?
keep_inds = np.arange(len(mg.nodes))[max_in_out_degree > 2]
remove_ids = np.setdiff1d(mg.nodes.index, mg.nodes.index[keep_inds])
print(f"Removed {len(remove_ids)} nodes when removing pendants.")
mg.nodes = mg.nodes.iloc[keep_inds]
mg.g.remove_nodes_from(remove_ids)
mg.to_largest_connected_component(verbose=True)
mg.nodes.sort_values("hemisphere", inplace=True)
mg.nodes["_inds"] = range(len(mg.nodes))
nodes = mg.nodes

raw_adj = mg.sum.adj.copy()

left_nodes = mg.nodes[mg.nodes["hemisphere"] == "L"]
left_inds = left_nodes["_inds"]
right_nodes = mg.nodes[mg.nodes["hemisphere"] == "R"]
right_inds = right_nodes["_inds"]

left_paired_inds, right_paired_inds = get_paired_inds(
    mg.nodes, pair_key="predicted_pair", pair_id_key="predicted_pair_id"
)
right_paired_inds_shifted = right_paired_inds - len(left_inds)
