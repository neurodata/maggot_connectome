#%%
import os
from collections import namedtuple
from copy import deepcopy

import numpy as np
import pandas as pd

from pkg.data import load_adjacency, load_node_meta


#%%


def get_paired_inds(meta, check_in=True, pair_key="pair", pair_id_key="pair_id"):
    # TODO make the assertions into actual errors
    # also this whole function is kinda garbage... open to improvement
    pair_meta = meta.copy()
    pair_meta["_inds"] = range(len(pair_meta))
    pair_meta = pair_meta[pair_meta["hemisphere"].isin(["L", "R"])]
    if check_in:
        pair_meta = pair_meta[pair_meta[pair_key].isin(pair_meta.index)]
    pair_group_size = pair_meta.groupby(pair_id_key).size()
    remove_pairs = pair_group_size[pair_group_size == 1].index
    pair_meta = pair_meta[~pair_meta[pair_id_key].isin(remove_pairs)]
    assert pair_meta.groupby(pair_id_key).size().min() == 2
    assert pair_meta.groupby(pair_id_key).size().max() == 2
    pair_meta.sort_values([pair_id_key, "hemisphere"], inplace=True)
    lp_inds = pair_meta[pair_meta["hemisphere"] == "L"]["_inds"]
    rp_inds = pair_meta[pair_meta["hemisphere"] == "R"]["_inds"]
    assert (
        meta.iloc[lp_inds][pair_id_key].values == meta.iloc[rp_inds][pair_id_key].values
    ).all()
    return lp_inds, rp_inds


class _LocIndexer:
    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, index):
        new_adj_dfs = {}
        for name, adj_df in self.parent._adj_dfs.items():
            new_adj_dfs[name] = adj_df.loc[index, index]
        new_meta = self.parent.meta.loc[index]
        return MaggotGraph(new_adj_dfs, new_meta)


class _ILocIndexer:
    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, index):
        new_adj_dfs = {}
        for name, adj_df in self.parent._adj_dfs.items():
            new_adj_dfs[name] = adj_df.iloc[index, index]
        new_meta = self.parent.meta.iloc[index]
        return MaggotGraph(new_adj_dfs, new_meta)


class MaggotGraph:
    def __init__(self, graphs, meta=None):
        if not isinstance(graphs, (list, dict)):
            graphs = {0: graphs}
        if isinstance(graphs, list):
            graphs = dict(zip(range(len(graphs)), graphs))
        self.graphs = graphs

        # TODO assert all same shape and match size of meta
        # TODO make dictionary? or named tuple?
        self._adj_dfs = {}
        for name, graph in graphs.items():
            if not isinstance(graph, pd.DataFrame):
                adj_df = pd.DataFrame(data=graph, index=meta.index, columns=meta.index)
            else:
                adj_df = graph
            self._adj_dfs[name] = adj_df

        self.meta = meta

        # These are indexers where I am trying to be like Pandas - I'm sure could be
        # implemented in a better way.
        self.loc = _LocIndexer(self)
        self.iloc = _ILocIndexer(self)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, val):
        return self.meta[val]

    def __repr__(self):
        return self.meta.__repr__()

    def __deepcopy__(self):
        # TODO im sure there's a more pythonic way of doing this correctly
        return MaggotGraph(self.adjs, self.meta)

    # TODO maybe also have a more official name for this property
    @property
    def adjs(self):
        adjs = {}
        for name, adj in self._adj_dfs.items():
            adjs[name] = adj.values
        return adjs

    def bisect(
        self, by_pairs=True, check_in=True, pair_key="pair", pair_id_key="pair_id"
    ):
        """Split the graph by left right. Return some new MaggotGraphs."""

        if not by_pairs:
            # TODO there is a version of this where we just return the same subgraphs
            # but don't guarantee anything is aligned or even necessarily has pairs
            raise NotImplementedError()

        meta = self.meta
        lp_inds, rp_inds = get_paired_inds(
            meta, check_in=check_in, pair_key=pair_key, pair_id_key=pair_id_key
        )

        # left-left, right-right, left-right, right-left
        split_adjs = {"ll": {}, "rr": {}, "lr": {}, "rl": {}}
        for name, adj in self.adjs.items():
            split_adjs["ll"][name] = adj[lp_inds][:, lp_inds]
            split_adjs["rr"][name] = adj[rp_inds][:, rp_inds]
            split_adjs["lr"][name] = adj[lp_inds][:, rp_inds]
            split_adjs["rl"][name] = adj[rp_inds][:, lp_inds]

        split_meta = {"l": meta.iloc[lp_inds].copy(), "r": meta.iloc[rp_inds].copy()}

        # TODO can also extract the left -> right and right -> left subgraph adjacencies
        # so I didn't go with this approach
        # left_induced_mg = deepcopy(self.iloc[lp_inds])
        # right_induced_mg = deepcopy(self.iloc[rp_inds])

        # TODO another (possibly more sensible option) here is to return 4 MaggotGraphs.
        # the problem is that for the L-R and R-L subgraphs, the rows and columns index
        # different objects. But we could somehow aggregate the node metadata by pair
        # to solve this problem (e.g. they are likely to have the same cell type,
        # for instance).
        return split_adjs, split_meta

    def largest_connected_component(self, union=True):
        # TODO
        raise NotImplementedError()


meta = load_node_meta()
ids = meta.index

graph_types = ["aa", "ad", "da", "dd"]
graphs = {}
union = np.zeros((len(meta), len(meta)))
for graph_type in graph_types:
    color_adj = load_adjacency(graph_type=f"G{graph_type}", nodelist=ids)
    graphs[graph_type] = color_adj
    union += color_adj
graphs["union"] = union

#%%
mg = MaggotGraph(graphs, meta=meta)
mg.adjs
#%%
bisected_adjs, bisected_meta = mg.bisect()

#%%
from src.visualization import adjplot

adjplot()

#%%

# getting lcc
# bisecting


#%%

#%%

# # Test data
# meta = [
#     ["A", "cat", 2, 1.0],
#     ["B", "dog", 5, 8.0],
#     ["X", "llama", 54, 3.0],
#     ["F", "alpaca", 1, 10.0],
# ]

# adj1 = np.array([[0, 1, 2, 0], [1, 0, 1, 0], [10, 3, 0, 0], [5, 0, 1, 1]])
# adj2 = np.array([[3, 1, 0, 1], [1, 0, 2, 0], [9, 3, 0, 0], [6, 0, 5, 1]])

# meta = pd.DataFrame(
#     data=meta, index=[33, 44, 55, 66], columns=["letter", "animal", "count", "weight"]
# )

# mmg = MaggotGraph([adj1, adj2], meta)
# print(mmg.meta)
# print(mmg.adjs[0])
# print(mmg.adjs[1])
# print()
# sub_mmg = mmg.loc[[66, 44, 33]]
# print(sub_mmg.meta)
# print(sub_mmg.adjs[0])
# print(sub_mmg.adjs[1])