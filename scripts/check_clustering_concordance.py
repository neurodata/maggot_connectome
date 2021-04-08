#%% [markdown]
# # Evaluate clustering concordance

#%%

import datetime
import pprint
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from anytree import LevelOrderGroupIter, Node, NodeMixin, Walker
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import adjusted_rand_score, rand_score

from giskard.plot import stacked_barplot
from graspologic.align import OrthogonalProcrustes, SeedlessProcrustes
from graspologic.cluster import DivisiveCluster
from graspologic.embed import (
    AdjacencySpectralEmbed,
    OmnibusEmbed,
    select_dimension,
    selectSVD,
)
from graspologic.plot import pairplot
from graspologic.utils import (
    augment_diagonal,
    binarize,
    is_fully_connected,
    multigraph_lcc_intersection,
    pass_to_ranks,
    to_laplacian,
)
from pkg.data import load_adjacency, load_maggot_graph, load_node_meta
from pkg.io import savefig
from pkg.plot import set_theme
from pkg.utils import get_paired_inds, get_paired_subgraphs, set_warnings
from src.visualization import CLASS_COLOR_DICT as palette
from src.visualization import adjplot  # TODO fix graspologic version and replace here
from pkg.flow import signal_flow
from pkg.utils import get_paired_inds

set_warnings()


t0 = time.time()


def stashfig(name, **kwargs):
    foldername = "cluster_concordance"
    savefig(name, foldername=foldername, **kwargs)


set_theme()

#%% [markdown]
# ### Load the data
#%%
mg = load_maggot_graph()
mg = mg[mg.nodes["paper_clustered_neurons"] | mg.nodes["accessory_neurons"]]
mg.to_largest_connected_component()
mg.fix_pairs()
mg.nodes["sf"] = signal_flow(mg.sum.adj)
mg.nodes["_inds"] = range(len(mg.nodes))
lp_inds, rp_inds = get_paired_inds(mg.nodes)
(mg.nodes.iloc[lp_inds]["pair"] == mg.nodes.iloc[rp_inds].index).all()
#%% [markdown]
# ## Evaluate cluster concordance between same or different hemispheres
#%% [markdown]
# ## Run multiple rounds of embedding/clustering each hemisphere independently
#%%


def preprocess_adjs(adjs, method="ase"):
    """Preprocessing necessary prior to embedding a graph, opetates on a list

    Parameters
    ----------
    adjs : list of adjacency matrices
        [description]
    method : str, optional
        [description], by default "ase"

    Returns
    -------
    [type]
        [description]
    """
    adjs = [pass_to_ranks(a) for a in adjs]
    adjs = [a + 1 / a.size for a in adjs]
    if method == "ase":
        adjs = [augment_diagonal(a) for a in adjs]
    elif method == "lse":  # haven't really used much. a few params to look at here
        adjs = [to_laplacian(a) for a in adjs]
    return adjs


def svd(X, n_components=None):
    return selectSVD(X, n_components=n_components, algorithm="full")[0]


n_omni_components = 8  # this is used for all of the embedings initially
n_svd_components = 16  # this is for the last step
method = "ase"  # one could also do LSE


n_init = 1
cluster_kws = dict(affinity=["euclidean", "manhattan", "cosine"])
rows = []
for side in ["left", "right"]:
    # TODO this is ignoring the contralateral connections!
    print("Preprocessing...")
    # side_mg = mg[mg.nodes[side]]
    # side_mg.to_largest_connected_component()
    # adj = side_mg.sum.adj
    if side == "left":
        inds = lp_inds
    else:
        inds = rp_inds
    adj = mg.sum.adj[np.ix_(inds, inds)]
    embed_adj = preprocess_adjs([adj])[0]
    # svd_embed = svd(embed_adj)
    print("Embedding...")
    latent = AdjacencySpectralEmbed(n_components=8, concat=True).fit_transform(
        embed_adj
    )

    print("Clustering...")
    for init in range(n_init):
        print(f"Init {init}")
        dc = DivisiveCluster(max_level=10, min_split=16, cluster_kws=cluster_kws)
        hier_pred_labels = dc.fit_predict(latent)
        row = {
            "hier_pred_labels": hier_pred_labels,
            "nodes": mg.nodes.iloc[inds].copy(),
            "init": init,
            "side": side,
        }
        rows.append(row)


#%%

comparison_rows = []
max_level = 7
for i, row1 in enumerate(rows):
    labels1 = row1["hier_pred_labels"]
    nodes1 = row1["nodes"]
    for j, row2 in enumerate(rows):
        if i > j:
            labels2 = row2["hier_pred_labels"]
            nodes2 = row2["nodes"]
            print((nodes1["pair"] == nodes2.index).all())
            for level in range(1, max_level):
                _, flat_labels1 = np.unique(labels1[:, :level], return_inverse=True)
                _, flat_labels2 = np.unique(labels2[:, :level], return_inverse=True)
                ari = adjusted_rand_score(flat_labels1, flat_labels2)
                row = {
                    "source": i,
                    "target": j,
                    "source_side": row1["side"],
                    "target_side": row2["side"],
                    "metric_val": ari,
                    "metric": "ARI",
                    "level": level,
                }
                comparison_rows.append(row)

                ri = rand_score(flat_labels1, flat_labels2)
                row = row.copy()
                row["metric_val"] = ri
                row["metric"] = "RI"
                comparison_rows.append(row)


comparison_results = pd.DataFrame(comparison_rows)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.lineplot(
    data=comparison_results,
    y="metric_val",
    hue="metric",
    x="level",
    style="metric",
    markers=True,
)
stashfig("pairwise-metrics-by-level")


#%%


class MetaNode(NodeMixin):
    def __init__(self, name, parent=None, children=None, meta=None):
        super().__init__()
        self.name = name
        self.parent = parent
        if children:
            self.children = children
        self.meta = meta

    def hierarchical_mean(self, key):
        if self.is_leaf:
            meta = self.meta
            var = meta[key]
            return np.mean(var)
        else:
            children = self.children
            child_vars = [child.hierarchical_mean(key) for child in children]
            return np.mean(child_vars)


def make_node(label, node_map):
    if label not in node_map:
        node = MetaNode(label)
        node_map[label] = node
    else:
        node = node_map[label]
    return node


def apply_flat_labels(meta, hier_pred_labels):
    cluster_meta = meta.copy()
    n_levels = hier_pred_labels.shape[1]
    last_max = -1
    cluster_map = {}
    for level in range(n_levels):
        uni_pred_labels, indicator = np.unique(
            hier_pred_labels[:, :level], axis=0, return_inverse=True
        )
        uni_pred_labels = [tuple(u) for u in uni_pred_labels]
        labels = indicator + last_max + 1
        level_map = dict(
            zip(np.arange(len(uni_pred_labels)) + last_max + 1, uni_pred_labels)
        )
        cluster_meta[f"lvl{level}_labels"] = labels
        last_max = np.max(labels)
        cluster_map.update(level_map)
    return cluster_meta, cluster_map


def get_x_y(xs, ys, orient):
    if orient == "h":
        return xs, ys
    elif orient == "v":
        return (ys, xs)


def plot_dendrogram(
    root,
    ax=None,
    index_key="_inds",
    orient="h",
    linewidth=0.7,
    cut=None,
    max_level=None,
):
    if max_level is None:
        max_level = root.height

    for node in (root.descendants) + (root,):
        node.y = node.hierarchical_mean(index_key)
        node.x = node.depth

    walker = Walker()
    walked = []

    for node in root.leaves:
        upwards, common, downwards = walker.walk(node, root)
        curr_node = node
        for up_node in (upwards) + (root,):
            edge = (curr_node, up_node)
            if edge not in walked:
                xs = [curr_node.x, up_node.x]
                ys = [curr_node.y, up_node.y]
                xs, ys = get_x_y(xs, ys, orient)
                ax.plot(
                    xs,
                    ys,
                    linewidth=linewidth,
                    color="black",
                    alpha=1,
                )
                walked.append(edge)
            curr_node = up_node
        y_max = node.meta[index_key].max()
        y_min = node.meta[index_key].min()
        xs = [node.x, node.x, node.x + 1, node.x + 1]
        ys = [node.y - linewidth * 2, node.y + linewidth * 2, y_max + 1, y_min]
        xs, ys = get_x_y(xs, ys, orient)
        ax.fill(xs, ys, facecolor="black")

    if orient == "h":
        ax.set(xlim=(-1, max_level + 1))
        if cut is not None:
            ax.axvline(cut - 1, linewidth=1, color="grey", linestyle=":")
    elif orient == "v":
        ax.set(ylim=(max_level + 1, -1))
        if cut is not None:
            ax.axhline(cut - 1, linewidth=1, color="grey", linestyle=":")
    ax.axis("off")
    return ax


def plot_colorstrip(
    meta, colors_var, ax=None, orient="v", index_key="_inds", palette="tab10"
):
    if ax is None:
        ax = plt.gca()

    color_data = meta[colors_var]
    uni_classes = list(np.unique(color_data))

    indicator = np.full((meta[index_key].max() + 1, 1), np.nan)

    # Create the color dictionary
    if isinstance(palette, dict):
        color_dict = palette
    elif isinstance(palette, str):
        color_dict = dict(
            zip(uni_classes, sns.color_palette(palette, len(uni_classes)))
        )

    # Make the colormap
    class_map = dict(zip(uni_classes, range(len(uni_classes))))

    color_sorted = list(map(color_dict.get, uni_classes))

    lc = ListedColormap(color_sorted)

    for idx, row in meta.iterrows():
        indicator[row[index_key]] = class_map[row[colors_var]]

    if orient == "v":
        indicator = indicator.T

    sns.heatmap(
        indicator,
        cmap=lc,
        cbar=False,
        yticklabels=False,
        xticklabels=False,
        ax=ax,
        square=False,
    )
    return ax


def sort_meta(meta, groups=[], group_orders=[], item_order=[]):
    # create new columns in the dataframe that correspond to the sorting order
    total_sort_by = []
    for group in groups:
        for group_order in group_orders:
            if group_order == "size":
                class_values = meta.groupby(group).size()
            else:
                class_values = meta.groupby(group)[group_order].mean()
            meta[f"_{group}_group_{group_order}"] = meta[group].map(class_values)
            total_sort_by.append(f"_{group}_group_{group_order}")
        total_sort_by.append(group)
    total_sort_by += item_order
    meta = meta.sort_values(total_sort_by, kind="mergesort", ascending=False)
    return meta


max_level = 7
sorters = []
item_orderers = ["merge_class", "pair"]
groupers = [f"lvl{i}_labels" for i in range(max_level)]
sorters += groupers
sorters += item_orderers
gap = 30


def preprocess_meta(meta, hier_labels, max_level=None):
    if max_level is None:
        max_level = hier_labels.shape[1]
    # sorting
    meta, cluster_map = apply_flat_labels(meta, hier_labels)

    # meta = meta.sort_values(sorters, kind="mergesort")
    meta = sort_meta(
        meta, groups=groupers, group_orders=["sf"], item_order=item_orderers
    )

    # add a new dummy variable to keep track of index
    meta["_inds"] = range(len(meta))

    # based on the grouping and the gap specification, map this onto the positional index
    ordered_lowest_clusters = meta[f"lvl{max_level-1}_labels"].unique()
    gap_map = dict(
        zip(ordered_lowest_clusters, np.arange(len(ordered_lowest_clusters)) * gap)
    )
    gap_vec = meta[f"lvl{max_level-1}_labels"].map(gap_map)
    meta["_inds"] += gap_vec
    return meta, cluster_map


def construct_meta_tree(meta, groupers, cluster_map):
    inv_cluster_map = {v: k for k, v in cluster_map.items()}
    node_map = {}
    for grouper in groupers[::-1][:-1]:
        level_labels = meta[grouper].unique()
        for label in level_labels:
            node = make_node(label, node_map)
            node.meta = meta[meta[grouper] == label]
            barcode = cluster_map[label]
            parent_label = inv_cluster_map[barcode[:-1]]
            if parent_label is not None:
                parent = make_node(parent_label, node_map)
                node.parent = parent
    parent.meta = meta
    root = parent
    return root


# #%%
# fig, dend_ax = plt.subplots(1, 1, figsize=(10, 4))
# orient = "v"
# plot_dendrogram(root, ax=dend_ax, index_key="_inds", orient=orient)
# divider = make_axes_locatable(dend_ax)
# color_ax = divider.append_axes("bottom", size="30%", pad=0, sharex=dend_ax)
# plot_colorstrip(meta, "merge_class", ax=color_ax, orient=orient, palette=palette)

# stashfig("test-dendrogram-bars", format="pdf")


class MatrixGrid:
    def __init__(
        self,
        data=None,
        row_meta=None,
        col_meta=None,
        plot_type="heatmap",
        col_group=None,  # single string, list of string, or np.ndarray
        row_group=None,  # can also represent a clustering?
        col_group_order="size",
        row_group_order="size",
        col_dendrogram=None,  # can this just be true false?
        row_dendrogram=None,
        col_item_order=None,  # single string, list of string, or np.ndarray
        row_item_order=None,
        col_colors=None,  # single string, list of string, or np.ndarray
        row_colors=None,
        col_palette="tab10",
        row_palette="tab10",
        col_ticks=True,
        row_ticks=True,
        col_tick_pad=None,
        row_tick_pad=None,
        ax=None,
        figsize=(10, 10),
        gap=False,
    ):
        self.data = data
        self.row_meta = row_meta
        self.col_meta = col_meta

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        else:
            fig = ax.figure
        divider = make_axes_locatable(ax)

        self.fig = fig
        self.ax = ax
        self.divider = divider
        self.top_axs = []
        self.left_axs = []
        self.bottom_axs = []
        self.right_axs = []

    def sort_values(self):
        raise NotImplementedError()

    def append_axes(self, side, size="10%", pad=0):
        kws = {}
        if side in ["top", "bottom"]:
            kws["sharex"] = self.ax
        elif side in ["left", "right"]:
            kws["sharey"] = self.ax
        ax = self.divider.append_axes(side, size=size, pad=pad, **kws)
        if side == "top":
            self.top_axs.append(ax)
        elif side == "bottom":
            self.bottom_axs.append(ax)
        elif side == "left":
            self.left_axs.append(ax)
        elif side == "right":
            self.right_axs.append(ax)
        return ax


colors_var = "merge_class"
pad = 0.01
data = np.eye(len(lp_inds))
matrixgrid = MatrixGrid(data)
ax = matrixgrid.ax
top_colors_ax = matrixgrid.append_axes("top", size="10%", pad=pad)
top_dendrogram_ax = matrixgrid.append_axes("top", size="20%", pad=pad)
left_colors_ax = matrixgrid.append_axes("left", size="10%", pad=pad)
left_dendrogram_ax = matrixgrid.append_axes("left", size="20%", pad=pad)

labels1 = rows[0]["hier_pred_labels"]
labels2 = rows[1]["hier_pred_labels"]
cluster1_meta = rows[0]["nodes"].copy()
cluster2_meta = rows[1]["nodes"].copy()
print((cluster1_meta["pair"] == cluster2_meta.index).all())
cluster1_meta["_inds"] = range(len(cluster1_meta))
cluster2_meta["_inds"] = range(len(cluster2_meta))
cluster1_meta, cluster1_map = preprocess_meta(
    cluster1_meta, labels1, max_level=max_level
)
cluster2_meta, cluster2_map = preprocess_meta(
    cluster2_meta, labels2, max_level=max_level
)

from graspologic.plot.plot_matrix import scattermap

# data = data.reindex(index=cluster1_meta.index, columns=cluster2_meta.index)
height = cluster1_meta["_inds"].max() + 1
width = cluster2_meta["_inds"].max() + 1
data = np.zeros((height, width))

for row_ind in range(len(cluster1_meta)):
    i = cluster1_meta.loc[cluster1_meta.index[row_ind], "_inds"]
    j = cluster2_meta.loc[
        cluster1_meta.loc[cluster1_meta.index[row_ind], "pair"], "_inds"
    ]
    data[i, j] = 1
scattermap(data, ax=ax, sizes=(4, 4))
for side in ["left", "right", "top", "bottom"]:
    ax.spines[side].set_visible(True)

cluster1_meta_tree = construct_meta_tree(cluster1_meta, groupers, cluster1_map)
cluster2_meta_tree = construct_meta_tree(cluster2_meta, groupers, cluster2_map)
plot_colorstrip(
    cluster1_meta, colors_var, ax=left_colors_ax, orient="h", palette=palette
)
plot_dendrogram(cluster1_meta_tree, ax=left_dendrogram_ax, orient="h")

plot_colorstrip(
    cluster2_meta, colors_var, ax=top_colors_ax, orient="v", palette=palette
)
plot_dendrogram(cluster2_meta_tree, ax=top_dendrogram_ax, orient="v")
stashfig("full-confusion-mat", format="pdf")
stashfig("full-confusion-mat", format="png")

#%%
level = 6
hier_labels1 = rows[0]["hier_pred_labels"]
hier_labels2 = rows[1]["hier_pred_labels"]
cluster1_meta = rows[0]["nodes"].copy()
cluster2_meta = rows[1]["nodes"].copy()
cluster1_meta, cluster1_map = apply_flat_labels(cluster1_meta, hier_labels1)
cluster2_meta, cluster2_map = apply_flat_labels(cluster2_meta, hier_labels2)
labels1 = cluster1_meta[f"lvl{level}_labels"].values
labels2 = cluster2_meta[f"lvl{level}_labels"].values
from sklearn.metrics import confusion_matrix
from graspologic.utils import remap_labels
from giskard.plot import confusionplot, stacked_barplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from giskard.plot import soft_axis_off, axis_on

# stacked_barplot()


labels2 = remap_labels(labels1, labels2)
cluster2_meta[f"lvl{level}_labels"] = labels2

ax, conf_mat = confusionplot(
    labels1,
    labels2,
    annot=False,
    xticklabels=False,
    yticklabels=False,
    return_confusion_matrix=True,
    title=False,
    normalize="true",
)
axis_on(ax)
divider = make_axes_locatable(ax)


def plot_stacked_bars(groups, colors, index, ax=None, normalize=False, orient="h"):
    if ax is None:
        ax = plt.gca()
    counts_by_cluster = pd.crosstab(
        index=groups,
        columns=colors,
    )
    for i, (cluster_idx, row) in enumerate(counts_by_cluster.iterrows()):
        row /= row.sum()
        i = np.squeeze(np.argwhere(index == cluster_idx))
        stacked_barplot(
            row,
            center=i + 0.5,
            palette=palette,
            ax=ax,
            orient=orient,
        )
        ax.set(xticks=[], yticks=[])
        soft_axis_off(ax)
    return ax


left_ax = divider.append_axes("left", size="20%", pad=0.01, sharey=ax)
plot_stacked_bars(
    cluster1_meta[f"lvl{level}_labels"],
    cluster1_meta["merge_class"],
    conf_mat.index,
    normalize=True,
    orient="h",
)

top_ax = divider.append_axes("top", size="20%", pad=0.02, sharex=ax)
plot_stacked_bars(
    cluster2_meta[f"lvl{level}_labels"],
    cluster2_meta["merge_class"],
    conf_mat.columns,
    normalize=True,
    orient="v",
)
ax.set_xlabel("Right clustering")
left_ax.set_ylabel("Left clustering")
top_ax.set_title("Confusion matrix (row normalized)")
stashfig(f"confusion-lvl{level}")