#%% [markdown]
# # Node matching
# Here we briefly explain the idea of using network (or graph) matching to uncover an
# alignment between the nodes of two networks.
#%% [markdown]
# ## Preliminaries
#%%
from networkx.relabel import relabel_nodes
from pkg.utils import set_warnings

import datetime
import pprint
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.utils import get_paired_inds
from graspologic.align import OrthogonalProcrustes, SeedlessProcrustes
from graspologic.embed import AdjacencySpectralEmbed, select_dimension
from graspologic.utils import augment_diagonal, binarize, pass_to_ranks
from hyppo.ksample import KSample
from pkg.data import (
    load_maggot_graph,
    load_navis_neurons,
    load_network_palette,
    load_node_palette,
)
from pkg.io import savefig
from pkg.plot import set_theme
from sklearn.metrics.pairwise import cosine_similarity
from pkg.plot import set_theme
from graspologic.simulations import er_corr
import networkx as nx
from giskard.plot import merge_axes, soft_axis_off
from matplotlib.colors import ListedColormap
from graspologic.match import GraphMatch
from graspologic.plot import heatmap


def stashfig(name, **kwargs):
    foldername = "introduce_node_matching"
    savefig(name, foldername=foldername, **kwargs)
    savefig(name, foldername=foldername, **kwargs)


network_palette, NETWORK_KEY = load_network_palette()
t0 = time.time()
set_theme()

#%% [markdown]
# ## Illustrate graph matching on a simulated dataset

#%%


def set_light_border(ax):
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.5)
        ax.spines[side].set_color("lightgrey")


# sample two networks
np.random.seed(8)
# A1, A2 = er_corr(16, 0.3, 0.9)
A1, A2 = er_corr(16, 0.3, 0.9)

# set up the plot
heatmap_kws = dict(cbar=False, center=None)
fig, axs = plt.subplots(
    2, 4, figsize=(13, 6), gridspec_kw=dict(width_ratios=[3, 3, 3, 3])
)

# plot graph 1 adjacency
ax = axs[0, 0]
colors = ["white", network_palette["Left"]]
cmap = ListedColormap(colors)
heatmap(A1, ax=ax, title="Network 1\n(random order)", cmap=cmap, **heatmap_kws)
set_light_border(ax)
ax.set_ylabel("Adjacency")

# plot graph 2 adjacency, with a random shuffling of the nodes
ax = axs[0, 1]
shuffle_inds = np.random.permutation(len(A2))
A2_shuffled = A2[shuffle_inds][:, shuffle_inds]
colors = ["white", network_palette["Right"]]
cmap = ListedColormap(colors)
heatmap(
    A2_shuffled,
    ax=ax,
    title="Network 2\n(random order)",
    cmap=cmap,
    **heatmap_kws,
)
set_light_border(ax)

# create networkx graphs from the adjacencies
g1 = nx.from_numpy_array(A1)
nodelist = np.arange(len(A2))[::-1]
g2 = nx.from_numpy_array(A2_shuffled)

# plot the first graph as a ball-and-stick
ax = axs[1, 0]
pos1 = nx.kamada_kawai_layout(g1)
nx.draw_networkx(
    g1,
    with_labels=False,
    pos=pos1,
    ax=ax,
    node_color=network_palette["Left"],
    node_size=100,
    edge_color=network_palette["Left"],
)
ax.axis("on")
soft_axis_off(ax)
ax.set_ylabel("Layout")

# plot the second graph as a ball-and-stick, with it's own predicted layout
ax = axs[1, 1]
# nx.relabel_nodes(g2, dict(zip(np.arange(len(A2)), shuffle_inds)))
pos2 = nx.kamada_kawai_layout(g2)
nx.draw_networkx(
    g2,
    with_labels=False,
    pos=pos2,
    ax=ax,
    node_color=network_palette["Right"],
    node_size=100,
    edge_color=network_palette["Right"],
)
soft_axis_off(ax)

# draw an arrow in the middle of the plot
# ax = merge_axes(fig, axs, rows=None, cols=2)
# ax.axis("off")
# text = "Network matching"
grey = (0.5, 0.5, 0.5)
ax = axs[0, 2]
colors = [
    (1, 1, 1),
    network_palette["Left"],
    network_palette["Right"],
    grey,
]
diff_cmap = ListedColormap(colors)
# heatmap_kws['center'] = None
diff_indicator = np.zeros_like(A1)
diff_indicator[(A1 == 1) & (A2_shuffled != 1)] = 1
diff_indicator[(A1 != 1) & (A2_shuffled == 1)] = 2
diff_indicator[(A1 == 1) & (A2_shuffled == 1)] = 3
heatmap(
    diff_indicator,
    ax=ax,
    title="Overlay\n(random order)",
    cmap=diff_cmap,
    **heatmap_kws,
)
set_light_border(ax)

print(np.linalg.norm(A1 - A2_shuffled) ** 2)


def color_edges(g1, g2):
    edge_colors = []
    edgelist = []
    for edge in g2.edges:
        edgelist.append(edge)
        if g1.has_edge(*edge):
            edge_colors.append(grey)
        else:
            edge_colors.append(network_palette["Right"])
    for edge in g1.edges:
        if edge not in edgelist:
            edgelist.append(edge)
            edge_colors.append(network_palette["Left"])
    return edgelist, edge_colors


ax = axs[1, 2]
edgelist, edge_colors = color_edges(g1, g2)
nx.draw_networkx(
    g2,
    with_labels=False,
    pos=pos1,
    ax=ax,
    node_color=grey,
    node_size=100,
    edge_color=edge_colors,
    edgelist=edgelist,
)
soft_axis_off(ax)
# ax.text(
#     0.5,
#     0.55,
#     text,
#     ha="center",
#     va="center",
# )
# text = r"$\rightarrow$"
# ax.text(
#     0.5,
#     0.45,
#     text,
#     ha="center",
#     va="center",
#     fontsize=60,
# )

# graph match to recover the pairing
gm = GraphMatch(n_init=30, shuffle_input=True)
perm_inds = gm.fit_predict(A1, A2)
ax = axs[0, 3]
A2_perm = A2[perm_inds][:, perm_inds]
diff_indicator = np.zeros_like(A1)
diff_indicator[(A1 == 1) & (A2_perm != 1)] = 1
diff_indicator[(A1 != 1) & (A2_perm == 1)] = 2
diff_indicator[(A1 == 1) & (A2_perm == 1)] = 3
heatmap(
    diff_indicator,
    ax=ax,
    title="Overlay\n(predicted matching)",
    cmap=diff_cmap,
    **heatmap_kws,
)
set_light_border(ax)

print(np.linalg.norm(A1 - A2_perm, ord="fro") ** 2)

# plot the second network after alignment, color edges based on matchedness

g2 = nx.from_numpy_array(A2_perm)
edgelist, edge_colors = color_edges(g1, g2)
ax = axs[1, 3]
nx.draw_networkx(
    g2,
    with_labels=False,
    pos=pos1,
    ax=ax,
    node_color=grey,
    node_size=100,
    edgelist=edgelist,
    edge_color=edge_colors,
)
soft_axis_off(ax)


import matplotlib as mpl

bounds = [0, 1, 2, 3, 4]
norm = mpl.colors.BoundaryNorm(bounds, diff_cmap.N)
cbar = fig.colorbar(
    mpl.cm.ScalarMappable(cmap=diff_cmap, norm=norm),
    ax=axs.ravel().tolist(),
    shrink=0.3,
    fraction=0.05,
    aspect=8,
)
cbar.set_ticks([0.5, 1.5, 2.5, 3.5])
cbar.set_ticklabels(["No edge", "Edge in 1", "Edge in 2", "Edge in both"])
cbar.outline.set_linewidth(1.5)
cbar.outline.set_edgecolor("lightgrey")

# set_light_border(cbar.ax)
# cbar.ax.spines['left'].set_visible(False)

# cbar = fig.colorbar(diff_cmap,  shrink=0.95)

# ax = merge_axes(fig, axs, cols=4)

# plt.colorbar(ListedColormap, ax)

# bounds = [0, 4]
# # norm = mpl.colors.BoundaryNorm(bounds, diff_cmap.N)
# cb = mpl.colorbar.ColorbarBase(
#     ax,
#     cmap=diff_cmap,
# #     # norm=norm,
#     # boundaries=[[0]] + bounds + [13],
# #     # extend="both",
#     # ticks=[0, 1, 2, 3],
# #     # spacing="proportional",
#     extendfrac=0.5,
#     orientation="vertical",
# )

# set_light_border(ax)
# plt.colorbar(cmap, ax)


stashfig("network-matching-explanation")
stashfig("network-matching-explanation", format="svg", pad_inches=0)

#%%


# cmap = mpl.colors.ListedColormap(colors)
# ax = heatmap(X, center=None, cmap=cmap, **kwargs)
# colorbar = ax.collections[0].colorbar
# cbar = kwargs.setdefault("cbar", True)
# if cbar:
#     colorbar.set_ticks([0.25, 0.75])
#     colorbar.set_ticklabels(colorbar_ticklabels)

# %%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
