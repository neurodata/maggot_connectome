#%% [markdown]
# # Look at RDPG models
#%% [markdown]
# ## Preliminaries
#%%
import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from giskard.plot import graphplot
from graspologic.align import OrthogonalProcrustes, SeedlessProcrustes
from graspologic.embed import (
    AdjacencySpectralEmbed,
    OmnibusEmbed,
    select_dimension,
    selectSVD,
)
from graspologic.match import GraphMatch
from graspologic.plot import pairplot
from graspologic.utils import (
    augment_diagonal,
    binarize,
    multigraph_lcc_intersection,
    pass_to_ranks,
)
from matplotlib.collections import LineCollection
from pkg.data import load_maggot_graph, load_palette
from pkg.io import savefig
from pkg.plot import set_theme

from pkg.utils import set_warnings
from sklearn.preprocessing import normalize
from src.visualization import adjplot
from src.visualization import CLASS_COLOR_DICT
from umap import AlignedUMAP
from factor_analyzer import Rotator
from src.visualization import matrixplot
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib import cm
from giskard.utils import get_paired_inds
from giskard.align import joint_procrustes
from pkg.data import load_network_palette, load_node_palette, select_nice_nodes

t0 = time.time()


def stashfig(name, **kwargs):
    foldername = "look_at_it_rdpg"
    savefig(name, foldername=foldername, **kwargs)


colors = sns.color_palette("Set2")

palette = dict(zip(["Left", "Right"], [colors[0], colors[1]]))
set_theme()

# %% [markdown]
# ## Load and process data
#%%


def calculate_weighted_degrees(adj):
    return np.sum(adj, axis=0) + np.sum(adj, axis=1)


t0 = time.time()
set_theme()

NETWORK_PALETTE, NETWORK_KEY = load_network_palette()
NODE_PALETTE, NODE_KEY = load_node_palette()

mg = load_maggot_graph()
mg = select_nice_nodes(mg)
nodes = mg.nodes
nodes["sum_signal_flow"] = -nodes["sum_signal_flow"]
adj = mg.sum.adj
nodes["degree"] = -calculate_weighted_degrees(adj)
left_nodes = nodes[nodes["hemisphere"] == "L"]
left_inds = left_nodes["_inds"]
right_nodes = nodes[nodes["hemisphere"] == "R"]
right_inds = right_nodes["_inds"]
left_paired_inds, right_paired_inds = get_paired_inds(
    nodes, pair_key="predicted_pair", pair_id_key="predicted_pair_id"
)
right_paired_inds_shifted = right_paired_inds - len(left_inds)
ll_adj = adj[np.ix_(left_inds, left_inds)]
rr_adj = adj[np.ix_(right_inds, right_inds)]


#%% [markdown]
# ## Embed the network using adjacency spectral embedding
#%%
def preprocess_for_embed(ll_adj, rr_adj, preprocess):
    if "binarize" in preprocess:
        ll_adj_to_embed = binarize(ll_adj)
        rr_adj_to_embed = binarize(rr_adj)

    if "rescale" in preprocess:
        ll_norm = np.linalg.norm(ll_adj_to_embed, ord="fro")
        rr_norm = np.linalg.norm(rr_adj_to_embed, ord="fro")
        mean_norm = (ll_norm + rr_norm) / 2
        ll_adj_to_embed *= mean_norm / ll_norm
        rr_adj_to_embed *= mean_norm / rr_norm
    return ll_adj_to_embed, rr_adj_to_embed


def embed(adj, n_components=40, ptr=False):
    if ptr:
        adj = pass_to_ranks(adj)
    elbow_inds, _ = select_dimension(augment_diagonal(adj), n_elbows=5)
    elbow_inds = np.array(elbow_inds)
    ase = AdjacencySpectralEmbed(n_components=n_components)
    out_latent, in_latent = ase.fit_transform(adj)
    return out_latent, in_latent, ase.singular_values_, elbow_inds


def split_adj(adj):
    ll_adj = adj[np.ix_(left_inds, left_inds)]
    rr_adj = adj[np.ix_(right_inds, right_inds)]
    lr_adj = adj[np.ix_(left_inds, right_inds)]
    rl_adj = adj[np.ix_(right_inds, left_inds)]
    return ll_adj, rr_adj, lr_adj, rl_adj


def prescale_for_embed(adjs):
    norms = [np.linalg.norm(adj, ord="fro") for adj in adjs]
    mean_norm = np.mean(norms)
    adjs = [adjs[i] * mean_norm / norms[i] for i in range(len(adjs))]
    return adjs


def ase(adj, n_components=None):
    U, S, Vt = selectSVD(adj, n_components=n_components, algorithm="full")
    S_sqrt = np.diag(np.sqrt(S))
    X = U @ S_sqrt
    Y = Vt.T @ S_sqrt
    return X, Y


max_n_components = 40
preprocess = ["binarize", "rescale"]

raw_adj = mg.sum.adj.copy()

ll_adj, rr_adj, _, _ = split_adj(raw_adj)

ll_adj_to_embed, rr_adj_to_embed = preprocess_for_embed(ll_adj, rr_adj, preprocess)

X_ll, Y_ll, left_sing_vals, left_elbow_inds = embed(
    ll_adj_to_embed, n_components=max_n_components
)
X_rr, Y_rr, right_sing_vals, right_elbow_inds = embed(
    rr_adj_to_embed, n_components=max_n_components
)

#%% [markdown]
# ### Plot screeplots
#%%


def screeplot(sing_vals, elbow_inds, color=None, ax=None, label=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))
    plt.plot(range(1, len(sing_vals) + 1), sing_vals, color=color, label=label)
    plt.scatter(
        elbow_inds, sing_vals[elbow_inds - 1], marker="x", s=50, zorder=10, color=color
    )
    ax.set(ylabel="Singular value", xlabel="Index")
    return ax


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
screeplot(left_sing_vals, left_elbow_inds, color=palette["Left"], ax=ax, label="Left")
screeplot(
    right_sing_vals, right_elbow_inds, color=palette["Right"], ax=ax, label="Right"
)
ax.legend()
stashfig(f"screeplot")


#%% [markdown]
# ## Align the left and the right embeddings
#%%


def ase(adj, n_components=None):
    U, S, Vt = selectSVD(adj, n_components=n_components, algorithm="full")
    S_sqrt = np.diag(np.sqrt(S))
    X = U @ S_sqrt
    Y = Vt.T @ S_sqrt
    return X, Y


n_align_components = 32
X_ll_unaligned = X_ll[:, :n_align_components]
Y_ll_unaligned = Y_ll[:, :n_align_components]
X_rr = X_rr[:, :n_align_components]
Y_rr = Y_rr[:, :n_align_components]

X_ll, Y_ll = joint_procrustes(
    (X_ll_unaligned, Y_ll_unaligned),
    (X_rr, Y_rr),
    method="seeded",
    seeds=(left_paired_inds, right_paired_inds_shifted),
)

XY_ll = np.concatenate((X_ll, Y_ll), axis=1)
XY_rr = np.concatenate((X_rr, Y_rr), axis=1)
n_final_components = 20
Z_ll, _ = ase(XY_ll, n_components=n_final_components)
Z_rr, _ = ase(XY_rr, n_components=n_final_components)

#%% [markdown]
# ### Plot the left and the right embeddings in the same space after the alignment
#%%


def add_connections(x1, x2, y1, y2, color="black", alpha=0.2, linewidth=0.2, ax=None):
    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)
    if ax is None:
        ax = plt.gca()

    coords = []
    for i in range(len(x1)):
        coords.append([[x1[i], y1[i]], [x2[i], y2[i]]])
    lc = LineCollection(
        coords,
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
        zorder=0,
    )
    ax.add_collection(lc)


def plot_latents(
    left,
    right,
    title="",
    n_show=4,
    alpha=0.6,
    linewidth=0.4,
    s=10,
    connections=False,
    palette=None,
):
    if n_show > left.shape[1]:
        n_show = left.shape[1]
    plot_data = np.concatenate([left, right], axis=0)
    labels = np.array(["Left"] * len(left) + ["Right"] * len(right))
    pg = pairplot(
        plot_data[:, :n_show], labels=labels, title=title, size=s, palette=palette
    )
    if connections:
        axs = pg.axes
        for i in range(n_show):
            for j in range(n_show):
                if i != j:
                    ax = axs[i, j]
                    add_connections(
                        left[:, j],
                        right[:, j],
                        left[:, i],
                        right[:, i],
                        ax=ax,
                        alpha=alpha,
                        linewidth=linewidth,
                    )
    pg._legend.remove()
    return pg


plot_latents(X_ll_unaligned, X_rr, palette=palette, connections=False)
stashfig("out-latent-unaligned")

plot_latents(X_ll, X_rr, palette=palette, connections=False)
stashfig("out-latent-aligned")
#%% [markdown]
# ## Examine the models

#%% [markdown]
# ### Plot the RDPG $\hat{P}$ for the left and the right hemispheres
#%%

vmin = -1
vmax = 1
cmap = cm.get_cmap("RdBu_r")
norm = Normalize(vmin, vmax)
norm = SymLogNorm(linthresh=0.1, linscale=2, vmin=vmin, vmax=vmax, base=10)


n_components = 16
P_ll = X_ll[:, :n_components] @ Y_ll[:, :n_components].T
P_ll[P_ll < 0] = 0
P_rr = X_rr[:, :n_components] @ Y_rr[:, :n_components].T
P_rr[P_rr < 0] = 0


fig, axs = plt.subplots(2, 2, figsize=(20, 20))
adjplot_kws = dict(
    colors=NODE_KEY,
    palette=NODE_PALETTE,
    sort_class=NODE_KEY,
    class_order="sum_signal_flow",
    item_order="degree",
    ticks=False,
    gridline_kws=dict(linewidth=0),
)
scattermap_kws = dict(sizes=(1, 1))
ax = axs[0, 0]
_, _, top_ax, left_ax = adjplot(
    ll_adj,
    plot_type="scattermap",
    meta=left_nodes,
    color=NETWORK_PALETTE["Left"],
    ax=ax,
    **adjplot_kws,
    **scattermap_kws,
)
top_ax.set_title(r"L $\to$ L", fontsize="xx-large")
left_ax.set_ylabel("Observed network", fontsize="xx-large")
ax = axs[0, 1]
_, _, top_ax, _ = adjplot(
    rr_adj,
    plot_type="scattermap",
    meta=right_nodes,
    color=NETWORK_PALETTE["Right"],
    ax=ax,
    **adjplot_kws,
    **scattermap_kws,
)
top_ax.set_title(r"R $\to$ R", fontsize="xx-large")
heatmap_kws = dict(
    cmap=cmap,
    norm=norm,
    center=0,
    vmin=vmin,
    vmax=vmax,
    cbar=False,
)
ax = axs[1, 0]
_, _, _, left_ax = adjplot(
    P_ll,
    meta=left_nodes,
    ax=ax,
    **adjplot_kws,
    **heatmap_kws,
)
left_ax.set_ylabel(f"RDPG model (d={n_components})", fontsize="xx-large")
ax = axs[1, 1]
adjplot(
    P_rr,
    meta=right_nodes,
    ax=ax,
    **adjplot_kws,
    **heatmap_kws,
)
plt.tight_layout()
stashfig("phat-comparison")

#%% [markdown]
# ### Experimental: try to make sense of the individual components
#%%
n_components = 20


def varimax(X):
    return Rotator(normalize=False).fit_transform(X)


n_left = len(X_ll)
X_concat = np.concatenate((X_ll[:, :n_components], X_rr[:, :n_components]), axis=0)

X_concat = varimax(X_concat)
X_ll_varimax = X_concat[:n_left]
X_rr_varimax = X_concat[n_left:]
plot_latents(X_ll_varimax, X_rr_varimax, palette=palette)

XY_rr = np.concatenate((X_rr[:, :n_components], Y_rr[:, :n_components]), axis=0)
XY_rr_varimax = varimax(XY_rr)

X_rr_varimax = XY_rr_varimax[: len(X_rr)]
Y_rr_varimax = XY_rr_varimax[len(X_rr) :]


#%% [markdown]
# #### Just for the right, examine the individual components
#%%

for dimension in X_rr_varimax.T[:16]:
    nodes = right_nodes.copy()
    nodes["dimension"] = dimension
    nodes.sort_values("dimension", inplace=True)
    nodes["index"] = np.arange(len(nodes))
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.scatterplot(
        data=nodes,
        x="index",
        y="dimension",
        hue="merge_class",
        palette=CLASS_COLOR_DICT,
        legend=False,
        linewidth=0,
        s=5,
    )

#%% [markdown]
# #### Look at what the components mean in the probability space
#%%

nodes = right_nodes.copy()
alpha = 10
quantile = 0.99
for i in range(16):
    x = X_rr_varimax[:, i]
    y = Y_rr_varimax[:, i]
    Phat = x[:, None] @ y[:, None].T
    abs_Phat = np.abs(Phat)
    q = np.quantile(abs_Phat, quantile)
    mask = abs_Phat > q
    row_used = mask.any(axis=0)
    col_used = mask.any(axis=1)
    col_nodes = nodes.iloc[col_used].copy()
    row_nodes = nodes.iloc[row_used].copy()
    sub_Phat = Phat[row_used][:, col_used]
    expected_out_degree = np.sum(np.abs(sub_Phat), axis=1)
    expected_in_degree = np.sum(np.abs(sub_Phat), axis=0)
    row_nodes["expected_out_degree"] = -expected_out_degree
    col_nodes["expected_in_degree"] = -expected_in_degree
    matrixplot(
        sub_Phat,
        row_meta=row_nodes,
        col_meta=col_nodes,
        row_colors="merge_class",
        col_colors="merge_class",
        row_palette=CLASS_COLOR_DICT,
        col_palette=CLASS_COLOR_DICT,
        row_item_order="expected_out_degree",
        col_item_order="expected_in_degree",
        cmap=cmap,
        norm=norm,
        center=0,
        vmin=vmin,
        vmax=vmax,
    )
    stashfig(f"right-phat-component-{i}")


#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
