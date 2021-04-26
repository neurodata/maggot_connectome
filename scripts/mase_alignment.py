#%% [markdown]
# # MASE and alignment
# Investigating the use of MASE as a method for joint embedding, and the effects of
# different alignment techniques
#%% [markdown]
# ## Preliminaries
#%%
import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graspologic.align import OrthogonalProcrustes, SeedlessProcrustes
from graspologic.embed import (
    AdjacencySpectralEmbed,
    MultipleASE,
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
from pkg.data import load_maggot_graph
from pkg.io import savefig
from pkg.plot import set_theme
from pkg.utils import set_warnings
from src.visualization import CLASS_COLOR_DICT as palette
from src.visualization import add_connections, adjplot
from scipy.stats import ortho_group
from giskard.plot import simple_scatterplot
from giskard.plot import merge_axes

t0 = time.time()
set_theme()


def stashfig(name, **kwargs):
    foldername = "mase_alignment"
    savefig(name, foldername=foldername, **kwargs)


#%%


def plot_pairs(
    X,
    labels,
    n_show=6,
    model=None,
    left_pair_inds=None,
    right_pair_inds=None,
    equal=False,
    palette=None,
):
    """Plots pairwise dimensional projections, and draws lines between known pair neurons

    Parameters

    ----------
    X : [type]
        [description]
    labels : [type]
        [description]
    model : [type], optional
        [description], by default None
    left_pair_inds : [type], optional
        [description], by default None
    right_pair_inds : [type], optional
        [description], by default None
    equal : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    if n_show is not None:
        n_dims = n_show
    else:
        n_dims = X.shape[1]

    fig, axs = plt.subplots(
        n_dims - 1, n_dims - 1, sharex=False, sharey=False, figsize=(20, 20)
    )
    data = pd.DataFrame(data=X[:, :n_dims], columns=[str(i) for i in range(n_dims)])
    data["label"] = labels

    for i in range(n_dims - 1):
        for j in range(n_dims):
            ax = axs[i, j - 1]
            ax.axis("off")
            if i < j:
                sns.scatterplot(
                    data=data,
                    x=str(j),
                    y=str(i),
                    ax=ax,
                    alpha=0.7,
                    linewidth=0,
                    s=8,
                    legend=False,
                    hue="label",
                    palette=palette,
                )
                if left_pair_inds is not None and right_pair_inds is not None:
                    add_connections(
                        data.iloc[left_pair_inds, j],
                        data.iloc[right_pair_inds, j],
                        data.iloc[left_pair_inds, i],
                        data.iloc[right_pair_inds, i],
                        ax=ax,
                    )

    plt.tight_layout()
    return fig, axs


def joint_procrustes(data1, data2):
    n = len(data1[0])
    data1 = np.concatenate(data1, axis=0)
    data2 = np.concatenate(data2, axis=0)
    data1_mapped = OrthogonalProcrustes().fit_transform(data1, data2)
    data1 = (data1_mapped[:n], data1_mapped[n:])
    return data1


def prescale_for_embed(adjs):
    """Want to avoid any excess alignment issues simply from the input matrices having
    different Frobenius norms"""
    norms = [np.linalg.norm(adj, ord="fro") for adj in adjs]
    mean_norm = np.mean(norms)
    adjs = [adjs[i] * mean_norm / norms[i] for i in range(len(adjs))]
    return adjs


def double_heatmap(
    matrices,
    axs=None,
    cbar_ax=None,
    figsize=(10, 5),
    square=True,
    vmin=None,
    vmax=None,
    center=0,
    cmap="RdBu_r",
    xticklabels=False,
    yticklabels=False,
    **kwargs,
):
    if axs is None and cbar_ax is None:
        fig, axs = plt.subplots(
            2, 2, figsize=figsize, gridspec_kw=dict(height_ratios=[0.9, 0.05])
        )
        cbar_ax = merge_axes(fig, axs, rows=1)
    if isinstance(matrices, (list, tuple)):
        matrices = np.stack(matrices, axis=0)
    if vmax is None:
        vmax = np.max(matrices)
    if vmin is None:
        vmin = np.min(matrices)
    heatmap_kws = dict(
        square=square,
        vmin=vmin,
        vmax=vmax,
        center=center,
        cmap=cmap,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
    )
    ax = axs[0, 0]
    sns.heatmap(matrices[0], ax=ax, cbar=False, **heatmap_kws)
    ax = axs[0, 1]
    sns.heatmap(
        matrices[1],
        ax=ax,
        cbar_ax=cbar_ax,
        cbar_kws={"orientation": "horizontal", "shrink": 0.6},
        **heatmap_kws,
    )
    return fig, axs


#%% [markdown]
# ## A simple simulation to validate the joint Procrustes method
#%%
np.random.seed(88888)
n = 32
X1 = np.random.uniform(0.1, 0.9, (n, 2))
Y1 = np.random.multivariate_normal([1, 1], np.eye(2), n)
Q = ortho_group.rvs(2)
X2 = X1 @ Q
Y2 = Y1 @ Q

pred_X2, pred_Y2 = joint_procrustes((X2, Y2), (X1, Y1))

colors = sns.color_palette("deep", 10, desat=1)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
ax = axs[0]
simple_scatterplot(X1, color=colors[0], alpha=1, ax=ax)
simple_scatterplot(X2, color=colors[1], alpha=1, ax=ax)
simple_scatterplot(
    pred_X2, color=colors[3], marker="s", s=50, alpha=0.7, zorder=-1, ax=ax
)
ax.set(title=r"$X$")

ax = axs[1]
simple_scatterplot(Y1, color=colors[0], alpha=1, ax=ax)
simple_scatterplot(Y2, color=colors[1], alpha=1, ax=ax)
simple_scatterplot(
    pred_Y2, color=colors[3], marker="s", s=50, alpha=0.7, zorder=-1, ax=ax
)
ax.set(title=r"$Y$")
stashfig("joint-procrustes-demo")


# %% [markdown]
# ## Load and process data
#%%
mg = load_maggot_graph()
mg = mg[mg.nodes["paper_clustered_neurons"]]

ll_mg, rr_mg, lr_mg, rl_mg = mg.bisect(paired=True)
ll_adj = ll_mg.sum.adj.copy()
rr_adj = rr_mg.sum.adj.copy()

nodes = ll_mg.nodes
nodes["_inds"] = range(len(nodes))
sorted_nodes = nodes.sort_values(["simple_group", "merge_class"])
sort_inds = sorted_nodes["_inds"]

ll_adj = ll_adj[np.ix_(sort_inds, sort_inds)]
rr_adj = rr_adj[np.ix_(sort_inds, sort_inds)]

adjs, lcc_inds = multigraph_lcc_intersection([ll_adj, rr_adj], return_inds=True)
ll_adj = adjs[0]
rr_adj = adjs[1]
sorted_nodes = sorted_nodes.iloc[lcc_inds]
print(f"{len(lcc_inds)} in intersection of largest connected components.")

#%% [markdown]
# ## Embed using MASE
#%%
ll_adj = binarize(ll_adj)
rr_adj = binarize(rr_adj)
adjs = prescale_for_embed([ll_adj, rr_adj])

#%%
colors = sns.color_palette("Set1")
side_palette = dict(zip(["Left", "Right"], colors))
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
adjplot(
    ll_adj,
    plot_type="scattermap",
    sizes=(1, 2),
    ax=axs[0],
    title=r"Left $\to$ left",
    color=side_palette["Left"],
)
adjplot(
    rr_adj,
    plot_type="scattermap",
    sizes=(1, 2),
    ax=axs[1],
    title=r"Right $\to$ right",
    color=side_palette["Right"],
)
stashfig("left-right-induced-adjs")


#%%
n_components = 32
mase = MultipleASE(n_components=n_components, algorithm="full")
mase.fit(adjs)

#%% [markdown]
# ### Look at the $\hat{R}$ matrices that MASE estimates
#%%

fig, axs = double_heatmap(mase.scores_)
axs[0, 0].set(title=r"$\hat{R}_{LL}$")
axs[0, 1].set(title=r"$\hat{R}_{RR}$")
stashfig("R-matrices")

#%% [markdown]
# ### Look at the $\hat{P}$ matrices that MASE estimates

U = mase.latent_left_
V = mase.latent_right_
R_ll = mase.scores_[0]
R_rr = mase.scores_[1]
true_phat_ll = U @ R_ll @ V.T
true_phat_rr = U @ R_rr @ V.T

fig, axs = double_heatmap((true_phat_ll, true_phat_rr), figsize=(15, 7.5))

#%% [markdown]
# ## Construct per-node, per-graph representations from MASE with alignment
#%% [markdown]
# ### Which alignment to do?
# - We can either align in the condensed space of the $R$ matrices (after we decompose
# using an SVD) or align in the projected node-wise latent space like we normally would.
# - We can either align the out and in representations jointly (to solve for the same
# $Q$ to be applied to both out and in) or separately.

#%%
scaled = True

for align_space in ["score", "latent"]:
    for align_mode in ["joint", "separate"]:

        Z_ll, S_ll, W_ll_T = selectSVD(R_ll, n_components=len(R_ll), algorithm="full")
        Z_rr, S_rr, W_rr_T = selectSVD(R_rr, n_components=len(R_rr), algorithm="full")
        W_ll = W_ll_T.T
        W_rr = W_rr_T.T
        S_ll_sqrt = np.diag(np.sqrt(S_ll))
        S_rr_sqrt = np.diag(np.sqrt(S_rr))

        if scaled:
            Z_ll = Z_ll @ S_ll_sqrt
            W_ll = W_ll @ S_ll_sqrt
            Z_rr = Z_rr @ S_rr_sqrt
            W_rr = W_rr @ S_rr_sqrt

        if align_space == "score":
            if align_mode == "joint":
                Z_ll, W_ll = joint_procrustes((Z_ll, W_ll), (Z_rr, W_rr))
            else:
                op_out = OrthogonalProcrustes()
                Z_ll = op_out.fit_transform(Z_ll, Z_rr)
                op_in = OrthogonalProcrustes()
                W_ll = op_in.fit_transform(W_ll, W_rr)

        X_ll = U @ Z_ll
        Y_ll = V @ W_ll
        X_rr = U @ Z_rr
        Y_rr = V @ W_ll

        if align_space == "latent":
            if align_mode == "joint":
                X_ll, Y_ll = joint_procrustes((X_ll, Y_ll), (X_rr, Y_rr))
            else:
                op_out = OrthogonalProcrustes()
                X_ll = op_out.fit_transform(X_ll, X_rr)
                op_in = OrthogonalProcrustes()
                Y_ll = op_in.fit_transform(Y_ll, Y_rr)

        norm = np.sqrt(
            np.linalg.norm(X_ll - X_rr) ** 2 + np.linalg.norm(Y_ll - Y_rr) ** 2
        )

        data = np.concatenate((X_ll, X_rr), axis=0)
        left_inds = np.arange(len(X_ll))
        right_inds = np.arange(len(X_rr)) + len(X_ll)
        labels = sorted_nodes["merge_class"].values
        labels = np.concatenate((labels, labels), axis=0)
        fig, axs = plot_pairs(
            data,
            labels,
            left_pair_inds=left_inds,
            right_pair_inds=right_inds,
            palette=palette,
        )
        fig.suptitle(
            f"Align mode = {align_mode}, align space = {align_space}, norm of difference = {norm:0.4f}",
            y=1.03,
            fontsize="xx-large",
        )

        phat_ll = X_ll @ Y_ll.T
        phat_rr = X_rr @ Y_rr.T
        fig, axs = double_heatmap((phat_ll, phat_rr), figsize=(15, 7.6))

        print(align_space)
        print(align_mode)
        print(np.linalg.norm(phat_ll - true_phat_ll))
        print(np.linalg.norm(phat_rr - true_phat_rr))
        print()

# #%%
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# # cbar_ax = merge_axes(fig, axs, rows=1)
# # vmax = np.max(ipsi_mase.scores_)
# # vmin = np.min(ipsi_mase.scores_)
# vmin = -1
# vmax = 1
# heatmap_kws = dict(
#     square=True,
#     vmin=vmin,
#     vmax=vmax,
#     center=0,
#     cmap="RdBu_r",
#     xticklabels=False,
#     yticklabels=False,
#     cbar=False,
# )
# ax = axs[0]
# sns.heatmap(op_out.Q_, ax=ax, **heatmap_kws)
# # ax.set(title=r"$\hat{R}_{LL}$")
# ax = axs[1]
# sns.heatmap(
#     op_in.Q_,
#     ax=ax,
#     **heatmap_kws,
# )
# # ax.set(title=r"$\hat{R}_{RR}$")
# ax = axs[2]
# sns.heatmap(
#     op_out.Q_ - op_in.Q_,
#     ax=ax,
#     **heatmap_kws,
# )
# # stashfig("ipsi-R-matrices")

# #%%
# sns.heatmap(op_out.Q_ @ op_in.Q_.T, **heatmap_kws)

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
