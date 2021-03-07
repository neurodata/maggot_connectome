#%%
from itertools import chain, combinations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from graspologic.models import DCSBMEstimator, SBMEstimator
from graspologic.plot import adjplot
from graspologic.utils import binarize, remove_loops
from pkg.data import load_adjacency, load_networkx, load_node_meta, load_palette
from pkg.plot import set_theme

palette = load_palette()

meta = load_node_meta()
meta = meta[meta["paper_clustered_neurons"]]

#%%
# make some new metadata columns
assert (
    meta[["ipsilateral_axon", "contralateral_axon", "bilateral_axon"]].sum(axis=1).max()
    == 1
)

meta["axon_lat"] = "other"
meta.loc[meta[meta["ipsilateral_axon"]].index, "axon_lat"] = "ipsi"
meta.loc[meta[meta["contralateral_axon"]].index, "axon_lat"] = "contra"
meta.loc[meta[meta["bilateral_axon"]].index, "axon_lat"] = "bi"


#%%

g = load_networkx(graph_type="G", node_meta=meta)
adj = nx.to_numpy_array(g, nodelist=meta.index)
adj = binarize(adj)


def group_repr(group_keys):
    if len(group_keys) == 1:
        return str(group_keys[0])
    else:
        out = f"{group_keys[0]}"
        for key in group_keys[1:]:
            out += " x "
            out += key
        return out


#%%


def powerset(iterable, ignore_empty=True):
    # REF: https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(
        chain.from_iterable(combinations(s, r) for r in range(ignore_empty, len(s) + 1))
    )


powerset(["hemisphere", "simple_group", "lineage", "axon_lat"])

#%%

single_group_keys = ["hemisphere", "simple_group", "lineage", "axon_lat"]
all_group_keys = powerset(single_group_keys)

rows = []
for group_keys in all_group_keys:
    group_keys = list(group_keys)
    print(group_keys)

    # this is essentially just grabbing the labels that are group_key1 cross group_key2
    # etc. and then applying this as a column in the metadata
    groupby = meta.groupby(group_keys)
    groups = groupby.groups
    unique_group_keys = list(sorted(groups.keys()))
    group_map = dict(zip(unique_group_keys, range(len(unique_group_keys))))
    if len(group_keys) > 1:
        meta["_group"] = list(meta[group_keys].itertuples(index=False, name=None))
    else:
        meta["_group"] = meta[group_keys]
    meta["_group_id"] = meta["_group"].map(group_map)

    # fitting the simple SBM
    Model = SBMEstimator
    estimator = Model(directed=True, loops=False)
    estimator.fit(adj, y=meta["_group_id"].values)

    # evaluating and saving results
    estimator.n_verts = len(adj)
    bic = estimator.bic(adj)
    lik = estimator.score(adj)
    n_params = estimator._n_parameters()
    penalty = 2 * np.log(len(adj)) * n_params
    row = {
        "bic": bic,
        "lik": lik,
        "group_keys": group_repr(group_keys),
        "n_params": n_params,
        "penalty": penalty,
    }
    for g in single_group_keys:
        if g in group_keys:
            row[g] = True
        else:
            row[g] = False
    rows.append(row)

#%%
results = pd.DataFrame(rows)
results.sort_values("bic", ascending=True, inplace=True)
results["rank"] = np.arange(len(results), 0, -1)
#%%

set_theme()

# colors = sns.color_palette("husl", n_colors=results["group_keys"].nunique())
# colors = sns.color_palette("cubehelix", )
colors = sns.cubehelix_palette(
    start=0.5, rot=-0.5, n_colors=results["group_keys"].nunique()
)[::-1]

palette = dict(zip(results["group_keys"].unique(), colors))

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
ax = axs[0]
sns.stripplot(data=results, x="group_keys", y="bic", ax=ax, s=10, palette=palette)
plt.setp(ax.get_xticklabels(), rotation=45, rotation_mode="anchor", ha="right")

ax = axs[1]
sns.stripplot(data=results, x="group_keys", y="bic", ax=ax, s=10, palette=palette)
plt.setp(ax.get_xticklabels(), rotation=45, rotation_mode="anchor", ha="right")
ax.set_yscale("log")

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.stripplot(
    data=results.iloc[:6],
    x="group_keys",
    y="bic",
    ax=ax,
    s=10,
    palette=palette,
)
plt.setp(ax.get_xticklabels(), rotation=45, rotation_mode="anchor", ha="right")


fig, axs = plt.subplots(1, 2, figsize=(12, 6))
ax = axs[0]
sns.scatterplot(
    data=results,
    x="n_params",
    y="lik",
    hue="group_keys",
    palette=palette,
    legend=False,
    size="rank",
    ax=ax,
)
ax = axs[1]
sns.scatterplot(
    data=results.iloc[:6],
    x="n_params",
    y="lik",
    hue="group_keys",
    palette=palette,
    legend=False,
    size="rank",
    ax=ax,
)

#%%
# adjplot(
#     adj,
#     meta=meta,
#     plot_type="scattermap",
#     group="simple_group",
#     ticks=False,
#     color="simple_group",
#     palette=palette,
#     sizes=(1, 2),
# )

# adjplot(
#     adj,
#     meta=meta,
#     plot_type="scattermap",
#     group=["hemisphere", "simple_group"],
#     ticks=False,
#     color="simple_group",
#     palette=palette,
#     sizes=(1, 2),
# )

#%%
from upsetplot import plot, UpSet


intersections = results.set_index(single_group_keys)
fig, axs = plt.subplots(
    2,
    1,
    figsize=(8, 6),
    gridspec_kw=dict(height_ratios=[0.75, 0.4], hspace=0),
    sharex=True,
)
ax = axs[0]
sns.stripplot(
    data=results, jitter=0, x="group_keys", y="bic", ax=ax, s=10, palette=palette
)
ax.set(xticks=[], xticklabels=[], xlabel="")
# plt.setp(ax.get_xticklabels(), rotation=45, rotation_mode="anchor", ha="right")
ax = axs[1]
plot_upset_indicators(intersections, ax=ax)
ax.set_xlabel("Grouping")

#%%

from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_upset_indicators(
    intersections,
    ax=None,
    facecolor="black",
    element_size=None,
    with_lines=True,
    horizontal=True,
    height_pad=0.7,
):
    # REF: https://github.com/jnothman/UpSetPlot/blob/e6f66883e980332452041cd1a6ba986d6d8d2ae5/upsetplot/plotting.py#L428
    """Plot the matrix of intersection indicators onto ax"""
    data = intersections
    n_cats = data.index.nlevels

    idx = np.flatnonzero(data.index.to_frame()[data.index.names].values)
    c = np.array(["lightgrey"] * len(data) * n_cats, dtype="O")
    c[idx] = facecolor
    x = np.repeat(np.arange(len(data)), n_cats)
    y = np.tile(np.arange(n_cats), len(data))
    if element_size is not None:
        s = (element_size * 0.35) ** 2
    else:
        # TODO: make s relative to colw
        s = 200
    ax.scatter(x, y, c=c.tolist(), linewidth=0, s=s)

    if with_lines:
        line_data = (
            pd.Series(y[idx], index=x[idx]).groupby(level=0).aggregate(["min", "max"])
        )
        ax.vlines(
            line_data.index.values,
            line_data["min"],
            line_data["max"],
            lw=2,
            colors=facecolor,
        )

    tick_axis = ax.yaxis
    tick_axis.set_ticks(np.arange(n_cats))
    tick_axis.set_ticklabels(data.index.names, rotation=0 if horizontal else -90)
    # ax.xaxis.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    if not horizontal:
        ax.yaxis.set_ticks_position("top")
    ax.set_frame_on(False)
    ax.set_ylim((-height_pad, n_cats - 1 + height_pad))


def upset_stripplot(data, x=None, y=None, ax=None, upset_ratio=0.3, **kwargs):

    divider = make_axes_locatable(ax)

    sns.stripplot(data=data, x=x, y=y, ax=ax, **kwargs)
    ax.set(xticks=[], xticklabels=[], xlabel="")

    upset_ax = divider.append_axes(
        "bottom", size=f"{upset_ratio*100}%", pad=0, sharex=ax
    )
    plot_upset_indicators(data, ax=upset_ax)
    return ax, upset_ax


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
upset_stripplot(
    intersections,
    jitter=0,
    x="group_keys",
    y="bic",
    ax=ax,
    s=10,
    palette=palette,
    upset_ratio=0.35,
)
