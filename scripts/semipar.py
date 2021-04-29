#%% [markdown]
# # Testing bilateral symmetry - semiparametric test
#%% [markdown]
# ## Preliminaries
#%%
from pkg.utils import set_warnings
import ast
import datetime
import pprint
import time


import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import histplot, scatterplot
from graspologic.align import OrthogonalProcrustes
from graspologic.embed import (
    AdjacencySpectralEmbed,
    MultipleASE,
    OmnibusEmbed,
    selectSVD,
)
from graspologic.inference import latent_position_test
from graspologic.simulations import p_from_latent
from graspologic.utils import (
    binarize,
    multigraph_lcc_intersection,
    remove_loops,
    symmetrize,
)
from joblib import Parallel, delayed
from numba import jit
from pkg.data import load_adjacency, load_node_meta
from pkg.inference import sample_rdpg
from pkg.io import get_out_dir, savefig
from pkg.plot import set_theme
from pkg.utils import get_paired_inds, get_paired_subgraphs, set_warnings

set_warnings()

t0 = time.time()

RECOMPUTE = False

foldername = "semipar"


def stashfig(name, **kwargs):
    savefig(name, foldername=foldername, **kwargs)


out_dir = get_out_dir(foldername=foldername)

colors = sns.color_palette("Set1")
palette = dict(zip(["Left", "Right"], colors))
set_theme()

#%% [markdown]
# ## Load the data
#%% [markdown]
# ### Load node metadata and select the subgraphs of interest
#%%
meta = load_node_meta()
meta = meta[meta["paper_clustered_neurons"]]

adj = load_adjacency(graph_type="G", nodelist=meta.index)

lp_inds, rp_inds = get_paired_inds(meta)
left_meta = meta.iloc[lp_inds]
right_meta = meta.iloc[rp_inds]

ll_adj, rr_adj, lr_adj, rl_adj = get_paired_subgraphs(adj, lp_inds, rp_inds)

adjs, lcc_inds = multigraph_lcc_intersection([ll_adj, rr_adj], return_inds=True)
ll_adj = adjs[0]
rr_adj = adjs[1]
print(f"{len(lcc_inds)} in intersection of largest connected components.")

print(f"Original number of valid pairs: {len(lp_inds)}")

left_meta = left_meta.iloc[lcc_inds]
right_meta = right_meta.iloc[lcc_inds]
meta = pd.concat((left_meta, right_meta))
n_pairs = len(ll_adj)
print(f"Number of pairs after taking LCC intersection: {n_pairs}")


#%%


def fix_P(P):
    P = P - np.diag(np.diag(P))
    P[P < 0] = 0
    P[P > 1] = 1
    return P


def estimate_models_and_tstat(
    adjacency1, adjacency2, n_components, method="ase", **kwargs
):
    if method == "ase" or method == "omni":
        if method == "ase":
            X1, Y1, X2, Y2 = aligned_ase(adjacency1, adjacency2, n_components, **kwargs)
        elif method == "omni":
            X1, Y1, X2, Y2 = omni(adjacency1, adjacency2, n_components, **kwargs)
        P1 = fix_P(X1 @ Y1.T)
        P2 = fix_P(X2 @ Y2.T)
        tstat = np.sqrt(
            np.linalg.norm(X1 - X2, ord="fro") ** 2
            + np.linalg.norm(Y1 - Y2, ord="fro") ** 2
        )
    elif method == "mase":
        U, V, R1, R2 = mase(adjacency1, adjacency2, n_components, **kwargs)
        P1 = fix_P(U @ R1 @ V.T)
        P2 = fix_P(U @ R2 @ V.T)
        tstat = np.linalg.norm(R1 - R2, ord="fro")
    return P1, P2, tstat


def aligned_ase(adjacency1, adjacency2, n_components, initial_n_components=None):
    if initial_n_components is None:
        initial_n_components = n_components
    ase = AdjacencySpectralEmbed(
        n_components=initial_n_components, check_lcc=False, diag_aug=True
    )
    n = len(adjacency1)
    X1, Y1 = ase.fit_transform(adjacency1)
    X2, Y2 = ase.fit_transform(adjacency2)
    data1 = np.concatenate((X1, Y1), axis=0)
    data2 = np.concatenate((X2, Y2), axis=0)
    data1 = OrthogonalProcrustes().fit_transform(data1, data2)
    X1 = data1[:n, :n_components]
    Y1 = data1[n:, :n_components]
    X2 = X2[:, :n_components]
    Y2 = Y2[:, :n_components]
    return X1, Y1, X2, Y2


def omni(adjacency1, adjacency2, n_components, **kwargs):
    omni = OmnibusEmbed(
        n_components=n_components, check_lcc=False, diag_aug=True, concat=False
    )
    Xs, Ys = omni.fit_transform([adjacency1, adjacency2])
    return Xs[0], Ys[0], Xs[1], Ys[1]


def mase(adjacency1, adjacency2, n_components, **kwargs):
    mase = MultipleASE(n_components=n_components, scaled=True, diag_aug=True)
    mase.fit([adjacency1, adjacency2])
    return mase.latent_left_, mase.latent_right_, mase.scores_[0], mase.scores_[1]


def bootstrap_sample(P, seed):
    rng = np.random.default_rng(seed)
    A1 = sample_rdpg(P, rng.integers(np.iinfo(np.int32).max))
    A2 = sample_rdpg(P, rng.integers(np.iinfo(np.int32).max))
    return A1, A2


def compute_tstat(A1, A2, method="ase", n_components=2, **kwargs):
    if method == "ase" or method == "omni":
        if method == "ase":
            X1, Y1, X2, Y2 = aligned_ase(A1, A2, n_components, **kwargs)
        elif method == "omni":
            X1, Y1, X2, Y2 = omni(A1, A2, n_components, **kwargs)
        tstat = np.sqrt(
            np.linalg.norm(X1 - X2, ord="fro") ** 2
            + np.linalg.norm(Y1 - Y2, ord="fro") ** 2
        )
    elif method == "mase":
        _, _, R1, R2 = mase(A1, A2, n_components, **kwargs)
        tstat = np.linalg.norm(R1 - R2, ord="fro")
    return tstat


def compute_null_distribution(P, n_bootstraps, rng=None, n_jobs=None, **kwargs):
    if rng is None:
        rng = np.random.default_rng()
    seeds = rng.integers(np.iinfo(np.int32).max, size=n_bootstraps)

    def sample_and_compute_tstat(seed, **kwargs):
        A1, A2 = bootstrap_sample(P, seed)
        tstat = compute_tstat(A1, A2, **kwargs)
        return tstat

    tstats = Parallel(n_jobs=n_jobs)(
        delayed(sample_and_compute_tstat)(seed) for seed in seeds
    )
    tstats = np.array(tstats)
    return tstats


def arrayize(string):
    return np.array(ast.literal_eval(string), dtype=float)


def compute_pvalue(observed, null):
    pvalue = (null > observed).mean()
    if pvalue == 0:
        pvalue = 1 / len(null)
    return pvalue


def prescale_for_embed(adjs):
    norms = [np.linalg.norm(adj, ord="fro") for adj in adjs]
    mean_norm = np.mean(norms)
    adjs = [adjs[i] * mean_norm / norms[i] for i in range(len(adjs))]
    return adjs


def run_semipar_experiment(
    A1,
    A2,
    min_n_components=6,
    max_n_components=10,
    n_bootstraps=50,
    methods=["omni", "mase"],
    random_seed=8888,
    initial_n_components=24,
    verbose=1,
    n_jobs=-2,
):
    rng = np.random.default_rng(8888)
    rows = []
    n_components_range = np.arange(min_n_components, max_n_components)
    for n_components in n_components_range:
        for method in methods:
            P1, P2, observed_tstat = estimate_models_and_tstat(
                A1,
                A2,
                n_components,
                method=method,
                initial_n_components=initial_n_components,
            )
            null_dist1 = compute_null_distribution(
                P1,
                n_bootstraps,
                rng=rng,
                n_jobs=n_jobs,
                n_components=n_components,
                method=method,
                initial_n_components=initial_n_components,
            )
            null_dist2 = compute_null_distribution(
                P2,
                n_bootstraps,
                rng=rng,
                n_jobs=n_jobs,
                method=method,
                n_components=n_components,
                initial_n_components=initial_n_components,
            )
            pvalue1 = compute_pvalue(observed_tstat, null_dist1)
            pvalue2 = compute_pvalue(observed_tstat, null_dist2)
            pvalues = (pvalue1, pvalue2)
            max_pvalue_ind = np.argmax(pvalues)
            pvalue = pvalues[max_pvalue_ind]
            null_dists = (null_dist1, null_dist2)
            max_pvalue_ind = np.argmax(pvalues)
            row = {
                "n_components": n_components,
                "method": method,
                "pvalue": pvalue,
                "pvalue1": pvalue1,
                "pvalue2": pvalue2,
                "tstat": observed_tstat,
                "null_dist": list(null_dists[max_pvalue_ind]),
            }
            rows.append(row)
            if verbose > 0:
                print(
                    f"n_components = {n_components}, method={method}, pvalue={pvalue}"
                )
    results = pd.DataFrame(rows)
    return results


def plot_tstat_distributions(results):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    set_theme()
    # expand out the test statistic null distributions
    explode_results = results[["n_components", "method", "null_dist"]].explode(
        "null_dist"
    )
    explode_results["tstat"] = explode_results["null_dist"].astype(float)

    # set up facet grid
    grid = sns.FacetGrid(
        explode_results,
        row="n_components",
        col="method",
        hue="method",
        aspect=10,
        height=1,
        sharex="col",
    )

    # plot kdes and a thin line for the bottom of the axis
    grid.map(sns.kdeplot, "tstat", fill=True, alpha=1, linewidth=1.5)
    grid.map(sns.kdeplot, "tstat", color="w", lw=2)
    grid.map(plt.axhline, y=0, lw=2, alpha=0.2, clip_on=False)

    # adjust
    grid.set_titles("")
    grid.set(yticks=[], xticks=[])
    grid.despine(bottom=True, left=True)

    # add some lines and labels
    for (n_components, method), ax in grid.axes_dict.items():
        obs_results = results[
            (results["method"] == method) & (results["n_components"] == n_components)
        ].iloc[0]
        tstat = obs_results["tstat"]
        pvalue = obs_results["pvalue"]

        # plot a line for the observed test statistic
        ax.axvline(tstat, 0, 0.6, color="black", linewidth=2, linestyle="-")
        if (
            n_components == results["n_components"].min()
            and method == results["method"].iloc[0]
        ):
            ax.text(
                tstat,
                0.63,
                "Observed",
                color="black",
                fontweight="bold",
                ha="center",
                va="bottom",
                transform=blended_transform_factory(ax.transData, ax.transAxes),
            )

        # plot the p value
        t = ax.text(
            0.95,
            0.25,
            f"p={pvalue:0.2f}",
            color="black",
            fontweight="bold",
            transform=ax.transAxes,
            va="center",
            ha="right",
            zorder=10,
        )
        t.set_bbox(dict(facecolor="white", alpha=0.7))

        # set the xaxis labels based on what is in that column
        if n_components == explode_results["n_components"].max():
            ax.set_xlabel(
                method.upper() + " test statistic", fontweight="bold", labelpad=10
            )

        # plot the number of components
        if method == results["method"].iloc[0]:
            ax.text(
                0,
                0.2,
                n_components,
                color="black",
                ha="left",
                va="center",
                transform=ax.transAxes,
            )

    grid.fig.subplots_adjust(hspace=-0.5)
    grid.fig.text(
        0, 0.45, "# of dimensions", rotation=90, fontweight="bold", va="center"
    )
    return grid


def saveload(results, name):
    results.to_csv(out_dir / name)
    results = pd.read_csv(
        out_dir / name,
        index_col=0,
        na_values="",
        keep_default_na=False,
        converters=dict(null_dist=arrayize),
    )
    return results


RECOMPUTE = True
experiment_kws = dict(
    min_n_components=1,
    max_n_components=10,
    n_bootstraps=100,
    methods=["ase", "omni", "mase"],
    random_seed=8888,
    initial_n_components=16,
    verbose=1,
    n_jobs=-2,
)
prescale = False
if RECOMPUTE:
    for null_n_components in [4, 8]:
        X1, Y1, X2, Y2 = aligned_ase(ll_adj, rr_adj, null_n_components)
        P1 = fix_P(X1 @ Y1.T)
        P2 = fix_P(X2 @ Y2.T)
        A1, A2 = bootstrap_sample(P1, 1)
        results = run_semipar_experiment(A1, A2, **experiment_kws)
        name = f"semipar-results-data=null-n_components={null_n_components}"
        results = saveload(results, name)
        grid = plot_tstat_distributions(results)
        grid.fig.suptitle(f"Data = RDPG, n_components={null_n_components}")
        stashfig(name)

    # compute on the real data
    A1 = ll_adj
    A2 = rr_adj
    preprocess = [binarize, remove_loops]
    graphs = [A1, A2]
    new_graphs = []
    for func in preprocess:
        for i, graph in enumerate(graphs):
            new_graphs.append(func(graph))
    if prescale:
        new_graphs = prescale_for_embed(new_graphs)
    A1 = new_graphs[0]
    A2 = new_graphs[1]
    results = run_semipar_experiment(A1, A2, **experiment_kws)
    name = "semipar-results-data=real"
    results = saveload(results, name)
    grid = plot_tstat_distributions(results)
    grid.fig.suptitle("Data = real")
    stashfig(name)


#%%


# stashfig("distribution-plot")
#

#%% [markdown]
# ## Plot p-values
#%%
# ax = scatterplot(
#     data=pvalues,
#     x="n_components",
#     y="pvalue",
#     hue="method",
#     shift="method",
#     shade=True,
#     shift_bounds=(-0.2, 0.2),
# )
# ax.set_yscale("log")
# styles = ["--", ":"]
# line_locs = [0.05, 1 / n_bootstraps]
# line_kws = dict(color="black", alpha=0.7, linewidth=1.5, zorder=-1)
# for loc, style in zip(line_locs, styles):
#     ax.axhline(loc, linestyle=style, **line_kws)
#     ax.text(ax.get_xlim()[-1] + 0.1, loc, loc, ha="left", va="center")
# stashfig("semipar-pvalues-by-dimension")

# #%%
# test_case = "rotation"
# embedding = "omnibus"
# n_components = 8
# n_bootstraps = 100
# n_repeats = 5
# rows = []
# for n_shuffle in [4, 8, 16]:

#     for repeat in range(n_repeats):
#         inds = np.arange(len(rr_adj))
#         choice_inds = np.random.choice(len(rr_adj), size=n_shuffle, replace=False)
#         shuffle_inds = choice_inds.copy()
#         np.random.shuffle(shuffle_inds)
#         inds[choice_inds] = inds[shuffle_inds]
#         rr_adj_shuffle = rr_adj[np.ix_(inds, inds)]
#         currtime = time.time()
#         pvalue, tstat, misc = latent_position_test(
#             ll_adj,
#             rr_adj_shuffle,
#             embedding=embedding,
#             n_components=n_components,
#             test_case=test_case,
#             n_bootstraps=n_bootstraps,
#         )
#         row = {
#             "pvalue": pvalue,
#             "tstat": tstat,
#             "n_shuffle": n_shuffle,
#             "n_components": n_components,
#             "n_bootstraps": n_bootstraps,
#             "embedding": embedding,
#             "test_case": test_case,
#             "repeat": repeat,
#         }
#         rows.append(row)
#         print(f"{time.time() - currtime:.3f} seconds elapsed.")
#         print(f"n_shuffle: {n_shuffle}")
#         print(f"test case: {test_case}")
#         print(f"embedding: {embedding}")
#         print(f"n_components: {n_components}")
#         print(f"p-value: {pvalue}")
#         print(f"tstat: {tstat}")
#         print()

# #%%
# results = pd.DataFrame(rows)
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# sns.scatterplot(data=results, x="n_shuffle", y="pvalue", ax=ax)
# stashfig("shuffle-p-values")

# %% [markdown]
# ## End
#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
