#%% [markdown]
# # Testing bilateral symmetry - semiparametric test
#%% [markdown]
# ## Preliminaries
#%%
import datetime
import pprint
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from giskard.plot import scatterplot
from graspologic.inference import latent_position_test
from graspologic.utils import binarize, multigraph_lcc_intersection, symmetrize
from pkg.data import load_adjacency, load_node_meta
from pkg.io import get_out_dir, savefig
from pkg.plot import set_theme
from pkg.utils import get_paired_inds, get_paired_subgraphs, set_warnings

from graspologic.align import OrthogonalProcrustes
from graspologic.embed import AdjacencySpectralEmbed, OmnibusEmbed, selectSVD
from numba import jit

from graspologic.embed import MultipleASE

from graspologic.simulations import p_from_latent
from joblib import Parallel, delayed


from graspologic.utils import remove_loops
from giskard.plot import histplot

# from src.visualization import adjplot  # TODO fix graspologic version and replace here

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


def joint_procrustes(data1, data2, method="orthogonal"):
    n = len(data1[0])
    procruster = OrthogonalProcrustes()
    data1 = np.concatenate(data1, axis=0)
    data2 = np.concatenate(data2, axis=0)
    data1_mapped = procruster.fit_transform(data1, data2)
    data1 = (data1_mapped[:n], data1_mapped[n:])
    return data1


# @jit(nopython=True)
def difference_norm(A, B):
    return np.linalg.norm(A - B, ord="fro")


def compute_ase_tstat(adjacency1, adjacency2, n_components, initial_n_components=None):
    if initial_n_components is None:
        initial_n_components = n_components
    ase = AdjacencySpectralEmbed(
        n_components=initial_n_components, check_lcc=False, diag_aug=True
    )
    X1, Y1 = ase.fit_transform(adjacency1)
    X2, Y2 = ase.fit_transform(adjacency2)
    data1 = np.concatenate((X1, Y1), axis=0)
    data2 = np.concatenate((X2, Y2), axis=0)
    data1 = OrthogonalProcrustes().fit_transform(data1, data2)
    return difference_norm(data1[:, :n_components], data2[:, :n_components])


def compute_omni_tstat(adjacency1, adjacency2, n_components):
    omni = OmnibusEmbed(
        n_components=n_components, check_lcc=False, diag_aug=True, concat=True
    )
    XYs = omni.fit_transform([adjacency1, adjacency2])
    return difference_norm(XYs[0], XYs[1])


def decompose_scores(scores, scaled=True, align=True):
    R1 = scores[0]
    R2 = scores[1]
    Z1, S1, W1t = selectSVD(R1, n_components=len(R1), algorithm="full")
    W1 = W1t.T
    Z2, S2, W2t = selectSVD(R2, n_components=len(R2), algorithm="full")
    W2 = W2t.T
    if scaled:
        S1_sqrt = np.diag(np.sqrt(S1))
        S2_sqrt = np.diag(np.sqrt(S2))
        Z1 = Z1 @ S1_sqrt
        Z2 = Z2 @ S2_sqrt
        W1 = W1 @ S1_sqrt
        W2 = W2 @ S2_sqrt
    if align:
        op = OrthogonalProcrustes()
        n = len(Z1)
        U1 = np.concatenate((Z1, W1), axis=0)
        U2 = np.concatenate((Z2, W2), axis=0)
        U1_mapped = op.fit_transform(U1, U2)
        Z1 = U1_mapped[:n]
        W1 = U1_mapped[n:]
    return Z1, W1, Z2, W2


def project_to_node_space(mase, scaled=True, align=True):
    U = mase.latent_left_
    V = mase.latent_right_
    Z1, W1, Z2, W2 = decompose_scores(mase.scores_, scaled=scaled, align=align)
    X1 = U @ Z1
    Y1 = V @ W1
    X2 = U @ Z2
    Y2 = V @ W2
    return X1, Y1, X2, Y2


def compute_mase_tstats(adjacency1, adjacency2, n_components):
    mase = MultipleASE(n_components=n_components, scaled=True, diag_aug=True)
    mase.fit([adjacency1, adjacency2])
    X1, Y1, X2, Y2 = project_to_node_space(mase)
    XY1 = np.concatenate((X1, Y1), axis=0)
    XY2 = np.concatenate((X2, Y2), axis=0)
    node_diff_norm = difference_norm(XY1, XY2)
    R_diff_norm = difference_norm(mase.scores_[0], mase.scores_[1])
    return node_diff_norm, R_diff_norm


def compute_tstats(
    adjacency1, adjacency2, n_components, initial_n_components=None, info={}
):
    rows = []
    master_row = {"n_components": n_components, **info}

    ase_tstat = compute_ase_tstat(
        adjacency1, adjacency2, n_components, initial_n_components=initial_n_components
    )
    row = master_row.copy()
    row["tstat"] = ase_tstat
    row["method"] = "ase"
    row["initial_n_components"] = initial_n_components
    rows.append(row)

    omni_tstat = compute_omni_tstat(adjacency1, adjacency2, n_components)
    row = master_row.copy()
    row["tstat"] = omni_tstat
    row["method"] = "omni"
    rows.append(row)

    mase_node_tstat, mase_R_tstat = compute_mase_tstats(
        adjacency1, adjacency2, n_components
    )
    row = master_row.copy()
    row["tstat"] = mase_node_tstat
    row["method"] = "mase-node"
    rows.append(row)
    row = master_row.copy()
    row["tstat"] = mase_R_tstat
    row["method"] = "mase-R"
    rows.append(row)

    return rows


# @jit(nopython=True)
def sample_rdpg(P, seed):
    np.random.seed(seed)
    A = np.random.binomial(1, P)
    A[np.arange(len(A)), np.arange(len(A))] = 0
    return A


def bootstrap_sample_tstats(P, n_components, seed):
    rng = np.random.default_rng(seed)
    A1 = sample_rdpg(P, rng.integers(np.iinfo(np.int32).max))
    A2 = sample_rdpg(P, rng.integers(np.iinfo(np.int32).max))
    rows = compute_tstats(A1, A2, n_components, info=dict(seed=seed, type="null"))
    return rows


def compute_null_tstats(A, n_bootstraps, n_components=2, rng=None):
    ase = AdjacencySpectralEmbed(n_components=n_components, check_lcc=False)
    X, Y = ase.fit_transform(A)
    P = p_from_latent(X, Y, rescale=False, loops=False)
    seeds = rng.integers(np.iinfo(np.int32).max, size=n_bootstraps)

    def _bootstrap(seed):
        return bootstrap_sample_tstats(P, n_components, seed)

    all_rows = Parallel(n_jobs=-2)(delayed(_bootstrap)(seed) for seed in seeds)
    rows = []
    for row_set in all_rows:
        rows += row_set
    # rows = []
    # for seed in seeds:
    #     rows += _bootstrap(seed)
    return rows


A1 = ll_adj
A2 = rr_adj

n_bootstraps = 100
verbose = 1
workers = -2
min_n_components = 2
max_n_components = 10
initial_n_components = 24  # only relevant for ASE
n_components_range = np.arange(min_n_components, max_n_components)
methods = ["ase", "omni", "mase-node", "mase-R"]

rng = np.random.default_rng(8888)

RECOMPUTE = False
if RECOMPUTE:
    preprocess = [binarize, remove_loops]
    graphs = [ll_adj, rr_adj]

    for func in preprocess:
        for i, graph in enumerate(graphs):
            graphs[i] = func(graph)

    ll_adj = graphs[0]
    rr_adj = graphs[1]

    rows = []
    for n_components in n_components_range:
        # compute the observed test statistics
        obs_rows = compute_tstats(
            A1,
            A2,
            n_components,
            initial_n_components=initial_n_components,
            info=dict(type="observed"),
        )
        rows += obs_rows

        # TODO should this have initial_n_components?
        null_rows = compute_null_tstats(
            A1, n_bootstraps, n_components=n_components, rng=rng
        )
        rows += null_rows

        # compute_p_values(obs_rows, null_rows)
        # results = pd.DataFrame(rows)
        # results.to_csv(out_dir / "semipar_results")
        # if verbose > 0:
        #     pprint.pprint(row)
        #     print()
        results = pd.DataFrame(rows)
        results.to_csv(out_dir / "semipar_tstats")

#%%
results = pd.read_csv(
    out_dir / "semipar_tstats", index_col=0, na_values="", keep_default_na=False
)
results
#%%
pvalue_rows = []
for n_components in n_components_range:
    for method in methods:
        sub_results = results[
            (results["method"] == method) & (results["n_components"] == n_components)
        ]
        obs_row = sub_results[sub_results["type"] == "observed"].iloc[0]
        obs_tstat = obs_row["tstat"]
        null = sub_results[sub_results["type"] != "observed"]
        null_tstat = null["tstat"]
        pvalue = (null_tstat > obs_tstat).mean()
        if pvalue == 0:
            pvalue = 1 / len(null)
        pvalue_rows.append(
            {"n_components": n_components, "method": method, "pvalue": pvalue}
        )
pvalues = pd.DataFrame(pvalue_rows)
pvalues.to_csv(out_dir / "semipar_pvalues")
#%%

n_components = 6
for method in methods:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    histplot(
        data=results[
            (results["method"] == method) & (results["n_components"] == n_components)
        ],
        x="tstat",
        hue="type",
        kde=True,
        ax=ax,
    )
    ax.set(title=method)


#%% [markdown]
# ## Plot p-values
#%%
ax = scatterplot(
    data=pvalues,
    x="n_components",
    y="pvalue",
    hue="method",
    shift="method",
    shade=True,
    shift_bounds=(-0.2, 0.2),
)
ax.set_yscale("log")
styles = ["--", ":"]
line_locs = [0.05, 1 / n_bootstraps]
line_kws = dict(color="black", alpha=0.7, linewidth=1.5, zorder=-1)
for loc, style in zip(line_locs, styles):
    ax.axhline(loc, linestyle=style, **line_kws)
    ax.text(ax.get_xlim()[-1] + 0.1, loc, loc, ha="left", va="center")
stashfig("semipar-pvalues-by-dimension")

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
