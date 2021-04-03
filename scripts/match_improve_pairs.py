#%% [markdown]
# # Matching when including the contralateral connections
#%% [markdown]
# ## Preliminaries
#%%
from pkg.utils import set_warnings

set_warnings()


import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import seaborn as sns
from numba import jit
from src.visualization import adjplot, matrixplot

from giskard.plot import matched_stripplot
from pkg.data import load_maggot_graph
from pkg.io import savefig
from pkg.plot import set_theme
from pkg.utils import get_paired_inds, get_paired_subgraphs


t0 = time.time()


def stashfig(name, **kwargs):
    foldername = "matching_improve_pairs"
    savefig(name, foldername=foldername, **kwargs)


set_theme()

colors = sns.color_palette("Set1")
palette = dict(zip(["Left", "Right", "Contra"], [colors[0], colors[1], colors[3]]))

#%% [markdown]
# ### Load the data
#%%
mg = load_maggot_graph()
mg = mg[mg.nodes["paper_clustered_neurons"]]
mg = mg[mg.nodes["hemisphere"].isin(["L", "R"])]
lp_inds, rp_inds = get_paired_inds(mg.nodes)

ll_mg, rr_mg, lr_mg, rl_mg = mg.bisect()

left_in_right = ll_mg.nodes["pair"].isin(rr_mg.nodes.index)
left_in_right_idx = left_in_right[left_in_right].index
right_in_left = rr_mg.nodes["pair"].isin(ll_mg.nodes.index)
right_in_left_idx = right_in_left[right_in_left].index
left_in_right_pair_ids = ll_mg.nodes.loc[left_in_right_idx, "pair_id"]
right_in_left_pair_ids = rr_mg.nodes.loc[right_in_left_idx, "pair_id"]
valid_pair_ids = np.intersect1d(left_in_right_pair_ids, right_in_left_pair_ids)
n_pairs = len(valid_pair_ids)
mg.nodes["valid_pair_id"] = False
mg.nodes.loc[mg.nodes["pair_id"].isin(valid_pair_ids), "valid_pair_id"] = True
mg.nodes.sort_values(
    ["hemisphere", "valid_pair_id", "pair_id"], inplace=True, ascending=False
)
mg.nodes["_inds"] = range(len(mg.nodes))
adj = mg.sum.adj
left_nodes = mg.nodes[mg.nodes["hemisphere"] == "L"].copy()
left_inds = left_nodes["_inds"]
right_nodes = mg.nodes[mg.nodes["hemisphere"] == "R"].copy()
right_inds = right_nodes["_inds"]

max_n_side = max(len(left_inds), len(right_inds))


def pad(A, size):
    # naive padding for now
    A_padded = np.zeros((size, size))
    rows = A.shape[0]
    cols = A.shape[1]
    A_padded[:rows, :cols] = A
    return A_padded


ll_adj = pad(adj[np.ix_(left_inds, left_inds)], max_n_side)
rr_adj = pad(adj[np.ix_(right_inds, right_inds)], max_n_side)
lr_adj = pad(adj[np.ix_(left_inds, right_inds)], max_n_side)
rl_adj = pad(adj[np.ix_(right_inds, left_inds)], max_n_side)

for i in range(max_n_side - len(left_inds)):
    left_nodes = left_nodes.append(pd.Series(name=-i, dtype="float"), ignore_index=True)

#%%
plot_kws = dict(plot_type="scattermap", sizes=(1, 1))
fig, axs = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw=dict(hspace=0, wspace=0))

ax = axs[0, 0]
adjplot(ll_adj, meta=left_nodes, ax=ax, color=palette["Left"], **plot_kws)

ax = axs[0, 1]
matrixplot(
    lr_adj,
    row_meta=left_nodes,
    col_meta=right_nodes,
    ax=ax,
    color=palette["Contra"],
    square=True,
    **plot_kws,
)

ax = axs[1, 0]
matrixplot(
    rl_adj,
    col_meta=left_nodes,
    row_meta=right_nodes,
    ax=ax,
    color=palette["Contra"],
    square=True,
    **plot_kws,
)

ax = axs[1, 1]
adjplot(rr_adj, meta=right_nodes, ax=ax, color=palette["Right"], **plot_kws)
#%% [markdown]
# ## Include the contralateral connections in graph matching
#%% [markdown]
# ### Run the graph matching experiment
#%%
np.random.seed(8888)
maxiter = 20
verbose = 1
ot = False
maximize = True
reg = np.nan  # TODO could try GOAT
thr = np.nan
tol = 1e-4
n_init = 20
n = len(ll_adj)
# construct an initialization
P0 = np.zeros((n, n))
P0[np.arange(n_pairs), np.arange(n_pairs)] = 1
P0[n_pairs:, n_pairs:] = 1 / (n - n_pairs)
P0


@jit(nopython=True)
def compute_gradient(A, B, AB, BA, P):
    return A @ P @ B.T + A.T @ P @ B + AB @ P.T @ BA.T + BA.T @ P.T @ AB


@jit(nopython=True)
def compute_step_size(A, B, AB, BA, P, Q):
    R = P - Q
    # TODO make these "smart" traces like in the scipy code, couldn't hurt
    # though I don't know how much Numba cares
    a_cross = np.trace(AB.T @ R @ BA @ R)
    b_cross = np.trace(AB.T @ R @ BA @ Q) + np.trace(AB.T @ Q @ BA @ R)
    a_intra = np.trace(A @ R @ B.T @ R.T)
    b_intra = np.trace(A @ Q @ B.T @ R.T + A @ R @ B.T @ Q.T)

    a = a_cross + a_intra
    b = b_cross + b_intra

    if a * obj_func_scalar > 0 and 0 <= -b / (2 * a) <= 1:
        alpha = -b / (2 * a)
    return alpha
    # else:
    #     alpha = np.argmin([0, (b + a) * obj_func_scalar])
    # return alpha


@jit(nopython=True)
def compute_objective_function(A, B, AB, BA, P):
    return np.trace(A @ P @ B.T @ P.T) + np.trace(AB.T @ P @ BA @ P)


rows = []
for init in range(n_init):
    if verbose > 0:
        print(f"Initialization: {init}")
    shuffle_inds = np.random.permutation(n)
    correct_perm = np.argsort(shuffle_inds)
    A_base = ll_adj.copy()
    B_base = rr_adj.copy()
    AB_base = lr_adj.copy()
    BA_base = rl_adj.copy()

    for between_term in [True]:
        init_t0 = time.time()
        if verbose > 0:
            print(f"Between term: {between_term}")
        A = A_base
        B = B_base[shuffle_inds][:, shuffle_inds]
        AB = AB_for_obj = AB_base[:, shuffle_inds]
        BA = BA_for_obj = BA_base[shuffle_inds]

        if not between_term:
            AB = np.zeros((n, n))
            BA = np.zeros((n, n))

        P = P0.copy()
        P = P[:, shuffle_inds]
        # _, iteration_perm = linear_sum_assignment(-P)
        # match_ratio = (correct_perm == iteration_perm)[:n_pairs].mean()
        # print(match_ratio)

        obj_func_scalar = 1
        if maximize:
            obj_func_scalar = -1

        for n_iter in range(1, maxiter + 1):

            # [1] Algorithm 1 Line 3 - compute the gradient of f(P)
            currtime = time.time()
            grad_fp = compute_gradient(A, B, AB, BA, P)
            if verbose > 1:
                print(f"{time.time() - currtime:.3f} seconds elapsed for grad_fp.")

            # [1] Algorithm 1 Line 4 - get direction Q by solving Eq. 8
            currtime = time.time()
            if ot:
                # TODO not implemented here yet
                Q = alap(grad_fp, n, maximize, reg, thr)
            else:
                _, cols = linear_sum_assignment(grad_fp, maximize=maximize)
                Q = np.eye(n)[cols]
            if verbose > 1:
                print(
                    f"{time.time() - currtime:.3f} seconds elapsed for LSAP/Sinkhorn step."
                )

            # [1] Algorithm 1 Line 5 - compute the step size
            currtime = time.time()

            alpha = compute_step_size(A, B, AB, BA, P, Q)

            if verbose > 1:
                print(
                    f"{time.time() - currtime:.3f} seconds elapsed for quadradic terms."
                )

            # [1] Algorithm 1 Line 6 - Update P
            P_i1 = alpha * P + (1 - alpha) * Q
            if np.linalg.norm(P - P_i1) / np.sqrt(n) < tol:
                P = P_i1
                break
            P = P_i1
            _, iteration_perm = linear_sum_assignment(-P)
            match_ratio = (correct_perm == iteration_perm)[:n_pairs].mean()

            objfunc = compute_objective_function(A, B, AB_for_obj, BA_for_obj, P)

            if verbose > 0:
                print(
                    f"Iteration: {n_iter},  Objective function: {objfunc:.2f},  Match ratio: {match_ratio:.2f}"
                )

            row = {
                "init": init,
                "iter": n_iter,
                "objfunc": objfunc,
                "match_ratio": match_ratio,
                "between_term": between_term,
                "time": time.time() - init_t0,
            }
            rows.append(row)

        if verbose > 0:
            print("\n")

    _, perm = linear_sum_assignment(-P)
    if verbose > 0:
        print("\n")

results = pd.DataFrame(rows)
results

#%% [markdown]
# ### Plot the results
#%%
last_results_idx = results.groupby(["between_term", "init"])["iter"].idxmax()
last_results = results.loc[last_results_idx].copy()

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
matched_stripplot(
    last_results,
    jitter=0.2,
    x="between_term",
    y="objfunc",
    match="init",
    hue="between_term",
)
stashfig("between-objfunc")


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
matched_stripplot(
    last_results,
    jitter=0.2,
    x="between_term",
    y="match_ratio",
    match="init",
    hue="between_term",
)
stashfig("between-match-ratio")

# %% [markdown]
# ## End
#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")