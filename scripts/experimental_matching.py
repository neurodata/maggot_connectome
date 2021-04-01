#%% [markdown]
# # Experimental matching
#%% [markdown]
# ## Preliminaries
#%%
from pkg.utils import set_warnings

set_warnings()

import datetime
import pprint
import time


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
from pkg.utils import get_paired_inds, get_paired_subgraphs

from src.visualization import adjplot  # TODO fix graspologic version and replace here

t0 = time.time()


def stashfig(name, **kwargs):
    foldername = "experimental_matching"
    savefig(name, foldername=foldername, **kwargs)


colors = sns.color_palette("Set1")
palette = dict(zip(["Left", "Right"], colors))
set_theme()


#%%

import operator

import numpy as np
from scipy._lib._util import check_random_state
from scipy.optimize import OptimizeResult, linear_sum_assignment


def _check_init_input(P0, n):
    row_sum = np.sum(P0, axis=0)
    col_sum = np.sum(P0, axis=1)
    tol = 1e-3
    msg = None
    if P0.shape != (n, n):
        msg = "`P0` matrix must have shape m' x m', where m'=n-m"
    elif (
        (~np.isclose(row_sum, 1, atol=tol)).any()
        or (~np.isclose(col_sum, 1, atol=tol)).any()
        or (P0 < 0).any()
    ):
        msg = "`P0` matrix must be doubly stochastic"
    if msg is not None:
        raise ValueError(msg)


def _split_matrix(X, n):
    # definitions according to Seeded Graph Matching [2].
    upper, lower = X[:n], X[n:]
    return upper[:, :n], upper[:, n:], lower[:, :n], lower[:, n:]


def _doubly_stochastic(P, tol=1e-3, max_iter=1000):
    # Adapted from @btaba implementation
    # https://github.com/btaba/sinkhorn_knopp
    # of Sinkhorn-Knopp algorithm
    # https://projecteuclid.org/euclid.pjm/1102992505

    c = 1 / P.sum(axis=0)
    r = 1 / (P @ c)
    P_eps = P

    for it in range(max_iter):
        if it % 100 == 0:  # only check every so often to speed up
            if (np.abs(P_eps.sum(axis=1) - 1) < tol).all() and (
                np.abs(P_eps.sum(axis=0) - 1) < tol
            ).all():
                # All column/row sums ~= 1 within threshold
                break

        c = 1 / (r @ P)
        r = 1 / (P @ c)
        P_eps = r[:, None] * P * c
    return P_eps


def quadratic_assignment_layered(A, B, method="faq", options=None):
    if options is None:
        options = {}

    method = method.lower()
    methods = {"faq": _quadratic_assignment_faq_ot}
    if method not in methods:
        raise ValueError(f"method {method} must be in {methods}.")
    res = methods[method](A, B, **options)
    return res


def _calc_score(A, B, perm):
    # equivalent to objective function but avoids matmul
    return np.sum(A * B[perm][:, perm])


def _common_input_validation(A, B, partial_match):
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)

    if partial_match is None:
        partial_match = np.array([[], []]).T
    partial_match = np.atleast_2d(partial_match).astype(int)

    msg = None
    if A.shape[0] != A.shape[1]:
        msg = "`A` must be square"
    elif B.shape[0] != B.shape[1]:
        msg = "`B` must be square"
    elif A.shape != B.shape:
        msg = "`A` and `B` matrices must be of equal size"
    elif partial_match.shape[0] > A.shape[0]:
        msg = "`partial_match` can have only as many seeds as there are nodes"
    elif partial_match.shape[1] != 2:
        msg = "`partial_match` must have two columns"
    elif partial_match.ndim != 2:
        msg = "`partial_match` must have exactly two dimensions"
    elif (partial_match < 0).any():
        msg = "`partial_match` must contain only positive indices"
    elif (partial_match >= len(A)).any():
        msg = "`partial_match` entries must be less than number of nodes"
    elif not len(set(partial_match[:, 0])) == len(partial_match[:, 0]) or not len(
        set(partial_match[:, 1])
    ) == len(partial_match[:, 1]):
        msg = "`partial_match` column entries must be unique"

    if msg is not None:
        raise ValueError(msg)

    return A, B, partial_match


import functools


def _layered_product(*args):
    # l is the index of the layer, which remains fixed
    # j is the index of summation in the matrix product
    # i is the row index of A
    # k is the col index of b
    return functools.reduce(
        lambda A, B: np.einsum("ijl,jkl->ikl", A, B, optimize=True), args
    )


@jit(nopython=True)
def _single_layered_product(A, B):
    # n_layers = max(A.shape[-1], B.shape[-1])
    if len(A.shape) == 3:
        n_layers = A.shape[-1]
    elif len(B.shape) == 3:
        n_layers = B.shape[-1]
    else:
        n_layers = 1
    output = np.empty((A.shape[0], B.shape[1], n_layers))
    if A.ndim == 2:
        for layer in range(n_layers):
            output[..., layer] = A @ B[..., layer]
    elif B.ndim == 2:
        for layer in range(n_layers):
            output[..., layer] = A[..., layer] @ B
    else:
        for layer in range(n_layers):
            output[..., layer] = A[..., layer] @ B[..., layer]
    return output


def _layered_product(*args):
    return functools.reduce(_single_layered_product, args)


def _transpose(A):
    return np.transpose(A, axes=(1, 0, 2))


def _quadratic_assignment_faq_ot(
    A,
    B,
    maximize=False,
    partial_match=None,
    rng=None,
    P0="barycenter",
    shuffle_input=False,
    maxiter=30,
    tol=0.03,
    reg=100,
    thr=5e-2,
    ot=False,
    verbose=False,
):

    maxiter = operator.index(maxiter)

    rng = check_random_state(rng)

    A, B, partial_match = _common_input_validation(A, B, partial_match)

    n = A.shape[0]  # number of vertices in graphs
    n_seeds = partial_match.shape[0]  # number of seeds
    n_unseed = n - n_seeds
    n_layers = A.shape[-1]

    obj_func_scalar = 1
    if maximize:
        obj_func_scalar = -1

    nonseed_B = np.setdiff1d(range(n), partial_match[:, 1])
    if shuffle_input:
        nonseed_B = rng.permutation(nonseed_B)
        # shuffle_input to avoid results from inputs that were already matched

    nonseed_A = np.setdiff1d(range(n), partial_match[:, 0])
    perm_A = np.concatenate([partial_match[:, 0], nonseed_A])
    perm_B = np.concatenate([partial_match[:, 1], nonseed_B])

    # definitions according to Seeded Graph Matching [2].
    A11, A12, A21, A22 = _split_matrix(A[perm_A][:, perm_A], n_seeds)
    B11, B12, B21, B22 = _split_matrix(B[perm_B][:, perm_B], n_seeds)
    # TODO also split contralaterals

    # [1] Algorithm 1 Line 1 - choose initialization
    if isinstance(P0, str):
        # initialize J, a doubly stochastic barycenter
        J = np.ones((n_unseed, n_unseed)) / n_unseed
        if P0 == "barycenter":
            P = J
        elif P0 == "randomized":
            # generate a nxn matrix where each entry is a random number [0, 1]
            # would use rand, but Generators don't have it
            # would use random, but old mtrand.RandomStates don't have it
            K = rng.uniform(size=(n_unseed, n_unseed))
            # Sinkhorn balancing
            K = _doubly_stochastic(K)
            P = J * 0.5 + K * 0.5
    else:
        P0 = np.atleast_2d(P0)
        _check_init_input(P0, n_unseed)
        P = P0

    currtime = time.time()
    const_sum = _layered_product(A21, _transpose(B21)) + _layered_product(
        _transpose(A12), B12
    )
    if verbose:
        print(f"{time.time() - currtime:.3f} seconds elapsed for const_sum.")

    # [1] Algorithm 1 Line 2 - loop while stopping criteria not met
    for n_iter in range(1, maxiter + 1):
        # [1] Algorithm 1 Line 3 - compute the gradient of f(P) = -tr(APB^tP^t)
        # TODO einsum

        currtime = time.time()
        # P = np.repeat(P, n_layers, axis=2)
        # grad_fp = (
        #     const_sum
        #     + _layered_product(A22, P, _transpose(B22))
        #     + _layered_product(_transpose(A22), P, B22)
        # )
        grad_fp = const_sum
        grad_fp += _layered_product(_layered_product(A22, P), _transpose(B22))
        grad_fp += _layered_product(_layered_product(_transpose(A22), P), B22)
        grad_fp = grad_fp.sum(axis=-1)
        if verbose:
            print(f"{time.time() - currtime:.3f} seconds elapsed for grad_fp.")
        # [1] Algorithm 1 Line 4 - get direction Q by solving Eq. 8
        currtime = time.time()
        if ot:
            Q = alap(grad_fp, n_unseed, maximize, reg, thr)
        else:
            _, cols = linear_sum_assignment(grad_fp, maximize=maximize)
            Q = np.eye(n_unseed)[cols]
        if verbose:
            print(f"{time.time() - currtime:.3f} seconds elapsed for LAP(ish) step.")
        #         Q = np.eye(n_unseed)[cols]

        # [1] Algorithm 1 Line 5 - compute the step size
        # Noting that e.g. trace(Ax) = trace(A)*x, expand and re-collect
        # terms as ax**2 + bx + c. c does not affect location of minimum
        # and can be ignored. Also, note that trace(A@B) = (A.T*B).sum();
        # apply where possible for efficiency.
        # TODO all einsums?
        currtime = time.time()
        R = P - Q
        b21 = (_layered_product(R.T, A21) * B21).sum()
        b12 = (_layered_product(R.T, _transpose(A12)) * _transpose(B12)).sum()
        AR22 = _layered_product(_transpose(A22), R)
        BR22 = _layered_product(B22, R.T)
        b22a = (AR22 * (_layered_product(Q, _transpose(B22)))).sum()
        b22b = (A22 * _layered_product(Q, BR22)).sum()
        a = (_transpose(AR22) * BR22).sum()
        b = b21 + b12 + b22a + b22b
        if verbose:
            print(f"{time.time() - currtime:.3f} seconds elapsed for quadradic terms.")
        # critical point of ax^2 + bx + c is at x = -d/(2*e)
        # if a * obj_func_scalar > 0, it is a minimum
        # if minimum is not in [0, 1], only endpoints need to be considered
        if a * obj_func_scalar > 0 and 0 <= -b / (2 * a) <= 1:
            alpha = -b / (2 * a)
        else:
            alpha = np.argmin([0, (b + a) * obj_func_scalar])

        # [1] Algorithm 1 Line 6 - Update P
        P_i1 = alpha * P + (1 - alpha) * Q
        if np.linalg.norm(P - P_i1) / np.sqrt(n_unseed) < tol:
            P = P_i1
            break
        P = P_i1
    # [1] Algorithm 1 Line 7 - end main loop

    # [1] Algorithm 1 Line 8 - project onto the set of permutation matrices
    #     print(P)
    _, col = linear_sum_assignment(-P)
    perm = np.concatenate((np.arange(n_seeds), col + n_seeds))

    unshuffled_perm = np.zeros(n, dtype=int)
    unshuffled_perm[perm_A] = perm_B[perm]

    score = _calc_score(A, B, unshuffled_perm)

    res = {"col_ind": unshuffled_perm, "fun": score, "nit": n_iter}

    return OptimizeResult(res)


# from numba import jit
# @jit(nopython=True)
def alap(P, n, maximize, reg, tol):
    power = 1 if maximize else -1
    lamb = reg / np.max(np.abs(P))
    P = np.exp(lamb * power * P)

    #     ones = np.ones(n)
    #     P_eps = sinkhorn(ones, ones, P, power/lamb, stopInnerThr=5e-02) # * (P > np.log(1/n)/lamb)

    P_eps = _doubly_stochastic(P, tol)

    return P_eps


from graspologic.simulations import er_corr


#%%

n = 20
p = 0.4
rs = [0.6, 0.7, 0.8, 0.9, 1.0]
n_sims = 20
options = dict(maximize=True, shuffle_input=True)
rows = []

arange = np.arange(n)


def compute_match_ratio(perm_inds):
    return (perm_inds == arange).mean()


for r in rs:
    for i in range(n_sims):
        A1, B1 = er_corr(n, p, r)
        A2, B2 = er_corr(n, p, r)
        A = np.stack((A1, A2), axis=2)
        B = np.stack((B1, B2), axis=2)

        layer_res = quadratic_assignment_layered(A, B, options=options)
        layer_res["method"] = "multilayer"
        layer_res["rho"] = r
        layer_res["match_ratio"] = compute_match_ratio(layer_res["col_ind"])
        rows.append(layer_res)
        # layer_perm_inds = layer_res["col_ind"]

        A_sum = A.sum(axis=-1).reshape((n, n, 1))
        B_sum = B.sum(axis=-1).reshape((n, n, 1))

        flat_res = quadratic_assignment(A_sum, B_sum, options=options)
        flat_res["method"] = "flat"
        flat_res["rho"] = r
        flat_res["match_ratio"] = compute_match_ratio(flat_res["col_ind"])
        rows.append(flat_res)


results = pd.DataFrame(rows)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

sns.lineplot(data=results, x="rho", y="match_ratio", hue="method")
stashfig("multilayer-er")

#%%

mg = load_maggot_graph()
mg = mg[mg.nodes["paper_clustered_neurons"]]

ll_mg, rr_mg, lr_mg, rl_mg = mg.bisect(paired=True)

edge_types = ["aa", "ad", "da", "dd"]
ll_adjs = []
rr_adjs = []
for edge_type in edge_types:
    ll_adjs.append(ll_mg.to_edge_type_graph(edge_type).adj)
    rr_adjs.append(rr_mg.to_edge_type_graph(edge_type).adj)
ll_adjs = np.stack(ll_adjs, axis=2)
rr_adjs = np.stack(rr_adjs, axis=2)
print(ll_adjs.shape)

ll_adjs_sum = ll_adjs.sum(axis=-1)
rr_adjs_sum = rr_adjs.sum(axis=-1)

#%%

from scipy.optimize import quadratic_assignment
from tqdm import tqdm

n = len(ll_adjs)
correct_perm = np.arange(n)

layered_options = dict(
    maximize=True, shuffle_input=False, ot=False, maxiter=20, verbose=False, tol=1e-4
)
vanilla_options = dict(maximize=True, shuffle_input=False, maxiter=20, tol=1e-4)

rows = []
n_init = 5
for i in tqdm(range(n_init)):
    shuffle_inds = np.random.permutation(n)
    correct_perm = np.argsort(shuffle_inds)
    currtime = time.time()
    res = quadratic_assignment_layered(
        ll_adjs, rr_adjs[shuffle_inds][:, shuffle_inds], options=layered_options
    )
    perm_inds = res["col_ind"]
    res["match_ratio"] = (perm_inds == correct_perm).mean()
    res["method"] = "layered"
    res["time"] = time.time() - currtime
    rows.append(res)

    currtime = time.time()
    res = quadratic_assignment(
        ll_adjs_sum, rr_adjs_sum[shuffle_inds][:, shuffle_inds], options=vanilla_options
    )
    perm_inds = res["col_ind"]
    res["match_ratio"] = (perm_inds == correct_perm).mean()
    res["method"] = "vanilla"
    res["time"] = time.time() - currtime
    rows.append(res)
results = pd.DataFrame(rows)
results

#%%
from numba import jit


def einsum_layered_product(A, B):
    return np.einsum("ijl,jkl->ikl", A, B)


@jit(nopython=True)
def forloop_layered_product(A, B):
    n_layers = B.shape[-1]
    output = np.empty((A.shape[0], B.shape[1], n_layers))
    for layer in range(n_layers):
        output[:, :, layer] = A[:, :, layer] @ B[:, :, layer]
    return output


def stack_layered_product(A, B):
    outs = []
    for layer in range(A.shape[-1]):
        outs.append(A[:, :, layer] @ B[:, :, layer])
    return np.stack(outs, axis=2)


for i in range(10):
    currtime = time.time()
    einsum_layered_product(ll_adjs, _transpose(rr_adjs))
    print(f"{time.time() - currtime:.3f} seconds elapsed for einsum layered product.")

    currtime = time.time()
    forloop_layered_product(ll_adjs, _transpose(rr_adjs))
    print(f"{time.time() - currtime:.3f} seconds elapsed for forloop layered product.")

    currtime = time.time()
    stack_layered_product(ll_adjs, _transpose(rr_adjs))
    print(f"{time.time() - currtime:.3f} seconds elapsed for stack layered product.")

    currtime = time.time()
    ll_adjs_sum @ rr_adjs_sum.T
    print(f"{time.time() - currtime:.3f} seconds elapsed for flat product.")
    print()