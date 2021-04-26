#%% [markdown]
# # Symmetry and graph matching

#%% [markdown]
# ## Preliminaries
#%%
from pkg.utils import set_warnings

import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from graspologic.align import OrthogonalProcrustes, SeedlessProcrustes
from graspologic.embed import (
    AdjacencySpectralEmbed,
    OmnibusEmbed,
    select_dimension,
)
from graspologic.match import GraphMatch
from graspologic.plot import pairplot
from graspologic.utils import (
    augment_diagonal,
    binarize,
    multigraph_lcc_intersection,
    pass_to_ranks,
)
from pkg.data import load_maggot_graph, load_palette
from pkg.io import savefig
from pkg.plot import set_theme


from src.visualization import adjplot  # TODO fix graspologic version and replace here

t0 = time.time()


def stashfig(name, **kwargs):
    foldername = "graph_match_symmetry"
    savefig(name, foldername=foldername, **kwargs)


colors = sns.color_palette("Set1")
palette = dict(zip(["Left", "Right"], colors))
set_theme()

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
sorted_nodes = nodes.sort_values(["simple_group"])
sort_inds = sorted_nodes["_inds"]

ll_adj = ll_adj[np.ix_(sort_inds, sort_inds)]
rr_adj = rr_adj[np.ix_(sort_inds, sort_inds)]

adjs, lcc_inds = multigraph_lcc_intersection([ll_adj, rr_adj], return_inds=True)
ll_adj = adjs[0]
rr_adj = adjs[1]
print(f"{len(lcc_inds)} in intersection of largest connected components.")

#%% [markdown]
# ## graph matching distance

from graspologic.utils import remove_loops, binarize

ll_adj = remove_loops(ll_adj)
rr_adj = remove_loops(rr_adj)
ll_adj = binarize(ll_adj)
rr_adj = binarize(rr_adj)

#%%

from graspologic.match import GraphMatch
import pandas as pd

n = len(ll_adj)
gm = GraphMatch()
perm_inds = gm.fit_predict(ll_adj, rr_adj)


def graph_match_distance(A, B, perm_inds=None):
    if perm_inds is None:
        perm_inds = np.random.permutation(len(A))
    diff = A - B[np.ix_(perm_inds, perm_inds)]
    dist = np.linalg.norm(diff, ord="fro")
    normalizer = len(A) ** 2
    # normalizer = np.linalg.norm(A) + np.linalg.norm(B)
    dist /= normalizer
    return dist


rows = []
observed_dist = graph_match_distance(ll_adj, rr_adj, perm_inds)
rows.append({"dist": observed_dist, "type": "observed"})
n_random = 10
for i in range(n_random):
    dist = graph_match_distance(ll_adj, rr_adj)
    rows.append({"dist": dist, "type": "random"})
results = pd.DataFrame(rows)

from giskard.plot import histplot

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
histplot(results, x="dist", hue="type", ax=ax)

#%%
from graspologic.embed import AdjacencySpectralEmbed
from graspologic.simulations import sample_edges
from tqdm import tqdm

rows = []
A_L = ll_adj
A_R = rr_adj
perm_inds = gm.fit_predict(A_L, A_R)
observed_dist = graph_match_distance(A_L, A_R, perm_inds)
observed_mean_norm = 0.5 * np.linalg.norm(A_L, ord="fro") + 0.5 * np.linalg.norm(
    A_R, ord="fro"
)
rows.append({"dist": observed_dist, "type": r"$A$", "mean_norm": observed_mean_norm})

n_components_range = np.geomspace(1, n, 10)
max_n_components = int(n_components_range[-1])
ase = AdjacencySpectralEmbed(
    n_components=max_n_components, check_lcc=False, algorithm="full"
)
X_L, Y_L = ase.fit_transform(ll_adj)
X_R, Y_R = ase.fit_transform(rr_adj)
n_samples = 1
# np.arange(1, max_n_components + 1)

for n_components in tqdm(n_components_range):
    n_components = int(n_components)

    P_L = X_L[:, :n_components] @ Y_L[:, :n_components].T
    P_R = X_R[:, :n_components] @ Y_R[:, :n_components].T
    # perm_inds = gm.fit_predict(P_L, P_R)
    # dist = graph_match_distance(P_L, P_R, perm_inds=perm_inds)
    # row = {"dist": dist, "type": r"$\hat{P}$", "n_components": n_components}
    # rows.append(row)
    P_L[P_L < 0] = 0
    P_L[P_L > 1] = 1
    P_R[P_R < 0] = 0
    P_R[P_R > 1] = 1
    for i in range(n_samples):
        A_L_sample = sample_edges(P_L, directed=True)
        A_R_sample = sample_edges(P_R, directed=True)
        mean_norm = 0.5 * np.linalg.norm(A_L_sample, ord="fro") + 0.5 * np.linalg.norm(
            A_R_sample, ord="fro"
        )
        perm_inds = gm.fit_predict(A_L_sample, A_R_sample)
        dist = graph_match_distance(P_L, P_R, perm_inds=perm_inds)
        row = {
            "dist": dist,
            "type": r"$\widetilde{A}$",
            "n_components": n_components,
            "sample": i,
            "mean_norm": mean_norm,
        }
        rows.append(row)

results = pd.DataFrame(rows)
results
#%%

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.lineplot(data=results, x="n_components", y="dist")
ax.axhline(observed_dist)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.lineplot(data=results, x="n_components", y="mean_norm")
ax.axhline(observed_mean_norm)
# rows = []
# observed_dist = graph_match_distance(ll_adj, rr_adj, perm_inds)
# rows.append({"dist": observed_dist, "type": "observed"})
# n_random = 10
# for i in range(n_random):
#     dist = graph_match_distance(ll_adj, rr_adj)
#     rows.append({"dist": dist, "type": "random"})
# results = pd.DataFrame(rows)

# from giskard.plot import histplot

# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# histplot(results, x="dist", hue="type", ax=ax)