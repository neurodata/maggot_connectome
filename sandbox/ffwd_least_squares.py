#%% [markdown]
from pkg.utils import set_warnings

import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed

from graspologic.models import DCEREstimator, EREstimator
from graspologic.plot import heatmap
from graspologic.simulations import sample_edges, sbm
from graspologic.utils import binarize, largest_connected_component, remove_loops
from pkg.data import load_maggot_graph
from pkg.flow import calculate_p_upper, rank_graph_match_flow
from pkg.io import savefig
from pathlib import Path
from pkg.plot import set_theme

#%%
# ## Preliminaries

#%%
set_warnings()

np.random.seed(8888)

t0 = time.time()

out_path = Path("maggot_connectome/results/outputs/feedforwardness_data")


def stashfig(name, **kwargs):
    foldername = "ffwd_least_squares"
    savefig(name, foldername=foldername, **kwargs)


colors = sns.color_palette("Set1")
set_theme()

#%% [markdown]
# ### Load the data

#%%

mg = load_maggot_graph()
mg = mg[mg.nodes["paper_clustered_neurons"]]
# mg = mg[mg.nodes["left"]]
# mg = mg[mg.nodes["class1"] == "KC"]

#%%
import networkx as nx

adj = mg.sum.adj.copy()
adj = remove_loops(adj)
adj = binarize(adj)
g = nx.from_numpy_array(adj, create_using=nx.DiGraph)
nodelist = sorted(g.nodes)
incidence = nx.incidence_matrix(g, nodelist=nodelist, oriented=True).T
weights = np.ones(incidence.shape[0])
from sklearn.linear_model import LinearRegression

lr = LinearRegression(fit_intercept=False, n_jobs=-1)

lr.fit(incidence, weights)


lr_score = lr.coef_
mg.nodes["lr_score"] = lr_score
mg.nodes.sort_values("lr_score", inplace=True)
from src.visualization import adjplot, CLASS_COLOR_DICT

adjplot(
    mg.sum.adj,
    meta=mg.nodes,
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    item_order="lr_score",
    plot_type="scattermap",
    sizes=(2, 2),
)

#%%

sns.histplot(lr_score)

#%%

adj = mg.sum.adj.copy()
adj = remove_loops(adj)
H = adj - adj.T

# is it indeed skew symmetric?
print((H == -H.T).all())

from graspologic.embed import selectSVD

U, S, V = selectSVD(H, n_components=2)

#%%
n = len(adj)
e = np.full(n, 1 / np.sqrt(n))
P = U @ U.T
# project ones vector onto span of the first two components
e_proj = P @ e
e_proj = e_proj / np.linalg.norm(e_proj)
# need to find the intersection of the orthogonal complement of e_proj and the span of U
P_e_proj = np.outer(e_proj, e_proj)
# get random vector in the span of U
u_rand = P @ np.random.normal(size=n)
# get vector orthogonal to e_proj
svd_score = (np.eye(n) - P_e_proj) @ u_rand

svd_argsort = np.argsort(svd_score)
triu_inds = np.triu_indices(n, k=1)
if adj[np.ix_(svd_argsort, svd_argsort)][triu_inds].sum() / adj.sum() < 0.5:
    svd_score = -svd_score

#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(x=U[:, 0], y=U[:, 1], ax=ax)
#%%
svd_score = (np.eye(n) - P_e_proj) @ u_rand

#%%
def projplot(x, ax=None, color=None):
    x1_projection = U[:, 0].T @ x
    x2_projection = U[:, 1].T @ x
    ax.plot([0, x1_projection], [0, x2_projection], color=color)
    # sns.scatterplot(x=x1_projection, y=x2_projection, ax=ax)
    return ax


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
projplot(u_rand, ax=ax)
projplot(e_proj, ax=ax, color="orange")
projplot(svd_score, ax=ax, color="green")
ax.set(xlim=(-1, 1), ylim=(-1, 1))

#%%
sns.histplot(svd_score)

#%%
mg.nodes["svd_score"] = svd_score
adjplot(
    mg.sum.adj,
    meta=mg.nodes,
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    item_order="svd_score",
    plot_type="scattermap",
    sizes=(2, 2),
)

#%%
adjplot(
    H,
    item_order=svd_score,
    plot_type="scattermap",
    sizes=(2, 2),
)
#%%
adjplot(
    H[:100, :100],
)

#%%
svd_score.T @ e_proj