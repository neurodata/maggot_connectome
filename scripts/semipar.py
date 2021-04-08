#%% [markdown]
# # Testing bilateral symmetry
# This notebook describes a first pass at testing for some notion of bilateral symmetry.
# Here, we focus only on the left-left and right-right induced subgraphs for simplicity.
# We also use the unweighted version of the maggot connectomes. Also, for now, we
# restrict ourselves to the set of neurons for which we know a pairing between neuron A
# on the left hemisphere and neuron A on the right hemisphere.
#
# To summarize, this notebook presents a strange phenomena where depending on the
# dimension of the embedding that we use to test bilateral symmetry, we get vastly
# different results.
#
# We also present a modified proceedure that fails to reject the null that the latent
# positions of the left-left induced subgraph and the right-right induced subgraph have
# the same distribution over many embedding dimensions, suggesting that for the current
# setup we fail to reject bilateral symmetry.
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
from hyppo.ksample import KSample
from scipy.stats import epps_singleton_2samp, ks_2samp


from graspologic.align import OrthogonalProcrustes, SeedlessProcrustes
from graspologic.embed import AdjacencySpectralEmbed, select_dimension
from graspologic.plot import pairplot
from graspologic.utils import (
    augment_diagonal,
    binarize,
    multigraph_lcc_intersection,
    pass_to_ranks,
)
from pkg.data import load_adjacency, load_node_meta
from pkg.io import savefig
from pkg.plot import set_theme
from pkg.utils import get_paired_inds, get_paired_subgraphs

from src.visualization import adjplot  # TODO fix graspologic version and replace here

t0 = time.time()


def stashfig(name, **kwargs):
    foldername = "bilateral_symmetry"
    savefig(name, foldername=foldername, **kwargs)


colors = sns.color_palette("Set1")
palette = dict(zip(["Left", "Right", "OP", "O-SP"], colors))
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

# TODO not sure what we wanna do about LCCs here
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

#%% [markdown]
# ### Plotting the aligned adjacency matrices
# At a high level, we see that the left-left and right-right induced subgraphs look
# quite similar when aligned by the known neuron pairs.
#%%
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
adjplot(
    ll_adj,
    plot_type="scattermap",
    sizes=(1, 2),
    ax=axs[0],
    title=r"Left $\to$ left",
    color=palette["Left"],
)
adjplot(
    rr_adj,
    plot_type="scattermap",
    sizes=(1, 2),
    ax=axs[1],
    title=r"Right $\to$ right",
    color=palette["Right"],
)
stashfig("left-right-induced-adjs")

#%% [markdown]
# ## Embedding the graphs
# Here I embed the unweighted, directed graphs using ASE.
#%%


def plot_latents(left, right, title="", n_show=4):
    plot_data = np.concatenate([left, right], axis=0)
    labels = np.array(["Left"] * len(left) + ["Right"] * len(right))
    pg = pairplot(plot_data[:, :n_show], labels=labels, title=title)
    return pg


def screeplot(sing_vals, elbow_inds, color=None, ax=None, label=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))
    plt.plot(range(1, len(sing_vals) + 1), sing_vals, color=color, label=label)
    plt.scatter(
        elbow_inds, sing_vals[elbow_inds - 1], marker="x", s=50, zorder=10, color=color
    )
    ax.set(ylabel="Singular value", xlabel="Index")
    return ax


def embed(adj, n_components=40, ptr=False):
    if ptr:
        adj = pass_to_ranks(adj)
    elbow_inds, elbow_vals = select_dimension(augment_diagonal(adj), n_elbows=4)
    elbow_inds = np.array(elbow_inds)
    ase = AdjacencySpectralEmbed(n_components=n_components)
    out_latent, in_latent = ase.fit_transform(adj)
    return out_latent, in_latent, ase.singular_values_, elbow_inds


#%% [markdown]
# ### Run the embedding
#%%

from graspologic.utils import symmetrize
from graspologic.inference import latent_position_test

n_components = 12
max_n_components = 40
preprocess = [symmetrize, binarize]
graphs = [ll_adj, rr_adj]

for func in preprocess:
    for i, graph in enumerate(graphs):
        graphs[i] = func(graph)

ll_adj = graphs[0]
rr_adj = graphs[1]
test_case = "scalar-rotation"
embedding = "ase"
currtime = time.time()
pvalue, tstat, misc = latent_position_test(
    ll_adj,
    rr_adj,
    embedding=embedding,
    n_components=n_components,
    test_case=test_case,
    n_bootstraps=500,
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

print(f"test case: {test_case}")
print(f"embedding: {embedding}")
print(f"n_components: {n_components}")
print(f"p-value: {pvalue}")
print(f"tstat: {tstat}")

#%%
test_case = "rotation"
embedding = "omnibus"
n_components = 8
n_bootstraps = 100
n_repeats = 2
rows = []
for n_shuffle in [2, 4, 8, 16]:
    for repeat in range(n_repeats):
        inds = np.arange(len(rr_adj))
        choice_inds = np.random.choice(len(rr_adj), size=n_shuffle, replace=False)
        shuffle_inds = choice_inds.copy()
        np.random.shuffle(shuffle_inds)
        inds[choice_inds] = inds[shuffle_inds]
        rr_adj_shuffle = rr_adj[np.ix_(inds, inds)]
        currtime = time.time()
        pvalue, tstat, misc = latent_position_test(
            ll_adj,
            rr_adj_shuffle,
            embedding=embedding,
            n_components=n_components,
            test_case=test_case,
            n_bootstraps=n_bootstraps,
        )
        row = {
            "pvalue": pvalue,
            "tstat": tstat,
            "n_shuffle": n_shuffle,
            "n_components": n_components,
            "n_bootstraps": n_bootstraps,
            "embedding": embedding,
            "test_case": test_case,
            "repeat": repeat,
        }
        rows.append(row)
        print(f"{time.time() - currtime:.3f} seconds elapsed.")
        print(f"n_shuffle: {n_shuffle}")
        print(f"test case: {test_case}")
        print(f"embedding: {embedding}")
        print(f"n_components: {n_components}")
        print(f"p-value: {pvalue}")
        print(f"tstat: {tstat}")
        print()