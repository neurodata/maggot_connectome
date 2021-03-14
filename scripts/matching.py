#%% [markdown]
# # Hemisphere matching
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
    foldername = "hemisphere_matching"
    savefig(name, foldername=foldername, **kwargs)


colors = sns.color_palette("Set1")
palette = dict(zip(["Left", "Right"], colors))
set_theme()

#%%
mg = load_maggot_graph()
mg = mg[mg.nodes["paper_clustered_neurons"]]
mg.fix_pairs()

#%%
ll_mg, rr_mg, lr_mg, rl_mg = mg.bisect(paired=True)
print(ll_mg)
print(rr_mg)

#%%
print(len(ll_mg.sum.adj))
print(len(rr_mg.sum.adj))

#%%
A = np.ones((1, 3, 2))
B = np.ones((3, 2, 2))
# l is the index of the layer, which remains fixed
# j is the index of summation in the matrix product
out = np.einsum("ijl,jkl->ikl", A, B)
out.shape