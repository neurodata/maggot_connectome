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

