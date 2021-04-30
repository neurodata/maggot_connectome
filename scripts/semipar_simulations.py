#%% [markdown]
# # Semiparametric latent position test - simulations
#%%
from pkg.utils import set_warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graspologic.inference import latent_position_test
from graspologic.simulations import rdpg
from joblib import Parallel, delayed
from pkg.io import savefig
from pkg.plot import set_theme
from tqdm import tqdm

foldername = "semipar_simulations"

set_theme()


def stashfig(name, **kwargs):
    savefig(name, foldername=foldername, **kwargs)


#%% [markdown]
# ## Simulation setup
# Here we sample from a simple RDPG model:
# - $d = 1$
# - $n = 64$
# - $X_i \overset{iid}{\sim} Uniform(0.1, 0.9)$
# - $Y_i \overset{iid}{\sim} Uniform(0.1, 0.9)$, independent of $X$
# - If directed, $P = X Y^T$
# - If undirected, $P = XX^T$
# - $A_1, A_2 \overset{iid}{\sim} Bernoulli(P)$
#
#
# Using this model, we run the following simulation under the null distribution:
# - For $i$ in $n_{trials}$:
#    - $A_1, A_2 \overset{iid}{\sim} Bernoulli(P)$
#    - $p$ = $Semipar(A_1, A_2)$
#    - Append p-value $p$ to the distribution of p-values under the null
#
# The above simulation is run using $ASE$ as the embedding method/test statistic.
#%%


def run_semipar_experiment(
    seed=None,
    directed=True,
    perturbation="none",
    n_verts=64,
    methods=["ase", "omnibus"],
):
    np.random.seed(seed)
    # sample latent positions
    X1 = np.random.uniform(0.1, 0.9, size=(n_verts, 1))
    if directed:
        Y1 = np.random.uniform(0.1, 0.9, size=(n_verts, 1))
    else:
        Y1 = None

    if perturbation == "none":
        X2 = X1
        Y2 = Y1

    # sample networks
    A1 = rdpg(X1, Y1, directed=directed, loops=False)
    A2 = rdpg(X2, Y2, directed=directed, loops=False)

    # run the tests
    rows = []
    for method in methods:
        lpt_result = latent_position_test(
            A1, A2, n_components=1, workers=1, embedding=method, n_bootstraps=200
        )
        lpt_result = lpt_result._asdict()
        lpt_result["method"] = method
        lpt_result["perturbation"] = perturbation
        lpt_result["directed"] = directed
        rows.append(lpt_result)
    return rows


n_trials = 5
rng = np.random.default_rng(8888)
all_rows = []
seeds = rng.integers(np.iinfo(np.int32).max, size=n_trials)
all_rows += Parallel(n_jobs=-1, verbose=10)(
    delayed(run_semipar_experiment)(seed, False) for seed in seeds
)
all_rows += Parallel(n_jobs=-2, verbose=10)(
    delayed(run_semipar_experiment)(seed, True) for seed in seeds
)
results = []
for rows in all_rows:
    results += rows
results = pd.DataFrame(results)

#%% [markdown]
# ## Plot ECDFs of p-values under the null
#%%
for directed in [False, True]:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.ecdfplot(
        data=results[(results["directed"] == directed)],
        x="p_value",
        hue="method",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2, color="darkred")
    ax.set(
        title=f"Null, directed={directed}",
        xlabel="p-value",
        ylabel="Cumulative frequency",
    )
    stashfig(f"semipar-null-ecdf-directed={directed}")
