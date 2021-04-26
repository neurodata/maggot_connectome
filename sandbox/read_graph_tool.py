#%%
import matplotlib.pyplot as plt
import pandas as pd

from giskard.plot import crosstabplot
from pkg.data import load_maggot_graph
from pkg.flow import signal_flow
from src.visualization import CLASS_COLOR_DICT as palette

mg = load_maggot_graph()
labels = pd.read_csv("maggot_connectome/sandbox/labels.csv", index_col=0, dtype="Int64")
mg.nodes["gt_labels"] = labels
adj = mg.sum.adj
mg.nodes["sf"] = signal_flow(adj)

#%%
fig, ax = plt.subplots(1, 1, figsize=(16, 8))
crosstabplot(
    mg.nodes,
    group="gt_labels",
    group_order="sf",
    hue="merge_class",
    palette=palette,
    ax=ax,
)
