#%%
from pkg.data import load_node_meta, load_adjacency, load_palette


meta = load_node_meta()
adj = load_adjacency("G", nodelist=meta.index)

from graspologic.plot import adjplot

palette = load_palette()

adjplot(
    adj,
    meta=meta,
    plot_type="scattermap",
    group="simple_group",
    ticks=False,
    color="simple_group",
    palette=palette,
    sizes=(1, 2),
)

adjplot(
    adj,
    meta=meta,
    plot_type="scattermap",
    group=["hemisphere", "simple_group"],
    ticks=False,
    color="simple_group",
    palette=palette,
    sizes=(1, 2),
)