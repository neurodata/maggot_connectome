#%% [markdown]
# # Comparing approaches to feedforwardness ordering
# For evaluating feedforwardness, we have:
# - 4 networks
#    - Axo-dendritic (AD)
#    - Axo-axonic (AA)
#    - Dendro-dendritic (DD)
#    - Dendro-axonic (DA)
# - 4+ algorithms for finding an ordering
#    - Signal flow (SF)
#    - Spring rank (SR)
#    - Graph match flow (GMF)
#    - Linear regression flow (LRF)
#    - Others...
#        - SyncRank
#        - SVD based, introduced in SyncRank paper, and a few regularized follow ups
# - 1+ test statistic for feedforwardness
#    - Proportion of edges in upper triangle after sorting ($p_{upper}$)
#    - Others...
#        - Spring rank energy
#
# This notebook compares the performance of the different ordering algorithms on the
# same data, as well as the ordering for each of the 4 networks predicted by a single
# algorithm.

#%% [markdown]
# ## Different algorithms, same dataset

#%% [markdown]
# ### Plot pairsplots of the ranks from each algorithm


#%% [markdown]
# ## Different datasets, same algorithm

#%% [markdown]
# ## Plot the adjacency matrices sorted by each algorithm
