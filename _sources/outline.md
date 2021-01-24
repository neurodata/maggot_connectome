# Outline

## Figure Panels
-  `[BDP]` "show the data"
    - What do the neurons look like? Plot the neurons in space, maybe show a few examples, etc.
    - What does the graph look like? Can plot adjacencies as well as some kind of graph layout possibly.
    - `[NDD?]` Some simple descripive statistics (# nodes, # edges, # synapses, degrees, weights, etc.)
        - Table: number of nodes, number of edges, number of synapses
        - Panel of edge weight distribution
        - Panel of in vs out degree with marginals
    - (Maybe) something describing the different edge types
- `[TL?/NDD?]` A priori modeling: using prior knowledge like cell type labels, left right hemisphere, etc. Testing hypotheses about these models 
    - Left/right hemisphere blockmodel  
        - testing for homophillic/assortative, the different SBM block probability hypotheses
        - Could do the above with the different 4 color graphs as well
- `[TL?]` A posteriori modeling
    - Hierarchical SBM estimation
    - Leiden hierarchical SBM estimation, how are these different/not different (do we want this?)
    - Comparison of model complexity (dDCSBM, SBM, RDPG-d, etc.)
- `[ASE]` Graph matching methods figure (I don't think these results from Youngser/Carey ever went to a paper anywhere? so I assume CEP would be okay with them here? And we should be able to replicate/improve in python now.)
    - `[BDP]` Show some examples of pairs in space.
    - `[ASE/YP?/CEP?]` Show results of vanilla GM, GM with some notion of similarity (maybe NBLAST and or spec sim?), GM with multigraph, GM with multigraph + similarity.
    - `[ASE]` (Maybe) Seeded graph matching with the known pairs as seeds? I actually use this in my work... so it is useful.
    - Some interesting inference about pairedness? Would be nice to show what this can be used for, or demonstrate which parts are more/less bilaterally similar?
- `[BDP/ASE/TL/NDD?]` Bilateral symmetry/testing
    - How similar are the SBM models? 
    - How similar are the RDPG models (nonpar/semipar)? 
    - Can we say anything about the correlation under these different models?
    - Testing homotopic affinity (by edge type)
- `[BDP/NDD-code?]` Feedforwardness: describing (and hopefully modeling) a feedforward pathway through the network, expanding to include multinetwork models. 
    - Some description of the chain predicted by signal flow or cascades or graph match flow etc.
    - Comparisons of the flows for different network types (e.g. AA, AD, etc.) 
    - Testing for feedforwardness with spring rank model
- `[BDP]` Directedness: testing for whether the graph or specific parts of it are meaningfully directed. (Do we know how to do this?)


## Code
- `[NDD]` [Flow/hierarchy/ranking](https://github.com/microsoft/graspologic/issues/636) into graspologic
- `[TL?]` Improve the estimation code to make it easier to fit to data in a useful way, examine the models, etc. (as necessary)
- `[NDD]` Adjacency with dendrogram for hierarchical clustering that is not complete
    - I have code, not pretty, probably not generalizable yet
- `[TL/NDD?]` [Tests from statistical connectomics](https://github.com/microsoft/graspologic/issues/570) into graspologic
    - We should talk to Eric/decide what we actually want first
- `[TL/NDD?]` Bar dendrogram plotting in general