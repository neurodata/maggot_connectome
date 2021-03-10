import numpy as np


def get_paired_inds(meta, check_in=True, pair_key="pair", pair_id_key="pair_id"):
    pair_meta = meta.copy()
    pair_meta["_inds"] = range(len(pair_meta))

    # remove any center neurons
    pair_meta = pair_meta[pair_meta["hemisphere"].isin(["L", "R"])]

    # remove any neurons for which the other in the pair is not in the metadata
    if check_in:
        pair_meta = pair_meta[pair_meta[pair_key].isin(pair_meta.index)]

    # remove any pairs for which there is only one neuron
    pair_group_size = pair_meta.groupby(pair_id_key).size()
    remove_pairs = pair_group_size[pair_group_size == 1].index
    pair_meta = pair_meta[~pair_meta[pair_id_key].isin(remove_pairs)]

    # make sure each pair is "valid" now
    assert pair_meta.groupby(pair_id_key).size().min() == 2
    assert pair_meta.groupby(pair_id_key).size().max() == 2

    # sort into pairs interleaved
    pair_meta.sort_values([pair_id_key, "hemisphere"], inplace=True)
    lp_inds = pair_meta[pair_meta["hemisphere"] == "L"]["_inds"]
    rp_inds = pair_meta[pair_meta["hemisphere"] == "R"]["_inds"]

    # double check that everything worked
    assert (
        meta.iloc[lp_inds][pair_id_key].values == meta.iloc[rp_inds][pair_id_key].values
    ).all()
    return lp_inds, rp_inds


def get_paired_subgraphs(adj, lp_inds, rp_inds):
    ll_adj = adj[np.ix_(lp_inds, lp_inds)]
    rr_adj = adj[np.ix_(rp_inds, rp_inds)]
    lr_adj = adj[np.ix_(lp_inds, rp_inds)]
    rl_adj = adj[np.ix_(rp_inds, lp_inds)]
    return (ll_adj, rr_adj, lr_adj, rl_adj)