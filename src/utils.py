import pandas as pd
import numpy as np



def create_dedupe_df(idxs, distances):
    """
    create a dataframe of matches from the output of a knn model
    @param idxs: array of indices of matches
    @param distances: array of distances of matches
    @return: dataframe of matches
    """
    try:
        deduped_df = pd.DataFrame({
            'orig_idxs': np.arange(len(idxs)),
            'match_idxs': idxs.tolist(), 
            'distance': distances.tolist()
            }).explode(['match_idxs', 'distance'])
    except:
        deduped_df = pd.DataFrame({
            'orig_idxs': np.arange(len(idxs)),
            'match_idxs': idxs, 
            'distance': distances
            }).explode(['match_idxs', 'distance'])


    ## remove self matches
    deduped_df = deduped_df[deduped_df['orig_idxs'] < deduped_df['match_idxs']]
    return deduped_df
