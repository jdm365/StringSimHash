import pandas as pd
import numpy as np



def create_dedupe_df(idxs, distances):
    """
    Create a dataframe of matches from the output of a knn model
    @param idxs: array of indices of matches
    @param distances: array of distances of matches
    @return: dataframe of matches
    """
    deduped_df = pd.DataFrame({
        'orig_idxs': np.arange(len(idxs)),
        'match_idxs': idxs.tolist(), 
        'distance': distances.tolist()
        }).explode(['match_idxs', 'distance'])

    ## Remove self matches
    deduped_df = deduped_df[deduped_df['orig_idxs'] < deduped_df['match_idxs']]
    return deduped_df
