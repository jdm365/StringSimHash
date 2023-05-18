import pandas as pd
import numpy as np
from time import perf_counter
from rapidfuzz import process
from rapidfuzz.distance.Indel import normalized_similarity
import faiss
import logging
from matplotlib import pyplot as plt


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


def get_sim_embeddings(items, dim=512):
    rand_strings = np.random.choice(items, size=dim)
    init = perf_counter()
    vectors = process.cdist(items, rand_strings, scorer=normalized_similarity, workers=-1)
    logging.info(f'Elapsed Time: {perf_counter() - init}')
    return vectors


def dedup(data, k=5, dim=128) -> pd.DataFrame:
    """
    Remove duplicates 
    @param items: list of items
    @param k: number of neighbors to consider
    @return: dataframe of matches
    """

    embeddings = get_sim_embeddings(data, dim=dim)
    index = faiss.IndexFlatL2(embeddings.shape[-1])
    index.add(embeddings)

    distances, idxs = index.search(embeddings, k=k)
    df = create_dedupe_df(idxs, distances)
    return df


def test_dedup(data, dedup_col, **kwargs):
    """
    Test deduplication function
    @param data: dataframe containing data to deduplicate
    @param dedup_col: column to deduplicate
    @param kwargs: keyword arguments to pass to deduplication function
    return: None
    """
    n_duplicates = len(data) - data['label'].nunique()

    start = perf_counter()
    match_df = dedup(data[dedup_col], **kwargs)
    logging.info('Time taken: {} seconds'.format(perf_counter() - start))

    match_df['orig_label']  = np.array(data['label'].values)[match_df['orig_idxs'].values.astype(int)]
    match_df['match_label'] = np.array(data['label'].values)[match_df['match_idxs'].values.astype(int)]

    match_df['is_match'] = (match_df['orig_label'] == match_df['match_label']).astype(int)

    logging.info(f'Recall:    {np.sum(match_df["is_match"]) / n_duplicates}')

    return match_df



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    FILENAME = 'data/corrupted_companies_dedup.feather'
    data = pd.read_feather(FILENAME)

    recalls = []
    dims    = [16, 32, 64, 128, 256, 512, 1024]
    for dim in dims:
        logging.info(f'Dimension: {dim}')
        match_df = test_dedup(data, 'company', k=5, dim=dim)
        recalls.append(np.sum(match_df['is_match']) / len(match_df))

    plt.plot(dims, recalls)
    plt.xlabel('Dimension')
    plt.ylabel('Recall @5')
    plt.title('Recall @5 vs Dimension')
    plt.show()

    plt.savefig('plots/recall@5_vs_dim.png')
