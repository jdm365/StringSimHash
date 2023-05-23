import pandas as pd
import numpy as np
from time import perf_counter
from rapidfuzz import process
from rapidfuzz.distance.Indel import normalized_similarity
from jarowinkler import jarowinkler_similarity
import faiss
import logging
from matplotlib import pyplot as plt
from utils import create_dedupe_df

from StringDedup import get_topk_strings_all


def get_sim_embeddings(items, dim=512):
    rand_strings = np.random.choice(items, size=dim)
    init = perf_counter()
    vectors = process.cdist(items, rand_strings, scorer=normalized_similarity, workers=-1)
    #vectors = process.cdist(items, rand_strings, scorer=jarowinkler_similarity, workers=-1)
    logging.info(f'Elapsed Time: {perf_counter() - init}')
    return vectors


def dedup(data, k=5, dim=128, exact=False) -> pd.DataFrame:
    embeddings = get_sim_embeddings(data, dim=dim)

    if exact:
        index = faiss.IndexFlatL2(embeddings.shape[-1])
    else:
        index = faiss.index_factory(dim, f"IVF64,PQ{32 if dim < 256 else 64}")
        sampled_embeddings = embeddings[np.random.choice(len(embeddings), size=16_384)]
        index.train(sampled_embeddings)
        index.nprobe = 8

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


def test_dedup_exact(data, dedup_col, **kwargs):
    """
    Test deduplication function
    @param data: dataframe containing data to deduplicate
    @param dedup_col: column to deduplicate
    @param kwargs: keyword arguments to pass to deduplication function
    return: None
    """
    n_duplicates = len(data) - data['label'].nunique()

    start = perf_counter()
    idxs = np.array(get_topk_strings_all(data[dedup_col], data[dedup_col], **kwargs)).reshape(-1, 5)
    match_df = create_dedupe_df(idxs, np.zeros_like(idxs))
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

    K = 5

    data = data.sample(100_000).reset_index(drop=True)
    match_df = test_dedup_exact(data, 'company', k=K)

    n_duplicates = len(data) - data['label'].nunique()

    recalls = []
    dims    = [32, 64, 128, 256, 512, 1024]
    for dim in dims:
        logging.info(f'Dimension: {dim}')
        match_df = test_dedup(data, 'company', k=K, dim=dim, exact=False)
        recalls.append(np.sum(match_df['is_match']) / n_duplicates)

    plt.plot(dims, recalls)
    plt.xlabel('Dimension')
    plt.ylabel('Recall @5')
    plt.title('Recall @5 vs Dimension')
    plt.show()

    plt.savefig(f'plots/recall@{K}_vs_dim.png')
