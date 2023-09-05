import pandas as pd
import numpy as np
import multiprocessing as mp
import os
from time import perf_counter
from collections import defaultdict

from datasketch import MinHashLSH
from tqdm import tqdm

from minhash import *
import faiss




def jaccard(a: np.ndarray, b: np.ndarray):
    """Compute Jaccard similarity between two sets of items.
    Args:
        a (np.ndarray): Set of items.
        b (np.ndarray): Set of items.
    Returns:
        float: Jaccard similarity.
    """
    return len(set(a).intersection(set(b))) / len(set(a).union(set(b)))

def get_lsh_index(hashes: np.ndarray, num_buckets: int = None):
    if num_buckets is None:
        num_buckets = hashes.shape[0] // 10

    nbits = int(np.ceil(np.log2(num_buckets)))

    index = faiss.IndexLSH(hashes.shape[-1], nbits)
    index.add(hashes)
    return index

# Function to divide MinHash signature into bands and hash them into buckets
def insert_into_lsh(signature, buckets, num_bands, band_length):
    for band_idx in range(num_bands):
        start = band_idx * band_length
        end = (band_idx + 1) * band_length
        band = tuple(signature[start:end])
        buckets[band].append(signature)

# Function to query similar items from LSH index
def query_lsh(signature, buckets, num_bands, band_length):
    candidates = set()
    for band_idx in range(num_bands):
        start = band_idx * band_length
        end = (band_idx + 1) * band_length
        band = tuple(signature[start:end])
        if band in buckets:
            candidates.update(tuple(map(tuple, buckets[band])))
    return candidates

# Function to create LSH index
def create_lsh_index(signatures, num_bands, band_length):
    buckets = defaultdict(list)
    for signature in tqdm(signatures, desc='Creating LSH index'):
        insert_into_lsh(signature, buckets, num_bands, band_length)
    return buckets



if __name__ == '__main__':
    FILENAME = '../data/corrupted_companies_dedup.feather'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    FILENAME = os.path.join(current_dir, FILENAME)
    data = pd.read_feather(FILENAME)

    '''
    init = perf_counter()
    candidate_pairs = get_candidate_pairs_ds(data['company'])
    print(f'Dedup time: {perf_counter() - init:.2f} seconds')
    '''

    init = perf_counter()
    hashes = get_minhashes(data['company'], 128)
    print(f'Hashing time: {perf_counter() - init:.2f} seconds')

    init = perf_counter()
    index = create_lsh_index(hashes, 32, 4)
    print(f'Indexing time: {perf_counter() - init:.2f} seconds')
