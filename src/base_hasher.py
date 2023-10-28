import pandas as pd
import numpy as np
import faiss
import os
from tqdm import tqdm
from time import perf_counter
from rapidfuzz.process import cdist 
from rapidfuzz.distance.Levenshtein import normalized_similarity as lev_sim 
from rapidfuzz.distance.Levenshtein import distance as lev_dist
from utils import create_dedupe_df
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import cdist as cdist_scipy

from jaccard_kernel import cdist_jaccard
from minhasher import hash_strings_parallel, shingle_hash, shingle_hash_set

import datasets
from datasketch import MinHash, MinHashLSH
from nltk import ngrams

from sentence_transformers import SentenceTransformer
from typing import List


# Function to choose words using k-means++ like initialization
def choose_kmeanspp_words(num_words, candidates):
    return [candidates[idx] for idx in np.random.randint(len(candidates), size=num_words)]
    SAMPLE_SIZE = 10_000 

    if len(candidates) > SAMPLE_SIZE:
        ## candidates = np.random.choice(candidates, SAMPLE_SIZE, replace=False)
        ## candidates = candidates[np.random.choice(len(candidates), size=num_words, replace=False)]
        candidates = [candidates[idx] for idx in np.random.randint(len(candidates), size=SAMPLE_SIZE)]

    progress_bar = tqdm(total=num_words)
    idx = np.random.randint(len(candidates))
    chosen_words = [candidates[idx]]

    candidates.pop(idx)

    for idx in range(1, num_words):
        '''
        distances = cdist(
                candidates,
                chosen_words,
                scorer=lev_dist,
                workers=-1
                )
        '''
        distances = cdist_jaccard(
                candidates,
                chosen_words,
                calculate_distance=True
                )
        distances = np.min(distances, axis=1)
        
        total_distance = np.sum(distances)
        probabilities = distances / total_distance
        word_idx = np.random.choice(len(candidates), p=probabilities)
        chosen_word = candidates[word_idx]
        chosen_words.append(chosen_word)
        ## candidates = np.delete(candidates, word_idx)
        candidates.pop(word_idx)

        progress_bar.update(1)
        progress_bar.set_description(f'Initializing Random Words (KMeans++) ({len(chosen_words)}/{num_words})')
    
    return chosen_words

def sparse_to_row_nonzero_indices(sparse_matrix):
    coo_matrix = sparse_matrix.tocoo()
    row_nonzero_dict = {}
    
    for row, col in zip(coo_matrix.row, coo_matrix.col):
        if row not in row_nonzero_dict:
            row_nonzero_dict[row] = []
        row_nonzero_dict[row].append(col)
    
    row_nonzero_list = [row_nonzero_dict.get(idx, []) for idx in range(sparse_matrix.shape[0])]
    
    return row_nonzero_list

def get_sim_embeddings(items: pd.Series, dim=128):
    items = items.str.lower()
    #rand_strings = [x[:16] for x in np.random.choice(items, size=dim, replace=True)]
    ## model = CountVectorizer(analyzer='char', ngram_range=(3, 3), max_features=131_072)
    ## encoded_items = model.fit_transform(items)
    ## encoded_items = sparse_to_row_nonzero_indices(encoded_items)
    ## items[1] = items[0]
    init = perf_counter()
    encoded_items = shingle_hash(items, n_gram_size=3)
    ## encoded_items = shingle_hash_set(items, n_gram_size=3)
    ## pirnt num 0's 
    print(f'Hashing time: {perf_counter() - init:.2f} seconds')
    ## print(items[0])
    ## print(items[1])
    ## print(encoded_items[0])
    ## print(encoded_items[1])
    ## assert all([x == y for x, y in zip(encoded_items[0], encoded_items[1])])

    rand_strings = choose_kmeanspp_words(dim, encoded_items)

    ## Choose random rows from scipy sparse matrix
    ## rand_strings = encoded_items[np.random.choice(encoded_items.shape[0], size=dim, replace=False)]
    ## Get rows of indices. Need to reshape. Currently returned as flat array
    ## rand_strings  = sparse_to_row_nonzero_indices(rand_strings)

    init = perf_counter()
    '''
    embeddings = cdist(
            encoded_items,
            rand_strings,
            scorer=lev_sim, 
            workers=-1
            )
    '''
    embeddings = cdist_jaccard(
            encoded_items,
            rand_strings
            )

    ## Plot distribution of embeddings values
    ## import matplotlib.pyplot as plt
    ## plt.hist(embeddings.flatten(), bins=100)
    ## plt.show()

    print(f'Embedding time: {perf_counter() - init:.2f} seconds')
    return embeddings 


def dedup(embeddings: np.ndarray, k=10) -> pd.DataFrame:
    dim = embeddings.shape[1]
    n_clusters = int(np.sqrt(embeddings.shape[0]))
    #n_clusters = 65536

    #index = faiss.IndexFlatL2(dim)
    #coarse_quantizer = faiss.IndexFlatL2(dim)
    #index = faiss.IndexIVFFlat(coarse_quantizer, dim, n_clusters, faiss.METRIC_L2)
    #index = faiss.IndexHNSWFlat(dim, 16)
    #index.hnsw.efConstruction = 64
    #index.hnsw.efSearch = 32
    ## index = faiss.IndexFlatL2(dim)
    #index = faiss.index_factory(dim, f'IVF{n_clusters}_HNSW128,PQ32', faiss.METRIC_L2)
    #index = faiss.index_factory(dim, f'IVF{n_clusters}_HNSW32,PQ32', faiss.METRIC_L2)
    assert dim % 4 == 0, 'PQ requires num subquantizers to be a multiple of dim. Num subquantizers = dim // 4'
    index = faiss.index_factory(dim, f'IVF{n_clusters},PQ{dim // 4}x4fs', faiss.METRIC_L2)

    #index = faiss.IndexLSH(dim, 256)
    init = perf_counter()
    ## Sample
    ## train_embeddings = embeddings
    train_embeddings = embeddings[np.random.choice(embeddings.shape[0], size=50 * n_clusters, replace=False)]
    ## train_embeddings = embeddings
    index.train(train_embeddings)
    index.add(embeddings)
    print(f'Indexing time: {perf_counter() - init:.2f} seconds')

    ## index.nprobe = int(np.sqrt(int(np.sqrt(n_clusters))))
    ## index.nprobe = int(np.sqrt(n_clusters))
    index.nprobe = n_clusters
    ## index.nprobe = 100

    init = perf_counter()
    distances, idxs = index.search(embeddings, k=k)
    print(f'Search time: {perf_counter() - init:.2f} seconds')
    df = create_dedupe_df(idxs, distances)
    return df


def minhashlsh_dedup(items: np.ndarray, k=10) -> pd.DataFrame:
    # Initialize MinHashLSH
    minhash_lsh = MinHashLSH(threshold=0.5, num_perm=128)
    
    # Create MinHash objects and insert them into MinHashLSH
    for idx, item in enumerate(tqdm(items, desc="Inserting into MinHashLSH")):
        m = MinHash(num_perm=128)
        for shingle in ngrams(item, 3):
            m.update("".join(list(shingle)).encode('utf8'))
        minhash_lsh.insert(idx, m)
    
    # Create MinHash objects and insert them into MinHashLSH
    matches = []
    for idx, item in enumerate(tqdm(items, desc="Querying MinHashLSH")):
        m = MinHash(num_perm=128)
        for shingle in ngrams(item, 3):
            m.update("".join(list(shingle)).encode('utf8'))
        matches.append(minhash_lsh.query(m))
    
    ## Extract indices of matches
    idxs = [np.array(x) for x in matches]
    distances = [np.ones_like(x) for x in matches]
    
    df = create_dedupe_df(idxs, distances)
    return df

import networkx as nx

def test_dedup(df, dedup_col, dim=128, k=10):
    """
    Test deduplication function
    @param data: dataframe containing data to deduplicate
    @param dedup_col: column to deduplicate
    @param k: number of nearest neighbors to return
    return: None
    """
    unique_id_groups = df[df['label'].apply(lambda x: len(x) > 0)]['label']
    n_duplicates = len(df) - len(unique_id_groups)

    _df = df[df['label'].apply(lambda x: len(x) > 0)][['id', 'label']].explode('label')
    edges = _df[['id', 'label']].values
    G = nx.Graph()
    G.add_edges_from(edges)
    n_unique_ids = nx.number_connected_components(G)
    n_duplicates = 2 * n_unique_ids

    start = perf_counter()
    embeddings = get_sim_embeddings(df[dedup_col], dim=dim)

    ## model = SentenceTransformer('all-mpnet-base-v2')
    ## embeddings = np.array(model.encode(df[dedup_col].values, show_progress_bar=True, device='cuda:0'))
    match_df = dedup(embeddings, k=k)

    ## match_df = minhashlsh_dedup(df[dedup_col].values, k=k)
    print('Time taken: {} seconds'.format(perf_counter() - start))

    left_df  = df.iloc[match_df['orig_idxs'].values.astype(int)].reset_index(drop=True)
    left_df.columns = [f'orig_{x}' for x in left_df.columns]
    right_df = df.iloc[match_df['match_idxs'].values.astype(int)].reset_index(drop=True)
    right_df.columns = [f'match_{x}' for x in right_df.columns]

    match_df = pd.concat([left_df, right_df], axis=1).reset_index(drop=True)

    match_df = match_df.explode('match_label').reset_index(drop=True)

    num_dups_found = np.sum(match_df['orig_id'] == match_df['match_label'])
    match_df['is_match'] = match_df['orig_id'] == match_df['match_label']
    print(match_df[['orig_id', 'match_label', 'is_match']].head(40))

    print(f'Total duplicates:           {n_duplicates}')
    print(f'Number of duplicates found: {num_dups_found}')
    print(f'Recall:                     {num_dups_found / n_duplicates}')
    print(121 * '=' + '\n')

    return match_df



if __name__ == '__main__':
    """
    FILENAME = '../data/corrupted_companies_dedup.feather'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    FILENAME = os.path.join(current_dir, FILENAME)
    data = pd.read_feather(FILENAME)
    """

    DATASET_NAME = 'pinecone/core-2020-05-10-deduplication'
    data = datasets.load_dataset(DATASET_NAME)['train'].to_pandas()
    data = data[['core_id', 'doi', 'processed_title', 'processed_abstract', 'labelled_duplicates']].rename(columns={
        'core_id': 'id',
        'processed_title': 'title',
        'processed_abstract': 'abstract',
        'labelled_duplicates': 'label'
    })
    ## data['label'] = data['label'].apply(lambda x: x[0] if len(x) else -1)
    data['text'] = data.apply(lambda x: x['title'] + ' ' + x['abstract'], axis=1)
    data['label'] = data['label'].apply(lambda x: [int(_x) for _x in x])
    data['id'] = data['id'].astype(int)


    ## ~1M rows
    ## data = pd.concat([data] * 40, ignore_index=True)

    ## n_copies = 1e6 // len(data)
    ## print(n_copies)
    ## data = pd.concat([data] * int(n_copies), ignore_index=True)
    K = 10
    DIM = 128 
    ## match_df = test_dedup(data, 'company', dim=DIM, k=K)
    ## DUP_COL = 'company'
    DUP_COL = 'text'

    match_df = test_dedup(data, DUP_COL, dim=DIM, k=K)

    ## print(match_df.head(40))
    #print(match_df[match_df['is_match'] == 1].head(40))
