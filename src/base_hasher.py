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

from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn

# Function to choose words using k-means++ like initialization
def choose_kmeanspp_words(num_words, candidates):
    if len(candidates) > 25_000:
        candidates = np.random.choice(candidates, 25_000, replace=False)

    progress_bar = tqdm(total=num_words)
    idx = np.random.randint(len(candidates))
    chosen_words = [candidates[idx]]

    candidates = np.delete(candidates, idx)

    for idx in range(1, num_words):
        distances = cdist(
                candidates,
                chosen_words,
                scorer=lev_dist,
                workers=-1
                )
        distances = np.min(distances, axis=1)
        
        total_distance = np.sum(distances)
        probabilities = distances / total_distance
        word_idx = np.random.choice(len(candidates), p=probabilities)
        chosen_word = candidates[word_idx]
        chosen_words.append(chosen_word)
        candidates = np.delete(candidates, word_idx)

        progress_bar.update(1)
        progress_bar.set_description(f'Initializing Random Words (KMeans++) ({len(chosen_words)}/{num_words})')
    
    return chosen_words

def get_sim_embeddings(items: pd.Series, dim=128):
    items = items.str.lower()
    #rand_strings = [x[:16] for x in np.random.choice(items, size=dim, replace=True)]
    rand_strings = choose_kmeanspp_words(dim, items)

    init = perf_counter()
    embeddings = cdist(
            items, 
            rand_strings,
            scorer=lev_sim, 
            workers=-1
            )

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
    #index = faiss.index_factory(dim, f'IVF{n_clusters}_HNSW32,PQ16', faiss.METRIC_L2)
    #index = faiss.index_factory(dim, f'IVF{n_clusters}_HNSW128,PQ32', faiss.METRIC_L2)
    #index = faiss.index_factory(dim, f'IVF{n_clusters}_HNSW32,PQ32', faiss.METRIC_L2)
    assert dim % 4 == 0, 'PQ requires num subquantizers to be a multiple of dim. Num subquantizers = dim // 4'
    index = faiss.index_factory(dim, f'IVF{n_clusters},PQ{dim // 4}x4fs', faiss.METRIC_L2)
    #index = faiss.index_factory(dim, 'IVF65536_HNSW32,PQ128x4fs', faiss.METRIC_L2)

    #index = faiss.IndexLSH(dim, 256)
    init = perf_counter()
    ## Sample
    train_embeddings = embeddings[np.random.choice(embeddings.shape[0], size=50 * n_clusters, replace=False)]
    index.train(train_embeddings)
    ## index.train(embeddings)
    index.add(embeddings)
    print(f'Indexing time: {perf_counter() - init:.2f} seconds')

    index.nprobe = int(np.sqrt(int(np.sqrt(n_clusters))))
    #index.nprobe = 2 * int(np.sqrt(n_clusters))

    init = perf_counter()
    distances, idxs = index.search(embeddings, k=k)
    print(f'Search time: {perf_counter() - init:.2f} seconds')
    df = create_dedupe_df(idxs, distances)
    return df

def extract_topn_from_scipy_sparse(results):
    distances = []
    indices   = []
    for idx in range(results.shape[0]):
        start_ptr, end_ptr = results.indptr[idx], results.indptr[idx + 1]
        values = results.data[start_ptr:end_ptr]
        idxs   = results.indices[start_ptr:end_ptr]

        distances.append(values)
        indices.append(idxs)
    return distances, indices


def dedup_sparse(embeddings, k=10):
    ## my_awesome_cossim_topn function from https://gist.github.com/ymwdalex/9a1c3746626b0e3d9e5a
    results = awesome_cossim_topn(
            embeddings, 
            embeddings.T, 
            ntop=k,
            use_threads=True,
            n_jobs=os.cpu_count()
            )
    distances, idxs = extract_topn_from_scipy_sparse(results)
    df = create_dedupe_df(idxs, distances)
    return df

def get_tfidf_embeddings(items: pd.Series):
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3, 4))
    X = tfidf.fit_transform(items)
    return X 

def test_dedup(data, dedup_col, dim=128, k=10):
    """
    Test deduplication function
    @param data: dataframe containing data to deduplicate
    @param dedup_col: column to deduplicate
    @param k: number of nearest neighbors to return
    return: None
    """
    n_duplicates = len(data) - data['label'].nunique()

    start = perf_counter()
    embeddings = get_sim_embeddings(data[dedup_col], dim=dim)
    match_df = dedup(embeddings, k=k)
    #embeddings = get_tfidf_embeddings(data[dedup_col])
    #match_df = dedup_sparse(embeddings, k=k)
    print('Time taken: {} seconds'.format(perf_counter() - start))

    match_df['orig_label']  = np.array(data['label'].values)[match_df['orig_idxs'].values.astype(int)]
    match_df['match_label'] = np.array(data['label'].values)[match_df['match_idxs'].values.astype(int)]

    match_df['is_match'] = (match_df['orig_label'] == match_df['match_label']).astype(int)

    print(f'Recall:     {np.sum(match_df["is_match"]) / n_duplicates}')
    print(121 * '=' + '\n')

    return match_df




if __name__ == '__main__':
    FILENAME = '../data/corrupted_companies_dedup.feather'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    FILENAME = os.path.join(current_dir, FILENAME)
    data = pd.read_feather(FILENAME)

    ## ~1M rows
    ## data = pd.concat([data] * 40, ignore_index=True)

    ## n_copies = 1e6 // len(data)
    ## print(n_copies)
    ## data = pd.concat([data] * int(n_copies), ignore_index=True)
    K = 10
    DIM = 512
    match_df = test_dedup(data, 'company', dim=DIM, k=K)

    match_df.drop(columns=['orig_label', 'match_label'], inplace=True)
    match_df['orig_name'] = np.array(data['company'].values)[match_df['orig_idxs'].values.astype(int)]
    match_df['match_name'] = np.array(data['company'].values)[match_df['match_idxs'].values.astype(int)]
    match_df.drop(columns=['orig_idxs', 'match_idxs'], inplace=True)
    print(match_df.head(40))
    #print(match_df[match_df['is_match'] == 1].head(40))
