import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
from time import perf_counter
from rapidfuzz import process
from rapidfuzz.distance.Indel import normalized_similarity as indel_sim 
from rapidfuzz.distance.Levenshtein import normalized_similarity as lev_sim 
from rapidfuzz.distance.JaroWinkler import normalized_similarity as jw_sim 
from rapidfuzz.distance.Prefix import normalized_similarity as prefix_sim 
from rapidfuzz.distance.Postfix import normalized_similarity as postfix_sim 
import faiss
import logging
from matplotlib import pyplot as plt
from utils import create_dedupe_df
from gensim.models import Word2Vec
from tqdm import tqdm
import os
from transformers import AutoModel, AutoTokenizer

#from StringDedup import get_topk_strings_all, get_dedup_candidates
from sentence_transformers import SentenceTransformer

from typing import List


def _n_gram_shingle(string, n=2) -> List[str]:
    return [string[idx:idx+n] for idx in range(len(string) - n + 1)]

n_gram_shingle = np.vectorize(_n_gram_shingle)

def get_word2vec_embeddings(items: List[str]):
    items = [n_gram_shingle(item) for item in items]

    THREAD_COUNT: int = os.cpu_count()
    model = Word2Vec(
            min_count=1, 
            vector_size=256, 
            workers=THREAD_COUNT
            )
    model.build_vocab(items)
    model.train(items, total_examples=model.corpus_count, epochs=5, compute_loss=True)

    embeddings = np.zeros((len(items), model.wv.vector_size))
    for idx, item in enumerate(tqdm(items, desc='Embedding items')):
        embeddings[idx] = np.mean([model.wv[word] for word in item], axis=0)
    return embeddings


def get_glove_embeddings(items):
    #model_name = 'average_word_embeddings_glove.840B.300d'
    model_name = 'shahrukhx01/paraphrase-mpnet-base-v2-fuzzy-matcher'
    #model_name = 'gtr-t5-xl'
    #model_name = 'openlm-research/open_llama_3b'

    '''
    model = SentenceTransformer(
            model_name, 
            device='cuda:0' if T.cuda.is_available() else 'cpu'
            )
    embeddings = model.encode(items, show_progress_bar=True)
    '''
    ## Load model from huggingface to get embeddings at fp16
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ## Use float16
    model = model.half()

    model.to('cuda:0' if T.cuda.is_available() else 'cpu')

    with T.no_grad():
        embeddings = []
        for item in tqdm(items, desc='Embedding items'):
            input_ids = tokenizer(item, return_tensors='pt').input_ids.to('cuda:0' if T.cuda.is_available() else 'cpu')
            ## Get embeddings from model
            outputs   = model(input_ids)
            embeddings.append(outputs[0].cpu().numpy()[0])
        embeddings = np.vstack(embeddings)
    return embeddings

@njit(parallel=True)
def get_entropy(sims: np.ndarray):
    eta = 1e-8
    return -np.sum(sims * np.log(sims + eta), axis=1) + np.log(np.sum(sims, axis=1) + eta)


def get_random_sample(items, dim=512, neg_sample_dim=128):
    #return np.random.choice(items, size=dim)
    str_list = []
    for idx in range(dim):
        if idx == 0:
            str_list.append(np.random.choice(items))
            continue

        rand_strings = np.random.choice(items, size=neg_sample_dim)
        sims = process.cdist(
                rand_strings, 
                str_list,
                scorer=lev_sim, 
                workers=-1
                )

        ## Get string which maximizes entropy of sims
        entropy = get_entropy(sims)
        str_list.append(rand_strings[np.argmax(entropy)])
    return str_list


def get_sim_embeddings(items, dim=512):
    SIM_FUNCS = {
            #indel_sim: 0.20,
            lev_sim: 1.00,#0.40,
            #jw_sim: 0.20,
            #prefix_sim: 0.10,
            #postfix_sim: 0.10
            }

    items = [x.lower() for x in items]
    rand_strings = get_random_sample(items, dim)

    init = perf_counter()


    embeddings = np.zeros((len(items), dim), dtype=np.float32)

    start = 0
    for sim_func, weight in SIM_FUNCS.items():
        embeddings[:, start:start + int(weight * len(rand_strings))] = np.array(
                process.cdist(
                    items, 
                    rand_strings[:int(weight * len(rand_strings))],
                    scorer=sim_func, 
                    workers=-1
                    )
                )
        start += int(len(rand_strings) * weight)

    logging.info(f'Elapsed time: {perf_counter() - init:.2f} seconds')
    return embeddings 


def fine_tune_embeddings(data, labels, dim=512):
    y = labels
    X_orig = get_sim_embeddings(data, dim=dim)

    df = create_training_df(X_orig.tolist(), labels)

    X_0 = T.tensor(np.stack(df['name_0'].values), dtype=T.float32)
    X_1 = T.tensor(np.stack(df['name_1'].values), dtype=T.float32)
    y   = df['is_match'].apply(lambda x: 1 if x else -1).values
    y   = T.tensor(y, dtype=T.float32)

    ## Train model which is just a no-bias linear projection to dim=256 embedding.
    ## Loss will be CosineSimilarityLoss
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    model = nn.Linear(X_0.shape[-1], dim // 4, bias=False)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CosineEmbeddingLoss()
    model.to(device)

    dataloader = DataLoader(
            TensorDataset(X_0, X_1, y),
            batch_size=32,
            shuffle=True
            )
    NUM_EPOCHS = 10
    losses = []

    progress_bar = tqdm(total=len(dataloader) * NUM_EPOCHS)
    for _ in range(NUM_EPOCHS):
        for X_0, X_1, y in dataloader:
            X_0 = X_0.to(device)
            X_1 = X_1.to(device)
            y   = y.to(device)

            emb0 = model(X_0)
            emb1 = model(X_1)
            loss = loss_fn(emb0, emb1, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())

            progress_bar.update(1)
            progress_bar.set_description(f'Loss: {np.mean(losses[-100:]):.4f}')

    progress_bar.close()

    ## Get embeddings from model
    model.eval()
    eval_loader = DataLoader(
            TensorDataset(T.tensor(X_orig, dtype=T.float32)),
            batch_size=32,
            shuffle=False
            )
    with T.no_grad():
        embeddings = []
        for X in tqdm(eval_loader, desc='Embedding items'):
            X = X[0].to(device)
            outputs = model(X)
            embeddings.append(outputs.cpu().numpy())
        embeddings = np.vstack(embeddings)
    return embeddings



def dedup(data, k=5, dim=128, exact=False, use_glove=False, labels=None) -> pd.DataFrame:
    if use_glove:
        embeddings = get_glove_embeddings(data)
        #sim_embeddings = get_sim_embeddings(data, dim=embeddings.shape[-1])

        #embeddings = sim_embeddings
        #embeddings = 0.0 * embeddings + 1.0 * sim_embeddings
    else:
        embeddings = get_sim_embeddings(data, dim=dim)
        #embeddings = fine_tune_embeddings(data, labels, dim=dim)
        #embeddings = get_word2vec_embeddings(data)

    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    print(121 * '=')
    logging.info(f'Using device: {device}')

    if exact:

        if device == T.device('cuda:0'):
            res   = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(embeddings.shape[-1])
            index = faiss.index_cpu_to_gpu(res, 0, index)

        elif device == T.device('cpu'):
            index = faiss.IndexFlatL2(embeddings.shape[-1])

    else:

        if device == T.device('cuda:0'):
            res   = faiss.StandardGpuResources()
            #index = faiss.index_factory(dim, f"IVF64,PQ{32 if dim < 256 else 64}")
            index = faiss.index_factory(dim, f"IVF256,PQ{32 if dim < 256 else 64}")
            #sampled_embeddings = embeddings[np.random.choice(len(embeddings), size=16_384)]
            #sampled_embeddings = T.tensor(sampled_embeddings, dtype=T.float16)

            index = faiss.index_cpu_to_gpu(res, 0, index)
            index.useFloat16LookupTables = True
            #index.train(sampled_embeddings, useFloat16LookupTables=True)
            embeddings = T.tensor(embeddings, dtype=T.float16)
            index.train(embeddings)
            index.nprobe = 8

        elif device == T.device('cpu'):
            index = faiss.index_factory(dim, f"IVF64,PQ{32 if dim < 256 else 64}")
            #sampled_embeddings = embeddings[np.random.choice(len(embeddings), size=16_384)]

            #index.train(sampled_embeddings)
            embeddings = T.tensor(embeddings, dtype=T.float16)
            index.train(embeddings)
            index.nprobe = 8

    logging.info(f'Using index: {index}')
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
    match_df = dedup(data[dedup_col], labels=data['label'], **kwargs)
    logging.info('Time taken: {} seconds'.format(perf_counter() - start))

    match_df['orig_label']  = np.array(data['label'].values)[match_df['orig_idxs'].values.astype(int)]
    match_df['match_label'] = np.array(data['label'].values)[match_df['match_idxs'].values.astype(int)]

    match_df['is_match'] = (match_df['orig_label'] == match_df['match_label']).astype(int)

    logging.info(f'Recall:     {np.sum(match_df["is_match"]) / n_duplicates}')
    print(121 * '=' + '\n')

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
    idxs = np.array(get_topk_strings_all(data[dedup_col], data[dedup_col], **kwargs)).reshape(-1, kwargs['k'])
    #idxs = np.array(get_dedup_candidates(data[dedup_col], **kwargs)).reshape(-1, 5)
    match_df = create_dedupe_df(idxs, np.zeros_like(idxs))
    logging.info('Time taken: {} seconds'.format(perf_counter() - start))

    match_df['orig_label']  = np.array(data['label'].values)[match_df['orig_idxs'].values.astype(int)]
    match_df['match_label'] = np.array(data['label'].values)[match_df['match_idxs'].values.astype(int)]

    match_df['is_match'] = (match_df['orig_label'] == match_df['match_label']).astype(int)

    logging.info(f'Recall:     {np.sum(match_df["is_match"]) / n_duplicates}')

    return match_df


def create_training_df(data, labels):
    df = pd.DataFrame({'name': data, 'label': labels})

    df['id'] = df.index
    matches_df = pd.merge(df, df, on='label', suffixes=('_0', '_1'), how='inner')
    matches_df = matches_df[matches_df['id_0'] != matches_df['id_1']].reset_index(drop=True)
    matches_df.drop(['id_0', 'id_1', 'label'], axis=1, inplace=True)

    matches_df['is_match'] = 1

    non_matches_df_a = df.sample(len(matches_df) * 2, replace=True).reset_index(drop=True)
    non_matches_df_b = df.sample(len(matches_df) * 2, replace=True).reset_index(drop=True)
    
    non_matches_df_a.drop(['id'], axis=1, inplace=True)
    non_matches_df_b.drop(['id'], axis=1, inplace=True)

    non_matches_df_a.rename(columns={'name': 'name_0', 'label': 'label_0'}, inplace=True)
    non_matches_df_b.rename(columns={'name': 'name_1', 'label': 'label_1'}, inplace=True)

    non_matches_df = pd.concat([non_matches_df_a, non_matches_df_b], axis=1)

    non_matches_df = non_matches_df[non_matches_df['label_0'] != non_matches_df['label_1']]
    non_matches_df = non_matches_df.reset_index(drop=True)
    non_matches_df['is_match'] = 0
    non_matches_df.drop(['label_0', 'label_1'], axis=1, inplace=True)

    final_df = pd.concat([matches_df, non_matches_df], axis=0).reset_index(drop=True)

    return final_df 



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    FILENAME = 'data/corrupted_companies_dedup.feather'
    data = pd.read_feather(FILENAME)

    K = 10

    #data = data.sample(100_000).reset_index(drop=True)
    #match_df = test_dedup_exact(data, 'company', k=K)

    n_duplicates = len(data) - data['label'].nunique()

    #print(data.head(10))

    match_df = test_dedup(
            data, 
            'company', 
            k=K, 
            dim=512, 
            exact=True,
            use_glove=False
            )
    recall = np.sum(match_df['is_match']) / n_duplicates
    #logging.info(f'Recall: {recall}')

    """
    recalls = []
    dims    = [32, 64, 128, 256, 512, 1024]
    for dim in dims:
        logging.info(f'Dimension: {dim}')
        match_df = test_dedup(data, 'company', k=K, dim=dim, exact=True)
        recalls.append(np.sum(match_df['is_match']) / n_duplicates)

    plt.plot(dims, recalls)
    plt.xlabel('Dimension')
    plt.ylabel('Recall @5')
    plt.title('Recall @5 vs Dimension')
    plt.show()

    plt.savefig(f'plots/recall@{K}_vs_dim.png')
    """
