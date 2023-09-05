import pandas as pd
import numpy as np
import multiprocessing as mp
import os
from time import perf_counter
from typing import List

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
from nltk import ngrams


# Function to create MinHash for a given text
def create_minhash(text, num_perm=32):
    m = MinHash(num_perm=num_perm)
    ##SHINGLE
    for d in ngrams(text, 3):
        m.update("".join(d).encode('utf-8'))
    return m

def create_minhash_parallel(items, num_perm=32) -> List[MinHash]:
    with mp.Pool(mp.cpu_count()) as pool:
        minhashes = pool.starmap(create_minhash, [(item, num_perm) for item in items])
    return minhashes

def fast_insertion(lsh, minhashes, buffer_size=50000):
    with lsh.insertion_session(buffer_size=buffer_size) as session:
        for idx, minhash in enumerate(minhashes):
            session.insert(idx, minhash)

def create_lsh_index(items, num_perm=32):
    lsh = MinHashLSH(threshold=0.8, num_perm=num_perm)
    minhashes = create_minhash_parallel(items, num_perm=num_perm)
    fast_insertion(lsh, minhashes)
    #for idx, minhash in enumerate(tqdm(minhashes, desc="Populating MinHashLSH")):
        #lsh.insert(idx, minhash)
    return lsh



if __name__ == '__main__':
    FILENAME = '../data/corrupted_companies_dedup.feather'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    FILENAME = os.path.join(current_dir, FILENAME)
    data = pd.read_feather(FILENAME)['company']

    # Create MinHashLSH index
    lsh = create_lsh_index(data)

    # Query to find candidate duplicates for each document
    for _data in tqdm(data, desc="Finding Candidates"):
        candidates = lsh.query(create_minhash(_data))
        if len(candidates) > 1:
            print(f"Candidate duplicates for {_data}: {[data[candidate] for candidate in candidates]}")
