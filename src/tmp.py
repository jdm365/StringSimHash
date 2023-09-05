import pandas as pd
import multiprocessing as mp

def match(row):
    ## do something
    pass


df = pd.read_csv('data.csv')

## paralell match over rows
pool = mp.Pool(mp.cpu_count())
df['match'] = pool.map(match, df.iterrows())


