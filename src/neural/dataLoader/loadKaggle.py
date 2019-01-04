import pandas as pd
import numpy as np
from multiprocessing import Pool
import time
from os import listdir

def loadAll():
    start = time.time()
    path = "src/neural/trainingData/kaggleProcessed/"
    kaggleList = [f for f in listdir(path) if f[-4:] == "json"]
    pool = Pool(processes=len(kaggleList))
    results = [pool.apply_async(pd.read_json, args=(path + x,)) for x in kaggleList]
    results = [p.get() for p in results]
    df = pd.concat(results).reset_index(drop=True)
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("--> dataset chargé en : {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    return df

if __name__ == "__main__":
    df = loadAll()