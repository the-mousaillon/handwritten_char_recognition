import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from multiprocessing import Pool
import time

# kaggle dataset, 28*28 centered letters, à concaténer avec la mnist

## !! premier B à la position 13870 !! ##

def get_lettre(l):
    def gene():
        iniLettre = "A"
        for i in range(26):
            yield chr(ord(iniLettre) + i)
    return list(gene())[l]


def convertImg(img):
    tmps = ""
    l = []
    tmpl=[]
    j=0
    for e in img:
        if e == ',':
            tmpl += [int(tmps)]
            tmps = ""
            j+=1
        
        else:
            tmps += e

        if j == 28:
            j=0
            l += [tmpl]
            tmpl = []
    
    tmpl += [int(tmps)]
    l += [tmpl]
    l = np.array(l)
    return l


class CalculateChunk():
    def __init__(self, chunk, processNB):
        self.chunk = chunk
        self.processed = pd.DataFrame(columns=["img", "imgNF", 'sparse_lettre', "lettre"])
        self.processNB = processNB
        self.progress = 0

    def shift(self, x):
        self.progress += 1
        #print("Process N°", self.processNB, " Progress : ", self.progress)
        l = x.split(",")[0]
        img = convertImg(x[len(l)+1:])
        l = int(l)
        dic = {"img" : img, "sparse_lettre": l, "imgNF": normalize(img, axis=1).flatten(), "lettre": get_lettre(l)}
        self.processed = self.processed.append(dic, ignore_index=True)
        return None


    def run(self):
        """Code à exécuter pendant l'exécution du thread."""
        self.chunk["img"].apply(lambda x: self.shift(x))
        return self.processed


def makeChunk(df, n):
    fractile = int(len(df)/n)
    chunkList = []
    a = 0
    b = 0
    for _ in range(n-1):
        b = a + fractile
        chunkList += [df.loc[a:b]]
        a = b+1
    chunkList += [df.loc[a:]]
    return chunkList

def run(x):
    return x.run()

def runWorkers(df, nb):
    pool = Pool(processes=nb)
    chunkList = makeChunk(df, nb)
    chunkList = [CalculateChunk(chunkList[i], i+1) for i in range(len(chunkList))]
    results = [pool.apply_async(run, args=(x,)) for x in chunkList]
    results = [p.get() for p in results]
    return results

def main():
    data = pd.read_csv("src/neural/trainingData/kaggle/A_Z Handwritten Data.csv", sep='\n')
    print("début de la construction ...")
    start = time.time()
    print("--> début à : ", start)
    dfList = runWorkers(data, 50)
    end = time.time()
    print("--> fin à : ", end)
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("--> dataset construit en : {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    print("finished building dataset !!!")
    return dfList

if __name__ == "__main__":
    dfList = main()
    for i in range(len(dfList)):
        dfList[i].to_json("src/neural/trainingData/processedKaggle" + str(i) + ".json")
# ipython -m src.neural.dataBuilder.buildKaggleV2