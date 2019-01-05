import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from multiprocessing import Pool
import time
from sklearn.model_selection import train_test_split

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
        self.processed = pd.DataFrame(columns=["imgNF", 'sparse_lettre', "lettre"])
        self.processNB = processNB
        self.progress = 0

    def shift(self, x):
        self.progress += 1
        #print("Process N°", self.processNB, " Progress : ", self.progress)
        l = x.split(",")[0]
        img = convertImg(x[len(l)+1:])
        l = int(l)
        dic = {"imgNF": normalize(img, axis=1), "sparse_lettre": l, "lettre": get_lettre(l)}
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

def buildDataList():
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

# split a hauteur de 10%
def split_train_valid(dfList):
    df = pd.concat(dfList)
    df = df.reset_index(drop=True)
    train, test = train_test_split(df, test_size=0.1)
    df = None
    trainList = makeChunk(train.reset_index(drop=True), 30)
    return (trainList, test.reset_index(drop=True))

def saveDatasets(trainList, test):
    for i in range(len(trainList)):
        trainList[i].to_json("src/neural/trainingData/kaggleV3/train/kagglePart" + str(i) + ".json")
    
    test.to_json("src/neural/trainingData/kaggleV3/test/test.json")

    print("dataset sauvegardé en 30 chunks")
    return 0

def main():
    dfList = buildDataList()
    trainList, test = split_train_valid(dfList)
    saveDatasets(trainList, test)
    return 0

if __name__ == "__main__":
    main()
# ipython -m src.neural.dataBuilder.buildKaggleV3