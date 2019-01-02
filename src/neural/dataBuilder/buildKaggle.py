import pandas as pd
import cv2
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from threading import Thread
from multiprocessing import Process, Queue


# kaggle dataset, 28*28 centered letters, à concaténer avec la mnist

## !! premier B à la position 13870 !! ##

data = pd.read_csv("src/neural/trainingData/bigCsvSample/A_Z Handwritten Data.csv", sep='\n')

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

class CalculateChunk(Thread):
    def __init__(self, chunk, threadNb, queue):
        Thread.__init__(self)
        self.chunk = chunk
        self.processed = pd.DataFrame(columns=["img", 'Lettre', "imgNF"])
        self.treadNb = threadNb
        self.progress = 0
        self.queue = queue

    def shift(self, x):
        self.progress += 1
        print("Thread N°", self.treadNb, " Preogress : ", self.progress)
        l = x.split(",")[0]
        img = convertImg(x[len(l)+1:])
        l = int(l)
        dic = {"img" : img, "Lettre": l, "imgNF": img}
        self.processed = self.processed.append(dic, ignore_index=True)
        return None


    def run(self):
        """Code à exécuter pendant l'exécution du thread."""
        self.chunk["img"].apply(lambda x: self.shift(x))
        self.queue.put([self.processed])

def makeChunk(df, n):
    fractile = int(len(df)/n)
    chunkList = []
    a = 0
    b = 0
    for i in range(n):
        b = a + fractile
        chunkList += [df.loc[a:b]]
        a = b+1
    return chunkList

def startComputation(chunkList, muliplicator, queue):
    threadList = []
    n=0
    for e in chunkList:
        n+=1
        threadList += [CalculateChunk(e, muliplicator*n, queue)]
    
    for e in threadList:
        e.start()
    
    for e in threadList:
        e.join()


def createProcess(thread, process):
    ThreadPerProcess = int(process/thread)
    chunkList = [{"chunk": makeChunk(data, ThreadPerProcess), "nb": p+1, "queue":Queue()} for p in range(process)]
    processList = [Process(target=startComputation, args=(e["chunk"],e["nb"], e["queue"])) for e in chunkList]
    return processList


def startProcess(processList):
    for p in processList:
        p.start()

    for p in processList:
        p.join()

if __name__ == "__main__":
    processList = createProcess(2, 4)
    startProcess(processList)