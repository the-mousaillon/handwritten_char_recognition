import pandas as pd
import cv2
import numpy as np

root = "src/neural/dataBuilder"

def lettreGenerator():
    iniLettre = 64
    for i in range(1,63):
        if i == 37:
            iniLettre = 64
        elif i < 11:
            yield {"lettre": str(i-1), "sample": i}
        else:
            iniLettre+=1
            yield {"lettre": chr(iniLettre), "sample": i}


def loadAllPath():
    pathDf = pd.DataFrame(columns=["path", "sample"])
    f = open("../trainingData/goodSample/Hnd/all.txt")
    for l in f:
        sample = str(int(l.split("/")[1].strip("Sample")))
        pathDf = pathDf.append({"path": l.strip("\n"), "sample": int(sample)}, ignore_index=True,)
    return pathDf


def loadAllImg():
    df = pd.DataFrame(columns=["image", "im_norm_flat", "lettre"])
    pathDf = loadAllPath()
    gene = lettreGenerator()
    for ind in gene:
        pathList = pathDf.loc[pathDf["sample"] == ind["sample"]]["path"].values
        for path in pathList:
            img = cv2.imread("../trainingData/goodSample/Hnd/" + path, 0)
            img = cv2.resize(img,(28,28))
            imNF = img.flatten()
            imNF = (imNF - imNF.mean())/ imNF.var()
            df = df.append({"image": img, "im_norm_flat": imNF, "lettre": ind["lettre"]}, ignore_index=True,)
    return df