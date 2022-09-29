'''
利用計算好的feature vectors建構kd tree.
使用kd tree搜尋相似的feature vectors, 並且記錄相似的feature vectors的index.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
import time
from sklearn.neighbors import KDTree
import pickle

def main():
    # 1. read kd tree
    # 2. read training data (feature vectors)
    # 3. Use kd tree to find similar feature vectors
    # 4. store similar feature vectors' indices
    pass

def constructKDTree():
    # 1. read feature vectors (單一動作底下所有subjects, 所有trials一起讀取) (左右手分開儲存)
    # 2. construct kd tree
    # 3. store kd tree
    
    # 1. 
    motionDirPath = 'data/swimming/'
    subjectDirPaths = [os.path.join(motionDirPath, i) for i in os.listdir(motionDirPath) if i.split('_')[-1] == 'featVecs']
    print(subjectDirPaths)
    trialsDirPaths = []
    for i in subjectDirPaths:
        for j in os.listdir(i):
            trialsDirPaths.append(os.path.join(i, j))
    print(trialsDirPaths)
    usedJointNms = ['lhand', 'rhand']
    featVecs = {_jointNm: [] for _jointNm in usedJointNms}

    for _trialDirPath in trialsDirPaths:
        for _jointNm in usedJointNms:
            featVecs[_jointNm].append(
                pd.read_csv(os.path.join(_trialDirPath, _jointNm+'.csv'))
            )
        #     break
        # break

    ## 合併所有trials的feature vectors
    fullFeatVecs = {_jointNm: pd.concat(featVecs[_jointNm], axis=0, ignore_index=True) for _jointNm in usedJointNms}
    # print(fullFeatVecs['lhand'].shape)

    # 2. 
    leftKdtree = KDTree(fullFeatVecs['lhand'].values)
    rightKdtree = KDTree(fullFeatVecs['rhand'].values)
    kdtrees = {
        'lhand': leftKdtree, 
        'rhand': rightKdtree
    }

    # 3. 
    kdtreeDirPath = os.path.join(motionDirPath, 'kdtree')
    if not os.path.isdir(kdtreeDirPath):
        os.makedirs(kdtreeDirPath)
    for _jointNm in usedJointNms:
        with open(os.path.join(kdtreeDirPath, _jointNm+'.pickle'), 'wb') as outPickle:
            pickle.dump(kdtrees[_jointNm], outPickle)

if __name__ == '__main__':
    main()
    pass