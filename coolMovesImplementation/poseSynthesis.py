'''
更搜尋到的similar feature vector對應的poses進行synthesis
1. candidate poses需要進行head up rotation alignment
2. 根據similarity進行weighted synthesis
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
import time
from sklearn.neighbors import KDTree
import pickle

def convertIndToTrialInd(trialsIndCount, sourceInd):
    '''
    Objective: 
        將使用kd tree搜尋到的raw index轉換成trial path以及在該trial內的index
    :trialsIndCount: 依照順序排列的trials資料夾地址, 以及該trial的feature vector數量
    :sourceInd: kd tree搜尋到的raw indices (一次傳入的量是top k個similar fea vecs' index)
    '''
    cumulativeCount = trialsIndCount['cumulativeCount'].values
    cumulativeCount = cumulativeCount[np.newaxis, :]
    sourceInd = sourceInd[:, np.newaxis]
    comparedInd = cumulativeCount-sourceInd
    trialInd = np.argmax(comparedInd>0, axis=1)
    indInTrial = np.take_along_axis(comparedInd, trialInd[:, np.newaxis]-1, axis=1)
    indInTrial = np.squeeze(np.abs(indInTrial))
    # 需要處理index是0的情況
    indInTrial[trialInd==0] = np.squeeze(sourceInd)[trialInd==0]

    # print(cumulativeCount)
    # print(sourceInd)
    # print(comparedInd)
    # print(trialInd)
    # print(indInTrial)
    return trialInd, indInTrial

def main():
    # 1. read similar vector index and corresponding distance
    # 2. use indices to retrieve corresponding poses (3d position)
    # 2.0 read trials index count 
    # 2.0.1 read all the 3d poses (all the 3d positions of all the joints)
    # 2.1 因為現在是所有trials綜合在一起的index, 
    #       所以需要將index做轉換. 
    #       raw index -> which trial and the index in that trial
    # 2.2 使用轉換後的trial path以及index讀取3d positions
    # ======= 底下的部分或許可以放到另一個function實作 ======= 
    # 3. candidate poses need to align to input head up rotation
    # 4. use distances to blend candidate poses

    # 1. 
    similarFVIndDirPath = 'data/swimming/similarInd/'
    similarFVDistDirPath = 'data/swimming/similarInd/'
    usedJointNm = ['lhand', 'rhand']
    similarFVInd = {
        _jointNm: pd.read_csv(os.path.join(similarFVIndDirPath, _jointNm+'.csv')).values for _jointNm in usedJointNm
    }
    similarFVDist = {
        _jointNm: pd.read_csv(os.path.join(similarFVDistDirPath, _jointNm+'.csv')).values for _jointNm in usedJointNm
    }

    # 2. 
    ## 2.0 read trials index count
    trialsIndCountFilePath = 'data/swimming/kdtree/featVecsCount.csv'
    trialsIndCount = pd.read_csv(trialsIndCountFilePath)
    trialsIndCount['cumulativeCount'] = trialsIndCount['featVecsCount'].cumsum()
    ## 2.0.1 read all the 3d poses (all the 3d positions of all the joints)
    motionDirPath = 'data/swimming/'
    subjectDirPaths = [os.path.join(motionDirPath, i) for i in os.listdir(motionDirPath) if i.split('_')[-1] == 'pos3d']
    trialsDirPaths = []
    for i in subjectDirPaths:
        for j in os.listdir(i):
            trialsDirPaths.append(os.path.join(i, j))
    
    allJointsNm = [i.replace('.csv', '') for i in os.listdir(trialsDirPaths[0])]
    # print(allJointsNm)
    fullPos3d = {_trialDirPath: {_jointNm: None for _jointNm in allJointsNm} for _trialDirPath in trialsDirPaths}
    for _trialDirPath in trialsDirPaths:
        for _jointNm in allJointsNm:
            fullPos3d[_trialDirPath][_jointNm] = pd.read_csv(os.path.join(_trialDirPath, _jointNm+'.csv'))
    # print(fullPos3d.keys())

    ## 2.1 convert index to trial path and the index in that trial
    similarFVTrialInd = {_jointNm: None for _jointNm in usedJointNm}
    similarFVIndInTrial = {_jointNm: None for _jointNm in usedJointNm}
    for _jointNm in usedJointNm:
        _trialInd, _indInTrial = convertIndToTrialInd(trialsIndCount, similarFVInd[_jointNm][0, :])
        # TODO: 兩者整理成一個array, 然後分別儲存到dictionary裡面
        break
    
    

    ## 2.2 use trial name and the index to get the corresponding 3d positions/poses
    pass

if __name__=='__main__':
    main()
    pass