'''
更搜尋到的similar feature vector對應的poses進行synthesis
1. candidate poses需要進行head up rotation alignment
2. 根據similarity進行weighted synthesis
'''

from genericpath import isdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
import time
from sklearn.neighbors import KDTree
import pickle
from similarFeatVecSearch import readAllFeatVecsOfAMotion

def convertIndToTrialInd(trialsIndCount, sourceInd):
    '''
    Objective: 
        將使用kd tree搜尋到的raw index轉換成trial path以及在該trial內的index
    :trialsIndCount: 依照順序排列的trials資料夾地址, 以及該trial的feature vector數量
    :sourceInd: kd tree搜尋到的raw indices (一次傳入的量是top k個similar fea vecs' index)
    ======= index 排列方式 簡單例子 =======
    0 1 2 3 | 4 | 5 6 7

    count 4 1 3 

    cumCount 4 5 8

    if ind=4
    ind - cumCount = 0 1 4

    if ind = 6
    ind - cumCount = -2 -1 2
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

def convertTrialIndTo3dPos(trialsIndCount, fullPos3d, trialInd, indInTrial):
    '''
    Objective: 
        trial index以及在該trial內的index轉換成3d positions
    :trialsIndCount: 依照順序排列的trials資料夾地址, 以及該trial的feature vector數量
    :trialInd: trial的indices (一次傳入的量是top k個similar fea vecs)
    :indInTrial: trial內部feature vectors的indices (一次傳入的量是top k個similar fea vecs)
    '''
    trialsIndCount['trial3dDirPath'] = trialsIndCount['trialDirPath'].map(lambda x: x.replace('featVecs', 'pos3d'))
    # print(trialInd)
    # print(indInTrial)
    # print(trialsIndCount['trial3dDirPath'].tolist())
    # 確認fullPos3d的trial path與trialsIndCount的trials path是相通的

    # 1. 將trial index轉換成trial 3d pos dir path
    trials3dDirPath = trialsIndCount['trial3dDirPath'].iloc[trialInd].values
    # print(trials3dDirPath)
    # 2. 讀取該trial內所有joints的對應index的3d positions
    corresponding3dPos = {_jointNm: [] for _jointNm in next(iter(fullPos3d.values())).keys()}
    # print(corresponding3dPos)
    for _trialPath, _indInTrial in zip(trials3dDirPath, indInTrial):
        for _jointNm in corresponding3dPos.keys():
            _indInTrial = _indInTrial % fullPos3d[_trialPath][_jointNm].shape[0]
            corresponding3dPos[_jointNm].append(fullPos3d[_trialPath][_jointNm].iloc[_indInTrial, :].values)
            # print(_indInTrial)
            # print(fullPos3d[_trialPath][_jointNm])
            # print(fullPos3d[_trialPath][_jointNm].iloc[_indInTrial, :])
    for _jointNm in corresponding3dPos.keys():
        corresponding3dPos[_jointNm] = np.concatenate(corresponding3dPos[_jointNm])
    return corresponding3dPos

def convertSimilarFVIndTo3dPos(
    similarFVIndDirPath = 'data/swimming/similarInd/',
    trialsIndCountFilePath = 'data/swimming/kdtree/featVecsCount.csv', 
    motionDirPath = 'data/swimming/'
):
    '''
    Objective:
        將similarFeatVecSearch.py找到的similar FV index轉換成對應的full-body 3d positions
    :similarFVIndDirPath: similarFeatVecSearch.py儲存similar FV index的directory
    :similarFVDistDirPath: similarFeatVecSearch.py儲存similar FV distance的directory
    :trialsIndCountFilePath: similarFeatVecSearch.py儲存每一個trial的FV數量以及FV儲存地址
    :motionDirPath: 整個動作motion的儲存directory
    '''
    # 1. read similar vector index and corresponding distance
    # 2. use indices to retrieve corresponding poses (3d position)
    # 2.0 read trials index count 
    # 2.0.1 read all the 3d poses (all the 3d positions of all the joints)
    # 2.1 因為現在是所有trials綜合在一起的index, 
    #       所以需要將index做轉換. 
    #       raw index -> which trial and the index in that trial
    # 2.2 使用轉換後的trial path以及index讀取3d positions
    # 3. output輸出找到的3d positions

    # 1. 
    usedJointNm = ['lhand', 'rhand']
    similarFVInd = {
        _jointNm: pd.read_csv(os.path.join(similarFVIndDirPath, _jointNm+'.csv')).values for _jointNm in usedJointNm
    }

    # 2. 
    ## 2.0 read trials index count
    trialsIndCount = pd.read_csv(trialsIndCountFilePath)
    trialsIndCount['cumulativeCount'] = trialsIndCount['featVecsCount'].cumsum()
    ## 2.0.1 read all the 3d poses (all the 3d positions of all the joints)
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
    similarFVTrialInd = {_jointNm: [] for _jointNm in usedJointNm}
    similarFVIndInTrial = {_jointNm: [] for _jointNm in usedJointNm}
    for _jointNm in usedJointNm:
        for i in range(similarFVInd[_jointNm].shape[0]):
            _trialInd, _indInTrial = convertIndToTrialInd(trialsIndCount, similarFVInd[_jointNm][i, :])
            similarFVTrialInd[_jointNm].append(_trialInd)
            similarFVIndInTrial[_jointNm].append(_indInTrial)
        # 兩者整理成一個array, 然後分別儲存到dictionary裡面
        similarFVTrialInd[_jointNm] = np.array(similarFVTrialInd[_jointNm])
        similarFVIndInTrial[_jointNm] = np.array(similarFVIndInTrial[_jointNm])
    # print(similarFVTrialInd['lhand'][-10:, :])
    # print(similarFVIndInTrial['lhand'][-10:, :])
    # print(similarFVTrialInd['lhand'].shape)
    # print(similarFVIndInTrial['lhand'].shape)    

    ## 2.2 use trial name and the index to get the corresponding 3d positions/poses
    similarFVPos3d = {_jointNm: {i: [] for i in next(iter(fullPos3d.values())).keys()} for _jointNm in usedJointNm}
    for _jointNm in usedJointNm:
        for i in range(similarFVTrialInd[_jointNm].shape[0]):
            print('processing: ', i, ' of total: ', similarFVTrialInd[_jointNm].shape[0])
            _3dPos = convertTrialIndTo3dPos(
                trialsIndCount, 
                fullPos3d,
                similarFVTrialInd[_jointNm][i, :], 
                similarFVIndInTrial[_jointNm][i, :]
            )
            # print(_3dPos)
            for k in _3dPos.keys():
                similarFVPos3d[_jointNm][k].append(_3dPos[k])
            ## For test (提早結束loop)
            # if i > 1:
            #     break
        # 將單一joint的資料合併成2d array
        for k in similarFVPos3d[_jointNm].keys():
            similarFVPos3d[_jointNm][k] = \
                np.concatenate([_arr[np.newaxis, :] for _arr in similarFVPos3d[_jointNm][k]], axis=0)
        ## For test (提早結束loop)
        # if _jointNm=='lhand':
        #     break
    # print(similarFVPos3d['lhand']['Chest'])
    # print(similarFVPos3d['lhand']['Chest'].shape)
    # print(similarFVPos3d['lhand'])

    # 3. output找到的3d positions
    similar3dPosDirPaths = os.path.join(motionDirPath, 'similar3dPos')
    similar3dPosDirPaths = {_jointNm: os.path.join(similar3dPosDirPaths, _jointNm) for _jointNm in usedJointNm}
    for i in similar3dPosDirPaths.values():
        if not os.path.isdir(i):
            os.makedirs(i)
    # print(similar3dPosDirPaths)
    for _jointNm in usedJointNm:
        for k, v in similarFVPos3d[_jointNm].items():
            # pd.DataFrame(v).to_csv(
            #     os.path.join(similar3dPosDirPaths[_jointNm], k+'.csv'), 
            #     index=False
            # )
            pass

def alignHeadUpRot(inputFV, candidateFVs, winSize):
    '''
    Objective:
        對齊candidates的head up rotation. 
        詳細公式參考coolMoves的文章
    :inputFV: input feature vector
    :candidateFVs: feature vectors similar to input feature vector
    '''
    print(inputFV)
    print(candidateFVs)
    c = list(range(winSize)) / winSize
    print(c)
    # TODO: finish this section
    pass

def main():
    # 1. read 3d position files and all feature vectors file.
    #       read similar feature vectors' indices
    # 2. candidate poses need to align to input head up rotation
    # 3. 輸出align完成的結果
    # ======= 底下可以放在其他function實作 =======
    # 4. use distances to blend candidate poses

    # 1. 
    motionDirPath = 'data/swimming/'
    similarFVIndDirPath = 'data/swimming/similarInd/'
    usedJointNm = ['lhand', 'rhand']
    windowSize = 9
    ## 讀取feature vectors
    allFeatVecs = readAllFeatVecsOfAMotion(motionDirPath)
    allFeatVecs = {k: v.values for k, v in allFeatVecs.items()}
    # print(allFeatVecs['lhand'].shape)

    ## 讀取similar FV's indices
    similarFVInd = {
        _jointNm: pd.read_csv(os.path.join(similarFVIndDirPath, _jointNm+'.csv')).values for _jointNm in usedJointNm
    }
    # print(similarFVInd['lhand'].shape)

    ## 讀取轉換完成的3d positions (對應到similar feature vectors)
    ## TODO: 或許這個部分的功能也應該獨立成一個function
    similarFV3dPosDirPaths = os.path.join(motionDirPath, 'similar3dPos')
    similarFV3dPosDirPaths = [os.path.join(similarFV3dPosDirPaths, _jointNm) for _jointNm in usedJointNm]
    similarFV3dPos = {_jointNm: None for _jointNm in usedJointNm}
    for _joint, _jointDirPath in zip(usedJointNm, similarFV3dPosDirPaths):
        _jointDirPaths = [os.path.join(_jointDirPath, i) for i in os.listdir(_jointDirPath)]
        _jointNms = [i.replace('.csv', '') for i in os.listdir(_jointDirPath)]
        _joint3dPos = {}
        for j, jPath in zip(_jointNms, _jointDirPaths):
            _joint3dPos[j] = pd.read_csv(jPath).values
        similarFV3dPos[_joint] = _joint3dPos
    # print(similarFV3dPos['lhand']['Chest'].shape)

    # 2. 
    ## TODO: 每一個3d positions/pose candidate都需要做一次head up rotation alignment (輸出一個旋轉角度)
    ##          做alignment只需要lhand與rhand的feature vector
    ##          需要5個candidate的feature vector以及input的feature vector
    ##          這邊為了方便, 姑且相信第一個candidate的feature vector就是input的feature vector
    alignRot = {_jointNm: None for _jointNm in usedJointNm}
    for _jointNm in usedJointNm:
        _inputFVInd = similarFVInd[_jointNm][0, 0]
        _similarFVsInd = similarFVInd[_jointNm][0, :]
        # print(_inputFVInd)
        # print(_similarFVsInd)
        alignHeadUpRot(
            allFeatVecs[_jointNm][_inputFVInd, :], 
            allFeatVecs[_jointNm][_similarFVsInd, :], 
            windowSize
        )
        break

    pass

if __name__=='__main__':
    # 需要先將similar FV的index轉換成對應的3d positions
    # convertSimilarFVIndTo3dPos(
    #     'data/swimming/similarInd/', 'data/swimming/kdtree/featVecsCount.csv', 
    #     'data/swimming/'
    # )
    main()
    pass