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
from scipy.spatial.transform import Rotation as R
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
    # print(inputFV)
    # print(candidateFVs)
    inputX = inputFV[:winSize]
    inputZ = inputFV[winSize*2:winSize*3]
    c = np.array([i / winSize for i in list(range(winSize))])
    # print(c)
    candidatesRot = np.zeros(candidateFVs.shape[0])
    numOfCandidates = candidateFVs.shape[0]
    for _candidateInd in range(numOfCandidates):
        _candidateFV = candidateFVs[_candidateInd, :]
        _candidateX = _candidateFV[:winSize]
        _candidateZ = _candidateFV[winSize*2:winSize*3]
        # print(_candidateX)
        # print(_candidateZ)
        ## 計算角度
        numeratorLeft = np.sum(c*(inputX*_candidateZ - _candidateX*inputZ))
        numeratorRight = \
            (np.sum(c*inputX)*np.sum(c*_candidateZ) - np.sum(c*_candidateX)*np.sum(c*inputZ)) / np.sum(c)
        numerator = numeratorLeft - numeratorRight
        denominatorLeft = np.sum(c*(inputX*_candidateX - _candidateZ*inputZ))
        denominatorRight = \
            (np.sum(c*inputX)*np.sum(c*_candidateX) + np.sum(c*_candidateZ)*np.sum(c*inputZ)) / np.sum(c)
        denominator = denominatorLeft - denominatorRight
        # print(numerator/denominator)
        candidatesRot[_candidateInd] = np.arctan(numerator/denominator)
    # print(candidatesRot)
    return candidatesRot

def readSimilarFV3dPos(motionDirPath):
    '''
    Objective:
        讀取similart feature vectors對應的3d positions (full-body 3d positions)
    :motionDirPath: 儲存整個動作資訊的資料夾地址
    '''
    usedJointNm = ['lhand', 'rhand']
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
    return similarFV3dPos

def computeAlignRot(
    motionDirPath = 'data/swimming/', 
    similarFVIndDirPath = 'data/swimming/similarInd/', 
    usedJointNm = ['lhand', 'rhand'], 
    windowSize = 9
):
    # 1. read 3d position files and all feature vectors file.
    #       read similar feature vectors' indices
    # 2. candidate poses need to align to input head up rotation
    # 3. 輸出/儲存計算出來的alignment rotation
    # ======= 底下可以放在其他function實作 =======
    # 4. use distances to blend candidate poses

    # 1. 
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
    ## output from convertSimilarFVIndTo3dPos()
    _similarFV3dPos = readSimilarFV3dPos(motionDirPath)
    # print(_similarFV3dPos['lhand']['Chest'].shape)
    # print(_similarFV3dPos['lhand']['Chest'][:10, :5])

    # 2. 
    ## 每一個3d positions/pose candidate都需要做一次head up rotation alignment (輸出一個旋轉角度)
    ##          做alignment只需要lhand與rhand的feature vector
    ##          需要5個candidate的feature vector以及input的feature vector
    ##          這邊為了方便, 姑且相信第一個candidate的feature vector就是input的feature vector
    alignRot = {_jointNm: np.zeros(similarFVInd[_jointNm].shape) for _jointNm in usedJointNm}
    for _jointNm in usedJointNm:
        for i in range(similarFVInd[_jointNm].shape[0]):
            _inputFVInd = similarFVInd[_jointNm][i, 0]
            _similarFVsInd = similarFVInd[_jointNm][i, :]
            # print(_inputFVInd)
            # print(_similarFVsInd)
            _rot = alignHeadUpRot(
                allFeatVecs[_jointNm][_inputFVInd, :], 
                allFeatVecs[_jointNm][_similarFVsInd, :], 
                windowSize
            )
            alignRot[_jointNm][i, :] = _rot
            # break
        # break

    # 3. 
    similarFVAlignRotDirPath = os.path.join(motionDirPath, 'similarAlignRot')
    if not os.path.isdir(similarFVAlignRotDirPath):
        os.makedirs(similarFVAlignRotDirPath)
    for _jointNm in usedJointNm:
        # 儲存檔案
        pd.DataFrame(alignRot[_jointNm]).to_csv(
            os.path.join(similarFVAlignRotDirPath, _jointNm+'.csv'), 
            index=False
        )

def applyAlignRotTo3dPos(
    motionDirPath = 'data/swimming/', 
    usedJointNm = ['lhand', 'rhand'], 
    numOfCandidate = 5
):
    # 1. read 3d positions
    # 2. read align rotations
    # 3. apply rotation to 3d positions
    # 4. output/store aligned 3d positions

    # 1. 
    similarFV3dPos = readSimilarFV3dPos(motionDirPath)
    # 2. 
    alignRotDirPath = os.path.join(motionDirPath, 'similarAlignRot')
    alignRot = {_jointNm: None for _jointNm in usedJointNm}
    for _jointNm in usedJointNm:
        alignRot[_jointNm] = \
            pd.read_csv(os.path.join(alignRotDirPath, _jointNm+'.csv')).values
    # 3. 
    aligned3dPos = {k: {i: np.zeros(similarFV3dPos[k][i].shape) for i in similarFV3dPos[k]} for k in similarFV3dPos}
    for _jointNm in usedJointNm:
        for _ind in range(alignRot[_jointNm].shape[0]):
            print('processing ', _ind, ' of total ', alignRot[_jointNm].shape[0])
            for k, _3dPos in similarFV3dPos[_jointNm].items():
                # print('======= =======')
                # print(k)
                # print(_3dPos[_ind, :])
                for j in range(numOfCandidate): 
                    r = R.from_euler('y', alignRot[_jointNm][_ind, j], degrees=True)
                    aligned3dPos[_jointNm][k][_ind, 3*j:3*(j+1)] = \
                        r.apply(_3dPos[_ind, 3*j:3*(j+1)])
                    # print(alignRot[_jointNm][_ind, j])
                    # print(_3dPos[_ind, 3*j:3*(j+1)])
                    # print(aligned3dPos[_jointNm][k][_ind, 3*j:3*(j+1)])
                # break
            ## For debug
            # if _ind > 30:
            #     break
        # break
    
    # print(aligned3dPos['lhand']['Chest'][:10, :6])
    # print(similarFV3dPos['lhand']['Chest'][:10, :6])

    # 4. 
    rotAligned3dPosDirPath = motionDirPath + 'similarAligned3dPos'
    for _jointNm in usedJointNm:
        _dirPath = os.path.join(rotAligned3dPosDirPath, _jointNm)
        print(_dirPath)
        if not os.path.isdir(_dirPath):
            os.makedirs(_dirPath)
        for k, v in aligned3dPos[_jointNm].items():
            # pd.DataFrame(v).to_csv(
            #     os.path.join(_dirPath, k+'.csv'), 
            #     index=False
            # )
            pass

def main():
    '''
    Objective:
        blend多個candidate poses
    :: 
    '''
    # 1.1 read rotation aligned 3d positions (所有joint才會構成一個pose)
    # 1.2 read FV's distances
    # 2. blend poses of each FV by corresponding weight
    # 2.1 compute global match weight for each pose
    # 3. Use EWMA with global match weight to blend poses inter-FVs
    # 4. output the blended 3d poses

    motionDirPath = 'data/swimming/'
    usedJointNm = ['lhand', 'rhand']
    # 1. 
    rotationAlignDirPath = os.path.join(motionDirPath, 'similarAligned3dPos')
    aligned3dPos = {_jointNm: {} for _jointNm in usedJointNm}
    fullPoseJoints = None
    # TODO: finish reading essensial data
    for _jointNm in usedJointNm:
        _jointPath = os.path.join(rotationAlignDirPath, _jointNm)
        _fullPoseAlignedPosPaths = os.listdir(_jointPath)
        fullPoseJoints = [i.replace('.csv', '') for i in _fullPoseAlignedPosPaths]
        _fullPoseAlignedPosPaths = [os.path.join(_jointPath, i) for i in _fullPoseAlignedPosPaths]
        # print(_fullPoseAlignedPosPaths)
        for _j, _jPath in zip(fullPoseJoints, _fullPoseAlignedPosPaths):
            aligned3dPos[_jointNm][_j] = pd.read_csv(_jPath)
    # print(aligned3dPos['lhand']['Chest'].shape)
    # 1.2 
    similarFVDisDirPath = os.path.join(motionDirPath, 'similarDist')
    similarFVdis = {
        _jointNm: pd.read_csv(os.path.join(similarFVDisDirPath, _jointNm+'.csv')).values for _jointNm in usedJointNm
    }
    print(similarFVdis['lhand'].shape)

    # 2. 
    ## TODO: normalize weights (全部一次處理)
    ## 這邊會遇到0沒有辦法轉成反比的情況 -> 我自己想的solution: 全部人都加上一個小數值10^-3
    ## 格式做成跟3d position一樣 (第二個維度變成15)
    for _jointNm in usedJointNm:
        _similarFVdis = similarFVdis[_jointNm] + 10**(-3)
        _similarFVdis = np.reciprocal(_similarFVdis)
        # ref: https://stackoverflow.com/questions/2850743/numpy-how-to-quickly-normalize-many-vectors
        _similarFVdis = _similarFVdis / np.linalg.norm(_similarFVdis, axis=1)[:, np.newaxis]
        _similarFVdis = _similarFVdis[:, [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]]
        # print(np.linalg.norm(_similarFVdis, axis=1))
        # print(np.linalg.norm(_similarFVdis, axis=1).shape)
        # print(_similarFVdis[:2, :])
        # break
    ## TODO: compute global match weight. 每一個blended pose都會有一個global match weight

    blendedPose = {}
    for _joint in fullPoseJoints:
        ## TODO:  hand, elbow, shoulder只會使用單手資料blending
        ## TODO: 確認這些joints的名稱
        if _joint == 'lhand' or _joint == 'rhand':
            pass
        ## 其餘joints使用雙手資料blending
        pass

    pass

if __name__=='__main__':
    # 需要先將similar FV的index轉換成對應的3d positions
    # convertSimilarFVIndTo3dPos(
    #     'data/swimming/similarInd/', 'data/swimming/kdtree/featVecsCount.csv', 
    #     'data/swimming/'
    # )
    # =======
    # 計算所有similat feature vectors的alignment rotation
    # computeAlignRot(
    #     motionDirPath = 'data/swimming/', 
    #     similarFVIndDirPath = 'data/swimming/similarInd/', 
    #     usedJointNm = ['lhand', 'rhand'], 
    #     windowSize = 9
    # )
    # =======
    # apply alignment rotation到所有3d positions/poses
    # applyAlignRotTo3dPos(
    #     motionDirPath = 'data/swimming/', 
    #     usedJointNm = ['lhand', 'rhand'], 
    #     numOfCandidate = 5
    # )
    # =======
    # blend candidate poses by corresponding weights
    main()
    pass