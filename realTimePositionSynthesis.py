'''
Goal: 
1. Implement real time 搜尋相似motion in DataBase, 
使用KDTree加速搜尋過程. 
2. 將DB Motions 計算好後儲存成檔案, 
未來做搜尋時, 直接讀取檔案加速搜尋過程. 
'''

import numpy as np 
import json
from sklearn.neighbors import KDTree
import pickle
import time
from positionAnalysis import positionDataPreproc, positionJsonDataParser, positionDataToPandasDf, setHipAsOrigin, rollingWindowSegRetrieve, jointsNames
from positionSynthesis import augFeatVecToPos

positionsJointCount = 7 # 用於比對motion similarity的joint數量(Upper leg*2, knee*2, foot*2, hip)
fullPositionsJointCount = 17    # 用於做motion synthesis的joint數量
rollingWinSize = 10
kSimilar = 5
# kSimilar = 1
augmentationRatio = [0.5, 0.7, 1, 1.3, 1.5]
EWMAWeight = 0.7
jointsInUsedToSyhthesis = [
    jointsNames.LeftLowerLeg, jointsNames.LeftFoot, jointsNames.RightLowerLeg, jointsNames.RightFoot
]

def storeDBEncodedMotionsToFile(DBPreprocResult, jointCount: int, saveDir: str):
    '''
    將DB motions encode成feature vectors後, 儲存成檔案
    '''
    for i in range(jointCount):
        np.save(saveDir+'{0}.npy'.format(i), DBPreprocResult[i].values)
    

def readDBEncodedMotionsFromFile(jointCount: int, saveDir: str):
    '''
    從檔案當中讀取DB motions encode後的feature vectors
    '''
    featureVecs = []
    for i in range(jointCount):
        featureVecs.append(
            np.load(saveDir+'{0}.npy'.format(i))
        )
    return featureVecs

# Read saved DB feature vectors and used it to construct KDTree
# and compute the nearest neighbor
# Measure the time consumption to estimate the pose 
# 先計算平均速度, 再計算瞬時速度
if __name__=='__main__':
    # 1. Read saved DB feature vectors and load the constructed KDTree pickles
    # Also read the full body 3D positions corresponding to feature vectors
    saveDirPath = 'DBPreprocFeatVec/leftFrontKick/'
    saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick/3DPos/'
    DBPreproc = readDBEncodedMotionsFromFile(fullPositionsJointCount, saveDirPath)
    DBPreproc3DPos = readDBEncodedMotionsFromFile(fullPositionsJointCount, saveDirPath3DPos)
    kdtrees = {k: None for k in jointsInUsedToSyhthesis}
    for i in jointsInUsedToSyhthesis:
        with open(saveDirPath+'{0}.pickle'.format(i), 'rb') as inPickle:
            kdtrees[i] = kdtree = pickle.load(inPickle)

    # 2. Read the hand position data, try to treat it as a input data stream 
    AfterMappingFileName = \
        './positionData/fromAfterMappingHand/leftFrontKickCombinations/leftFrontKick(True, False, False, False, True, True).json'
    AfterMapDf = None
    with open(AfterMappingFileName, 'r') as fileIn:
        jsonStr=json.load(fileIn)
        positionsDB = positionJsonDataParser(jsonStr, positionsJointCount)
        AfterMapDf = positionDataToPandasDf(positionsDB, positionsJointCount)
    AfterMapPreproc = positionDataPreproc(AfterMapDf, positionsJointCount, rollingWinSize, False, augmentationRatio, False)
    AfterMapPreprocArr = [_df.values for _df in AfterMapPreproc]

    # 4. Find similar feature vectors for each joints(lower body joints)
    # TODO: 把hand position當作streaming data搜尋similar DB feature vectors
    # TODO: 將這邊的指令改成function, 並且能夠執行單一的hand輸入(很容易, 單純的把query後面的輸入提出即可)
    # TODO: 計算所需時間
    multiJointsKSimilarDBIdx = {k: None for k in jointsInUsedToSyhthesis}
    multiJointskSimilarDBDist = {k: None for k in jointsInUsedToSyhthesis}
    for i in jointsInUsedToSyhthesis:
        kdtree = kdtrees[i]   # Default metric is the euclidean distance(l2 dist)
        dist, ind = kdtree.query(AfterMapPreprocArr[i], k=kSimilar)
        multiJointsKSimilarDBIdx[i] = ind
        multiJointskSimilarDBDist[i] = dist
        print(ind[:30, :])
        # print(dist[:30, :])

    # 5. Use the k similar feature vectors to construct full body pose (includes the EMWA technique)
    pass

# Encode and save DB motions' feature vectors to file
# Save used joints' KDTree into file
# Save DB motions' positions corresponding to feature vectors to file
# (紀錄每一個feature vector對應的3D position, 加速synthesis過程)
if __name__=='__main01__':
    # 1. 讀取DB motion
    DBFileName = './positionData/fromDB/leftFrontKickPositionFullJointsWithHead.json'
    posDBDf = None
    with open(DBFileName, 'r') as fileIn:
        jsonStr=json.load(fileIn)
        positionsDB = positionJsonDataParser(jsonStr, fullPositionsJointCount)
        posDBDf = positionDataToPandasDf(positionsDB, fullPositionsJointCount)

    # 2. DB motion preprocessing and encode to feature vectors
    DBPreproc = positionDataPreproc(posDBDf, fullPositionsJointCount, rollingWinSize, True, augmentationRatio)
    print(len(DBPreproc))
    print(DBPreproc[0].shape)

    # 3. Store feature vectors to files
    saveDirPath = 'DBPreprocFeatVec/leftFrontKick/'
    storeDBEncodedMotionsToFile(DBPreproc, fullPositionsJointCount, saveDirPath)

    # 3.1 Store 3D positions corresponding to the feature vectors to file
    saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick/3DPos/'
    DBPosNoAug = [augFeatVecToPos(i.values, rollingWinSize) for i in DBPreproc]
    for i in range(fullPositionsJointCount):
        np.save(saveDirPath3DPos+'{0}.npy'.format(i), DBPosNoAug[i])

    # 4. Check if the saved file is still the same by reading and compare it with the origin
    # tmpDBPreproc = readDBEncodedMotionsFromFile(fullPositionsJointCount, saveDirPath)
    # for i in range(fullPositionsJointCount):
    #     print(np.array_equal(DBPreproc[i].values, tmpDBPreproc[i]))

    # 5. Store Feature vector constructed KDTree to file
    for i in jointsInUsedToSyhthesis:
        kdtree = KDTree(DBPreproc[i].values)
        with open(saveDirPath+'{0}.pickle'.format(i), 'wb') as outPickle:
            pickle.dump(kdtree, outPickle)
    
    # 6. Read back the store KDTree pickle
    # for i in jointsInUsedToSyhthesis:
    #     with open(saveDirPath+'{0}.pickle'.format(i), 'rb') as inPickle:
    #         kdtree = pickle.load(inPickle)
    #     print(type(kdtree))
    