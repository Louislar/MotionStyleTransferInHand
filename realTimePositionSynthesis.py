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
from collections import deque
from positionAnalysis import positionDataPreproc, positionJsonDataParser, positionDataToPandasDf, setHipAsOrigin, rollingWindowSegRetrieve, jointsNames
from positionSynthesis import augFeatVecToPos, kSimilarFeatureVectorsBlending

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
jointsBlendingRef = {
        jointsNames.LeftUpperLeg: {jointsNames.LeftFoot: 0.9, jointsNames.LeftLowerLeg: 0.1}, 
        jointsNames.LeftLowerLeg: {jointsNames.LeftFoot: 0.9, jointsNames.LeftLowerLeg: 0.1},
        jointsNames.LeftFoot: {jointsNames.LeftFoot: 1.0}, 
        jointsNames.RightUpperLeg: {jointsNames.RightFoot: 0.9, jointsNames.RightLowerLeg: 0.1},
        jointsNames.RightLowerLeg: {jointsNames.RightFoot: 0.9, jointsNames.RightLowerLeg: 0.1}, 
        jointsNames.RightFoot: {jointsNames.RightFoot: 1.0}, 
        jointsNames.Spine: {jointsNames.LeftFoot: 0.5, jointsNames.RightFoot: 0.5},
        jointsNames.Chest: {jointsNames.LeftFoot: 0.5, jointsNames.RightFoot: 0.5},
        jointsNames.UpperChest: {jointsNames.LeftFoot: 0.5, jointsNames.RightFoot: 0.5},
        jointsNames.LeftUpperArm: {jointsNames.LeftFoot: 0.5, jointsNames.RightFoot: 0.5},
        jointsNames.LeftLowerArm: {jointsNames.LeftFoot: 0.5, jointsNames.RightFoot: 0.5},
        jointsNames.LeftHand: {jointsNames.LeftFoot: 0.5, jointsNames.RightFoot: 0.5},
        # jointsNames.LeftHand: {jointsNames.LeftFoot: 1.0},
        jointsNames.RightUpperArm: {jointsNames.LeftFoot: 0.5, jointsNames.RightFoot: 0.5},
        jointsNames.RightLowerArm: {jointsNames.LeftFoot: 0.5, jointsNames.RightFoot: 0.5},
        jointsNames.RightHand: {jointsNames.LeftFoot: 0.5, jointsNames.RightFoot: 0.5}, 
        # jointsNames.RightHand: {jointsNames.LeftFoot: 1.0}, 
        jointsNames.Head: {jointsNames.LeftFoot: 0.5, jointsNames.RightFoot: 0.5}
    }   # 第一層的key是main joint, 第二層的key是reference joints, 第二層value是reference joints之間的weight

# global variable
preLowerBodyPos = deque([], rollingWinSize)
preVel = deque([], rollingWinSize-1)
preAcc = deque([], rollingWinSize-2) 

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

def kSimilarFromKDTree(sourceData, kdtrees, k):
    '''
    Goal: 利用預先建立好的KDTree搜尋最相似的k個instances
    Input:
    :sourceData: 搜尋的參考feature vectors構成的dict, 
        key is the joint, value is a 2D array (number of source data, feature count)
    :kdtrees: 所有joint的kdtrees' list
    :k: 找前k個相似的DB feature vectors
    '''
    # print(sourceData)
    kSimilarIdx = {i: None for i in sourceData}
    kSimilarDist = {i: None for i in sourceData}
    for aJoint in sourceData:
        dist, ind = kdtrees[aJoint].query(sourceData[aJoint], k=k)
        kSimilarIdx[aJoint] = ind
        kSimilarDist[aJoint] = dist
    return kSimilarDist, kSimilarIdx

def kSimilarPoseBlendingSingleTime(DBFullJoint3DPos, refJointsKSimiarFeatVecIdx, refJointsKSimiarFeatVecDist):
    '''
    Goal: 類似positionSynthesis.py當中的kSimilarFeatureVectorsBlending()的wrapper, 
            只是, 改為輸出單一時間點所有joint的blending結果
            利用前k個相似的3D position, blend新的3D position
    Input: 
    :DBFullJoint3DPos: DB中所有joint的 3D position資訊
    :refJointsKSimiarFeatVecIdx: 所有參考點在單一時間點下的前k個相似的feature vector index, 
        內含的vector是二維的向量 (時間點數量, k)
    :refJointsKSimiarFeatVecDist: 所有參考點在單一時間點下的前k個相似的feature vector 距離
    '''
    # print(refJointsKSimiarFeatVecIdx)
    blendingRet = []
    for mainJoint in jointsBlendingRef:
        refJoints = jointsBlendingRef[mainJoint].keys()
        refJointsWeights = list(jointsBlendingRef[mainJoint].values())
        multiRefJointsResult = []
        for aRefJoint in refJoints:
            multiRefJointsResult.append(
                kSimilarFeatureVectorsBlending(
                    DBFullJoint3DPos[mainJoint], 
                    refJointsKSimiarFeatVecIdx[aRefJoint], 
                    refJointsKSimiarFeatVecDist[aRefJoint]
                )
            )
        for i in range(len(refJointsWeights)):
            multiRefJointsResult[i] = multiRefJointsResult[i]*refJointsWeights[i]
        blendingRet.append(sum(multiRefJointsResult))
    # print(blendingRet)
    return blendingRet

def EWMAForStreaming(streamPose1, streamPose2, weight):
    '''
    對streaming data做EWMA(Exponential weighted moving average)
    Input: 
    :streamPose1: 上一個時間點的Pose, 一個list, 每個element都是一個joint的3D position array
    :streamPose2: 這個時間點的Pose
    :weight: p_t = p_t*weight + p_{t-1}*(1-weight)

    Output:
    :: 兩個時間點的pose做EWMA的結果
    '''
    EWMAPose = []
    remainWeight = 1 - weight
    for i in range(len(streamPose1)):
        EWMAPose.append(streamPose1[i] * remainWeight + streamPose2[i] * weight)
    return EWMAPose

def blendingStreamResultToJson(blendStreamResult, jointCount):
    '''
    Goal: blending streaming result轉換為可以輸出成json檔案的資料結構
    Input: 
    :blendStreamResult: 多個時間點的streaming blending結果, list當中每個element
    :jointCount: blending輸出的joint數量
    '''
    timeCount = len(blendStreamResult)
    axesUsed = ['x', 'y', 'z']
    outputdata = [{'time': i, 'data': [{k: 0 for k in ['x', 'y', 'z']} for j in range(jointCount)]} for i in range(timeCount)]
    print(timeCount)
    for t in range(timeCount):
        for j in range(jointCount):
            for axisInd, axisNm in enumerate(axesUsed):
                outputdata[t]['data'][j][axisNm] = blendStreamResult[t][j][0, axisInd]
    return outputdata

def posPreprocStream(lowerBodyPos, rollingWinSize):
    '''
    Goal: 對單一時間點的lower body position資訊做preprocessing
    Input:
    :lowerBodyPos: 單一時間點的lower body position資訊
    :rollingWinSize: 計算一個feature vector使用多少時間點的position資訊

    Output: 
    :: 單一時間點preprocess好的feature vector
    '''
    print(lowerBodyPos)
    global preLowerBodyPos  # 過去時間點的lower body position資訊
    global preVel   # 過去時間點的速度
    global preAcc   # 過去時間點的加速度

    # 1. set hip as origin
    # 2. rolling window retrieve
    # (現在只有"一組data", 這步驟會變成對過去資訊queue的存儲)
    # 2.1 update position queue
    # 3. compute velocity and acceleration
    # 3.1 update velocity and accleration queue
    # 4. construct the feature vector and return

    jointsCount = len(lowerBodyPos)
    axesCat = ['x', 'y', 'z']
    # 1. 
    for aJoint in range(jointsCount):
            if aJoint != jointsNames.Hip:
                lowerBodyPos[aJoint] = \
                    lowerBodyPos[aJoint] - lowerBodyPos[jointsNames.Hip]
    lowerBodyPos[jointsNames.Hip] = np.zeros(3)
    
    # 2. 
    # 需要先更新previous position queue
    # 轉換成list or array再加入queue
    # 如果為第一筆資料, pos重複加入至滿queue, vel and acc都是加入0至滿queue
    if len(preLowerBodyPos) == 0:
        for i in range(rollingWinSize):
            preLowerBodyPos.append(lowerBodyPos)
            preVel.append({j: np.zeros(3) for j in range(jointsCount)})
            preAcc.append({j: np.zeros(3) for j in range(jointsCount)})
    else:
        preLowerBodyPos.append(lowerBodyPos)

    # 3. 
    # 之前算過的速度與加速度, 不要再重新計算, 只算最新的速度與加速度
    lastPos = preLowerBodyPos[-2]
    lastVel = preVel[-1]
    vel = {j: lowerBodyPos[j] - lastPos[j] for j in range(jointsCount)}
    acc = {j: vel[j] - lastVel[j] for j in range(jointsCount)}
    preVel.append(vel)
    preAcc.append(acc)
    print(preLowerBodyPos)
    print('=======')
    # print(preVel)
    # print('=======')
    # print(preAcc)
    # print('=======')
    # print('=======')

    # 4. 
    # conver deque to feature vector, 每個joint都有獨立的一個feature vector,
    #       最終使用list儲存所有feature vectors
    # 排列方式為 XXXXX|YYYYY|ZZZZZ
    featVec = {j: np.zeros(rollingWinSize*3+(rollingWinSize-1)*3+(rollingWinSize-2)*3) \
        for j in range(jointsCount)}
    ## position
    for t, aPos in enumerate(preLowerBodyPos):
        for j in range(jointsCount):
            featVec[j][t] = aPos[j][0]
            featVec[j][t+rollingWinSize] = aPos[j][1]
            featVec[j][t+rollingWinSize*2] = aPos[j][2]
    ## velocity
    for t, aVel in enumerate(preVel):
        for j in range(jointsCount):
            featVec[j][t+rollingWinSize*3] = aVel[j][0]
            featVec[j][t+rollingWinSize*3+(rollingWinSize-1)] = aVel[j][1]
            featVec[j][t+rollingWinSize*3+(rollingWinSize-1)*2] = aVel[j][2]
    ## acceleration
    for t, aAcc in enumerate(preAcc):
        for j in range(jointsCount):
            featVec[j][t+rollingWinSize*3+(rollingWinSize-1)*3] = aAcc[j][0]
            featVec[j][t+rollingWinSize*3+(rollingWinSize-1)*3+(rollingWinSize-2)] = aAcc[j][1]
            featVec[j][t+rollingWinSize*3+(rollingWinSize-1)*3+(rollingWinSize-2)*2] = aAcc[j][2]
    print(featVec)
    return featVec

# Read saved DB feature vectors and used it to construct KDTree
# and compute the nearest neighbor
# Measure the time consumption to estimate the pose 
# 先計算平均速度, 再計算瞬時速度
if __name__=='__main01__':
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
    saveDirPathHand = 'HandPreprocFeatVec/leftFrontKick/'
    AfterMapPreprocArr = readDBEncodedMotionsFromFile(positionsJointCount, saveDirPathHand)

    # 3. TODO: Transfer hand position data to streaming data
    # 要寫個streaming data的converter會比較好, 包含轉換過去的function


    # 4. Find similar feature vectors for each joints(lower body joints)
    startTime = time.time()
    timeCount = AfterMapPreprocArr[0].shape[0]
    print('time count: ', timeCount)
    multiJointsKSimilarDBIdx = {k: None for k in jointsInUsedToSyhthesis}
    multiJointskSimilarDBDist = {k: None for k in jointsInUsedToSyhthesis}
    for i in jointsInUsedToSyhthesis:
        kdtree = kdtrees[i]   # Default metric is the euclidean distance(l2 dist)
        dist, ind = kdtree.query(AfterMapPreprocArr[i], k=kSimilar)
        multiJointsKSimilarDBIdx[i] = ind
        multiJointskSimilarDBDist[i] = dist
        # print(ind[:30, :])
        # print(dist[:30, :])
    lapTime1 = time.time()
    print('find similar feature vec cost:', lapTime1-startTime)

    ## Streaming data version
    kSimilarSearchTimeLaps = np.zeros(timeCount)
    kSimilarDistStream = []
    kSimilarIdxStream = []
    for t in range(timeCount):
        # Single time but multiple joints data
        singleTimeData = {i: None for i in jointsInUsedToSyhthesis}
        for i in jointsInUsedToSyhthesis:
            singleTimeData[i] = AfterMapPreprocArr[i][t:t+1, :]
        # print(singleTimeData)
        kSimilarDist, kSimilarIdx = kSimilarFromKDTree(singleTimeData, kdtrees, kSimilar)
        kSimilarDistStream.append(kSimilarDist)
        kSimilarIdxStream.append(kSimilarIdx)
        # print(kSimilarDist)
        # print(kSimilarIdx)
        kSimilarSearchTimeLaps[t] = time.time()
    kSimilarSearchCost = kSimilarSearchTimeLaps[1:] - kSimilarSearchTimeLaps[:-1]
    print('k similar search avg time: ', np.mean(kSimilarSearchCost))
    print('k similar search time std: ', np.std(kSimilarSearchCost))
    print('k similar search max time cost: ', np.max(kSimilarSearchCost))
    print('k similar search min time cost: ', np.min(kSimilarSearchCost))

    # 5. Use the k similar feature vectors to construct full body pose (includes the EMWA technique)
    # 寫成讀取stream hand data形式的function，符合之後的使用情境
    print(DBPreproc3DPos[0].shape)
    synthesisTimeLaps = np.zeros(timeCount)
    blendingResultStream = []
    for t in range(timeCount):
        # blendingRetsult1 = kSimilarPoseBlendingSingleTime(
        #     DBPreproc3DPos, 
        #     {k: multiJointsKSimilarDBIdx[k][t:t+1, :] for k in multiJointsKSimilarDBIdx}, 
        #     {k: multiJointskSimilarDBDist[k][t:t+1, :] for k in multiJointskSimilarDBDist}
        # )
        blendingResult = kSimilarPoseBlendingSingleTime(
            DBPreproc3DPos,
            kSimilarIdxStream[t], 
            kSimilarDistStream[t]
        )
        blendingResultStream.append(blendingResult)
        synthesisTimeLaps[t] = time.time()
    synthesisTimeCost = synthesisTimeLaps[1:] - synthesisTimeLaps[:-1]
    print('blending avg time: ', np.mean(synthesisTimeCost))
    print('blending time std: ', np.std(synthesisTimeCost))
    print('blending max time cost: ', np.max(synthesisTimeCost))
    print('blending min time cost: ', np.min(synthesisTimeCost))
    # TODO: 把耗時的distribution可以畫一下(30Hz ~= 0.033, 60Hz ~= 0.0166, 90Hz ~= 0.011)

    # 6. EWMA
    EWMAResult = []
    EWMATimeLaps = np.zeros(timeCount)
    for i in range(len(blendingResultStream)-1):
        EWMAResult.append(
            EWMAForStreaming(blendingResultStream[i], blendingResultStream[i+1], EWMAWeight)
        )
        EWMATimeLaps[i] = time.time()
    EWMAResult.append(
        EWMAForStreaming(blendingResultStream[-1], blendingResultStream[-1], EWMAWeight)
    )
    EWMATimeLaps[-1] = time.time()
    EWMATimeCost = EWMATimeLaps[1:] - EWMATimeLaps[:-1]
    print('EWMA avg time: ', np.mean(EWMATimeCost))
    print('EWMA time std: ', np.std(EWMATimeCost))
    print('EWMA max time cost: ', np.max(EWMATimeCost))
    print('EWMA min time cost: ', np.min(EWMATimeCost))

    # 7. 將stream資料整理成與之前相同格式的計算結果(real time執行時不會做這一步)
    # 這一步的目的是為了與之前的結果比較
    blendingStreamJson = blendingStreamResultToJson(EWMAResult, len(jointsBlendingRef))
    # with open('./positionData/afterSynthesis/leftFrontKick_stream_EWMA.json', 'w') as WFile: 
    #     json.dump(blendingStreamJson, WFile)
    
# For test(streaming版本的feature vector preprocessing)
if __name__=='__main__':
    AfterMappingFileName = \
        './positionData/fromAfterMappingHand/leftFrontKickCombinations/leftFrontKick(True, False, False, False, True, True).json'
    afterMapJson = None
    with open(AfterMappingFileName, 'r') as fileIn:
        afterMapJson=json.load(fileIn)['results']
    timeCount = len(afterMapJson)

    # convert json to np array
    jointCount = len(afterMapJson[0]['data'])
    afterMapArrs = [{i: np.zeros(3) for i in range(jointCount)} for t in range(timeCount)]
    for t in range(timeCount):
        for j in range(jointCount):
            afterMapArrs[t][j][0] = afterMapJson[t]['data'][j]['x']
            afterMapArrs[t][j][1] = afterMapJson[t]['data'][j]['y']
            afterMapArrs[t][j][2] = afterMapJson[t]['data'][j]['z']

    for t in range(timeCount):
        if t==1:
            break
        posPreprocStream(afterMapArrs[t], rollingWinSize)

    # TODO: compare to the origin preprocess function's reault

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

    # 7. Hand motion也儲存成npy, 方便debug使用, 不會在testing stage使用
    AfterMappingFileName = \
        './positionData/fromAfterMappingHand/leftFrontKickCombinations/leftFrontKick(True, False, False, False, True, True).json'
    AfterMapDf = None
    with open(AfterMappingFileName, 'r') as fileIn:
        jsonStr=json.load(fileIn)
        positionsDB = positionJsonDataParser(jsonStr, positionsJointCount)
        AfterMapDf = positionDataToPandasDf(positionsDB, positionsJointCount)
    AfterMapPreproc = positionDataPreproc(AfterMapDf, positionsJointCount, rollingWinSize, False, augmentationRatio, False)
    
    saveDirPath = 'HandPreprocFeatVec/leftFrontKick/'
    storeDBEncodedMotionsToFile(AfterMapPreproc, positionsJointCount, saveDirPath)