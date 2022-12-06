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
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
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
jointsBlendingRef = {
        # jointsNames.LeftUpperLeg: {jointsNames.LeftFoot: 0.9, jointsNames.LeftLowerLeg: 0.1}, 
        # jointsNames.LeftLowerLeg: {jointsNames.LeftFoot: 0.9, jointsNames.LeftLowerLeg: 0.1},
        jointsNames.LeftUpperLeg: {jointsNames.LeftFoot: 1.0},
        jointsNames.LeftLowerLeg: {jointsNames.LeftFoot: 1.0},
        jointsNames.LeftFoot: {jointsNames.LeftFoot: 1.0}, 
        # jointsNames.RightUpperLeg: {jointsNames.RightFoot: 0.9, jointsNames.RightLowerLeg: 0.1},
        # jointsNames.RightLowerLeg: {jointsNames.RightFoot: 0.9, jointsNames.RightLowerLeg: 0.1}, 
        jointsNames.RightUpperLeg: {jointsNames.LeftFoot: 1.0},
        jointsNames.RightLowerLeg: {jointsNames.LeftFoot: 1.0},
        jointsNames.RightFoot: {jointsNames.LeftFoot: 1.0}, 
        jointsNames.Spine: {jointsNames.LeftFoot: 1.0}, 
        jointsNames.Chest: {jointsNames.LeftFoot: 1.0}, 
        jointsNames.UpperChest: {jointsNames.LeftFoot: 1.0}, 
        jointsNames.LeftUpperArm: {jointsNames.LeftFoot: 1.0}, 
        jointsNames.LeftLowerArm: {jointsNames.LeftFoot: 1.0}, 
        jointsNames.LeftHand: {jointsNames.LeftFoot: 1.0}, 
        # jointsNames.LeftHand: {jointsNames.LeftFoot: 1.0},
        jointsNames.RightUpperArm: {jointsNames.LeftFoot: 1.0}, 
        jointsNames.RightLowerArm: {jointsNames.LeftFoot: 1.0}, 
        jointsNames.RightHand: {jointsNames.LeftFoot: 1.0}, 
        # jointsNames.RightHand: {jointsNames.LeftFoot: 1.0}, 
        jointsNames.Head: {jointsNames.LeftFoot: 1.0}, 
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
    # print(lowerBodyPos)
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
    # print(preLowerBodyPos)
    # print('=======')
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
    # print(featVec)
    return featVec

# For debug 
# (畫出"after mapping的position軌跡"以及"animation的position軌跡"以及"synthesis結果的position軌跡")
if __name__=='__main01__':
    # 1.1 read animation position time series (without hip rotation)
    # 1.2 read animation position time series (with hip rotation)
    # 1.3 read after mapping position time series
    # 1.4 read after synthesis position time series
    # 1.5 read 相似的feature vector index, 以及feature vector對應的3D position
    # 2. extract specific joint's position time series for drawing 3D plot
    # 3. plot the lines in 3D space

    # 1.1 
    animationJson=None
    # with open('./positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withoutHip.json', 'r') as WFile: 
    with open('./positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withoutHip_075.json', 'r') as WFile: 
    # with open('./positionData/fromDB/genericAvatar/leftSideKickPositionFullJointsWithHead_withoutHip.json', 'r') as WFile:
    # with open('./positionData/fromDB/genericAvatar/runSprintPositionFullJointsWithHead_withoutHip.json', 'r') as WFile:
    # with open('./positionData/fromDB/genericAvatar/walkInjuredPositionFullJointsWithHead_withoutHip.json', 'r') as WFile:
        animationJson = json.load(WFile)['results']
    # 1.2
    animationHipJson=None
    with open('./positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withHip_075.json', 'r') as WFile: 
    # with open('./positionData/fromDB/leftFrontKickPositionFullJointsWithHead.json', 'r') as WFile: 
    # with open('./positionData/fromDB/genericAvatar/leftSideKickPositionFullJointsWithHead_withHip.json', 'r') as WFile: 
    # with open('./positionData/fromDB/genericAvatar/runSprintPositionFullJointsWithHead_withHip.json', 'r') as WFile: 
    # with open('./positionData/fromDB/genericAvatar/walkInjuredPositionFullJointsWithHead_withHip.json', 'r') as WFile: 
        animationHipJson = json.load(WFile)['results']
    # 1.3
    afterMappingJson=None
    # with open('./positionData/fromAfterMappingHand/leftFrontKickStreamLinearMapping/leftFrontKick(True, False, False, True, True, True).json', 'r') as WFile: 
    # with open('./positionData/fromAfterMappingHand/leftFrontKickStreamLinearMapping_TFFTTT.json', 'r') as WFile: 
    with open('./positionData/fromAfterMappingHand/newMappingMethods/leftFrontKick_quat_BSpline_TFTTTT.json', 'r') as WFile: 
    # with open('./positionData/fromAfterMappingHand/leftSideKickStreamLinearMapping_FTTFFF.json', 'r') as WFile: 
    # with open('./positionData/fromAfterMappingHand/runSprintStreamLinearMapping_TFTTFT.json', 'r') as WFile: 
    # with open('./positionData/fromAfterMappingHand/walkInjuredStreamLinearMapping_TFTTFT.json', 'r') as WFile: 
        # afterMappingJson = json.load(WFile)['results']
        afterMappingJson = json.load(WFile)
    # 1.4
    afterSynthesisJson=None
    with open('./positionData/afterSynthesis/leftFrontKickStreamLinearMapping_TFFTTT_EWMA.json', 'r') as WFile: 
    # with open('./positionData/afterSynthesis/leftSideKickStreamLinearMapping_FTTFFF_EWMA.json', 'r') as WFile: 
        afterSynthesisJson = json.load(WFile)

    # 1.5
    # saveDirPathIdx = './similarFeatVecIdx/leftFrontKickStreamLinearMapping_TFFTTT/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_withoutHip/3DPos/'
    # saveDirPathIdx = './similarFeatVecIdx/leftFrontKickStreamLinearMapping_TFFTTT_075/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_withoutHip_075/3DPos/'
    # saveDirPathIdx = './similarFeatVecIdx/leftFrontKickStreamLinearMapping_TFFTTT_075_normalized/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_withoutHip_075_normalized/3DPos/'
    # saveDirPathIdx = './similarFeatVecIdx/leftFrontKickStreamLinearMapping_TFFTTT_withHip_075_normalized/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_withHip_075_normalized/3DPos/'
    saveDirPathIdx = './similarFeatVecIdx/leftFrontKick_quat_BSpline_TFTTTT_withHip_075_normalized/'
    saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_withHip_075_quat_BSpline_normalized/3DPos/'
    # saveDirPathIdx = './similarFeatVecIdx/leftSideKickStreamLinearMapping_FTTFFF/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftSideKick_withoutHip/3DPos/'
    # saveDirPathIdx = './similarFeatVecIdx/runSprintStreamLinearMapping_TFTTFT/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/runSprint_withoutHip/3DPos/'
    # saveDirPathIdx = './similarFeatVecIdx/walkInjuredStreamLinearMapping_TFTTFT/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/walkInjured_withoutHip/3DPos/'
    similarIdx = {}
    for i in jointsInUsedToSyhthesis:
        similarIdx[i] = np.load(saveDirPathIdx+'{0}.npy'.format(i))
    DBPreproc3DPos = readDBEncodedMotionsFromFile(fullPositionsJointCount, saveDirPath3DPos)
    firstSimilarIdx = similarIdx[2][:, 0]
    print(len(firstSimilarIdx))
    print(len(DBPreproc3DPos[2]))   # TODO: 發現feature vector的數量實際上比想像中少

    # 2. 
    # TODO: 確認哪一個index代表腳, 腿, 每一個輸出資料的joint數量或許會不同
    axisKeys = ['x', 'y', 'z']
    animationPos = [[animationJson[t]['data'][2][k] for k in axisKeys] for t in range(len(animationJson))]
    animationHipPos = [[animationHipJson[t]['data'][2][k] for k in axisKeys] for t in range(len(animationHipJson))]
    afterMappingPos = [[afterMappingJson[t]['data']['2'][k] - afterMappingJson[t]['data']['6'][k] for k in axisKeys] for t in range(len(afterMappingJson))]
    afterSynthesisPos = [[afterSynthesisJson[t]['data'][2][k] for k in axisKeys] for t in range(len(afterSynthesisJson))]
    similarFeatVecPos = [DBPreproc3DPos[2][firstSimilarIdx[t]] for t in range(len(firstSimilarIdx))]
    # plot預先計算好的3d position, 他應該要與animation的position完全相同, 但是結果卻不同(there is a bug)
    # --> 發現問題出在有沒有把hip設為原點, 並且已經修正完成
    fullFeatVecPos = [DBPreproc3DPos[2][t] for t in range(len(DBPreproc3DPos[2]))] 

    # 2.0 研究一下hip的數值範圍
    # 需要將hip設為origin
    def printJointMeanStd(pos3d):
        hip_x = np.array([t[0] for t in pos3d])
        hip_y = np.array([t[1] for t in pos3d])
        hip_z = np.array([t[2] for t in pos3d])
        print('x: ', hip_x.mean(), ', ', hip_x.std())
        print('y: ', hip_y.mean(), ', ', hip_y.std())
        print('z: ', hip_z.mean(), ', ', hip_z.std())
    
    print('animation without hip')
    printJointMeanStd(animationPos)
    print('after mapping')
    printJointMeanStd(afterMappingPos)
    print('full feature vector')
    printJointMeanStd(fullFeatVecPos)
    
    # 2.1 限制取值時間點範圍, 部分時間的的資料或許會比較清楚
    # animationPos = animationPos[50:70]
    # animationHipPos = animationHipPos[50:70]
    # afterMappingPos = afterMappingPos[100:]

    # 3. 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter([d[0] for d in animationPos], [d[1] for d in animationPos], [d[2] for d in animationPos], label='animation_hip0')
    # ax.scatter([d[0] for d in animationHipPos], [d[1] for d in animationHipPos], [d[2] for d in animationHipPos], label='animation')
    ax.plot([d[0] for d in afterMappingPos], [d[1] for d in afterMappingPos], [d[2] for d in afterMappingPos], label='after_mapping', color='g')
    # ax.scatter([d[0] for d in afterSynthesisPos], [d[1] for d in afterSynthesisPos], [d[2] for d in afterSynthesisPos], label='after_synthesis')
    ax.scatter([d[0] for d in fullFeatVecPos], [d[1] for d in fullFeatVecPos], [d[2] for d in fullFeatVecPos], label='full_featVec')
    ax.scatter([d[0] for d in similarFeatVecPos], [d[1] for d in similarFeatVecPos], [d[2] for d in similarFeatVecPos], label='similar_featVec', color='r')


    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    plt.tight_layout()
    plt.legend()
    plt.show()
    
# For debug
# 使用mapping後的position找到相似的animation pose, 並且把那些pose對應到的index記錄下來
# 提供其他functcion visualize那些pose 以及 mapping後的position trajectory
if __name__=='__main01__':
    # 1. 讀取mapping後的positions轉換成的feature vectors (先在底下的區塊將其轉換成feature vectors, 並儲存成pickles讀取)
    # 2. 讀取animation position轉換成feature vectors. 再建立的kdtree資料結構
    # 3. 尋找每個時間點與finger feature vector相似的animation feature vector, 並且記錄其index(time point)
    # 4. 儲存得到的index, 在其他地方繪製成trajectory做比較

    # 1.
    # saveDirPathHand = 'HandPreprocFeatVec/leftFrontKickStreamLinearMapping_TFFTTT/'
    saveDirPathHand = 'HandPreprocFeatVec/leftFrontKick_quat_BSpline_TFTTTT/'
    # saveDirPathHand = 'HandPreprocFeatVec/leftSideKickStreamLinearMapping_FTTFFF/'
    # saveDirPathHand = 'HandPreprocFeatVec/runSprintStreamLinearMapping_TFTTFT/'
    # saveDirPathHand = 'HandPreprocFeatVec/walkInjuredStreamLinearMapping_TFTTFT/'
    AfterMapPreprocArr = readDBEncodedMotionsFromFile(positionsJointCount, saveDirPathHand)

    # 2. 
    # saveDirPath = 'DBPreprocFeatVec/leftFrontKick_withoutHip/'
    # saveDirPath = 'DBPreprocFeatVec/leftFrontKick_withoutHip_075/'
    # saveDirPath = 'DBPreprocFeatVec/leftFrontKick_withoutHip_075_transformed/'
    # saveDirPath = 'DBPreprocFeatVec/leftFrontKick_withHip_075_normalized/'
    saveDirPath = 'DBPreprocFeatVec/leftFrontKick_withHip_075_quat_BSpline_normalized/'
    # saveDirPath = 'DBPreprocFeatVec/leftSideKick_withoutHip/'
    # saveDirPath = 'DBPreprocFeatVec/runSprint_withoutHip/'
    # saveDirPath = 'DBPreprocFeatVec/walkInjured_withoutHip/'
    kdtrees = {k: None for k in jointsInUsedToSyhthesis}
    for i in jointsInUsedToSyhthesis:
        with open(saveDirPath+'{0}.pickle'.format(i), 'rb') as inPickle:
            kdtrees[i] = pickle.load(inPickle)
    
    # 3. 
    timeCount = AfterMapPreprocArr[0].shape[0]
    print('time count: ', timeCount)
    multiJointsKSimilarDBIdx = {k: None for k in jointsInUsedToSyhthesis}
    multiJointskSimilarDBDist = {k: None for k in jointsInUsedToSyhthesis}
    for i in jointsInUsedToSyhthesis:
        kdtree = kdtrees[i]   # Default metric is the euclidean distance(l2 dist)
        dist, ind = kdtree.query(AfterMapPreprocArr[i], k=kSimilar)
        multiJointsKSimilarDBIdx[i] = ind
        multiJointskSimilarDBDist[i] = dist
    # print(multiJointsKSimilarDBIdx[2])
    # print(type(multiJointsKSimilarDBIdx[2]))

    # 4. 
    # saveDirPath = './similarFeatVecIdx/leftFrontKickStreamLinearMapping_TFFTTT/'
    # saveDirPath = './similarFeatVecIdx/leftFrontKickStreamLinearMapping_TFFTTT_075/'
    # saveDirPath = './similarFeatVecIdx/leftFrontKickStreamLinearMapping_TFFTTT_075_transformed/'
    # saveDirPath = './similarFeatVecIdx/leftFrontKickStreamLinearMapping_TFFTTT_075_normalized/'
    # saveDirPath = './similarFeatVecIdx/leftFrontKickStreamLinearMapping_TFFTTT_withHip_075_normalized/'
    saveDirPath = './similarFeatVecIdx/leftFrontKick_quat_BSpline_TFTTTT_withHip_075_normalized/'
    # saveDirPath = './similarFeatVecIdx/leftSideKickStreamLinearMapping_FTTFFF/'
    # saveDirPath = './similarFeatVecIdx/runSprintStreamLinearMapping_TFTTFT/'
    # saveDirPath = './similarFeatVecIdx/walkInjuredStreamLinearMapping_TFTTFT/'
    for i in jointsInUsedToSyhthesis:
        np.save(saveDirPath+'{0}.npy'.format(i), multiJointsKSimilarDBIdx[i])

# Read saved DB feature vectors and used it to construct KDTree
# and compute the nearest neighbor
# Measure the time consumption to estimate the pose 
# 先計算平均速度, 再計算瞬時速度
if __name__=='__main01__':
    # 1. Read saved DB feature vectors and load the constructed KDTree pickles
    # Also read the full body 3D positions corresponding to feature vectors
    # saveDirPath = 'DBPreprocFeatVec/leftFrontKick/'
    # saveDirPath = 'DBPreprocFeatVec/leftFrontKick_withoutHip/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick/3DPos/'
    # saveDirPath = 'DBPreprocFeatVec/leftFrontKick_withoutHip_075/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_075/3DPos/'
    saveDirPath = 'DBPreprocFeatVec/leftFrontKick_withHip_075_quat_BSpline_normalized/'
    saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_075/3DPos/'
    # saveDirPath = 'DBPreprocFeatVec/leftSideKick_withoutHip/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftSideKick/3DPos/'
    # saveDirPath = 'DBPreprocFeatVec/runSprint_withoutHip/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/runSprint/3DPos/'
    # saveDirPath = 'DBPreprocFeatVec/walkInjured_withoutHip/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/walkInjured/3DPos/'
    DBPreproc = readDBEncodedMotionsFromFile(fullPositionsJointCount, saveDirPath)
    DBPreproc3DPos = readDBEncodedMotionsFromFile(fullPositionsJointCount, saveDirPath3DPos)
    kdtrees = {k: None for k in jointsInUsedToSyhthesis}
    for i in jointsInUsedToSyhthesis:
        with open(saveDirPath+'{0}.pickle'.format(i), 'rb') as inPickle:
            kdtrees[i] = pickle.load(inPickle)

    # 2. Read the hand position data, try to treat it as a input data stream 
    # saveDirPathHand = 'HandPreprocFeatVec/leftFrontKick/'
    # saveDirPathHand = 'HandPreprocFeatVec/leftFrontKickStreamLinearMapping_TFFTTT/'
    saveDirPathHand = 'HandPreprocFeatVec/leftFrontKick_quat_BSpline_TFTTTT/'
    # saveDirPathHand = 'HandPreprocFeatVec/leftSideKickStreamLinearMapping_FTTFFF/'
    # saveDirPathHand = 'HandPreprocFeatVec/runSprintStreamLinearMapping_TFTTFT/'
    # saveDirPathHand = 'HandPreprocFeatVec/walkInjuredStreamLinearMapping_TFTTFT/'
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
    # with open('./positionData/afterSynthesis/leftFrontKickStreamLinearMapping_TFFTTT_EWMA.json', 'w') as WFile: 
    # with open('./positionData/afterSynthesis/leftFrontKickStreamLinearMapping_TFFTTT_075_EWMA.json', 'w') as WFile: 
    with open('./positionData/afterSynthesis/leftFrontKick_quat_BSpline_TFTTTT_075_EWMA.json', 'w') as WFile: 
    # with open('./positionData/afterSynthesis/leftSideKickStreamLinearMapping_FTTFFF_EWMA.json', 'w') as WFile: 
    # with open('./positionData/afterSynthesis/runSprintStreamLinearMapping_TFTTFT_EWMA.json', 'w') as WFile: 
    # with open('./positionData/afterSynthesis/walkInjuredStreamLinearMapping_TFTTFT_EWMA.json', 'w') as WFile: 
        json.dump(blendingStreamJson, WFile)
    
# For test(streaming版本的feature vector preprocessing)
if __name__=='__main01__':
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

    print(afterMapArrs[1])
    featVecsStream = []
    for t in range(timeCount):
        featureVec = posPreprocStream(afterMapArrs[t], rollingWinSize)
        featVecsStream.append(featureVec)

    # compare to the origin preprocess function's reault
    saveDirPathHand = 'HandPreprocFeatVec/leftFrontKick/'
    AfterMapPreprocArr = readDBEncodedMotionsFromFile(positionsJointCount, saveDirPathHand)
    # print(AfterMapPreprocArr[1])
    # print(AfterMapPreprocArr[1].shape)

    ## plot two results
    plt.figure()
    plt.plot(range(AfterMapPreprocArr[1].shape[0]), AfterMapPreprocArr[1][:, 35], label='old')
    plt.plot(range(len(featVecsStream)), [i[1][35] for i in featVecsStream], label='new')
    plt.legend()
    plt.show()

# Encode and save DB motions' feature vectors to file
# Save used joints' KDTree into file
# Save DB motions' positions corresponding to feature vectors to file
# (紀錄每一個feature vector對應的3D position, 加速synthesis過程)
if __name__=='__main01__':
    # 1. 讀取DB motion
    # DBFileName = './positionData/fromDB/leftFrontKickPositionFullJointsWithHead.json'
    # DBFileName = './positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withoutHip.json'
    # DBFileName = './positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withoutHip_075.json'
    # DBFileName = './positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withoutHip_075_transformed.json'
    # DBFileName = './positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withoutHip_075_normalized.json'
    # DBFileName = './positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withHip_075_normalized.json'
    DBFileName = './positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withHip_075_quat_BSpline_normalized.json'
    # DBFileName = './positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withHip.json'
    # DBFileName = './positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withHip_075.json'
    # DBFileName = './positionData/fromDB/genericAvatar/leftSideKickPositionFullJointsWithHead_withoutHip.json'
    # DBFileName = './positionData/fromDB/genericAvatar/leftSideKickPositionFullJointsWithHead_withHip.json'
    # DBFileName = './positionData/fromDB/leftSideKickPositionFullJointsWithHead.json'
    # DBFileName = './positionData/fromDB/genericAvatar/runSprintPositionFullJointsWithHead_withoutHip.json'
    # DBFileName = './positionData/fromDB/genericAvatar/runSprintPositionFullJointsWithHead_withHip.json'
    # DBFileName = './positionData/fromDB/runSprintPositionFullJointsWithHead.json'
    # DBFileName = './positionData/fromDB/genericAvatar/walkInjuredPositionFullJointsWithHead_withoutHip.json'
    # DBFileName = './positionData/fromDB/genericAvatar/walkInjuredPositionFullJointsWithHead_withHip.json'
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
    # saveDirPath = 'DBPreprocFeatVec/leftFrontKick_withoutHip/'
    # saveDirPath = 'DBPreprocFeatVec/leftFrontKick_withoutHip_075/'
    # saveDirPath = 'DBPreprocFeatVec/leftFrontKick/'
    # saveDirPath = 'DBPreprocFeatVec/leftFrontKick_075/'
    # saveDirPath = 'DBPreprocFeatVec/leftFrontKick_withoutHip_075_normalized/'
    # saveDirPath = 'DBPreprocFeatVec/leftFrontKick_withHip_075_normalized/'
    saveDirPath = 'DBPreprocFeatVec/leftFrontKick_withHip_075_quat_BSpline_normalized/'
    # saveDirPath = 'DBPreprocFeatVec/leftSideKick_withoutHip/'
    # saveDirPath = 'DBPreprocFeatVec/leftSideKick/'
    # saveDirPath = 'DBPreprocFeatVec/runSprint/'
    # saveDirPath = 'DBPreprocFeatVec/runSprint_withoutHip/'
    # saveDirPath = 'DBPreprocFeatVec/walkInjured_withoutHip/'
    # saveDirPath = 'DBPreprocFeatVec/walkInjured/'
    storeDBEncodedMotionsToFile(DBPreproc, fullPositionsJointCount, saveDirPath)

    # 3.1 Store 3D positions corresponding to the feature vectors to file
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick/3DPos/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_075/3DPos/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_withoutHip_075_normalized/3DPos/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_withHip_075_normalized/3DPos/'
    saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_withHip_075_quat_BSpline_normalized/3DPos/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_withoutHip/3DPos/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_withoutHip_075/3DPos/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftSideKick_withoutHip/3DPos/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftSideKick/3DPos/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/runSprint/3DPos/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/runSprint_withoutHip/3DPos/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/walkInjured_withoutHip/3DPos/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/walkInjured/3DPos/'
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
    #   TODO: 這邊最好修改成使用streaming版本的feature vector preprocessing方法. 
    #           估計的結果才會是testing stage時會看到的結果. 
    # AfterMappingFileName = \
    #     './positionData/fromAfterMappingHand/leftFrontKickCombinations/leftFrontKick(True, False, False, False, True, True).json'
    # AfterMappingFileName = \
    #     './positionData/fromAfterMappingHand/leftFrontKickStream.json'
    # AfterMappingFileName = \
    #     './positionData/fromAfterMappingHand/leftFrontKickStreamLinearMapping_TFFTTT.json'
    AfterMappingFileName = \
        './positionData/fromAfterMappingHand/newMappingMethods/leftFrontKick_quat_BSpline_TFTTTT.json'
    # AfterMappingFileName = \
    #     './positionData/fromAfterMappingHand/leftSideKickStreamLinearMapping_FTTFFF.json'
    # AfterMappingFileName = \
    #     './positionData/fromAfterMappingHand/runSprintStreamLinearMapping_TFTTFT.json'
    # AfterMappingFileName = \
    #     './positionData/fromAfterMappingHand/walkInjuredStreamLinearMapping_TFTTFT.json'
    AfterMapDf = None
    with open(AfterMappingFileName, 'r') as fileIn:
        jsonStr=json.load(fileIn)   
        jsonStr = \
            [{'time': i['time'], 'data': {int(k): i['data'][k] for k in i['data']}} for i in jsonStr] # For python output
        jsonStr = {'results':jsonStr}   # for python output
        positionsDB = positionJsonDataParser(jsonStr, positionsJointCount)
        AfterMapDf = positionDataToPandasDf(positionsDB, positionsJointCount)
    AfterMapPreproc = positionDataPreproc(AfterMapDf, positionsJointCount, rollingWinSize, False, augmentationRatio, False)
    
    # saveDirPath = 'HandPreprocFeatVec/leftFrontKick/'
    # saveDirPath = 'HandPreprocFeatVec/leftFrontKickStreamLinearMapping_TFFTTT/'
    saveDirPath = 'HandPreprocFeatVec/leftFrontKick_quat_BSpline_TFTTTT/'
    # saveDirPath = 'HandPreprocFeatVec/leftSideKickStreamLinearMapping_FTTFFF/'
    # saveDirPath = 'HandPreprocFeatVec/runSprintStreamLinearMapping_TFTTFT/'
    # saveDirPath = 'HandPreprocFeatVec/walkInjuredStreamLinearMapping_TFTTFT/'
    storeDBEncodedMotionsToFile(AfterMapPreproc, positionsJointCount, saveDirPath)