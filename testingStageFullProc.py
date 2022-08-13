'''
Goal: 從手勢偵測一路到最終的position synthesis在real time的狀態下執行, 
        將整個process包裝成一個function提供呼叫使用, 
        輸入為hand landmark streaming data(source可以是讀取檔案或是real time detect), 
        以及precomputed: 
            1. Rotation mapping function(.pickle)
            2. T pose position data(.pickle)
            3. Preprocessed DB motions, and store in KDTree structure(.pickle)

        輸出為avatar full body joints position streaming data(不包含存檔, 僅僅是輸出streaming data)
Process: 
    1. Read in hand landmarks streaming data
    2. Compute the hand joints' rotations
    3. Mapped the computed hand joints' rotations
    4. Apply the mapped rotations to the avatar, get lower body positions
    5. Use the lower body positions to synthesis full body positions
'''

import numpy as np 
import json
import pickle
import time
import matplotlib.pyplot as plt 
from positionAnalysis import jointsNames
from realTimeHandRotationCompute import negateAxes, heightWidthCorrection, kalmanFilter, \
        computeUsedVectors, computeUsedRotations, negateXYZMask, \
        kalmanParamQ, kalmanParamR, kalmanX, kalmanK, kalmanP
from realTimeHandRotationCompute import jointsNames as handJointsNames
from realTimeRotationMapping import rotationMappingStream,tmpRotations
from realTimeRotToAvatarPos import forwardKinematic, loadTPosePosAndVecs, upperLegXRotAdj, leftUpperLegZRotAdj, \
        usedLowerBodyJoints
from realTimePositionSynthesis import posPreprocStream, preLowerBodyPos, preVel, preAcc, \
    rollingWinSize, readDBEncodedMotionsFromFile, jointsInUsedToSyhthesis, fullPositionsJointCount, \
        kSimilarFromKDTree, kSimilarPoseBlendingSingleTime, EWMAForStreaming

handLandMarkFilePath = 'complexModel/frontKick.json'
rotationMappingFuncFilePath = 'preprocBSpline/leftFrontKick/'
usedJointIdx = [['x','z'], ['x'], ['x','z'], ['x']]
usedJointIdx1 = [(i,j) for i in range(len(usedJointIdx)) for j in usedJointIdx[i]]  
mappingStrategy = [['x'], [], ['z'], ['x']]  # 設計的跟usedJointIdx相同即可, 缺一些element而已
TPosePosDataFilePath = 'TPoseInfo/'
DBMotionKDTreeFilePath = 'DBPreprocFeatVec/leftFrontKick/'
DBMotion3DPosFilePath = 'DBPreprocFeatVec/leftFrontKick/3DPos/'
ksimilar = 5
EWMAWeight = 0.7

# Global variables
preBlendResult = None

def testingStage(
    handLandMark, 
    mappingfunction, mappingStrategy, 
    TPoseLeftKinematic, TPoseRightKinematic, TPosePositions, 
    kdtrees, DBMotion3DPos, ksimilar, EWMAweight
):
    '''
    Goal: 
    Input:
    :handLandMark: 單一時間點下, 偵測到的hand landmarks, 
        共21個joints
    :mappingFunction: 預先計算好的BSpline sample points
    :mappingStrategy: 決定哪一些joint's axis需要被map

    Output: 
    '''
    # 1. hand rotation compute
    # print(handLandMark)
    usedJoints = [
        handJointsNames.wrist, 
        handJointsNames.indexMCP, handJointsNames.indexPIP, handJointsNames.indexDIP, 
        handJointsNames.middleMCP, handJointsNames.middlePIP, handJointsNames.middleDIP
    ]
    # 1.1 hand landmarks preprocessing
    handLandMark = negateAxes(handLandMark, negateXYZMask, usedJoints)
    handLandMark = heightWidthCorrection(handLandMark, usedJoints, 848, 480)
    handLandMark = kalmanFilter(handLandMark, usedJoints)
    # 1.2 hand rotation compute
    usedVecs = computeUsedVectors(handLandMark)
    handRotations = computeUsedRotations(*usedVecs)
    # print(handRotations)
    # return handRotations

    # 2. hand rotation mapping
    mappedHandRotations = [{aAxis: None for aAxis in i} for i in usedJointIdx]
    # 2.1 set max rotation to 180, min rotation to -180
    handRotations = [r-360 if r>180 else r for r in handRotations]
    # 2.2 estimate increase or decrease segment and map the hand rotation    
    for i, _tuple in enumerate(usedJointIdx1):
        mappedHandRotations[_tuple[0]][_tuple[1]] = handRotations[i]
    # print(mappedHandRotations)
    mappedHandRotations = rotationMappingStream(mappedHandRotations, mappingfunction, mappingStrategy)
    # print(mappedHandRotations)
    # return mappedHandRotations
    
    # 3. apply mapped rotation to avatar
    lowerBodyPositions = {aJoint: None for aJoint in usedLowerBodyJoints} 
    # 3.1 left kinematic
    leftKinematicNew = forwardKinematic(
        TPoseLeftKinematic, 
        [
            mappedHandRotations[0]['x']+upperLegXRotAdj, 
            mappedHandRotations[0]['z']+leftUpperLegZRotAdj, 
            mappedHandRotations[1]['x']
        ]
    )
    lowerBodyPositions[jointsNames.LeftLowerLeg] = leftKinematicNew[0] + leftKinematicNew[1]
    lowerBodyPositions[jointsNames.LeftFoot] = leftKinematicNew[0] + leftKinematicNew[1] + leftKinematicNew[2]
    # 3.2 right kinematic
    rightKinematicNew = forwardKinematic(
            rightKinematic, 
            [
                mappedHandRotations[2]['x']+upperLegXRotAdj, 
                mappedHandRotations[2]['z'], 
                mappedHandRotations[3]['x']
            ]
        )
    lowerBodyPositions[jointsNames.RightLowerLeg] = rightKinematicNew[0] + rightKinematicNew[1]
    lowerBodyPositions[jointsNames.RightFoot] = rightKinematicNew[0] + rightKinematicNew[1] + rightKinematicNew[2]

    lowerBodyPositions[jointsNames.Hip] = TPosePositions[jointsNames.Hip]
    lowerBodyPositions[jointsNames.LeftUpperLeg] = TPosePositions[jointsNames.LeftUpperLeg]
    lowerBodyPositions[jointsNames.RightUpperLeg] = TPosePositions[jointsNames.RightUpperLeg]
    # print(lowerBodyPositions)
    # return lowerBodyPositions

    # 4. motion synthesis/blending
    # 4.1 hand vector preprocessing
    # streaming版本的feature vector preprocessing, 
    #       寫在realTimePositionSynthesis當中
    handFeatVec = posPreprocStream(lowerBodyPositions, rollingWinSize)
    # print(handFeatVec)
    # return handFeatVec
    # 4.2 find similar feature vector for each joint
    # 要將array改成2D array
    for k in handFeatVec:
        handFeatVec[k] = handFeatVec[k][np.newaxis, :]
    kSimilarDist, kSimilarIdx = kSimilarFromKDTree(
        {k:handFeatVec[k] for k in jointsInUsedToSyhthesis}, 
        kdtrees, 
        ksimilar
    )
    # 4.3 use k similar feature vector to construct full body pose
    blendingResult = kSimilarPoseBlendingSingleTime(DBMotion3DPos, kSimilarIdx, kSimilarDist)
    # print(blendingResult)
    # 4.4 EWMA
    global preBlendResult
    if preBlendResult is None:
        preBlendResult = blendingResult
    for i in range(len(blendingResult)):
        blendingResult[i] = blendingResult[i]*EWMAWeight + preBlendResult[i]*(1-EWMAweight)
    preBlendResult = blendingResult
    return blendingResult

# For test the process
if __name__=='__main01__':
    
    # 讀取hand landmark data(假裝是streaming data輸入)
    handLMJson = None
    with open(handLandMarkFilePath, 'r') as fileOpen: 
        handLMJson=json.load(fileOpen)

    # 讀取pre computed mapping function, 也就是BSpline的sample points
    BSplineSamplePoints = [
        [{aAxis: None for aAxis in usedJointIdx[aJoint]} for aJoint in range(len(usedJointIdx))], 
        [{aAxis: None for aAxis in usedJointIdx[aJoint]} for aJoint in range(len(usedJointIdx))]
    ]
    for aJoint in range(len(usedJointIdx)):
        for aAxis in usedJointIdx[aJoint]:
            for i in range(2):
                BSplineSamplePoints[i][aJoint][aAxis] = \
                    np.load(rotationMappingFuncFilePath+'{0}.npy'.format(str(i)+'_'+aAxis+'_'+str(aJoint)))

    # 讀取T pose position以及vectors, 計算left and right kinematics
    TPosePositions, TPoseVectors  = loadTPosePosAndVecs(TPosePosDataFilePath)
    leftKinematic = [
        TPosePositions[jointsNames.LeftUpperLeg], 
        TPoseVectors[0], 
        TPoseVectors[1]
    ]  # upper leg position, upper leg vector, lower leg vector
    rightKinematic = [
        TPosePositions[jointsNames.RightUpperLeg], 
        TPoseVectors[2], 
        TPoseVectors[3]
    ]
    # 讀取預先建立的KDTree, 當中儲存DB motion feature vectors
    # 讀取與feature vector相對應的3D positions
    DBPreproc3DPos = readDBEncodedMotionsFromFile(fullPositionsJointCount, DBMotion3DPosFilePath)
    kdtrees = {k: None for k in jointsInUsedToSyhthesis}
    for i in jointsInUsedToSyhthesis:
        with open(DBMotionKDTreeFilePath+'{0}.pickle'.format(i), 'rb') as inPickle:
            kdtrees[i] = kdtree = pickle.load(inPickle)

    # testing stage full processes
    timeCount = len(handLMJson)

    testingStageResult = []
    fullTimeLaps = np.zeros(timeCount)
    for t in range(timeCount):
        result = testingStage(
            handLMJson[t]['data'], 
            BSplineSamplePoints, mappingStrategy, 
            leftKinematic, rightKinematic, TPosePositions, 
            kdtrees, DBPreproc3DPos, ksimilar, EWMAWeight
        )
        testingStageResult.append(result)
        fullTimeLaps[t] = time.time()

    fullTimeCost = fullTimeLaps[1:] - fullTimeLaps[:-1]
    print('full avg time: ', np.mean(fullTimeCost))
    print('full time std: ', np.std(fullTimeCost))
    print('full max time cost: ', np.max(fullTimeCost))
    print('full min time cost: ', np.min(fullTimeCost))

    # TODO: compare with old method's result
    # Old result comes from realTimePositionSynthesis.py
    
    plt.figure()
    # computed hand rotation, 
    # find the bug, the output of computeUsedRotation() is not straintforward
    # rotComputeRetSaveDirPath = 'HandRotationOuputFromHomePC/'
    # handRot = None
    # with open(rotComputeRetSaveDirPath+'leftFrontKickStream.json', 'r') as WFile: 
    #     handRot = json.load(WFile)
    # print(testingStageResult[0])
    # plt.plot(range(len(handRot)), [i['data'][0]['x'] for i in handRot], label='old')
    # plt.plot(range(len(testingStageResult)), [i[0] for i in testingStageResult], label='new')

    # rotation mapping result, huge difference(修正後相同)
    # 這邊已經使用real time的結果來比較, 理論上結果要相似
    # TODO: 繼續驗證下面的部分結果是否相同
    # rotMapRetSaveDirPath = 'handRotaionAfterMapping/'
    # rotMapResult = None
    # with open(rotMapRetSaveDirPath+'leftFrontKickStreamTFFFTT.json', 'r') as WFile: 
    #     rotMapResult = json.load(WFile)
    # plt.plot(range(len(rotMapResult)), [i['data'][0]['x'] for i in rotMapResult], label='old')
    # plt.plot(range(len(testingStageResult)), [i[0]['x'] for i in testingStageResult], label='new')

    # rotation output apply to avatar result, huge difference(修正後相同)
    # rotApplySaveDirPath='positionData/fromAfterMappingHand/'
    # lowerBodyPosition=None
    # with open(rotApplySaveDirPath+'leftFrontKickStream.json', 'r') as WFile: 
    #     lowerBodyPosition=json.load(WFile)
    # plt.plot(range(len(lowerBodyPosition)), [i['data']['2']['y'] for i in lowerBodyPosition], label='old')
    # plt.plot(range(len(testingStageResult)), [i[2][1] for i in testingStageResult], label='new')
    
    # after position preprocessing, the different is huge that cannot be neglect(修正後相同, 有些微項位上的不同)
    # saveDirPathHand = 'HandPreprocFeatVec/leftFrontKick/'
    # AfterMapPreprocArr = readDBEncodedMotionsFromFile(7, saveDirPathHand)
    # plt.plot(range(AfterMapPreprocArr[1].shape[0]), AfterMapPreprocArr[1][:, 30], label='old')
    # plt.plot(range(len(testingStageResult)), [i[1][30] for i in testingStageResult], label='new')
    plt.legend()
    plt.show()
    


# Execute the full process
if __name__=='__main01__':
    # 讀取hand landmark data(假裝是streaming data輸入)
    handLMJson = None
    with open(handLandMarkFilePath, 'r') as fileOpen: 
        handLMJson=json.load(fileOpen)

    # 讀取pre computed mapping function, 也就是BSpline的sample points
    BSplineSamplePoints = [
        [{aAxis: None for aAxis in usedJointIdx[aJoint]} for aJoint in range(len(usedJointIdx))], 
        [{aAxis: None for aAxis in usedJointIdx[aJoint]} for aJoint in range(len(usedJointIdx))]
    ]
    for aJoint in range(len(usedJointIdx)):
        for aAxis in usedJointIdx[aJoint]:
            for i in range(2):
                BSplineSamplePoints[i][aJoint][aAxis] = \
                    np.load(rotationMappingFuncFilePath+'{0}.npy'.format(str(i)+'_'+aAxis+'_'+str(aJoint)))

    # 讀取T pose position以及vectors, 計算left and right kinematics
    TPosePositions, TPoseVectors  = loadTPosePosAndVecs(TPosePosDataFilePath)
    leftKinematic = [
        TPosePositions[jointsNames.LeftUpperLeg], 
        TPoseVectors[0], 
        TPoseVectors[1]
    ]  # upper leg position, upper leg vector, lower leg vector
    rightKinematic = [
        TPosePositions[jointsNames.RightUpperLeg], 
        TPoseVectors[2], 
        TPoseVectors[3]
    ]
    # 讀取預先建立的KDTree, 當中儲存DB motion feature vectors
    # 讀取與feature vector相對應的3D positions
    DBPreproc3DPos = readDBEncodedMotionsFromFile(fullPositionsJointCount, DBMotion3DPosFilePath)
    kdtrees = {k: None for k in jointsInUsedToSyhthesis}
    for i in jointsInUsedToSyhthesis:
        with open(DBMotionKDTreeFilePath+'{0}.pickle'.format(i), 'rb') as inPickle:
            kdtrees[i] = kdtree = pickle.load(inPickle)

    # testing stage full processes
    timeCount = len(handLMJson)

    testingStageResult = []
    fullTimeLaps = np.zeros(timeCount)
    for t in range(timeCount):
        result = testingStage(
            handLMJson[t]['data'], 
            BSplineSamplePoints, mappingStrategy, 
            leftKinematic, rightKinematic, TPosePositions, 
            kdtrees, DBPreproc3DPos, ksimilar, EWMAWeight
        )
        testingStageResult.append(result)
        fullTimeLaps[t] = time.time()

    fullTimeCost = fullTimeLaps[1:] - fullTimeLaps[:-1]
    print('full avg time: ', np.mean(fullTimeCost))
    print('full time std: ', np.std(fullTimeCost))
    print('full max time cost: ', np.max(fullTimeCost))
    print('full min time cost: ', np.min(fullTimeCost))

    # TODO: 耗時使用15, 30, 60, 90Hz分別統計, 可能用relative frequency
    # 15Hz = 0.0666s, 30Hz = 0.0333, 60Hz = 0.0166, 90Hz = 0.0111
    hzCount = np.array([
        np.sum(fullTimeCost>=0.0666), 
        np.sum((fullTimeCost>=0.0333) & (fullTimeCost<=0.0666)), 
        np.sum((fullTimeCost>=0.0166) & (fullTimeCost<=0.0333)), 
        np.sum((fullTimeCost>=0.0111) & (fullTimeCost<=0.0166)), 
        np.sum(fullTimeCost<=0.0111)
    ])
    hzRel = hzCount/np.sum(hzCount)
    print('below 15Hz count: ', hzCount[0], ', ', hzRel[0])
    print('15 to 30Hz count: ', hzCount[1], ', ', hzRel[1])
    print('30 to 60Hz count: ', hzCount[2], ', ', hzRel[2])
    print('60 to 90Hz count: ', hzCount[3], ', ', hzRel[3])
    print('greater 90Hz count: ', hzCount[4], ', ', hzRel[4])

    # compare with old method's result
    # Old result comes from realTimePositionSynthesis.py
    oldBlendResult = None
    with open('./positionData/afterSynthesis/leftFrontKick_stream_EWMA.json', 'r') as WFile: 
        oldBlendResult = json.load(WFile)
    # print(testingStageResult[0])
    plt.figure()
    plt.plot(range(len(oldBlendResult)), [i['data'][1]['y'] for i in oldBlendResult], label='old')
    plt.plot(range(len(testingStageResult)), [i[1][0, 1] for i in testingStageResult], label='new')
    plt.legend()
    plt.show()
    
# 串聯真實streaming data的輸入, 使用webcam加上mediaPipe
if __name__=='__main__':
    # 讀取pre computed mapping function, 也就是BSpline的sample points
    BSplineSamplePoints = [
        [{aAxis: None for aAxis in usedJointIdx[aJoint]} for aJoint in range(len(usedJointIdx))], 
        [{aAxis: None for aAxis in usedJointIdx[aJoint]} for aJoint in range(len(usedJointIdx))]
    ]
    for aJoint in range(len(usedJointIdx)):
        for aAxis in usedJointIdx[aJoint]:
            for i in range(2):
                BSplineSamplePoints[i][aJoint][aAxis] = \
                    np.load(rotationMappingFuncFilePath+'{0}.npy'.format(str(i)+'_'+aAxis+'_'+str(aJoint)))

    # 讀取T pose position以及vectors, 計算left and right kinematics
    TPosePositions, TPoseVectors  = loadTPosePosAndVecs(TPosePosDataFilePath)
    leftKinematic = [
        TPosePositions[jointsNames.LeftUpperLeg], 
        TPoseVectors[0], 
        TPoseVectors[1]
    ]  # upper leg position, upper leg vector, lower leg vector
    rightKinematic = [
        TPosePositions[jointsNames.RightUpperLeg], 
        TPoseVectors[2], 
        TPoseVectors[3]
    ]
    # 讀取預先建立的KDTree, 當中儲存DB motion feature vectors
    # 讀取與feature vector相對應的3D positions
    DBPreproc3DPos = readDBEncodedMotionsFromFile(fullPositionsJointCount, DBMotion3DPosFilePath)
    kdtrees = {k: None for k in jointsInUsedToSyhthesis}
    for i in jointsInUsedToSyhthesis:
        with open(DBMotionKDTreeFilePath+'{0}.pickle'.format(i), 'rb') as inPickle:
            kdtrees[i] = kdtree = pickle.load(inPickle)

    # Streaming data
    from HandGestureMediaPipe import captureByMediaPipe
    captureByMediaPipe(
        0, 
        # 這個function call會把一些需要預先填入的database資訊放入, 
        # 只需要再輸入streaming data即可預測avatar position
        lambda streamData: testingStage(
            streamData, 
            BSplineSamplePoints, mappingStrategy, 
            leftKinematic, rightKinematic, TPosePositions, 
            kdtrees, DBPreproc3DPos, ksimilar, EWMAWeight
        )
    )
    pass