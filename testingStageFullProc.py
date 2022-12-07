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
from realTimeRotationMapping import rotationMappingStream,tmpRotations, linearRotationMappingStream
from realTimeRotToAvatarPos import forwardKinematic, loadTPosePosAndVecs, \
        usedLowerBodyJoints
from realTimePositionSynthesis import posPreprocStream, preLowerBodyPos, preVel, preAcc, \
    rollingWinSize, readDBEncodedMotionsFromFile, jointsInUsedToSyhthesis, fullPositionsJointCount, \
        kSimilarFromKDTree, kSimilarPoseBlendingSingleTime, EWMAForStreaming

# handLandMarkFilePath = 'complexModel/frontKick.json'
# rotationMappingFuncFilePath = 'preprocBSpline/leftFrontKick/'   # From realTimeRotationMapping.py
# linearMappingFuncFilePath = './preprocLinearPolyLine/leftFrontKickStream/'   # From realTimeRotationMapping.py
# usedJointIdx = [['x','z'], ['x'], ['x','z'], ['x']]
# usedJointIdx1 = [(i,j) for i in range(len(usedJointIdx)) for j in usedJointIdx[i]]  
# mappingStrategy = [['x'], [], ['z'], ['x']]  # 設計的跟usedJointIdx相同即可, 缺一些element而已
# TPosePosDataFilePath = 'TPoseInfo/' # From realTimeRotToAvatarPos.py
# DBMotionKDTreeFilePath = 'DBPreprocFeatVec/leftFrontKick/'  # From realTimePositionSynthesis.py
# DBMotion3DPosFilePath = 'DBPreprocFeatVec/leftFrontKick/3DPos/' # From realTimePositionSynthesis.py
# ksimilar = 5
# EWMAWeight = 0.7

## Front kick linear mapping
# handLandMarkFilePath = 'complexModel/frontKick.json'
# linearMappingFuncFilePath = './preprocLinearPolyLine/leftFrontKickStream/'   # From realTimeRotationMapping.py
# usedJointIdx = [['x','z'], ['x'], ['x','z'], ['x']]
# usedJointIdx1 = [(i,j) for i in range(len(usedJointIdx)) for j in usedJointIdx[i]]  
# mappingStrategy = [['x'], [], ['z'], ['x']]  # 設計的跟usedJointIdx相同即可, 缺一些element而已
# negMappingStrategy = [['z'], ['x'], ['x'], ['z']] # 因為upper leg需要修正沒有作mapping的角度, 所以把沒有mapping的旋轉軸列出
# # TPosePosDataFilePath = 'TPoseInfo/' # From realTimeRotToAvatarPos.py
# TPosePosDataFilePath = 'TPoseInfo/genericAvatar/' # From realTimeRotToAvatarPos.py
# DBMotionKDTreeFilePath = 'DBPreprocFeatVec/leftFrontKick_withoutHip/'  # From realTimePositionSynthesis.py
# DBMotion3DPosFilePath = 'DBPreprocFeatVec/leftFrontKick/3DPos/' # From realTimePositionSynthesis.py
# ksimilar = 5
# EWMAWeight = 0.7
# upperLegXAxisRotAdj = -30
# leftUpperLegZAxisRotAdj = -20

## Front kick linear mapping (new TFFTTT)
# handLandMarkFilePath = 'complexModel/frontKick.json'
# linearMappingFuncFilePath = './preprocLinearPolyLine/leftFrontKickStream/'   # From realTimeRotationMapping.py
# usedJointIdx = [['x','z'], ['x'], ['x','z'], ['x']]
# usedJointIdx1 = [(i,j) for i in range(len(usedJointIdx)) for j in usedJointIdx[i]]  
# mappingStrategy = [['x'], [], ['x', 'z'], ['x']]  # 設計的跟usedJointIdx相同即可, 缺一些element而已
# negMappingStrategy = [['z'], ['x'], [], []] # 因為upper leg需要修正沒有作mapping的角度, 所以把沒有mapping的旋轉軸列出
# # TPosePosDataFilePath = 'TPoseInfo/' # From realTimeRotToAvatarPos.py
# TPosePosDataFilePath = 'TPoseInfo/genericAvatar/' # From realTimeRotToAvatarPos.py
# DBMotionKDTreeFilePath = 'DBPreprocFeatVec/leftFrontKick_withoutHip/'  # From realTimePositionSynthesis.py
# DBMotion3DPosFilePath = 'DBPreprocFeatVec/leftFrontKick/3DPos/' # From realTimePositionSynthesis.py
# ksimilar = 5
# EWMAWeight = 0.7
# upperLegXAxisRotAdj = -30
# leftUpperLegZAxisRotAdj = -20

## Front kick linear mapping (new TFFTTT)
handLandMarkFilePath = 'complexModel/frontKick.json'
linearMappingFuncFilePath = './preprocLinearPolyLine/leftFrontKickStream/'   # From realTimeRotationMapping.py
usedJointIdx = [['x','z'], ['x'], ['x','z'], ['x']]
usedJointIdx1 = [(i,j) for i in range(len(usedJointIdx)) for j in usedJointIdx[i]]  
mappingStrategy = [['x'], [], ['x', 'z'], ['x']]  # 設計的跟usedJointIdx相同即可, 缺一些element而已
negMappingStrategy = [['z'], ['x'], [], []] # 因為upper leg需要修正沒有作mapping的角度, 所以把沒有mapping的旋轉軸列出
# TPosePosDataFilePath = 'TPoseInfo/' # From realTimeRotToAvatarPos.py
TPosePosDataFilePath = 'TPoseInfo/genericAvatar/' # From realTimeRotToAvatarPos.py
DBMotionKDTreeFilePath = 'DBPreprocFeatVec/leftFrontKick_withoutHip_075/'  # From realTimePositionSynthesis.py
DBMotion3DPosFilePath = 'DBPreprocFeatVec/leftFrontKick_075/3DPos/' # From realTimePositionSynthesis.py
ksimilar = 5
EWMAWeight = 0.7
upperLegXAxisRotAdj = -30
leftUpperLegZAxisRotAdj = -20

## Front kick quaternion B-Spline mapping (new TFTTTT)
handLandMarkFilePath = 'complexModel/frontKick.json'
usedJointIdx = [['x','z'], ['x'], ['x','z'], ['x']]
usedJointIdx1 = [(i,j) for i in range(len(usedJointIdx)) for j in usedJointIdx[i]]
mappingCategory = 0 # 0: euler linear, 1: euler B-Spline, 2: quat linear, 3: quat B-Spline
isLinearMapping = False
isQuatBSplineMapping = True
mappingStrategy = [['x'], ['x'], ['z'], ['x']]
unusedJointAxis = [['y', 'z'], ['y', 'z'], ['y'], ['y', 'z']]    # 與mappingStrategy是互補的關係
BSplineHandSPFilePath='rotationMappingQuaternionData/leftFrontKickBSpline/handNormMapSamplePts.pickle'
BSplineBodySPFilePath='rotationMappingQuaternionData/leftFrontKickBSpline/bodyNormMapSamplePts.pickle'
TPosePosDataFilePath = 'TPoseInfo/genericAvatar/' # From realTimeRotToAvatarPos.py
DBMotionKDTreeFilePath = 'DBPreprocFeatVec/leftFrontKick_withoutHip_075/'  # From realTimePositionSynthesis.py
DBMotion3DPosFilePath = 'DBPreprocFeatVec/leftFrontKick_075/3DPos/' # From realTimePositionSynthesis.py
ksimilar = 5
EWMAWeight = 0.7
upperLegXAxisRotAdj = -30
leftUpperLegZAxisRotAdj = -20


## Side kick linear mapping (new FTTFFF)
# handLandMarkFilePath = 'complexModel/leftSideKick.json'
# linearMappingFuncFilePath = './preprocLinearPolyLine/leftSideKickStream/'   # From realTimeRotationMapping.py
# usedJointIdx = [['x','z'], ['x'], ['x','z'], ['x']]
# usedJointIdx1 = [(i,j) for i in range(len(usedJointIdx)) for j in usedJointIdx[i]]  
# mappingStrategy = [['z'], ['x'], [], []]  # 設計的跟usedJointIdx相同即可, 缺一些element而已
# negMappingStrategy = [['x'], [], ['x', 'z'], ['x']] # 因為upper leg需要修正沒有作mapping的角度, 所以把沒有mapping的旋轉軸列出
# # TPosePosDataFilePath = 'TPoseInfo/' # From realTimeRotToAvatarPos.py
# TPosePosDataFilePath = 'TPoseInfo/genericAvatar/' # From realTimeRotToAvatarPos.py
# DBMotionKDTreeFilePath = 'DBPreprocFeatVec/leftSideKick_withoutHip/'  # From realTimePositionSynthesis.py
# DBMotion3DPosFilePath = 'DBPreprocFeatVec/leftSideKick/3DPos/' # From realTimePositionSynthesis.py
# ksimilar = 5
# EWMAWeight = 0.7
# upperLegXAxisRotAdj = -30
# leftUpperLegZAxisRotAdj = -20

## run sprint linear mapping (new TFTTFT)
# handLandMarkFilePath = 'complexModel/runSprint.json'
# linearMappingFuncFilePath = './preprocLinearPolyLine/runSprintStream/'   # From realTimeRotationMapping.py
# usedJointIdx = [['x','z'], ['x'], ['x','z'], ['x']]
# usedJointIdx1 = [(i,j) for i in range(len(usedJointIdx)) for j in usedJointIdx[i]]  
# mappingStrategy = [['x'], ['x'], ['x'], ['x']]  # 設計的跟usedJointIdx相同即可, 缺一些element而已
# negMappingStrategy = [['z'], [], ['z'], []] # 因為upper leg需要修正沒有作mapping的角度, 所以把沒有mapping的旋轉軸列出
# # TPosePosDataFilePath = 'TPoseInfo/' # From realTimeRotToAvatarPos.py
# TPosePosDataFilePath = 'TPoseInfo/genericAvatar/' # From realTimeRotToAvatarPos.py
# DBMotionKDTreeFilePath = 'DBPreprocFeatVec/runSprint_withoutHip/'  # From realTimePositionSynthesis.py
# DBMotion3DPosFilePath = 'DBPreprocFeatVec/runSprint/3DPos/' # From realTimePositionSynthesis.py
# ksimilar = 5
# EWMAWeight = 0.7
# upperLegXAxisRotAdj = -30
# leftUpperLegZAxisRotAdj = -20

## walk injured linear mapping (new TFTTFT)
# handLandMarkFilePath = 'complexModel/runSprint.json'
# linearMappingFuncFilePath = './preprocLinearPolyLine/walkInjuredStream/'   # From realTimeRotationMapping.py
# usedJointIdx = [['x','z'], ['x'], ['x','z'], ['x']]
# usedJointIdx1 = [(i,j) for i in range(len(usedJointIdx)) for j in usedJointIdx[i]]  
# mappingStrategy = [['x'], ['x'], ['x'], ['x']]  # 設計的跟usedJointIdx相同即可, 缺一些element而已
# negMappingStrategy = [['z'], [], ['z'], []] # 因為upper leg需要修正沒有作mapping的角度, 所以把沒有mapping的旋轉軸列出
# # TPosePosDataFilePath = 'TPoseInfo/' # From realTimeRotToAvatarPos.py
# TPosePosDataFilePath = 'TPoseInfo/genericAvatar/' # From realTimeRotToAvatarPos.py
# DBMotionKDTreeFilePath = 'DBPreprocFeatVec/walkInjured_withoutHip/'  # From realTimePositionSynthesis.py
# DBMotion3DPosFilePath = 'DBPreprocFeatVec/walkInjured/3DPos/' # From realTimePositionSynthesis.py
# ksimilar = 5
# EWMAWeight = 0.7
# upperLegXAxisRotAdj = -30
# leftUpperLegZAxisRotAdj = -20

# Global variables
preBlendResult = None

def testingStage(
    handLandMark, 
    mappingfunction, mappingStrategy, negMappingStrategy, isLinearMapping, 
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
    if isLinearMapping:
        mappedHandRotations = linearRotationMappingStream(mappedHandRotations, mappingfunction, mappingStrategy)
    else:
        mappedHandRotations = rotationMappingStream(mappedHandRotations, mappingfunction, mappingStrategy)
    # 2.3 之前做的180度校正(2.1做的), 現在要校正回來
    for jointIdx in range(len(usedJointIdx)):
        for aAxis in usedJointIdx[jointIdx]:
            if mappedHandRotations[jointIdx][aAxis] < 0:
                mappedHandRotations[jointIdx][aAxis] += 360
    # 2.4 沒有作mapping的旋轉軸角度需要作旋轉補正
    #       upper leg flexion -30
    #       left upper leg abduction -20
    for jointIdx in range(len(negMappingStrategy)):
        for aAxis in negMappingStrategy[jointIdx]:
            if (jointIdx == 0 or jointIdx == 2) and aAxis == 'x':
                mappedHandRotations[jointIdx][aAxis] += upperLegXAxisRotAdj
            if jointIdx == 0 and aAxis == 'z':
                mappedHandRotations[jointIdx][aAxis] += leftUpperLegZAxisRotAdj

    # print(mappedHandRotations)
    # return mappedHandRotations
    
    # 3. apply mapped rotation to avatar
    lowerBodyPositions = {aJoint: None for aJoint in usedLowerBodyJoints} 
    # 3.1 left kinematic
    leftKinematicNew = forwardKinematic(
        TPoseLeftKinematic, 
        [
            mappedHandRotations[0]['x'], 
            mappedHandRotations[0]['z'], 
            mappedHandRotations[1]['x']
        ]
    )
    lowerBodyPositions[jointsNames.LeftLowerLeg] = leftKinematicNew[0] + leftKinematicNew[1]
    lowerBodyPositions[jointsNames.LeftFoot] = leftKinematicNew[0] + leftKinematicNew[1] + leftKinematicNew[2]
    # 3.2 right kinematic
    rightKinematicNew = forwardKinematic(
            TPoseRightKinematic, 
            [
                mappedHandRotations[2]['x'], 
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
        blendingResult[i] = blendingResult[i]*EWMAweight + preBlendResult[i]*(1-EWMAweight)
    preBlendResult = blendingResult
    # print(blendingResult)
    return blendingResult

# For test the process
# New: 加入對於linear mapping的測試
if __name__=='__main01__':
    
    # 讀取hand landmark data(假裝是streaming data輸入)
    handLMJson = None
    with open(handLandMarkFilePath, 'r') as fileOpen: 
        handLMJson=json.load(fileOpen)
        # side kick 需要篩選部分輸入
        # indexInterval = [2600, 3500]
        # handLMJson = handLMJson[indexInterval[0]:indexInterval[1]]

    # 讀取pre computed mapping function, 也就是BSpline的sample points
    # BSplineSamplePoints = [
    #     [{aAxis: None for aAxis in usedJointIdx[aJoint]} for aJoint in range(len(usedJointIdx))], 
    #     [{aAxis: None for aAxis in usedJointIdx[aJoint]} for aJoint in range(len(usedJointIdx))]
    # ]
    # for aJoint in range(len(usedJointIdx)):
    #     for aAxis in usedJointIdx[aJoint]:
    #         for i in range(2):
    #             BSplineSamplePoints[i][aJoint][aAxis] = \
    #                 np.load(rotationMappingFuncFilePath+'{0}.npy'.format(str(i)+'_'+aAxis+'_'+str(aJoint)))

    # 讀取pre computed linear mapping function用於計算mapped rotation
    fittedLinearLine = [{aAxis: None for aAxis in usedJointIdx[aJoint]} for aJoint in range(len(usedJointIdx))]
    for aJoint in range(len(usedJointIdx)):
        for aAxis in usedJointIdx[aJoint]:
            fittedLinearLine[aJoint][aAxis] = \
                np.load(linearMappingFuncFilePath+'{0}.npy'.format(aAxis+'_'+str(aJoint)))

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
            fittedLinearLine, mappingStrategy, negMappingStrategy, True, 
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
    # with open(rotComputeRetSaveDirPath+'runSprintStream.json', 'r') as WFile: 
    #     handRot = json.load(WFile)
    # print(testingStageResult[0])
    # plt.plot(range(len(handRot)), [i['data'][0]['x'] for i in handRot], label='old')
    # plt.plot(range(len(testingStageResult)), [i[0] for i in testingStageResult], label='new')

    # rotation mapping result, huge difference(修正後相同)
    # 這邊已經使用real time的結果來比較, 理論上結果要相似
    # New, linear mapping的結果與之前使用rotationAnalysis.py有差異
    #       感覺上是因為沒有作180度的校正回來的關係
    #       看過[1, x][2, x]之後又覺得是固定數值偏移的問題
    #       沒錯!!!, 是數值補正問題(沒有mapping的數值需要作補正)
    #       upper leg flexion補正-30
    #       index/left upper leg abduction補正-20
    # rotMapRetSaveDirPath = 'handRotaionAfterMapping/walkInjuredStreamLinearMapping/'
    # rotMapResult = None
    # with open(rotMapRetSaveDirPath+'walkInjured(True, False, True, True, False, True).json', 'r') as WFile: 
    #     rotMapResult = json.load(WFile)
    # plt.plot(range(len(rotMapResult)), [i['data'][2]['x'] for i in rotMapResult], label='old')
    # plt.plot(range(len(testingStageResult)), [i[2]['x'] for i in testingStageResult], label='new')

    # rotation output apply to avatar result, huge difference(修正後相同)
    # 這邊做的forward kinematic與Unity端的結果差異很小
    # 使用新的t pose資訊重新計算結果
    # rotApplySaveDirPath='positionData/fromAfterMappingHand/'
    # # rotApplySaveDirPath='positionData/fromAfterMappingHand/leftSideKickStreamLinearMappingCombinations/'
    # lowerBodyPosition=None
    # # with open(rotApplySaveDirPath+'leftFrontKickStream.json', 'r') as WFile: 
    # with open(rotApplySaveDirPath+'walkInjuredStreamLinearMapping_TFTTFT.json', 'r') as WFile: 
    #     # lowerBodyPosition=json.load(WFile)['results']
    #     lowerBodyPosition=json.load(WFile)  # For python output
    # plt.plot(range(len(lowerBodyPosition)), [i['data']['1']['x'] for i in lowerBodyPosition], label='old')
    # # plt.plot(range(len(lowerBodyPosition)), [i['data'][1]['x'] for i in lowerBodyPosition], label='old')
    # plt.plot(range(len(testingStageResult)), [i[1][0] for i in testingStageResult], label='new')
    
    # after position preprocessing, the different is huge that cannot be neglect(修正後相同, 有些微項位上的不同)
    # AfterMapPreprocArr[joint index][time index, feature index]
    # Multiple array in list. Evey array represents a joint. 
    # 1st dimension in array is time count, 2nd dimension is number of features
    # =======
    # testingStageResult[t][joint index][feature index]
    # Array in dict in list. Every dict in list represents a data at a time.
    # Every array in dict represents a joint's data.
    # array has only one dimension represents number of features
    # =======
    # 1st feature is x position, 11th y pos, 21th z pos
    # saveDirPathHand = 'HandPreprocFeatVec/leftFrontKickStreamLinearMapping_TFFTTT/'
    # # saveDirPathHand = 'HandPreprocFeatVec/leftSideKickStreamLinearMapping_FTTFFF/'
    # # saveDirPathHand = 'HandPreprocFeatVec/runSprintStreamLinearMapping_TFTTFT/'
    # # saveDirPathHand = 'HandPreprocFeatVec/walkInjuredStreamLinearMapping_TFTTFT/'
    # AfterMapPreprocArr = readDBEncodedMotionsFromFile(7, saveDirPathHand)
    # # plt.plot(range(AfterMapPreprocArr[1].shape[0]), AfterMapPreprocArr[1][:, 20], label='old')
    # # plt.plot(range(len(testingStageResult)), [i[1][20] for i in testingStageResult], label='new')
    # plt.plot(range(AfterMapPreprocArr[1].shape[0]), AfterMapPreprocArr[1][:, 2], label='old')
    # plt.plot(range(len(testingStageResult)), [i[1][2] for i in testingStageResult], label='new')

    # after position synthesis
    # testingStageResult
    # array in list in list.
    # saveDirPath = './positionData/afterSynthesis/'
    # posSynRes = None
    # # with open(saveDirPath+'walkInjuredStreamLinearMapping_TFTTFT_EWMA.json') as RFile:
    # with open(saveDirPath+'leftFrontKickStreamLinearMapping_TFFTTT_075_EWMA.json') as RFile:
    #     posSynRes = json.load(RFile)
    # plt.plot(range(len(posSynRes)), [i['data'][5]['x'] for i in posSynRes], label='old')
    # plt.plot(range(len(testingStageResult)), [i[5][0, 0] for i in testingStageResult], label='new')
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
            BSplineSamplePoints, mappingStrategy, negMappingStrategy, False, 
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

# 使用linear mapping 串聯真實streaming data的輸入
if __name__=='__main01__':
    # 讀取預先計算好的linear mapping function
    fittedLinearLine = [{aAxis: None for aAxis in usedJointIdx[aJoint]} for aJoint in range(len(usedJointIdx))]
    for aJoint in range(len(usedJointIdx)):
        for aAxis in usedJointIdx[aJoint]:
            fittedLinearLine[aJoint][aAxis] = \
                np.load(linearMappingFuncFilePath+'{0}.npy'.format(aAxis+'_'+str(aJoint)))

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

    # Open server in another thread
    # 回傳到瀏覽器測試成功, MediaPipe+Camera也測試成功(但是傳送的是string不是json)
    # 傳送到Unity測試看看(unity可以吃string的json)
    from HandLMServer import HandLMServer
    newHttpServer = HandLMServer(hostIP='localhost', hostPort=8080)
    newHttpServer.startHTTPServerThread()

    # Streaming data
    from HandGestureMediaPipe import captureByMediaPipe
    captureByMediaPipe(
        # 0, 
        'C:/Users/liangch/Desktop/MotionStyleHandData/newRecord_2022_9_12/frontKickNew_rgb.avi',
        # 'C:/Users/liangch/Desktop/MotionStyleHandData/newRecord_2022_9_12/sideKickNew_rgb.avi',
        # 'C:/Users/liangch/Desktop/MotionStyleHandData/newRecord_2022_9_12/walkNormal_rgb.avi',
        # 'C:/Users/liangch/Desktop/MotionStyleHandData/newRecord_2022_9_12/walkIInjured_rgb.avi',
        # 這個function call會把一些需要預先填入的database資訊放入, 
        # 只需要再輸入streaming data即可預測avatar position
        lambda streamData: testingStage(
            streamData, 
            fittedLinearLine, mappingStrategy, negMappingStrategy, True, 
            leftKinematic, rightKinematic, TPosePositions, 
            kdtrees, DBPreproc3DPos, ksimilar, EWMAWeight
        ), 
        newHttpServer.curSentMsg
    )
    pass

# 串聯真實streaming data的輸入, 使用webcam加上mediaPipe
if __name__=='__main01__':
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
    
    # Open server in another thread
    # 回傳到瀏覽器測試成功, MediaPipe+Camera也測試成功(但是傳送的是string不是json)
    # 傳送到Unity測試看看(unity可以吃string的json)
    from HandLMServer import HandLMServer
    newHttpServer = HandLMServer(hostIP='localhost', hostPort=8080)
    newHttpServer.startHTTPServerThread()
    # curTime = time.time()
    # while True:
    #     try:
    #         if time.time() - curTime > 3: 
    #             newHttpServer.curSentMsg[0] = str(curTime)
    #             curTime=time.time()
    #         pass
    #     except KeyboardInterrupt:
    #         print('keyboard interrupt')
    #         newHttpServer.stopHTTPServer()

    # Streaming data
    from HandGestureMediaPipe import captureByMediaPipe
    captureByMediaPipe(
        0, 
        # 這個function call會把一些需要預先填入的database資訊放入, 
        # 只需要再輸入streaming data即可預測avatar position
        lambda streamData: testingStage(
            streamData, 
            BSplineSamplePoints, mappingStrategy, negMappingStrategy, False, 
            leftKinematic, rightKinematic, TPosePositions, 
            kdtrees, DBPreproc3DPos, ksimilar, EWMAWeight
        ), 
        newHttpServer.curSentMsg
    )