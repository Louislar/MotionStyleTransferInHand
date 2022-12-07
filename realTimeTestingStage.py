'''
參考testingStageFullProc.py而撰寫的新的code.
將參數的部分獨立到另一個class file, 
也可以使用.json的方式存取獨立的config設定檔
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
from realTimeRotationMapping import rotationMappingStream,tmpRotations, linearRotationMappingStream, \
    quatBSplineRotationMappingStream
from realTimeRotToAvatarPos import forwardKinematic, loadTPosePosAndVecs, \
        usedLowerBodyJoints, forwardKinematicQuat
from realTimePositionSynthesis import posPreprocStream, preLowerBodyPos, preVel, preAcc, \
    rollingWinSize, readDBEncodedMotionsFromFile, jointsInUsedToSyhthesis, fullPositionsJointCount, \
        kSimilarFromKDTree, kSimilarPoseBlendingSingleTime, EWMAForStreaming
from testingStageConfig import TestStageConfig 

# Read config 
config = TestStageConfig()

# Global variables
preBlendResult = None

def testingStage(
    handLandMark, 
    mappingfunction, 
    TPoseLeftKinematic, TPoseRightKinematic, TPosePositions, 
    kdtrees, DBMotion3DPos, 
    config: TestStageConfig
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
    mappedHandRotations = [{aAxis: None for aAxis in i} for i in config.usedJointIdx]
    # 2.1 set max rotation to 180, min rotation to -180
    handRotations = [r-360 if r>180 else r for r in handRotations]
    # 2.2 estimate increase or decrease segment and map the hand rotation    
    for i, _tuple in enumerate(config.usedJointIdx1):
        mappedHandRotations[_tuple[0]][_tuple[1]] = handRotations[i]
    # print(mappedHandRotations)
    if config.mappingCategory == 0:
        mappedHandRotations = linearRotationMappingStream(mappedHandRotations, mappingfunction, config.mappingStrategy)
    elif config.mappingCategory == 1:
        mappedHandRotations = rotationMappingStream(mappedHandRotations, mappingfunction, config.mappingStrategy)
    elif config.mappingCategory == 3:
        mappedHandRotations = quatBSplineRotationMappingStream(
            mappedHandRotations, mappingfunction[0], mappingfunction[1], 
            config.unusedJointAxis, config.quatIndex
        )
    # 2.3 之前做的180度校正(2.1做的), 現在要校正回來
    ## 僅限於euler版本的mapping才會需要校正 
    if config.mappingCategory == 0 or config.mappingCategory == 1: 
        for jointIdx in range(len(config.usedJointIdx)):
            for aAxis in config.usedJointIdx[jointIdx]:
                if mappedHandRotations[jointIdx][aAxis] < 0:
                    mappedHandRotations[jointIdx][aAxis] += 360
    # 2.4 沒有作mapping的旋轉軸角度需要作旋轉補正
    #       upper leg flexion -30
    #       left upper leg abduction -20
    ## 僅限於euler版本的mapping才會需要補正 
    if config.mappingCategory == 0 or config.mappingCategory == 1: 
        for jointIdx in range(len(config.negMappingStrategy)):
            for aAxis in config.negMappingStrategy[jointIdx]:
                if (jointIdx == 0 or jointIdx == 2) and aAxis == 'x':
                    mappedHandRotations[jointIdx][aAxis] += config.upperLegXAxisRotAdj
                if jointIdx == 0 and aAxis == 'z':
                    mappedHandRotations[jointIdx][aAxis] += config.leftUpperLegZAxisRotAdj

    # print(mappedHandRotations)
    # return mappedHandRotations
    
    # 3. apply mapped rotation to avatar
    lowerBodyPositions = {aJoint: None for aJoint in usedLowerBodyJoints} 
    # 3.1 left kinematic
    leftKinematicNew=None
    if config.mappingCategory == 0 or config.mappingCategory == 1: 
        leftKinematicNew = forwardKinematic(
            TPoseLeftKinematic, 
            [
                mappedHandRotations[0]['x'], 
                mappedHandRotations[0]['z'], 
                mappedHandRotations[1]['x']
            ]
        )
    elif config.mappingCategory == 2 or config.mappingCategory == 3: 
        leftKinematicNew = forwardKinematicQuat(
            TPoseLeftKinematic, 
            [
                [
                    mappedHandRotations[0]['x'], 
                    mappedHandRotations[0]['y'], 
                    mappedHandRotations[0]['z'], 
                    mappedHandRotations[0]['w']
                ], 
                [
                    mappedHandRotations[1]['x'], 
                    mappedHandRotations[1]['y'], 
                    mappedHandRotations[1]['z'], 
                    mappedHandRotations[1]['w']
                ]
            ]
        )
    lowerBodyPositions[jointsNames.LeftLowerLeg] = leftKinematicNew[0] + leftKinematicNew[1]
    lowerBodyPositions[jointsNames.LeftFoot] = leftKinematicNew[0] + leftKinematicNew[1] + leftKinematicNew[2]
    # 3.2 right kinematic
    rightKinematicNew = None
    if config.mappingCategory == 0 or config.mappingCategory == 1: 
        rightKinematicNew = forwardKinematic(
                TPoseRightKinematic, 
                [
                    mappedHandRotations[2]['x'], 
                    mappedHandRotations[2]['z'], 
                    mappedHandRotations[3]['x']
                ]
            )
    elif config.mappingCategory == 2 or config.mappingCategory == 3: 
        rightKinematicNew = forwardKinematicQuat(
                TPoseRightKinematic, 
                [
                    [
                        mappedHandRotations[2]['x'], 
                        mappedHandRotations[2]['y'], 
                        mappedHandRotations[2]['z'], 
                        mappedHandRotations[2]['w']
                    ],
                    [
                        mappedHandRotations[3]['x'], 
                        mappedHandRotations[3]['y'], 
                        mappedHandRotations[3]['z'], 
                        mappedHandRotations[3]['w']
                    ]
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
        config.ksimilar
    )
    # 4.3 use k similar feature vector to construct full body pose
    blendingResult = kSimilarPoseBlendingSingleTime(DBMotion3DPos, kSimilarIdx, kSimilarDist)
    # print(blendingResult)
    # 4.4 EWMA
    global preBlendResult
    if preBlendResult is None:
        preBlendResult = blendingResult
    for i in range(len(blendingResult)):
        blendingResult[i] = blendingResult[i]*config.EWMAWeight + preBlendResult[i]*(1-config.EWMAWeight)
    preBlendResult = blendingResult
    # print(blendingResult)
    return blendingResult

# For test the process
# New: 加入對於linear mapping的測試
# 加入對quaternion B-Spline mapping的測試 
if __name__=='__main01__':
    
    # 讀取hand landmark data(假裝是streaming data輸入)
    handLMJson = None
    with open(config.handLandMarkFilePath, 'r') as fileOpen: 
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

    # 讀取pre computed quaternion B-Spline mapping function, 當中包含hand與body的sample points
    BSplineHandSP = None
    BSplineBodySP = None
    with open(config.BSplineHandSPFilePath, 'rb') as RFile:
        BSplineHandSP = pickle.load(RFile)
    with open(config.BSplineBodySPFilePath, 'rb') as RFile:
        BSplineBodySP = pickle.load(RFile)
    ## 修正沒有使用的sample points (修改成0 mapping到0)
    for _jointInd in range(len(BSplineHandSP)):
        for _axis in BSplineHandSP[_jointInd]:
            if BSplineHandSP[_jointInd][_axis] is None: 
                BSplineHandSP[_jointInd][_axis] = np.array([0, 0, 0])
                BSplineBodySP[_jointInd][_axis] = np.array([0, 0, 0])
            else:   
                BSplineHandSP[_jointInd][_axis] = np.array(BSplineHandSP[_jointInd][_axis])
                BSplineBodySP[_jointInd][_axis] = np.array(BSplineBodySP[_jointInd][_axis])
    bsMappingFunc = [BSplineHandSP, BSplineBodySP]

    # 讀取pre computed linear mapping function用於計算mapped rotation
    # fittedLinearLine = [{aAxis: None for aAxis in config.usedJointIdx[aJoint]} for aJoint in range(len(config.usedJointIdx))]
    # for aJoint in range(len(config.usedJointIdx)):
    #     for aAxis in config.usedJointIdx[aJoint]:
    #         fittedLinearLine[aJoint][aAxis] = \
    #             np.load(config.linearMappingFuncFilePath+'{0}.npy'.format(aAxis+'_'+str(aJoint)))

    # 讀取T pose position以及vectors, 計算left and right kinematics
    TPosePositions, TPoseVectors  = loadTPosePosAndVecs(config.TPosePosDataFilePath)
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
    DBPreproc3DPos = readDBEncodedMotionsFromFile(fullPositionsJointCount, config.DBMotion3DPosFilePath)
    kdtrees = {k: None for k in jointsInUsedToSyhthesis}
    for i in jointsInUsedToSyhthesis:
        with open(config.DBMotionKDTreeFilePath+'{0}.pickle'.format(i), 'rb') as inPickle:
            kdtrees[i] = kdtree = pickle.load(inPickle)

    # testing stage full processes
    timeCount = len(handLMJson)

    testingStageResult = []
    fullTimeLaps = np.zeros(timeCount)
    for t in range(timeCount):
        result = testingStage(
            handLMJson[t]['data'], 
            bsMappingFunc,  
            leftKinematic, rightKinematic, TPosePositions, 
            kdtrees, DBPreproc3DPos, 
            config
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
    ## computed hand rotation, 
    ## find the bug, the output of computeUsedRotation() is not straintforward
    # rotComputeRetSaveDirPath = 'HandRotationOuputFromHomePC/'
    # handRot = None
    # with open(rotComputeRetSaveDirPath+'leftFrontKickStream.json', 'r') as WFile: 
    #     handRot = json.load(WFile)
    # print(testingStageResult[0])
    # plt.plot(range(len(handRot)), [i['data'][1]['x'] for i in handRot], label='old')
    # plt.plot(range(len(testingStageResult)), [i[2] for i in testingStageResult], label='new')

    # rotation mapping result, huge difference(修正後相同)
    # 這邊已經使用real time的結果來比較, 理論上結果要相似
    # New, linear mapping的結果與之前使用rotationAnalysis.py有差異
    #       感覺上是因為沒有作180度的校正回來的關係
    #       看過[1, x][2, x]之後又覺得是固定數值偏移的問題
    #       沒錯!!!, 是數值補正問題(沒有mapping的數值需要作補正)
    #       upper leg flexion補正-30
    #       index/left upper leg abduction補正-20
    # rotMapRetSaveDirPath = 'rotationMappingQuaternionData/leftFrontKick/'
    # rotMapResult = None
    # with open(rotMapRetSaveDirPath+'leftFrontKick_quat_BSpline_TFTTTT.json', 'r') as WFile: 
    #     rotMapResult = json.load(WFile)
    # plt.plot(range(len(rotMapResult)), [i['data'][0]['x'] for i in rotMapResult], label='old')
    # plt.plot(range(len(testingStageResult)), [i[0]['x'] for i in testingStageResult], label='new')

    # rotation output apply to avatar result, huge difference(修正後相同)
    # 這邊做的forward kinematic與Unity端的結果差異很小
    # 使用新的t pose資訊重新計算結果
    # rotApplySaveDirPath='positionData/fromAfterMappingHand/'
    # # rotApplySaveDirPath='positionData/fromAfterMappingHand/leftSideKickStreamLinearMappingCombinations/'
    # lowerBodyPosition=None
    # # with open(rotApplySaveDirPath+'leftFrontKickStream.json', 'r') as WFile: 
    # with open(rotApplySaveDirPath+'newMappingMethods/leftFrontKick_quat_BSpline_TFTTTT.json', 'r') as WFile: 
    #     # lowerBodyPosition=json.load(WFile)['results']
    #     lowerBodyPosition=json.load(WFile)  # For python output
    # plt.plot(range(len(lowerBodyPosition)), [i['data']['2']['z'] for i in lowerBodyPosition], label='old')
    # # plt.plot(range(len(lowerBodyPosition)), [i['data'][1]['x'] for i in lowerBodyPosition], label='old')
    # plt.plot(range(len(testingStageResult)), [i[2][2] for i in testingStageResult], label='new')
    
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
    # with open(saveDirPath+'leftFrontKick_quat_BSpline_TFTTTT_075_EWMA.json') as RFile:
    #     posSynRes = json.load(RFile)
    # plt.plot(range(len(posSynRes)), [i['data'][2]['y'] for i in posSynRes], label='old')
    # plt.plot(range(len(testingStageResult)), [i[2][0, 1] for i in testingStageResult], label='new')
    plt.legend()
    plt.show()

## TODO: 使用quaternion and B-Spline mapping
## 串聯真實streaming data的輸入 (影片或是webcam)
if __name__=='__main01__':
    # 1. 讀取mapping function 
    ## 讀取pre computed quaternion B-Spline mapping function, 當中包含hand與body的sample points
    BSplineHandSP = None
    BSplineBodySP = None
    with open(config.BSplineHandSPFilePath, 'rb') as RFile:
        BSplineHandSP = pickle.load(RFile)
    with open(config.BSplineBodySPFilePath, 'rb') as RFile:
        BSplineBodySP = pickle.load(RFile)
    ## 修正沒有使用的sample points (修改成0 mapping到0)
    for _jointInd in range(len(BSplineHandSP)):
        for _axis in BSplineHandSP[_jointInd]:
            if BSplineHandSP[_jointInd][_axis] is None: 
                BSplineHandSP[_jointInd][_axis] = np.array([0, 0, 0])
                BSplineBodySP[_jointInd][_axis] = np.array([0, 0, 0])
            else:   
                BSplineHandSP[_jointInd][_axis] = np.array(BSplineHandSP[_jointInd][_axis])
                BSplineBodySP[_jointInd][_axis] = np.array(BSplineBodySP[_jointInd][_axis])
    bsMappingFunc = [BSplineHandSP, BSplineBodySP]

    # 讀取T pose position以及vectors, 計算left and right kinematics
    TPosePositions, TPoseVectors  = loadTPosePosAndVecs(config.TPosePosDataFilePath)
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
    DBPreproc3DPos = readDBEncodedMotionsFromFile(fullPositionsJointCount, config.DBMotion3DPosFilePath)
    kdtrees = {k: None for k in jointsInUsedToSyhthesis}
    for i in jointsInUsedToSyhthesis:
        with open(config.DBMotionKDTreeFilePath+'{0}.pickle'.format(i), 'rb') as inPickle:
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
            bsMappingFunc, 
            leftKinematic, rightKinematic, TPosePositions, 
            kdtrees, DBPreproc3DPos,
            config
        ), 
        newHttpServer.curSentMsg
    )
    pass