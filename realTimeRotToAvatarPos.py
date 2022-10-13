'''
Goal: after mapping後的rotation需要apply到avatar的lower body上, 
才能夠的到position資訊, 得到position資訊後才能與DB當中的motion做比較
'''

import numpy as np 
import time
import json
import pickle
import os 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from scipy.spatial.transform import Rotation as R
from positionAnalysis import jointsNames
from realTimeHandRotationCompute import jointsNames as handJointNames

usedLowerBodyJoints = [
    jointsNames.LeftUpperLeg, jointsNames.LeftLowerLeg, jointsNames.LeftFoot, 
    jointsNames.RightUpperLeg, jointsNames.RightLowerLeg, jointsNames.RightFoot, 
    jointsNames.Hip
]

def loadTPosePosAndVecs(saveDirPath):
    '''
    Goal: load儲存的T pose position以及vectors資訊
    '''
    TPosePositions=None
    TPoseVectors=None
    with open(saveDirPath+'TPosePositions.pickle', 'rb') as inPickle:
            TPosePositions = pickle.load(inPickle)
    with open(saveDirPath+'TPoseVectors.pickle', 'rb') as inPickle:
        TPoseVectors = pickle.load(inPickle)
    return TPosePositions, TPoseVectors

def visualize3DVecs(startPts, vectors):
    '''
    (original version is in realTimeHandRotationCompute)
    Goal: 顯示3D空間向量, 方便debug, 顯示部分joints以及vertex即可
    Input: 
    :*args: 多個3d position array組成的list代表向量的起點 + 多個3d position array組成的list代表多個向量
            以上兩者長度要相同
    '''
    # origin = [0, 0, 0]
    # x = [1, 2, 3]
    # y = [4, 5, 6]
    # xCy = np.cross(x, y)
    xyzZip = list(zip(*[startPts[i]+vectors[i] for i in range(len(vectors))]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colormap = matplotlib.cm.inferno
    colors = list(range(len(vectors)))
    norm = matplotlib.colors.Normalize()
    norm.autoscale(colors)
    for i in range(len(vectors)):
        ax.quiver(xyzZip[0][i], xyzZip[1][i], xyzZip[2][i], xyzZip[3][i], xyzZip[4][i], xyzZip[5][i], arrow_length_ratio =0.2, color=colormap(norm(colors[i])), label=i)
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 2])
    ax.set_zlim([0, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.legend()

def forwardKinematic(kinematicChain, forwardRotations):
    '''
    Goal: 給定一個kinematic chain(1 position, 2 vectors), 
        給予第一個joint的X, Z旋轉, 以及第二個joint的X旋轉, 
        求第2及第3個點的旋轉後position
        Note, z軸旋轉要正負相反, 因為Unity與python的z軸是反過來的
        TODO: 第一個vector的旋轉, 會影響到第2個vector
    '''
    outputKC=[kinematicChain[0], None, None]
    # upperLegRotMat = R.from_euler('zyx', [forwardRotations[1], 0, forwardRotations[0]], degrees=True) # upper leg rotation matrix
    # 看起來unity是以zxy的順序以extrinsic rotation進行旋轉
    # ref: https://forum.unity.com/threads/which-euler-angles-convention-used-in-unity.41114/#post-6828104
    upperLegRotMat = R.from_euler('zxy', [forwardRotations[1], forwardRotations[0], 0], degrees=True) # upper leg rotation matrix
    upperLegRotMat = upperLegRotMat.as_matrix()
    outputKC[1] = np.dot(upperLegRotMat, kinematicChain[1])
    # print(upperLegRotMat)
    
    # lowerLegRotMat = R.from_euler('zyx', [0, 0, forwardRotations[2]], degrees=True)
    lowerLegRotMat = R.from_euler('zxy', [0, forwardRotations[2], 0], degrees=True)
    lowerLegRotMat = lowerLegRotMat.as_matrix()
    outputKC[2] = np.dot(lowerLegRotMat, kinematicChain[2])
    outputKC[2] = np.dot(upperLegRotMat, outputKC[2])
    # print(lowerLegRotMat)
    

    # another method(maybe wrong) -> it's wrong, it assume the link(bone) is always on the x axis
    # link1Length = np.linalg.norm(kinematicChain[1])
    # link2Length = np.linalg.norm(kinematicChain[2])
    # firstRotResult = np.dot(upperLegRotMat, np.array([link1Length, 0, 0]))
    # newSecondPoint = firstRotResult + forwardRotations[0]
    # outputKC[1] = firstRotResult

    # secondRotResult = np.dot(lowerLegRotMat, np.array([link2Length, 0, 0]))
    # secondRotResult = np.dot(upperLegRotMat, secondRotResult)
    # newThirdPoint = secondRotResult + newSecondPoint
    # outputKC[2] = secondRotResult

    return outputKC

# For test, compare the position result in unity and python
if __name__=='__main01__':
    rotApplyUnitySaveDirPath = 'positionData/fromAfterMappingHand/leftFrontKickCombinations/'
    unityPosJson = None
    with open(rotApplyUnitySaveDirPath+'leftFrontKick(True, False, False, False, True, True).json', 'r') as fileIn:
        unityPosJson = json.load(fileIn)['results']

    rotApplyPythonSaveDirPath = 'positionData/fromAfterMappingHand/'
    pythonPosJson = None
    with open(rotApplyPythonSaveDirPath+'leftFrontKickStream.json', 'r') as fileIn:
        pythonPosJson = json.load(fileIn)

    unityTimeCount = len(unityPosJson)
    pythonTimeCount = len(pythonPosJson)
    unityJointCount = len(unityPosJson[0]['data'])
    pythonJointCount = len(pythonPosJson[0]['data'])
    print('unity time count: ', unityTimeCount)
    print('python time count: ', pythonTimeCount)
    print('unity joint count: ', unityJointCount)
    print('python joint count: ', pythonJointCount)
    jointInd = 0
    jointAxis = 'z'
    unityData = [unityPosJson[t]['data'][jointInd][jointAxis] for t in range(unityTimeCount)]
    pythonData = [pythonPosJson[t]['data'][str(jointInd)][jointAxis] for t in range(pythonTimeCount)]
    plt.figure()
    plt.plot(range(unityTimeCount), unityData, label='unity')
    plt.plot(range(pythonTimeCount), pythonData, label='real time')
    plt.legend()
    plt.show()

# For test, 
# apply unity收集到的animation rotation data到python的avatar
# 觀察結果是否合理
if __name__=='__main__':
    # 1. 讀取預存好的T pose position以及vectors
    # 2. 讀取animation rotation
    # 3. (real time)Apply rotation到T pose vectors
    
    # 1. 
    saveDirPath='TPoseInfo/genericAvatar/'
    TPosePositions, TPoseVectors = loadTPosePosAndVecs(saveDirPath)
    # 2. 
    animationRotDirPath = 'bodyDBRotation/genericAvatar/'
    animRotJson = None
    with open(os.path.join(animationRotDirPath, 'leftFrontKick0.03_withoutHip.json')) as fileIn:
        animRotJson = json.load(fileIn)['results']
    timeCount = len(animRotJson)
    print('timeCount: ', timeCount)
    # 3. 
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
    lowerBodyPosition = [{'time': t, 'data': {aJoint: None for aJoint in usedLowerBodyJoints}} for t in range(timeCount)]
    testKinematic1 = None
    rotApplyTimeLaps = np.zeros(timeCount)
    for t in range(timeCount):
        testKinematic1 = forwardKinematic(
            leftKinematic, 
            [
                animRotJson[t]['data'][0]['x'], 
                animRotJson[t]['data'][0]['z'], 
                animRotJson[t]['data'][1]['x']
            ]
        )
        lowerBodyPosition[t]['data'][jointsNames.LeftLowerLeg] = testKinematic1[0] + testKinematic1[1]
        lowerBodyPosition[t]['data'][jointsNames.LeftFoot] = testKinematic1[0] + testKinematic1[1] + testKinematic1[2]
        
        testKinematic2 = forwardKinematic(
            rightKinematic, 
            [
                animRotJson[t]['data'][2]['x'], 
                animRotJson[t]['data'][2]['z'], 
                animRotJson[t]['data'][3]['x']
            ]
        )
        lowerBodyPosition[t]['data'][jointsNames.RightLowerLeg] = testKinematic2[0] + testKinematic2[1]
        lowerBodyPosition[t]['data'][jointsNames.RightFoot] = testKinematic2[0] + testKinematic2[1] + testKinematic2[2]
        rotApplyTimeLaps[t] = time.time()
    rotApplyCost = rotApplyTimeLaps[1:] - rotApplyTimeLaps[:-1]
    print('rotation compute avg time: ', np.mean(rotApplyCost))
    print('rotation compute time std: ', np.std(rotApplyCost))
    print('rotation compute max time cost: ', np.max(rotApplyCost))
    print('rotation compute min time cost: ', np.min(rotApplyCost))
    testKinematic1 = forwardKinematic(leftKinematic, [90, 90, 90])
    ## 轉換資料成方便儲存的格式
    for t in range(timeCount):
        lowerBodyPosition[t]['data'][jointsNames.Hip] = TPosePositions[jointsNames.Hip]
        lowerBodyPosition[t]['data'][jointsNames.LeftUpperLeg] = TPosePositions[jointsNames.LeftUpperLeg]
        lowerBodyPosition[t]['data'][jointsNames.RightUpperLeg] = TPosePositions[jointsNames.RightUpperLeg]
    for t in range(timeCount):
        for aJoint in usedLowerBodyJoints:
            if lowerBodyPosition[t]['data'][aJoint] is not None:
                lowerBodyPosition[t]['data'][aJoint] = {k: lowerBodyPosition[t]['data'][aJoint][i] for i, k in enumerate(['x', 'y', 'z'])} 
    # Store data into file
    rotApplySaveDirPath='positionData/'
    with open(os.path.join(rotApplySaveDirPath, 'testLeftFrontKickAnimRotToAvatar.json'), 'w') as WFile: 
        # json.dump(lowerBodyPosition, WFile)
        pass

# Implement rotation apply to avatar
if __name__=='__main01__':
    # 1. 讀取預存好的T pose position以及vectors
    # 2. 讀取mapped hand rotations
    # 3. (real time)Apply mapped hand rotations到T pose position以及vectors上
    # 4. Store the applied result(avatar lower body motions)

    # 1. 
    saveDirPath='TPoseInfo/genericAvatar/'
    TPosePositions, TPoseVectors = loadTPosePosAndVecs(saveDirPath)
    print(TPosePositions)
    print(TPoseVectors)

    # 2. 
    # mappedHandRotSaveDirPath='handRotaionAfterMapping/leftFrontKick/'
    # mappedHandRotSaveDirPath='handRotaionAfterMapping/' # python real time版本計算的結果
    mappedHandRotSaveDirPath='handRotaionAfterMapping/leftFrontKickStreamLinearMapping/' 
    # mappedHandRotSaveDirPath='handRotaionAfterMapping/leftSideKickStreamLinearMapping/' 
    # mappedHandRotSaveDirPath='handRotaionAfterMapping/runSprintStreamLinearMapping/' 
    # mappedHandRotSaveDirPath='handRotaionAfterMapping/walkInjuredStreamLinearMapping/' 
    mappedHandRotJson = None
    # with open(mappedHandRotSaveDirPath+'leftFrontKick(True, False, False, False, True, True).json', 'r') as fileIn:
    # with open(mappedHandRotSaveDirPath+'leftFrontKickStreamTFFFTT.json', 'r') as fileIn: # python real time版本計算的結果
    with open(mappedHandRotSaveDirPath+'leftFrontKick(True, False, False, True, True, True).json', 'r') as fileIn:
    # with open(mappedHandRotSaveDirPath+'leftSideKick(False, True, True, False, False, False).json', 'r') as fileIn:
    # with open(mappedHandRotSaveDirPath+'runSprint(True, False, True, True, False, True).json', 'r') as fileIn:
    # with open(mappedHandRotSaveDirPath+'walkInjured(True, False, True, True, False, True).json', 'r') as fileIn:
        mappedHandRotJson = json.load(fileIn)
    timeCount = len(mappedHandRotJson)
    # print(mappedHandRotJson)

    # 3. 
    # visualize 三個joints的heirarchy結構(origin position, two vectors)
    # 這邊只會有兩個獨立的heirarchy結構: 左腿, 右腿
    ## 3.1 切成兩個kinematic chain(left and right), 並且接下來的處理都是以chain為單位
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
    # testKinematic = [
    #     np.array([0, 0, 0]), 
    #     np.array([0, 0, 1]), 
    #     np.array([1, 1, 0])
    # ]

    ## 3.2 forward kinematic
    # visualize3DVecs(
    #     [testKinematic[0].tolist(), (testKinematic[0]+testKinematic[1]).tolist()], 
    #     [testKinematic[1].tolist(), testKinematic[2].tolist()]
    # )
    # visualize3DVecs(
    #     [leftKinematic[0].tolist(), (leftKinematic[0]+leftKinematic[1]).tolist(), leftKinematic[0].tolist(), leftKinematic[0].tolist(), leftKinematic[0].tolist()], 
    #     [leftKinematic[1].tolist(), leftKinematic[2].tolist(), [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # )
    lowerBodyPosition = [{'time': t, 'data': {aJoint: None for aJoint in usedLowerBodyJoints}} for t in range(timeCount)]
    testKinematic1 = None
    rotApplyTimeLaps = np.zeros(timeCount)
    for t in range(timeCount):
        # print(mappedHandRotJson[t]['data'][0])
        # print(mappedHandRotJson[t]['data'][1])
        testKinematic1 = forwardKinematic(
            leftKinematic, 
            [
                mappedHandRotJson[t]['data'][0]['x'], 
                mappedHandRotJson[t]['data'][0]['z'], 
                mappedHandRotJson[t]['data'][1]['x']
            ]
        )
        lowerBodyPosition[t]['data'][jointsNames.LeftLowerLeg] = testKinematic1[0] + testKinematic1[1]
        lowerBodyPosition[t]['data'][jointsNames.LeftFoot] = testKinematic1[0] + testKinematic1[1] + testKinematic1[2]
        
        testKinematic2 = forwardKinematic(
            rightKinematic, 
            [
                mappedHandRotJson[t]['data'][2]['x'], 
                mappedHandRotJson[t]['data'][2]['z'], 
                mappedHandRotJson[t]['data'][3]['x']
            ]
        )
        lowerBodyPosition[t]['data'][jointsNames.RightLowerLeg] = testKinematic2[0] + testKinematic2[1]
        lowerBodyPosition[t]['data'][jointsNames.RightFoot] = testKinematic2[0] + testKinematic2[1] + testKinematic2[2]
        rotApplyTimeLaps[t] = time.time()
    rotApplyCost = rotApplyTimeLaps[1:] - rotApplyTimeLaps[:-1]
    print('rotation compute avg time: ', np.mean(rotApplyCost))
    print('rotation compute time std: ', np.std(rotApplyCost))
    print('rotation compute max time cost: ', np.max(rotApplyCost))
    print('rotation compute min time cost: ', np.min(rotApplyCost))
    testKinematic1 = forwardKinematic(leftKinematic, [90, 90, 90])
    # visualize3DVecs(
    #     [testKinematic1[0].tolist(), (testKinematic1[0]+testKinematic1[1]).tolist()], 
    #     [testKinematic1[1].tolist(), testKinematic1[2].tolist()]
    # )
    # visualize3DVecs(
    #     [[0,0,0], [0,0,1]], 
    #     [[0,1,0], [0,1,1]]
    # )
    # plt.show()

    # 4. 
    # Unity store 7 joints
    # (left/right) upper, lowerleg , foot, hips
    for t in range(timeCount):
        lowerBodyPosition[t]['data'][jointsNames.Hip] = TPosePositions[jointsNames.Hip]
        lowerBodyPosition[t]['data'][jointsNames.LeftUpperLeg] = TPosePositions[jointsNames.LeftUpperLeg]
        lowerBodyPosition[t]['data'][jointsNames.RightUpperLeg] = TPosePositions[jointsNames.RightUpperLeg]
    for t in range(timeCount):
        for aJoint in usedLowerBodyJoints:
            if lowerBodyPosition[t]['data'][aJoint] is not None:
                lowerBodyPosition[t]['data'][aJoint] = {k: lowerBodyPosition[t]['data'][aJoint][i] for i, k in enumerate(['x', 'y', 'z'])}
    
    
    rotApplySaveDirPath='positionData/fromAfterMappingHand/'
    # with open(rotApplySaveDirPath+'leftFrontKickStream.json', 'w') as WFile: 
    with open(rotApplySaveDirPath+'leftFrontKickStreamLinearMapping_TFFTTT.json', 'w') as WFile: 
    # with open(rotApplySaveDirPath+'leftSideKickStreamLinearMapping_FTTFFF.json', 'w') as WFile:
    # with open(rotApplySaveDirPath+'runSprintStreamLinearMapping_TFTTFT.json', 'w') as WFile: 
    # with open(rotApplySaveDirPath+'walkInjuredStreamLinearMapping_TFTTFT.json', 'w') as WFile: 
        json.dump(lowerBodyPosition, WFile)
        pass

    # 5. 
    # compare with unity result
    # rotApplyUnitySaveDirPath = 'positionData/fromAfterMappingHand/leftFrontKickCombinations/'
    rotApplyUnitySaveDirPath = 'positionData/fromAfterMappingHand/leftFrontKickStreamLinearMapping/'
    # rotApplyUnitySaveDirPath = 'positionData/fromAfterMappingHand/leftSideKickStreamLinearMapping/'
    # rotApplyUnitySaveDirPath = 'positionData/fromAfterMappingHand/runSprintStreamLinearMappingCombinations/'
    unityPosJson = None
    with open(rotApplyUnitySaveDirPath+'leftFrontKick(True, False, False, True, True, True).json', 'r') as fileIn:
    # with open(rotApplyUnitySaveDirPath+'leftSideKick(False, True, True, False, False, False).json', 'r') as fileIn:
    # with open(rotApplyUnitySaveDirPath+'runSprint(True, False, True, True, False, True).json', 'r') as fileIn:
        unityPosJson = json.load(fileIn)['results']
    unityTimeCount = len(unityPosJson)
    pythonTimeCount = len(lowerBodyPosition)
    print('unity time count: ', unityTimeCount)
    print('python time count: ', pythonTimeCount)
    vizJoint = 5
    vizAxis = 'x'
    unityData = [unityPosJson[t]['data'][vizJoint][vizAxis] for t in range(unityTimeCount)]
    pythonData = [lowerBodyPosition[t]['data'][vizJoint][vizAxis] for t in range(pythonTimeCount)]
    plt.figure()
    plt.plot(range(unityTimeCount), unityData, label='unity')
    plt.plot(range(pythonTimeCount), pythonData, label='real time')
    plt.legend()
    plt.show()

if __name__=='__main01__':
    # 1. 讀取檔案, 得到TPose狀態下的position資訊
    #   1.1 Hip, upper leg, lower leg, foot
    # 2. 計算lower body的bone length(改為計算TPose時的向量就好, 他就包含了bone length的資訊)
    #   2.1 upper leg, lower leg兩個bone lengths(vectors)
    # 3. Store bone lengths and TPose positions

    # 1. 
    saveDirPath = 'positionData/fromDB/'
    saveDirPath = 'positionData/fromDB/genericAvatar/'
    TPoseJson = None
    with open(saveDirPath+'TPose.json', 'r') as fileIn:
        TPoseJson = json.load(fileIn)['results']
    jointCount = len(TPoseJson[0]['data'])
    # print(TPoseJson[0])
    # print('=======')
    # print(TPoseJson[2])
    print('joint count: ', jointCount)
    # 1.1
    # 只擷取一個時間點的TPose資訊即可, 特別是lower body的部分
    # 擷取的時間點不要第一個時間點就好
    TPosePositions = {aJoint: np.array([TPoseJson[2]['data'][aJoint][aAxis] for aAxis in ['x', 'y', 'z']]) for aJoint in usedLowerBodyJoints}
    print(TPosePositions)

    # 2. 
    # 4 vectors, (left/right)(upper/lower leg)
    TPoseVectors = [
        TPosePositions[jointsNames.LeftLowerLeg] - TPosePositions[jointsNames.LeftUpperLeg], 
        TPosePositions[jointsNames.LeftFoot] - TPosePositions[jointsNames.LeftLowerLeg], 
        TPosePositions[jointsNames.RightLowerLeg] - TPosePositions[jointsNames.RightUpperLeg], 
        TPosePositions[jointsNames.RightFoot] - TPosePositions[jointsNames.RightLowerLeg]
    ]
    print(TPoseVectors)
    # 3. 
    # saveDirPath='TPoseInfo/'
    saveDirPath='TPoseInfo/genericAvatar/'
    # with open(saveDirPath+'TPosePositions.pickle', 'wb') as outPickle:
    #     pickle.dump(TPosePositions, outPickle)
    # with open(saveDirPath+'TPoseVectors.pickle', 'wb') as outPickle:
    #     pickle.dump(TPoseVectors, outPickle)
