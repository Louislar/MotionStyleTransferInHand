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

from turtle import forward
import numpy as np 
import json
import pickle
from positionAnalysis import jointsNames
from realTimeHandRotationCompute import negateAxes, heightWidthCorrection, kalmanFilter, \
        computeUsedVectors, computeUsedRotations, negateXYZMask, \
        kalmanParamQ, kalmanParamR, kalmanX, kalmanK, kalmanP
from realTimeHandRotationCompute import jointsNames as handJointsNames
from realTimeRotationMapping import rotationMappingStream,tmpRotations
from realTimeRotToAvatarPos import forwardKinematic, loadTPosePosAndVecs, upperLegXRotAdj, leftUpperLegZRotAdj, \
        usedLowerBodyJoints

handLandMarkFilePath = 'complexModel/frontKick.json'
rotationMappingFuncFilePath = 'preprocBSpline/leftFrontKick/'
usedJointIdx = [['x','z'], ['x'], ['x','z'], ['x']]
usedJointIdx1 = [(i,j) for i in range(len(usedJointIdx)) for j in usedJointIdx[i]]  
mappingStrategy = [['x'], [], ['z'], ['x']]  # 設計的跟usedJointIdx相同即可, 缺一些element而已
TPosePosDataFilePath = ''
DBMotionKDTreeFilePath = ''

def testingStage(
    handLandMark, 
    mappingfunction, mappingStrategy, 
    TPoseLeftKinematic, TPoseRightKinematic
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
    print(handLandMark)
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
    print(handRotations)

    # 2. hand rotation mapping
    mappedHandRotations = [{aAxis: None for aAxis in i} for i in usedJointIdx]
    # 2.1 set max rotation to 180, min rotation to -180
    handRotations = [r-360 if r>180 else r for r in handRotations]
    # 2.2 estimate increase or decrease segment and map the hand rotation    
    for i, _tuple in enumerate(usedJointIdx1):
        mappedHandRotations[_tuple[0]][_tuple[1]] = handRotations[i]
    print(mappedHandRotations)
    mappedHandRotations = rotationMappingStream(mappedHandRotations, mappingfunction, mappingStrategy)
    print(mappedHandRotations)
    
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
    # TODO: finish this

# Execute the full process
if __name__=='__main__':
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

    # testing stage full processes
    timeCount = len(handLMJson)

    for t in range(timeCount):
        testingStage(
            handLMJson[t]['data'], 
            BSplineSamplePoints, mappingStrategy, 
            leftKinematic, rightKinematic
        )
        break
    pass