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
from realTimeHandRotationCompute import negateAxes, heightWidthCorrection, kalmanFilter, \
        computeUsedVectors, computeUsedRotations, negateXYZMask, \
        kalmanParamQ, kalmanParamR, kalmanX, kalmanK, kalmanP
from realTimeHandRotationCompute import jointsNames as handJointsNames

handLandMarkFilePath = 'complexModel/frontKick.json'
rotationMappingFuncFilePath = ''
TPosePosDataFilePath = ''
DBMotionKDTreeFilePath = ''

def testingStage(handLandMark, mappingfunction):
    '''
    Goal: 
    Input:
    :handLandMark: 單一時間點下, 偵測到的hand landmarks, 
        共21個joints
    :mappingFunction: 預先計算好的

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
    pass

# Execute the full process
if __name__=='__main__':
    # 讀取hand landmark data(假裝是streaming data輸入)
    handLMJson = None
    with open(handLandMarkFilePath, 'r') as fileOpen: 
        handLMJson=json.load(fileOpen)
    # TODO: 讀取pre computed mapping function
    mappingFunction = None


    # testing stage full processes
    timeCount = len(handLMJson)

    for t in range(timeCount):
        testingStage(handLMJson[t]['data'], mappingFunction)
        break
    pass