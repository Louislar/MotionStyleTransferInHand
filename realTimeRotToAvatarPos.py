'''
Goal: after mapping後的rotation需要apply到avatar的lower body上, 
才能夠的到position資訊, 得到position資訊後才能與DB當中的motion做比較
'''

import numpy as np 
import json
import pickle
from positionAnalysis import jointsNames

usedLowerBodyJoints = [
    jointsNames.LeftUpperLeg, jointsNames.LeftLowerLeg, jointsNames.LeftFoot, 
    jointsNames.RightUpperLeg, jointsNames.RightLowerLeg, jointsNames.RightFoot
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

if __name__=='__main__':
    # 1. 讀取預存好的T pose position以及vectors
    # 2. 讀取mapped hand rotations
    # 3. (real time)Apply mapped hand rotations到T pose position以及vectors上
    # 4. Store the applied result(avatar lower body motions)

    # 1. 
    saveDirPath='TPoseInfo/'
    TPosePositions, TPoseVectors = loadTPosePosAndVecs(saveDirPath)
    print(TPosePositions)
    print(TPoseVectors)
    # 2. 

if __name__=='__main01__':
    # 1. 讀取檔案, 得到TPose狀態下的position資訊
    #   1.1 Hip, upper leg, lower leg, foot
    # 2. 計算lower body的bone length(改為計算TPose時的向量就好, 他就包含了bone length的資訊)
    #   2.1 upper leg, lower leg兩個bone lengths(vectors)
    # 3. Store bone lengths and TPose positions

    # 1. 
    saveDirPath = 'positionData/fromDB/'
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
    saveDirPath='TPoseInfo/'
    # with open(saveDirPath+'TPosePositions.pickle', 'wb') as outPickle:
    #     pickle.dump(TPosePositions, outPickle)
    # with open(saveDirPath+'TPoseVectors.pickle', 'wb') as outPickle:
    #     pickle.dump(TPoseVectors, outPickle)
