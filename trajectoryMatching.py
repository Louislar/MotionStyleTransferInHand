'''
利用transformation matrix, 
將hand mapped trajectory以及body trajectory對齊
最後, 將對齊前後的body trajectory以及hand mapped trajectory畫出來
'''

import pandas as pd 
import numpy as np 
from scipy.spatial.transform import Rotation
import json 
import os 
import copy
import matplotlib.pyplot as plt 
from testingStageViz import jsonToDf
from realTimePositionSynthesis import readDBEncodedMotionsFromFile, fullPositionsJointCount

def dfToJson(dfs):
    '''
    將dataframes in dict轉換成json
    '''
    jointsNm = list(dfs.keys())
    timeCount = dfs[jointsNm[0]].shape[0]
    jsonData = []
    for t in range(timeCount):
        _jointsDataInSingleTime = []
        for _jointInd in jointsNm:
            _jointsDataInSingleTime.append(
                dfs[_jointInd].iloc[t, :].to_dict()
            )
        jsonData.append(
            {
                'time': t,
                'data': _jointsDataInSingleTime
            }
        )
    jsonData = {'results': jsonData}
    return jsonData

# Construct transformation matrix and apply it to positions/trajectory
# input rotation in x, y, z order    
# input transition in x, y, z order 
def constructAndApplyTransMat(
    handMappedPosDirPath = 'positionData/fromAfterMappingHand/',
    body3dPosDirPath = 'DBPreprocFeatVec/leftFrontKick_withoutHip_075/3DPos/',
    rotationAngles = [0, 0, 0],
    translationValues = [0.1, 0, 0]
):
    # 1. read hand mapped positions
    # 2. read body positions
    # 3. construct transformation matrix
    ## ref: https://towardsdatascience.com/the-one-stop-guide-for-transformation-matrices-cea8f609bdb1
    # 3.1 apply transformation matrix to body trajectory
    # 4. store transformed result 
    
    # 1. 
    handMappedPosJson = None
    with open(os.path.join(handMappedPosDirPath, 'leftFrontKickStreamLinearMapping_TFFTTT.json'), 'r') as RFile: 
        handMappedPosJson = json.load(RFile)
    # joint key value改為數值而非string
    for t in range(len(handMappedPosJson)):
        _newDict = {}
        for k, v in handMappedPosJson[t]['data'].items():
            _newDict[int(k)]=v
        handMappedPosJson[t]['data']=_newDict
    # print(handMappedPosJson[1])
    handMappedPosJson = jsonToDf(handMappedPosJson)
    # print(handMappedPosJson[1])

    # 2. 
    DBPreproc3DPos = readDBEncodedMotionsFromFile(fullPositionsJointCount, body3dPosDirPath)
    ## to dataframe
    bodyJoint3dPos = {
        _jointInd: pd.DataFrame(
            DBPreproc3DPos[_jointInd],
            columns=['x','y','z'],
            index=range(DBPreproc3DPos[_jointInd].shape[0])
        ) for _jointInd in range(fullPositionsJointCount)
    }
    # print(bodyJoint3dPos[0])

    # 3. 
    transMat = np.eye(4)
    R = Rotation.from_euler("XYZ", rotationAngles, degrees=True).as_matrix()
    transMat[:3,:3] = R
    transMat[:3,3] = np.array(translationValues)
    # 3.1
    def _applyTransMat(vec: pd.Series, transMat: np.array):
        vecNp = vec.values
        vecNp = np.append(vecNp, 1)
        vecNp = np.dot(transMat, vecNp)
        return pd.Series(vecNp[:-1], index=vec.index)

    transformedBodyJoint3dPos = {} #copy.deepcopy(bodyJoint3dPos)
    for _jointInd in range(fullPositionsJointCount):
        transformedBodyJoint3dPos[_jointInd] = bodyJoint3dPos[_jointInd].apply(
            lambda _aRow: _applyTransMat(_aRow, transMat), 
            axis=1
        )
    # print(bodyJoint3dPos[0])
    # print(transformedBodyJoint3dPos[0])

    # 4. 
    transformedPosDirPath = 'transformedPosData/leftFrontKick_withoutHip_075/'
    for _jointInd in range(fullPositionsJointCount):
        transformedBodyJoint3dPos[_jointInd].to_csv(
            os.path.join(transformedPosDirPath, '{0}.csv'.format(_jointInd)),
            index=False
        )

# visualize trajectory
# (including before and after applying transformation trjectory and hand trajectory)
def visualizeTransResult(
    handMappedPosDirPath = 'positionData/fromAfterMappingHand/',
    body3dPosDirPath = 'DBPreprocFeatVec/leftFrontKick_withoutHip_075/3DPos/',
    transformed3dPosDirPath = 'transformedPosData/leftFrontKick_withoutHip_075/'
):
    # 1. read hand mapped positions
    # 2. read body positions
    # 3. read transformed body positions
    # 4. visualize
    
    # 1. 
    handMappedPosJson = None
    with open(os.path.join(handMappedPosDirPath, 'leftFrontKickStreamLinearMapping_TFFTTT.json'), 'r') as RFile: 
        handMappedPosJson = json.load(RFile)
    # joint key value改為數值而非string
    for t in range(len(handMappedPosJson)):
        _newDict = {}
        for k, v in handMappedPosJson[t]['data'].items():
            _newDict[int(k)]=v
        handMappedPosJson[t]['data']=_newDict
    # print(handMappedPosJson[1])
    handMappedPos = jsonToDf(handMappedPosJson)
    # print(handMappedPos[1])
    # print(handMappedPos[6])
    ## 需要使用校正hip為原點
    for _jointInd in handMappedPos.keys():
        if _jointInd != 6:
            handMappedPos[_jointInd] = handMappedPos[_jointInd] - handMappedPos[6]
    handMappedPos[6] = handMappedPos[6] - handMappedPos[6]
    # print(handMappedPos[1])
    # 2. 
    DBPreproc3DPos = readDBEncodedMotionsFromFile(fullPositionsJointCount, body3dPosDirPath)
    ## to dataframe
    bodyJoint3dPos = {
        _jointInd: pd.DataFrame(
            DBPreproc3DPos[_jointInd],
            columns=['x','y','z'],
            index=range(DBPreproc3DPos[_jointInd].shape[0])
        ) for _jointInd in range(fullPositionsJointCount)
    }
    # 3. 
    transformedPos = {}
    for _jointInd in range(fullPositionsJointCount):
        transformedPos[_jointInd] = pd.read_csv(
            os.path.join(transformed3dPosDirPath, '{0}.csv'.format(_jointInd))
        )
    
    # 4. 
    vizJointInd = 2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.plot(
        handMappedPos[2]['x'],
        handMappedPos[2]['y'],
        handMappedPos[2]['z'],
        '.',
        label = 'mapped position'
    )
    ax.plot(
        bodyJoint3dPos[2]['x'],
        bodyJoint3dPos[2]['y'],
        bodyJoint3dPos[2]['z'],
        '.',
        label='original body trajectory'
    )
    ax.plot(
        transformedPos[2]['x'],
        transformedPos[2]['y'],
        transformedPos[2]['z'],
        '.',
        label='transformed body trajectory'
    )
    plt.legend()
    plt.show()
    pass

# construct transformation matrix (4x4 matrix)
# input rotation in x, y, z order    
# input transition in x, y, z order 
def constructTransMat(
    rotationAngles=[0,0,0],
    translationValues=[0.3,0,0.2]
):
    transMat = np.eye(4)
    R = Rotation.from_euler("XYZ", rotationAngles, degrees=True).as_matrix()
    transMat[:3,:3] = R
    transMat[:3,3] = np.array(translationValues)
    return transMat

# 對所有body joint的點apply transformation matrix
# input rotation in x, y, z order    
# input transition in x, y, z order 
def applyTransMatToEntireAnimation(
    bodyPosFilePath = 'positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withoutHip_075.json',
    transformedPosOutputFilePath = 'positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withoutHip_075_transformed.json',
    rotationAngles=[0,0,0],
    translationValues=[0.4,0,0.2]
):
    # 1. read body positions
    # 2. construct and apply transformation (matrix)
    ## Warning: Do not transform the body joint position since it is the origin. 
    ## And other processes/function 會將origin歸0, 導致transformation失效
    # 3. store transformed body positions

    # 1. 
    posDBDf = None
    with open(os.path.join(bodyPosFilePath), 'r') as fileIn:
        jsonStr=json.load(fileIn)['results']
        posDBDf=jsonToDf(jsonStr)
    # print(len(posDBDf))
    # print(posDBDf[0])
    # 2. 
    transMat = constructTransMat(rotationAngles, translationValues)
    def _applyTransMat(vec: pd.Series, transMat: np.array):
        vecNp = vec.values  # convert to NumPy vector
        vecNp = np.append(vecNp, 1)
        vecNp = np.dot(transMat, vecNp)
        return pd.Series(vecNp[:-1], index=vec.index)
    transformedBodyPos = {}
    for _jointInd in range(len(posDBDf)):
        if _jointInd != 6:
            transformedBodyPos[_jointInd] = posDBDf[_jointInd].apply(
                lambda _aRow: _applyTransMat(_aRow, transMat),
                axis=1
            )
        else: 
            transformedBodyPos[_jointInd] = posDBDf[_jointInd]
    # 3. 
    transformedBodyPosJson = dfToJson(transformedBodyPos)
    with open(transformedPosOutputFilePath, 'w') as WFile:
        json.dump(transformedBodyPosJson, WFile)

if __name__=='__main__':
    ## construct and apply transformation matrix to body positions/trajectory
    constructAndApplyTransMat(
        rotationAngles=[0,0,0],
        translationValues=[0.4,0,0.2]
    )
    ## visualize transformation applying result
    visualizeTransResult()
    ## apply tranformation to body positions in DB
    # applyTransMatToEntireAnimation(
    #     bodyPosFilePath = 'positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withoutHip_075.json',
    #     transformedPosOutputFilePath = 'positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withoutHip_075_transformed.json',
    #     rotationAngles=[0,0,0],
    #     translationValues=[0.4,0,0.2]
    # )
    pass
