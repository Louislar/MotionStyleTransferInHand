import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import json
from positionSynthesis import positionJsonDataParser, positionDataToPandasDf, positionDataPreproc, augFeatVecToPos, rollingWinSize, augmentationRatio

DBFullJointfilePath = './positionData/fromDB/leftFrontKickPositionFullJointsWithHead.json'
AfterSynthesisFullJointfilePath = './positionData/afterSynthesis/leftFrontKick_EWMA.json'
DBFullJointCount = 17
afterSynthesisFullJointCount = 16


def readInSynthesis3DPos(filePath, jointCount: int, fromUnity: bool=False):
    '''
    讀取存儲在json格式檔案當中的3d position資訊, 
    並且轉換成numpy array格式
    '''
    joints3dPosArr = [None for i in range(jointCount)]
    with open(filePath, 'r') as fileIn:
        jsonData=json.load(fileIn)
        if fromUnity:
            jsonData = jsonData['results']
        for i in range(jointCount):
            tmpArr = np.zeros((len(jsonData), 3))
            for t in range(len(jsonData)):
                tmpArr[t, 0] = jsonData[t]['data'][i]['x']
                tmpArr[t, 1] = jsonData[t]['data'][i]['y']
                tmpArr[t, 2] = jsonData[t]['data'][i]['z']
            joints3dPosArr[i] = tmpArr
    return joints3dPosArr
    
def readInDBPosByOldWay(DBFilePath, jointCount: int):
    '''
    使用positionSynthesis.py當中的方法讀取DB的positions
    '''
    posDBFullJointsDf = None
    with open(DBFilePath, 'r') as fileIn:
        jsonStr=json.load(fileIn)
        positionsDB = positionJsonDataParser(jsonStr, jointCount)
        posDBFullJointsDf = positionDataToPandasDf(positionsDB, jointCount)
    print(posDBFullJointsDf.shape)
    DBFullJointsPreproc = positionDataPreproc(posDBFullJointsDf, jointCount, rollingWinSize, True, augmentationRatio)
    DBFullJointsPosNoAug = [augFeatVecToPos(i.values, rollingWinSize) for i in DBFullJointsPreproc]
    return DBFullJointsPosNoAug

def DBPosSetHipOrigin(DBPosArrs):
    '''
    設定DB(from)Unity的position以Hip的座標為中心, 
    假設Hip的index是6

    Input:
    :DBPosArrs: 多個joints的position array構成的list
    '''
    for i in range(len(DBPosArrs)):
        if i != 6:
            DBPosArrs[i] = DBPosArrs[i] - DBPosArrs[6]
    DBPosArrs[6] = DBPosArrs[6] - DBPosArrs[6]
    return DBPosArrs

def findJointsAxesMinMax(PosArrs):
    '''
    尋找每個joint's axes的position最大最小值

    Input: 
    :PosArrs: 多個joints的position array構成的list
    '''
    minmaxInJoints = [[None, None] for i in range(len(PosArrs))]
    for i in range(len(PosArrs)):
        minmaxInJoints[i][0] = PosArrs[i].max(axis=0)
        minmaxInJoints[i][1] = PosArrs[i].min(axis=0)
    return minmaxInJoints

def plot3D(positions3d, isNewFigure:bool=False):
    '''
    繪製3D positions的trajectory

    Input: 
    :positions3d: np.array, 維度為(time point, 3)
    '''
    fig=None
    ax=None
    if isNewFigure:
        fig= plt.figure()
        ax = fig.add_subplot(projection='3d')
    plt.plot(positions3d[:, 0], positions3d[:, 1], positions3d[:, 2], '.-')
    

if __name__=='__main__':
    DBPos3dArrs = readInSynthesis3DPos(DBFullJointfilePath, DBFullJointCount, True)
    DBPos3dArrs = DBPosSetHipOrigin(DBPos3dArrs)
    AugDBPos = readInDBPosByOldWay(DBFullJointfilePath, DBFullJointCount)
    AfterSynthesisPos3dArrs = readInSynthesis3DPos(AfterSynthesisFullJointfilePath, afterSynthesisFullJointCount)
    
    # print(AugDBPos[0].shape)
    # print(DBPos3dArrs[0].shape)
    # print(AfterSynthesisPos3dArrs[0].shape)
    plot3D(AfterSynthesisPos3dArrs[2], True)
    plot3D(DBPos3dArrs[2], False)
    plot3D(AugDBPos[2], False)

    # Find min and max in each joint/axes
    DBPosMinMax = findJointsAxesMinMax(DBPos3dArrs)
    AfterSynthesisMinMax = findJointsAxesMinMax(AfterSynthesisPos3dArrs)
    for i in range(len(AfterSynthesisPos3dArrs)):
        if i >= 6:
            # print('Max: ', DBPosMinMax[i+1][0], ', ', AfterSynthesisMinMax[i][0])
            # print('Min: ', DBPosMinMax[i+1][1], ', ', AfterSynthesisMinMax[i][1])
            continue
        # print('Max: ', DBPosMinMax[i][0], ', ', AfterSynthesisMinMax[i][0])
        # print('Min: ', DBPosMinMax[i][1], ', ', AfterSynthesisMinMax[i][1])
        

    
    plt.show()
    