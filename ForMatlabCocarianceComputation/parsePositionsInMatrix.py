from turtle import left
import numpy as np
import pandas as pd
import json
import os 
import enum
import itertools

'''
Goal: Parse the position data in json format to .csv format. 
The output .csv format includes a matrix with each column represents a pose in a specific time, 
and a row represents a specific joint axis' values through the while time line. 

TODO: 左右腳的position分開輸出成獨立的dataframe/.csv files
TODO: joint displacement計算
'''

class jointsNames(enum.IntEnum):
    LeftUpperLeg = 0
    LeftLowerLeg = 1
    LeftFoot = 2
    RightUpperLeg = 3
    RightLowerLeg = 4
    RightFoot = 5
    Hip = 6

leftRightJointPairs=[
    [1, 2], [4, 5]
]

# 計算多種jointPairs的displacement
displacmentJointPairs = [
    [0, 2], [3, 5], [2, 5]
]

def positionJsonDataParser(jsonDict: dict, jointCount: int):
    '''
    target format: 
    x list:[x1, x2, x3, ...]
    knee: {x list, y list, z list}
    left upper leg, left knee, right upper leg, right knee
    '''
    timeSeries=jsonDict['results']
    parsedPositionData=[{'x': [], 'y': [], 'z': []} for i in range(jointCount)]
    for jointIdx in range(jointCount):
        for oneData in timeSeries:
            parsedPositionData[jointIdx]['x'].append(oneData['data'][jointIdx]['x'])
            parsedPositionData[jointIdx]['y'].append(oneData['data'][jointIdx]['y'])
            parsedPositionData[jointIdx]['z'].append(oneData['data'][jointIdx]['z'])
    return parsedPositionData

def positionDataToPandasDf(parsedPosData, jointCount: int):
    '''
    Convert parsed position data to pandas DataFrame
    Columns(left to right): | joint 1 x | joint 1 y | joint 1 z | joint 2 x | ... |
    Rows(Top to bottom): | 1st sample pt | 2nd sample pt | ... |
    '''
    posDf = pd.DataFrame()
    for jointIdx in range(jointCount):
        for k, _data in parsedPosData[jointIdx].items():
            posDf['{0}_{1}'.format(jointIdx, k)] = _data
    return posDf

def setHipAsOrigin(posDf, jointCount: int):
    '''
    Set hip position as origin (0, 0, 0)
    '''
    axesStr = ['x', 'y', 'z']
    for i in range(jointCount):
        for _axis in axesStr:
            posDf.loc[:, '{0}_'.format(i)+_axis] = \
                posDf['{0}_'.format(i)+_axis] - posDf['{0}_'.format(jointsNames.Hip)+_axis]
    return posDf

def readPosToDf(fileName, posJointCount):
    '''
    使用檔名讀取position資料為DataFrame
    '''
    ## Read Position data 
    posDf = None
    with open(fileName, 'r') as fileIn:
        jsonStr=json.load(fileIn)
        positionsData = positionJsonDataParser(jsonStr, posJointCount)

        ### Position data to dataframe
        posDf = positionDataToPandasDf(positionsData, posJointCount)
        return posDf

def divideLeftRightPosData(posDf):
    '''
    Input: 
    :posDf: readPosToDf()的輸出
    :leftRightJointPairs: global variable
    '''
    leftRightDfList = []
    for _joints in leftRightJointPairs:
        tmpSrList = []
        for j in _joints:
            j = j * 3
            tmpSrList.append(posDf.iloc[:, j:j+3])
        leftRightDfList.append(pd.concat(tmpSrList, axis=1))
    return leftRightDfList

def computeJointPairsDisplacments(posDf):
    '''
    Input: 
    :posDf: readPosToDf()的輸出
    :leftRightJointPairs: global variable
    '''
    displacementSrList = []
    for _pair in displacmentJointPairs:
        tmpSrList = []
        for j in _pair:
            j = j * 3
            tmpSrList.append(posDf.iloc[:, j:j+3])
        displacementSrList.append(tmpSrList[1]-tmpSrList[0].values)
    displacementDf = pd.concat(displacementSrList, axis=1)
    return displacementDf



if __name__=='__main__':
    # Read position data
    # DBFileName = '../positionData/fromDB/leftFrontKickPosition.json'
    # DBFileName = '../positionData/fromDB/leftSideKickPositionFullJointsWithHead.json'
    # DBFileName = '../positionData/fromDB/walkCrossoverPositionFullJointsWithHead.json'
    # DBFileName = '../positionData/fromDB/walkInjuredPositionFullJointsWithHead.json'
    DBFileName = '../positionData/fromDB/runSprintPositionFullJointsWithHead.json'
    # AfterMappingFileName = '../positionData/fromAfterMappingHand/leftFrontKickPosition(True, True, True, True, True, True).json'
    trueFalseValue = list(itertools.product([True, False], repeat=6))
    # baseFileName = '../positionData/fromAfterMappingHand/leftFrontKickCombinations/leftFrontKick{0}.json'
    # baseFileName = '../positionData/fromAfterMappingHand/leftSideKickCombinations/leftSideKick{0}.json'
    # baseFileName = '../positionData/fromAfterMappingHand/walkCrossoverCombinations/walkCrossover{0}.json'
    # baseFileName = '../positionData/fromAfterMappingHand/walkInjuredCombinations/walkInjured{0}.json'
    baseFileName = '../positionData/fromAfterMappingHand/runSprintCombinations/runSprint{0}.json'
    AfterMappingFileNames = [
        baseFileName.format(str(_tfComb)) for _tfComb in trueFalseValue
    ]
    # AfterMappingFileNames = [
    #     '../positionData/fromAfterMappingHand/leftFrontKickCombinations/leftFrontKickPosition(True, True, True, True, True, True).json', 
    #     '../positionData/fromAfterMappingHand/leftFrontKickCombinations/leftFrontKickPosition(True, False, True, True, True, True).json', 
    #     '../positionData/fromAfterMappingHand/leftFrontKickCombinations/leftFrontKickPosition(True, False, True, False, False, False).json'
    # ]
    positionsJointCount = 7

    DBPosDf = readPosToDf(DBFileName, positionsJointCount)
    DBPosDf = setHipAsOrigin(DBPosDf, positionsJointCount)
    # DBPosDf.to_csv('dataDBMatrix.csv', index=False, header=False)
    # separate left and right
    leftAndRightDBDf = divideLeftRightPosData(DBPosDf)
    # leftAndRightDBDf[0].to_csv('./leftLeg/dataDBMatrix.csv', index=False, header=False)
    # leftAndRightDBDf[1].to_csv('./rightLeg/dataDBMatrix.csv', index=False, header=False)

    # Use displacement as feature
    displacementDBDf = computeJointPairsDisplacments(DBPosDf)
    # displacementDBDf.to_csv('displacement/leftFrontKick/displacementDBMatrix.csv', index=False, header=False)
    # displacementDBDf.to_csv('displacement/leftSideKick/displacementDBMatrix.csv', index=False, header=False)
    # displacementDBDf.to_csv('displacement/walkCrossover/displacementDBMatrix.csv', index=False, header=False)
    # displacementDBDf.to_csv('displacement/walkInjured/displacementDBMatrix.csv', index=False, header=False)
    displacementDBDf.to_csv('displacement/runSprint/displacementDBMatrix.csv', index=False, header=False)
    
    
    for _fileNM in AfterMappingFileNames:
        posDf = readPosToDf(_fileNM, positionsJointCount)
        posDf = setHipAsOrigin(posDf, positionsJointCount)
        # separate left and right joints data
        leftAndRightDf = divideLeftRightPosData(posDf)
        # Use displacement as feature
        displacementDf = computeJointPairsDisplacments(posDf)
        _fileNM = os.path.basename(_fileNM)
        _fileNM = os.path.splitext(_fileNM)[0] + '.csv'
        # posDf.to_csv(_fileNM, index=False, header=False)
        # leftAndRightDf[0].to_csv('./leftLeg/'+_fileNM, index=False, header=False)
        # leftAndRightDf[1].to_csv('./rightLeg/'+_fileNM, index=False, header=False)
        # displacementDf.to_csv('displacement/leftFrontKick/displacement'+_fileNM, index=False, header=False)
        # displacementDf.to_csv('displacement/leftSideKick/displacement'+_fileNM, index=False, header=False)
        # displacementDf.to_csv('displacement/walkCrossover/displacement'+_fileNM, index=False, header=False)
        # displacementDf.to_csv('displacement/walkInjured/displacement'+_fileNM, index=False, header=False)
        displacementDf.to_csv('displacement/runSprint/displacement'+_fileNM, index=False, header=False)
        print(_fileNM)

