import numpy as np
import pandas as pd
import json
import os 

'''
Goal: Parse the position data in json format to .csv format. 
The output .csv format includes a matrix with each column represents a pose in a specific time, 
and a row represents a specific joint axis' values through the while time line. 
'''

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


if __name__=='__main__':
    # Read position data
    DBFileName = '../positionData/fromDB/leftFrontKickPosition.json'
    AfterMappingFileName = '../positionData/fromAfterMappingHand/leftFrontKickPosition(True, True, True, True, True, True).json'
    AfterMappingFileNames = [
        '../positionData/fromAfterMappingHand/leftFrontKickPosition(True, True, True, True, True, True).json', 
        '../positionData/fromAfterMappingHand/leftFrontKickPosition(True, False, True, True, True, True).json', 
        '../positionData/fromAfterMappingHand/leftFrontKickPosition(True, False, True, False, False, False).json'
    ]
    positionsJointCount = 7

    DBPosDf = readPosToDf(DBFileName, positionsJointCount)
    DBPosDf.to_csv('dataDBMatrix.csv', index=False, header=False)
    
    for _fileNM in AfterMappingFileNames:
        posDf = readPosToDf(_fileNM, positionsJointCount)
        _fileNM = os.path.basename(_fileNM)
        _fileNM = os.path.splitext(_fileNM)[0] + '.csv'
        posDf.to_csv(_fileNM, index=False, header=False)
