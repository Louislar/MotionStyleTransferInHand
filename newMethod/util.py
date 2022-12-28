'''
各式雜項功能
'''

import numpy as np 
import matplotlib.pyplot as plt 
import json 
import sys
import copy
sys.path.append("../")
from rotationAnalysis import rotationJsonDataParser 

def readHandPerformance(filePath = '../HandRotationOuputFromHomePC/leftFrontKickStream.json'):
    '''
    讀取hand performance rotation data 
    '''
    handJointsRotations=None
    with open(filePath, 'r') as fileOpen: 
        rotationJson=json.load(fileOpen)
        handJointsRotations = rotationJsonDataParser({'results': rotationJson}, jointCount=4)    # For python output
    return handJointsRotations

def cropHandPerformance(handPerformanceData, startInd, endInd): 
    numJoint = len(handPerformanceData)
    catAxis = handPerformanceData[0].keys()
    # timeCount = len(handPerformanceData[0]['x'])

    data = copy.deepcopy(handPerformanceData) 
    for _joint in range(numJoint):
        for _axis in catAxis:
            data[_joint][_axis] = data[_joint][_axis][startInd:endInd+1]
    return data

def handPerformanceToMatrix(handPerformanceData, usedJointAxis):
    '''
    Make hand rotation data to a matrix with d * t dimension 
    每一行是所有維度的資料, 依序是joint 0 x, joint 0 y, joint 0 z, joint 1 x, .....
    總共的行數是多少個時間點 
    :usedJointAxis: 使用到的joint axis. List of pairs of joint and axis. e.g. [[0, 'x'], [2, 'y']]
    '''
    numFeat = len(usedJointAxis)
    timeCount = len(handPerformanceData[0]['x'])
    targetMat = np.zeros((numFeat, timeCount))
    for i, _pair in enumerate(usedJointAxis):
        _joint=_pair[0]
        _axis=_pair[1]
        targetMat[i, :] = handPerformanceData[_joint][_axis]
    return targetMat

def scaleToReferenceControlSequence(signal, referenceControlSequence):
    '''
    使用min max scaling將input signal根據reference control sequence轉換到[-1, 1]
    大於或是小於reference control sequence的數值, 使用control sequence的最大最小值代替
    '''
    refMin = np.min(referenceControlSequence)
    refMax = np.max(referenceControlSequence)
    signal = np.copy(signal)
    signal[signal>refMax] = refMax
    signal[signal<refMin] = refMin
    signal = -1 + (signal-refMin)*(1-(-1)) / (refMax-refMin)
    return signal

if __name__=='__main__':
    data = readHandPerformance()
    cropData = cropHandPerformance(data, 1430, 1530)
    dataMat = handPerformanceToMatrix(cropData, [[0, 'x'], [1, 'x']])
    print(dataMat.shape)
    pass 