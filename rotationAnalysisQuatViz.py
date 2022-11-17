'''
visualize rotationAnalysisQuaternion.py的計算結果
'''
import numpy as np 
import pickle
import os
import matplotlib.pyplot as plt 
import json
from rotationAnalysis import usedJointIdx
from rotationAnalysisViz import plotRotationCurveInARow, saveFigs, plotAutoCorrelation, \
    plotLinearMapFunc

quatIndex = [['x','y','z','w'], ['x','y','z','w'], ['x','y','z','w'], ['x','y','z','w']]

def readAllTheData(dataFilePath: str):
    def _readFile(fileNm):
        with open(os.path.join(dataFilePath, fileNm+'.pickle'), 'rb') as RFile:
            return pickle.load(RFile)
    handJointsRotations = _readFile('handOrigin')
    afterAdjRangeJointRots = _readFile('handAfterAdjRange')
    afterLowPassJointRots = _readFile('handAfterLowPass')
    filteredHandJointRots = _readFile('handAfterGaussian')
    quatJointRots  = _readFile('quatJointRots')
    quatGaussianRots = _readFile('quatGaussianRots')
    handAutoCorr =  _readFile('handAutoCorr')
    handAutoCorrLocalMaxInd = _readFile('handAutoCorrLocalMaxInd')
    # =======
    originBodyRot = _readFile('bodyOrigin')
    bodyJointRotations = _readFile('bodyAfterAdjRange')
    bodyQuatJointRots= _readFile('bodyQuatJointRots')
    # =======
    mappingFuncs = _readFile('mappingFuncs')
    handMinMax = _readFile('handMinMax')
    bodyMinMax = _readFile('bodyMinMax')
    
    
    return handJointsRotations, afterAdjRangeJointRots, afterLowPassJointRots, \
        filteredHandJointRots, quatJointRots, quatGaussianRots, handAutoCorr, \
        handAutoCorrLocalMaxInd, originBodyRot, bodyJointRotations, bodyQuatJointRots, \
        mappingFuncs, handMinMax, bodyMinMax

def visualizeLinearMapRes(dataFilePath, saveFigsFilePath):
    '''
    - Plot range adjust, low pass and gaussian results (hand)
    - Plot after converting to quaternion (hand) 
    '''

    # 1. read data
    handJointsRotations, afterAdjRangeJointRots, afterLowPassJointRots, \
        filteredHandJointRots, quatJointRots, quatGaussianRots, handAutoCorr, \
        handAutoCorrLocalMaxInd, originBodyRot, bodyJointRotations, bodyQuatJointRots, \
        mappingFuncs, handMinMax, bodyMinMax = readAllTheData(dataFilePath)
    ## Plot range adjust, low pass and gaussian results (hand)
    handPreprocFigs = plotRotationCurveInARow(
        [afterAdjRangeJointRots, afterLowPassJointRots, filteredHandJointRots],
        ['adjust range', 'low pass', 'gaussian'],
        usedJointIdx,
        'hand'
    )
    ## Plot after converting to quaternion (hand) 
    handQuatFigs = plotRotationCurveInARow(
        [quatJointRots, quatGaussianRots],
        ['quat', 'quat gaussian'],
        quatIndex,
        'hand quat'
    )
    saveFigs(handQuatFigs, os.path.join(saveFigsFilePath, 'quat'))
    ## Plot autocorrelation
    ## 濾除掉數值都是0的curve and autocorrelation, 之前分析時給None
    for _jointInd in handAutoCorr:
        for _axis in handAutoCorr[_jointInd]:
            if handAutoCorr[_jointInd][_axis] is None:
                handAutoCorr[_jointInd][_axis] = np.array([0, 0, 0])
                handAutoCorrLocalMaxInd[_jointInd][_axis] = 0
    autoCorrFigs = plotAutoCorrelation(handAutoCorr, handAutoCorrLocalMaxInd, quatIndex)
    saveFigs(autoCorrFigs, os.path.join(saveFigsFilePath, 'autoCorr'))
    ## Plot body rotation (in quat)
    bodyQuatFigs = plotRotationCurveInARow(
        [bodyQuatJointRots],
        ['body quat'],
        quatIndex,
        'body'
    )
    saveFigs(bodyQuatFigs, os.path.join(saveFigsFilePath, 'bodyQuat'))
    ## Plot mapping function
    ## 我需要hand與body的最大最小值, 才能標出那兩個點的位置
    ## curve是常數0的旋轉軸之前給None, 現在給0
    for _jointInd in range(len(mappingFuncs)):
        for _axis in mappingFuncs[_jointInd]:
            if mappingFuncs[_jointInd][_axis] is None: 
                mappingFuncs[_jointInd][_axis] = np.array([0, 0])
    mapFuncFigs = plotLinearMapFunc(
        mappingFuncs, handMinMax, bodyMinMax, quatIndex
    )
    saveFigs(mapFuncFigs, os.path.join(saveFigsFilePath, 'mappingFunc'))
    pass

if __name__=='__main__':
    visualizeLinearMapRes(
        dataFilePath='rotationMappingQuaternionData/leftFrontKick/', 
        saveFigsFilePath='rotationMappingQuaternionFigs/leftFrontKick/'
    )