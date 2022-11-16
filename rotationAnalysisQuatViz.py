'''
visualize rotationAnalysisQuaternion.py的計算結果
'''
import numpy as np 
import pickle
import os
import matplotlib.pyplot as plt 
import json
from rotationAnalysis import usedJointIdx
from rotationAnalysisViz import plotRotationCurveInARow, saveFigs, plotAutoCorrelation

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
    
    return handJointsRotations, afterAdjRangeJointRots, afterLowPassJointRots, \
        filteredHandJointRots, quatJointRots, quatGaussianRots, handAutoCorr, \
        handAutoCorrLocalMaxInd

def visualizeLinearMapRes(dataFilePath, saveFigsFilePath):
    '''
    - Plot range adjust, low pass and gaussian results (hand)
    - Plot after converting to quaternion (hand) 
    '''

    # 1. read data
    handJointsRotations, afterAdjRangeJointRots, afterLowPassJointRots, \
        filteredHandJointRots, quatJointRots, quatGaussianRots, handAutoCorr, \
        handAutoCorrLocalMaxInd = readAllTheData(dataFilePath)
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
    autoCorrFigs = plotAutoCorrelation(handAutoCorr, handAutoCorrLocalMaxInd, quatIndex)
    saveFigs(autoCorrFigs, os.path.join(saveFigsFilePath, 'autoCorr'))
    pass

if __name__=='__main__':
    visualizeLinearMapRes(
        dataFilePath='rotationMappingQuaternionData/leftFrontKick/', 
        saveFigsFilePath='rotationMappingQuaternionFigs/leftFrontKick/'
    )