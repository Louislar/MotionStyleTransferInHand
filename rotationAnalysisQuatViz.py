'''
visualize rotationAnalysisQuaternion.py的計算結果
'''
import numpy as np 
import pickle
import os
import matplotlib.pyplot as plt 
import matplotlib as mpl
import json
from rotationAnalysis import usedJointIdx
from rotationAnalysisViz import plotRotationCurveInARow, saveFigs, plotAutoCorrelation, \
    plotLinearMapFunc, plotMultiSegSamplePts

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

def readBSplineData(dataFilePath: str):
    def _readFile(fileNm):
        with open(os.path.join(dataFilePath, fileNm+'.pickle'), 'rb') as RFile:
            return pickle.load(RFile)
    bodyQuatGaussian = _readFile('bodyQuatGaussian')
    bodyAutoCorr = _readFile('bodyAutoCorr')
    bodyJointFreq = _readFile('bodyJointFreq')
    bodySamplePointsArrs = _readFile('bodySamplePointsArrs')
    handSamplePointsArrs = _readFile('handSamplePointsArrs')
    handAvgSamplePts = _readFile('handAvgSamplePts')
    bodyAvgSamplePts = _readFile('bodyAvgSamplePts')
    handMapSamplePts = _readFile('handMapSamplePts')
    bodyMapSamplePts = _readFile('bodyMapSamplePts')
    handNormMapSamplePts = _readFile('handNormMapSamplePts')
    bodyNormMapSamplePts = _readFile('bodyNormMapSamplePts')

    return bodyQuatGaussian, bodyAutoCorr, bodyJointFreq, bodySamplePointsArrs, \
        handSamplePointsArrs, handAvgSamplePts, bodyAvgSamplePts, handMapSamplePts, \
        bodyMapSamplePts, handNormMapSamplePts, bodyNormMapSamplePts

# 畫B-Spline fitting前後的sample points 
def plotSamplePtsBeforeAfterBS(beforeBS, afterBS, usedJointIdx, labels=None):
    figs = []
    _defaultMarkerSize = mpl.rcParams['lines.markersize']
    for _jointInd in range(len(usedJointIdx)):
        for _axis in usedJointIdx[_jointInd]:
            fig = plt.figure(num='beforeAfterBS_{0}_{1}'.format(_jointInd, _axis))
            fig.suptitle('{0}_{1}'.format(_jointInd, _axis))
            ax = plt.subplot(111)
            ax.plot(
                beforeBS[0][_jointInd][_axis], 
                beforeBS[1][_jointInd][_axis], 
                '.',
                markersize=int(_defaultMarkerSize*2),
                label='before B-Spline fitting' if labels is None else labels[0]
            )
            ax.plot(
                afterBS[0][_jointInd][_axis], 
                afterBS[1][_jointInd][_axis], 
                '*',
                label='after B-Spline fitting' if labels is None else labels[1]
            )
            plt.legend()
            figs.append(fig)
    return figs

## visualize B-Spline mapping function建構過程的資料 
def vizBSplineMapFunc(dataFilePath, BSDataFilePath, saveFigsFilePath):
    # 1. read data and BSpline data
    handJointsRotations, afterAdjRangeJointRots, afterLowPassJointRots, \
        filteredHandJointRots, quatJointRots, quatGaussianRots, handAutoCorr, \
        handAutoCorrLocalMaxInd, originBodyRot, bodyJointRotations, bodyQuatJointRots, \
        mappingFuncs, handMinMax, bodyMinMax = readAllTheData(dataFilePath)
    bodyQuatGaussian, bodyAutoCorr, bodyJointFreq, bodySamplePointsArrs, \
        handSamplePointsArrs, handAvgSamplePts, bodyAvgSamplePts, handMapSamplePts, \
        bodyMapSamplePts, handNormMapSamplePts, bodyNormMapSamplePts  = readBSplineData(BSDataFilePath)

    # 2. plot body quat and after gaussian 
    bodyQuatFigs = plotRotationCurveInARow(
        [bodyQuatJointRots, bodyQuatGaussian],
        ['body quat', 'after gaussian'],
        quatIndex,
        'body'
    )
    saveFigs(bodyQuatFigs, os.path.join(saveFigsFilePath, 'bodyQuat'))

    # 3. Plot body auto correlation 
    ## 濾除掉數值都是0的curve and autocorrelation, 之前分析時給None
    for _jointInd in bodyAutoCorr:
        for _axis in bodyAutoCorr[_jointInd]:
            if bodyAutoCorr[_jointInd][_axis] is None:
                bodyAutoCorr[_jointInd][_axis] = np.array([0, 0, 0])
                bodyJointFreq[_jointInd][_axis] = 0
    bodyAutoCorrFigs = plotAutoCorrelation(bodyAutoCorr, bodyJointFreq, quatIndex)
    saveFigs(bodyAutoCorrFigs, os.path.join(saveFigsFilePath, 'bodyAutoCorr'))

    # 4. Plot inc and dec segments, also the average result 
    ## 濾除掉None的旋轉軸, 塞0資料. 為了visualize方便而已. 
    for _jointInd in range(len(bodyAvgSamplePts)):
        for _axis in bodyAvgSamplePts[_jointInd]:
            if handAvgSamplePts[_jointInd][_axis] is None: 
                handAvgSamplePts[_jointInd][_axis] = np.array([0, 0, 0])
                handSamplePointsArrs[0][_jointInd][_axis] = np.array([0, 0, 0])
                handSamplePointsArrs[1][_jointInd][_axis] = np.array([0, 0, 0])
                bodyAvgSamplePts[_jointInd][_axis] = np.array([0, 0, 0])
                bodySamplePointsArrs[0][_jointInd][_axis] = np.array([0, 0, 0])
                bodySamplePointsArrs[1][_jointInd][_axis] = np.array([0, 0, 0])
            
    ## hand 
    handMultiSegFigs = plotMultiSegSamplePts(
        handSamplePointsArrs, handAvgSamplePts, quatIndex, 'hand'
    )
    ## body 
    bodyMultiSegFigs = plotMultiSegSamplePts(
        bodySamplePointsArrs, bodyAvgSamplePts, quatIndex, 'body'
    )
    saveFigs(handMultiSegFigs, os.path.join(saveFigsFilePath, 'handSeg'))
    saveFigs(bodyMultiSegFigs, os.path.join(saveFigsFilePath, 'bodySeg'))
    
    # 5. Plot avg result再fit一次B-Spline的結果
    ## 注意, 原始資料點與fitting後的sample points都要畫出來
    ## 濾除都是0的None旋轉軸
    for _jointInd in range(len(handMapSamplePts)):
        for _axis in handMapSamplePts[_jointInd]:
            if handMapSamplePts[_jointInd][_axis] is None: 
                handMapSamplePts[_jointInd][_axis] = np.array([0, 0, 0])
                bodyMapSamplePts[_jointInd][_axis] = np.array([0, 0, 0])
    
    mapFuncBeforeAfterBSFigs = plotSamplePtsBeforeAfterBS(
        [handAvgSamplePts, bodyAvgSamplePts], 
        [handMapSamplePts, bodyMapSamplePts], 
        quatIndex
    )
    saveFigs(mapFuncBeforeAfterBSFigs, os.path.join(saveFigsFilePath, 'mapFuncBeforeAfterBS'))

    # 6. Plot normalization後的結果 
    for _jointInd in range(len(handNormMapSamplePts)):
        for _axis in handNormMapSamplePts[_jointInd]:
            if handNormMapSamplePts[_jointInd][_axis] is None: 
                handNormMapSamplePts[_jointInd][_axis] = np.array([0, 0, 0])
                bodyNormMapSamplePts[_jointInd][_axis] = np.array([0, 0, 0])
    mapFuncAfterNormFigs = plotSamplePtsBeforeAfterBS(
        [handMapSamplePts, bodyMapSamplePts], 
        [handNormMapSamplePts, bodyNormMapSamplePts], 
        quatIndex,
        ['before normalization', 'after normalization']
    )
    saveFigs(mapFuncAfterNormFigs, os.path.join(saveFigsFilePath, 'mapFuncAfterNorm')) 

# 將兩種不同方法的mapping function畫在一起比較差異
def vizDiffMapFunc(dataFilePath, BSDataFilePath, saveFigsFilePath):
    '''
    1. read data (linear, B-Spline)
    2. plot two mapping function in a single figure
    3. store figures
    '''
    # 1. 
    handJointsRotations, afterAdjRangeJointRots, afterLowPassJointRots, \
        filteredHandJointRots, quatJointRots, quatGaussianRots, handAutoCorr, \
        handAutoCorrLocalMaxInd, originBodyRot, bodyJointRotations, bodyQuatJointRots, \
        mappingFuncs, handMinMax, bodyMinMax = readAllTheData(dataFilePath)
    bodyQuatGaussian, bodyAutoCorr, bodyJointFreq, bodySamplePointsArrs, \
        handSamplePointsArrs, handAvgSamplePts, bodyAvgSamplePts, handMapSamplePts, \
        bodyMapSamplePts, handNormMapSamplePts, bodyNormMapSamplePts  = readBSplineData(BSDataFilePath)
    # 1.1 處理None旋轉軸資料
    ## linear mapping function
    for _jointInd in range(len(mappingFuncs)):
        for _axis in mappingFuncs[_jointInd]:
            if mappingFuncs[_jointInd][_axis] is None: 
                mappingFuncs[_jointInd][_axis] = np.array([0, 0])
    ## B-Spline mapping function sample points 
    for _jointInd in range(len(handNormMapSamplePts)):
        for _axis in handNormMapSamplePts[_jointInd]:
            if handNormMapSamplePts[_jointInd][_axis] is None: 
                handNormMapSamplePts[_jointInd][_axis] = np.array([0, 0, 0])
                bodyNormMapSamplePts[_jointInd][_axis] = np.array([0, 0, 0])

    # 2. 
    figs = []
    for _jointInd in range(len(quatIndex)):
        for _axis in quatIndex[_jointInd]:
            fittedLine = np.poly1d(mappingFuncs[_jointInd][_axis])
            _x = np.linspace(handMinMax[0][_jointInd][_axis], handMinMax[1][_jointInd][_axis])
            _y = fittedLine(_x)

            fig = plt.figure(num='two mapping func_{0}_{1}'.format(_jointInd, _axis))
            fig.suptitle('{0}_{1}'.format(_jointInd, _axis))
            ax = plt.subplot(111)
            ax.plot(
                handNormMapSamplePts[_jointInd][_axis],
                bodyNormMapSamplePts[_jointInd][_axis],
                '.-',
                label='B-Spline'
            )
            ax.plot(_x, _y, label='linear function')
            ax.plot(
                [handMinMax[0][_jointInd][_axis], handMinMax[1][_jointInd][_axis]],
                [bodyMinMax[0][_jointInd][_axis], bodyMinMax[1][_jointInd][_axis]],
                '.r',
                label='linear function min max'
            )
            figs.append(fig)
    # 3. 
    saveFigs(figs, os.path.join(saveFigsFilePath, 'compareMapFunc'))

## 畫出linear and B-Spline mapping後的rotation以及mapping前的rotation
def vizDiffApplyResult(dataFilePath, saveFigsFilePath):
    '''
    1. read all mapping results (include the old ones)
    2. visualize 
    3. store visualization figures 
    '''
    # 1. 
    def _readFile(fileNm):
        with open(os.path.join(dataFilePath, fileNm+'.pickle'), 'rb') as RFile:
            return pickle.load(RFile)
    handLinearMappedRot = _readFile('linearMappedRot')
    bodyLinearMappedRot = _readFile('BSMappedRot')
    # 2. 
    def plotNewFig(data, dataName):
        _fig = plt.figure()
        for _d, _n in zip(data, dataName):
            plt.plot(range(len(_d)), _d, label=_n)
        plt.legend()
        return _fig
    ## 全部的旋轉軸都要畫圖
    figs=[]
    for _jointInd in range(len(quatIndex)):
        for _axis in quatIndex[_jointInd]:
            _fig = plotNewFig(
                [handLinearMappedRot[_jointInd][_axis], bodyLinearMappedRot[_jointInd][_axis]],
                ['linear', 'B-Spline']
            )
            _fig.suptitle('{0}_{1}'.format(_jointInd, _axis))
            figs.append(_fig)
    saveFigs(figs, os.path.join(saveFigsFilePath, 'applyDiffMapFunc'))
    pass

if __name__=='__main__':
    # visualizeLinearMapRes(
    #     dataFilePath='rotationMappingQuaternionData/leftFrontKick/', 
    #     saveFigsFilePath='rotationMappingQuaternionFigs/leftFrontKick/'
    # )
    # vizBSplineMapFunc(
    #     dataFilePath='rotationMappingQuaternionData/leftFrontKick/', 
    #     BSDataFilePath='rotationMappingQuaternionData/leftFrontKickBSpline/',
    #     saveFigsFilePath='rotationMappingQuaternionFigs/leftFrontKickBSpline/'
    # )
    # vizDiffMapFunc(
    #     dataFilePath='rotationMappingQuaternionData/leftFrontKick/', 
    #     BSDataFilePath='rotationMappingQuaternionData/leftFrontKickBSpline/',
    #     saveFigsFilePath='rotationMappingQuaternionFigs/leftFrontKick/'
    # )
    vizDiffApplyResult(
        dataFilePath='rotationMappingQuaternionData/leftFrontKick/', 
        saveFigsFilePath='rotationMappingQuaternionFigs/leftFrontKick/'
    )