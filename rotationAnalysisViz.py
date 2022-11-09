'''
visualize rotationAnalysisNew.py 輸出的資料
'''

import numpy as np 
import pickle
import os
import matplotlib.pyplot as plt 
from rotationAnalysis import usedJointIdx


def readAllTheData(dataFilePath: str):
    def _readFile(fileNm):
        with open(os.path.join(dataFilePath, fileNm+'.pickle'), 'rb') as RFile:
            return pickle.load(RFile)
    
    handJointsRotations = _readFile('handOrigin')
    afterAdjRangeJointRots = _readFile('handAfterAdjRange')
    afterLowPassJointRots = _readFile('handAfterLowPass')
    filteredHandJointRots = _readFile('handAfterGaussian')
    
    jointsACorr = _readFile('handAutoCorrelation')
    jointsACorrLocalMaxIdx = _readFile('handAutoCorrelationLocalMaxIdx')

    handJointsPatternData = _readFile('handJointsPatternData')

    originBodyRot = _readFile('bodyOrigin')
    bodyAfterRangeAdj = _readFile('bodyAfterAdjRange')
    bodyAfterGaussian = _readFile('bodyAfterGaussian')

    mappingFuncs = _readFile('mappingFuncs')
    handMinMax = _readFile('handMinMax')
    bodyMinMax = _readFile('bodyMinMax')

    return handJointsRotations, afterAdjRangeJointRots, afterLowPassJointRots, filteredHandJointRots, \
        jointsACorr, jointsACorrLocalMaxIdx, handJointsPatternData, originBodyRot, bodyAfterRangeAdj, \
           bodyAfterGaussian, mappingFuncs, handMinMax, bodyMinMax

def plotRotationCurveInARow(rotCurves, curveNms, usedJointIdx, figNm=''):
    '''
    原始rotation, range adjust, low pass, gaussian. 
    以上所有訊號都垂直plot在同一個figure
    注意, 每個joint, axis是畫在不同的figure當中
    Input:
    :rots: (list) 多個joint以及axis的原始rotation curve
                list的element是不同步驟的rotation curve
    '''
    numCurves = len(rotCurves)
    figs = []
    
    for _jointInd in range(len(usedJointIdx)):
        for _axis in usedJointIdx[_jointInd]:
            fig = plt.figure(num='{0}_preproc_{1}_{2}'.format(figNm, _jointInd, _axis))
            axs = []
            for i in range(1, numCurves+1):
                axs.append(plt.subplot(numCurves,1,i))
            for i in range(numCurves):
                axs[i].set_title(curveNms[i])
                axs[i].plot(range(len(rotCurves[i][_jointInd][_axis])), rotCurves[i][_jointInd][_axis])
            figs.append(fig)
    # plt.show()
    return figs

def plotAutoCorrelation(autoCorrCurves, autoCorrLocalMaxIdx, usedJointIdx):
    '''
    顯示一張圖片, 需要標上auto correlation找到的frequency位置 (或是/並且在明顯的地方顯示數值)
    auto correlation
    Input: 
    :autoCorrCurves: (list) 多個joint以及axis的autocorrelation數值
    :autoCorrLocalMaxIdx: (list) 多個joint以及axis的autocorrelation第一個正數波峰
    '''
    figs = []
    for _jointInd in range(len(usedJointIdx)):
        for _axis in usedJointIdx[_jointInd]:
            fig = plt.figure(num='autocorr_{0}_{1}_frequency_{2}'.format(_jointInd, _axis, autoCorrLocalMaxIdx[_jointInd][_axis]))
            ax = plt.subplot(111)
            ax.plot(range(len(autoCorrCurves[_jointInd][_axis])), autoCorrCurves[_jointInd][_axis])
            ax.plot(autoCorrLocalMaxIdx[_jointInd][_axis], autoCorrCurves[_jointInd][_axis][autoCorrLocalMaxIdx[_jointInd][_axis]], '.r')
            figs.append(fig)            
    # plt.show()
    return figs
    pass

def plotLinearMapFunc(mapFuncParam, handMinMax, bodyMinMax, usedJointIdx):
    '''
    plot linear mapping function with hand and body minimum and maximum point
    '''
    figs = []
    for _jointInd in range(len(usedJointIdx)):
        for _axis in usedJointIdx[_jointInd]:
            fittedLine = np.poly1d(mapFuncParam[_jointInd][_axis])
            _x = np.linspace(handMinMax[0][_jointInd][_axis], handMinMax[1][_jointInd][_axis])
            _y = fittedLine(_x)

            fig = plt.figure(num='map func_{0}_{1}'.format(_jointInd, _axis))
            ax = plt.subplot(111)
            ax.plot(_x, _y)
            ax.plot(
                [handMinMax[0][_jointInd][_axis], handMinMax[1][_jointInd][_axis]],
                [bodyMinMax[0][_jointInd][_axis], bodyMinMax[1][_jointInd][_axis]],
                '.r'
            )

            figs.append(fig)
    return figs

    pass

def saveFigs(figs, filePath):
    if not os.path.isdir(filePath):
        os.makedirs(filePath)
    for _fig in figs:
        _fig.savefig(os.path.join(
            filePath, '{0}.png'.format(_fig.number)
        ))
        plt.close(_fig)
    pass

def vizLinearMapResult(
    dataFilePath = 'rotationMappingData/leftFrontKick/', 
    saveFigsFilePath = 'rotationMappingFigs/leftFrontKick/'
):
    handJointsRotations, afterAdjRangeJointRots, afterLowPassJointRots, filteredHandJointRots, \
        jointsACorr, jointsACorrLocalMaxIdx, handJointsPatternData, \
            originBodyRot, bodyAfterRangeAdj, bodyAfterGaussians, \
                mappingFuncs, handMinMax, bodyMinMax = readAllTheData(dataFilePath)
    ## Plot range adjust, low pass and gaussian results (hand)
    handPreprocFigs = plotRotationCurveInARow(
        [afterAdjRangeJointRots, afterLowPassJointRots, filteredHandJointRots],
        ['adjust range', 'low pass', 'gaussian'],
        usedJointIdx,
        'hand'
    )
    ## Plot auto correlation result
    autoCorrFigs = plotAutoCorrelation(
        jointsACorr, jointsACorrLocalMaxIdx, usedJointIdx
    )
    ## Plot body rotation 
    bodyRotFigs = plotRotationCurveInARow(
        [originBodyRot, bodyAfterRangeAdj, bodyAfterGaussians],
        ['origin body rotation', 'after range adjustment', 'after gaussian'],
        usedJointIdx,
        'body'
    )
    ## Plot mapping function
    # 我需要hand與body的最大最小值, 才能標出那兩個點的位置
    mapFuncFigs = plotLinearMapFunc(
        mappingFuncs, handMinMax, bodyMinMax, usedJointIdx
    )

    # plt.show()
    ## Store figures
    saveFigs(handPreprocFigs, os.path.join(saveFigsFilePath, 'handPreproc'))
    ## ------- 
    saveFigs(autoCorrFigs, os.path.join(saveFigsFilePath, 'autoCorr'))
    ## ------- 
    saveFigs(bodyRotFigs, os.path.join(saveFigsFilePath, 'bodyPreproc'))
    ## ------- 
    saveFigs(mapFuncFigs, os.path.join(saveFigsFilePath, 'mappingFunc'))

    # print(afterAdjRangeJointRots[0]['x'])
    # print(afterLowPassJointRots[0]['x'])
    # print(jointsACorrLocalMaxIdx[0]['x'])
    pass

def readBSplineData(dataFilePath: str):
    '''
    讀取B-Spline fitting多計算的一些資料 
    '''
    def _readFile(fileNm):
        with open(os.path.join(dataFilePath, fileNm+'.pickle'), 'rb') as RFile:
            return pickle.load(RFile)
    # TODO: 
    # _outputData(bodyJointRotations, 'bodyAutoCorrelation')
    # _outputData(bodyRepeatPatternCycle, 'bodyRepeatPatternCycle')
    # _outputData(handAvgSamplePts, 'handAvgSamplePts')
    # _outputData(bodyAvgSamplePts, 'bodyAvgSamplePts')
    # _outputData(handSamplePointsArrs, 'handSamplePointsArrs')
    # _outputData(bodySamplePointsArrs, 'bodySamplePointsArrs')
    bodyACorr = _readFile('bodyAutoCorrelation')
    bodyRepeatPatternCycle = _readFile('bodyRepeatPatternCycle')
    handSamplePointsArrs = _readFile('handSamplePointsArrs')
    bodySamplePointsArrs = _readFile('bodySamplePointsArrs')
    handAvgSamplePts = _readFile('handAvgSamplePts')
    bodyAvgSamplePts = _readFile('bodyAvgSamplePts')
    return bodyACorr, bodyRepeatPatternCycle, handSamplePointsArrs, bodySamplePointsArrs, \
        handAvgSamplePts, bodyAvgSamplePts

def plotMultiSegSamplePts(decIncMapFuncSP, avgMapFuncSP, usedJointIdx, figNm=''):
    '''
    畫出increase and decrease segments. 並且一起畫出最終取平均的結果
    '''
    # print(decIncMapFuncSP[0][0]['x'])
    figs = []
    for _jointInd in range(len(usedJointIdx)):
        for _axis in usedJointIdx[_jointInd]:
            fig = plt.figure(num='{2}_segment merge_{0}_{1}'.format(_jointInd, _axis, figNm))
            ax = plt.subplot(111)
            ax.plot(
                np.linspace(0, 1, len(decIncMapFuncSP[0][_jointInd][_axis])), 
                decIncMapFuncSP[0][_jointInd][_axis][::-1], 
                '.',
                label='decrease segment'
            )
            ax.plot(
                np.linspace(0, 1, len(decIncMapFuncSP[1][_jointInd][_axis])), 
                decIncMapFuncSP[1][_jointInd][_axis], 
                '.',
                label='increase segment'
            )
            ax.plot(
                np.linspace(0, 1, len(avgMapFuncSP[_jointInd][_axis])), 
                avgMapFuncSP[_jointInd][_axis], 
                '.-',
                label='average'
            )
            plt.legend()
            figs.append(fig)
            pass
    return figs

def plotBSplineMapFunc(handSP, bodySP, usedJointIdx):
    figs = []
    for _jointInd in range(len(usedJointIdx)):
        for _axis in usedJointIdx[_jointInd]:
            ## show min and max in mapping function
            handMinMax = [np.min(handSP[_jointInd][_axis]), np.max(handSP[_jointInd][_axis])]
            bodyMinMax = [np.min(bodySP[_jointInd][_axis]), np.max(bodySP[_jointInd][_axis])]
            print(_jointInd, ', ', _axis)
            print('hand min: {0}, max: {1}'.format(handMinMax[0], handMinMax[1]))
            print('body min: {0}, max: {1}'.format(bodyMinMax[0], bodyMinMax[1]))
            

            fig = plt.figure(num='mapping func_{0}_{1}'.format(_jointInd, _axis))
            ax = plt.subplot(111)
            ax.plot(handSP[_jointInd][_axis], bodySP[_jointInd][_axis], '.-')
            figs.append(fig)
    return figs

# visualize B-Spline fitting result
def vizBSplineMapFunc(
    dataFilePath = 'rotationMappingData/leftFrontKick/', 
    BSplineDataFilePath = 'rotationMappingData/leftFrontKickBSpline/',
    saveFigsFilePath = 'rotationMappingFigs/leftFrontKickBSpline/'
):
    '''
    # 1. read all the data, include those are only computed in BSpline fitting 
    # 2. plot body autocorrelation result. autocorrelation curve and found frequency index 
    # 3. plot inc and dec segment, 並且畫出average後的線段 
    # 4. plot 最終的sample points 組合的mapping function, 並且顯示hand and body的最大與最小值
    # 5. store figures
    '''

    # 1. 
    handJointsRotations, afterAdjRangeJointRots, afterLowPassJointRots, filteredHandJointRots, \
        jointsACorr, jointsACorrLocalMaxIdx, handJointsPatternData, \
            originBodyRot, bodyAfterRangeAdj, bodyAfterGaussians, \
                mappingFuncs, handMinMax, bodyMinMax = readAllTheData(dataFilePath)
    ## read BSpline fitting data 
    bodyACorr, bodyRepeatPatternCycle, \
        handSamplePointsArrs, bodySamplePointsArrs, handAvgSamplePts, bodyAvgSamplePts = \
            readBSplineData(BSplineDataFilePath)    
    # 2. autocorrelation 
    autoCorrFigs = plotAutoCorrelation(
        [{'x': bodyACorr}], [{'x': bodyRepeatPatternCycle}], [['x']]
    )

    # 3. plot inc and dec segments, also the average result 
    # hand 
    handMultiSegFigs = plotMultiSegSamplePts(
        handSamplePointsArrs, handAvgSamplePts, usedJointIdx, 'hand'
    )
    ## body 
    bodyMultiSegFigs = plotMultiSegSamplePts(
        bodySamplePointsArrs, bodyAvgSamplePts, usedJointIdx, 'body'
    )

    # 4. plot mapping function and show the min and max
    BSplineMapFuncFigs = plotBSplineMapFunc(handAvgSamplePts, bodyAvgSamplePts, usedJointIdx)

    # 5. 
    saveFigs(autoCorrFigs, os.path.join(saveFigsFilePath, 'autoCorrelation'))
    saveFigs(handMultiSegFigs, os.path.join(saveFigsFilePath, 'handSegments'))
    saveFigs(bodyMultiSegFigs, os.path.join(saveFigsFilePath, 'bodySegments'))
    saveFigs(BSplineMapFuncFigs, os.path.join(saveFigsFilePath, 'BSplineMapFunc')) 
    # plt.show()
    pass

# TODO: 將兩種不同方法的mapping function畫在一起比較差異
def vizDiffMapFunc(dataFilePath, BSplineDataFilePath, saveFigsFilePath):
    '''
    1. read all the data
    2. plot two mapping function in a single figure
    3. store figures
    '''
    # 1. 
    handJointsRotations, afterAdjRangeJointRots, afterLowPassJointRots, filteredHandJointRots, \
        jointsACorr, jointsACorrLocalMaxIdx, handJointsPatternData, \
            originBodyRot, bodyAfterRangeAdj, bodyAfterGaussians, \
                mappingFuncs, handMinMax, bodyMinMax = readAllTheData(dataFilePath)
    ## read BSpline fitting data 
    bodyACorr, bodyRepeatPatternCycle, \
        handSamplePointsArrs, bodySamplePointsArrs, handAvgSamplePts, bodyAvgSamplePts = \
            readBSplineData(BSplineDataFilePath) 
    # 2. 
    figs = []
    for _jointInd in range(len(usedJointIdx)):
        for _axis in usedJointIdx[_jointInd]:
            fittedLine = np.poly1d(mappingFuncs[_jointInd][_axis])
            _x = np.linspace(handMinMax[0][_jointInd][_axis], handMinMax[1][_jointInd][_axis])
            _y = fittedLine(_x)

            fig = plt.figure(num='two mapping func_{0}_{1}'.format(_jointInd, _axis))
            ax = plt.subplot(111)
            ax.plot(
                handAvgSamplePts[_jointInd][_axis],
                bodyAvgSamplePts[_jointInd][_axis],
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
    # plt.show()
    pass

if __name__=='__main__':
    vizLinearMapResult(dataFilePath = 'rotationMappingData/leftFrontKick/')
    vizBSplineMapFunc(
        dataFilePath = 'rotationMappingData/leftFrontKick/', 
        BSplineDataFilePath = 'rotationMappingData/leftFrontKickBSpline/',
        saveFigsFilePath = 'rotationMappingFigs/leftFrontKickBSpline/'
    )
    vizDiffMapFunc(
        dataFilePath = 'rotationMappingData/leftFrontKick/', 
        BSplineDataFilePath = 'rotationMappingData/leftFrontKickBSpline/',
        saveFigsFilePath = 'rotationMappingFigs/leftFrontKick/'
    )
    pass