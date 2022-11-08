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

def plotMapFunc(mapFuncParam, handMinMax, bodyMinMax, usedJointIdx):
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
    mapFuncFigs = plotMapFunc(
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

# TODO: visualize B-Spline fitting result
def vizBSplineMapFunc():
    pass

if __name__=='__main__':
    vizLinearMapResult(dataFilePath = 'rotationMappingData/leftFrontKick/')
    pass