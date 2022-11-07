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
    return handJointsRotations, afterAdjRangeJointRots, afterLowPassJointRots, filteredHandJointRots, \
        jointsACorr, jointsACorrLocalMaxIdx, handJointsPatternData, originBodyRot, bodyAfterRangeAdj, \
           bodyAfterGaussian, mappingFuncs

def plotRotationCurveInARow(rotCurves, curveNms, usedJointIdx):
    '''
    原始rotation, range adjust, low pass, gaussian. 
    以上所有訊號都垂直plot在同一個figure
    注意, 每個joint, axis是畫在不同的figure當中
    Input:
    :rots: (list) 多個joint以及axis的原始rotation curve
                list的element是不同步驟的rotation curve
    '''
    numCurves = len(rotCurves)
    
    for _jointInd in range(len(usedJointIdx)):
        for _axis in usedJointIdx[_jointInd]:
            fig = plt.figure(num='{0}_{1}'.format(_jointInd, _axis))
            axs = []
            for i in range(1, numCurves+1):
                axs.append(plt.subplot(numCurves,1,i))
            for i in range(numCurves):
                axs[i].set_title(curveNms[i])
                axs[i].plot(range(len(rotCurves[i][_jointInd][_axis])), rotCurves[i][_jointInd][_axis])
    # plt.show()
    pass

def plotAutoCorrelation(autoCorrCurves, autoCorrLocalMaxIdx):
    '''
    顯示一張圖片, 需要標上auto correlation找到的frequency位置 (或是/並且在明顯的地方顯示數值)
    auto correlation
    Input: 
    :autoCorrCurves: (list) 多個joint以及axis的autocorrelation數值
    :autoCorrLocalMaxIdx: (list) 多個joint以及axis的autocorrelation第一個正數波峰
    '''
    for _jointInd in range(len(usedJointIdx)):
        for _axis in usedJointIdx[_jointInd]:
            fig = plt.figure(num='{0}_{1}_frequency_{2}'.format(_jointInd, _axis, autoCorrLocalMaxIdx[_jointInd][_axis]))
            ax = plt.subplot(111)
            ax.plot(range(len(autoCorrCurves[_jointInd][_axis])), autoCorrCurves[_jointInd][_axis])
            ax.plot(autoCorrLocalMaxIdx[_jointInd][_axis], autoCorrCurves[_jointInd][_axis][autoCorrLocalMaxIdx[_jointInd][_axis]], '.r')
    # plt.show()
    pass

def vizLinearMapResult(dataFilePath = 'rotationMappingData/leftFrontKick/'):
    handJointsRotations, afterAdjRangeJointRots, afterLowPassJointRots, filteredHandJointRots, \
        jointsACorr, jointsACorrLocalMaxIdx, handJointsPatternData, \
            originBodyRot, bodyAfterRangeAdj, bodyAfterGaussians, \
                mappingFuncs = readAllTheData(dataFilePath)
    ## Plot range adjust, low pass and gaussian results (hand)
    # plotRotationCurveInARow(
    #     [afterAdjRangeJointRots, afterLowPassJointRots, filteredHandJointRots],
    #     ['adjust range', 'low pass', 'gaussian'],
    #     usedJointIdx
    # )
    ## Plot auto correlation result
    plotAutoCorrelation(
        jointsACorr, jointsACorrLocalMaxIdx
    )
    ## Plot body rotation 
    plotRotationCurveInARow(
        [originBodyRot, bodyAfterRangeAdj, bodyAfterGaussians],
        ['origin body rotation', 'after range adjustment', 'after gaussian'],
        usedJointIdx
    )
    ## Plot mapping function
    # TODO: 

    plt.show()
    # print(afterAdjRangeJointRots[0]['x'])
    # print(afterLowPassJointRots[0]['x'])
    print(jointsACorrLocalMaxIdx[0]['x'])
    pass

if __name__=='__main__':
    vizLinearMapResult(dataFilePath = 'rotationMappingData/leftFrontKick/')
    pass