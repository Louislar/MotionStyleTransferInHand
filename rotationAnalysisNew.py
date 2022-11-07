'''
rewrite rotation analysis的部分main function
需要輸出計算過程各個步驟的資料 (提供給rotationAnalysisViz.py)
沿用大部分的function
rewrite兩個重要的計算
1. linear mapping function construction
2. B-Spline mapping function construction 

需要**確認**rewrite的輸出結果會與原始function相同
'''

import json
import pandas as pd 
import numpy as np 
import pickle
import os 
import copy 
from rotationAnalysis import *

def constructLinearMapFunc(
    handRotationFilePath = './HandRotationOuputFromHomePC/leftFrontKickStream.json',
    bodyRotationFilePath = './bodyDBRotation/genericAvatar/leftFrontKick0.03_withHip.json',
    outputFilePath = 'rotationMappingData/leftFrontKick/'
):
    '''
    # 1. (hand) read hand rotation
    # 2. (hand) adjust range to [-180, 180] 
    # 3. (hand) low pass and average filter
    # 4. (hand) autocorrelation for finding frequency
    # 5. (hand) average repeating patterns with high correlation between others
    # 6. (hand) extract min and max
    # 7. (body) read body rotation 
    # 8. (body) adjust range to [-180, 180] also extract min and max
    # 9. (mixed) linear fitting by maximum and minimum value 
    # 10. Save all the data 
    '''

    # 1. 
    handJointsRotations=None
    with open(handRotationFilePath, 'r') as fileOpen: 
        rotationJson=json.load(fileOpen)
        # handJointsRotations = rotationJsonDataParser(rotationJson, jointCount=4)    # For Unity output
        handJointsRotations = rotationJsonDataParser({'results': rotationJson}, jointCount=4)    # For python output
    
    # 2. 3.  Filter the time series data
    afterAdjRangeJointRots = copy.deepcopy(handJointsRotations)
    afterLowPassJointRots = copy.deepcopy(handJointsRotations)
    filteredHandJointRots = copy.deepcopy(handJointsRotations)
    for aJointIdx in range(len(handJointsRotations)):
        aJointData=handJointsRotations[aJointIdx]
        for k, aAxisRotationData in aJointData.items():
            aAxisRotationData = adjustRotationDataTo180(aAxisRotationData)

            filteredRotaion = butterworthLowPassFilter(aAxisRotationData)

            gaussainRotationData = gaussianFilter(filteredRotaion, 2)

            filteredHandJointRots[aJointIdx][k] = gaussainRotationData
            afterAdjRangeJointRots[aJointIdx][k] = aAxisRotationData
            afterLowPassJointRots[aJointIdx][k] = filteredRotaion
    
    ## Data need to output
    # afterAdjRangeJointRots
    # afterLowPassJointRots
    # filteredHandJointRots
    # print(filteredHandJointRots)

    # 4. Find repeat patterns of the filtered data (use autocorrelation)
    jointsACorr = {i: {k: [] for k in aJointAxis} for i, aJointAxis in enumerate(usedJointIdx)}
    jointsACorrLocalMaxIdx = {i: {k: [] for k in aJointAxis} for i, aJointAxis in enumerate(usedJointIdx)}
    for aJointIdx in range(len(filteredHandJointRots)):
        aJointData=filteredHandJointRots[aJointIdx]
        for k in usedJointIdx[aJointIdx]:
            jointsACorr[aJointIdx][k] = autoCorrelation(aJointData[k], False)
            localMaxIdx, = findLocalMaximaIdx(jointsACorr[aJointIdx][k])
            localMaxIdx = [i for i in localMaxIdx if jointsACorr[aJointIdx][k][i]>0]# The local maximum need to correspond to a positive correlation
            jointsACorrLocalMaxIdx[aJointIdx][k] = localMaxIdx[0]

    allFrequency = [jointsACorrLocalMaxIdx[i][k] for i, aJointAxis in enumerate(usedJointIdx) for k in aJointAxis]
    repeatingPatternCycle = sum(allFrequency) / len(allFrequency)    # 出現重複模式的週期
    # TODO: 改成使用weighted sum來計算repeating pattern可能會比較好，weight的來源可以使用下一步計算的correlation
    print('Hand cycle: ', repeatingPatternCycle)

    ## Data need to output
    # jointsACorr 
    # jointsACorrLocalMaxIdx 

    # 5. Compute the repeat pattern by simply average all the pattern candidates
    # [Alter choice] Just pick top 3(k) patterns that has most correlation with each other and average them
    # [Use weighted average, since not all the pattern's quality are equal
    # the pattern has high correlation with more other patterns get higher weight]
    handJointsPatternData = \
        {i: {k: [] for k in aJointInd} for i, aJointInd in enumerate(usedJointIdx)}
    for aJointIdx in range(len(filteredHandJointRots)):
        aJointData=filteredHandJointRots[aJointIdx]
        for k in usedJointIdx[aJointIdx]:
            splitedRotation = splitRotation(aJointData[k], repeatingPatternCycle, True)
            highestCorrIdx, highestCorrs = correlationBtwMultipleSeq(splitedRotation, k=3)
            highestCorrRots = [splitedRotation[i] for i in highestCorrIdx]
            avgHighCorrPattern = averageMultipleSeqs(highestCorrRots, highestCorrs/sum(highestCorrs))
            handJointsPatternData[aJointIdx][k] = avgHighCorrPattern
    
    ## Data need to output
    # handJointsPatternData
    # print(handJointsPatternData[0]['x'])

    # 6. extract min and max in hand patterns
    handGlobalMin = {i: {k: [] for k in aJointInd} for i, aJointInd in enumerate(usedJointIdx)}
    handGlobalMax = {i: {k: [] for k in aJointInd} for i, aJointInd in enumerate(usedJointIdx)}
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            _max, _maxIdx = findGlobalMaxAndIdx(np.array(handJointsPatternData[aJointIdx][k]))
            _min, _minIdx = findGlobalMinAndIdx(np.array(handJointsPatternData[aJointIdx][k]))
            handGlobalMax[aJointIdx][k] = _max
            handGlobalMin[aJointIdx][k] = _min
    
    ## ======= ======= ======= ======= ======= ======= ======= 
    # Body rotation analysis
    # 7. 
    bodyJointRotations=None
    with open(bodyRotationFilePath, 'r') as fileOpen: 
        rotationJson=json.load(fileOpen)
        bodyJointRotations = rotationJsonDataParser(rotationJson, jointCount=4)
        bodyJointRotations = [{k: bodyJointRotations[aJointIdx][k] for k in bodyJointRotations[aJointIdx]} for aJointIdx in range(len(bodyJointRotations))]
    
    ## 8. Adjust body rotation data to [-180, 180]
    originBodyRot = bodyJointRotations.copy()
    bodyOriginMin = [{k:None for k in usedJointIdx[aJointIdx]} for aJointIdx in range(len(usedJointIdx))]
    bodyOriginMax = [{k:None for k in usedJointIdx[aJointIdx]} for aJointIdx in range(len(usedJointIdx))]
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            bodyJointRotations[aJointIdx][k] = adjustRotationDataTo180(bodyJointRotations[aJointIdx][k])
            bodyOriginMin[aJointIdx][k] = min(bodyJointRotations[aJointIdx][k])
            bodyOriginMax[aJointIdx][k] = max(bodyJointRotations[aJointIdx][k])
    
    ## 9. fit linear function by max and min 
    ## body的min max確定與先前實驗相同
    ## 因為hand rotation的min and max與先前的實驗不同, 所以fitting結果也不同
    ## 不過差異沒有非常大. 推測差異來源是因為先前的實驗有座B-Spline fitting
    mappingFuncs = [{k: [] for k in axis} for axis in usedJointIdx]
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            fittedPolyLine = simpleLinearFitting(
                [handGlobalMin[aJointIdx][k], handGlobalMax[aJointIdx][k]], 
                [bodyOriginMin[aJointIdx][k], bodyOriginMax[aJointIdx][k]], 
                degree=1
            )
            mappingFuncs[aJointIdx][k] = fittedPolyLine
    print(mappingFuncs[0]['x'])

    ## Data need to output
    # mappingFuncs

    ## ======= ======= ======= ======= ======= ======= =======
    # 10. store results in each processing step
    def _outputData(data, fileNm):
        with open(os.path.join(outputFilePath, fileNm+'.pickle'), 'wb') as WFile:
            pickle.dump(data, WFile)
    
    # TODO: hand origin rotation still the same with handAfterAdjRange
    # dont know where the bug is 
    _outputData(handJointsRotations, 'handOrigin')
    _outputData(afterAdjRangeJointRots, 'handAfterAdjRange')
    _outputData(afterLowPassJointRots, 'handAfterLowPass')
    _outputData(filteredHandJointRots, 'handAfterGaussian')
    
    _outputData(jointsACorr, 'handAutoCorrelation')
    _outputData(jointsACorrLocalMaxIdx, 'handAutoCorrelationLocalMaxIdx')

    _outputData(handJointsPatternData, 'handJointsPatternData')

    _outputData(originBodyRot, 'bodyOrigin')
    _outputData(bodyJointRotations, 'bodyAfterAdjRange')

    _outputData(mappingFuncs, 'mappingFuncs')
    pass

if __name__ == '__main__':
    constructLinearMapFunc(
        handRotationFilePath = './HandRotationOuputFromHomePC/leftFrontKickStream.json',
        bodyRotationFilePath = './bodyDBRotation/genericAvatar/leftFrontKick0.03_withHip.json',
        outputFilePath = 'rotationMappingData/leftFrontKick/'
    )
    pass