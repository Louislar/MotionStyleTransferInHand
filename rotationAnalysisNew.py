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
    # 8. (body) 需要將body rotation curve 去掉前面幾個signal, 因為大機率包含雜訊
    # 9. (body) adjust range to [-180, 180] also extract min and max
    # 10. (body) Apply gaussian filter
    # 11. (mixed) linear fitting by maximum and minimum value 
    # (放到realTimeRotMapping.py處理) apply mapping function到預處理好的hand rotation
    # 12. Save all the data 
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
    
    ## 8. 不要使用body rotation前面的幾個時間點的訊號
    ## 目前指定不要使用前10個訊號
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            bodyJointRotations[aJointIdx][k] = bodyJointRotations[aJointIdx][k][10:]


    ## 9. Adjust body rotation data to [-180, 180] and find min and max
    originBodyRot = copy.deepcopy(bodyJointRotations)
    bodyOriginMin = [{k:None for k in usedJointIdx[aJointIdx]} for aJointIdx in range(len(usedJointIdx))]
    bodyOriginMax = [{k:None for k in usedJointIdx[aJointIdx]} for aJointIdx in range(len(usedJointIdx))]
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            bodyJointRotations[aJointIdx][k] = adjustRotationDataTo180(bodyJointRotations[aJointIdx][k])
            bodyOriginMin[aJointIdx][k] = min(bodyJointRotations[aJointIdx][k])
            bodyOriginMax[aJointIdx][k] = max(bodyJointRotations[aJointIdx][k])

    ## 10. Apply gaussian filter
    bodyAfterRangeAdjust = copy.deepcopy(bodyJointRotations)
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            bodyJointRotations[aJointIdx][k] = gaussianFilter(bodyJointRotations[aJointIdx][k], 2)
    
    ## 11. fit linear function by max and min 
    ## body的min max確定與先前實驗相同
    ## 因為hand rotation的min and max與先前的實驗不同, 所以fitting結果也不同
    ## 不過差異沒有非常大. 推測差異來源是因為先前的實驗有做B-Spline fitting
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
    # 12. store results in each processing step
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
    _outputData(bodyAfterRangeAdjust, 'bodyAfterAdjRange')
    _outputData(bodyJointRotations, 'bodyAfterGaussian')

    _outputData(mappingFuncs, 'mappingFuncs')
    _outputData([handGlobalMin, handGlobalMax], 'handMinMax')
    _outputData([bodyOriginMin, bodyOriginMax], 'bodyMinMax')
    pass

def handRotPreproc(handRotationFilePath):
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
    return handJointsPatternData

def constructBSplineMapFunc(
    handRotationFilePath = './HandRotationOuputFromHomePC/leftFrontKickStream.json',
    bodyRotationFilePath = './bodyDBRotation/genericAvatar/leftFrontKick0.03_withHip.json',
    outputFilePath = 'rotationMappingData/leftFrontKickBSpline/'
):
    '''
    # 1. (hand) read hand rotation
    # 2. (hand) adjust range to [-180, 180] 
    # 3. (hand) low pass and average filter
    # 4. (hand) autocorrelation for finding frequency
    # 5. (hand) average repeating patterns with high correlation between others
    # 6. (hand) extract min and max
    # 7. (body) read body rotation 
    # 8. (body) 需要將body rotation curve 去掉前面幾個signal, 因為大機率包含雜訊
    # 9. (body) adjust range to [-180, 180] also extract min and max
    # 10. (body) Apply gaussian filter
    # ======= 以上與linear mapping function的過程相同 =======
    # 11. TODO: 需要做min max normalization, 把最大與最小值調整回原始訊號的數值範圍 
    # 12. (body) use autocorrelation to find frequency
    ## TODO: 這邊多了幾個步驟使用DTW取特定的body rotation區段, 我感覺有點多餘
    ## 先不要加入這個部分, 先隨便取frequency長度的segment. 但是最後要觀察這種作法與之前的結果是否相似
    # 13. (body) crop segment in single cycle length
    # 14. (body) find max and min then crop inc and dec segment 
    # TODO: (Mixed) scale hand and body curve to the same time scale via minmaxNormalization
    # TODO: 上面rescale的步驟感覺是多餘的, 先刪除. 要觀察最終結果與原本的結果會不會差太多
    # 15. (Mixed) BSpline fit hand and body's inc and dec segments with 1000 sample points
    #           這邊sample points的數量就要取到最終搜尋時使用的sample points數量 (1000是原本的方法使用的數量)
    # TODO: 接下來還有一個N dimensional B-Spline fitting. 目前也省略掉. 
    # 16. average inc and dec segments which belongs to the same joint, axis
    # (放到realTimeRotMapping.py處理) apply mapping function到預處理好的hand rotation
    # 17. Save all the data 
    '''
    # 1. 2. 3. 4. 5. 6. 
    handJointsPatternData = handRotPreproc(handRotationFilePath)
    # ======= ======= ======= ======= ======= ======= =======
    # 7. 
    bodyJointRotations=None
    with open(bodyRotationFilePath, 'r') as fileOpen: 
        rotationJson=json.load(fileOpen)
        bodyJointRotations = rotationJsonDataParser(rotationJson, jointCount=4)
        bodyJointRotations = [{k: bodyJointRotations[aJointIdx][k] for k in bodyJointRotations[aJointIdx]} for aJointIdx in range(len(bodyJointRotations))]
    
    ## 8. 不要使用body rotation前面的幾個時間點的訊號
    ## 目前指定不要使用前10個訊號
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            bodyJointRotations[aJointIdx][k] = bodyJointRotations[aJointIdx][k][10:]

    ## 9. Adjust body rotation data to [-180, 180]
    originBodyRot = copy.deepcopy(bodyJointRotations)
    bodyOriginMin = [{k:None for k in usedJointIdx[aJointIdx]} for aJointIdx in range(len(usedJointIdx))]
    bodyOriginMax = [{k:None for k in usedJointIdx[aJointIdx]} for aJointIdx in range(len(usedJointIdx))]
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            bodyJointRotations[aJointIdx][k] = adjustRotationDataTo180(bodyJointRotations[aJointIdx][k])
            bodyOriginMin[aJointIdx][k] = min(bodyJointRotations[aJointIdx][k])
            bodyOriginMax[aJointIdx][k] = max(bodyJointRotations[aJointIdx][k])

    ## 10. Apply gaussian filter
    bodyAfterRangeAdjust = copy.deepcopy(bodyJointRotations)
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            bodyJointRotations[aJointIdx][k] = gaussianFilter(bodyJointRotations[aJointIdx][k], 2)
    
    ## ------- 與linear mapping不同的地方 -------
    ## 11. normalize the body average sample points to the original body rotation range 
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            bodyJointRotations[aJointIdx][k] = minMaxNormalization(
                bodyJointRotations[aJointIdx][k], 
                bodyOriginMin[aJointIdx][k],
                bodyOriginMax[aJointIdx][k]
            )

    ## 12. find frequency via autocorrelation 
    bodyRepeatPatternCycle=None
    bodyACorr = autoCorrelation(bodyJointRotations[0]['x'], True) # left front kick
    # bodyACorr = autoCorrelation(bodyJointRotations[0]['z'], False)   # left side kick
    bodyLocalMaxIdx, = findLocalMaximaIdx(bodyACorr)
    bodyLocalMaxIdx = [i for i in bodyLocalMaxIdx if bodyACorr[i]>0]
    bodyRepeatPatternCycle=bodyLocalMaxIdx[0]
    print('body cycle: ', bodyRepeatPatternCycle)

    ## output data
    # bodyACorr, bodyRepeatPatternCycle

    ## 13. crop cycle length (frequency) segment
    startCropInd = 5
    bodyJointsPatterns = [{k: [] for k in axis} for axis in usedJointIdx]
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            bodyJointsPatterns[aJointIdx][k]=bodyJointRotations[aJointIdx][k][startCropInd:startCropInd+bodyRepeatPatternCycle]

    ## output data 
    # bodyJointsPatterns
    
    # 14. find min max and crop into inc and dec segments
    bodyDecreaseSegs = [{k: [] for k in axis} for axis in usedJointIdx]
    bodyIncreaseSegs = [{k: [] for k in axis} for axis in usedJointIdx]
    handDecreaseSegs = [{k: [] for k in axis} for axis in usedJointIdx]
    handIncreaseSegs = [{k: [] for k in axis} for axis in usedJointIdx]

    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            bodyJointCurve = np.array(bodyJointsPatterns[aJointIdx][k]*3)
            handJointCurve = np.array(handJointsPatternData[aJointIdx][k]*3)
            bodyGlobalMax, bodyGlobalMaxIdx = findGlobalMaxAndIdx(bodyJointCurve)
            bodyGlobalMin, bodyGlobalMinIdx = findGlobalMinAndIdx(bodyJointCurve)

            handGlobalMax, handGlobalMaxIdx = findGlobalMaxAndIdx(handJointCurve)
            handGlobalMin, handGlobalMinIdx = findGlobalMinAndIdx(handJointCurve)

            bodyDecreaseSegs[aJointIdx][k], bodyIncreaseSegs[aJointIdx][k] = \
                cropIncreaseDecreaseSegments(bodyJointCurve, bodyGlobalMaxIdx, bodyGlobalMinIdx)
            handDecreaseSegs[aJointIdx][k], handIncreaseSegs[aJointIdx][k] = \
                cropIncreaseDecreaseSegments(handJointCurve, handGlobalMaxIdx, handGlobalMinIdx)
    bodySegs = [
        bodyDecreaseSegs, bodyIncreaseSegs
    ]
    handSegs = [
        handDecreaseSegs, handIncreaseSegs
    ]

    ## output data
    # bodySegs, handSegs

    # 15. fit B-Spline then sample some points
    numberOfSamplePt = 1000
    bodySplines = [
        [{k: [] for k in axis} for axis in usedJointIdx], [{k: [] for k in axis} for axis in usedJointIdx]
    ]
    handSplines = [
        [{k: [] for k in axis} for axis in usedJointIdx], [{k: [] for k in axis} for axis in usedJointIdx]
    ]
    handSamplePointsArrs = [
        [{k: [] for k in axis} for axis in usedJointIdx], [{k: [] for k in axis} for axis in usedJointIdx]
    ]
    bodySamplePointsArrs = [
        [{k: [] for k in axis} for axis in usedJointIdx], [{k: [] for k in axis} for axis in usedJointIdx]
    ]
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            for inc_dec in range(2):
                bodySplines[inc_dec][aJointIdx][k] = bSplineFitting(bodySegs[inc_dec][aJointIdx][k], isDrawResult=False)
                handSplines[inc_dec][aJointIdx][k] = bSplineFitting(handSegs[inc_dec][aJointIdx][k], isDrawResult=False)
                bodySamplePointsArrs[inc_dec][aJointIdx][k] = splev(np.linspace(0, len(bodySegs[inc_dec][aJointIdx][k]), numberOfSamplePt), bodySplines[inc_dec][aJointIdx][k])
                handSamplePointsArrs[inc_dec][aJointIdx][k] = splev(np.linspace(0, len(handSegs[inc_dec][aJointIdx][k]), numberOfSamplePt), handSplines[inc_dec][aJointIdx][k])
    ## output data 
    # bodySplines, handSplines, handSamplePointsArrs, bodySamplePointsArrs

    ## 16. average inc and dec segments which belongs to the same joint, axis
    ## 注意, decrease的時間先後順序要相反過來, 不然大小變化會與increase相反 (其中一個相反即可)
    handAvgSamplePts = [{k: [] for k in axis} for axis in usedJointIdx]
    bodyAvgSamplePts = [{k: [] for k in axis} for axis in usedJointIdx] 
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            handAvgSamplePts[aJointIdx][k] = \
                (handSamplePointsArrs[0][aJointIdx][k][::-1] + handSamplePointsArrs[1][aJointIdx][k]) / 2
            bodyAvgSamplePts[aJointIdx][k] = \
                (bodySamplePointsArrs[0][aJointIdx][k][::-1] + bodySamplePointsArrs[1][aJointIdx][k]) / 2
            
    ## output data 
    # handAvgSamplePts, bodyAvgSamplePts    

    ## 17. store all the data 
    def _outputData(data, fileNm):
        with open(os.path.join(outputFilePath, fileNm+'.pickle'), 'wb') as WFile:
            pickle.dump(data, WFile)

    _outputData(bodyACorr, 'bodyAutoCorrelation')
    _outputData(bodyRepeatPatternCycle, 'bodyRepeatPatternCycle')
    _outputData(bodyJointsPatterns, 'bodyJointsPatterns')
    _outputData(handSegs, 'handSegs')
    _outputData(bodySegs, 'bodySegs')
    _outputData(bodySplines, 'bodySplines')
    _outputData(handSplines, 'handSplines')
    _outputData(handSamplePointsArrs, 'handSamplePointsArrs')
    _outputData(bodySamplePointsArrs, 'bodySamplePointsArrs')
    _outputData(handAvgSamplePts, 'handAvgSamplePts')
    _outputData(bodyAvgSamplePts, 'bodyAvgSamplePts')


    pass

if __name__ == '__main__':
    constructLinearMapFunc(
        handRotationFilePath = './HandRotationOuputFromHomePC/leftFrontKickStream.json',
        bodyRotationFilePath = './bodyDBRotation/genericAvatar/leftFrontKick0.03_withHip.json',
        outputFilePath = 'rotationMappingData/leftFrontKick/'
    )
    # ======= 
    constructBSplineMapFunc(
        handRotationFilePath = './HandRotationOuputFromHomePC/leftFrontKickStream.json',
        bodyRotationFilePath = './bodyDBRotation/genericAvatar/leftFrontKick0.03_withHip.json',
        outputFilePath = 'rotationMappingData/leftFrontKickBSpline/'
    )
    pass