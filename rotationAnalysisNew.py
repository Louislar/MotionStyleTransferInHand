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
    print('======= ======= ======= ======= ======= ======= ')
    print('Constructing linear mapping function')
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
            # 顯示hand最大最小值
            print('hand')
            print(aJointIdx, ', ', k)
            print(handGlobalMin[aJointIdx][k], handGlobalMax[aJointIdx][k])
    
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
            # 顯示body最大最小值
            print('body')
            print(aJointIdx, ', ', k)
            print(bodyOriginMin[aJointIdx][k], bodyOriginMax[aJointIdx][k])

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
    # 11. 需要做min max normalization, 把最大與最小值調整回原始訊號的數值範圍 
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
    # 17. B-Spline fit the mapping function and sample points from it 
    # 18. Normalize the sample points to original range (min and max)
    # 19. Save all the data 
    '''
    print('======= ======= ======= ======= ======= ======= ')
    print('Constructing B-Spline mapping function')
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
    bodyACorr = autoCorrelation(bodyJointRotations[0]['x'], False) # left front kick
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
    # TODO: 這邊在找global min and max的地方出現問題
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
                bodySamplePointsArrs[inc_dec][aJointIdx][k] = splev(np.linspace(0, len(bodySegs[inc_dec][aJointIdx][k])-1, numberOfSamplePt), bodySplines[inc_dec][aJointIdx][k])
                handSamplePointsArrs[inc_dec][aJointIdx][k] = splev(np.linspace(0, len(handSegs[inc_dec][aJointIdx][k])-1, numberOfSamplePt), handSplines[inc_dec][aJointIdx][k])
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

    ## 17. B-Spline fit the mapping function and sample points from it 
    handMapSamplePts = [{k: [] for k in axis} for axis in usedJointIdx]
    bodyMapSamplePts = [{k: [] for k in axis} for axis in usedJointIdx]
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            ## Fit B-Spline最重要的兩個議題要注意
            ## 1. x要由小到大
            ## 2. x不能有重複的數值
            ## 這邊如果只排序x軸資料, y軸的資料就會亂掉. 所以, y軸也由小到大排序
            handAvgSamplePts[aJointIdx][k] = np.sort(handAvgSamplePts[aJointIdx][k])
            bodyAvgSamplePts[aJointIdx][k] = np.sort(bodyAvgSamplePts[aJointIdx][k])
            ## Fit B-Spline 
            ## Decide smooth factor used in B-Spline fitting 
            ## factor = number of data * variance of data
            ## refer to the post: https://stackoverflow.com/questions/8719754/scipy-interpolate-univariatespline-not-smoothing-regardless-of-parameters?rq=1
            smoothFactor = numberOfSamplePt * 0.01
            # _bspline = bSplineFitting(
            #     bodyAvgSamplePts[aJointIdx][k], handAvgSamplePts[aJointIdx][k], False
            # )
            _bspline = splrep(
                handAvgSamplePts[aJointIdx][k], bodyAvgSamplePts[aJointIdx][k], s=smoothFactor
            )
            ## Sample points
            handMapSamplePts[aJointIdx][k] = np.linspace(
                handAvgSamplePts[aJointIdx][k][0], 
                handAvgSamplePts[aJointIdx][k][-1], 
                numberOfSamplePt
            )
            bodyMapSamplePts[aJointIdx][k] = splev(
                handMapSamplePts[aJointIdx][k], 
                _bspline
            ) 
            pass

    ## 18. Normalize the sample points to original range (min and max)
    ##      Hand rotation的原始大小是找到的repeating pattern的數值範圍
    ##      Body rotation的原始大小是adjust範圍到[-180, 180]後, 得到的min and max 
    handNormMapSamplePts = [{k: [] for k in axis} for axis in usedJointIdx]
    bodyNormMapSamplePts = [{k: [] for k in axis} for axis in usedJointIdx]
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            handNormMapSamplePts[aJointIdx][k] = minMaxNormalization(
                handMapSamplePts[aJointIdx][k],
                np.min(handJointsPatternData[aJointIdx][k]),
                np.max(handJointsPatternData[aJointIdx][k])
            )
            bodyNormMapSamplePts[aJointIdx][k] = minMaxNormalization(
                bodyMapSamplePts[aJointIdx][k],
                bodyOriginMin[aJointIdx][k],
                bodyOriginMax[aJointIdx][k]
            )
            pass

    ## 19. store all the data 
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
    _outputData(handNormMapSamplePts, 'handNormMapSamplePts')
    _outputData(bodyNormMapSamplePts, 'bodyNormMapSamplePts')

    pass

def applyLinearMapFunc(mappingFuncs, rotation, usedJointIdx):
    afterMapping = [{k: [] for k in axis} for axis in usedJointIdx]
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            fitLine = np.poly1d(mappingFuncs[aJointIdx][k])
            mappedRot = fitLine(rotation[aJointIdx][k])
            afterMapping[aJointIdx][k] = mappedRot
    return afterMapping

def applyBSplineMapFunc(handSP, bodySP, rotation, usedJointIdx):
    '''
    尋找最接近的hand sample point, 輸出相對應的body sample point
    '''
    afterMapping = [{k: [] for k in axis} for axis in usedJointIdx]
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            _rot = rotation[aJointIdx][k][:, np.newaxis]
            _dist = np.abs(handSP[aJointIdx][k] - _rot)
            _minInd = np.argmin(_dist, axis=1)
            afterMapping[aJointIdx][k] = bodySP[aJointIdx][k][_minInd]
    return afterMapping

def applyMapFuncToRot(
    handRotationFilePath, linearMapFuncFilePath, 
    BSplineHandSPFilePath, BSplineBodySPFilePath,
    outputFilePath
):
    '''
    Object:
        套用兩種mapping function到hand rotation. 
        hand rotation也分成預處理與沒有預處理兩種
    1. read hand rotation (after low pass and gaussian filter)
    2. read mapping function (2 types of mapping function)
    3. apply mapping function 
    4. output mapping result 
    4.0 決定沒有要mapping的轉軸, 給予原始手部旋轉數值
    4.1 output mapping result in json format 
    Input: 
    (以下兩者搭配在一起當作mapping function使用)
    :BSplineHandSPFilePath: hand B-Spline sample points 
    :BSplineBodySPFilePath: body B-Spline sample points 
    '''

    # 1. 
    handJointsRotations=None
    with open(handRotationFilePath, 'r') as fileOpen: 
        rotationJson=json.load(fileOpen)
        # handJointsRotations = rotationJsonDataParser(rotationJson, jointCount=4)    # For Unity output
        handJointsRotations = rotationJsonDataParser({'results': rotationJson}, jointCount=4)    # For python output

    ## Filter the time series data
    originHandJointRots = copy.deepcopy(handJointsRotations)
    filteredHandJointRots = copy.deepcopy(handJointsRotations)
    for aJointIdx in range(len(filteredHandJointRots)):
        aJointData=filteredHandJointRots[aJointIdx]
        for k, aAxisRotationData in aJointData.items():
            aAxisRotationData = adjustRotationDataTo180(aAxisRotationData)
            filteredRotaion = butterworthLowPassFilter(aAxisRotationData)
            gaussainRotationData = gaussianFilter(filteredRotaion, 2)
            filteredHandJointRots[aJointIdx][k] = gaussainRotationData

    ## if origin rotation data is not array then change it to numpy array
    if not isinstance(originHandJointRots[0]['x'], np.ndarray):
        for aJointIdx in range(len(usedJointIdx)):
            for aAxis in usedJointIdx[aJointIdx]:
                originHandJointRots[aJointIdx][aAxis] = np.array(originHandJointRots[aJointIdx][aAxis])

    # 2. read mapping function 
    ## linear mapping function 
    linearMapFunc = None
    with open(linearMapFuncFilePath, 'rb') as RFile:
        linearMapFunc = pickle.load(RFile)
    ## B-Spline fitting mapping function 
    BSplineHandSP = None
    BSplineBodySP = None
    with open(BSplineHandSPFilePath, 'rb') as RFile:
        BSplineHandSP = pickle.load(RFile)
    with open(BSplineBodySPFilePath, 'rb') as RFile:
        BSplineBodySP = pickle.load(RFile)
    
    # 3. apply mapping function 
    ## linear mapping function
    originHandLinearMap = applyLinearMapFunc(linearMapFunc, originHandJointRots, usedJointIdx)
    filteredHandLinearMap = applyLinearMapFunc(linearMapFunc, filteredHandJointRots, usedJointIdx)
    plt.figure()
    # plt.plot(range(len(originHandLinearMap[0]['x'])), originHandLinearMap[0]['x'], label='origin linear')
    # plt.plot(range(len(filteredHandLinearMap[0]['x'])), filteredHandLinearMap[0]['x'], label='filtered linear')
    # plt.plot(range(len(originHandJointRots[0]['x'])), originHandJointRots[0]['x'], label='origin')
    # plt.plot(range(len(filteredHandJointRots[0]['x'])), filteredHandJointRots[0]['x'], label='filtered')
    # plt.legend()
    # plt.show()

    ## B-Spline mapping function 
    originHandBSplineMap = applyBSplineMapFunc(BSplineHandSP, BSplineBodySP, originHandJointRots, usedJointIdx)
    filteredHandBSplineMap = applyBSplineMapFunc(BSplineHandSP, BSplineBodySP, filteredHandJointRots, usedJointIdx)

    # plt.figure()
    # plt.plot(range(len(originHandBSplineMap[0]['x'])), originHandBSplineMap[0]['x'], label='origin BSpline')
    # plt.plot(range(len(filteredHandBSplineMap[0]['x'])), filteredHandBSplineMap[0]['x'], label='filtered BSpline')
    # plt.plot(range(len(originHandJointRots[0]['x'])), originHandJointRots[0]['x'], label='origin')
    # plt.plot(range(len(filteredHandJointRots[0]['x'])), filteredHandJointRots[0]['x'], label='filtered')
    # plt.legend()
    # plt.show()

    # 4. output mapping result 
    def _outputData(data, fileNm):
        with open(os.path.join(outputFilePath, fileNm+'.pickle'), 'wb') as WFile:
            pickle.dump(data, WFile)

    _outputData(originHandLinearMap, 'originHandLinearMap')
    _outputData(filteredHandLinearMap, 'filteredHandLinearMap')
    _outputData(originHandBSplineMap, 'originHandBSplineMap')
    _outputData(filteredHandBSplineMap, 'filteredHandBSplineMap')

    ## 4.0 決定沒有要mapping的轉軸, 給予原始手部旋轉數值 (不包含'y')
    ## TODO: 修改成, 沒有要mapping的轉軸, 給予數值0
    unMappedAxis = [['z'], [], [], []]
    for i in range(len(unMappedAxis)):
        for k in unMappedAxis[i]:
            ## 給予原始手部旋轉數值
            # originHandLinearMap[i][k] = originHandJointRots[i][k]
            # filteredHandLinearMap[i][k] = filteredHandJointRots[i][k]
            # originHandBSplineMap[i][k] = originHandJointRots[i][k]
            # filteredHandBSplineMap[i][k] = filteredHandJointRots[i][k]
            ## 給予0
            originHandLinearMap[i][k] = np.zeros_like(originHandJointRots[i][k])
            filteredHandLinearMap[i][k] = np.zeros_like(filteredHandJointRots[i][k])
            originHandBSplineMap[i][k] = np.zeros_like(originHandJointRots[i][k])
            filteredHandBSplineMap[i][k] = np.zeros_like(filteredHandJointRots[i][k])
    ## 4.1 output mapping result in json format 
    timeCount = len(originHandLinearMap[0]['x'])
    jointCount = len(originHandLinearMap)
    ## 輸出格式要轉換成list, 因為json輸出不支援np.array型別
    ## 沒有旋轉的軸'y', 給予數值0 
    linearMappedRot = [{k: np.zeros(timeCount).tolist() for k in ['x', 'y', 'z']} for i in range(jointCount)]
    BSMappedRot = [{k: np.zeros(timeCount).tolist() for k in ['x', 'y', 'z']} for i in range(jointCount)]
    for i in range(len(usedJointIdx)):
        for k in usedJointIdx[i]:
            linearMappedRot[i][k] = originHandLinearMap[i][k].tolist()
            BSMappedRot[i][k] = originHandBSplineMap[i][k].tolist()
    print('usedJointIdx: ', usedJointIdx)
    print('timeCount: ', timeCount)
    print('jointCount: ', jointCount)
    linearMappedJson = [
        {
            'time':t, 'data': [
                {k: linearMappedRot[i][k][t] for k in ['x', 'y', 'z']} for i in range(jointCount)
            ]
        } for t in range(timeCount)
    ]
    BSMappedJson = [
        {
            'time':t, 'data': [
                {k: BSMappedRot[i][k][t] for k in ['x', 'y', 'z']} for i in range(jointCount)
            ]
        } for t in range(timeCount)
    ]
    with open(outputFilePath+'leftFrontKick_eular_linear_TFTTTT.json', 'w') as WFile: 
        json.dump(linearMappedJson, WFile) 
    with open(outputFilePath+'leftFrontKick_eular_BSpline_TFTTTT.json', 'w') as WFile: 
        json.dump(BSMappedJson, WFile) 
    pass


if __name__ == '__main__':
    # constructLinearMapFunc(
    #     handRotationFilePath = './HandRotationOuputFromHomePC/leftFrontKickStream.json',
    #     bodyRotationFilePath = './bodyDBRotation/genericAvatar/leftFrontKick0.03_withHip.json',
    #     outputFilePath = 'rotationMappingData/leftFrontKick/'
    # )
    ## ======= 
    constructBSplineMapFunc(
        handRotationFilePath = './HandRotationOuputFromHomePC/leftFrontKickStream.json',
        bodyRotationFilePath = './bodyDBRotation/genericAvatar/leftFrontKick0.03_withHip.json',
        outputFilePath = 'rotationMappingData/leftFrontKickBSpline/'
    )
    ## ======= 
    ## apply mapping function to hand rotation 
    # applyMapFuncToRot(
    #     handRotationFilePath = './HandRotationOuputFromHomePC/leftFrontKickStream.json',
    #     linearMapFuncFilePath = 'rotationMappingData/leftFrontKick/mappingFuncs.pickle',
    #     BSplineHandSPFilePath='rotationMappingData/leftFrontKickBSpline/handAvgSamplePts.pickle',
    #     BSplineBodySPFilePath='rotationMappingData/leftFrontKickBSpline/bodyAvgSamplePts.pickle',
    #     outputFilePath='rotationMappingData/leftFrontKick/'
    # )
    pass