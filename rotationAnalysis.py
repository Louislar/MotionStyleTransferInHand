import numpy as np
import pandas as pd
from scipy.interpolate import splev, splrep, splprep
import scipy.signal as sig
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import PassiveAggressiveClassifier
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import json
import matplotlib.pyplot as plt
import itertools
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

'''
The goal of this code is trying to find a mapping function between hand and body motion, 
which the hand motion is from a motion capture system with noise and biased signal. 
'''

def rotationJsonDataParser(jsonDict: dict, jointCount: int):
    '''
    target format: 
    x list:[x1, x2, x3, ...]
    knee: {x list, y list, z list}
    left upper leg, left knee, right upper leg, right knee
    '''
    timeSeries=jsonDict['results']
    parsedRotationData=[{'x': [], 'y': [], 'z': []} for i in range(jointCount)]
    for jointIdx in range(jointCount):
        for oneData in timeSeries:
            parsedRotationData[jointIdx]['x'].append(oneData['data'][jointIdx]['x'])
            parsedRotationData[jointIdx]['y'].append(oneData['data'][jointIdx]['y'])
            parsedRotationData[jointIdx]['z'].append(oneData['data'][jointIdx]['z'])
    return parsedRotationData

def adjustRotationDataTo180(rotations: list):
    '''
    Make 180 degree to 0 and greater than 180 be negative degree, 因為在0與360度左右的資料不少
    '''
    return [i-360 if i>180 else i for i in rotations]

def adjustRotationDataFrom180To360(rotations: list):
    '''
    Reverse the transform of making -180 degree as 180
    '''
    return [i+360 if i<0 else i for i in rotations]

def adjustRotationByFFT(rotations: list):
    '''
    Use FFT to compute the frequency domain of the rotation data
    '''
    return np.fft.rfft(rotations)

def butterworthLowPassFilter(rotationsInFreq: list, order=5, cutoff=0.6):
    '''
    Use low pass filter to remove high frequency signals
    '''
    hpFilter = sig.butter(N=order, Wn=cutoff, btype='lowpass', output='sos')
    # b, a = sig.butter(N=order, Wn=cutoff, btype='lowpass', output='ba')
    # w, h = sig.freqs(b, a)
    # plt.figure()
    # plt.plot(w, h)
    return sig.sosfilt(hpFilter, rotationsInFreq)

def gaussianFilter(rotations: list, sd):
    '''
    Use gaussian filter to smooth the rotation curve
    '''
    return gaussian_filter1d(rotations, sigma=sd)

def autoCorrelation(rotations: list, drawACFPlot: bool = False):
    '''
    Use auto correlation to find repeat pattern, and use it to construct mapping function
    '''
    acorr = sm.tsa.acf(rotations, nlags = len(rotations)-1)
    # Draw AutoCorrelationFunction plot
    if drawACFPlot: 
        plot_acf(rotations, lags=len(rotations)-1)
        # plot_pacf(rotations, lags=len(rotations)/2-1)
    return acorr

def findLocalMaximaIdx(autoCorrs: list):
    '''
    Find local maximum in the given signal(usaully a autocorrelation series)
    '''
    return sig.argrelmax(autoCorrs)

def findLocalMinimaIdx(autoCorrs: list):
    '''
    Find local minimum in the given signal(usaully a autocorrelation series)
    '''
    return sig.argrelmin(autoCorrs)

def findGlobalMaxAndIdx(rotations: list):
    '''
    Find global maximum and corresponding index, 
    the global maximum will be one of the local maximums
    if no local maximum is found then the maximum value and corresponding id will return
    '''
    localMaxIdx, = findLocalMaximaIdx(rotations)
    if localMaxIdx.size == 0:
        return max(rotations), np.array([np.argmax(rotations)])
    globalMax = max(rotations[localMaxIdx])
    globalMaxIdx, = np.where(rotations==globalMax)
    return globalMax, globalMaxIdx

def findGlobalMinAndIdx(rotations: list):
    '''
    Find global minimum and corresponding index, 
    the global minimum will be one of the local minimums
    if no local minimum is found then the minimum value and corresponding id will return
    '''
    localMinIdx, = findLocalMinimaIdx(rotations)
    if localMinIdx.size == 0:
        return min(rotations), np.array([np.argmin(rotations)])
    globalMin = min(rotations[localMinIdx])
    globalMinIdx, = np.where(rotations==globalMin)
    return globalMin, globalMinIdx

def minMaxNormalization(rotations: list, targetMin, targetMax):
    '''
    Re-scale the rotations to a specific range [min, max]
    '''
    minRot=min(rotations)
    maxRot=max(rotations)
    return [(srcRot-minRot)*(targetMax-targetMin)/(maxRot-minRot) + targetMin for srcRot in rotations]

def splitRotation(rotations: list, size = 5, removeRemainData=False):
    '''
    Split rotation data into small buckets, which includes a repeat pattern
    ref: https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
    Input: 
    :removeRemainData: 不足size的尾段資料要不要移除
    '''
    size = int(size)
    if removeRemainData:
        return [rotations[i:i + size] for i in range(0, len(rotations), size) if len(rotations[i:i + size])==size]
    return [rotations[i:i + size] for i in range(0, len(rotations), size)]

def correlationBtwMultipleSeq(splitedRotations: list, k: int):
    '''
    Compute multiple sequences' correlation with each other, the correlation matrix
    Give every sequence a score about its correlation to the remainings
    它與其他所有sequences之間的相似程度的總和分數(負數記為0，不相似與非常不相似是相同的)
    Input: 
    :k: return k most highest aggrgate correlation sequence index
    '''
    splitedRotDf = pd.DataFrame({i: splitedRotations[i] for i in range(len(splitedRotations))})
    corrMatrixDf = splitedRotDf.corr()
    corrMatrixDf[corrMatrixDf<0] = 0
    splitedRotCorrScore = corrMatrixDf.sum(axis=0).values
    highestCorrIdx = np.argsort(splitedRotCorrScore)[::-1]
    return highestCorrIdx[:k], splitedRotCorrScore[highestCorrIdx[:k]]

def averageMultipleSeqs(rotations: list, weights: list = None):
    '''
    Averaging multiple sequnces, they must have same number of datapoints and in same time scale
    Weighted average if weights is not None
    '''
    avgSeq = []
    if weights is not None:
        tmpSeq = []
        for aSeq, aWeight in zip(rotations, weights):
            tmpSeq.append(aSeq * aWeight)
        for t in zip (*tmpSeq):
            avgSeq.append(sum(t))
        return avgSeq
    for t in zip(*rotations):
        avgSeq.append(sum(t)/len(t))
    return avgSeq

def rollingWindowSplitRotation(rotations: list, winSize: int):
    '''
    Apply rolling/sliding window to rotation curve
    Input: 
    :winSize: window size
    '''
    lastWindowIdx = len(rotations) - winSize
    slidingWindowResults=[]
    for i in range(lastWindowIdx):
        slidingWindowResults.append(rotations[i:i+winSize])
    return slidingWindowResults

def computeDTWBtwMultiSeqs(motherWave: list, multiRotations: list, drawWarpResult=False):
    '''
    Compute DTW(Dynamic time warpping) distance between multiple rotaions seqences 
    to a single target rotation seqence(Similar to the Mother wave in wavelet transform)
    Input: 
    :drawWarpResult: Draw the warping result, the best and the worst
    Output: 
    :DTWResult: The DTW distance between the rotations and the mother rotation curve in the input order
    '''
    DTWResults=[]
    for aRotation in multiRotations:
        DTWResults.append(
            dtw.distance(motherWave, aRotation)
        )
    sortedDTWResult = np.argsort(DTWResults)
    if drawWarpResult:
        path = dtw.warping_path(motherWave, multiRotations[sortedDTWResult[0]])
        fig, ax = plt.subplots(nrows=2)
        dtwvis.plot_warping(motherWave, multiRotations[sortedDTWResult[0]], path, fig=fig, axs=ax)

        path = dtw.warping_path(motherWave, multiRotations[sortedDTWResult[-1]])
        fig, ax = plt.subplots(nrows=2)
        dtwvis.plot_warping(motherWave, multiRotations[sortedDTWResult[-1]], path, fig=fig, axs=ax)

    return DTWResults

def cropIncreaseDecreaseSegments(rotations: list, globalMaxIdx, globalMinIdx):
    '''
    Use global max/minimum to crop increase and decrease segments from rotation curve
    Input:
    :globalMaxIdx: global maximum indices
    :globalMinIdx: global minimum indices
    '''
    DecreaseSeg=None
    IncreaseSeg=None
    if globalMaxIdx[0]<globalMinIdx[0]:
        DecreaseSeg = rotations[globalMaxIdx[0]:globalMinIdx[0]]
        IncreaseSeg = rotations[globalMinIdx[0]:globalMaxIdx[1]]
    elif globalMaxIdx[0]>globalMinIdx[0]:
        IncreaseSeg = rotations[globalMinIdx[0]:globalMaxIdx[0]]
        DecreaseSeg = rotations[globalMaxIdx[0]:globalMinIdx[1]]
    return DecreaseSeg, IncreaseSeg

def scaleSegmentsToSameCycleLen(handSegment, bodySegment):
    '''
    TODO: body與hand的上升或下降segment調整成有相同的time scale，時間點數量相同
    調整的方式是統一成時間點數量最多的
    輸出的是兩者的時間點資料，時間點數量較多者與輸入時一樣
    '''
    bodyCurveLen = len(bodySegment)
    handCurveLen = len(handSegment)
    bodyCurveTimePoints = None
    handCurveTimePoints = None
    if bodyCurveLen >= handCurveLen:
        handCurveTimePoints = minMaxNormalization(range(handCurveLen), 0, bodyCurveLen-1)
        bodyCurveTimePoints = minMaxNormalization(range(bodyCurveLen), 0, bodyCurveLen-1)
    elif bodyCurveLen < handCurveLen: 
        handCurveTimePoints = minMaxNormalization(range(handCurveLen), 0, handCurveLen-1)
        bodyCurveTimePoints = minMaxNormalization(range(bodyCurveLen), 0, handCurveLen-1)
    return handCurveTimePoints, bodyCurveTimePoints

def bSplineFitting(rotations: list, timeline: list=None, isDrawResult: bool=False):
    ''' 
    B-spline fitting
    Assuming that the sample points is sample in fequency of 1
    '''
    timeline = range(len(rotations)) if timeline is None else timeline
    print('time line :', len(timeline))
    print('rotations: ', len(rotations))
    spl = splrep(timeline, rotations)
    if isDrawResult:
        x = np.linspace(0, timeline[-1], len(rotations)*5)
        y = splev(x, spl)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.plot(timeline, rotations, '.r')
    return spl

def NDbSplineFitting(rotationsPairs: list, smoothingRatio: float=None): 
    '''
    N dimensional B-spline fitting
    The dimensio depends on the input rotations list
    Larger smoothingRatio means more smoothing while smaller values of smoothingRatio indicate less smoothing
    '''
    if smoothingRatio is None:
        spl, _ = splprep(rotationsPairs)
        return spl, _
    spl, _ = splprep(rotationsPairs, s=smoothingRatio)
    return spl, _

def NDBSplineMapping(NDSpline, handRotations, nSamples: int=1000):
    '''
    Use N dimensional B-spline fitting result, 
    to map hand rotation to body rotation by sampling **some** sample points, 
    then find the one most close to the point that want to be mapped
    '''
    interpolatePoints = splev(np.linspace(0, 1, nSamples), NDSpline)
    newHandRotation=handRotations[:, np.newaxis]
    subHandRotations = np.abs(interpolatePoints[0]-newHandRotation)
    minSamplePtIdx = np.argmin(subHandRotations, axis=1)

    # print(interpolatePoints[0].shape)
    # print(handRotations.shape)
    # print(newHandRotation.shape)
    # print((interpolatePoints[0]-newHandRotation).shape)
    # print(minSamplePtIdx)
    # print(subHandRotations)

    return interpolatePoints[1][minSamplePtIdx]

def drawPlot(x, y):
    plt.figure()
    plt.plot(x, y, '.-')
    
usedJointIdx = [['x','z'], ['x'], ['x','z'], ['x']]

if __name__=="__main__":
    handJointsRotations=None
    # fileName = './HandRotationOuputFromHomePC/leftFrontKick.json'
    fileName = './HandRotationOuputFromHomePC/leftSideKick.json'
    # fileName = 'leftFrontKickingBody.json'
    with open(fileName, 'r') as fileOpen: 
        rotationJson=json.load(fileOpen)
        # print(type(rotationJson))
        # print(list(rotationJson.keys()))
        # print(type(rotationJson['results']))
        handJointsRotations = rotationJsonDataParser(rotationJson, jointCount=4)
    # Filter the time series data
    filteredHandJointRots = handJointsRotations.copy()
    for aJointIdx in range(len(handJointsRotations)):
        aJointData=handJointsRotations[aJointIdx]
        for k, aAxisRotationData in aJointData.items():
            aAxisRotationData = adjustRotationDataTo180(aAxisRotationData)
            # drawPlot(range(len(aAxisRotationData)), aAxisRotationData)
            # aFreqRotationData = adjustRotationByFFT(aAxisRotationData)
            # drawPlot(range(len(aFreqRotationData)), aFreqRotationData)

            filteredRotaion = butterworthLowPassFilter(aAxisRotationData)
            # drawPlot(range(len(aAxisRotationData)), filteredRotaion)
            # filteredFreqRotaion = adjustRotationByFFT(filteredRotaion)
            # drawPlot(range(len(filteredFreqRotaion)), filteredFreqRotaion)

            # gaussainRotationData = gaussianFilter(filteredRotaion, 0.7)
            gaussainRotationData = gaussianFilter(filteredRotaion, 2)
            # drawPlot(range(len(gaussainRotationData)), gaussainRotationData)

            # autoCorrelation(gaussainRotationData, True)
            filteredHandJointRots[aJointIdx][k]=gaussainRotationData
            
    # Find repeat patterns of the filtered data(using autocorrelation)
    jointsACorr = []
    jointsACorrLocalMaxIdx = []
    for aJointIdx in range(len(filteredHandJointRots)):
        aJointData=filteredHandJointRots[aJointIdx]
        for k in usedJointIdx[aJointIdx]:
            # drawPlot(range(len(aAxisRotationData)), aAxisRotationData)
            jointsACorr.append(autoCorrelation(aJointData[k], False))
            localMaxIdx, = findLocalMaximaIdx(jointsACorr[-1])
            localMaxIdx = [i for i in localMaxIdx if jointsACorr[-1][i]>0]# The local maximum need to correspond to a positive correlation
            jointsACorrLocalMaxIdx.append(localMaxIdx[0])
            # plt.plot(localMaxIdx, [jointsACorr[-1][i] for i in localMaxIdx], 'r.')

    repeatingPatternCycle = sum(jointsACorrLocalMaxIdx) / len(jointsACorrLocalMaxIdx)    # 出現重複模式的週期
    # TODO: 改成使用weighted sum來計算repeating pattern可能會比較好，weight的來源可以使用下一步計算的correlation
    print('Hand cycle: ', repeatingPatternCycle)

    # Compute the repeat pattern by simply average all the pattern candidates
    # [Alter choice] Just pick top 3(k) patterns that has most correlation with each other and average them
    # [Use weighted average, since not all the pattern's quality are equal
    # the pattern has high correlation with more other patterns get higher weight]
    handJointsPatternData=[{k: [] for k in axis} for axis in usedJointIdx]
    for aJointIdx in range(len(filteredHandJointRots)):
        aJointData=filteredHandJointRots[aJointIdx]
        for k in usedJointIdx[aJointIdx]:
            splitedRotation = splitRotation(aJointData[k], repeatingPatternCycle, True)
            highestCorrIdx, highestCorrs = correlationBtwMultipleSeq(splitedRotation, k=3)
            highestCorrRots = [splitedRotation[i] for i in highestCorrIdx]
            avgHighCorrPattern = averageMultipleSeqs(highestCorrRots, highestCorrs/sum(highestCorrs))
            handJointsPatternData[aJointIdx][k] = avgHighCorrPattern

            # for rotsIdx in highestCorrIdx:
            #     drawPlot(range(len(splitedRotation[rotsIdx])), splitedRotation[rotsIdx])
            # drawPlot(range(len(avgHighCorrPattern)), avgHighCorrPattern)
    drawPlot(range(len(handJointsPatternData[3]['x'])), handJointsPatternData[3]['x'])
    drawPlot(range(len(filteredHandJointRots[3]['x'])), filteredHandJointRots[3]['x'])

    # ======= ======= ======= ======= ======= ======= =======
    # Compare hand and body curve, then compute the mapping function
    # Scale the hand curve to the same time frquency in the body curve
    ## load body curve
    bodyJointRotations=None
    # fileName = 'leftFrontKickingBody.json'
    fileName = './bodyDBRotation/leftSideKick.json'
    with open(fileName, 'r') as fileOpen: 
        rotationJson=json.load(fileOpen)
        bodyJointRotations = rotationJsonDataParser(rotationJson, jointCount=4)
        bodyJointRotations = [{k: bodyJointRotations[aJointIdx][k] for k in bodyJointRotations[aJointIdx]} for aJointIdx in range(len(bodyJointRotations))]
    
    ## Adjust body rotation data
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            bodyJointRotations[aJointIdx][k] = adjustRotationDataTo180(bodyJointRotations[aJointIdx][k])
    

    ## Use average filter on the body rotation data，since we only want a "feasible" body motion
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            bodyJointRotations[aJointIdx][k] = gaussianFilter(bodyJointRotations[aJointIdx][k], 2)
    # drawPlot(range(len(bodyJointRotations[0]['x'])), bodyJointRotations[0]['x'])        

    ## Find repeat pattern's frequency in the body curve
    ## Cause body curve is perfect so only one curve from a single joint single axis need to be computed
    ## [new] 發現autocorrelation還是會出現偶發性錯誤，調整body的數值為
    bodyRepeatPatternCycle=None
    # bodyACorr = autoCorrelation(bodyJointRotations[0]['x'], True) # left front kick
    bodyACorr = autoCorrelation(bodyJointRotations[0]['z'], True)   # left side kick
    bodyLocalMaxIdx, = findLocalMaximaIdx(bodyACorr)
    bodyLocalMaxIdx = [i for i in bodyLocalMaxIdx if bodyACorr[i]>0]
    bodyRepeatPatternCycle=bodyLocalMaxIdx[0]
    print('body cycle: ', bodyRepeatPatternCycle)

    ## [暫且捨棄]Generate each phase of the body curve in a singel cycle length, 
    ## by rolling window method
    bodyJointsRollingWindows = [{k: [] for k in axis} for axis in usedJointIdx]
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            bodyJointsRollingWindows[aJointIdx][k]=\
                rollingWindowSplitRotation(bodyJointRotations[aJointIdx][k][bodyRepeatPatternCycle:3*bodyRepeatPatternCycle], bodyRepeatPatternCycle)
    # drawPlot(range(len(bodyJointsRollingWindows[0]['x'][0])), bodyJointsRollingWindows[0]['x'][0])

    ##  [暫且捨棄]Compute these windows signals with the hand signal's DTW distance,
    ##  and find the one that has the most correlation 
    ## TODO: 這邊hand curve的時間長度與body不同，雖然用了DTW方法，但是效果不佳，或許可以考慮將hand curve先scale到與body curve相同再比較相似度
    handBodyDTWDistances = [{k: [] for k in axis} for axis in usedJointIdx]
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            handBodyDTWDistances[aJointIdx][k]=\
                computeDTWBtwMultiSeqs(handJointsPatternData[aJointIdx][k], bodyJointsRollingWindows[aJointIdx][k], drawWarpResult=False)
    
    minDTWDis = [[[min(disDict[k]), np.argmin(disDict[k])] for k in disDict] for disDict in handBodyDTWDistances]
    minDTWDis2 = []
    for aJointDTWDis in minDTWDis:
        minDTWDis2.extend(aJointDTWDis)
    minDTWStartIdx = min(minDTWDis2, key=lambda x: x[0])
    minDTWStartIdx = minDTWStartIdx[1] + bodyRepeatPatternCycle

    ## [暫且捨棄]Crop the body rotation data from the computed start index to the length of a cycle
    ## See if the data has same number of data points, between hand and body's rotations curve
    ## if the number of data points is not the same, interpolation must be done
    bodyJointsPatterns = [{k: [] for k in axis} for axis in usedJointIdx]
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            bodyJointsPatterns[aJointIdx][k]=bodyJointRotations[aJointIdx][k][minDTWStartIdx:minDTWStartIdx+bodyRepeatPatternCycle]
    drawPlot(range(len(bodyJointRotations[0]['z'])), bodyJointRotations[0]['z'])
    drawPlot(range(len(bodyJointsPatterns[0]['z'])), bodyJointsPatterns[0]['z'])

    ## Find the global maximum and minimum in the hand and body rotation curve
    ## Crop the increase and decrease segment
    bodyDecreaseSegs = [{k: [] for k in axis} for axis in usedJointIdx]
    bodyIncreaseSegs = [{k: [] for k in axis} for axis in usedJointIdx]
    handDecreaseSegs = [{k: [] for k in axis} for axis in usedJointIdx]
    handIncreaseSegs = [{k: [] for k in axis} for axis in usedJointIdx]

    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            bodyJointCurve = np.array(bodyJointsPatterns[aJointIdx][k].tolist()*3)
            handJointCurve = np.array(handJointsPatternData[aJointIdx][k]*3)
            bodyGlobalMax, bodyGlobalMaxIdx = findGlobalMaxAndIdx(bodyJointCurve)
            bodyGlobalMin, bodyGlobalMinIdx = findGlobalMinAndIdx(bodyJointCurve)

            handGlobalMax, handGlobalMaxIdx = findGlobalMaxAndIdx(handJointCurve)
            handGlobalMin, handGlobalMinIdx = findGlobalMinAndIdx(handJointCurve)

            if aJointIdx==0 and k=='x':
                drawPlot(range(len(bodyJointCurve)), bodyJointCurve)
                plt.plot(bodyGlobalMaxIdx, bodyJointCurve[bodyGlobalMaxIdx], 'r.')
                plt.plot(bodyGlobalMinIdx, bodyJointCurve[bodyGlobalMinIdx], 'r.')
                drawPlot(range(len(handJointCurve)), handJointCurve)
                plt.plot(handGlobalMaxIdx, handJointCurve[handGlobalMaxIdx], 'r.')
                plt.plot(handGlobalMinIdx, handJointCurve[handGlobalMinIdx], 'r.')

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

    drawPlot(range(len(bodyDecreaseSegs[0]['x'])), bodyDecreaseSegs[0]['x'])
    drawPlot(range(len(bodyIncreaseSegs[0]['x'])), bodyIncreaseSegs[0]['x'])
    drawPlot(range(len(handDecreaseSegs[0]['x'])), handDecreaseSegs[0]['x'])
    drawPlot(range(len(handIncreaseSegs[0]['x'])), handIncreaseSegs[0]['x'])

    print('Hand pattern length: ', len(handJointsPatternData[0]['z']))
    print('Body pattern length: ', len(bodyJointsPatterns[0]['z']))
    print('body increase segment length: ', len(bodyIncreaseSegs[1]['x']))
    print('hand increase segment length: ', len(handIncreaseSegs[1]['x']))
    print('body decrease segment length: ', len(bodyDecreaseSegs[1]['x']))
    print('hand decrease segment length: ', len(handDecreaseSegs[1]['x']))

    ## Scale the hand curve and make it competible with body rotation curve
    ## (Scaling method: B-Spline fitting then interpolation, minMax)
    ## 先做minmax scaling將兩者總時長調整成相同，再使用B-Spline fitting
    bodyTimePoints = [
        [{k: [] for k in axis} for axis in usedJointIdx], [{k: [] for k in axis} for axis in usedJointIdx]
    ]
    handTimePoints = [
        [{k: [] for k in axis} for axis in usedJointIdx], [{k: [] for k in axis} for axis in usedJointIdx]
    ]
    bodySplines = [
        [{k: [] for k in axis} for axis in usedJointIdx], [{k: [] for k in axis} for axis in usedJointIdx]
    ]
    handSplines = [
        [{k: [] for k in axis} for axis in usedJointIdx], [{k: [] for k in axis} for axis in usedJointIdx]
    ]

    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            for inc_dec in range(2):
                handTimePoints[inc_dec][aJointIdx][k], bodyTimePoints[inc_dec][aJointIdx][k] = \
                    scaleSegmentsToSameCycleLen(handSegs[inc_dec][aJointIdx][k], bodySegs[inc_dec][aJointIdx][k])
                bodySplines[inc_dec][aJointIdx][k] = bSplineFitting(bodySegs[inc_dec][aJointIdx][k], timeline=bodyTimePoints[inc_dec][aJointIdx][k], isDrawResult=False)
                handSplines[inc_dec][aJointIdx][k] = bSplineFitting(handSegs[inc_dec][aJointIdx][k], timeline=handTimePoints[inc_dec][aJointIdx][k], isDrawResult=False)

    print(bodySplines[1][0]['z'])

    # handTimePoint, bodyTimePoint = \
    #     scaleSegmentsToSameCycleLen(handIncreaseSegs[0]['z'], bodyIncreaseSegs[0]['z'])

    # bodyDecreaseSegBSpline = bSplineFitting(bodyIncreaseSegs[0]['z'], timeline=bodyTimePoint, isDrawResult=True)
    # handDecreaseSegBSpline = bSplineFitting(handIncreaseSegs[0]['z'], timeline=handTimePoint, isDrawResult=True)
    # print(bodyDecreaseSegBSpline)

    ## Sample points from the body and hand curves(increase and decrease segments)
    ## number of sample points are set as the double of the frequency fo the curve which is higher
    ## Use the sampled points to construct the final mapping function and fit by a B-Spline
    ## Compute all the joints and axes' rotations
    timelinesArrs = [
        [{k: [] for k in axis} for axis in usedJointIdx], [{k: [] for k in axis} for axis in usedJointIdx]
    ]
    handSamplePointsArrs = [
        [{k: [] for k in axis} for axis in usedJointIdx], [{k: [] for k in axis} for axis in usedJointIdx]
    ]
    bodySamplePointsArrs = [
        [{k: [] for k in axis} for axis in usedJointIdx], [{k: [] for k in axis} for axis in usedJointIdx]
    ]
    mappingFuncBSplines = [
        [{k: [] for k in axis} for axis in usedJointIdx], [{k: [] for k in axis} for axis in usedJointIdx]
    ]

    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            for inc_dec in range(2):
                numSamplePoints = max(len(handTimePoints[inc_dec][aJointIdx][k]), len(bodyTimePoints[inc_dec][aJointIdx][k]))*2
                timelinesArrs[inc_dec][aJointIdx][k] = np.linspace(0, bodyTimePoints[inc_dec][aJointIdx][k][-1], numSamplePoints)
                bodySamplePointsArrs[inc_dec][aJointIdx][k] = splev(timelinesArrs[inc_dec][aJointIdx][k], bodySplines[inc_dec][aJointIdx][k])
                handSamplePointsArrs[inc_dec][aJointIdx][k] = splev(timelinesArrs[inc_dec][aJointIdx][k], handSplines[inc_dec][aJointIdx][k])
                mappingFuncBSplines[inc_dec][aJointIdx][k] = NDbSplineFitting(
                    [
                        handSamplePointsArrs[inc_dec][aJointIdx][k], 
                        bodySamplePointsArrs[inc_dec][aJointIdx][k]
                    ], 
                    smoothingRatio=len(handSamplePointsArrs[inc_dec][aJointIdx][k]) - \
                        (2*len(handSamplePointsArrs[inc_dec][aJointIdx][k]))**(1/2)
                )
    print(bodySamplePointsArrs[0][0]['x'])
    print(handSamplePointsArrs[0][0]['x'])
    fig, ax=plt.subplots()
    ax.plot(handSamplePointsArrs[0][0]['x'], bodySamplePointsArrs[0][0]['x'], '.-')

    fig, ax=plt.subplots()
    ax.plot(handSamplePointsArrs[0][0]['x'], bodySamplePointsArrs[0][0]['x'], '.-')
    interpolatePoints = splev(np.linspace(0, 1, 1000), mappingFuncBSplines[0][0]['x'][0])
    ax.plot(interpolatePoints[0], interpolatePoints[1], 'r--')

    fig, ax=plt.subplots()
    ax.plot(handSamplePointsArrs[1][0]['x'], bodySamplePointsArrs[1][0]['x'], '.-')
    interpolatePoints = splev(np.linspace(0, 1, 1000), mappingFuncBSplines[1][0]['x'][0])
    ax.plot(interpolatePoints[0], interpolatePoints[1], 'r--')

    ## Use mapping function(BSpline) to map a hand rotation to body rotation
    ## This is just a testing(verification), 
    ## the input hand sequence is the sequence used to construct mapping function
    mappedBodyRotations = NDBSplineMapping(mappingFuncBSplines[0][0]['x'][0], handSamplePointsArrs[0][0]['x'])
    # print(mappedBodyRotations)

    ## Map the hand rotations(the entire signals) to body rotation
    ## Use the signal after low pass filter and moving average filter maybe a good choice
    ## Split hand rotation by the repeat pattern frequency computed earlier, 
    ## Finally use the splitted data to find the global maximum and minimum's index in the 
    ## original full signals
    ## splitedRotation = splitRotation(aJointData[k], repeatingPatternCycle, True), line 343
    repeatingPatternCycle = int(repeatingPatternCycle)
    handJointsSplitedData=[{k: [] for k in axis} for axis in usedJointIdx]
    handCurvesGlobalMaxMinIdx=[{k: [] for k in axis} for axis in usedJointIdx]
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            handJointsSplitedData[aJointIdx][k] = \
                splitRotation(filteredHandJointRots[aJointIdx][k], repeatingPatternCycle, False)
            # 每個splited segment都要找到min and max
            minmaxLabelSeq = []  # 紀錄最大最小值的
            for i, splitedSeg in enumerate(handJointsSplitedData[aJointIdx][k]):
                handGlobalMax, handGlobalMaxIdx = findGlobalMaxAndIdx(splitedSeg)
                handGlobalMin, handGlobalMinIdx = findGlobalMinAndIdx(splitedSeg)

                # fig, ax=plt.subplots()
                # ax.plot(range(len(splitedSeg)), splitedSeg)
                # ax.plot(handGlobalMaxIdx, splitedSeg[handGlobalMaxIdx], 'r.')
                # ax.plot(handGlobalMinIdx, splitedSeg[handGlobalMinIdx], 'r.')
                minMaxIdx = [handGlobalMinIdx[0]+i*repeatingPatternCycle, handGlobalMaxIdx[0]+i*repeatingPatternCycle] \
                    if handGlobalMinIdx[0]<handGlobalMaxIdx[0] else [handGlobalMaxIdx[0]+i*repeatingPatternCycle, handGlobalMinIdx[0]+i*repeatingPatternCycle]
                handCurvesGlobalMaxMinIdx[aJointIdx][k].extend(minMaxIdx)

                minmaxLabel = [0, 1] if handGlobalMinIdx[0]<handGlobalMaxIdx[0] else [1, 0]
                minmaxLabelSeq.extend(minmaxLabel)

            globalMaxMinIdx = handCurvesGlobalMaxMinIdx[aJointIdx][k]
            deleteIndices = []
            # 處理最大或最小相鄰的問題，合併後兩個最大值取較大的作為最大值，兩個最小值取較小的作為最小值
            for i in range(0, len(minmaxLabelSeq)-2, 2):
                if minmaxLabelSeq[i+1]==minmaxLabelSeq[i+2]:
                    # if consecutive minimums, choose the lower one. Same way apply to maximum
                    if filteredHandJointRots[aJointIdx][k][globalMaxMinIdx[i+1]] < filteredHandJointRots[aJointIdx][k][globalMaxMinIdx[i+2]]:
                        if minmaxLabelSeq[i+1] == 0:
                            deleteIndices.append(i+2)
                        else:
                            deleteIndices.append(i+1)
                    else: 
                        if minmaxLabelSeq[i+1] == 0:
                            deleteIndices.append(i+1)
                        else:
                            deleteIndices.append(i+2)
            handCurvesGlobalMaxMinIdx[aJointIdx][k] = \
                np.delete(handCurvesGlobalMaxMinIdx[aJointIdx][k], deleteIndices)


    fig, ax=plt.subplots()
    ax.plot(range(len(handJointsSplitedData[2]['x'][0])), handJointsSplitedData[2]['x'][0], '.-')
    fig, ax=plt.subplots()
    ax.plot(range(len(filteredHandJointRots[0]['z'])), filteredHandJointRots[0]['z'], '.-')
    ax.plot(handCurvesGlobalMaxMinIdx[0]['z'], filteredHandJointRots[0]['z'][handCurvesGlobalMaxMinIdx[0]['z']], 'r.')
    fig, ax=plt.subplots()
    ax.plot(range(len(bodyJointRotations[0]['z'])), bodyJointRotations[0]['z'], '.-')

    # Apply the mapping function to each increase and decrease segments
    afterMappingBodyCurve = [{k: [] for k in axis} for axis in usedJointIdx]
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            minmaxindices = handCurvesGlobalMaxMinIdx[aJointIdx][k]
            handCurve = filteredHandJointRots[aJointIdx][k]
            incOrDec = True if handCurve[minmaxindices[0]] < handCurve[minmaxindices[1]] else False
            # Signal before first relative extrema need to be mapped too
            if not incOrDec:
                afterMappingBodyCurve[aJointIdx][k].extend(
                    NDBSplineMapping(mappingFuncBSplines[1][aJointIdx][k][0], handCurve[:minmaxindices[0]])
                )
            else: 
                afterMappingBodyCurve[aJointIdx][k].extend(
                    NDBSplineMapping(mappingFuncBSplines[0][aJointIdx][k][0], handCurve[:minmaxindices[0]])
                )
            # Signal between relative extrema mapping
            for i in range(len(minmaxindices)-1):
                if incOrDec:
                    afterMappingBodyCurve[aJointIdx][k].extend(
                        NDBSplineMapping(mappingFuncBSplines[1][aJointIdx][k][0], handCurve[minmaxindices[i]:minmaxindices[i+1]])
                    )
                else: 
                    afterMappingBodyCurve[aJointIdx][k].extend(
                        NDBSplineMapping(mappingFuncBSplines[0][aJointIdx][k][0], handCurve[minmaxindices[i]:minmaxindices[i+1]])
                    )
                incOrDec = not incOrDec
            # Signal after last relative extrema need to be mapped too
            if incOrDec:
                afterMappingBodyCurve[aJointIdx][k].extend(
                    NDBSplineMapping(mappingFuncBSplines[1][aJointIdx][k][0], handCurve[minmaxindices[-1]:])
                )
            else: 
                afterMappingBodyCurve[aJointIdx][k].extend(
                    NDBSplineMapping(mappingFuncBSplines[0][aJointIdx][k][0], handCurve[minmaxindices[-1]:])
                )
    
    fig, ax=plt.subplots()
    ax.plot(range(len(afterMappingBodyCurve[0]['z'])), afterMappingBodyCurve[0]['z'], '.-')

    # 從-180~180轉換回0~360
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            afterMappingBodyCurve[aJointIdx][k] = adjustRotationDataFrom180To360(afterMappingBodyCurve[aJointIdx][k])

    ## 輸出mapping過後的rotations curve
    ## 轉換成與之前輸出landmark相同的格式，最後以json檔輸出
    import json
    outputJointCat = [{'x', 'y', 'z'}, {'x', 'y', 'z'}, {'x', 'y', 'z'}, {'x', 'y', 'z'}]
    outputData = [{'time': i, 'data': [{k: 0 for k in axis} for axis in outputJointCat]} for i in range(len(afterMappingBodyCurve[0]['x']))]
    
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            for i in range(len(afterMappingBodyCurve[aJointIdx][k])):
                outputData[i]['data'][aJointIdx][k] = \
                    afterMappingBodyCurve[aJointIdx][k][i]
            
    # Output by json dump
    # with open('./handRotaionAfterMapping/leftFrontKick.json', 'w') as WFile: 
    #     json.dump(outputData, WFile)
    #     # print(json.dumps(afterMappingBodyCurve))
    #     pass

    ## 需要能夠指定哪一些joint需要mapping，哪一些joint不要mapping，並且輸出所有種類的排列組合
    ## Unmapped hand data: filteredHandJointRots
    ## Mapped hand data: afterMappingBodyCurve
    usedJointEnum = []
    for i, k in enumerate(usedJointIdx):
        for j in k:
            usedJointEnum.append([i, j])
    trueFalseValue = list(itertools.product([True, False], repeat=len(usedJointEnum)))
    outputData = [{'time': i, 'data': [{k: 0 for k in axis} for axis in outputJointCat]} for i in range(len(afterMappingBodyCurve[0]['x']))]

    for _trueFalseVal in trueFalseValue:
        for _idx, _idxAxisPair in enumerate(usedJointEnum):
            i = _idxAxisPair[0]
            k = _idxAxisPair[1]
            if _trueFalseVal[_idx]:
                for t in range(len(afterMappingBodyCurve[i][k])):
                    outputData[t]['data'][i][k] = \
                        afterMappingBodyCurve[i][k][t]
            elif not _trueFalseVal[_idx]:
                for t in range(len(filteredHandJointRots[i][k])):
                    outputData[t]['data'][i][k] = \
                        filteredHandJointRots[i][k][t]
        # with open('./handRotaionAfterMapping/leftSideKick/leftSideKick{0}.json'.format(str(_trueFalseVal)), 'w') as WFile: 
        #     json.dump(outputData, WFile)




    ## TODO: 可能要先確定一下，為什兩者的角度範圍差異有點大(<-- working on this task)
    
    plt.show()