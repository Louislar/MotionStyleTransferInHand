import numpy as np
import pandas as pd
import scipy
import scipy.signal as sig
from scipy.ndimage import gaussian_filter1d
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import json
import matplotlib.pyplot as plt
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
    '''
    localMaxIdx, = findLocalMaximaIdx(rotations)
    globalMax = max(rotations[localMaxIdx])
    globalMaxIdx, = np.where(rotations==globalMax)
    return globalMax, globalMaxIdx

def findGlobalMinAndIdx(rotations: list):
    '''
    Find global minimum and corresponding index, 
    the global minimum will be one of the local minimums
    '''
    localMinIdx, = findLocalMinimaIdx(rotations)
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

def bSplineFitting(handRots, bodyRots):
    # TODO: Finish B-spline fitting
    pass

def drawPlot(x, y):
    plt.figure()
    plt.plot(x, y, '.-')
    
usedJointIdx = [['x','z'], ['x'], ['x','z'], ['x']]

if __name__=="__main__":
    handJointsRotations=None
    fileName = 'leftFrontKick.json'
    # fileName = 'leftFrontKickingBody.json'
    with open(fileName, 'r') as fileOpen: 
        rotationJson=json.load(fileOpen)
        print(type(rotationJson))
        print(list(rotationJson.keys()))
        print(type(rotationJson['results']))
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
            
    # Find repeat patterns of the filtered data(using autocrrelation)
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
    # TODO: 改成使用weighted sum來計算repearing pattern可能會比較好，weight的來源可以使用下一步計算的correlation
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
    drawPlot(range(len(handJointsPatternData[0]['z'])), handJointsPatternData[0]['z'])
    drawPlot(range(len(filteredHandJointRots[0]['z'])), filteredHandJointRots[0]['z'])
    

    # ======= ======= ======= ======= ======= ======= =======
    # Compare hand and body curve, then compute the mapping function
    # Scale the hand curve to the same time frquency in the body curve
    ## load body curve
    bodyJointRotations=None
    fileName = 'leftFrontKickingBody.json'
    with open(fileName, 'r') as fileOpen: 
        rotationJson=json.load(fileOpen)
        bodyJointRotations = rotationJsonDataParser(rotationJson, jointCount=4)
        bodyJointRotations = [{k: bodyJointRotations[aJointIdx][k] for k in bodyJointRotations[aJointIdx]} for aJointIdx in range(len(bodyJointRotations))]
    
    ## Adjust body rotation data
    # for aJointIdx in range(len(usedJointIdx)):
    #     for k in usedJointIdx[aJointIdx]:
    #         bodyJointRotations[aJointIdx][k] = adjustRotationDataTo180(bodyJointRotations[aJointIdx][k])
    drawPlot(range(len(bodyJointRotations[0]['z'])), bodyJointRotations[0]['z'])

    ## Use average filter on the body rotation data，since we only want a "feasible" body motion
    for aJointIdx in range(len(usedJointIdx)):
        for k in usedJointIdx[aJointIdx]:
            bodyJointRotations[aJointIdx][k] = gaussianFilter(bodyJointRotations[aJointIdx][k], 2)
            

    ## Find repeat pattern's frequency in the body curve
    ## Cause body curve is perfect so only one curve from a single joint single axis need to be computed
    bodyRepeatPatternCycle=None
    bodyACorr = autoCorrelation(bodyJointRotations[0]['x'], True)
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

    ## Find the global maximum and minimum in the hand and body rotation curve
    ## TODO: Crop the increase and decrease segment
    
    bodyJointCurve = np.array(bodyJointsPatterns[0]['z'].tolist()*3)
    handJointCurve = np.array(handJointsPatternData[0]['z']*3)
    bodyGlobalMax, bodyGlobalMaxIdx = findGlobalMaxAndIdx(bodyJointCurve)
    bodyGlobalMin, bodyGlobalMinIdx = findGlobalMinAndIdx(bodyJointCurve)

    handGlobalMax, handGlobalMaxIdx = findGlobalMaxAndIdx(handJointCurve)
    handGlobalMin, handGlobalMinIdx = findGlobalMinAndIdx(handJointCurve)

    # drawPlot(range(len(bodyJointCurve)), bodyJointCurve)
    # plt.plot(bodyGlobalMaxIdx, bodyJointCurve[bodyGlobalMaxIdx], 'r.')
    # plt.plot(bodyGlobalMinIdx, bodyJointCurve[bodyGlobalMinIdx], 'r.')
    # drawPlot(range(len(handJointCurve)), handJointCurve)
    # plt.plot(handGlobalMaxIdx, handJointCurve[handGlobalMaxIdx], 'r.')
    # plt.plot(handGlobalMinIdx, handJointCurve[handGlobalMinIdx], 'r.')

    bodyDecreaseSeg, bodyIncreaseSeg = cropIncreaseDecreaseSegments(bodyJointCurve, bodyGlobalMaxIdx, bodyGlobalMinIdx)
    handDecreaseSeg, handIncreaseSeg = cropIncreaseDecreaseSegments(handJointCurve, handGlobalMaxIdx, handGlobalMinIdx)
    
    # drawPlot(range(len(bodyDecreaseSeg)), bodyDecreaseSeg)
    # drawPlot(range(len(bodyIncreaseSeg)), bodyIncreaseSeg)
    # drawPlot(range(len(handDecreaseSeg)), handDecreaseSeg)
    # drawPlot(range(len(handIncreaseSeg)), handIncreaseSeg)

    print('Hand pattern length: ', len(handJointsPatternData[0]['x']))
    print('Body pattern length: ', len(bodyJointsPatterns[0]['x']))
    print('body increase segment length: ', len(bodyIncreaseSeg))
    print('hand increase segment length: ', len(handIncreaseSeg))
    print('body decrease segment length: ', len(bodyDecreaseSeg))
    print('hand decrease segment length: ', len(handDecreaseSeg))

    ## TODO: Scale the hand curve and make it competible with body rotation curve
    ## (Two scaling method: B-Spline fitting than interpolation, minMax)
    ## TODO: 可能要先確定一下，為什兩者的角度範圍差異有點大
    ## TODO: Sample points from the body and hand curves(increase and decrease segments)
    ## TODO: Use the sampled points to construct the final mapping function and fit by a B-Spline
    
    plt.show()