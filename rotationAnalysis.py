import numpy as np
import scipy
import scipy.signal as sig
from scipy.ndimage import gaussian_filter1d
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import json
import matplotlib.pyplot as plt

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
    return [rotations[i:i + size] for i in range(0, len(rotations), size)]

def correlationBtwMultipleSeq(splitedRotations: list):
    '''
    Compute multiple sequences' correlation with each other
    '''
    pass
    
def drawPlot(x, y):
    plt.figure()
    plt.plot(x, y, '.-')
    

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
    # drawPlot(range(len(filteredHandJointRots[0]['x'])), filteredHandJointRots[0]['x'])
    jointsACorr.append(autoCorrelation(filteredHandJointRots[0]['x'], False))
    localMaxIdx, = findLocalMaximaIdx(jointsACorr[-1])
    localMaxIdx = [i for i in localMaxIdx if jointsACorr[-1][i]>0]# The local maximum need to correspond to a positive correlation
    jointsACorrLocalMaxIdx.append(localMaxIdx[0])
    # plt.plot(localMaxIdx, [jointsACorr[-1][i] for i in localMaxIdx], 'r.')

    # drawPlot(range(len(filteredHandJointRots[0]['z'])), filteredHandJointRots[0]['z'])
    jointsACorr.append(autoCorrelation(filteredHandJointRots[0]['z'], False))
    localMaxIdx, = findLocalMaximaIdx(jointsACorr[-1])
    localMaxIdx = [i for i in localMaxIdx if jointsACorr[-1][i]>0]# The local maximum need to correspond to a positive correlation
    jointsACorrLocalMaxIdx.append(localMaxIdx[0])
    # plt.plot(localMaxIdx, [jointsACorr[-1][i] for i in localMaxIdx], 'r.')
    
    # drawPlot(range(len(filteredHandJointRots[1]['x'])), filteredHandJointRots[1]['x'])
    jointsACorr.append(autoCorrelation(filteredHandJointRots[1]['x'], False))
    localMaxIdx, = findLocalMaximaIdx(jointsACorr[-1])
    localMaxIdx = [i for i in localMaxIdx if jointsACorr[-1][i]>0]# The local maximum need to correspond to a positive correlation
    jointsACorrLocalMaxIdx.append(localMaxIdx[0])
    # plt.plot(localMaxIdx, [jointsACorr[-1][i] for i in localMaxIdx], 'r.')

    # drawPlot(range(len(filteredHandJointRots[2]['x'])), filteredHandJointRots[2]['x'])
    jointsACorr.append(autoCorrelation(filteredHandJointRots[2]['x'], False))
    localMaxIdx, = findLocalMaximaIdx(jointsACorr[-1])
    localMaxIdx = [i for i in localMaxIdx if jointsACorr[-1][i]>0]# The local maximum need to correspond to a positive correlation
    jointsACorrLocalMaxIdx.append(localMaxIdx[0])
    # plt.plot(localMaxIdx, [jointsACorr[-1][i] for i in localMaxIdx], 'r.')

    # drawPlot(range(len(filteredHandJointRots[2]['z'])), filteredHandJointRots[2]['z'])
    jointsACorr.append(autoCorrelation(filteredHandJointRots[2]['z'], False))
    localMaxIdx, = findLocalMaximaIdx(jointsACorr[-1])
    localMaxIdx = [i for i in localMaxIdx if jointsACorr[-1][i]>0]# The local maximum need to correspond to a positive correlation
    jointsACorrLocalMaxIdx.append(localMaxIdx[0])
    # plt.plot(localMaxIdx, [jointsACorr[-1][i] for i in localMaxIdx], 'r.')

    # drawPlot(range(len(filteredHandJointRots[3]['x'])), filteredHandJointRots[3]['x'])
    jointsACorr.append(autoCorrelation(filteredHandJointRots[3]['x'], False))
    localMaxIdx, = findLocalMaximaIdx(jointsACorr[-1])
    localMaxIdx = [i for i in localMaxIdx if jointsACorr[-1][i]>0]# The local maximum need to correspond to a positive correlation
    jointsACorrLocalMaxIdx.append(localMaxIdx[0])
    # plt.plot(localMaxIdx, [jointsACorr[-1][i] for i in localMaxIdx], 'r.')

    repeatingPatternCycle = sum(jointsACorrLocalMaxIdx) / len(jointsACorrLocalMaxIdx)    # 出現重複模式的週期
    print(repeatingPatternCycle)

    # Compute the repeat pattern by simply average all the pattern candidates
    # Just pick top 3(k) that has most correlation with each other and average them
    # [Use weighted average, since not all the pattern's quality are equal
    # the pattern has high correlation with more other patterns get higher weight]
    jointsPatternData=None
    splitedRotation = splitRotation(filteredHandJointRots[2]['x'], repeatingPatternCycle)
    print(len(splitedRotation))
    for rots in splitedRotation:
        drawPlot(range(len(rots)), rots)
    
    # Compare hand and body curve, then compute the mapping function
    # Scale the hand curve to the same time frquency in the body curve
    plt.show()