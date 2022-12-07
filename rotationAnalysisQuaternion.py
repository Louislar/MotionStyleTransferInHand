'''
轉換成四元數後再進行linear mapping或是B-Spline mapping
大部分功能參考rotationAnalysisNew.py 
'''

import json
import pandas as pd 
import numpy as np 
import pickle
import os 
import copy 
import matplotlib.pyplot as plt 
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splev, splrep 
from rotationAnalysis import rotationJsonDataParser, \
    adjustRotationDataTo180, butterworthLowPassFilter, gaussianFilter, \
    minMaxNormalization, autoCorrelation, findLocalMaximaIdx, splitRotation, \
    correlationBtwMultipleSeq, averageMultipleSeqs, findGlobalMaxAndIdx, \
    findGlobalMinAndIdx, simpleLinearFitting, cropIncreaseDecreaseSegments, \
    bSplineFitting
from rotationAnalysisNew import applyLinearMapFunc, applyBSplineMapFunc

quatIndex = [['x','y','z','w'], ['x','y','z','w'], ['x','y','z','w'], ['x','y','z','w']]
# unusedJointAxisIdx = [['y', 'z'], ['y', 'z'], ['y'], ['y', 'z']]     # 需要先決定mapping strategy 
unusedJointAxisIdx = [['x', 'y'], ['y', 'z'], ['y', 'z'], ['y', 'z']]    # For side kick

def eularToQuat(eularStream):
    '''
    TODO: 目前還不知道這邊轉換成quaternion時, 順序會不會有很大的影響. 
    Ref: https://stackoverflow.com/questions/53033620/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr
    '''
    quatStream = []
    for _eularRot in eularStream:
        quatStream.append(
            Rotation.from_euler('xyz', _eularRot, degrees=True).as_quat()
        )
    return np.array(quatStream)

# 建立四元數版本的linear mapping function 
def constructLinearMappingOnQuat(handRotationFilePath, bodyRotationFilePath, outputFilePath):
    '''
    1. (hand) read hand rotation
    2. (hand) adjust range to [-180, 180] 
    3. (hand) low pass and average filter
    --> 這邊就要決定需要mapping哪一些joint's axes. 需要決定好mapping strategy.
        轉換到quat的rotations只有那些需要mapping的轉軸, 其餘旋轉數值清成0 
    4. (hand) convert to quaternion
    5. (hand) apply gaussian filter to quaternion 
    --> 注意, 有過gaussian數值就要調整回原本的範圍. (包含上面在eular做gaussian也要調整回來)
    6. (hand) autocorrelation for repeating pattern finding 
    7. (hand) find best repeating pattern
    8. (hand) extract min and max (以上做這些只是了避免找到雜訊與outlier)
    === === ===
    9. (body) read body rotation 
    10. (body) 不要前幾個時間點的訊號  
    11. (body) adjust range to [-180, 180] 
    12. (body) convert to quaternion
    13. (body) find min and max in quat
    === === === 
    14. (mixed) construct linear mapping function by 
                min and max of hand  and body quat 
    === === === 
    n. save all the data 
    '''

    # 1. 
    handJointsRotations=None
    with open(handRotationFilePath, 'r') as fileOpen: 
        rotationJson=json.load(fileOpen)
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
    # 4. convert to quat
    quatJointRots = {i: None for i in range(len(handJointsRotations))}
    ## bad mapping轉軸要先清成0
    for aJointIdx in range(len(unusedJointAxisIdx)): 
        for k in unusedJointAxisIdx[aJointIdx]:
            filteredHandJointRots[aJointIdx][k] = [0 for i in filteredHandJointRots[aJointIdx][k]]
    for aJointIdx in range(len(handJointsRotations)): 
        _eularStream = [
            list(i) for i in zip(*[filteredHandJointRots[aJointIdx][k] for k in ['x', 'y', 'z']])
        ]
        _quatStream = eularToQuat(_eularStream)
        # print(_quatStream)
        _quatX = _quatStream.T[0]
        _quatY = _quatStream.T[1]
        _quatZ = _quatStream.T[2]
        _quatW = _quatStream.T[3]
        quatJointRots[aJointIdx] = {
            'x': _quatX,
            'y': _quatY, 
            'z': _quatZ,
            'w': _quatW
        }
    # 5. gaussian on quat
    quatGaussianRots = copy.deepcopy(quatJointRots)
    for aJointIdx in range(len(handJointsRotations)):
        for k, _quat in quatGaussianRots[aJointIdx].items():
            quatGaussianRots[aJointIdx][k] = gaussian_filter1d(_quat, sigma=3)
            _min = np.min(quatJointRots[aJointIdx][k])
            _max = np.max(quatJointRots[aJointIdx][k])
            ## 調整回原本的數值範圍
            ## 如果數值都一樣就不用調整
            if _min == _max:
                continue
            quatGaussianRots[aJointIdx][k] = np.array(minMaxNormalization(quatGaussianRots[aJointIdx][k], _min, _max))
    
    # 6. (hand) autocorrelation on quat for repeating pattern finding 
    handAutoCorr = {i: {j: None for j in quatIndex[i]} for i in range(len(handJointsRotations))}
    handAutoCorrLocalMaxInd = {i: {j: None for j in quatIndex[i]} for i in range(len(handJointsRotations))}
    for aJointIdx in range(len(handJointsRotations)):
        for k, _quat in quatGaussianRots[aJointIdx].items():
            # 這邊有機會會出錯. 因為有的curve是完全等於0 (或是一個常數), 他的autocorrelation結果也會是常數0
            # 這邊需要想辦法排除掉這種旋轉軸的資料
            # 這種轉軸會找不到local max index, 給予None. 
            # 並且在接下來球frequency的計算當中, 不加入計算
            if not np.any(_quat):
                handAutoCorr[aJointIdx][k] = None
                handAutoCorrLocalMaxInd[aJointIdx][k] = None
                continue
            handAutoCorr[aJointIdx][k] = autoCorrelation(_quat, False)
            localMaxIdx, = findLocalMaximaIdx(handAutoCorr[aJointIdx][k])
            localMaxIdx = [i for i in localMaxIdx if handAutoCorr[aJointIdx][k][i]>0]
            handAutoCorrLocalMaxInd[aJointIdx][k] = localMaxIdx[0]
    allFrequency = \
        [handAutoCorrLocalMaxInd[i][k] for i, aJointAxis in enumerate(quatIndex) for k in aJointAxis]
    allFrequency = [i for i in allFrequency if i is not None]
    handRepeatingCycle = int(sum(allFrequency) / len(allFrequency))
    print('hand frequency: ', handRepeatingCycle)
    
    # 7. (hand) find best curve of repeating pattern  
    handReapeatingPattern = {i: {j: None for j in quatIndex[i]} for i in range(len(handJointsRotations))}
    for aJointIdx in range(len(handJointsRotations)):
        for k, _quat in quatGaussianRots[aJointIdx].items():
            _rotSegs = splitRotation(_quat, handRepeatingCycle, True)
            highestCorrIdx, highestCorrs = correlationBtwMultipleSeq(_rotSegs, 3)
            highestCorrRots = [_rotSegs[i] for i in highestCorrIdx]
            # rotation全部都是0的curve, correlation會全部都是0, 給予none就是普通average
            _weight = highestCorrs/sum(highestCorrs) if sum(highestCorrs) != 0 else None
            avgHighCorrPattern = averageMultipleSeqs(highestCorrRots, _weight)
            handReapeatingPattern[aJointIdx][k] = avgHighCorrPattern

    # 8. (hand) extract min and max from repeating patterns 
    handGlobalMin = {i: {j: None for j in quatIndex[i]} for i in range(len(handJointsRotations))}
    handGlobalMax = {i: {j: None for j in quatIndex[i]} for i in range(len(handJointsRotations))}
    for aJointIdx in range(len(handJointsRotations)):
        for k, _repeatingPattern in handReapeatingPattern[aJointIdx].items():
            _max, _maxIdx = findGlobalMaxAndIdx(np.array(_repeatingPattern))
            _min, _minIdx = findGlobalMinAndIdx(np.array(_repeatingPattern))
            handGlobalMax[aJointIdx][k] = _max
            handGlobalMin[aJointIdx][k] = _min
            # 顯示hand最大最小值
            print('hand')
            print(aJointIdx, ', ', k)
            print(handGlobalMin[aJointIdx][k], handGlobalMax[aJointIdx][k])
    
    # ======= ======= ======= ======= ======= ======= ======= 
    # 9. (body) read body rot
    bodyJointRotations=None
    with open(bodyRotationFilePath, 'r') as fileOpen: 
        rotationJson=json.load(fileOpen)
        bodyJointRotations = rotationJsonDataParser(rotationJson, jointCount=4)
        bodyJointRotations = [{k: bodyJointRotations[aJointIdx][k] for k in bodyJointRotations[aJointIdx]} for aJointIdx in range(len(bodyJointRotations))]
    
    # 10. (body) 不要前幾個時間點的訊號 
    ## 目前指定不要使用前10個訊號
    for aJointIdx in range(len(bodyJointRotations)):
        for k in bodyJointRotations[aJointIdx]:
            bodyJointRotations[aJointIdx][k] = bodyJointRotations[aJointIdx][k][10:]
    
    # 11. (body) adjust range to [-180, 180] 
    originBodyRot = copy.deepcopy(bodyJointRotations)
    for aJointIdx in range(len(bodyJointRotations)):
        for k in bodyJointRotations[aJointIdx]:
            bodyJointRotations[aJointIdx][k] = adjustRotationDataTo180(bodyJointRotations[aJointIdx][k])
    
    # 12. (body) convert to quat
    bodyQuatJointRots = {i: None for i in range(len(bodyJointRotations))}
    # Body joint要注意y軸以及部分z軸要清成0
    for _jointInd in range(len(unusedJointAxisIdx)):
        for _axis in unusedJointAxisIdx[_jointInd]:
          bodyJointRotations[_jointInd][_axis] = [0 for i in bodyJointRotations[_jointInd][_axis]] 
    for aJointIdx in range(len(bodyJointRotations)):
        _eularStream = [
            list(i) for i in zip(*[bodyJointRotations[aJointIdx][k] for k in ['x', 'y', 'z']])
        ]
        _quatStream = eularToQuat(_eularStream)
        # print(_quatStream)
        _quatX = _quatStream.T[0]
        _quatY = _quatStream.T[1]
        _quatZ = _quatStream.T[2]
        _quatW = _quatStream.T[3]
        bodyQuatJointRots[aJointIdx] = {
            'x': _quatX,
            'y': _quatY, 
            'z': _quatZ,
            'w': _quatW
        }

    # gaussian沒有用, 因為最終我只會看最大與最小值, 直接拿它們建構linear mapping function就好
    # 13. (body) find min and max in quat
    bodyQuatMax = {i: {j: None for j in quatIndex[i]} for i in range(len(quatIndex))}
    bodyQuatMin = {i: {j: None for j in quatIndex[i]} for i in range(len(quatIndex))}
    for aJointIdx in range(len(bodyQuatJointRots)):
        for k in bodyQuatJointRots[aJointIdx].keys():
            _min = np.min(bodyQuatJointRots[aJointIdx][k])
            _max = np.max(bodyQuatJointRots[aJointIdx][k])
            bodyQuatMin[aJointIdx][k] = _min
            bodyQuatMax[aJointIdx][k] = _max
            
    # ======= ======= ======= ======= ======= ======= ======= 
    # 14. (mixed) construct linear mapping function by 
    #           min and max of hand and body quat 
    mappingFuncs = [{k: [] for k in axis} for axis in quatIndex]
    for aJointIdx in range(len(quatIndex)):
        for k in quatIndex[aJointIdx]:
            ## 最大最小值都是0的hand或是body的四元數, 就給None就好, 代表不用做mapping
            if handGlobalMin[aJointIdx][k] == handGlobalMax[aJointIdx][k] and handGlobalMax[aJointIdx][k] == 0:
                mappingFuncs[aJointIdx][k] = None
                continue
            _fittedLine = simpleLinearFitting(
                [handGlobalMin[aJointIdx][k], handGlobalMax[aJointIdx][k]],
                [bodyQuatMin[aJointIdx][k], bodyQuatMax[aJointIdx][k]],
                degree=1
            )
            mappingFuncs[aJointIdx][k] = _fittedLine
            ## print hand and body min and max
            print('=======')
            print(aJointIdx, ', ', k)
            print('hand')
            print([handGlobalMin[aJointIdx][k], handGlobalMax[aJointIdx][k]])
            print('body')
            print([bodyQuatMin[aJointIdx][k], bodyQuatMax[aJointIdx][k]])

    # n. save all the data 
    def _outputData(data, fileNm):
        with open(os.path.join(outputFilePath, fileNm+'.pickle'), 'wb') as WFile:
            pickle.dump(data, WFile)
    _outputData(handJointsRotations, 'handOrigin')
    _outputData(afterAdjRangeJointRots, 'handAfterAdjRange')
    _outputData(afterLowPassJointRots, 'handAfterLowPass')
    _outputData(filteredHandJointRots, 'handAfterGaussian')
    _outputData(quatJointRots, 'quatJointRots')
    _outputData(quatGaussianRots, 'quatGaussianRots')
    _outputData(handAutoCorr, 'handAutoCorr')
    _outputData(handAutoCorrLocalMaxInd, 'handAutoCorrLocalMaxInd')
    _outputData(handReapeatingPattern, 'handReapeatingPattern')
    # =======
    _outputData(originBodyRot, 'bodyOrigin')
    _outputData(bodyJointRotations, 'bodyAfterAdjRange')
    _outputData(bodyQuatJointRots, 'bodyQuatJointRots')
    # ======= 
    _outputData(mappingFuncs, 'mappingFuncs')
    _outputData([handGlobalMin, handGlobalMax], 'handMinMax')
    _outputData([bodyQuatMin, bodyQuatMax], 'bodyMinMax')

def handRotPreproc(handRotationFilePath):
    '''
    Copy from constructLinearMappingOnQuat()
    手部的旋轉預處理
    Output: 
    :handReapeatingPattern: 各個旋轉軸的repeating pattern segments
    '''
    # 1. 
    handJointsRotations=None
    with open(handRotationFilePath, 'r') as fileOpen: 
        rotationJson=json.load(fileOpen)
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
    # 4. convert to quat
    quatJointRots = {i: None for i in range(len(handJointsRotations))}
    ## bad mapping轉軸要先清成0
    for aJointIdx in range(len(unusedJointAxisIdx)): 
        for k in unusedJointAxisIdx[aJointIdx]:
            filteredHandJointRots[aJointIdx][k] = [0 for i in filteredHandJointRots[aJointIdx][k]]
    for aJointIdx in range(len(handJointsRotations)): 
        _eularStream = [
            list(i) for i in zip(*[filteredHandJointRots[aJointIdx][k] for k in ['x', 'y', 'z']])
        ]
        _quatStream = eularToQuat(_eularStream)
        # print(_quatStream)
        _quatX = _quatStream.T[0]
        _quatY = _quatStream.T[1]
        _quatZ = _quatStream.T[2]
        _quatW = _quatStream.T[3]
        quatJointRots[aJointIdx] = {
            'x': _quatX,
            'y': _quatY, 
            'z': _quatZ,
            'w': _quatW
        }
    # 5. gaussian on quat
    quatGaussianRots = copy.deepcopy(quatJointRots)
    for aJointIdx in range(len(handJointsRotations)):
        for k, _quat in quatGaussianRots[aJointIdx].items():
            quatGaussianRots[aJointIdx][k] = gaussian_filter1d(_quat, sigma=3)
            _min = np.min(quatJointRots[aJointIdx][k])
            _max = np.max(quatJointRots[aJointIdx][k])
            ## 調整回原本的數值範圍
            ## 如果數值都一樣就不用調整
            if _min == _max:
                continue
            quatGaussianRots[aJointIdx][k] = np.array(minMaxNormalization(quatGaussianRots[aJointIdx][k], _min, _max))
    
    # 6. (hand) autocorrelation on quat for repeating pattern finding 
    handAutoCorr = {i: {j: None for j in quatIndex[i]} for i in range(len(handJointsRotations))}
    handAutoCorrLocalMaxInd = {i: {j: None for j in quatIndex[i]} for i in range(len(handJointsRotations))}
    for aJointIdx in range(len(handJointsRotations)):
        for k, _quat in quatGaussianRots[aJointIdx].items():
            # 這邊有機會會出錯. 因為有的curve是完全等於0 (或是一個常數), 他的autocorrelation結果也會是常數0
            # 這邊需要想辦法排除掉這種旋轉軸的資料
            # 這種轉軸會找不到local max index, 給予None. 
            # 並且在接下來球frequency的計算當中, 不加入計算
            if not np.any(_quat):
                handAutoCorr[aJointIdx][k] = None
                handAutoCorrLocalMaxInd[aJointIdx][k] = None
                continue
            handAutoCorr[aJointIdx][k] = autoCorrelation(_quat, False)
            localMaxIdx, = findLocalMaximaIdx(handAutoCorr[aJointIdx][k])
            localMaxIdx = [i for i in localMaxIdx if handAutoCorr[aJointIdx][k][i]>0]
            handAutoCorrLocalMaxInd[aJointIdx][k] = localMaxIdx[0]
    allFrequency = \
        [handAutoCorrLocalMaxInd[i][k] for i, aJointAxis in enumerate(quatIndex) for k in aJointAxis]
    allFrequency = [i for i in allFrequency if i is not None]
    handRepeatingCycle = int(sum(allFrequency) / len(allFrequency))
    print('hand frequency: ', handRepeatingCycle)
    
    # 7. (hand) find best curve of repeating pattern  
    handReapeatingPattern = {i: {j: None for j in quatIndex[i]} for i in range(len(handJointsRotations))}
    for aJointIdx in range(len(handJointsRotations)):
        for k, _quat in quatGaussianRots[aJointIdx].items():
            _rotSegs = splitRotation(_quat, handRepeatingCycle, True)
            highestCorrIdx, highestCorrs = correlationBtwMultipleSeq(_rotSegs, 3)
            highestCorrRots = [_rotSegs[i] for i in highestCorrIdx]
            # rotation全部都是0的curve, correlation會全部都是0, 給予none就是普通average
            _weight = highestCorrs/sum(highestCorrs) if sum(highestCorrs) != 0 else None
            avgHighCorrPattern = averageMultipleSeqs(highestCorrRots, _weight)
            handReapeatingPattern[aJointIdx][k] = avgHighCorrPattern
    return handReapeatingPattern

# 建立四元數版本的B-Spline mapping function 
def constructBSplineMapFunc(handRotationFilePath, bodyRotationFilePath, outputFilePath):
    '''
    1. (hand) read hand rotation, preproc hand rotation and get repeating pattern in quat
    ======= ======= ======= 
    2. (body) read body rotation 
    3. (body) 不要前幾個時間點的訊號 
    4. (body) adjust range to [-180, 180] 
    5. (body) convert to quat
    6. (body) apply gaussian to quat (並且偵測都是0的旋轉軸, 不要做之後的計算)
    7. (body) autocorrelation for frequency finding 
    8. (body) crop cycle length (frequency) segment
    ======= ======= ======= 
    9. (mixed) find min max and crop into inc and dec segments
    10. (mixed) fit B-Spline then sample some points
    11. (mixed) average inc and dec segments which belongs to the same joint, axis
    12. (mixed) 利用這些sample points組合成的function再做一次B-Spline fitting 
                and sample points from it 
    13. (mixed) hand and body最大最小值要轉換回原始範圍
    ======= ======= ======= 
    n. store result 
    '''

    # 1. (hand) read hand rotation, preproc hand rotation and get repeating pattern in quat
    handRepeatingPattern = handRotPreproc(handRotationFilePath)
    # ======= ======= ======= 
    # 2. (body) read body rotation 
    bodyJointRotations=None
    with open(bodyRotationFilePath, 'r') as fileOpen: 
        rotationJson=json.load(fileOpen)
        bodyJointRotations = rotationJsonDataParser(rotationJson, jointCount=4)
        bodyJointRotations = [{k: bodyJointRotations[aJointIdx][k] for k in bodyJointRotations[aJointIdx]} for aJointIdx in range(len(bodyJointRotations))]
    
    # 3. (body) 不要前幾個時間點的訊號 
    ## 目前指定不要使用前10個訊號
    for aJointIdx in range(len(bodyJointRotations)):
        for k in bodyJointRotations[aJointIdx]:
            bodyJointRotations[aJointIdx][k] = bodyJointRotations[aJointIdx][k][10:]
    
    # 4. (body) adjust range to [-180, 180] 
    originBodyRot = copy.deepcopy(bodyJointRotations)
    for aJointIdx in range(len(bodyJointRotations)):
        for k in bodyJointRotations[aJointIdx]:
            bodyJointRotations[aJointIdx][k] = adjustRotationDataTo180(bodyJointRotations[aJointIdx][k])
    
    # 5. (body) convert to quat
    bodyQuatJointRots = {i: None for i in range(len(bodyJointRotations))}
    # Body joint要注意y軸以及部分z軸要清成0
    for _jointInd in range(len(unusedJointAxisIdx)):
        for _axis in unusedJointAxisIdx[_jointInd]:
          bodyJointRotations[_jointInd][_axis] = [0 for i in bodyJointRotations[_jointInd][_axis]] 
    for aJointIdx in range(len(bodyJointRotations)):
        _eularStream = [
            list(i) for i in zip(*[bodyJointRotations[aJointIdx][k] for k in ['x', 'y', 'z']])
        ]
        _quatStream = eularToQuat(_eularStream)
        # print(_quatStream)
        _quatX = _quatStream.T[0]
        _quatY = _quatStream.T[1]
        _quatZ = _quatStream.T[2]
        _quatW = _quatStream.T[3]
        bodyQuatJointRots[aJointIdx] = {
            'x': _quatX,
            'y': _quatY, 
            'z': _quatZ,
            'w': _quatW
        }
    # 6. (body) apply gaussian to quat, also normalize to original min max
    bodyQuatGaussian = copy.deepcopy(bodyQuatJointRots)
    zeroJointAxes = [[] for i in range(len(quatIndex))]
    for _jointInd in range(len(bodyQuatJointRots)):
        for _axis in bodyQuatJointRots[_jointInd]:
            bodyQuatGaussian[_jointInd][_axis] = gaussian_filter1d(
                bodyQuatJointRots[_jointInd][_axis], sigma=2
            )
            _min = np.min(bodyQuatJointRots[_jointInd][_axis])
            _max = np.max(bodyQuatJointRots[_jointInd][_axis])
            ## 如果min and max == 0 維持原樣就好, 不用做normalization 
            ## 這些旋轉軸要記錄下來, 之後都不用做處理
            if _min == _max and _min == 0:
                zeroJointAxes[_jointInd].append(_axis)
                continue
            bodyQuatGaussian[_jointInd][_axis] = minMaxNormalization(bodyQuatGaussian[_jointInd][_axis], _min, _max)

    # 7. (body) autocorrelation for frequency finding 
    ## 找所有軸的autocorrelation
    bodyAutoCorr = {_jointInd: {_axis: None for _axis in quatIndex[_jointInd]} for _jointInd in range(len(quatIndex))}
    bodyJointFreq = {_jointInd: {_axis: None for _axis in quatIndex[_jointInd]} for _jointInd in range(len(quatIndex))}
    for _jointInd in range(len(bodyQuatGaussian)):
        for _axis in bodyQuatGaussian[_jointInd]:
            ## 如果全部數值都是0的就不用求autocorrelation, 維持原本的None 
            if not np.any(bodyQuatGaussian[_jointInd][_axis]):
                continue
            _autoCorr = autoCorrelation(bodyQuatGaussian[_jointInd][_axis], False)
            bodyLocalMaxIdx, = findLocalMaximaIdx(_autoCorr)
            bodyLocalMaxIdx = [i for i in bodyLocalMaxIdx if _autoCorr[i]>0]
            bodyAutoCorr[_jointInd][_axis] = _autoCorr
            bodyJointFreq[_jointInd][_axis] = bodyLocalMaxIdx
    ## 取所有frequency的中位數 (所有四元數的frequency concate在一起)
    bodyFullFreq = []
    for _jointInd in range(len(bodyQuatGaussian)):
        for _axis in bodyQuatGaussian[_jointInd]:
            if bodyJointFreq[_jointInd][_axis] is not None:
                _freq = np.diff(bodyJointFreq[_jointInd][_axis])[:2]
                _freq = np.append(_freq, bodyJointFreq[_jointInd][_axis][0]).tolist()
                bodyFullFreq.extend(_freq)
    bodyFreq = np.median(bodyFullFreq).astype(int)  # For front kick
    from scipy.stats import mode
    bodyFreq = mode(bodyFullFreq).mode[0].astype(int)  # For side kick
    print('body frequency: ', bodyFreq)
    print(bodyFullFreq)

    # 8. (body) crop cycle length (frequency) segment
    startCropInd = 5
    bodyJointsPatterns = [{k: [] for k in axis} for axis in quatIndex]
    for _jointInd in range(len(quatIndex)):
        for _axis in quatIndex[_jointInd]:
            bodyJointsPatterns[_jointInd][_axis] = \
                bodyQuatGaussian[_jointInd][_axis][startCropInd: startCropInd + bodyFreq]

    # ======= ======= ======= 
    # 9. (mixed) find min max and crop into inc and dec segments
    bodyDecreaseSegs = [{k: [] for k in axis} for axis in quatIndex]
    bodyIncreaseSegs = [{k: [] for k in axis} for axis in quatIndex]
    handDecreaseSegs = [{k: [] for k in axis} for axis in quatIndex]
    handIncreaseSegs = [{k: [] for k in axis} for axis in quatIndex]
    for _jointInd in range(len(quatIndex)):
        for _axis in quatIndex[_jointInd]:
            ## 都是常數0的旋轉軸不做處理, 給None
            if not np.any(bodyJointsPatterns[_jointInd][_axis]):
                bodyDecreaseSegs[_jointInd][_axis] = None
                bodyIncreaseSegs[_jointInd][_axis] = None
                handDecreaseSegs[_jointInd][_axis] = None
                handIncreaseSegs[_jointInd][_axis] = None
                continue
            
            if isinstance(bodyJointsPatterns[_jointInd][_axis], np.ndarray):
                bodyJointsPatterns[_jointInd][_axis] = \
                    bodyJointsPatterns[_jointInd][_axis].tolist()
            bodyJointCurve = np.array(bodyJointsPatterns[_jointInd][_axis]*3)
            handJointCurve = np.array(handRepeatingPattern[_jointInd][_axis]*3)
            bodyGlobalMax, bodyGlobalMaxIdx = findGlobalMaxAndIdx(bodyJointCurve)
            bodyGlobalMin, bodyGlobalMinIdx = findGlobalMinAndIdx(bodyJointCurve)

            handGlobalMax, handGlobalMaxIdx = findGlobalMaxAndIdx(handJointCurve)
            handGlobalMin, handGlobalMinIdx = findGlobalMinAndIdx(handJointCurve)
            
            bodyDecreaseSegs[_jointInd][_axis], bodyIncreaseSegs[_jointInd][_axis] = \
                cropIncreaseDecreaseSegments(bodyJointCurve, bodyGlobalMaxIdx, bodyGlobalMinIdx)
            handDecreaseSegs[_jointInd][_axis], handIncreaseSegs[_jointInd][_axis] = \
                cropIncreaseDecreaseSegments(handJointCurve, handGlobalMaxIdx, handGlobalMinIdx)
            pass
    bodySegs = [
        bodyDecreaseSegs, bodyIncreaseSegs
    ]
    handSegs = [
        handDecreaseSegs, handIncreaseSegs
    ]
    # 10. fit B-Spline then sample some points
    numberOfSamplePt = 1000
    bodySplines = [
        [{k: [] for k in axis} for axis in quatIndex], [{k: [] for k in axis} for axis in quatIndex]
    ]
    handSplines = [
        [{k: [] for k in axis} for axis in quatIndex], [{k: [] for k in axis} for axis in quatIndex]
    ]
    handSamplePointsArrs = [
        [{k: [] for k in axis} for axis in quatIndex], [{k: [] for k in axis} for axis in quatIndex]
    ]
    bodySamplePointsArrs = [
        [{k: [] for k in axis} for axis in quatIndex], [{k: [] for k in axis} for axis in quatIndex]
    ]
    for _jointInd in range(len(quatIndex)):
        for _axis in quatIndex[_jointInd]:
            for inc_dec in range(2):
                # 那些是0的旋轉軸做B-Spline fitting會出錯
                # 他們的segment沒有取出任何資料是None. 因為他們沒有最大最小值
                # 這些旋轉軸不做B-Spline fitting 
                if bodySegs[inc_dec][_jointInd][_axis] is None:
                    bodySplines[inc_dec][_jointInd][_axis]=None
                    handSplines[inc_dec][_jointInd][_axis]=None
                    bodySamplePointsArrs[inc_dec][_jointInd][_axis]=None
                    handSamplePointsArrs[inc_dec][_jointInd][_axis]=None
                    continue
                bodySplines[inc_dec][_jointInd][_axis] = bSplineFitting(bodySegs[inc_dec][_jointInd][_axis], isDrawResult=False)
                handSplines[inc_dec][_jointInd][_axis] = bSplineFitting(handSegs[inc_dec][_jointInd][_axis], isDrawResult=False)
                bodySamplePointsArrs[inc_dec][_jointInd][_axis] = splev(np.linspace(0, len(bodySegs[inc_dec][_jointInd][_axis])-1, numberOfSamplePt), bodySplines[inc_dec][_jointInd][_axis])
                handSamplePointsArrs[inc_dec][_jointInd][_axis] = splev(np.linspace(0, len(handSegs[inc_dec][_jointInd][_axis])-1, numberOfSamplePt), handSplines[inc_dec][_jointInd][_axis]) 
                pass
    

    # 11. (mixed) average inc and dec segments which belongs to the same joint, axis
    ## 注意, decrease的時間先後順序要相反過來, 不然大小變化會與increase相反 (其中一個相反即可)
    handAvgSamplePts = [{k: [] for k in axis} for axis in quatIndex]
    bodyAvgSamplePts = [{k: [] for k in axis} for axis in quatIndex] 
    for aJointIdx in range(len(quatIndex)):
        for k in quatIndex[aJointIdx]:
            ## 都是0的旋轉軸不用取平均, 給None
            if handSamplePointsArrs[0][aJointIdx][k] is None:
                handAvgSamplePts[aJointIdx][k]=None
                bodyAvgSamplePts[aJointIdx][k]=None
                continue
            
            handAvgSamplePts[aJointIdx][k] = \
                (handSamplePointsArrs[0][aJointIdx][k][::-1] + handSamplePointsArrs[1][aJointIdx][k]) / 2
            bodyAvgSamplePts[aJointIdx][k] = \
                (bodySamplePointsArrs[0][aJointIdx][k][::-1] + bodySamplePointsArrs[1][aJointIdx][k]) / 2

    # 12. (mixed) 利用這些sample points組合成的function再做一次B-Spline fitting 
    #            and sample points from it
    handMapSamplePts = [{k: [] for k in axis} for axis in quatIndex]
    bodyMapSamplePts = [{k: [] for k in axis} for axis in quatIndex]
    for aJointIdx in range(len(quatIndex)):
        for k in quatIndex[aJointIdx]:
            ## 都是0的旋轉軸會是None, 不用重新fit B-Spline
            if handAvgSamplePts[aJointIdx][k] is None:
                handMapSamplePts[aJointIdx][k]=None
                bodyMapSamplePts[aJointIdx][k]=None
                continue
            
            ## Fit B-Spline最重要的兩個議題要注意
            ## 1. x要由小到大
            ## 2. x不能有重複的數值
            ## 這邊如果只排序x, y軸的資料就會亂掉. 所以, y軸也由小到大排序
            ## check if hand sample points have duplicates (no duplicates)
            # _duplicates, _counts = np.unique(handAvgSamplePts[aJointIdx][k], return_counts=True)
            ## sample points need to be sorted
            _sortedInd = np.argsort(handAvgSamplePts[aJointIdx][k])
            handAvgSamplePts[aJointIdx][k] = handAvgSamplePts[aJointIdx][k][_sortedInd]
            _sortedInd = np.argsort(bodyAvgSamplePts[aJointIdx][k])
            bodyAvgSamplePts[aJointIdx][k] = bodyAvgSamplePts[aJointIdx][k][_sortedInd]
            # print(handAvgSamplePts[aJointIdx][k])
            ## fit BSpline
            # _bspline = bSplineFitting(
            #     bodyAvgSamplePts[aJointIdx][k], handAvgSamplePts[aJointIdx][k],
            #     False
            # )
            smoothFactor = numberOfSamplePt*0.0001
            _bspline = splrep(
                handAvgSamplePts[aJointIdx][k], bodyAvgSamplePts[aJointIdx][k],
                s=smoothFactor
            )
            ## sample points
            handMapSamplePts[aJointIdx][k] = np.linspace(
                handAvgSamplePts[aJointIdx][k][0], 
                handAvgSamplePts[aJointIdx][k][-1], 
                numberOfSamplePt
            )
            bodyMapSamplePts[aJointIdx][k] = splev(
                handMapSamplePts[aJointIdx][k], 
                _bspline
            ) 

    # 13. (mixed) hand and body的B-Spline sample points的最大最小值要轉換回原始範圍
    # 使用normalization
    handNormMapSamplePts = [{k: [] for k in axis} for axis in quatIndex]
    bodyNormMapSamplePts = [{k: [] for k in axis} for axis in quatIndex]
    for aJointIdx in range(len(quatIndex)):
        for k in quatIndex[aJointIdx]:
            ## 不要處理None的資料
            if handMapSamplePts[aJointIdx][k] is None:
                handNormMapSamplePts[aJointIdx][k]=None
                bodyNormMapSamplePts[aJointIdx][k]=None
                continue
            handNormMapSamplePts[aJointIdx][k] = minMaxNormalization(
                handMapSamplePts[aJointIdx][k],
                np.min(handRepeatingPattern[aJointIdx][k]),
                np.max(handRepeatingPattern[aJointIdx][k])
            )
            bodyNormMapSamplePts[aJointIdx][k] = minMaxNormalization(
                bodyMapSamplePts[aJointIdx][k],
                np.min(bodyQuatGaussian[aJointIdx][k]),
                np.max(bodyQuatGaussian[aJointIdx][k])
            )


    # n. store result 
    def _outputData(data, fileNm):
        with open(os.path.join(outputFilePath, fileNm+'.pickle'), 'wb') as WFile:
            pickle.dump(data, WFile)
    _outputData(bodyQuatGaussian, 'bodyQuatGaussian')
    _outputData(bodyAutoCorr, 'bodyAutoCorr')
    _outputData(bodyJointFreq, 'bodyJointFreq')
    _outputData(bodySegs, 'bodySegs')
    _outputData(handSegs, 'handSegs')
    _outputData(bodySamplePointsArrs, 'bodySamplePointsArrs')
    _outputData(handSamplePointsArrs, 'handSamplePointsArrs')
    _outputData(handAvgSamplePts, 'handAvgSamplePts')
    _outputData(bodyAvgSamplePts, 'bodyAvgSamplePts')
    _outputData(handMapSamplePts, 'handMapSamplePts')
    _outputData(bodyMapSamplePts, 'bodyMapSamplePts')
    _outputData(handNormMapSamplePts, 'handNormMapSamplePts')
    _outputData(bodyNormMapSamplePts, 'bodyNormMapSamplePts')
    pass

## apply兩種mapping function到hand rotation
def applyMapFuncToRot(
    handRotationFilePath, linearMapFuncFilePath, 
    BSplineHandSPFilePath, BSplineBodySPFilePath,
    outputFilePath, linearMappedResultFileNm, BSplineMappedResultFileNm
):
    '''
    Object:
        套用兩種mapping function到hand rotation. 
        一律使用沒有預處理的hand rotation
    1. read hand rotation (quaternion, without preprocessing)
    1.1 adjust to [-180, 180] 
    2. read mapping function (2 types of mapping function)
    3. apply mapping function 
    4. output mapping result 
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
        handJointsRotations = rotationJsonDataParser({'results': rotationJson}, jointCount=4)    # For python output
    ## adjust to [-180, 180]
    for aJointIdx in range(len(handJointsRotations)):
        for aAxis in handJointsRotations[aJointIdx]:
            handJointsRotations[aJointIdx][aAxis] = \
                adjustRotationDataTo180(handJointsRotations[aJointIdx][aAxis])
    ## 先將不使用的轉軸資料清0
    for aJointIdx in range(len(unusedJointAxisIdx)): 
        for k in unusedJointAxisIdx[aJointIdx]:
            handJointsRotations[aJointIdx][k] = [0 for i in handJointsRotations[aJointIdx][k]]
    ## convert to quaternion 
    for aJointIdx in range(len(handJointsRotations)): 
        _eularStream = [
            list(i) for i in zip(*[handJointsRotations[aJointIdx][k] for k in ['x', 'y', 'z']])
        ]
        _quatStream = eularToQuat(_eularStream)
        # print(_quatStream)
        _quatX = _quatStream.T[0]
        _quatY = _quatStream.T[1]
        _quatZ = _quatStream.T[2]
        _quatW = _quatStream.T[3]
        handJointsRotations[aJointIdx] = {
            'x': _quatX,
            'y': _quatY, 
            'z': _quatZ,
            'w': _quatW
        }

    # 2. 
    ## linear mapping function 
    linearMapFunc = None
    with open(linearMapFuncFilePath, 'rb') as RFile:
        linearMapFunc = pickle.load(RFile)
    ## 修正沒有使用的linear mapping function 
    for _jointInd in range(len(linearMapFunc)):
        for _axis in linearMapFunc[_jointInd]:
            if linearMapFunc[_jointInd][_axis] is None: 
                linearMapFunc[_jointInd][_axis] = np.array([0, 0])
    ## B-Spline fitting mapping function 
    BSplineHandSP = None
    BSplineBodySP = None
    with open(BSplineHandSPFilePath, 'rb') as RFile:
        BSplineHandSP = pickle.load(RFile)
    with open(BSplineBodySPFilePath, 'rb') as RFile:
        BSplineBodySP = pickle.load(RFile)
    ## 修正沒有使用的sample points 
    for _jointInd in range(len(BSplineHandSP)):
        for _axis in BSplineHandSP[_jointInd]:
            if BSplineHandSP[_jointInd][_axis] is None: 
                BSplineHandSP[_jointInd][_axis] = np.array([0, 0, 0])
                BSplineBodySP[_jointInd][_axis] = np.array([0, 0, 0])
            else:   
                BSplineHandSP[_jointInd][_axis] = np.array(BSplineHandSP[_jointInd][_axis])
                BSplineBodySP[_jointInd][_axis] = np.array(BSplineBodySP[_jointInd][_axis])
    
    # 3. 
    ## linear mapping function
    linearMappedRot = applyLinearMapFunc(linearMapFunc, handJointsRotations, quatIndex)
    ## B-Spline mapping function 
    BSMappedRot = applyBSplineMapFunc(BSplineHandSP, BSplineBodySP, handJointsRotations, quatIndex)

    # 4. output
    def _outputData(data, fileNm):
        with open(os.path.join(outputFilePath, fileNm+'.pickle'), 'wb') as WFile:
            pickle.dump(data, WFile)

    _outputData(linearMappedRot, 'linearMappedRot')
    _outputData(BSMappedRot, 'BSMappedRot')

    ## 4.1 output mapping result in json format 
    ##      輸出格式與handRotaionAfterMapping/leftFrontKickStreamLinearMapping/內的資料相同. 
    ##      但是, 儲存的內容是quaternion資料. 
    timeCount = len(linearMappedRot[0]['x'])
    jointCount = len(quatIndex)
    print('timeCount: ', timeCount)
    ## 輸出格式要轉換成list, 因為json輸出不支援np.array型別
    for i in range(jointCount):
        for k in quatIndex[i]:
            linearMappedRot[i][k] = linearMappedRot[i][k].tolist()
            BSMappedRot[i][k] = BSMappedRot[i][k].tolist()
    linearMappedJson = [
        {
            'time':t, 'data': [
                {k: linearMappedRot[i][k][t] for k in quatIndex[i]} for i in range(jointCount)
            ]
        } for t in range(timeCount)
    ]
    BSMappedJson = [
        {
            'time':t, 'data': [
                {k: BSMappedRot[i][k][t] for k in quatIndex[i]} for i in range(jointCount)
            ]
        } for t in range(timeCount)
    ]
    with open(outputFilePath+linearMappedResultFileNm, 'w') as WFile: 
        json.dump(linearMappedJson, WFile) 
    with open(outputFilePath+BSplineMappedResultFileNm, 'w') as WFile: 
        json.dump(BSMappedJson, WFile) 
    


if __name__=='__main01__':
    ## construct quaternion linear mapping function 
    constructLinearMappingOnQuat(
        handRotationFilePath = './HandRotationOuputFromHomePC/leftSideKickStream.json',
        bodyRotationFilePath = './bodyDBRotation/genericAvatar/leftSideKick0.03_withHip.json',
        outputFilePath = 'rotationMappingQuaternionData/leftSideKick/'
    )
    ## construct quaternion B-Spline mapping function 
    constructBSplineMapFunc(
        handRotationFilePath = './HandRotationOuputFromHomePC/leftSideKickStream.json', 
        bodyRotationFilePath = './bodyDBRotation/genericAvatar/leftSideKick0.03_withHip.json', 
        outputFilePath = 'rotationMappingQuaternionData/leftSideKickBSpline/'
    )
    ## apply mapping function to hand rotation
    applyMapFuncToRot(
        handRotationFilePath='./HandRotationOuputFromHomePC/leftSideKickStream.json', 
        linearMapFuncFilePath='rotationMappingQuaternionData/leftSideKick/mappingFuncs.pickle', 
        BSplineHandSPFilePath='rotationMappingQuaternionData/leftSideKickBSpline/handNormMapSamplePts.pickle', 
        BSplineBodySPFilePath='rotationMappingQuaternionData/leftSideKickBSpline/bodyNormMapSamplePts.pickle',
        outputFilePath='rotationMappingQuaternionData/leftSideKick/', 
        linearMappedResultFileNm = 'leftSideKick_quat_BSpline_FTTTFT.json', 
        BSplineMappedResultFileNm = 'leftSideKick_quat_linear_FTTTFT.json'
    )
    pass