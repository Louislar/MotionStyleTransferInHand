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
from rotationAnalysis import rotationJsonDataParser, \
    adjustRotationDataTo180, butterworthLowPassFilter, gaussianFilter, \
    minMaxNormalization, autoCorrelation, findLocalMaximaIdx

quatIndex = [['x','y','z','w'], ['x','y','z','w'], ['x','y','z','w'], ['x','y','z','w']]

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

# 建立四元數版本的mapping function 
def constructLinearMappingOnQuat(handRotationFilePath, bodyRotationFilePath, outputFilePath):
    '''
    1. (hand) read hand rotation
    2. (hand) adjust range to [-180, 180] 
    3. (hand) low pass and average filter
    4. (hand) convert to quaternion
    5. (hand) apply gaussian filter to quaternion 
    --> 注意, 有過gaussian數值就要調整回原本的範圍. (包含上面在eular做gaussian也要調整回來)
    6. (hand) autocorrelation for repeating pattern finding 
    7. (hand) find best repeating pattern
    8. (hand) extract min and max (以上做這些只是了避免找到雜訊與outlier)
    x. (body) read body rotation 
    x. (body) convert to quaternion
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
    # 4. 
    quatJointRots = {i: None for i in range(len(handJointsRotations))}
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
            handAutoCorr[aJointIdx][k] = autoCorrelation(_quat, True)
            plt.show()
            localMaxIdx, = findLocalMaximaIdx(handAutoCorr[aJointIdx][k])
            localMaxIdx = [i for i in localMaxIdx if handAutoCorr[aJointIdx][k][i]>0]
            handAutoCorrLocalMaxInd[aJointIdx][k] = localMaxIdx[0]
            # TODO: 這邊會出錯. 因為有的curve是完全等於0, 他的autocorrelation結果也會是常數0
            # 這邊需要想辦法排除掉這種旋轉軸的資料
            
    
    # 7. (hand) find best repeating pattern
    # TODO: 
    # 8. (hand) extract min and max from repeating patterns 
    # TODO: 
    

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
    

if __name__=='__main__':
    constructLinearMappingOnQuat(
        handRotationFilePath = './HandRotationOuputFromHomePC/leftFrontKickStream.json',
        bodyRotationFilePath = './bodyDBRotation/genericAvatar/leftFrontKick0.03_withHip.json',
        outputFilePath = 'rotationMappingQuaternionData/leftFrontKick/'
    )
    pass