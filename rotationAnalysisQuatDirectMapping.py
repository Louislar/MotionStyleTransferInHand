'''
建立quaternion rotation的reference sequence 
手部的旋轉也是建立reference sequence
兩者有相同的sample points, 並且會是one to one mapping 
兩者都需要從原始訊號當中手動標記 

testing時, 搜尋手部最相近的旋轉, 直接map到對應的quaternion 
'''

import numpy as np 
import pandas as pd 
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import copy 
from newMethod.util import readHandPerformance, cropHandPerfJointWise, \
    cropHandPerformance, handPerformanceToMatrix 

def generateRefSeq(rot, interval, numSamplePts, outputFilePath=None): 
    '''
    generate reference sequence, 可用於hand performace以及DB animation的資料 
    crop指定區間的quaternion資料, 並且使用slerp interpolate指定的sample points數量
    若有給定輸出位置, 則輸出成json檔案 
    :interval: 每一個joint都會有獨立指定要crop的區間 
    Output
    :interpData: array of array. 每個array代表一個joint. 
        儲存quaternion的array. dimension是numSamplePts * 4
    '''
    # Crop data by given intervals 
    cropData = cropHandPerfJointWise(rot, interval) 
    numOfJoint = len(cropData)
    axisNames = list(cropData[0].keys())
    print('number of joints: ', numOfJoint)
    print('axis names: ', axisNames)

    # interpolation via slerp to quaternion 
    interpData = np.zeros((numOfJoint, numSamplePts, len(axisNames))) 
    for _jointInd in range(numOfJoint):
        keyTimes = list(range(len(cropData[_jointInd]['x'])))
        keyRots = np.array([cropData[_jointInd][_axis] for _axis in axisNames]).T
        keyRots = R.from_quat(keyRots)
        _slerp = Slerp(keyTimes, keyRots)
        newTimes = np.linspace(keyTimes[0], keyTimes[-1], numSamplePts)
        _interpRet = _slerp(newTimes).as_quat()
        interpData[_jointInd, :, :] = _interpRet

        # print(keyRots.as_quat())
        # print(keyRots.as_quat().shape)
        # print(_interpRet)
        # print(_interpRet.shape)
    # Output to npy
    if outputFilePath is not None: 
        with open(outputFilePath, 'wb') as OutFile:
            np.save(OutFile, interpData)
    return interpData

def genHandPerfRefSeq(rot, intervalInd, axisPair, numSamplePts, outputFilePath=None):
    '''
    只給最大最小值的index, 根據給定數值產生sample points
    Output 
    :: array of array (2 dimensional array). 每個array代表一個joint axis. 
        array的dimension只有一個是numSamplePts
    '''
    numOfJoint = len(rot)
    interpData = np.zeros((numOfJoint, numSamplePts)) 
    for _jointInd in range(numOfJoint):
        _axis = axisPair[_jointInd]
        _interpRet = np.linspace(
            rot[_jointInd][_axis][intervalInd[_jointInd][0]], 
            rot[_jointInd][_axis][intervalInd[_jointInd][1]], 
            numSamplePts
        )
        interpData[_jointInd, :] = _interpRet
    if outputFilePath is not None: 
        with open(outputFilePath, 'wb') as OutFile:
            np.save(OutFile, interpData)
    return interpData

def main():
    pass

if __name__=='__main__':
    main()

    DBRotFilePath = 'bodyDBRotation/genericAvatar/quaternion/leftFrontKick0.03_075_withHip.json'
    DBRefSeqOutputFilePath = 'rotationMappingQuatDirectMappingData/leftFrontKick_body_ref.npy'
    handPerfFilePath = 'HandRotationOuputFromHomePC/leftFrontKickStream.json'
    handPerfRefSeqOutputFilePath = 'rotationMappingQuatDirectMappingData/leftFrontKick_hand_ref.npy'
    cropInterval = {0: [101, 124], 1: [121, 133], 2: [101, 124], 3: [121, 133]}
    handPerfCropInterval = {0: [1392, 1363], 1: [1145, 1161], 2: [1392, 1363], 3: [1145, 1161]} # 最大值與最小值的index 
    handPerfAxisPair = {0: 'x', 1: 'x', 2: 'x', 3: 'x'}
    # read DB animation rotation in quaternion 
    data = readHandPerformance(DBRotFilePath, isFromUnity=True)
    # read hand performance rotation 
    handPerfData = readHandPerformance(handPerfFilePath)
    
    # 產生interpolated DB animation資料
    DBRefSeq = generateRefSeq(data, cropInterval, 100, outputFilePath=DBRefSeqOutputFilePath) 
    # 產生interpolated hand performance資料. 每個joint有獨立的interval (min, max index)
    handPerfRefSeq = genHandPerfRefSeq(handPerfData, handPerfCropInterval, handPerfAxisPair, 100, handPerfRefSeqOutputFilePath)
    print(handPerfRefSeq.shape)

    pass