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
import json 
import time 
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

def applyQuatDirectMapFunc(handRef, bodyRef, handInput, usedJointAxis):
    '''
    參考: rotationAnalysisNew.py -> applyBSplineMapFunc() 
    ### Input 
        - handRef: 2 dimension array. number of joints * sample points
        - bodyRef: 3 dimension array. number of joints * sample points * 4 (dimension of quaternion)
        - handInput: list of dictionary. each dict represent a joint's axes' rotation. rotations of XYZ in dict. 
        - usedJointAxis: dict. each element represent the axis used for searching for the corresponding joint (same as the index in that list). 
            only single axis will used for searching in a joint. (一個joint只會使用單一個axis做搜尋)
    ### Output
        - afterMapping:  list of dict. each dict represent a joint's quaternion, includes XYZW. 
    '''
    afterMapping = [{k: [] for k in ['x', 'y', 'z', 'w']} for i in range(len(usedJointAxis))]
    for _jointInd in range(len(usedJointAxis)): 
        _usedAxis = usedJointAxis[_jointInd]
        _dist = np.abs(handRef[_jointInd, :] - handInput[_jointInd][_usedAxis])
        _minInd = np.argmin(_dist)
        afterMapping[_jointInd] = {
            'x': bodyRef[_jointInd][_minInd][0], 
            'y': bodyRef[_jointInd][_minInd][1], 
            'z': bodyRef[_jointInd][_minInd][2], 
            'w': bodyRef[_jointInd][_minInd][3]
        }
    return afterMapping

def main():
    pass

if __name__=='__main__':
    main()

    DBRotFilePath = 'bodyDBRotation/genericAvatar/quaternion/jumpJoy0.03_075_withHip.json'
    DBRefSeqOutputFilePath = 'rotationMappingQuatDirectMappingData/jumpJoy_body_ref.npy'
    handPerfFilePath = 'HandRotationOuputFromHomePC/jumpJoyStream.json'
    handPerfRefSeqOutputFilePath = 'rotationMappingQuatDirectMappingData/jumpJoy_hand_ref.npy'
    MappedRotSaveDataPath = 'handRotaionAfterMapping/jumpJoy_quat_directMapping.json'
    cropInterval = {0: [87, 104], 1: [94, 103], 2: [87, 104], 3: [94, 103]}
    handPerfCropInterval = {0: [725, 904], 1: [800, 103], 2: [725, 904], 3: [800, 103]} # 最大值與最小值的index 
    handPerfAxisPair = {0: 'x', 1: 'x', 2: 'x', 3: 'x'}
    bodyRefReverse = {0: True, 1: True, 2: True, 3: True}
    # read DB animation rotation in quaternion 
    data = readHandPerformance(DBRotFilePath, isFromUnity=True)
    # read hand performance rotation 
    handPerfData = readHandPerformance(handPerfFilePath)
    
    # 產生interpolated DB animation資料
    DBRefSeq = generateRefSeq(data, cropInterval, 100, outputFilePath=DBRefSeqOutputFilePath) 
    # New: animation的部分joint sequence可能需要反轉 
    for _jointInd, _ifReverse in bodyRefReverse.items():
        if _ifReverse:
            DBRefSeq[_jointInd, :, :] = DBRefSeq[_jointInd, ::-1, :]
    ## 重新存檔一次, 因為之前存的是沒有反轉的結果
    with open(DBRefSeqOutputFilePath, 'wb') as OutFile:
        np.save(OutFile, DBRefSeq)
    # 產生interpolated hand performance資料. 每個joint有獨立的interval (min, max index)
    handPerfRefSeq = genHandPerfRefSeq(handPerfData, handPerfCropInterval, handPerfAxisPair, 100, handPerfRefSeqOutputFilePath)
    print('hand reference sequence shape: ', handPerfRefSeq.shape)

    # 實際mapping一次, 使用原始的hand performance旋轉數值 
    handJointsRotations = None
    with open(handPerfFilePath, 'r') as fileOpen: 
        handJointsRotations=json.load(fileOpen)
    print(handJointsRotations[0]['data'])
    
    timeCount = len(handJointsRotations)
    mapRet = [None for i in range(timeCount)]
    rotMapTimeLaps = np.zeros(timeCount)
    for t in range(timeCount):
        mapRet[t] = applyQuatDirectMapFunc(
            handPerfRefSeq,
            DBRefSeq,
            handJointsRotations[t]['data'],
            handPerfAxisPair
        )
        rotMapTimeLaps[t] = time.time()
    rotMapCost = rotMapTimeLaps[1:] - rotMapTimeLaps[:-1]
    print('rotation map avg time: ', np.mean(rotMapCost))
    print('rotation map time std: ', np.std(rotMapCost))
    print('rotation map max time cost: ', np.max(rotMapCost))
    print('rotation map min time cost: ', np.min(rotMapCost))

    # Store mapping result 
    mapResultJson = [{'time': t, 'data': mapRet[t]} for t in range(timeCount)]
    with open(MappedRotSaveDataPath, 'w') as WFile: 
        json.dump(mapResultJson, WFile) 
    pass