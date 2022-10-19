'''
重現Motion editing with data glove的system
'''

from genericpath import isdir
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import json
import os 
import sys
import copy
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import splev, splrep
sys.path.append("../")
from testingStageViz import jsonToDf
from rotationAnalysis import butterworthLowPassFilter, bSplineFitting

def preproc(srcRot, avgFilterSize, cutOffPt, butterWorthOrder):
    '''
    average filter and low pass filter
    Input:
    :avgFilterSize: average filter window size 
    :cutOffPt: low pass filter cutoff point
    :butterWorthOrder: low pass butter worth filter order
    '''
    srcRot = copy.deepcopy(srcRot)
    srcRot = uniform_filter1d(srcRot, size=avgFilterSize)
    srcRot = butterworthLowPassFilter(
        srcRot, 
        order=butterWorthOrder,
        cutoff=cutOffPt
    )
    return srcRot


def constructMappingFunc():
    # 1.1 read hand rotation data
    # 1.2 read body rotation data
    # 2. Average filter apply to both rotation curves (注意, not gaussian filter)
    # 3. Low pass filter apply to both rotation curves via FFT
    # 4. Compute tangent in both curves and find the time point that tanget(slope; 斜率) is 0
    # 4.1 Decide the frequency of hand and body joint's rotation data
    # sp: 某些joint需要獨立處理, 因為使用的參數與別人不同 
    # 5. Construct multiple mapping functions with discrete sample points
    # 5.0 Use B-Spline fitting to interpolate each joint's segment (increase and decrease segments)
    # 5.1 儲存每個joint的B-Spline interpolation結果即可 
    # 需要mapping的時候只需要從finger rotation尋找最相似的sample point, 
    # 然後根據index找到對應的body rotation sample point
    # 5.1.1 left shoulder
    # 5.1.2 left upper leg
    # 5.1.3 left knee
    # 5.1.4 right shoulder
    # 5.1.5 right shoulder
    # 5.1.6 right shoulder
    # 5.1.7 Index finger 
    # 5.1.8 Middle finger 
    # 6. Store each mapping function in file

    # 1. 
    # 1.1
    handRotDirPath = '../HandRotationOuputFromHomePC/'
    handRot = None
    with open(os.path.join(handRotDirPath, 'leftFrontKickStream.json'), 'r') as RFile:
        handRot = json.load(RFile)
    # 1.2 
    bodyRotDirPath = '../bodyDBRotation/genericAvatar/'
    bodyRot = None
    with open(os.path.join(bodyRotDirPath, 'leftFrontKick0.03_withoutHip.json'), 'r') as RFile:
        bodyRot = json.load(RFile)['results']
    # print(bodyRot[0].keys())
    handRot = jsonToDf(handRot)
    bodyRot = jsonToDf(bodyRot)
    # print(handRot[0])
    # print(bodyRot[0])
    handJointsCount = len(list(handRot.keys()))
    handTimeCount = handRot[0].shape[0]
    bodyJointsCount = len(list(bodyRot.keys()))
    bodyTimeCount = bodyRot[0].shape[0]
    print('hand joints count: ', handJointsCount)
    print('hand time count: ', handTimeCount)
    print('body joints count: ', bodyJointsCount)
    print('body time count: ', bodyTimeCount)

    # 2. 
    ## apply average filter in different size
    ## average filter會不會自動取到整數? (我不想要自動取到整數) (輸入的array含有浮點數即可)
    ## TODO: 這邊必定改變最大最小值. 所以, 如果不做mix max的校正, rotation數值會與原始訊號不符.
    avgFilterSize = 50
    beforeAvgFilterHandRot = copy.deepcopy(handRot)
    beforeAvgFilterBodyRot = copy.deepcopy(bodyRot)
    for _jointInd in range(handJointsCount):
        for _axis in ['x', 'y', 'z']:
            handRot[_jointInd].loc[:, _axis] = \
                uniform_filter1d(handRot[_jointInd].loc[:, _axis], size=avgFilterSize)
            bodyRot[_jointInd].loc[:, _axis] = \
                uniform_filter1d(bodyRot[_jointInd].loc[:, _axis], size=avgFilterSize)
    # print(uniform_filter1d([1.1, 2, 5, 8, 9, 11], size=3))

    ## visualize before average filter and after average filter
    vizAxis = 'x'
    vizJoint = 0
    vizTarget = 'body'    # or 'body' or 'hand'
    vizData = beforeAvgFilterHandRot if vizTarget == 'hand' else beforeAvgFilterBodyRot
    vizData2 = handRot if vizTarget == 'hand' else bodyRot
    vizTimeCount = handTimeCount if vizTarget == 'hand' else bodyTimeCount
    # plt.figure()
    # ax1 = plt.subplot(2, 1, 1)
    # ax2 = plt.subplot(2, 1, 2)
    # ax1.set_title('before avg filter')
    # ax2.set_title('after avg filter')
    # ax1.plot(range(vizTimeCount), vizData[vizJoint][vizAxis])
    # ax2.plot(range(vizTimeCount), vizData2[vizJoint][vizAxis])
    # plt.show()

    # 3. 
    cutOffPt = 0.6
    butterWorthOrder = 15
    beforeLowPassHandRot = copy.deepcopy(handRot)
    beforeLowPassBodyRot = copy.deepcopy(bodyRot)
    for _jointInd in range(handJointsCount):
        for _axis in ['x', 'y', 'z']:
            handRot[_jointInd].loc[:, _axis] = butterworthLowPassFilter(
                handRot[_jointInd].loc[:, _axis], 
                order=butterWorthOrder,
                cutoff=cutOffPt
            )
            bodyRot[_jointInd].loc[:, _axis] = butterworthLowPassFilter(
                bodyRot[_jointInd].loc[:, _axis], 
                order=butterWorthOrder,
                cutoff=cutOffPt
            )
    
    ## visualize before and after low pass filter
    # vizAxis = 'x'
    # vizJoint = 2
    # vizTarget = 'body'    # or 'body' or 'hand'
    # vizData = beforeLowPassHandRot if vizTarget == 'hand' else beforeLowPassBodyRot
    # vizData2 = handRot if vizTarget == 'hand' else bodyRot
    # vizTimeCount = handTimeCount if vizTarget == 'hand' else bodyTimeCount
    # plt.figure()
    # ax1 = plt.subplot(2, 1, 1)
    # ax2 = plt.subplot(2, 1, 2)
    # ax1.set_title('before low pass filter')
    # ax2.set_title('after low pass filter')
    # ax1.plot(range(vizTimeCount), vizData[vizJoint][vizAxis])
    # ax2.plot(range(vizTimeCount), vizData2[vizJoint][vizAxis])
    # plt.show()

    # 4. 
    ## TODO: alternative method. 直接使用尋找local extrema的function
    gradientBound = 0.05    # 小於這個bound就算是tanget=0, 就是端點(local extrema)
    gradientHandRot = copy.deepcopy(handRot)
    gradientBodyRot = copy.deepcopy(bodyRot)
    handLocalExtremaInd = {_jointInd: {_axis: [] for _axis in ['x', 'y', 'z']} for _jointInd in range(handJointsCount)}
    bodyLocalExtremaInd = {_jointInd: {_axis: [] for _axis in ['x', 'y', 'z']} for _jointInd in range(bodyJointsCount)}
    for _jointInd in range(handJointsCount):
        for _axis in ['x', 'y', 'z']:
            gradientHandRot[_jointInd].loc[:, _axis] = np.gradient(gradientHandRot[_jointInd].loc[:, _axis])
            gradientBodyRot[_jointInd].loc[:, _axis] = np.gradient(gradientBodyRot[_jointInd].loc[:, _axis])
            gradientHandRot[_jointInd].loc[:, _axis] = gradientHandRot[_jointInd].loc[:, _axis].abs()
            gradientBodyRot[_jointInd].loc[:, _axis] = gradientBodyRot[_jointInd].loc[:, _axis].abs()
            handLocalExtremaInd[_jointInd][_axis] = \
                gradientHandRot[_jointInd].loc[:, _axis].index[gradientHandRot[_jointInd].loc[:, _axis]<gradientBound]
            bodyLocalExtremaInd[_jointInd][_axis] = \
                gradientBodyRot[_jointInd].loc[:, _axis].index[gradientBodyRot[_jointInd].loc[:, _axis]<gradientBound]
    
    ## visualize time point that tangent (slpoe) equals to 0
    # vizAxis = 'x'
    # vizJoint = 2
    # vizTarget = 'body'    # or 'body' or 'hand'
    # vizData = handRot if vizTarget == 'hand' else bodyRot
    # vizData2 = handLocalExtremaInd if vizTarget == 'hand' else bodyLocalExtremaInd
    # vizTimeCount = handTimeCount if vizTarget == 'hand' else bodyTimeCount
    # plt.figure()
    # ax1 = plt.subplot(1, 1, 1)
    # ax1.set_title('after low pass filter with tangent=0')
    # ax1.plot(range(vizTimeCount), vizData[vizJoint][vizAxis])
    # ax1.plot(
    #     vizData2[vizJoint][vizAxis], 
    #     vizData[vizJoint][vizAxis].iloc[vizData2[vizJoint][vizAxis]],
    #     '.',
    #     label='extrema'
    # )
    # plt.legend()
    # plt.show()

    # 4.1 
    ## 目前只能先用肉眼判斷rotation curve的frequency
    ## 並且挑出上升與下降區段的indices. 
    ## 這些區段在之後的步驟會用來construct mapping function
    ## Hint: 只需要算x軸的旋轉
    ## Hint: 手只需要PIP的rotation.
    ## 目前使用肉眼指定tip and pit的位置/index
    handTipAndPitInd = {_jointInd: [] for _jointInd in range(handJointsCount)}   # 之後用來construct mapping function的區段
    bodyTipAndPitInd = {_jointInd: [] for _jointInd in range(bodyJointsCount)}
    # 排列方式為[[谷, 峰], [峰, 谷]]
    handTipAndPitInd[1] = [[1468, 1499], [1499, 1537]]    # 谷, 峰, 谷. avgFilter:50, lowpass cutout:0.6, gradient bound:0.05
    handTipAndPitInd[3] = [[1053, 1098], [1033, 1053]]    # 峰, 谷, 峰. avgFilter:50, lowpass cutout:0.6, gradient bound:0.05
    
    bodyTipAndPitInd[0] = [[300, 334], [265, 300]]    # 峰, 谷, 峰. avgFilter:50, lowpass cutout:0.6, gradient bound:0.05
    bodyTipAndPitInd[1] = [[125, 152], [152, 194]]    # 谷, 峰, 谷. avgFilter:50, lowpass cutout:0.6, gradient bound:0.05
    bodyTipAndPitInd[2] = [[291, 322], [322, 360]]    # 谷, 峰, 谷. avgFilter:100, lowpass cutout:0.2, gradient bound:0.05
    bodyTipAndPitInd[3] = [[205, 234], [234, 274]]    # 谷, 峰, 谷. avgFilter:50, lowpass cutout:0.6, gradient bound:0.05
    # print(bodyLocalExtremaInd[3]['x'].tolist())

    ## visualize 選擇的tip and pit的位置
    # vizAxis = 'x'
    # vizJoint = 2
    # vizTarget = 'body'    # or 'body' or 'hand'
    # vizData = handRot if vizTarget == 'hand' else bodyRot
    # vizData2 = handTipAndPitInd if vizTarget == 'hand' else bodyTipAndPitInd
    # vizData2 = {k: np.array(v).flatten() for k, v in vizData2.items()}
    # vizTimeCount = handTimeCount if vizTarget == 'hand' else bodyTimeCount
    # plt.figure()
    # ax1 = plt.subplot(1, 1, 1)
    # ax1.set_title('after low pass filter with tangent=0')
    # ax1.plot(range(vizTimeCount), vizData[vizJoint][vizAxis])
    # ax1.plot(
    #     vizData2[vizJoint], 
    #     vizData[vizJoint][vizAxis].iloc[vizData2[vizJoint]],
    #     '.',
    #     label='select tip and pit'
    # )
    # plt.legend()
    # plt.show()

    # sp: body的joint 2需要特殊參數才能找到tip and pit, 所以需要獨立重新做計算
    bodyRot[2].loc[:, 'x'] = preproc(
        beforeAvgFilterBodyRot[2]['x'],
        100,
        0.2,
        15
    )

    # 5. 
    ## Hint: 只會拿x軸的旋轉作mapping function
    ## TODO: 目前收集的body rotation只有lower body, 
    ## 但是這篇文章會使用到left and right shoulder的旋轉.
    # 5.0 所有data先做一次B-Spiline fitting and interpolation, 取相同數量的sample points
    ## 上升與下降區段分開做B-Spline interpolation
    ## B-Spline fitting可以參考rotationAnalysis.py 
    numSamplePts = 1000
    handSeg = {_jointInd: [] for _jointInd in range(handJointsCount)}
    bodySeg = {_jointInd: [] for _jointInd in range(handJointsCount)}
    handBSplineSamplePts = {_jointInd: [] for _jointInd in range(handJointsCount)}
    bodyBSplineSamplePts = {_jointInd: [] for _jointInd in range(handJointsCount)}
    for _jointInd in [1, 3]:
        for j in range(2):  # 0: [谷, 峰], 1: [峰, 谷]
            # print(handRot[_jointInd]['x'])
            _seg = handRot[_jointInd]['x'].iloc[handTipAndPitInd[_jointInd][j][0]:handTipAndPitInd[_jointInd][j][1]+1]
            # print(_seg)
            _bspline = bSplineFitting(_seg)
            _x = np.linspace(0, len(_seg)-1, numSamplePts)
            _sp = splev(_x, _bspline)
            handSeg[_jointInd].append(_seg)
            handBSplineSamplePts[_jointInd].append(_sp)
    for _jointInd in range(bodyJointsCount):
        for j in range(2):  # 0: [谷, 峰], 1: [峰, 谷]
            _seg = bodyRot[_jointInd]['x'].iloc[bodyTipAndPitInd[_jointInd][j][0]:bodyTipAndPitInd[_jointInd][j][1]+1]
            _bspline = bSplineFitting(_seg)
            _x = np.linspace(0, len(_seg)-1, numSamplePts)
            _sp = splev(_x, _bspline)
            bodySeg[_jointInd].append(_seg)
            bodyBSplineSamplePts[_jointInd].append(_sp)


    ## visualize B-Spline interpolation result
    # vizJoint = 3
    # vizTarget = 'body'    # or 'body' or 'hand'
    # vizIncDec = 1   # 0: increase, 1: decrease
    # vizData = handSeg if vizTarget == 'hand' else bodySeg
    # vizData2 = handBSplineSamplePts if vizTarget == 'hand' else bodyBSplineSamplePts
    # plt.figure()
    # ax1 = plt.subplot(1, 1, 1)
    # ax1.set_title('B-Spline interpolation')
    # ax1.plot(
    #     range(len(vizData[vizJoint][vizIncDec])), 
    #     vizData[vizJoint][vizIncDec],
    #     '.',
    #     label='original segment'
    # )
    # ax1.plot(
    #     np.linspace(0, len(vizData[vizJoint][vizIncDec]), len(vizData2[vizJoint][vizIncDec])), 
    #     vizData2[vizJoint][vizIncDec],
    #     '.-',
    #     label='interpolated'
    # )
    # plt.legend()
    # plt.show()

    # 5.1 儲存所有B-Spline interpolates sample point
    ## 使用.npy儲存
    saveDirPath = 'data'
    if not os.path.isdir(saveDirPath):
        os.makedirs(saveDirPath)
    for _jointInd in [1, 3]:
        for j in range(2):  # 0: [谷, 峰], 1: [峰, 谷]
            with open(
                os.path.join(saveDirPath, 'hand_{0}_{1}.npy'.format(_jointInd, j)), 'wb'
            ) as OutFile:
                np.save(OutFile, handBSplineSamplePts[_jointInd][j])
    for _jointInd in range(bodyJointsCount):
        for j in range(2):  # 0: [谷, 峰], 1: [峰, 谷]
            with open(
                os.path.join(saveDirPath, 'body_{0}_{1}.npy'.format(_jointInd, j)), 'wb'
            ) as OutFile:
                np.save(OutFile, bodyBSplineSamplePts[_jointInd][j])

def findNearestNum(srcArr, tarArr):
    '''
    Objective:
        使用srcArr當中的每一個元素, 找到再tarArr當中最相近的數值, 並且return它在tarArr中的index
        假設兩個array的維度都是1
        找最相似 = 相減後取絕對值的最小
    Output:
    :: index of the nearest number
    :: nearest number
    '''
    srcArr = copy.deepcopy(srcArr)
    tarArr = copy.deepcopy(tarArr)
    srcArr = srcArr[:, np.newaxis]
    tarArr = tarArr[np.newaxis, :]
    _absDiff = np.abs(srcArr - tarArr)
    _minInd = np.argmin(_absDiff, axis=1)

    # print(srcArr)
    # print(tarArr)
    # print(_absDiff.shape)
    # print(_minInd.shape)
    return _minInd

# 將建立好的mapping function套用到hand rotation
def main(): 
    # 1. read hand rotation
    # 2. read sample points of mapping function
    # 3. use mapping function map hand rotation
    # 4. store mapping results

    # 1. 
    handRotDirPath = '../HandRotationOuputFromHomePC/'
    handRot = None
    with open(os.path.join(handRotDirPath, 'leftFrontKickStream.json'), 'r') as RFile:
        handRot = json.load(RFile)
    handRot = jsonToDf(handRot)
    # 2. 
    samplePtDirPath = 'data/'
    handJoints = [1, 3]
    bodyJoints = [0, 1, 2, 3]
    handSamplePts = {_jointInd: [] for _jointInd in handJoints}
    bodySamplePts = {_jointInd: [] for _jointInd in bodyJoints}
    for _jointInd in handJoints:
        for j in range(2):
            with open(
                os.path.join(samplePtDirPath, 'hand_{0}_{1}.npy'.format(_jointInd, j)), 'rb'
            ) as RFile:
                handSamplePts[_jointInd].append(np.load(RFile))
            # print(handSamplePts[_jointInd][j].shape)
    for _jointInd in bodyJoints:
        for j in range(2):
            with open(
                os.path.join(samplePtDirPath, 'body_{0}_{1}.npy'.format(_jointInd, j)), 'rb'
            ) as RFile:
                bodySamplePts[_jointInd].append(np.load(RFile))
            # print(bodySamplePts[_jointInd][j].shape)
    
    # 3. 
    ## find index and middle finger PIP rotations' similar index in sample points
    ## TODO: 因為目前完全不知道甚麼時候該套用incremental period mapping function
    ## 以及decremental period mapping function. 所以, 兩者分開來個別產生一個結果
    handIncSimilarInd = {_jointInd: None for _jointInd in handJoints}
    handDecSimilarInd = {_jointInd: None for _jointInd in handJoints}
    for _jointInd in handJoints:
        handIncSimilarInd[_jointInd] = findNearestNum(handRot[1]['x'].values, handSamplePts[1][0]) # 上升區段
        handDecSimilarInd[_jointInd] = findNearestNum(handRot[1]['x'].values, handSamplePts[1][1]) # 上升區段
    
    ## use similar indices to find corresponding body rotation in every joint
    bodyIncMappingResult = {_jointInd: [] for _jointInd in bodyJoints}
    bodyDecMappingResult = {_jointInd: [] for _jointInd in bodyJoints}
    ## right shoulder, left upper leg and left knee use Index finger information
    ## Hint: body sample points也有分incremental與decremental
    # print(handIncSimilarInd[1])
    for _jointInd in [0,1]:
        bodyIncMappingResult[_jointInd] = bodySamplePts[_jointInd][0][handIncSimilarInd[1]]
        bodyDecMappingResult[_jointInd] = bodySamplePts[_jointInd][1][handDecSimilarInd[1]]
    ## left shoulder, right upper leg and right knee use Index finger information
    for _jointInd in [2,3]:
        bodyIncMappingResult[_jointInd] = bodySamplePts[_jointInd][0][handIncSimilarInd[3]]
        bodyDecMappingResult[_jointInd] = bodySamplePts[_jointInd][1][handDecSimilarInd[3]]
    
    # print(bodyIncMappingResult[0])

    # 4. 
    saveDirPath = 'data/mappedResult/'
    if not os.path.isdir(saveDirPath):
        os.makedirs(saveDirPath)
    for _jointInd in bodyJoints:
        # incremental
        with open(
            os.path.join(saveDirPath, 'inc_{0}.npy'.format(_jointInd)), 
            'wb'
        ) as OFile:
            np.save(OFile, bodyIncMappingResult[_jointInd])
        # decremental
        with open(
            os.path.join(saveDirPath, 'dec_{0}.npy'.format(_jointInd)), 
            'wb'
        ) as OFile:
            np.save(OFile, bodyDecMappingResult[_jointInd])
        print(type(bodyIncMappingResult[_jointInd]))

def visualizeRotApplyToAvatar():
    pass

if __name__=='__main__':
    # constructMappingFunc()
    main()
    pass