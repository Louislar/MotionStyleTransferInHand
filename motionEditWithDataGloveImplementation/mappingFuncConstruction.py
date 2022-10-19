'''
重現Motion editing with data glove的system
'''

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import json
import os 
import sys
import copy
from scipy.ndimage import uniform_filter1d
sys.path.append("../")
from testingStageViz import jsonToDf
from rotationAnalysis import butterworthLowPassFilter, bSplineFitting

def main():
    # 1.1 read hand rotation data
    # 1.2 read body rotation data
    # 2. Average filter apply to both rotation curves (注意, not gaussian filter)
    # 3. Low pass filter apply to both rotation curves via FFT
    # 4. Compute tangent in both curves and find the time point that tanget(slope; 斜率) is 0
    # 4.1 Decide the frequency of hand and body joint's rotation data
    # 5. Construct multiple mapping functions with discrete sample points
    # 5.1 Index finger to right shoulder
    # 5.2 Index finger to left upper leg
    # 5.3 Index finger to left knee
    # 5.4 Middle finger to left shoulder
    # 5.5 Middle finger to right shoulder
    # 5.6 Middle finger to right shoulder
    # 6. Use B-Spline fitting to interpolate each mapping function
    # 7. Store each mapping function in file

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
    # vizJoint = 3
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
    handTipAndPitInd[1] = [[1468, 1499], [1499, 1537]]    # 谷, 峰, 谷. avgFilter:35, lowpass cutout:0.6, gradient bound:0.05
    handTipAndPitInd[3] = [[1053, 1098], [1033, 1053]]    # 峰, 谷, 峰. avgFilter:50, lowpass cutout:0.6, gradient bound:0.05
    
    bodyTipAndPitInd[0] = [[300, 334], [265, 300]]    # 峰, 谷, 峰. avgFilter:50, lowpass cutout:0.6, gradient bound:0.05
    bodyTipAndPitInd[1] = [[125, 152], [152, 194]]    # 谷, 峰, 谷. avgFilter:50, lowpass cutout:0.6, gradient bound:0.05
    bodyTipAndPitInd[2] = [[291, 322], [322, 360]]    # 谷, 峰, 谷. avgFilter:100, lowpass cutout:0.2, gradient bound:0.05
    bodyTipAndPitInd[3] = [[205, 234], [234, 274]]    # 谷, 峰, 谷. avgFilter:50, lowpass cutout:0.6, gradient bound:0.05
    # print(bodyLocalExtremaInd[3]['x'].tolist())

    ## visualize 選擇的tip and pit的位置
    # vizAxis = 'x'
    # vizJoint = 3
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

    # 5. 
    ## Hint: 只會拿x軸的旋轉作mapping function
    ## TODO: 目前收集的body rotation只有lower body, 
    ## 但是這篇文章會使用到left and right shoulder的旋轉.
    # 5.0 所有data先做一次B-Spiline fitting and interpolation, 取相同數量的sample points
    ## 上升與下降區段分開做B-Spline interpolation
    ## B-Spline fitting可以參考rotationAnalysis.py 
    numSamplePts = 100
    handBSplineSamplePts = {_jointInd: [] for _jointInd in range(handJointsCount)}
    bodyBSplineSamplePts = {_jointInd: [] for _jointInd in range(handJointsCount)}
    for _jointInd in [1, 3]:
        print(handRot[_jointInd]['x'])
        handTipAndPitInd[_jointInd]
        break

    # 5.1 index and right shoulder (TODO)
    # 5.2 index and left upper leg 


if __name__=='__main__':
    main()
    pass