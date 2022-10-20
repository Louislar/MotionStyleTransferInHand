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
import time
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import splev, splrep
sys.path.append("../")
from testingStageViz import jsonToDf
from rotationAnalysis import butterworthLowPassFilter, bSplineFitting
from testingStageViz import set_axes_equal
from realTimeRotToAvatarPos import loadTPosePosAndVecs, forwardKinematic
from realTimeHandRotationCompute import negateAxes, heightWidthCorrection, kalmanFilter, negateXYZMask
from positionAnalysis import jointsNames
from realTimeHandRotationCompute import jointsNames as handJointsNames

# ref: testingStageViz.py
class Pos3DVisualizer():
    def __init__(self, handLMDataIn, handSkeletonIn, 
        lowerBodymappedDataIn, lowerBodySkeletonIn
        ) -> None:
        '''
        :dataIn: (dict) key: joint name, 
            value: DataFrame with XYZ as columns, frame number as rows
        '''
        self.handLMData = handLMDataIn
        self.handSkeleton = handSkeletonIn
        self.lowerBodyData = lowerBodymappedDataIn
        self.lowerBodySkeleton = lowerBodySkeletonIn
        self.axList = []

    def plotJoints(self, ax, jointsData, frameNum=0, **kwargs):
        jointLine, = ax.plot(
            [_jointDf['x'][frameNum] for _jointDf in jointsData.values()],  # x Values
            [_jointDf['y'][frameNum] for _jointDf in jointsData.values()],  # y Values
            [_jointDf['z'][frameNum] for _jointDf in jointsData.values()],   # z Values
            '.', 
            **kwargs
        )
        return jointLine
    
    def updateJoints(self, jointLine, jointsData, frameNum):
        jointLine.set_data(
                [_jointDf['x'][frameNum] for _jointDf in jointsData.values()],  # x Values
                [_jointDf['y'][frameNum] for _jointDf in jointsData.values()],  # y Values
            )
        jointLine.set_3d_properties([_jointDf['z'][frameNum] for _jointDf in jointsData.values()], 'z')

    def plotBones(self, ax, jointsData, boneChain, frameNum=0, **kwargs):
        _color = kwargs['color'] if 'color' in kwargs else None   # For keeping all the lines(bones) in same color
        kwargs.pop('color', None)   # color交由_color變數儲存, 避免重複輸入
        bonesLines = []
        # boneChain = boneChain.iloc[1:, :]
        for i in range(0, boneChain.shape[0]): # First joint pair is Hip with no parent
            _parentJointNM = boneChain['parent'][i]
            _jointNM = boneChain['joint'][i]
            if _color is not None:
                _p = ax.plot(
                    [jointsData[_parentJointNM]['x'][frameNum], jointsData[_jointNM]['x'][frameNum]], 
                    [jointsData[_parentJointNM]['y'][frameNum], jointsData[_jointNM]['y'][frameNum]], 
                    [jointsData[_parentJointNM]['z'][frameNum], jointsData[_jointNM]['z'][frameNum]], 
                    color=_color, 
                    **kwargs
                )
                bonesLines.append(_p[0])
            else: 
                _p = ax.plot(
                    [jointsData[_parentJointNM]['x'][frameNum], jointsData[_jointNM]['x'][frameNum]], 
                    [jointsData[_parentJointNM]['y'][frameNum], jointsData[_jointNM]['y'][frameNum]], 
                    [jointsData[_parentJointNM]['z'][frameNum], jointsData[_jointNM]['z'][frameNum]],
                    **kwargs
                )
                _color = _p[0].get_color()
                bonesLines.append(_p[0])
        return bonesLines

    def updateBones(self, bonesLines, jointsData, boneChain, frameNum=0):
        for _bLineIdx in range(len(bonesLines)):
            _parentJointNM = boneChain['parent'][_bLineIdx]
            _jointNM = boneChain['joint'][_bLineIdx]
            bonesLines[_bLineIdx].set_data(
                [jointsData[_parentJointNM]['x'][frameNum], jointsData[_jointNM]['x'][frameNum]], 
                [jointsData[_parentJointNM]['y'][frameNum], jointsData[_jointNM]['y'][frameNum]]
            )
            bonesLines[_bLineIdx].set_3d_properties(
                [jointsData[_parentJointNM]['z'][frameNum], jointsData[_jointNM]['z'][frameNum]],
                'z'
            )

    def plotMultiFrames(self, numOfPlot=2, frameInterval:list=[0, 100]):
        '''
        :frameInterval: 目標展示的frame區間. 如果為None則展示所有frame.
        '''
        plt.ion()   # For drawing multiple figures
        fig = plt.figure()
        for i in range(numOfPlot):
            ax = fig.add_subplot(1, numOfPlot, i+1, projection='3d')
            ax.set_xlabel('x axis')
            ax.set_ylabel('y axis')
            ax.set_zlabel('z axis')
            self.axList.append(ax)
        
        ## plot hand landmarks
        handLMLine = self.plotJoints(
            self.axList[0], self.handLMData, frameNum=10, 
            color='r', markersize=15
        )
        handBonesLines = self.plotBones(
            self.axList[0], self.handLMData, self.handSkeleton, frameNum=10,
            color='r'
        )
        ## plot mapped lower body poses
        mappedLBLine = self.plotJoints(
            self.axList[1], self.lowerBodyData, frameNum=10, 
            color='g', markersize=15
        )
        mappedLBBonesLine = self.plotBones(
            self.axList[1], self.lowerBodyData, self.lowerBodySkeleton, frameNum=10,
            color='g'
        )
        ## For test debug
        ## 左腳的點多畫一次, 使用不同顏色
        tmpLine = self.plotJoints(
            self.axList[1], {2: self.lowerBodyData[2]}, frameNum=10, 
            color='c', markersize=17
        )
        for i in range(numOfPlot):
            set_axes_equal(self.axList[i])  # Keep axis in same scale

        for i in range(frameInterval[0], frameInterval[1]):
            # Update joint and bone's data
            self.updateJoints(handLMLine, self.handLMData, i)
            self.updateBones(handBonesLines, self.handLMData, self.handSkeleton, i)
            self.updateJoints(mappedLBLine, self.lowerBodyData, i)
            self.updateBones(mappedLBBonesLine, self.lowerBodyData, self.lowerBodySkeleton, i)
            ## For test debug
            self.updateJoints(tmpLine, {2: self.lowerBodyData[2]}, i)

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.03)
    
    def plot2dDot(self, ax, jointData, dataAxis='x', frameNum=0, **kwargs):
        jointLine, = ax.plot(
            frameNum,
            jointData[dataAxis][frameNum],
            '.',
            **kwargs
        )
        return jointLine
    
    def update2dDot(self, jointLine, jointData, dataAxis, frameNum):
        jointLine.set_data(
            frameNum,
            jointData[dataAxis][frameNum]
        )

    def plotFrameAndPrintRot(self, lowerBodyData, rotData, handPosData, handRotData, frameInterval:list=[0, 100]):
        # 想要確認該時間點的動作是對應到哪一個rotation數值
        plt.ion()   # For drawing multiple figures
        fig = plt.figure()
        plt.subplot_tool()
        ax = fig.add_subplot(2, 2, 3, projection='3d')
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.set_xlabel('x axis')
        ax1.set_ylabel('y axis')
        ax1.set_zlabel('z axis')
        # rotation plots
        axRot1 = fig.add_subplot(2, 2, 4)
        axRot1.set_xlabel('frame')
        axRot1.set_ylabel('rotation')
        axRot2 = fig.add_subplot(2, 2, 2)
        axRot2.set_xlabel('frame')
        axRot2.set_ylabel('rotation')

        ## plot rotation curve
        axRot1.set_title('mapped rotation')
        axRot1.plot(
            range(rotData[0].shape[0]),
            rotData[0]['x'], '.-'
        )
        rot1Line = self.plot2dDot(
            axRot1, rotData[0], dataAxis='x', frameNum=0,
            color='c', markersize=15
        )
        axRot2.set_title('hand rotation')
        axRot2.plot(
            range(handRotData[1].shape[0]),
            handRotData[1]['x'], '.-'
        )
        rot2Line = self.plot2dDot(
            axRot2, handRotData[1], dataAxis='x', frameNum=0,
            color='c', markersize=15
        )
        ## plot hand positions (landmarks)
        handPosLine = self.plotJoints(
            ax1, handPosData, frameNum=10, 
            color='b', markersize=15
        )
        handPosBonesLine = self.plotBones(
            ax1, handPosData, self.handSkeleton, frameNum=10,
            color='b'
        )

        ## plot mapped lower body poses
        mappedLBLine = self.plotJoints(
            ax, lowerBodyData, frameNum=10, 
            color='g', markersize=15
        )
        mappedLBBonesLine = self.plotBones(
            ax, lowerBodyData, self.lowerBodySkeleton, frameNum=10,
            color='g'
        )
        ## 左腳的點多畫一次, 使用不同顏色
        tmpLine = self.plotJoints(
            ax, {2: lowerBodyData[2]}, frameNum=10, 
            color='c', markersize=17
        )
        set_axes_equal(ax)  # Keep axis in same scale
        set_axes_equal(ax1)  # Keep axis in same scale
        for i in range(frameInterval[0], frameInterval[1]):
            # Update joint and bone's data
            self.updateJoints(mappedLBLine, lowerBodyData, i)
            self.updateBones(mappedLBBonesLine, lowerBodyData, self.lowerBodySkeleton, i)
            self.updateJoints(handPosLine, handPosData, i)
            self.updateBones(handPosBonesLine, handPosData, self.handSkeleton, i)
            self.update2dDot(rot1Line, rotData[0], 'x', i)
            self.update2dDot(rot2Line, handRotData[1], 'x', i)
            ## For test debug (畫出明顯的左腳joint)
            self.updateJoints(tmpLine, {2: lowerBodyData[2]}, i)

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.03)

# average filter and low pass filter一次處理
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
    vizJoint = 1
    vizTarget = 'hand'    # or 'body' or 'hand'
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
    vizAxis = 'x'
    vizJoint = 1
    vizTarget = 'hand'    # or 'body' or 'hand'
    vizData = beforeLowPassHandRot if vizTarget == 'hand' else beforeLowPassBodyRot
    vizData2 = handRot if vizTarget == 'hand' else bodyRot
    vizTimeCount = handTimeCount if vizTarget == 'hand' else bodyTimeCount
    plt.figure()
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    ax1.set_title('before low pass filter')
    ax2.set_title('after low pass filter')
    ax1.plot(range(vizTimeCount), vizData[vizJoint][vizAxis])
    ax2.plot(range(vizTimeCount), vizData2[vizJoint][vizAxis])
    plt.show()

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
    numSamplePts = int(1e5)
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
    vizJoint = 0
    vizTarget = 'body'    # or 'body' or 'hand'
    vizIncDec = 0   # 0: increase, 1: decrease
    vizData = handSeg if vizTarget == 'hand' else bodySeg
    vizData2 = handBSplineSamplePts if vizTarget == 'hand' else bodyBSplineSamplePts
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

# plot mapping function with B-Spline interpolated sample points
def plotMappingFunction():
    # 1. read hand and body sample points
    # 2. plot pair of sample points
    # 2.1 index finger and left upper leg
    # 2.2  

    samplePtsDirPath = 'data/'
    handJoints = [1,3]
    bodyJoints = [0,1,2,3]
    # 1. 
    handSamplePtsInc = {}
    handSamplePtsDec = {}
    bodySamplePtsInc = {}
    bodySamplePtsDec = {}
    for _jointInd in handJoints:
        with open(
            os.path.join(samplePtsDirPath, 'hand_{0}_{1}.npy'.format(_jointInd, 0)), 
            'rb'
        ) as RFile:
            handSamplePtsInc[_jointInd] = np.load(RFile)
        with open(
            os.path.join(samplePtsDirPath, 'hand_{0}_{1}.npy'.format(_jointInd, 1)), 
            'rb'
        ) as RFile:
            handSamplePtsDec[_jointInd] = np.load(RFile)
    for _jointInd in bodyJoints:
        with open(
            os.path.join(samplePtsDirPath, 'body_{0}_{1}.npy'.format(_jointInd, 0)), 
            'rb'
        ) as RFile:
            bodySamplePtsInc[_jointInd] = np.load(RFile)
        with open(
            os.path.join(samplePtsDirPath, 'body_{0}_{1}.npy'.format(_jointInd, 1)), 
            'rb'
        ) as RFile:
            bodySamplePtsDec[_jointInd] = np.load(RFile)

    # 2.1 
    vizHandJoint = 1
    vizBodyJoint = 0
    incOrDec = 0    # 0: inc, 1: dec
    vizHandData = handSamplePtsInc if incOrDec==0 else handSamplePtsDec
    vizBodyData = bodySamplePtsInc if incOrDec==0 else bodySamplePtsDec
    plt.figure()
    plt.plot(vizHandData[vizHandJoint], vizBodyData[vizBodyJoint], '.')
    plt.show()

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

# TODO: 實現文章當中提到的最大最小值教正
def reScaleMappingFunc(newValue, originMin, originMax, newMin, newMax):
    
    pass

# 將建立好的mapping function套用到hand rotation
# TODO: 最大最小值的校正, 其實文章當中有提出 (感覺不是很好的re-scale方式)
def mapHandRotationToBodyRotation(): 
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
        handDecSimilarInd[_jointInd] = findNearestNum(handRot[1]['x'].values, handSamplePts[1][1]) # 下降區段
    
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

# mapping後的rotation apply到avatar身上
def applyRotToAvatar():
    # 1. read mapping result
    # 1.1 convert mapping result to 標準格式 (dataframe with x,y,z 三個column)
    # 2. read T pose positions
    # 3. apply to avatar (ref: realTimeRotToAvatarPos.py)
    # 4. store avatar positions

    mappingResDirPath = 'data/mappedResult/'
    bodyJoints = [0,1,2,3]
    # 1. 
    incMappingRes = {_jointInd: [] for _jointInd in bodyJoints}
    decMappingRes = {_jointInd: [] for _jointInd in bodyJoints}
    for _jointInd in bodyJoints:
        with open(
            os.path.join(mappingResDirPath, 'inc_{0}.npy'.format(_jointInd)), 
            'rb'
        ) as RFile:
            incMappingRes[_jointInd] = np.load(RFile)
        with open(
            os.path.join(mappingResDirPath, 'dec_{0}.npy'.format(_jointInd)), 
            'rb'
        ) as RFile:
            decMappingRes[_jointInd] = np.load(RFile)
    # print(incMappingRes[0])
    # print(incMappingRes[0].shape)
    # 1.1 
    incMapResDf = {}
    decMapResDf = {}
    for _jointInd in bodyJoints:
        incMapResDf[_jointInd] = pd.DataFrame({
            'x': incMappingRes[_jointInd],
            'y': np.zeros(incMappingRes[_jointInd].shape[0]),
            'z': np.zeros(incMappingRes[_jointInd].shape[0])
        })
        decMapResDf[_jointInd] = pd.DataFrame({
            'x': decMappingRes[_jointInd],
            'y': np.zeros(incMappingRes[_jointInd].shape[0]),
            'z': np.zeros(incMappingRes[_jointInd].shape[0])
        })
    # print(incMapResDf[0])
    # print(incMapResDf[0].shape)

    # 2. 
    tPoseDirPath='../TPoseInfo/genericAvatar/'
    TPosePositions, TPoseVectors = loadTPosePosAndVecs(tPoseDirPath)

    # 3. 
    usedLowerBodyJoints = [
        jointsNames.LeftUpperLeg, jointsNames.LeftLowerLeg, jointsNames.LeftFoot, 
        jointsNames.RightUpperLeg, jointsNames.RightLowerLeg, jointsNames.RightFoot, 
        jointsNames.Hip
    ]
    leftKinematic = [
        TPosePositions[jointsNames.LeftUpperLeg], 
        TPoseVectors[0], 
        TPoseVectors[1]
    ]
    rightKinematic = [
        TPosePositions[jointsNames.RightUpperLeg], 
        TPoseVectors[2], 
        TPoseVectors[3]
    ]
    timeCount = incMappingRes[0].shape[0]
    # lowerBodyPositionInc = [{'time': t, 'data': {aJoint: None for aJoint in usedLowerBodyJoints}} for t in range(timeCount)]
    # lowerBodyPositionDec = [{'time': t, 'data': {aJoint: None for aJoint in usedLowerBodyJoints}} for t in range(timeCount)]
    lowerBodyPositionInc = {aJoint: np.zeros((timeCount, 3)) for aJoint in usedLowerBodyJoints}
    lowerBodyPositionDec = {aJoint: np.zeros((timeCount, 3)) for aJoint in usedLowerBodyJoints}
    for t in range(timeCount):
        ## incremental period
        testKinematic1 = forwardKinematic(
            leftKinematic, 
            [
                incMappingRes[0][t], 
                0, 
                incMappingRes[1][t]
            ]
        )
        # lowerBodyPositionInc[t]['data'][jointsNames.LeftLowerLeg] = testKinematic1[0] + testKinematic1[1]
        # lowerBodyPositionInc[t]['data'][jointsNames.LeftFoot] = testKinematic1[0] + testKinematic1[1] + testKinematic1[2]
        lowerBodyPositionInc[jointsNames.LeftLowerLeg][t, :] = testKinematic1[0] + testKinematic1[1]
        lowerBodyPositionInc[jointsNames.LeftFoot][t, :] = testKinematic1[0] + testKinematic1[1] + testKinematic1[2]
        
        testKinematic2 = forwardKinematic(
            rightKinematic, 
            [
                incMappingRes[2][t], 
                0, 
                incMappingRes[3][t]
            ]
        )
        # lowerBodyPositionInc[t]['data'][jointsNames.RightLowerLeg] = testKinematic2[0] + testKinematic2[1]
        # lowerBodyPositionInc[t]['data'][jointsNames.RightFoot] = testKinematic2[0] + testKinematic2[1] + testKinematic2[2]
        lowerBodyPositionInc[jointsNames.RightLowerLeg][t, :] = testKinematic2[0] + testKinematic2[1]
        lowerBodyPositionInc[jointsNames.RightFoot][t, :] = testKinematic2[0] + testKinematic2[1] + testKinematic2[2]

        ## decremental period
        testKinematic1 = forwardKinematic(
            leftKinematic, 
            [
                decMappingRes[0][t], 
                0, 
                decMappingRes[1][t]
            ]
        )
        lowerBodyPositionDec[jointsNames.LeftLowerLeg][t, :] = testKinematic1[0] + testKinematic1[1]
        lowerBodyPositionDec[jointsNames.LeftFoot][t, :] = testKinematic1[0] + testKinematic1[1] + testKinematic1[2]
        
        testKinematic2 = forwardKinematic(
            rightKinematic, 
            [
                decMappingRes[2][t], 
                0, 
                decMappingRes[3][t]
            ]
        )
        lowerBodyPositionDec[jointsNames.RightLowerLeg][t, :] = testKinematic2[0] + testKinematic2[1]
        lowerBodyPositionDec[jointsNames.RightFoot][t, :] = testKinematic2[0] + testKinematic2[1] + testKinematic2[2]
    
    lowerBodyPositionInc[jointsNames.LeftUpperLeg] = np.tile(TPosePositions[jointsNames.LeftUpperLeg], (timeCount,1))
    lowerBodyPositionInc[jointsNames.RightUpperLeg] = np.tile(TPosePositions[jointsNames.RightUpperLeg], (timeCount,1))
    lowerBodyPositionInc[jointsNames.Hip] = np.tile(TPosePositions[jointsNames.Hip], (timeCount,1))
    lowerBodyPositionDec[jointsNames.LeftUpperLeg] = np.tile(TPosePositions[jointsNames.LeftUpperLeg], (timeCount,1))
    lowerBodyPositionDec[jointsNames.RightUpperLeg] = np.tile(TPosePositions[jointsNames.RightUpperLeg], (timeCount,1))
    lowerBodyPositionDec[jointsNames.Hip] = np.tile(TPosePositions[jointsNames.Hip], (timeCount,1))

    posIncDf = {}
    posDecDf = {}
    for aJoint in usedLowerBodyJoints:
        posIncDf[aJoint] = pd.DataFrame({
            'x': lowerBodyPositionInc[aJoint][:,0],
            'y': lowerBodyPositionInc[aJoint][:,1],
            'z': lowerBodyPositionInc[aJoint][:,2]
        })
        posDecDf[aJoint] = pd.DataFrame({
            'x': lowerBodyPositionDec[aJoint][:,0],
            'y': lowerBodyPositionDec[aJoint][:,1],
            'z': lowerBodyPositionDec[aJoint][:,2]
        }) 
    
    # 4. 
    posDirPath = 'data/mappedPos/'
    if not os.path.isdir(posDirPath):
        os.makedirs(posDirPath)
    for aJoint in usedLowerBodyJoints:
        posIncDf[aJoint].to_csv(
            os.path.join(posDirPath, 'inc_{0}.csv'.format(aJoint)),
            index=False
        )
        posDecDf[aJoint].to_csv(
            os.path.join(posDirPath, 'dec_{0}.csv'.format(aJoint)),
            index=False
        )
    pass

# visualize avatar with mapped positions
def main():
    # 1. read mapped position
    # 2. read hand position
    # 3. read mapped rotation
    # 4.1 read 還未map的hand rotation
    # 4. visualize together

    posIncDirPath = 'data/mappedPos/'
    handPosDirPath = '../complexModel/'
    mappedRotDirPath = 'data/mappedResult/'
    handRotDirPath = '../HandRotationOuputFromHomePC/'
    bodyJoints = [0,1,2,3]
    usedLowerBodyJoints = [
        jointsNames.LeftUpperLeg, jointsNames.LeftLowerLeg, jointsNames.LeftFoot, 
        jointsNames.RightUpperLeg, jointsNames.RightLowerLeg, jointsNames.RightFoot, 
        jointsNames.Hip
    ]
    handLMUsedJoints = [
        handJointsNames.wrist, 
        handJointsNames.indexMCP, handJointsNames.indexPIP, handJointsNames.indexDIP, 
        handJointsNames.middleMCP, handJointsNames.middlePIP, handJointsNames.middleDIP
    ]
    handBoneStructure = pd.DataFrame({
        'parent': [
            handJointsNames.wrist, handJointsNames.wrist, 
            handJointsNames.indexMCP, handJointsNames.indexPIP, 
            handJointsNames.middleMCP, handJointsNames.middlePIP
        ], 
        'joint': [
            handJointsNames.indexMCP, handJointsNames.middleMCP, 
            handJointsNames.indexPIP, handJointsNames.indexDIP, 
            handJointsNames.middlePIP, handJointsNames.middleDIP
        ]
    })
    lowerBodyBoneStructure = pd.DataFrame({
        'parent': [
            jointsNames.Hip, jointsNames.Hip, 
            jointsNames.LeftUpperLeg, jointsNames.LeftLowerLeg,
            jointsNames.RightUpperLeg, jointsNames.RightLowerLeg
        ], 
        'joint': [
            jointsNames.LeftUpperLeg, jointsNames.RightUpperLeg,
            jointsNames.LeftLowerLeg, jointsNames.LeftFoot,
            jointsNames.RightLowerLeg, jointsNames.RightFoot
        ]
    })
    # 1. 
    posIncDf = {}
    posDecDf = {}
    for _jointInd in usedLowerBodyJoints:
        posIncDf[_jointInd] = pd.read_csv(
            os.path.join(posIncDirPath, 'inc_{0}.csv'.format(_jointInd))
        )
        posDecDf[_jointInd] = pd.read_csv(
            os.path.join(posIncDirPath, 'dec_{0}.csv'.format(_jointInd))
        )
    print('mapped position time count: ', posIncDf[0].shape[0])
    
    # 2. 
    ## ref: testingStageViz.py 
    handLMJson = None
    with open(os.path.join(handPosDirPath, 'frontKick.json'), 'r') as fileOpen: 
        handLMJson=json.load(fileOpen)
    timeCount = len(handLMJson)
    handLMPreproc = [{'data': None} for t in range(len(handLMJson))]
    ## preprocess
    for t in range(len(handLMJson)):
        handLandMark = negateAxes(handLMJson[t]['data'], negateXYZMask, handLMUsedJoints)
        handLandMark = heightWidthCorrection(handLandMark, handLMUsedJoints, 848, 480)
        handLandMark = kalmanFilter(handLandMark, handLMUsedJoints)
        handLMPreproc[t]['data'] = handLandMark
    ## convert to dataframe
    handLMDf = jsonToDf(handLMPreproc)
    handLMDf = {k: handLMDf[k] for k in handLMUsedJoints}
    ## make wrist origin for plotting convenience
    for k in handLMUsedJoints:
        if k != handJointsNames.wrist:
            handLMDf[k].iloc[:, :] = handLMDf[k] - handLMDf[handJointsNames.wrist]
    handLMDf[handJointsNames.wrist].iloc[:, :] = 0
    print('hand time count: ', handLMDf[0].shape[0])

    # 3. 
    mappedRotInc = {}
    mappedRotDec = {}
    for _jointInd in bodyJoints:
        with open(
            os.path.join(mappedRotDirPath, 'inc_{0}.npy'.format(_jointInd)), 
            'rb'
        ) as RFile:
            mappedRotInc[_jointInd] = np.load(RFile)
        with open(
            os.path.join(mappedRotDirPath, 'dec_{0}.npy'.format(_jointInd)), 
            'rb'
        ) as RFile:
            mappedRotDec[_jointInd] = np.load(RFile)
        ## convert to df (y, z rotation is 0)
        mappedRotInc[_jointInd] = pd.DataFrame({
            'x': mappedRotInc[_jointInd],
            'y': np.zeros(mappedRotInc[_jointInd].shape[0]),
            'z': np.zeros(mappedRotInc[_jointInd].shape[0])
        })
        mappedRotDec[_jointInd] = pd.DataFrame({
            'x': mappedRotDec[_jointInd],
            'y': np.zeros(mappedRotDec[_jointInd].shape[0]),
            'z': np.zeros(mappedRotDec[_jointInd].shape[0])
        })

    print('mapped rotation time count: ', mappedRotInc[0].shape[0])
    # 3.1 
    handRot = None
    with open(os.path.join(handRotDirPath, 'leftFrontKickStream.json'), 'r') as RFile:
        handRot = json.load(RFile)
    handRot = jsonToDf(handRot)
    print('original hand rotation time count', handRot[0].shape[0])

    # 4. TODO: 
    ## 顯示手, 下半身, rotation curve
    plotter = Pos3DVisualizer(
        handLMDf, handBoneStructure, 
        posIncDf, lowerBodyBoneStructure
    )
    # plotter.plotMultiFrames(numOfPlot=2, frameInterval=[0, handLMDf[0].shape[0]])
    plotter.plotFrameAndPrintRot(
        posIncDf, mappedRotInc, handLMDf, handRot, frameInterval=[0, mappedRotInc[0].shape[0]-1]
    )
    pass

if __name__=='__main__':
    # constructMappingFunc()
    ## 想要實際畫出mapping function的2d plot
    # plotMappingFunction()
    # mapHandRotationToBodyRotation()
    # applyRotToAvatar()
    ## 將rotation curve以及position畫在同一個figure當中
    ## TODO: 把avatar全身都畫出來會不會比較好觀察 (比較容易看懂正面以及背面)
    ##      需要把全身的skeleton hierarchy關係建立好
    ## TODO: 如果有做min max縮放回原始rotation的數值範圍, 結果會不會比較好?
    ##         應該會比較容易做
    main()
    pass