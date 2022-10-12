'''
visualize testing stage各個階段的結果
1. 手指關節移動
2. rotation mapping後的lower body motion
3. blending後的lower body motion
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
import time
import json
from realTimeHandRotationCompute import jointsNames as handJointsNames
from realTimeHandRotationCompute import negateAxes, heightWidthCorrection, kalmanFilter, negateXYZMask
from positionAnalysis import jointsNames

# Ref: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

class Pos3DVisualizer():
    def __init__(self, handLMDataIn, handSkeletonIn, lowerBodymappedDataIn, lowerBodySkeletonIn, lowerBodyBlendedDataIn) -> None:
        '''
        :dataIn: (dict) key: joint name, 
            value: DataFrame with XYZ as columns, frame number as rows
        '''
        self.handLMData = handLMDataIn
        self.handSkeleton = handSkeletonIn
        self.lowerBodyData = lowerBodymappedDataIn
        self.lowerBodySkeleton = lowerBodySkeletonIn
        self.lowerBodyBlendData = lowerBodyBlendedDataIn
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

    def plotMultiFrames(self, numOfPlot=3, frameInterval:list=[0, 100]):
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
        ## plot blended lower body poses
        blendedLBLine = self.plotJoints(
            self.axList[2], self.lowerBodyBlendData, frameNum=10, 
            color='b', markersize=15
        )
        blendedLBBonesLine = self.plotBones(
            self.axList[2], self.lowerBodyBlendData, self.lowerBodySkeleton, frameNum=10,
            color='b'
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
            self.updateJoints(blendedLBLine, self.lowerBodyBlendData, i)
            self.updateBones(blendedLBBonesLine, self.lowerBodyBlendData, self.lowerBodySkeleton, i)
            ## For test debug
            self.updateJoints(tmpLine, {2: self.lowerBodyData[2]}, i)

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.03)

def jsonToDf(lmJson):
    '''
    land mark json轉換成dataframe
    column為x, y, z
    '''
    timeCount = len(lmJson)
    numOfJoints = len(lmJson[0]['data'])
    axisNms = ['x', 'y', 'z']
    timeSeries = {_joint: {_axis: [] for _axis in axisNms} for _joint in range(numOfJoints)}
    for _joint in range(numOfJoints):
        for t in range(timeCount):
            for _axis in axisNms:
                timeSeries[_joint][_axis].append(
                    lmJson[t]['data'][_joint][_axis]
                )
    jointsSeries = {
        _joint: pd.DataFrame(timeSeries[_joint]) for _joint in range(numOfJoints)
    }
    # print(jointsSeries[1])

    return jointsSeries

def main():
    # 1. 讀取手指關節資訊. 以streaming方式做完三個預處理動作後. 
    #       轉換成dataframe格式
    # 2. rotation mapping後apply到avatar的lower body motion
    # 3. blending後的avatar lower body motion
    # 4. 所有資料visualize


    handLandMarkFilePath = 'complexModel/frontKick.json'
    handLMUsedJoints = [
        handJointsNames.wrist, 
        handJointsNames.indexMCP, handJointsNames.indexPIP, handJointsNames.indexDIP, 
        handJointsNames.middleMCP, handJointsNames.middlePIP, handJointsNames.middleDIP
    ]
    usedLowerBodyJoints = [
    jointsNames.LeftUpperLeg, jointsNames.LeftLowerLeg, jointsNames.LeftFoot, 
    jointsNames.RightUpperLeg, jointsNames.RightLowerLeg, jointsNames.RightFoot, 
    jointsNames.Hip
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
    handLMJson = None
    with open(handLandMarkFilePath, 'r') as fileOpen: 
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
    print('timeCount: ', timeCount)

    # 2. 
    rotApplySaveDirPath='positionData/fromAfterMappingHand/'
    lowerBodyPosition = None
    with open(rotApplySaveDirPath+'leftFrontKickStreamLinearMapping_TFFTTT.json', 'r') as WFile:
        lowerBodyPosition=json.load(WFile)
    timeCount = len(lowerBodyPosition)
    jointsIndices = [i for i in list(lowerBodyPosition[0]['data'].keys())]
    lowerBodyPosition = [{'data': {int(i): lowerBodyPosition[t]['data'][i] for i in jointsIndices}} for t in range(timeCount)]
    lowerBodyDf = jsonToDf(lowerBodyPosition)
    # print({k: lowerBodyDf[k] for k in lowerBodyDf if k<3})
    # print('lower body joint indices:', jointsIndices)
    print('timeCount: ', timeCount)

    # 3. 
    blendingResultDirPath = './positionData/afterSynthesis/'
    blendingResult = None
    with open(blendingResultDirPath+'leftFrontKickStreamLinearMapping_TFFTTT_075_EWMA.json', 'r') as WFile:
        blendingResult=json.load(WFile)
    timeCount = len(blendingResult)
    blendingResultDf = jsonToDf(blendingResult)
    blendingResultDf = {_joint: blendingResultDf[_joint] for _joint in usedLowerBodyJoints}
    print('timeCount: ', timeCount)
    
    

    # 4. 
    plotter = Pos3DVisualizer(
        handLMDf, handBoneStructure, 
        lowerBodyDf, lowerBodyBoneStructure,
        blendingResultDf
    )
    plotter.plotMultiFrames(3, [0, 1000])

if __name__=='__main__':
    main()
    pass