import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
import time

boneHeirarchy = []

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
    def __init__(self, jointsDataIn, skeletonIn) -> None:
        '''
        :dataIn: (dict) key: joint name, 
            value: DataFrame with XYZ as columns, frame number as rows
        '''
        self.jointsData = jointsDataIn
        self.boneHeirarchy = skeletonIn

    def plotJoints(self, ax, jointsData, frameNum=0):
        jointLine, = ax.plot(
            [_jointDf['x'][frameNum] for _jointDf in jointsData.values()],  # x Values
            [_jointDf['y'][frameNum] for _jointDf in jointsData.values()],  # y Values
            [_jointDf['z'][frameNum] for _jointDf in jointsData.values()],   # z Values
            '.'
        )
        return jointLine

    def plotBones(self, ax, jointsData, boneChain, frameNum=0):
        _p = None   # For keeping all the lines(bones) in same color
        bonesLines = []
        boneChain = boneChain.iloc[1:, :]
        for i in range(1, boneChain.shape[0]): # First joint pair is Hip with no parent
            _parentJointNM = boneChain['parent'][i]
            _jointNM = boneChain['joint'][i]
            if _p is not None:
                _p = ax.plot(
                    [jointsData[_parentJointNM]['x'][frameNum], jointsData[_jointNM]['x'][frameNum]], 
                    [jointsData[_parentJointNM]['y'][frameNum], jointsData[_jointNM]['y'][frameNum]], 
                    [jointsData[_parentJointNM]['z'][frameNum], jointsData[_jointNM]['z'][frameNum]], 
                    color=_p[0].get_color()
                )
                bonesLines.append(_p[0])
            else: 
                _p = ax.plot(
                    [jointsData[_parentJointNM]['x'][frameNum], jointsData[_jointNM]['x'][frameNum]], 
                    [jointsData[_parentJointNM]['y'][frameNum], jointsData[_jointNM]['y'][frameNum]], 
                    [jointsData[_parentJointNM]['z'][frameNum], jointsData[_jointNM]['z'][frameNum]]
                )
                bonesLines.append(_p[0])
        return bonesLines

    def plotOneFrame(self, frameNum=0, elapseTime=10):
        '''
        :frameNum: 目標展示的frame index. 預設為第一個frame.
        :elapseTime: 圖片展示時間長度. 單位是second.
        '''
        plt.ion()   # For drawing multiple figures
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Ref: https://github.com/fabro66/GAST-Net-3DPoseEstimation/issues/51
        # But not useful...
        # ax.set_box_aspect([1,1,1])
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        
        self.plotJoints(ax, self.jointsData, frameNum)
        self.plotBones(ax, self.jointsData, self.boneHeirarchy, frameNum)
        set_axes_equal(ax)  # Keep axis in same scale

        for i in range(int(elapseTime/0.05)):
            fig.canvas.draw()
            fig.canvas.flush_events()

            time.sleep(0.05)

    def plotMultiFrames(self, frameInterval:list=None):
        '''
        :frameInterval: 目標展示的frame區間. 如果為None則展示所有frame.
        '''
        plt.ion()   # For drawing multiple figures
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        
        jointLine = self.plotJoints(ax, self.jointsData, frameNum=0)
        bonesLines = self.plotBones(ax, self.jointsData, self.boneHeirarchy, frameNum=0)
        set_axes_equal(ax)  # Keep axis in same scale

        for i in range(frameInterval[0], frameInterval[1]):
            # Update joint and bone's data
            jointLine.set_data(
                [_jointDf['x'][i] for _jointDf in self.jointsData.values()],  # x Values
                [_jointDf['y'][i] for _jointDf in self.jointsData.values()],  # y Values
            )
            jointLine.set_3d_properties([_jointDf['z'][i] for _jointDf in self.jointsData.values()], 'z')
            for _bLineIdx in range(len(bonesLines)):
                _parentJointNM = self.boneHeirarchy['parent'][_bLineIdx+1]
                _jointNM = self.boneHeirarchy['joint'][_bLineIdx+1]
                bonesLines[_bLineIdx].set_data(
                    [self.jointsData[_parentJointNM]['x'][i], self.jointsData[_jointNM]['x'][i]], 
                    [self.jointsData[_parentJointNM]['y'][i], self.jointsData[_jointNM]['y'][i]]
                )
                bonesLines[_bLineIdx].set_3d_properties(
                    [self.jointsData[_parentJointNM]['z'][i], self.jointsData[_jointNM]['z'][i]],
                    'z'
                )

            fig.canvas.draw()
            fig.canvas.flush_events()

            time.sleep(0.05)

def main():
    # 1. Read multiple joints' 3d pos time series from csv
    # 1.1 Read bone heirarchy
    # 2. Put data into Pos3DVisualizer

    # 1. 
    dataDirPath = 'data/swimming/125_parsed/125_01/'
    dataFiles = os.listdir(dataDirPath)
    dataFilesWithoutExt = [os.path.splitext(_file)[0] for _file in dataFiles]
    dataFilesPaths = [os.path.join(dataDirPath, _file) for _file in dataFiles]
    print(dataFiles)
    print(dataFilesWithoutExt)
    print(dataFilesPaths)
    
    jointsNMs = dataFilesWithoutExt
    jointsData = {_nm: None for _nm in jointsNMs}
    for i in range(len(jointsNMs)):
        jointsData[jointsNMs[i]] = pd.read_csv(dataFilesPaths[i])
    # print(jointsData['Hips'].head(10))

    # 1.1 
    skeletonFilePath = 'data/skeleton.csv'
    skeletonDf = pd.read_csv(skeletonFilePath)

    # 2. TODO: Frame rate 120, 要再調整這邊的播放速率
    pos3dviz = Pos3DVisualizer(jointsData, skeletonDf)
    # pos3dviz.plotOneFrame(0, 10)
    pos3dviz.plotMultiFrames([0, 100])



if __name__=='__main__':
    main()
    pass