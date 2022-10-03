'''
Visualize pose blending結果, 
以及blending過程的similar poses.
可能的話使用按鍵控制畫的frame (播放, 暫停)
Ref: visualize3DPos.py

1. plot blended EWMA result
2. plot original input full joints
3. plot original input 3 points (head, left hand, right hand)
4. plot nearest FV's corresponding 3d positions
5. plot nearest FV's poses (left hand and right hand)
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
import time
from poseSynthesis import readSimilarFV3dPos
from visualize3DPos import set_axes_equal

class multiFigsPloter():
    def __init__(self, inputPos, fullBodyPose) -> None:
        self.inputPos = inputPos
        self.fullBodyPose = fullBodyPose
        self.numOfFigure = 4
        self.axList = []
        pass

    def plotJoints(self, ax, jointsData, frameNum=0):
        jointLine, = ax.plot(
            [_jointDf['x'][frameNum] for _jointDf in jointsData.values()],  # x Values
            [_jointDf['y'][frameNum] for _jointDf in jointsData.values()],  # y Values
            [_jointDf['z'][frameNum] for _jointDf in jointsData.values()],   # z Values
            '.'
        )
        return jointLine
    
    def updateJoints(self, jointLine, jointsData, frameNum):
        jointLine.set_data(
                [_jointDf['x'][frameNum] for _jointDf in jointsData.values()],  # x Values
                [_jointDf['y'][frameNum] for _jointDf in jointsData.values()],  # y Values
            )
        jointLine.set_3d_properties([_jointDf['z'][frameNum] for _jointDf in jointsData.values()], 'z')
    
    def plotInputPos(self, ax, jointsData, frameNum=0):
        '''
        類似self.plotJoints(). 
        但是, 要改變點的大小, 以及點的顏色. (變大, 紅色)
        '''
        # TODO: finish this
        jointLine, = ax.plot(
            [_jointDf['x'][frameNum] for _jointDf in jointsData.values()],  # x Values
            [_jointDf['y'][frameNum] for _jointDf in jointsData.values()],  # y Values
            [_jointDf['z'][frameNum] for _jointDf in jointsData.values()],   # z Values
            '.', 
            color = 'r', 
            markersize=15
        )
        return jointLine

    def plotOneFrame(self, frameNum=0, elapseTime=10):
        '''
        :frameNum: 目標展示的frame index. 預設為第一個frame.
        :elapseTime: 圖片展示時間長度. 單位是second.
        '''
        plt.ion()   # For drawing multiple figures
        fig = plt.figure()
        for i in range(1, 4+1):
            ax = fig.add_subplot(1, 4, i, projection='3d')
            ax.set_xlabel('x axis')
            ax.set_ylabel('y axis')
            ax.set_zlabel('z axis')
            self.axList.append(ax)
        plt.subplot_tool()  # 開啟調整子圖排列的GUI. Ref: https://www.delftstack.com/zh-tw/howto/matplotlib/how-to-improve-subplot-size-or-spacing-with-many-subplots-in-matplotlib/
        # plot input position in the right most plot
        inputPosLine = self.plotInputPos(self.axList[0], self.inputPos, 0)

        # keep axes in same scale
        for i in self.axList:
            set_axes_equal(i)

        for i in range(int(elapseTime/0.05)):
            fig.canvas.draw()
            fig.canvas.flush_events()

            time.sleep(0.05)
    
    def plotMultiFrames(self, frameInterval:list=None, framePerSec=20):
        '''
        :frameInterval: 目標展示的frame區間. 如果為None則展示所有frame.
        '''
        plt.ion()   # For drawing multiple figures
        fig = plt.figure()
        for i in range(1, 4+1):
            ax = fig.add_subplot(1, 4, i, projection='3d')
            ax.set_xlabel('x axis')
            ax.set_ylabel('y axis')
            ax.set_zlabel('z axis')
            self.axList.append(ax)
        plt.subplot_tool()
        # plot input position in the right most plot
        inputPosLine = self.plotInputPos(self.axList[0], self.inputPos, 0)
        # keep axes in same scale
        for i in self.axList:
            set_axes_equal(i)

        for i in range(frameInterval[0], frameInterval[1]):
            self.updateJoints(inputPosLine, self.inputPos, i)

            fig.canvas.draw()
            fig.canvas.flush_events()

            time.sleep(0.05)

    pass

def main():
    ## 需要注意, 統一3d position格式為pandas dataframe, column為x, y, z
    # 1. Read input position data (只有左右手. head為0) (similar3dPos的第一個XYZ)
    # 2. Read Ground true (full-body pose) (similar3dPos的第一個XYZ)
    # 3. read blended EWMA result (blendedEWMA)
    # 4. read nearest FV's corresponding positions (只有左右手) (similar3dPos的所有XYZ)
    # 5. plot above information

    motionDirPath = 'data/swimming/'
    usedJointNm = ['lhand', 'rhand']
    # 1. 
    # 這邊需要一個像similarFeatVecsSearch.readAllFeatVecsOfAMotion()的函數. 
    #       他能夠讀取所有與FV對應的3d positions (包含所有trials)
    #       -> 已實作過 poseSynthesis.readSimilarFV3dPos()
    # [注意] 這邊假裝FV最相似的pose就是input pose
    ## Change to pd.dataframe and make columns name to x, y, z
    similarFV3dPos = readSimilarFV3dPos(motionDirPath)
    similarFV3dPos = {
        _refJoint: {
            k: pd.DataFrame(v[:, 0:3]).rename(columns={0:'x', 1:'y', 2:'z'}, inplace=False) for k, v in _dic.items()
        } for _refJoint, _dic in similarFV3dPos.items()
    }
    # print(similarFV3dPos['lhand']['Chest'])

    # 2. 
    # TODO: read full-body pose
    # 上面(1. )已經讀取過了. 接下來只需要轉換成pd.DataFrame的格式, 再放到ploter當中 (下方)

    # 5. 
    ploter = multiFigsPloter(
        inputPos={
            'lhand': similarFV3dPos['lhand']['lhand'],
            'rhand': similarFV3dPos['rhand']['rhand'], 
            'Head': pd.DataFrame(np.zeros(similarFV3dPos['rhand']['rhand'].shape)).rename(columns={0:'x', 1:'y', 2:'z'}, inplace=False)
        },
        fullBodyPose={
            # TODO: 
        }
    )
    # ploter.plotOneFrame()
    ploter.plotMultiFrames([0, 5000], 20)
    pass

if __name__ == '__main__':
    main()
    pass