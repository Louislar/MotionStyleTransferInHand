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

class multiFigsPloter():
    def __init__(self, inputPos) -> None:
        self.inputPos = inputPos
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
        pass
    
    def plotInputPos(self, ax, jointsData, frameNum=0):
        '''
        類似self.plotJoints(). 
        但是, 要改變點的大小, 以及點的顏色. (變大, 紅色)
        '''
        jointLine, = ax.plot(
            [_jointDf['x'][frameNum] for _jointDf in jointsData.values()],  # x Values
            [_jointDf['y'][frameNum] for _jointDf in jointsData.values()],  # y Values
            [_jointDf['z'][frameNum] for _jointDf in jointsData.values()],   # z Values
            '.'
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

        for i in range(int(elapseTime/0.05)):
            fig.canvas.draw()
            fig.canvas.flush_events()

            time.sleep(0.05)
    pass

def main():
    ## 需要注意, 統一3d position格式為pandas dataframe, column為x, y, z
    # 1. Read input position data (只有head和左右手) (similar3dPos的第一個XYZ)
    # 2. Read Ground true (full-body pose) (similar3dPos的第一個XYZ)
    # 3. read blended EWMA result (blendedEWMA)
    # 4. read nearest FV's corresponding positions (只有左右手) (similar3dPos的所有XYZ)
    # 5. plot above information

    motionDirPath = 'data/swimming/'
    usedJointNm = ['lhand', 'rhand']
    # 1. 
    # TODO: 這邊需要一個像similarFeatVecsSearch.readAllFeatVecsOfAMotion()的函數. 
    #       他能夠讀取所有與FV對應的3d positions (包含所有trials)
    inputPosDirPath = os.path.join(motionDirPath, 'similar3dPos')
    inputPosData = {_joint: None for _joint in usedJointNm}
    for _joint in usedJointNm:
        inputPosData[_joint] = pd.read_csv(
            os.path.join(inputPosDirPath, _joint+'.csv')
        )
    print(inputPosData['lhand'])

    # 5. 
    ploter = multiFigsPloter()
    # ploter.plotOneFrame()
    pass

if __name__ == '__main__':
    main()
    pass