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
    def __init__(
        self, 
        skeletonIn, inputPos, fullBodyPose, blendedPose, similarFV3dPos,
        similar3dPoses
    ) -> None:
        self.skeleton = skeletonIn
        self.inputPos = inputPos
        self.fullBodyPose = fullBodyPose
        self.blendedPose = blendedPose
        self.similarFV3dPos = similarFV3dPos
        self.similar3dPoses = similar3dPoses
        self.numOfFigure = 4
        self.axList = []
        self.figList = []
        pass

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
    
    def plotInputPos(self, ax, jointsData, frameNum=0):
        '''
        類似self.plotJoints(). 
        但是, 要改變點的大小, 以及點的顏色. (變大, 紅色)
        '''
        jointLine, = ax.plot(
            [_jointDf['x'][frameNum] for _jointDf in jointsData.values()],  # x Values
            [_jointDf['y'][frameNum] for _jointDf in jointsData.values()],  # y Values
            [_jointDf['z'][frameNum] for _jointDf in jointsData.values()],   # z Values
            '.', 
            color = 'r', 
            markersize=15
        )
        return jointLine

    def plotFullBodyJoints(self, ax, jointsData, frameNum=0, **kwargs):
        jointLine, = ax.plot(
            [_jointDf['x'][frameNum] for _jointDf in jointsData.values()],  # x Values
            [_jointDf['y'][frameNum] for _jointDf in jointsData.values()],  # y Values
            [_jointDf['z'][frameNum] for _jointDf in jointsData.values()],   # z Values
            '.', 
            **kwargs
        )
        return jointLine

    def plotFullBodyBones(self, ax, jointsData, boneChain, frameNum=0, **kwargs):
        _color = kwargs['color'] if 'color' in kwargs else None   # For keeping all the lines(bones) in same color
        kwargs.pop('color', None)   # color交由_color變數儲存, 避免重複輸入
        bonesLines = []
        boneChain = boneChain.iloc[1:, :]
        for i in range(1, boneChain.shape[0]): # First joint pair is Hip with no parent
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

    def updateFullBodyBones(self, bonesLines, jointsData, boneChain, frameNum=0):
        for _bLineIdx in range(len(bonesLines)):
            _parentJointNM = boneChain['parent'][_bLineIdx+1]
            _jointNM = boneChain['joint'][_bLineIdx+1]
            bonesLines[_bLineIdx].set_data(
                [jointsData[_parentJointNM]['x'][frameNum], jointsData[_jointNM]['x'][frameNum]], 
                [jointsData[_parentJointNM]['y'][frameNum], jointsData[_jointNM]['y'][frameNum]]
            )
            bonesLines[_bLineIdx].set_3d_properties(
                [jointsData[_parentJointNM]['z'][frameNum], jointsData[_jointNM]['z'][frameNum]],
                'z'
            )

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
        # Plot full-body joint
        groundTruthLine = self.plotFullBodyJoints(
            self.axList[1], self.fullBodyPose, 0, 
            color='g'
        )
        groundTruthBoneLines = self.plotFullBodyBones(
            self.axList[1], self.fullBodyPose, self.skeleton, 0,
            color='g'
        )
        # Plot predicted pose
        blendedPoseLine = self.plotFullBodyJoints(
            self.axList[2], self.blendedPose, 0, 
            color='b'
        )
        blendedPoseBoneLines = self.plotFullBodyBones(
            self.axList[2], self.blendedPose, self.skeleton, 0,
            color='b'
        )
        # Plot similar FV's 3d positions
        similarFVLine = self.plotInputPos(self.axList[3], self.inputPos, 0)
        similarLeftFV = self.plotJoints(
            self.axList[3], self.similarFV3dPos['lhand'], 
            color='c', alpha=0.7, markersize=15
        )
        similarrightFV = self.plotJoints(
            self.axList[3], self.similarFV3dPos['rhand'], 
            color='m', alpha=0.7, markersize=15
        )

        # keep axes in same scale
        for i in self.axList:
            set_axes_equal(i)
        
        # plot similar poses in another figure
        self.plotSimilarPoses(
            numColumn=5,
            numRow=1
        )

        for i in range(int(elapseTime/0.05)):
            fig.canvas.draw()
            fig.canvas.flush_events()

            time.sleep(0.05)
    
    def plotMultiFrames(self, frameInterval:list=None, framePerSec=20, numColumn=4, numRow=1):
        '''
        :frameInterval: 目標展示的frame區間. 如果為None則展示所有frame.
        '''
        plt.ion()   # For drawing multiple figures
        fig = plt.figure()
        for i in range(1, (numColumn*numRow)+1):
            ax = fig.add_subplot(numRow, numColumn, i, projection='3d')
            ax.set_xlabel('x axis')
            ax.set_ylabel('y axis')
            ax.set_zlabel('z axis')
            self.axList.append(ax)
        plt.subplot_tool()
        # plot input position in the right most plot
        self.axList[0].set_title('input')
        inputPosLine = self.plotInputPos(self.axList[0], self.inputPos, 0)
        # Plot full-body joints and bones
        self.axList[1].set_title('ground truth')
        groundTruthLine = self.plotFullBodyJoints(
            self.axList[1], self.fullBodyPose, 0, 
            color='g'
        )
        groundTruthBoneLines = self.plotFullBodyBones(
            self.axList[1], self.fullBodyPose, self.skeleton, 0,
            # color='g'
        )
        # Plot predicted pose
        self.axList[2].set_title('predicted pose')
        blendedPoseLine = self.plotFullBodyJoints(
            self.axList[2], self.blendedPose, 0, 
            color='b'
        )
        blendedPoseBoneLines = self.plotFullBodyBones(
            self.axList[2], self.blendedPose, self.skeleton, 0,
            color='b'
        )
        # Plot similar FV's 3d positions
        self.axList[3].set_title('similar neighbors')
        similarFVLine = self.plotInputPos(self.axList[3], self.inputPos, 0)
        similarLeftFVLine = self.plotJoints(
            self.axList[3], self.similarFV3dPos['lhand'], 
            color='c', alpha=0.5, markersize=15
        )
        similarrightFVLine = self.plotJoints(
            self.axList[3], self.similarFV3dPos['rhand'], 
            color='m', alpha=0.5, markersize=15
        )
        # keep axes in same scale
        for i in self.axList:
            set_axes_equal(i)

        # plot similar poses in another figure
        leftJointLines, leftBoneLines, rightJointLines, rightBoneLines = \
            self.plotSimilarPoses(
                numColumn=5,
                numRow=1
            )

        for i in range(frameInterval[0], frameInterval[1]):
            self.updateJoints(inputPosLine, self.inputPos, i)
            self.updateJoints(groundTruthLine, self.fullBodyPose, i)
            self.updateFullBodyBones(
                groundTruthBoneLines, self.fullBodyPose, self.skeleton, 
                i
            )
            self.updateJoints(blendedPoseLine, self.blendedPose, i)
            self.updateFullBodyBones(
                blendedPoseBoneLines, self.blendedPose, self.skeleton, 
                i
            )
            self.updateJoints(similarFVLine, self.inputPos, i)
            self.updateJoints(similarLeftFVLine, self.similarFV3dPos['lhand'], i)
            self.updateJoints(similarrightFVLine, self.similarFV3dPos['rhand'], i)
            self.updateSimilarPoses(
                leftJointLines, leftBoneLines, rightJointLines, rightBoneLines,
                self.similar3dPoses['lhand'], self.similar3dPoses['rhand'], self.skeleton,
                5, 1, i
            )

            fig.canvas.draw()
            fig.canvas.flush_events()

            time.sleep(0.03)

    def plotSimilarPoses(self, numColumn=5, numRow=1):
        '''
        TODO: 展示前幾相似的full-body poses.
        這個函數應該要在plotMultiFrames當中被呼叫
        '''
        # plt.ion()   # For drawing multiple figures
        fig = plt.figure()
        _axList = []
        for i in range(1, (numColumn*numRow)+1):
            ax = fig.add_subplot(numRow, numColumn, i, projection='3d')
            ax.set_title('top {0}'.format(i))
            ax.set_xlabel('x axis')
            ax.set_ylabel('y axis')
            ax.set_zlabel('z axis')
            _axList.append(ax)
        plt.subplot_tool()

        leftJointLines = []
        leftBoneLines = []
        rightJointLines = []
        rightBoneLines = []
        # plot left hand similar poses
        for i in range(numColumn*numRow):
            _j = self.plotFullBodyJoints(
                _axList[i], self.similar3dPoses['lhand'][i], 0,
                color='c', alpha=0.7
            )
            _b = self.plotFullBodyBones(
                _axList[i], self.similar3dPoses['lhand'][i], self.skeleton, 0,
                color='c', alpha=0.7
            )
            leftJointLines.append(_j)
            leftBoneLines.append(_b)
        # plot right hand similar poses
        for i in range(numColumn*numRow):
            _j = self.plotFullBodyJoints(
                _axList[i], self.similar3dPoses['rhand'][i], 0,
                color='m', alpha=0.7
            )
            _b = self.plotFullBodyBones(
                _axList[i], self.similar3dPoses['rhand'][i], self.skeleton, 0,
                color='m', alpha=0.7
            )
            rightJointLines.append(_j)
            rightBoneLines.append(_b)

        # keep axes in same scale
        for i in _axList:
            set_axes_equal(i)
        
        return leftJointLines, leftBoneLines, rightJointLines, rightBoneLines

    def updateSimilarPoses(
        self, leftJointLines, leftBoneLines, rightJointLines, rightBoneLines, 
        leftJointsData, rightJointsData, boneChain, numColumn=5, numRow=1, frameNum=0
    ):
        for i in range(numColumn*numRow):
            self.updateJoints(leftJointLines[i], leftJointsData[i], frameNum)
            self.updateJoints(rightJointLines[i], rightJointsData[i], frameNum)
            self.updateFullBodyBones(leftBoneLines[i], leftJointsData[i], boneChain, frameNum)
            self.updateFullBodyBones(rightBoneLines[i], rightJointsData[i], boneChain, frameNum)

def main():
    ## 需要注意, 統一3d position格式為pandas dataframe, column為x, y, z
    # 0. read skeleton for drawing bones
    # 1. Read input position data (只有左右手. head為0) (similar3dPos的第一個XYZ)
    # 2. Read Ground true (full-body pose) (similar3dPos的第一個XYZ)
    # 3. read blended EWMA result (blendedEWMA)
    # 4. read nearest FV's corresponding positions (只有左右手) (similar3dPos的第一個XYZ)
    # 5. read similar FV's poses (full-body joints) (similar3dPos的所有XYZ)
    # 6. plot above information

    motionDirPath = 'data/swimming/'
    usedJointNm = ['lhand', 'rhand']
    skeletonFilePath = 'data/skeleton.csv'
    numOfCandidate = 5
    # 0. 
    skeletonDf = pd.read_csv(skeletonFilePath)
    # 1. 
    # 這邊需要一個像similarFeatVecsSearch.readAllFeatVecsOfAMotion()的函數. 
    #       他能夠讀取所有與FV對應的3d positions (包含所有trials)
    #       -> 已實作過 poseSynthesis.readSimilarFV3dPos()
    # [注意] 這邊假裝FV最相似的pose就是input pose
    ## Change to pd.dataframe and make columns name to x, y, z
    similarFV3dPos = readSimilarFV3dPos(motionDirPath)
    similarFVFullBody3dPos = similarFV3dPos
    similarFV3dPos = {
        _refJoint: {
            k: pd.DataFrame(v[:, 0:3]).rename(columns={0:'x', 1:'y', 2:'z'}, inplace=False) for k, v in _dic.items()
        } for _refJoint, _dic in similarFV3dPos.items()
    }
    # print(similarFV3dPos['lhand']['Chest'])

    # 2. 
    ## read full-body pose
    ## 上面(1. )已經讀取過了. 接下來只需要轉換成pd.DataFrame的格式, 再放到ploter當中 (下方)

    # 3. 
    ## 讀取經過blended + EWMA的pose
    blendedEWMAPoseDirPath = os.path.join(motionDirPath, 'blendedEWMA')
    _fileNms = [i for i in os.listdir(blendedEWMAPoseDirPath)]
    _filePaths = [os.path.join(blendedEWMAPoseDirPath, i) for i in _fileNms]
    _fileNms = [i.replace('.csv', '') for i in _fileNms]
    blendedEWMAPose = {_joint: None for _joint in _fileNms}
    for _joint, _path in zip(_fileNms, _filePaths):
        blendedEWMAPose[_joint] = pd.read_csv(
            os.path.join(_path)
        ).rename(columns={'0': 'x', '1': 'y', '2': 'z'}, inplace=False)
    # print(blendedEWMAPose['Chest'])

    # 4. 
    ## 讀取多個similar FV對應的3d positions. 
    ## Hint: 每個相似的點, 各自整理成一個dataframe方便作圖
    ## 資料已經讀取過了(1. )只需要lhand與rhand的資料
    leftSimilarFV3dPos = {i: None for i in range(numOfCandidate)}
    rightSimilarFV3dPos = {i: None for i in range(numOfCandidate)}
    for i in range(numOfCandidate):
        leftSimilarFV3dPos[i] = \
            pd.DataFrame(
                similarFVFullBody3dPos['lhand']['lhand'][:, 3*i:3*(i+1)]
            ).rename(columns={0: 'x', 1: 'y', 2: 'z'}, inplace=False)
        rightSimilarFV3dPos[i] = \
            pd.DataFrame(
                similarFVFullBody3dPos['rhand']['rhand'][:, 3*i:3*(i+1)]
            ).rename(columns={0: 'x', 1: 'y', 2: 'z'}, inplace=False)
    # print(leftSimilarFV3dPos[1])

    # 5. 
    ## 讀取多個similar FV對應的3d poses. 
    fullBodyJointNms = _fileNms
    leftSimilarPoses = {i: {_joint: None for _joint in fullBodyJointNms} for i in range(numOfCandidate)}
    rightSimilarPoses = {i: {_joint: None for _joint in fullBodyJointNms} for i in range(numOfCandidate)}
    for i in range(numOfCandidate):
        for _joint in fullBodyJointNms:
            leftSimilarPoses[i][_joint] = \
                pd.DataFrame(
                    similarFVFullBody3dPos['lhand'][_joint][:, 3*i:3*(i+1)]
                ).rename(columns={0: 'x', 1: 'y', 2: 'z'}, inplace=False)
            rightSimilarPoses[i][_joint] = \
                pd.DataFrame(
                    similarFVFullBody3dPos['rhand'][_joint][:, 3*i:3*(i+1)]
                ).rename(columns={0: 'x', 1: 'y', 2: 'z'}, inplace=False)
    
    # 6. 
    ploter = multiFigsPloter(
        skeletonDf,
        inputPos={
            'lhand': similarFV3dPos['lhand']['lhand'],
            'rhand': similarFV3dPos['rhand']['rhand'], 
            'Head': pd.DataFrame(np.zeros(similarFV3dPos['rhand']['rhand'].shape)).rename(columns={0:'x', 1:'y', 2:'z'}, inplace=False)
        },
        fullBodyPose = similarFV3dPos['lhand'], 
        blendedPose = blendedEWMAPose, 
        similarFV3dPos={
            'lhand': leftSimilarFV3dPos, 
            'rhand': rightSimilarFV3dPos
        },
        similar3dPoses={
            'lhand': leftSimilarPoses, 
            'rhand': rightSimilarPoses
        }
    )
    # ploter.plotOneFrame()
    ploter.plotMultiFrames([0, 300], 20)
    pass

if __name__ == '__main__':
    main()
    pass