'''
使用預處理完成的3d positions time series計算feature vectors
這裡只需要計算left and right hand的feature vector就好
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
import time

from dataPreprocessing import resampleTo90Hz

def rollingWinResample(aJoint3dPos, winSize, overlapSize):
    '''
    Objective:
        將單一joint的3d position time series轉換成多個rolling window構成的資料
        rolling window的資料排列方式|XXXXX...|YYYYY...|ZZZZZ...|
        注意rollling window取的是該時間點以前的資料, 
        所以該時間點的資料排在window的最後面
    :aJoint3dPos: 單一joint的3d position time series
    '''
    # print(aJoint3dPos.head(20))
    # print(aJoint3dPos.shape)
    rollingWinData = []
    for i, a in enumerate(aJoint3dPos.rolling(window=winSize)):
        if not i < winSize-1:
            rollingWinData.append(a)
    # 每拿一筆資料, 就要捨棄掉未來的winSize-overlapSize-1筆資料
    # 我只能拿index相差winSize-overlapSize的資料
    resampleIdx = list(range(0, len(rollingWinData), winSize-overlapSize))
    rollingWinData = [rollingWinData[i] for i in resampleIdx]

    return rollingWinData
    # TODO: (放到後面再進行) 3d positions encode成vector
    featVecData = []
    for _rollWinData in rollingWinData:
        _featVec = pd.concat(
            [_rollWinData['x'], _rollWinData['y'], _rollWinData['z']], 
            ignore_index=True
        )
        featVecData.append(_featVec)
    # print(len(featVecData))
    return featVecData

def computeVelAndAcc(aJointRollWin, winSize):
    '''
    Objective:
        計算單一joint的feature vector time series的velocity以及acceleration.
        並且, 把兩個數值接續在feature vector後方
    :aJointRollWin: 單一joint在多個時間下的rolling window (由rollingWinResample()計算)
    :winSize: window size
    '''
    rollWinVelData = []
    rollWinAccData = []
    for _aWin in aJointRollWin:
        _vel = _aWin.diff().iloc[1:, :]
        _acc = _vel.diff().iloc[1:, :]
        rollWinVelData.append(_vel)
        rollWinAccData.append(_acc)
        # print(_aWin)
        # print(_vel)
        # print(_acc)
    return rollWinVelData, rollWinAccData

def augmentationWithVelAcc(rollWinData, velData, accData, augRatio):
    '''
    Objectives:
        利用增加velocity的數值改變rolling window內的3d position. 
        但是, 最終的終點位置是不動的.
        Acceleration也會改變數值, 其實是相同意思.
        總之加速度兩倍 = 速度兩倍 
    :rollWinData: single joint's 3d position time series
    :velData: single joint's velocity data
    :accData: single joint's acceleration data
    :augRatio: target augmentation ratios of velocity
    '''
    print(len(rollWinData))
    print(len(velData))
    print(len(accData))
    # TODO: save new augment data into these dicts
    augRollWinData = {_augRatio for _augRatio in augRatio}
    augVelData = {_augRatio for _augRatio in augRatio}
    augAccData = {_augRatio for _augRatio in augRatio}
    for i in range(len(rollWinData)):
        for _augRatio in augRatio:
            
            _newAccData = accData[i]*_augRatio
            # _newAccData = accData[i]
            _newAccDataCumsum = _newAccData[::-1].cumsum()[::-1]
            _newVelData = velData[i].iloc[-1, :] - _newAccDataCumsum
            _newVelData = pd.concat([_newVelData, velData[i].iloc[-1:, :]])
            _newVelDataCumsum = _newVelData[::-1].cumsum()[::-1]
            _newRollWinData = rollWinData[i].iloc[-1, :] - _newVelDataCumsum
            _newRollWinData = pd.concat([_newRollWinData, rollWinData[i].iloc[-1:, :]])
            # print(accData[i])
            # print(velData[i])
            # print(rollWinData[i])
            # print(_newAccData)
            # print(_newVelData)
            # print(_newRollWinData)
            break
        break
    pass

def xyzToFeatVec(aJoint3dPos, winSize, overlapSize, augRatios):
    '''
    Objective:
        將joint的3d position time series轉換成feature vectors
        假設每一筆資料的間隔時間相同.
    :aJoint3dPos: 單一joint的3d position time series
    '''
    # 1. Rolling window resample
    # 2. velocity and acceleration computation
    # 3. augmentation with velocity 
    # 4. return 

    # 1. 
    rollingWinData = rollingWinResample(aJoint3dPos, winSize, overlapSize)

    # 2. 
    velData, AccData = computeVelAndAcc(rollingWinData, winSize)

    # 3. 
    augmentationWithVelAcc(rollingWinData, velData, AccData, augRatios)
    pass

def main():
    # 1. read processed data
    #       only left hand and right hand need to be read
    # 2. encode 3d position time series to feature vector
    # 3. output encoded feature vectors

    # 0. parameters
    windowSize = 9
    overlapSize = 5
    augRatios = [0.5, 0.75, 1, 1.25, 1.5]

    # 1. 
    subjectDirPath = 'data/swimming/79_processed/'
    trialsDirPaths = [os.path.join(subjectDirPath, i) for i in os.listdir(subjectDirPath) if os.path.isdir(os.path.join(subjectDirPath, i))]
    print(trialsDirPaths)

    usedJointNms = ['lhand', 'rhand']
    
    # 2. 
    trialsFeatVecs = {_trialDirPath: {_jointNm: None for _jointNm in usedJointNms} for _trialDirPath in trialsDirPaths}
    for _trialDirPath in trialsDirPaths:
        usedJointsData = {i: None for i in usedJointNms}

        for _jointNm in usedJointNms:
            usedJointsData[_jointNm] = pd.read_csv(os.path.join(_trialDirPath, _jointNm+'.csv'))
            # compute feature vectors
            trialsFeatVecs[_trialDirPath][_jointNm] = xyzToFeatVec(
                usedJointsData[_jointNm], 
                windowSize, 
                overlapSize, 
                augRatios
            )
            break
        break


if __name__=='__main__':
    main()
    pass