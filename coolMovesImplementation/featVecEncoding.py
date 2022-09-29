'''
使用預處理完成的3d positions time series計算feature vectors
這裡只需要計算left and right hand的feature vector就好
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
import time

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

def extract3dPosFromRollWin(rollWinData):
    '''
    Objective:
        從rolling windows當中, 取得每個window對應的3d position
        對應的3d position是window當中最後一個3d position
    :rollWinData: (list of pd.DataFrame) single joint's 3d position time series, list of rolling windows
    '''
    corresponding3dPos = []
    for _win in rollWinData:
        corresponding3dPos.append(_win.iloc[-1, :])
    corresponding3dPos = pd.concat(corresponding3dPos, axis=1, ignore_index=True)
    return corresponding3dPos.T

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
    :rollWinData: (list of pd.DataFrame) single joint's 3d position time series, list of rolling windows 
    :velData: (list of pd.DataFrame) single joint's velocity data
    :accData: (list of pd.DataFrame) single joint's acceleration data
    :augRatio: target augmentation ratios of velocity
    '''
    # print(len(rollWinData))
    # print(len(velData))
    # print(len(accData))
    # TODO: save new augment data into these dicts
    augRollWinData = {_augRatio:[] for _augRatio in augRatio}
    augVelData = {_augRatio:[] for _augRatio in augRatio}
    augAccData = {_augRatio:[] for _augRatio in augRatio}
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
            augRollWinData[_augRatio].append(_newRollWinData)
            augVelData[_augRatio].append(_newVelData)
            augAccData[_augRatio].append(_newAccData)

    return augRollWinData, augVelData, augAccData

def convertRollWinToFeatVec(rollWinData):
    '''
    Objective: 
        將list of rolling window data轉換成feature vectors
    :rollWinData: (list of df.DataFrame) list of windowed data with XYZ 3 columns
    '''
    featVecs = []
    for _win in rollWinData:
        # print(_win)
        _featVec = pd.concat([_win['x'], _win['y'], _win['z']], ignore_index=True)
        featVecs.append(_featVec)
    featVecs = pd.concat(featVecs, axis=1, ignore_index=True)
    # print(rollWinData[:2])
    # print(featVecs)
    return featVecs.T

def xyzToFeatVec(aJoint3dPos, winSize, overlapSize, augRatios):
    '''
    Objective:
        將joint的3d position time series轉換成feature vectors
        假設每一筆資料的間隔時間相同.
    :aJoint3dPos: 單一joint的3d position time series
    '''
    # 1. Rolling window resample
    # 1.1 TODO: 額外紀錄每個window對應的3d position, 取最後一個數值
    # 2. velocity and acceleration computation
    # 3. augmentation with velocity 
    # 4. convert to feature vectors
    # 5. return 

    # 1. 
    rollingWinData = rollingWinResample(aJoint3dPos, winSize, overlapSize)
    corresponding3dPos = extract3dPosFromRollWin(rollingWinData)

    # 2. 
    velData, AccData = computeVelAndAcc(rollingWinData, winSize)

    # 3. 
    augRollWinData, augVelData, augAccData = augmentationWithVelAcc(rollingWinData, velData, AccData, augRatios)
    # print(len(augRollWinData[0.5]))
    # print(len(augVelData[0.5]))
    # print(len(augAccData[0.5]))

    # 4. 
    augFeatVecs = {_augRatio: None for _augRatio in augRatios}
    for _augRatio in augRatios:
        augRollWinFeatVec = convertRollWinToFeatVec(augRollWinData[_augRatio])
        augVelFeatVec = convertRollWinToFeatVec(augVelData[_augRatio])
        augAccFeatVec = convertRollWinToFeatVec(augAccData[_augRatio])
        # 將三者合併
        augFeatVecs[_augRatio] = pd.concat(
            [augRollWinFeatVec, augVelFeatVec, augAccFeatVec], 
            axis=1, 
            ignore_index=True
        )
        # print(augFeatVecs[_augRatio])
    
    # 5. 
    allFeatVecs = pd.concat(augFeatVecs.values(), axis=0, ignore_index=True)
    # print(allFeatVecs)
    return allFeatVecs, corresponding3dPos

def main():
    # 1. read processed data
    #       only left hand and right hand need to be read
    # 2. encode 3d position time series to feature vector
    # 3. output and store encoded feature vectors

    # 0. parameters
    windowSize = 9
    overlapSize = 5
    augRatios = [0.5, 0.75, 1, 1.25, 1.5]

    # 1. 
    # subjectDirPath = 'data/swimming/79_processed/'
    # subjectDirPath = 'data/swimming/80_processed/'
    # subjectDirPath = 'data/swimming/125_processed/'
    subjectDirPath = 'data/swimming/126_processed/'
    trialsDirPaths = [os.path.join(subjectDirPath, i) for i in os.listdir(subjectDirPath) if os.path.isdir(os.path.join(subjectDirPath, i))]
    print(trialsDirPaths)

    usedJointNms = ['lhand', 'rhand']
    
    # 2. 
    trialsFeatVecs = {_trialDirPath: {_jointNm: None for _jointNm in usedJointNms} for _trialDirPath in trialsDirPaths}
    trials3dPos = {_trialDirPath: {_jointNm: None for _jointNm in usedJointNms} for _trialDirPath in trialsDirPaths}
    for _trialDirPath in trialsDirPaths:
        usedJointsData = {i: None for i in usedJointNms}

        for _jointNm in usedJointNms:
            usedJointsData[_jointNm] = pd.read_csv(os.path.join(_trialDirPath, _jointNm+'.csv'))
            # compute feature vectors
            _featVecs, _corresponding3dPos = trialsFeatVecs[_trialDirPath][_jointNm] = xyzToFeatVec(
                usedJointsData[_jointNm], 
                windowSize, 
                overlapSize, 
                augRatios
            )
            trialsFeatVecs[_trialDirPath][_jointNm] = _featVecs
            trials3dPos[_trialDirPath][_jointNm] = _corresponding3dPos
        #     break
        # break

    # 3. TODO: 儲存成3d position以及kd tree形式檔案 (兩種檔案分開儲存)
    #      先將所有feature vector儲存成DataFrame格式, 最後再將所有DataFrame整合成單一一個kdtree
    pos3dDirPath = os.path.dirname(subjectDirPath).replace('processed', 'pos3d')
    featVecDirPath = os.path.dirname(subjectDirPath).replace('processed', 'featVecs')
    print(pos3dDirPath)
    print(featVecDirPath)
    for _trialDirPath in trialsDirPaths:
        for _jointNm in usedJointNms:
            _trialNm = os.path.basename(_trialDirPath)
            _pos3dDirPath = os.path.join(*[pos3dDirPath, _trialNm])
            _featVecDirPath = os.path.join(*[featVecDirPath, _trialNm])
            if not os.path.isdir(_pos3dDirPath):
                os.makedirs(_pos3dDirPath)
            if not os.path.isdir(_featVecDirPath):
                os.makedirs(_featVecDirPath)
            # print(_pos3dDirPath)
            # print(_featVecDirPath)
            trialsFeatVecs[_trialDirPath][_jointNm].to_csv(
                os.path.join(_featVecDirPath, _jointNm+'.csv'), 
                index=False
            )
            trials3dPos[_trialDirPath][_jointNm].to_csv(
                os.path.join(_pos3dDirPath, _jointNm+'.csv'), 
                index=False
            )

if __name__=='__main01__':
    main()
    pass