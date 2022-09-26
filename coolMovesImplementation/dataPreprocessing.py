'''
1. Resample to 90Hz
2. Make head coordinate system
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
import time

def convertToHeadCoordSys(jointsData):
    '''
    Objective:
        將人體座標軸的原點設定在head. 
        做法是, 所有joint的3d position減去head的3d position
    :jointsData: 每個joint的3d positions time series
    '''
    headCoordJointsData = {_jointNm: None for _jointNm in jointsData.keys()}
    for _jointNm, _3dPosSeries in jointsData.items():
        # 注意, head自己的座標要最後才能更動. 
        if _jointNm != 'Head':
            headCoordJointsData[_jointNm] = _3dPosSeries - jointsData['Head']
    jointsData['Head'] = jointsData['Head'] - jointsData['Head']
    return headCoordJointsData

def resampleTo90Hz(jointsData, frameCount):
    '''
    Objective: 
        假設原本是120Hz的資料, 將之轉換成90Hz的資料. 等同移除1/4的資料.
        做法是, 每4筆資料移除一筆. 
    :jointsData: 每個joint的3d positions time series
    '''
    resampledJointsData = {_jointNm: None for _jointNm in jointsData.keys()}
    # 2, 6, 10, 14, ... 都是要被移除的資料. 每四筆資料移除第2筆.
    removeIdx = [i for i in range(frameCount) if (i+2)%4==0]
    for _jointNm, _3dPosSeries in jointsData.items():
        # print(_jointNm)
        # print(_3dPosSeries.shape)
        # print(_3dPosSeries.head(10))
        resampledJointsData[_jointNm] = _3dPosSeries.iloc[removeIdx, :]
    return resampledJointsData

def preprocessData(dataDirPath):
    '''
    Objective:
        執行所有preprocessing的functions
    :dataDirPath: 被preprocess的trial
    '''
    # 1. read data
    # 2. resample data to 90Hz
    # 3. convert data to head coordinate system
    # 4. return preprocessed data

    # 1.
    dataFiles = os.listdir(dataDirPath)
    dataFilesWithoutExt = [os.path.splitext(_file)[0] for _file in dataFiles]
    dataFilesPaths = [os.path.join(dataDirPath, _file) for _file in dataFiles]
    # print(dataFiles)
    # print(dataFilesWithoutExt)
    # print(dataFilesPaths)
    
    jointsNMs = dataFilesWithoutExt
    jointsData = {_nm: None for _nm in jointsNMs}
    for i in range(len(jointsNMs)):
        jointsData[jointsNMs[i]] = pd.read_csv(dataFilesPaths[i])

    # 2. 
    resampledData = resampleTo90Hz(jointsData, jointsData['Hips'].shape[0])

    # 3. 
    headCoordData = convertToHeadCoordSys(resampledData)

    return headCoordData

def main():
    # 1. Parse directories
    # 2. preprocess data
    # 3. output data

    subjectDirPath = 'data/swimming/125_parsed/'
    trialsDirPaths = [os.path.join(subjectDirPath, i) for i in os.listdir(subjectDirPath)]
    print(trialsDirPaths)

    trialsData = {i for i in trialsDirPaths}
    for _trialDirPath in trialsDirPaths:
        preprocessData(_trialDirPath)
        # TODO: finish this section

    dataDirPath = 'data/swimming/125_parsed/125_01/'
    preprocData = preprocessData(dataDirPath)
    # print(preprocData['Hips'].shape)
    

if __name__=='__main__':
    main()
    pass