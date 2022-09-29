'''
透過分別計算XYZ的pearson correlation coefficient, 再將它們平均.
把平均後分數低於0.1的trials記錄起來, 之後的實驗要排除這些資料.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
import time

def computeCorrCoef(handData, feetData):
    '''
    Objective:
        計算hand and feet的correlation coeficient, 
        X, Y, Z分開計算後再取平均當作最終的分數.
    :handData: (pd.DataFrame) 手的3d position time series.
                row代表某個時間點, column代表XYZ
    :feetData: (pd.DataFrame) 腳的3d position time series.
    '''
    corrcoef_x = pd.concat([handData['x'], feetData['x']], axis=1, ignore_index=True).corr()
    corrcoef_x = corrcoef_x.iloc[0, 1]

    corrcoef_y = pd.concat([handData['y'], feetData['y']], axis=1, ignore_index=True).corr()
    corrcoef_y = corrcoef_y.iloc[0, 1]

    corrcoef_z = pd.concat([handData['z'], feetData['z']], axis=1, ignore_index=True).corr()
    corrcoef_z = corrcoef_z.iloc[0, 1]

    return (corrcoef_x + corrcoef_y + corrcoef_z)/3

def compute3dCorrCoef(handData, feetData):
    '''
    Objective:
        計算hand and feet的correlation coeficient, 
        X, Y, Z視作三維time series計算一個correlation coefficient
    :handData: (pd.DataFrame) 手的3d position time series.
                row代表某個時間點, column代表XYZ
    :feetData: (pd.DataFrame) 腳的3d position time series.
    '''
    pass

def main():
    # 1. Read preprocessed data (only hand and feet need to be used)
    #      'lhand', 'rhand', 'LeftToe', 'RightToe'
    # 2. compute correlation coefficient of each trial
    # 3. store the correlation coefficient in a text file

    # 1. 
    # subjectDirPath = 'data/swimming/125_processed/'
    # subjectDirPath = 'data/swimming/126_processed/'
    # subjectDirPath = 'data/swimming/79_processed/'
    subjectDirPath = 'data/swimming/80_processed/'
    trialsDirPaths = [os.path.join(subjectDirPath, i) for i in os.listdir(subjectDirPath) if os.path.isdir(os.path.join(subjectDirPath, i))]
    print(trialsDirPaths)

    usedJointNms = ['lhand', 'rhand', 'LeftToe', 'RightToe']

    # 2. 
    trialsCorrCoef = {_trialDirPath: None for _trialDirPath in trialsDirPaths}
    for _trialDirPath in trialsDirPaths:
        usedJointsData = {i: None for i in usedJointNms}

        for _jointNm in usedJointNms:
            usedJointsData[_jointNm] = pd.read_csv(os.path.join(_trialDirPath, _jointNm+'.csv'))

        corrcoefLeft = computeCorrCoef(usedJointsData['lhand'], usedJointsData['LeftToe'])
        corrcoefRight = computeCorrCoef(usedJointsData['rhand'], usedJointsData['RightToe'])
        print(corrcoefLeft)
        print(corrcoefRight)

        trialsCorrCoef[_trialDirPath] = (corrcoefLeft+corrcoefRight)/2
        # break
    
    # 3. 
    outDf = pd.DataFrame(
        {
            'trialPath': list(trialsCorrCoef.keys()), 
            'corrcoef': list(trialsCorrCoef.values())
        }
    )
    outDf.to_csv(os.path.join(subjectDirPath,'corrcoef.csv'), index=False)

if __name__=='__main__':
    main()