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
    pass

def main():
    # 1. Read preprocessed data (only hand and feet need to be used)
    #      'lhand', 'rhand', 'LeftToe', 'RightToe'
    # 2. compute correlation coefficient of each trial
    # 3. store the correlation coefficient in a text file

    # 1. 
    subjectDirPath = 'data/swimming/125_processed/'
    trialsDirPaths = [os.path.join(subjectDirPath, i) for i in os.listdir(subjectDirPath)]
    print(trialsDirPaths)

    usedJointNms = ['lhand', 'rhand', 'LeftToe', 'RightToe']

    # 2. 
    for _trialDirPath in trialsDirPaths:
        usedJointsData = {i: None for i in usedJointNms}

        for _jointNm in usedJointNms:
            usedJointsData[_jointNm] = pd.read_csv(os.path.join(_trialDirPath, _jointNm+'.csv'))
        print(usedJointsData['lhand'].head(10))
        print(usedJointsData['lhand'].shape)
        break
    
    # 3. 
    
    pass

if __name__=='__main__':
    main()