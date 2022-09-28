'''
使用預處理完成的3d positions time series計算feature vectors
這裡只需要計算left and right hand的feature vector就好
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
import time

def main():
    # 1. read processed data
    #       only left hand and right hand need to be read
    # 2. encode 3d position time series to feature vector
    # 3. output encoded feature vectors

    # 1. 
    subjectDirPath = 'data/swimming/80_processed/'
    trialsDirPaths = [os.path.join(subjectDirPath, i) for i in os.listdir(subjectDirPath)]
    print(trialsDirPaths)

    usedJointNms = ['lhand', 'rhand']
    pass

if __name__=='__main__':
    pass