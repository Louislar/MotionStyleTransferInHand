'''
Goal: after mapping後的rotation需要apply到avatar的lower body上, 
才能夠的到position資訊, 得到position資訊後才能與DB當中的motion做比較
'''

import numpy as np 
import json

if __name__=='__main__':
    # 1. 讀取檔案, 得到TPose狀態下的position資訊
    #   1.1 Hip, upper leg, lower leg, foot
    # 2. 計算lower body的bone length
    #   2.1 upper leg, lower leg兩個bone lengths
    # 3. Store bone lengths and TPose positions

    # 1. 
    saveDirPath = 'positionData/fromDB/'
    TPoseJson = None
    with open(saveDirPath+'TPose.json', 'r') as fileIn:
        TPoseJson = json.load(fileIn)['results']
    print(TPoseJson[0])
    # TODO: 只擷取一個時間點的TPose即可, 特別是lower body的部分
    pass