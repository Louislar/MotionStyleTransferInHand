'''
建構rotation sequence, 
按照原本的finger rotation的曲線做interpolation, 
若是最大最小值出現變動, 再使用min max normalization做修正 
'''
import json 
import pandas as pd 
import numpy as np 
from scipy import interpolate

def jsonToDf(lmJson):
    '''
    land mark json轉換成dataframe
    column為x, y, z
    '''
    timeCount = len(lmJson)
    numOfJoints = len(lmJson[0]['data'])
    axisNms = ['x', 'y', 'z']
    timeSeries = {_joint: {_axis: [] for _axis in axisNms} for _joint in range(numOfJoints)}
    for _joint in range(numOfJoints):
        for t in range(timeCount):
            for _axis in axisNms:
                timeSeries[_joint][_axis].append(
                    lmJson[t]['data'][_joint][_axis]
                )
    jointsSeries = {
        _joint: pd.DataFrame(timeSeries[_joint]) for _joint in range(numOfJoints)
    }
    # print(jointsSeries[1])

    return jointsSeries

def minmaxRescaling(seq: np.array, targetMin, targetMax):
    return targetMin + ( (targetMax - targetMin) * (seq - min(seq)) / (np.max(seq) - np.min(seq)) )

def readRotSeq():
    pass

def main():
    '''
    param: 
    :: rotation raw資料檔案
    :: 事前決定的目標rotation區間index
    :: 事前決定的目標rotation數值範圍
    :: 事前決定的目標rotation數量

    process: 
    1. 讀取rotation raw資料檔案
    2. 根據事前決定好的區間, 切割目標區間rotation數值
    3. 透過minmax rescaling的方式, 將目標區間rotation數值的範圍修正為目標數值範圍
    4. 透過linear interpolation的方式將rotation數量修正成目標數量
    '''
    handRotFilePath = '../HandRotationOuputFromHomePC/leftFrontKickStream.json' 
    targetInterval = [867, 936] # [front index, rear index]
    # (min value, max value); ['0x', '0z', '1x']
    # targetValInterval = [(-23.647, 19.315), (6, 50), (None, None)]  
    targetValInterval = [(None, None), (None, None), (None, None)]  # No rescaling 
    valIntervalInd = [[0, 'x'], [0, 'z'], [1, 'x']]
    targetRotCount = 100

    # 1. 
    handRotJson = None
    with open(handRotFilePath, 'r') as fileOpen: 
        handRotJson=json.load(fileOpen)
    # 2. 
    handRotJson = handRotJson[targetInterval[0]: targetInterval[1]+1] 
    handRotDf = jsonToDf(handRotJson)
    print(handRotDf.keys())
    print(handRotDf[0])
    # 3. 
    for _ind, _interval in zip(valIntervalInd, targetValInterval):
        if _interval[0] is None:
            continue
        _min = _interval[0]
        _max = _interval[1] 
        _seq = minmaxRescaling(handRotDf[_ind[0]][_ind[1]].values, _min, _max)
        handRotDf[_ind[0]][_ind[1]] = _seq
    # 4. 
    rotSeqResult = {}
    for _ind in valIntervalInd:
        _seq = handRotDf[_ind[0]][_ind[1]].values
        _f = interpolate.interp1d(np.arange(0, len(_seq)), _seq)
        _xnew = np.linspace(0, len(_seq)-1, targetRotCount)
        _seqNew = _f(_xnew)
        # TODO: 儲存到輸出格式當中
        pass
    
    pass

if __name__ == '__main__':
    # TODO: 需要把interpolation的結果與interpolate之前的數值化在一起觀察
    #       interpolation的結果有可能異常 
    main()
    pass