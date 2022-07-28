'''
Goal: 
1. Rotation mapping需要的bBSpline資訊儲存成檔案備用, 
    儲存的檔案會是sample很多點的結果
    這部分實作在原本的rotationAnalysis.py當中
2. 將stream rotation data做mapping的function
'''

import numpy as np 
from scipy.interpolate import splev
import json
import pickle
import time 
from rotationAnalysis import rotationJsonDataParser

usedJointIdx = [['x','z'], ['x'], ['x','z'], ['x']]
jointCount = 4
# left front kick best strategy: TFFFTT
mappingStrategy = [['x'], [], ['z'], ['x']]  # 設計的跟usedJointIdx相同即可, 缺一些element而已
tmpRotations = \
    [{aAxis: np.zeros(5) for aAxis in mappingStrategy[aJoint]} for aJoint in range(len(mappingStrategy))]  # For inc dec判斷使用, 當作queue, 每個joint/axis都要有獨立的一個queue


def samplePointsFromBSpline(BSplineParam, nSamplePoints):
    '''
    Goal: 從給定的BSpline當中取特定數量的sample points
    Input: 
    :BSplineParam: BSpline parameters from splprep()
    :nSamplePoints: number of sample points
    Output: 
    :samplePoints: 含有兩個row的2D array, 第一個row是hand rotation array, 
        第二個row是mapping後的body rotation array
    '''
    return np.array(splev(np.linspace(0, 1, nSamplePoints), BSplineParam))

def convertJsonToStreamingData(ReadInJsonData, jointCount):
    '''
    Goal: 將原始讀取的json資料, 轉換成streaming data, 
            特別是單一時間點的所有joint資訊統整成一個array
    '''

    pass

def estimateIncDecSeg(shortRotArr):
    '''
    Goal: 給定一小段落的rotation資訊, 
        判斷使用的mapping方法是increase seg還是decrease segment
    Input: 
    :shortRotArr: 小段落資訊, 由array構成

    Output: 
    :IncDec: 0 -> decrease, 1 -> increase segment
    '''
    pass

def rotationMappingStream(rotationData, mappingFuncSamplePts, mappingStrategy):
    '''
    Goal: 將streaming rotation data做mapping, 
            包含要real time估計increase or decrease segment
            TODO: 需要能夠暫存5個rotation資訊的variables, 使用queue會是不錯的選擇
    Input:
    :rotationData: 單一時間點的rotation資訊
    :mappingFuncSamplePts: mapping function的sample points
    :mappingStrategy: 需要被mapping的joint/axis

    Output: 
    ::
    '''
    # Define a queue(array) for storing temporary rotations
    global tmpRotations
    print(rotationData)
    for aJoint in range(len(mappingStrategy)):
        for aAxis in mappingStrategy[aJoint]:
            tmpRotations[aJoint][aAxis][:-1] = tmpRotations[aJoint][aAxis][1:]
            tmpRotations[aJoint][aAxis][-1] = rotationData[aJoint][aAxis]
    # 1. increase or decrease
    # TODO: finish this
    estimateIncDecSeg(tmpRotations)
    # 2. roataion mapping

if __name__=='__main__':
    # 1. Read in BSpline sample points
    # 2. Read in hand rotations and convert it as streaming data 
    #       2.1 [需要計時; 進入real time執行階段] Need to adjust rotation's max as 180, min as -180
    # 3. Find closest rotation in BSpline sample points and 
    #       the corresponding body rotation is the answer
    #       *Note that do not map bad mapping axes
    #       3.1 TODO: Need to Dynamiclly decide the streaming data 
    #           is in increasing segment or decreasing segment
    # 4. Save the mapping results(body rotations)(儲存檔案不用計時)
    #       4.1 [需要計時; 結束 real time執行階段] Need to reverse the rotation to max as 360, min as 0

    # 1. 
    BSplineSamplePoints = [
        [{aAxis: None for aAxis in usedJointIdx[aJoint]} for aJoint in range(len(usedJointIdx))], 
        [{aAxis: None for aAxis in usedJointIdx[aJoint]} for aJoint in range(len(usedJointIdx))]
    ]
    saveDirPath = 'preprocBSpline/leftFrontKick/'
    for aJoint in range(len(usedJointIdx)):
        for aAxis in usedJointIdx[aJoint]:
            for i in range(2):
                BSplineSamplePoints[i][aJoint][aAxis] = \
                    np.load(saveDirPath+'{0}.npy'.format(str(i)+'_'+aAxis+'_'+str(aJoint)))
    # print(BSplineSamplePoints[0][0]['x'])

    # 2. 
    HandRotSaveDirPath = './HandRotationOuputFromHomePC/leftFrontKick.json' # 從Unity output出來的
    handJointsRotations = None
    with open(HandRotSaveDirPath, 'r') as fileOpen: 
        handJointsRotations=json.load(fileOpen)
        handJointsRotations = handJointsRotations['results']
    ## TODO: [考慮看看要不要做, 不知道有沒有加速的效果]Convert to streaming data, 特別是將每一個時間點的資料都轉換成array
    # handJointsRotations = convertJsonToStreamingData(handJointsRotations, jointCount)
    ## Set max to 180, min to -180
    timeCount = len(handJointsRotations)
    print('time count: ', timeCount)
    set180TimeLaps = np.zeros(timeCount)
    for t in range(timeCount):
        for aJoint in range(len(mappingStrategy)):
            for aAxis in mappingStrategy[aJoint]:
                tmp = handJointsRotations[t]['data'][aJoint][aAxis]
                handJointsRotations[t]['data'][aJoint][aAxis] = \
                    tmp-360 if tmp>180 else tmp
        set180TimeLaps[t] = time.time()
    set180Cost = set180TimeLaps[1:] - set180TimeLaps[:-1]
    print('k similar search avg time: ', np.mean(set180Cost))
    print('k similar search time std: ', np.std(set180Cost))
    print('k similar search max time cost: ', np.max(set180Cost))
    print('k similar search min time cost: ', np.min(set180Cost))

    # 3. 
    # 3.1 Decide increase segment or decrease segment
    #       使用小段落rotations估計increase or decrease segment
    #       小段落暫時設定為5個連續時間點的roations
    # 3.2 rotation mapping
    for t in range(timeCount):
        rotationMappingStream(handJointsRotations[t]['data'], BSplineSamplePoints, mappingStrategy)
        break

if __name__=='__main01__':
    # 1. Read in the pre compute BSpline parameter (from rotationAnalysis.py)
    saveDirPath = 'preprocBSpline/leftFrontKick/'
    BSplineParam = [
        [{aAxis: None for aAxis in usedJointIdx[aJoint]} for aJoint in range(len(usedJointIdx))], 
        [{aAxis: None for aAxis in usedJointIdx[aJoint]} for aJoint in range(len(usedJointIdx))]
    ]
    for aJoint in range(len(usedJointIdx)):
        for aAxis in usedJointIdx[aJoint]:
            for i in range(2):
                with open(saveDirPath+'{0}.pickle'.format(str(i)+'_'+aAxis+'_'+str(aJoint)), 'rb') as inPickle:
                    BSplineParam[i][aJoint][aAxis] = pickle.load(inPickle)
    print(BSplineParam[0][0]['x'])
    # 2. Sample points from the pre compute BSpline parameter 
    BSplineSamplePoints = [
        [{aAxis: None for aAxis in usedJointIdx[aJoint]} for aJoint in range(len(usedJointIdx))], 
        [{aAxis: None for aAxis in usedJointIdx[aJoint]} for aJoint in range(len(usedJointIdx))]
    ]
    for aJoint in range(len(usedJointIdx)):
        for aAxis in usedJointIdx[aJoint]:
            for i in range(2):
                BSplineSamplePoints[i][aJoint][aAxis] = \
                    samplePointsFromBSpline(BSplineParam[i][aJoint][aAxis], 1000)
    # print(samplePointsFromBSpline(BSplineParam[0][0]['x'], 100))
    # 3. Save the sample points
    saveDirPath = 'preprocBSpline/leftFrontKick/'
    for aJoint in range(len(usedJointIdx)):
        for aAxis in usedJointIdx[aJoint]:
            for i in range(2):
                # np.save(
                #     saveDirPath+'{0}.npy'.format(str(i)+'_'+aAxis+'_'+str(aJoint)), 
                #     BSplineSamplePoints[i][aJoint][aAxis]
                # )
                pass
                