import numpy as np
import pandas as pd
import json
import enum
from sklearn.metrics.pairwise import euclidean_distances

'''
目標: 讀取position隨時間變動的資料，計算出對應的features，並且比較兩個motion間的相似性
計算features的方法，目前都是按照CoolMove的方法計算position features
'''

class jointsNames(enum.IntEnum):
    LeftUpperLeg = 0
    LeftLowerLeg = 1
    LeftFoot = 2
    RightUpperLeg = 3
    RightLowerLeg = 4
    RightFoot = 5
    Hip = 6


def positionJsonDataParser(jsonDict: dict, jointCount: int):
    '''
    target format: 
    x list:[x1, x2, x3, ...]
    knee: {x list, y list, z list}
    left upper leg, left knee, right upper leg, right knee
    '''
    timeSeries=jsonDict['results']
    parsedPositionData=[{'x': [], 'y': [], 'z': []} for i in range(jointCount)]
    for jointIdx in range(jointCount):
        for oneData in timeSeries:
            parsedPositionData[jointIdx]['x'].append(oneData['data'][jointIdx]['x'])
            parsedPositionData[jointIdx]['y'].append(oneData['data'][jointIdx]['y'])
            parsedPositionData[jointIdx]['z'].append(oneData['data'][jointIdx]['z'])
    return parsedPositionData

def positionDataToPandasDf(parsedPosData, jointCount: int):
    '''
    Convert parsed position data to pandas DataFrame
    Columns(left to right): | joint 1 x | joint 1 y | joint 1 z | joint 2 x | ... |
    Rows(Top to bottom): | 1st sample pt | 2nd sample pt | ... |
    '''
    posDf = pd.DataFrame()
    for jointIdx in range(jointCount):
        for k, _data in parsedPosData[jointIdx].items():
            posDf['{0}_{1}'.format(jointIdx, k)] = _data
    return posDf

def setHipAsOrigin(posDf, jointCount: int):
    '''
    Set hip position as origin (0, 0, 0)
    '''
    axesStr = ['x', 'y', 'z']
    for i in range(jointCount):
        for _axis in axesStr:
            posDf.loc[:, '{0}_'.format(i)+_axis] = \
                posDf['{0}_'.format(i)+_axis] - posDf['{0}_'.format(jointsNames.Hip)+_axis]
    return posDf

def rollingWindowSegRetrieve(posDf, winSize: int, jointCount: int):
    '''
    Extract the window segment values
    Rolling window之間的重疊區域只有150ms, 5筆資料, 但是現在是除了兩端點之外的部分都重疊的作法
    CoolMove: 每個時間點左右手各一個feature vector，每個feature vector都是由一個window內的position資料構成
    Ours: 每一個joint都有自己的一個DataFrame紀錄feature vectors
    ref: https://stackoverflow.com/questions/70670079/get-indexes-of-pandas-rolling-window
    '''
    jointWindowDfs = [] # window segments of each joint
    axesStr = ['x', 'y', 'z']

    curJoint= 0 
    curAxis = axesStr[0]
    aJointAxisDf = posDf['{0}_'.format(curJoint)+curAxis]
    print(aJointAxisDf.head(10))

    for curJoint in range(jointCount): 
        aJointDfs = []
        for curAxis in axesStr:
            aJointAxisDf = posDf['{0}_'.format(curJoint)+curAxis]
            rollingWins = [i.reset_index(drop=True, inplace=False) for i in aJointAxisDf.rolling(winSize) if len(i)>=winSize]
            rollingWins = [_win for i, _win in enumerate(rollingWins) if i % (winSize/2) == 0]# 重疊區域目前設定為window size的一半
            # print(len(rollingWins))
            # print(rollingWins[0])
            # print(rollingWins[1])
            # Each window to a time point
            tmpDf = pd.concat(rollingWins, axis=1).T.reset_index(drop=True, inplace=False)
            # print(tmpDf)
            aJointDfs.append(tmpDf)
        jointWindowDfs.append(
            pd.concat(aJointDfs, axis=1)
        )
    return jointWindowDfs

    # TODO: 注意rolling()的參數，檢查有無包含最右邊或是最左邊端點
    # rollingWins = [i.reset_index(drop=True, inplace=False) for i in aJointAxisDf.rolling(winSize) if len(i)>=winSize]
    # print(rollingWins[0])
    # print(rollingWins[1])
    # # Each window to a time point
    # tmpDf = pd.concat(rollingWins, axis=1).T.reset_index(drop=True, inplace=False)
    # print(tmpDf)
    # TODO: X, Y, Z三個軸的position是同一個feature vector還是儲存成多個feature vectors? Ans: 是同一個
    # 可能要從論文的reference當中找答案了


def computeVelocityAndAcceleration(aJointWindowSegDf, winSize):
    '''
    compute velocity and acceleration of a single joint(速度還會分成X, Y, Z，還是一個joint只會有一個速度) 
    應該是X, Y, Z個一個速度，這樣才能夠將速度的方向加入feature vector當中
    是windows的平均速度，還是每一個資料點的瞬時速度

    Input:
    :aJointWindowSegDf: 單一joint使用window function切割後的DataFrame, dimension為(時間點數量, window size*3)
    X, Y, Z的排序方式為, window size個X, window size個Y, window size個Z
    '''
    # print(aJointWindowSegDf)
    # print(aJointWindowSegDf.iloc[0, :])
    # print(aJointWindowSegDf.iloc[0, :][0])

    avgVel = [None]*3
    avgAcc = [None]*3
    windowsVelAccSrs = []
    for i in range(aJointWindowSegDf.shape[0]): # iterate all the windows
        aWindowVel = []
        for _axis in range(3):
            for t in range(winSize-1):  # time point in each window
                # print(aJointWindowSegDf.iloc[i, :][t])
                # print(aJointWindowSegDf.iloc[i, :][t].iloc[_axis])
                # print(aJointWindowSegDf.iloc[i, :][t+1])
                # print(aJointWindowSegDf.iloc[i, :][t+1].iloc[_axis])
                aWindowVel.append(
                    aJointWindowSegDf.iloc[i, :][t+1].iloc[_axis] - aJointWindowSegDf.iloc[i, :][t].iloc[_axis]
                )

        # Use velocity to compute acceleration
        # 有27個速度，應該只有24個加速度，因為X, Y, Z軸是分開計算的
        aWindowAcc = [aWindowVel[i+1]-aWindowVel[i] for i in range(len(aWindowVel)-1) if i % (winSize-1) != 0 or i == 0]
        aWindowVelAcc = aWindowVel + aWindowAcc
        windowsVelAccSrs.append(
            pd.Series(aWindowVelAcc)
        )
    windowsVelAccDf = pd.DataFrame(windowsVelAccSrs)
    return pd.concat([aJointWindowSegDf, windowsVelAccDf], axis=1)


def velocityAccelerationAugmentation(aJointWindowSegDf, winSize, augSpeeds):
    '''
    Augment the feature vectors by changing the velocities and accelerations
    速度改變後, 對應的position data也要改變
    目前的想法: 使用window內最後一個motion作為起始點, 將每一個Velocity乘上相對應的倍數, 
    最後再利用這些調整後的velocity, 從最後一個motion往前推所有window內的motion

    Input: 
    :aJointWindowSegDf: 加入速度與加速度的window feature vectors
    :augSpeeds: 調整的目標速度有哪一些, e.g. [0.5, 0.7, 1.0, 1.3, 1.5]
    以上等同 [減速50%, 減速30%, 原速, ..., 加速50%]
    '''
    multiSpeedWinSegs = []  # 各種Speed Ratio調整後的feauture vectors, each element(DataFrame) in the list is a specific ratio
    velocityIdxStart = winSize*3
    accelerationIdxStart = velocityIdxStart + (winSize-1)*3
    for curSpeedRatio in augSpeeds:
        newVelocitySr = aJointWindowSegDf.iloc[:, velocityIdxStart:accelerationIdxStart]*curSpeedRatio
        newAccelerationSr = aJointWindowSegDf.iloc[:, accelerationIdxStart:]*curSpeedRatio
        # back-propagate the velocity to the position data
        # Velocity的back-propagate, 需要X, Y, Z三個軸的資料分開處理, 並且要加上最原始的motion在最後面, 整體也要reverse
        axesPosData = []
        for _posStartIdx, _velStartIdx in [(0, 0), (winSize, winSize-1), (winSize*2, (winSize-1)*2)]:
            accumulateVelSr = newVelocitySr.iloc[:, _velStartIdx:_velStartIdx+winSize-1].iloc[:, ::-1].cumsum(axis=1)
            lastPos = aJointWindowSegDf.iloc[:, _posStartIdx+winSize-1] # 該axis的最後一個position資料
            newPosData = accumulateVelSr.add(lastPos, axis=0)
            newPosData = pd.concat([lastPos, newPosData], axis=1)
            axesPosData.append(newPosData)
            # print(lastPos)
            # print(newPosData.iloc[:, ::-1])
            # print(aJointWindowSegDf.iloc[:, _posStartIdx:_posStartIdx+winSize])
        newPosVelAcc = pd.concat(axesPosData+[newVelocitySr, newAccelerationSr], axis=1)
        multiSpeedWinSegs.append(newPosVelAcc)
    return multiSpeedWinSegs

def positionDataPreproc(posDf, posJointCount, winSize, ifAug, augRatio):
    '''
    Full position data preprocessing pipeline
    '''
    # - Make Hip joint as origin
    originAdjPosDBDf = setHipAsOrigin(posDf, posJointCount)

    # - Get the windowed segments of each joints' position data(10 data in a row)
    jointsWindowSegs = rollingWindowSegRetrieve(originAdjPosDBDf, winSize, posJointCount)

    # - Compute the velocity and acceleration
    jointsWindowSegsWithVelAcc = []
    for i in range(len(jointsWindowSegs)): 
        jointsWindowSegsWithVelAcc.append(
            computeVelocityAndAcceleration(jointsWindowSegs[i], winSize)
        )

    if ifAug is False:
        return jointsWindowSegsWithVelAcc

    # - Augment feature vectors 
    jointsFeatureVecAug = []
    for i in range(6):
        multiSpeedRatioFeatureVec = \
            velocityAccelerationAugmentation(jointsWindowSegsWithVelAcc[i], winSize, augRatio)
        # 組合不同速度的feature vector, 一個joint最終只會有一個DataFrame，當中包含不同速度產生的feature vectors
        jointsFeatureVecAug.append(pd.concat(multiSpeedRatioFeatureVec, axis=0))    
    
    return jointsFeatureVecAug

def kSimilarfeatureVec(aJointDBFeatVecs, aJointMappedFeatVecs):
    '''
    Find K similar feature vectors with each feature vector in the DB motion, 
    also compute and return the l2 distance of them 
    '''
    # TODO: finish this section, Move the code below in the main function to here
    euclidean_distances(aJointDBFeatVecs, aJointMappedFeatVecs)
    pass

if __name__=='__main__':
    # Read position data
    DBFileName = './positionData/fromDB/leftFrontKickPosition.json'
    AfterMappingFileName = './positionData/fromAfterMappingHand/leftFrontKickPosition(True, True, True, True, True, True).json'
    positionsJointCount = 7
    rollingWinSize = 10
    augmentationRatio = [0.5, 0.7, 1.3, 1.5]
    k_similar = 5

    ## Read Position data in DB
    positionsDB = None
    with open(DBFileName, 'r') as fileIn:
        jsonStr=json.load(fileIn)
        positionsDB = positionJsonDataParser(jsonStr, positionsJointCount)

    ## Read position data of mapped hand rotations
    positionsMapped = None
    with open(AfterMappingFileName, 'r') as fileIn:
        jsonStr=json.load(fileIn)
        positionsMapped = positionJsonDataParser(jsonStr, positionsJointCount)

    ### Position data to dataframe
    posDBDf = positionDataToPandasDf(positionsDB, positionsJointCount)
    posMappedDf = positionDataToPandasDf(positionsMapped, positionsJointCount)

    # - Preprocessing the position data
    DBfeatureVecs = positionDataPreproc(posDBDf, positionsJointCount, rollingWinSize, False, augmentationRatio)
    mappedFeatureVecs = positionDataPreproc(posMappedDf, positionsJointCount, rollingWinSize, False, augmentationRatio)
    print(DBfeatureVecs[0])
    print(mappedFeatureVecs[0])

    # - Compute the similarities between DB motions and the mapped motion
    # - 找最像的幾個feature vectors，看看它們大多屬於哪一個motion類別，target motion(mapped motion)就可以說最像該類別
    l2BtwDBAndMapped = euclidean_distances(DBfeatureVecs[0].values, mappedFeatureVecs[0].values)
    kSimilarL2 = np.sort(l2BtwDBAndMapped, axis=1)[:, ::-1]# 每個row都是DB feature與所有mapped feauter vector的距離列表
    kSimilarL2 = kSimilarL2[:, :k_similar]
    print(kSimilarL2)
    

# Obsolete section
if __name__=='__main01__':
    # Read position data
    DBFileName = './positionData/fromDB/leftFrontKickPosition.json'
    AfterMappingFileName = './positionData/fromAfterMappingHand/leftFrontKickPosition(True, True, True, True, True, True).json'
    positionsJointCount = 7
    rollingWinSize = 10
    augmentationRatio = [0.5, 0.7, 1.3, 1.5]

    ## Read Position data in DB
    positionsDB = None
    with open(DBFileName, 'r') as fileIn:
        jsonStr=json.load(fileIn)
        positionsDB = positionJsonDataParser(jsonStr, positionsJointCount)

    ## Read position data of mapped hand rotations
    positionsMapped = None
    with open(AfterMappingFileName, 'r') as fileIn:
        jsonStr=json.load(fileIn)
        positionsMapped = positionJsonDataParser(jsonStr, positionsJointCount)

    ### Position data to dataframe
    posDBDf = positionDataToPandasDf(positionsDB, positionsJointCount)
    posMappedDf = positionDataToPandasDf(positionsMapped, positionsJointCount)
    print(posDBDf.head(10))

    # - Make Hip joint as origin
    originAdjPosDBDf = setHipAsOrigin(posDBDf, positionsJointCount)
    print(originAdjPosDBDf.head(10))

    # - Get the windowed segments of each joints' position data(10 data in a row)
    jointsWindowSegs = rollingWindowSegRetrieve(originAdjPosDBDf, rollingWinSize, positionsJointCount)

    # - Compute the velocity and acceleration
    jointsWindowSegsWithVelAcc = []
    for i in range(len(jointsWindowSegs)): 
        jointsWindowSegsWithVelAcc.append(
            computeVelocityAndAcceleration(jointsWindowSegs[i], rollingWinSize)
        )
    print(jointsWindowSegsWithVelAcc[0])

    # Save , cause the execution time
    # for i, _df in enumerate(jointsWindowSegsWithVelAcc):
    #     _df.to_csv('tmp/{0}.csv'.format(i), index=False)
    
    # Load tmp csv
    # jointsWindowSegsWithVelAcc = []
    # for i in range(6):
    #     jointsWindowSegsWithVelAcc.append(
    #         pd.read_csv('tmp/{0}.csv'.format(i))
    #     )

    # - Augment feature vectors 
    jointsFeatureVecAug = []
    for i in range(6):
        multiSpeedRatioFeatureVec = \
            velocityAccelerationAugmentation(jointsWindowSegsWithVelAcc[i], rollingWinSize, augmentationRatio)
        # 組合不同速度的feature vector, 一個joint最終只會有一個DataFrame，當中包含不同速度產生的feature vectors
        jointsFeatureVecAug.append(pd.concat(multiSpeedRatioFeatureVec, axis=0))    
    print(jointsFeatureVecAug[0])

    # - Compute the similarities between DB motions and the mapped motion


    # - 找最像的幾個feature vectors，看看它們大多屬於哪一個motion類別，target motion就可以說最像該類別
    pass