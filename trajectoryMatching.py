'''
利用transformation matrix, 
將hand mapped trajectory以及body trajectory對齊
最後, 將對齊前後的body trajectory以及hand mapped trajectory畫出來.
=======
(另一種作法)
利用normalization的方式, 將trajectory對齊. 
X, Y, Z三個position數值分別使用min max normalization的方式, 
將body trajectory對齊hand mapped trajectory的數值範圍
'''

import pandas as pd 
import numpy as np 
from scipy.spatial.transform import Rotation
import json 
import os 
import copy
import matplotlib.pyplot as plt 
from testingStageViz import jsonToDf
from realTimePositionSynthesis import readDBEncodedMotionsFromFile, fullPositionsJointCount
from rotationAnalysis import minMaxNormalization 
from positionSynthesis import jointsNames 

# Ref: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
# Copy from testingStageViz.py 
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def dfToJson(dfs):
    '''
    將dataframes in dict轉換成json
    '''
    jointsNm = list(dfs.keys())
    timeCount = dfs[jointsNm[0]].shape[0]
    jsonData = []
    for t in range(timeCount):
        _jointsDataInSingleTime = []
        for _jointInd in jointsNm:
            _jointsDataInSingleTime.append(
                dfs[_jointInd].iloc[t, :].to_dict()
            )
        jsonData.append(
            {
                'time': t,
                'data': _jointsDataInSingleTime
            }
        )
    jsonData = {'results': jsonData}
    return jsonData

# Construct transformation matrix and apply it to positions/trajectory
# input rotation in x, y, z order    
# input transition in x, y, z order 
def constructAndApplyTransMat(
    handMappedPosDirPath = 'positionData/fromAfterMappingHand/',
    body3dPosDirPath = 'DBPreprocFeatVec/leftFrontKick_withoutHip_075/3DPos/',
    transformedPosDirPath = 'transformedPosData/leftFrontKick_withoutHip_075/',
    rotationAngles = [0, 0, 0],
    translationValues = [0.1, 0, 0]
):
    # 1. read hand mapped positions
    # 2. read body positions
    # 3. construct transformation matrix
    ## ref: https://towardsdatascience.com/the-one-stop-guide-for-transformation-matrices-cea8f609bdb1
    # 3.1 apply transformation matrix to body trajectory
    # 4. store transformed result 
    
    # 1. 
    handMappedPosJson = None
    with open(os.path.join(handMappedPosDirPath, 'leftFrontKickStreamLinearMapping_TFFTTT.json'), 'r') as RFile: 
        handMappedPosJson = json.load(RFile)
    # joint key value改為數值而非string
    for t in range(len(handMappedPosJson)):
        _newDict = {}
        for k, v in handMappedPosJson[t]['data'].items():
            _newDict[int(k)]=v
        handMappedPosJson[t]['data']=_newDict
    # print(handMappedPosJson[1])
    handMappedPosJson = jsonToDf(handMappedPosJson)
    # print(handMappedPosJson[1])

    # 2. 
    DBPreproc3DPos = readDBEncodedMotionsFromFile(fullPositionsJointCount, body3dPosDirPath)
    ## to dataframe
    bodyJoint3dPos = {
        _jointInd: pd.DataFrame(
            DBPreproc3DPos[_jointInd],
            columns=['x','y','z'],
            index=range(DBPreproc3DPos[_jointInd].shape[0])
        ) for _jointInd in range(fullPositionsJointCount)
    }
    # print(bodyJoint3dPos[0])

    # 3. 
    transMat = np.eye(4)
    R = Rotation.from_euler("XYZ", rotationAngles, degrees=True).as_matrix()
    transMat[:3,:3] = R
    transMat[:3,3] = np.array(translationValues)
    # 3.1
    def _applyTransMat(vec: pd.Series, transMat: np.array):
        vecNp = vec.values
        vecNp = np.append(vecNp, 1)
        vecNp = np.dot(transMat, vecNp)
        return pd.Series(vecNp[:-1], index=vec.index)

    transformedBodyJoint3dPos = {} #copy.deepcopy(bodyJoint3dPos)
    for _jointInd in range(fullPositionsJointCount):
        transformedBodyJoint3dPos[_jointInd] = bodyJoint3dPos[_jointInd].apply(
            lambda _aRow: _applyTransMat(_aRow, transMat), 
            axis=1
        )
    # print(bodyJoint3dPos[0])
    # print(transformedBodyJoint3dPos[0])

    # 4. 
    for _jointInd in range(fullPositionsJointCount):
        transformedBodyJoint3dPos[_jointInd].to_csv(
            os.path.join(transformedPosDirPath, '{0}.csv'.format(_jointInd)),
            index=False
        )

# visualize trajectory
# (including before and after applying transformation trjectory and hand trajectory)
def visualizeTransResult(
    handMappedPosDirPath = 'positionData/fromAfterMappingHand/',
    body3dPosDirPath = 'DBPreprocFeatVec/leftFrontKick_withoutHip_075/3DPos/',
    transformed3dPosDirPath = 'transformedPosData/leftFrontKick_withoutHip_075/'
):
    # 1. read hand mapped positions
    # 2. read body positions
    # 3. read transformed body positions
    # 4. visualize
    
    # 1. 
    handMappedPosJson = None
    with open(os.path.join(handMappedPosDirPath, 'leftFrontKickStreamLinearMapping_TFFTTT.json'), 'r') as RFile: 
        handMappedPosJson = json.load(RFile)
    # joint key value改為數值而非string
    for t in range(len(handMappedPosJson)):
        _newDict = {}
        for k, v in handMappedPosJson[t]['data'].items():
            _newDict[int(k)]=v
        handMappedPosJson[t]['data']=_newDict
    # print(handMappedPosJson[1])
    handMappedPos = jsonToDf(handMappedPosJson)
    # print(handMappedPos[1])
    # print(handMappedPos[6])
    ## 需要使用校正hip為原點
    for _jointInd in handMappedPos.keys():
        if _jointInd != 6:
            handMappedPos[_jointInd] = handMappedPos[_jointInd] - handMappedPos[6]
    handMappedPos[6] = handMappedPos[6] - handMappedPos[6]
    # print(handMappedPos[1])
    # 2. 
    DBPreproc3DPos = readDBEncodedMotionsFromFile(fullPositionsJointCount, body3dPosDirPath)
    ## to dataframe
    bodyJoint3dPos = {
        _jointInd: pd.DataFrame(
            DBPreproc3DPos[_jointInd],
            columns=['x','y','z'],
            index=range(DBPreproc3DPos[_jointInd].shape[0])
        ) for _jointInd in range(fullPositionsJointCount)
    }
    # 3. 
    transformedPos = {}
    for _jointInd in range(fullPositionsJointCount):
        transformedPos[_jointInd] = pd.read_csv(
            os.path.join(transformed3dPosDirPath, '{0}.csv'.format(_jointInd))
        )
    
    # 4. 
    vizJointInd = 2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.plot(
        handMappedPos[2]['x'],
        handMappedPos[2]['y'],
        handMappedPos[2]['z'],
        '.',
        label = 'mapped position'
    )
    ax.plot(
        bodyJoint3dPos[2]['x'],
        bodyJoint3dPos[2]['y'],
        bodyJoint3dPos[2]['z'],
        '.',
        label='original body trajectory'
    )
    ax.plot(
        transformedPos[2]['x'],
        transformedPos[2]['y'],
        transformedPos[2]['z'],
        '.',
        color='r',
        label='transformed body trajectory'
    )
    plt.legend()
    plt.show()
    pass

# construct transformation matrix (4x4 matrix)
# input rotation in x, y, z order    
# input transition in x, y, z order 
def constructTransMat(
    rotationAngles=[0,0,0],
    translationValues=[0.3,0,0.2]
):
    transMat = np.eye(4)
    R = Rotation.from_euler("XYZ", rotationAngles, degrees=True).as_matrix()
    transMat[:3,:3] = R
    transMat[:3,3] = np.array(translationValues)
    return transMat

# 對所有body joint的點apply transformation matrix
# input rotation in x, y, z order    
# input transition in x, y, z order 
def applyTransMatToEntireAnimation(
    bodyPosFilePath = 'positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withoutHip_075.json',
    transformedPosOutputFilePath = 'positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withoutHip_075_transformed.json',
    rotationAngles=[0,0,0],
    translationValues=[0.4,0,0.2]
):
    # 1. read body positions
    # 2. construct and apply transformation (matrix)
    ## Warning: Do not transform the body joint position since it is the origin. 
    ## And other processes/function 會將origin歸0, 導致transformation失效
    # 3. store transformed body positions

    # 1. 
    posDBDf = None
    with open(os.path.join(bodyPosFilePath), 'r') as fileIn:
        jsonStr=json.load(fileIn)['results']
        posDBDf=jsonToDf(jsonStr)
    # print(len(posDBDf))
    # print(posDBDf[0])
    # 2. 
    transMat = constructTransMat(rotationAngles, translationValues)
    def _applyTransMat(vec: pd.Series, transMat: np.array):
        vecNp = vec.values  # convert to NumPy vector
        vecNp = np.append(vecNp, 1)
        vecNp = np.dot(transMat, vecNp)
        return pd.Series(vecNp[:-1], index=vec.index)
    transformedBodyPos = {}
    for _jointInd in range(len(posDBDf)):
        if _jointInd != 6:
            transformedBodyPos[_jointInd] = posDBDf[_jointInd].apply(
                lambda _aRow: _applyTransMat(_aRow, transMat),
                axis=1
            )
        else: 
            transformedBodyPos[_jointInd] = posDBDf[_jointInd]
    # 3. 
    transformedBodyPosJson = dfToJson(transformedBodyPos)
    with open(transformedPosOutputFilePath, 'w') as WFile:
        json.dump(transformedBodyPosJson, WFile)

# 使用min max normalization的方式, 將body trajectory對齊hand mapped trajectory
def matchTrajectoryViaNormalization(
    bodyPosFilePath = 'positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withoutHip_075.json', 
    handMappedPosDirPath = 'positionData/fromAfterMappingHand/', 
    normalizedBodyPosFilePath = 'positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withoutHip_075_normalized.json'
): 
    '''
    Note: 這邊為了方便, 只有對used joint (left feet)的position進行調整. 其他joint的positions沒有做更動.
    '''
    # 1. read body trajectory/positions
    # 2. read hand mapped trajectory/positions
    # 3. compute min and max of hand mapped trajectory
    # 4. use min and max to normalize body trajectory
    # 5. store the normalized result

    usedJoint = 2   # left feet
    # 1. 
    posDBDf = None
    with open(os.path.join(bodyPosFilePath), 'r') as fileIn:
        jsonStr=json.load(fileIn)['results']
        posDBDf=jsonToDf(jsonStr)
    # print(posDBDf[0])
    # 2. 
    handMappedPosJson = None
    with open(os.path.join(handMappedPosDirPath, 'leftFrontKickStreamLinearMapping_TFFTTT.json'), 'r') as RFile: 
        handMappedPosJson = json.load(RFile)
    ## joint key value改為數值而非string
    for t in range(len(handMappedPosJson)):
        _newDict = {}
        for k, v in handMappedPosJson[t]['data'].items():
            _newDict[int(k)]=v
        handMappedPosJson[t]['data']=_newDict
    handMappedPos = jsonToDf(handMappedPosJson)
    ## 需要使用校正hip為原點
    for _jointInd in handMappedPos.keys():
        if _jointInd != 6:
            handMappedPos[_jointInd] = handMappedPos[_jointInd] - handMappedPos[6]
    handMappedPos[6] = handMappedPos[6] - handMappedPos[6]
    # print(handMappedPos[1])

    # 3. 
    handMappedMin = handMappedPos[usedJoint].min(axis=0)
    handMappedMax = handMappedPos[usedJoint].max(axis=0)
    posRange = {
        _axis: [handMappedMin[_axis], handMappedMax[_axis]] for _axis in ['x', 'y', 'z']
    } # first element: min, second element: max
    # print(handMappedPos[2])
    # print(handMappedPos[2].min(axis=0))
    # print(handMappedPos[2].max(axis=0))
    print(posRange)

    # 4. 
    # print(posDBDf[2])
    for _axis in ['x', 'y', 'z']:
        posDBDf[usedJoint].loc[:, _axis] = \
            minMaxNormalization(posDBDf[usedJoint][ _axis], posRange[_axis][0], posRange[_axis][1])
    # print(posDBDf[2])

    # TODO: 左腳normalize後, 需要把hip的所有positions校正成0
    #       因為, 後面的處理會將hip設為原點, 導致normalized的結果受到影響
    posDBDf[6].loc[:, 'x'].values[:] = 0
    posDBDf[6].loc[:, 'y'].values[:] = 0
    posDBDf[6].loc[:, 'z'].values[:] = 0

    # 5. 
    normalizedBodyPosJson = dfToJson(posDBDf)
    with open(normalizedBodyPosFilePath, 'w') as WFile:
        json.dump(normalizedBodyPosJson, WFile)
    pass

# 與matchTrajectoryViaNormalization()相同. 只是可以指定最大最小值的percentile, 以及指定mapping的軸. 
def trajectoryNormalization(
    bodyPosFilePath = 'positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withoutHip_075.json', 
    handMappedPosDirPath = 'positionData/fromAfterMappingHand/newMappingMethods/leftFrontKick_quat_BSpline_TFTTTT.json', 
    normalizedBodyPosFilePath = 'positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withoutHip_075_normalized.json',
    maxPercentile = 0.8,
    minPercentile = 0.2,
    normalizeAxis = ['y']    
):
    '''
    參考matchTrajectoryViaNormalization()
    但是, 增加"部分軸的資料作normalization", 並且增加"取前幾%的數值當作max, 後幾%的數值當作min"
    '''
    # 1. read body trajectory/positions
    # 1.1 校正hip回原點 
    # 1.2 DB trajectory需要去除前5筆資料做normalization
    # 2. read hand mapped trajectory/positions
    # 3. compute min and max of hand mapped trajectory
    # 4. use min and max to normalize body trajectory
    # 4.1 補回之前沒有使用的前5筆資料
    # 5. store the normalized result

    # 修改以下程式碼, 需要能夠接受三個參數 (完成normalize axis)
    usedJoint = 2   # left feet
    # 1. 
    posDBDf = None
    with open(os.path.join(bodyPosFilePath), 'r') as fileIn:
        jsonStr=json.load(fileIn)['results']
        posDBDf=jsonToDf(jsonStr)
    jointCount = len(posDBDf)
    axisNames = list(posDBDf[0].keys())
    print('joint count: ', jointCount)
    print('axis category: ', axisNames)
    # 1.1 校正hip回原點 
    for _jointInd in range(jointCount):
        if _jointInd != jointsNames.Hip:
            posDBDf[_jointInd] = posDBDf[_jointInd] - posDBDf[jointsNames.Hip]
    posDBDf[jointsNames.Hip].iloc[:, :] = 0
    # 1.2 去除前5筆資料
    first5DBDf = []
    for _jointInd in range(jointCount):
        first5DBDf.append(posDBDf[_jointInd].iloc[:5, :])
        posDBDf[_jointInd] = posDBDf[_jointInd].iloc[5:, :]
    print('First 5 DataBase animation size: ', first5DBDf[0].shape)
    # 2. 
    handMappedPosJson = None
    with open(handMappedPosDirPath, 'r') as RFile: 
        handMappedPosJson = json.load(RFile)
    ## joint key value改為數值而非string
    for t in range(len(handMappedPosJson)):
        _newDict = {}
        for k, v in handMappedPosJson[t]['data'].items():
            _newDict[int(k)]=v
        handMappedPosJson[t]['data']=_newDict
    handMappedPos = jsonToDf(handMappedPosJson)
    ## 需要使用校正hip為原點
    for _jointInd in handMappedPos.keys():
        if _jointInd != 6:
            handMappedPos[_jointInd] = handMappedPos[_jointInd] - handMappedPos[6]
    handMappedPos[6] = handMappedPos[6] - handMappedPos[6]
    # print(handMappedPos[1])

    # 3. 
    # XYZ分別計算80%高的數值以及20%低的數值
    handMappedMin = handMappedPos[usedJoint].quantile(minPercentile, axis=0)
    handMappedMax = handMappedPos[usedJoint].quantile(maxPercentile, axis=0)
    posRange = {
        _axis: [handMappedMin[_axis], handMappedMax[_axis]] for _axis in ['x', 'y', 'z']
    } # first element: min, second element: max
    # print(handMappedPos[2])
    print('min: ', handMappedPos[2].min(axis=0))
    print('max: ', handMappedPos[2].max(axis=0))
    print(posRange)

    # 4. 
    # print(posDBDf[2])
    for _axis in normalizeAxis:
        posDBDf[usedJoint].loc[:, _axis] = \
            minMaxNormalization(posDBDf[usedJoint][ _axis], posRange[_axis][0], posRange[_axis][1])
    # print(posDBDf[2])

    # 4.1 把前5筆資料補回來 
    for _jointInd in range(jointCount):
        posDBDf[_jointInd] = pd.DataFrame(first5DBDf[_jointInd], columns=posDBDf[_jointInd].columns).append(posDBDf[_jointInd])

    # 左腳normalize後, 需要把hip的所有positions校正成0
    # 因為, 後面的處理會將hip設為原點, 導致normalized的結果受到影響
    # New: 前面校正過, 所以不用再校正一次
    # posDBDf[6].loc[:, 'x'].values[:] = 0
    # posDBDf[6].loc[:, 'y'].values[:] = 0
    # posDBDf[6].loc[:, 'z'].values[:] = 0

    # 5. 
    normalizedBodyPosJson = dfToJson(posDBDf)
    with open(normalizedBodyPosFilePath, 'w') as WFile:
        json.dump(normalizedBodyPosJson, WFile)
    pass

def visualizeNormalizeResult(
    normalizedBodyPosFilePath = 'positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withoutHip_075_normalized.json', 
    bodyPosFilePath = 'positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withoutHip_075.json', 
    handMappedPosDirPath = 'positionData/fromAfterMappingHand/newMappingMethods/leftFrontKick_quat_BSpline_TFTTTT.json'
):
    # 1. read normalized body positions/trajectory
    # 2. read original body positions/trajectory
    # 2.1 校正hip回原點 
    # 3. read read hand mapped trajectory/positions
    # 4. visualize all the data

    usedJoint = 2   # left feet
    # 1. 
    normPosDBDf = None
    with open(os.path.join(normalizedBodyPosFilePath), 'r') as fileIn:
        jsonStr=json.load(fileIn)['results']
        normPosDBDf=jsonToDf(jsonStr)
    # 2. 
    posDBDf = None
    with open(os.path.join(bodyPosFilePath), 'r') as fileIn:
        jsonStr=json.load(fileIn)['results']
        posDBDf=jsonToDf(jsonStr)
    # print(normPosDBDf[usedJoint])
    # print(posDBDf[usedJoint])
    # 2.1 校正hip回原點 
    for _jointInd in range(len(posDBDf)):
        if _jointInd != jointsNames.Hip:
            posDBDf[_jointInd] = posDBDf[_jointInd] - posDBDf[jointsNames.Hip]
    posDBDf[jointsNames.Hip].iloc[:, :] = 0
    # 3. 
    handMappedPosJson = None
    with open(handMappedPosDirPath, 'r') as RFile: 
        handMappedPosJson = json.load(RFile)
    ## joint key value改為數值而非string
    for t in range(len(handMappedPosJson)):
        _newDict = {}
        for k, v in handMappedPosJson[t]['data'].items():
            _newDict[int(k)]=v
        handMappedPosJson[t]['data']=_newDict
    handMappedPos = jsonToDf(handMappedPosJson)
    ## 需要使用校正hip為原點
    for _jointInd in handMappedPos.keys():
        if _jointInd != 6:
            handMappedPos[_jointInd] = handMappedPos[_jointInd] - handMappedPos[6]
    handMappedPos[6] = handMappedPos[6] - handMappedPos[6]

    # 4. 
    vizJointInd = 2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.plot(
        handMappedPos[vizJointInd]['x'],
        handMappedPos[vizJointInd]['y'],
        handMappedPos[vizJointInd]['z'],
        '.',
        label = 'mapped position'
    )
    ax.plot(
        posDBDf[vizJointInd]['x'],
        posDBDf[vizJointInd]['y'],
        posDBDf[vizJointInd]['z'],
        '.',
        label='original body trajectory'
    )
    ax.plot(
        normPosDBDf[vizJointInd]['x'],
        normPosDBDf[vizJointInd]['y'],
        normPosDBDf[vizJointInd]['z'],
        '.',
        label='normalized body trajectory'
    )
    set_axes_equal(ax)
    plt.tight_layout()
    plt.legend()
    plt.show()
    pass

if __name__=='__main__':
    ## construct and apply transformation matrix to body positions/trajectory
    # constructAndApplyTransMat(
    #     handMappedPosDirPath = 'positionData/fromAfterMappingHand/',
    #     body3dPosDirPath = 'DBPreprocFeatVec/leftFrontKick_075/3DPos/',
    #     transformedPosDirPath = 'transformedPosData/leftFrontKick_075/',
    #     rotationAngles=[0.2,0,-0.2],
    #     translationValues=[0,-0.1,0]
    # )
    ## visualize transformation applying result
    ## 使用肉眼判斷兩個trajectory重合的效果好不好
    # visualizeTransResult(
    #     handMappedPosDirPath = 'positionData/fromAfterMappingHand/',
    #     body3dPosDirPath = 'DBPreprocFeatVec/leftFrontKick_075/3DPos/',
    #     transformed3dPosDirPath = 'transformedPosData/leftFrontKick_075/'
    # )
    ## apply tranformation to body positions in DB
    # applyTransMatToEntireAnimation(
    #     bodyPosFilePath = 'positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withoutHip_075.json',
    #     transformedPosOutputFilePath = 'positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withoutHip_075_transformed.json',
    #     rotationAngles=[0,0,0],
    #     translationValues=[0.4,0,0.2]
    # )
    ## ======= ======= ======= ======= ======= ======= ======= 
    ## 使用minmax normalization方式對齊
    # matchTrajectoryViaNormalization(
    #     bodyPosFilePath = 'positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withoutHip_075.json', 
    #     handMappedPosDirPath = 'positionData/fromAfterMappingHand/', 
    #     normalizedBodyPosFilePath = 'positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withoutHip_075_normalized.json'
    # )
    ## 與matchTrajectoryViaNormalization() 相似, 
    ## 但是只對特定axis做normalization. 並且, normalization的min max是取前80%與後20%percentile. 
    trajectoryNormalization(
        bodyPosFilePath = 'positionData/fromDB/genericAvatar/twoLegJumpPositionFullJointsWithHead_withoutHip_075.json', 
        handMappedPosDirPath = 'positionData/twoLegJump_quat_directMapping.json', 
        normalizedBodyPosFilePath = 'positionData/fromDB/genericAvatar/twoLegJumpPositionFullJointsWithHead_withoutHip_075_quat_direct_normalized.json',
        maxPercentile = 0.95,
        minPercentile = 0.05,
        normalizeAxis = ['x', 'y', 'z']  
    )
    ## visualize normalization result
    visualizeNormalizeResult(
        normalizedBodyPosFilePath = 'positionData/fromDB/genericAvatar/twoLegJumpPositionFullJointsWithHead_withoutHip_075_quat_direct_normalized.json', 
        bodyPosFilePath = 'positionData/fromDB/genericAvatar/twoLegJumpPositionFullJointsWithHead_withoutHip_075.json', 
        handMappedPosDirPath = 'positionData/twoLegJump_quat_directMapping.json'
    )
    pass
