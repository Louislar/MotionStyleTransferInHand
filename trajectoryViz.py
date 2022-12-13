'''
Goal: visualize 某個joint的trajectory以及相對應的animation positions
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl 
import json
import time
from positionAnalysis import jointsNames
from realTimePositionSynthesis import readDBEncodedMotionsFromFile
from rotationAnalysisViz import saveFigs

positionsJointCount = 7 # 用於比對motion similarity的joint數量(Upper leg*2, knee*2, foot*2, hip)
fullPositionsJointCount = 17    # 用於做motion synthesis的joint數量
rollingWinSize = 10
kSimilar = 5
jointsInUsedToSyhthesis = [
    jointsNames.LeftLowerLeg, jointsNames.LeftFoot, jointsNames.RightLowerLeg, jointsNames.RightFoot
]

def main():
    '''
    visualize after mapping position time series, with hip animation position
    and without hip animation position
    '''
    # 1.1 read after mapping position time series
    # 1.2 read 相似的feature vector index, 以及feature vector對應的3D position
    # 1.3 read with hip 3d positions
    # 2. extract specific joint's position time series for drawing 3D plot
    # 3. plot the lines in 3D space

    # 1.1 
    afterMappingJson=None
    # with open('./positionData/fromAfterMappingHand/leftFrontKickStreamLinearMapping/leftFrontKick(True, False, False, True, True, True).json', 'r') as WFile: 
    # with open('./positionData/fromAfterMappingHand/leftFrontKickStreamLinearMapping_TFFTTT.json', 'r') as WFile: 
    # with open('./positionData/fromAfterMappingHand/newMappingMethods/leftFrontKick_quat_BSpline_TFTTTT.json', 'r') as WFile: 
    # with open('./positionData/fromAfterMappingHand/newMappingMethods/leftSideKick_quat_BSpline_FTTTFT.json', 'r') as WFile: 
    # with open('./positionData/fromAfterMappingHand/leftSideKickStreamLinearMapping_FTTFFF.json', 'r') as WFile: 
    with open('./positionData/fromAfterMappingHand/newMappingMethods/runSprint_quat_BSpline_TFTTFT.json', 'r') as WFile: 
    # with open('./positionData/fromAfterMappingHand/runSprintStreamLinearMapping_TFTTFT.json', 'r') as WFile: 
    # with open('./positionData/fromAfterMappingHand/walkInjuredStreamLinearMapping_TFTTFT.json', 'r') as WFile: 
        # afterMappingJson = json.load(WFile)['results']
        afterMappingJson = json.load(WFile)
    print(len(afterMappingJson))
    # 1.2 
    # saveDirPathIdx = './similarFeatVecIdx/leftFrontKickStreamLinearMapping_TFFTTT_075/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_withoutHip_075/3DPos/'
    # saveDirPathIdx = './similarFeatVecIdx/leftFrontKickStreamLinearMapping_TFFTTT_075_transformed/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_withoutHip_075_transformed/3DPos/'
    # saveDirPathIdx = './similarFeatVecIdx/leftFrontKickStreamLinearMapping_TFFTTT_075_normalized/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_withoutHip_075_normalized/3DPos/'
    # saveDirPathIdx = './similarFeatVecIdx/leftFrontKickStreamLinearMapping_TFFTTT_withHip_075_normalized/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_withHip_075_normalized/3DPos/'
    # saveDirPathIdx = './similarFeatVecIdx/leftFrontKick_quat_BSpline_TFTTTT_withHip_075_normalized/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_withHip_075_quat_BSpline_normalized/3DPos/'
    # saveDirPathIdx = './similarFeatVecIdx/leftSideKick_quat_BSpline_FTTTFT_withoutHip_075_normalized/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftSideKick_withoutHip_075_quat_BSpline_normalized/3DPos/'
    # saveDirPathIdx = './similarFeatVecIdx/leftSideKickStreamLinearMapping_FTTFFF/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftSideKick_withoutHip/3DPos/'
    # saveDirPathIdx = './similarFeatVecIdx/runSprintStreamLinearMapping_TFTTFT/'
    saveDirPathIdx = './similarFeatVecIdx/runSprint_quat_BSpline_TFTTFT_withHip_05_normalized/'
    saveDirPath3DPos = 'DBPreprocFeatVec/runSprint_withHip_05_quat_BSpline_normalized/3DPos/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/runSprint_withoutHip/3DPos/'
    # saveDirPathIdx = './similarFeatVecIdx/walkInjuredStreamLinearMapping_TFTTFT/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/walkInjured_withoutHip/3DPos/'
    similarIdx = {}
    for i in jointsInUsedToSyhthesis:
        similarIdx[i] = np.load(saveDirPathIdx+'{0}.npy'.format(i))
    DBPreproc3DPos = readDBEncodedMotionsFromFile(fullPositionsJointCount, saveDirPath3DPos)
    fullSimilarIdx = similarIdx[2]
    print(len(DBPreproc3DPos[2]))   # TODO: 發現feature vector的數量實際上比想像中少
    print(len(fullSimilarIdx))

    # 1.3
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_075/3DPos/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftSideKick_075/3DPos/'
    saveDirPath3DPos = 'DBPreprocFeatVec/runSprint_withHip_05/3DPos/'
    DBPreproc3DPos_withHip = readDBEncodedMotionsFromFile(fullPositionsJointCount, saveDirPath3DPos)
    print(len(DBPreproc3DPos_withHip[2]))

    # 改成一次只plot一個time point的資料
    # 顯示所有找到的5個similar animation position
    # 加大顯示單一個的mapped position

    timeIdx = 1
    axisKeys = ['x', 'y', 'z']
    afterMappingPos = [[afterMappingJson[timeIdx]['data']['2'][k] - afterMappingJson[timeIdx]['data']['6'][k] for k in axisKeys]]
    similarFeatVecPos = DBPreproc3DPos[2][fullSimilarIdx[timeIdx, :]]
    similarFeatVecPos = [similarFeatVecPos[rowIdx, :] for rowIdx in range(similarFeatVecPos.shape[0])]
    similarFeatVecPos_withHip = DBPreproc3DPos_withHip[2][fullSimilarIdx[timeIdx, :]]
    similarFeatVecPos_withHip = [similarFeatVecPos_withHip[rowIdx, :] for rowIdx in range(similarFeatVecPos_withHip.shape[0])]
    fullFeatVecPos = [DBPreproc3DPos[2][t] for t in range(len(DBPreproc3DPos[2]))] 
    fullFeatVecPos_withHip = [DBPreproc3DPos_withHip[2][t] for t in range(len(DBPreproc3DPos_withHip[2]))] 
    print(similarFeatVecPos)
    
    # 3. 
    plt.ion()   # For drawing multiple figures
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    line1, = ax.plot([d[0] for d in afterMappingPos], [d[1] for d in afterMappingPos], [d[2] for d in afterMappingPos], '*', label='after_mapping', color='g')
    line2, = ax.plot([d[0] for d in fullFeatVecPos], [d[1] for d in fullFeatVecPos], [d[2] for d in fullFeatVecPos], '.', label='full_featVec')
    line3, = ax.plot([d[0] for d in similarFeatVecPos], [d[1] for d in similarFeatVecPos], [d[2] for d in similarFeatVecPos], '.', markersize=10, label='similar_featVec', color='r')
    # with hip positions, 真正會被用來做synthesis的positions
    line4, = ax.plot([d[0] for d in fullFeatVecPos_withHip], [d[1] for d in fullFeatVecPos_withHip], [d[2] for d in fullFeatVecPos_withHip], '.', label='full_featVec_withHip')
    line5, = ax.plot([d[0] for d in similarFeatVecPos_withHip], [d[1] for d in similarFeatVecPos_withHip], [d[2] for d in similarFeatVecPos_withHip], '.', markersize=10, label='similar_featVec_withHip')

    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    plt.legend()
    # plt.show()
    
    # for i in range(700, 1500):    # for front kick 
    for i in range(0, len(fullSimilarIdx)):
        # ref: https://www.geeksforgeeks.org/how-to-update-a-plot-on-same-figure-during-the-loop/
        # update data
        afterMappingPos = [[afterMappingJson[i]['data']['2'][k] - afterMappingJson[i]['data']['6'][k] for k in axisKeys]]
        similarFeatVecPos = DBPreproc3DPos[2][fullSimilarIdx[i, :]]
        similarFeatVecPos = [similarFeatVecPos[rowIdx, :] for rowIdx in range(similarFeatVecPos.shape[0])]
        similarFeatVecPos_withHip = DBPreproc3DPos_withHip[2][fullSimilarIdx[i, :]]
        similarFeatVecPos_withHip = [similarFeatVecPos_withHip[rowIdx, :] for rowIdx in range(similarFeatVecPos_withHip.shape[0])]

        line1.set_data([d[0] for d in afterMappingPos], [d[1] for d in afterMappingPos])
        line1.set_3d_properties([d[2] for d in afterMappingPos], 'z')
        line3.set_data([d[0] for d in similarFeatVecPos], [d[1] for d in similarFeatVecPos])
        line3.set_3d_properties([d[2] for d in similarFeatVecPos], 'z')
        line5.set_data([d[0] for d in similarFeatVecPos_withHip], [d[1] for d in similarFeatVecPos_withHip])
        line5.set_3d_properties([d[2] for d in similarFeatVecPos_withHip], 'z')
        

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.05)

def main01():
    '''
    從index 0到最後, 從with hip與without hip的animation feature vectors visualize相應的positions
    這邊只是單純按照時間順序visualize with hip與without hip的animation, 
    目的是觀察兩者的時間與位置是否有對齊
    '''
    ## 1.1 without hip的3d positions
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_withoutHip/3DPos/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_withoutHip_075/3DPos/'
    # saveDirPathIdx = './similarFeatVecIdx/leftSideKickStreamLinearMapping_FTTFFF/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftSideKick_withoutHip/3DPos/'
    saveDirPath3DPos = 'DBPreprocFeatVec/leftSideKick_withoutHip_075/3DPos/'
    # saveDirPathIdx = './similarFeatVecIdx/runSprintStreamLinearMapping_TFTTFT/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/runSprint_withoutHip/3DPos/'
    # saveDirPathIdx = './similarFeatVecIdx/walkInjuredStreamLinearMapping_TFTTFT/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/walkInjured_withoutHip/3DPos/'
    
    DBPreproc3DPos = readDBEncodedMotionsFromFile(fullPositionsJointCount, saveDirPath3DPos)
    print(len(DBPreproc3DPos[2]))   # TODO: 發現feature vector的數量實際上比想像中少

    # 1.2 with hip的3d positions
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick/3DPos/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_075/3DPos/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftSideKick/3DPos/'
    saveDirPath3DPos = 'DBPreprocFeatVec/leftSideKick_075/3DPos/'
    DBPreproc3DPos_withHip = readDBEncodedMotionsFromFile(fullPositionsJointCount, saveDirPath3DPos)
    print(len(DBPreproc3DPos_withHip[2]))

    # 改成一次只plot一個time point的資料
    # 顯示所有找到的5個similar animation position
    # 加大顯示單一個的mapped position

    timeIdx = 1
    axisKeys = ['x', 'y', 'z']
    similarFeatVecPos = DBPreproc3DPos[2][timeIdx:timeIdx+2]
    similarFeatVecPos = [similarFeatVecPos[rowIdx, :] for rowIdx in range(similarFeatVecPos.shape[0])]
    similarFeatVecPos_withHip = DBPreproc3DPos_withHip[2][timeIdx:timeIdx+2]
    similarFeatVecPos_withHip = [similarFeatVecPos_withHip[rowIdx, :] for rowIdx in range(similarFeatVecPos_withHip.shape[0])]
    fullFeatVecPos = [DBPreproc3DPos[2][t] for t in range(len(DBPreproc3DPos[2]))] 
    fullFeatVecPos_withHip = [DBPreproc3DPos_withHip[2][t] for t in range(len(DBPreproc3DPos_withHip[2]))] 
    
    # 3. 
    plt.ion()   # For drawing multiple figures
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    line2, = ax.plot([d[0] for d in fullFeatVecPos], [d[1] for d in fullFeatVecPos], [d[2] for d in fullFeatVecPos], '.', label='full_featVec')
    line3, = ax.plot([d[0] for d in similarFeatVecPos], [d[1] for d in similarFeatVecPos], [d[2] for d in similarFeatVecPos], '.', markersize=10, label='similar_featVec', color='r')
    # with hip positions, 真正會被用來做synthesis的positions
    line4, = ax.plot([d[0] for d in fullFeatVecPos_withHip], [d[1] for d in fullFeatVecPos_withHip], [d[2] for d in fullFeatVecPos_withHip], '.', label='full_featVec_withHip')
    line5, = ax.plot([d[0] for d in similarFeatVecPos_withHip], [d[1] for d in similarFeatVecPos_withHip], [d[2] for d in similarFeatVecPos_withHip], '.', markersize=10, label='similar_featVec_withHip')

    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    plt.legend()
    # plt.show()
    
    for i in range(0, 200):
        # ref: https://www.geeksforgeeks.org/how-to-update-a-plot-on-same-figure-during-the-loop/
        # update data
        similarFeatVecPos = DBPreproc3DPos[2][i:i+1]
        similarFeatVecPos = [similarFeatVecPos[rowIdx, :] for rowIdx in range(similarFeatVecPos.shape[0])]
        similarFeatVecPos_withHip = DBPreproc3DPos_withHip[2][i:i+1]
        similarFeatVecPos_withHip = [similarFeatVecPos_withHip[rowIdx, :] for rowIdx in range(similarFeatVecPos_withHip.shape[0])]

        line3.set_data([d[0] for d in similarFeatVecPos], [d[1] for d in similarFeatVecPos])
        line3.set_3d_properties([d[2] for d in similarFeatVecPos], 'z')
        line5.set_data([d[0] for d in similarFeatVecPos_withHip], [d[1] for d in similarFeatVecPos_withHip])
        line5.set_3d_properties([d[2] for d in similarFeatVecPos_withHip], 'z')
        

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.5)

# visualize多種mapping function造成的trajectory
def vizMultiTrajectories(positionFilePaths, fileDataNm, saveDirPath):
    '''
    visualize多種mapping function造成的trajectory.
    總共製作四張圖片 
    Input:
    :positionFilePaths: (list) position json檔案位址 
        順序是eular linear, eular B-Spline, quat linear, quat B-Spline 
    :fileDataNm: (list) position檔案儲存的資料的代表名稱 (用於畫圖顯示資料類別)
    '''
    posData = []
    for _filePath in positionFilePaths:
        with open(_filePath, 'r') as RFile:
            posData.append(
                json.load(RFile)
            )
    ## convert data to 單一轉軸的time series
    timeCount = len(posData[0])
    jointCount = len(posData[0][0]['data'].keys())
    print('timeCount: ', timeCount)
    print('jointCount: ', jointCount)
    posTimeSeries = [{k: None for k in range(jointCount)} for i in range(len(posData))]
    for i in range(len(posData)):
        for k in range(jointCount):
            _aJointTimeSeries = []
            for t in range(timeCount):
                _aJointTimeSeries.append(
                    posData[i][t]['data'][str(k)]
                )
            posTimeSeries[i][k] = _aJointTimeSeries
    ## Plot
    _defaultMarkerSize = mpl.rcParams['lines.markersize']
    plotIndPair = [[0,1], [2,3], [0,2], [1,3]]    # 指定繪製的資料配對index
    figs = []
    for _indPair in plotIndPair:
        _ind1 = _indPair[0]
        _ind2 = _indPair[1]
        _fig = plt.figure()
        _ax = _fig.add_subplot(111, projection='3d')
        
        _ax.plot(
            [i['x'] for i in posTimeSeries[_ind1][2]], 
            [i['y'] for i in posTimeSeries[_ind1][2]], 
            [i['z'] for i in posTimeSeries[_ind1][2]], 
            '.-',
            markersize=_defaultMarkerSize*(3/2),
            label=fileDataNm[_ind1]
        )
        _ax.plot(
            [i['x'] for i in posTimeSeries[_ind2][2]], 
            [i['y'] for i in posTimeSeries[_ind2][2]], 
            [i['z'] for i in posTimeSeries[_ind2][2]], 
            label=fileDataNm[_ind2]
        )
        plt.legend()
        plt.show()
    ## Store figures
    # saveFigs(figs, saveDirPath)            
    pass

if __name__=='__main__':
    main()
    # main01()
    ## visualize多種mapping function造成的trajectory
    # vizMultiTrajectories(
    #     positionFilePaths=[
    #         'positionData/fromAfterMappingHand/newMappingMethods/leftFrontKick_eular_linear_TFTTTT.json',
    #         'positionData/fromAfterMappingHand/newMappingMethods/leftFrontKick_eular_BSpline_TFTTTT.json',
    #         'positionData/fromAfterMappingHand/newMappingMethods/leftFrontKick_quat_linear_TFTTTT.json',
    #         'positionData/fromAfterMappingHand/newMappingMethods/leftFrontKick_quat_BSpline_TFTTTT.json'
    #     ],
    #     fileDataNm=[
    #         'eular linear',
    #         'eular B-Spline',
    #         'quat linear',
    #         'quat B-Spline'
    #     ],
    #     saveDirPath = 'rotationMappingQuaternionFigs/leftFrontKick/trajectoryCompare/'
    # )

    