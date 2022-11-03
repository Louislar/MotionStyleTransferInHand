'''
Goal: visualize 某個joint的trajectory以及相對應的animation positions
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import time
from positionAnalysis import jointsNames
from realTimePositionSynthesis import readDBEncodedMotionsFromFile

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
    with open('./positionData/fromAfterMappingHand/leftFrontKickStreamLinearMapping_TFFTTT.json', 'r') as WFile: 
    # with open('./positionData/fromAfterMappingHand/leftSideKickStreamLinearMapping_FTTFFF.json', 'r') as WFile: 
    # with open('./positionData/fromAfterMappingHand/runSprintStreamLinearMapping_TFTTFT.json', 'r') as WFile: 
    # with open('./positionData/fromAfterMappingHand/walkInjuredStreamLinearMapping_TFTTFT.json', 'r') as WFile: 
        # afterMappingJson = json.load(WFile)['results']
        afterMappingJson = json.load(WFile)

    # 1.2 
    # saveDirPathIdx = './similarFeatVecIdx/leftFrontKickStreamLinearMapping_TFFTTT_075/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_withoutHip_075/3DPos/'
    # saveDirPathIdx = './similarFeatVecIdx/leftFrontKickStreamLinearMapping_TFFTTT_075_transformed/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_withoutHip_075_transformed/3DPos/'
    saveDirPathIdx = './similarFeatVecIdx/leftFrontKickStreamLinearMapping_TFFTTT_075_normalized/'
    saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_withoutHip_075_normalized/3DPos/'
    # saveDirPathIdx = './similarFeatVecIdx/leftSideKickStreamLinearMapping_FTTFFF/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftSideKick_withoutHip/3DPos/'
    # saveDirPathIdx = './similarFeatVecIdx/runSprintStreamLinearMapping_TFTTFT/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/runSprint_withoutHip/3DPos/'
    # saveDirPathIdx = './similarFeatVecIdx/walkInjuredStreamLinearMapping_TFTTFT/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/walkInjured_withoutHip/3DPos/'
    similarIdx = {}
    for i in jointsInUsedToSyhthesis:
        similarIdx[i] = np.load(saveDirPathIdx+'{0}.npy'.format(i))
    DBPreproc3DPos = readDBEncodedMotionsFromFile(fullPositionsJointCount, saveDirPath3DPos)
    fullSimilarIdx = similarIdx[2]
    print(len(DBPreproc3DPos[2]))   # TODO: 發現feature vector的數量實際上比想像中少

    # 1.3
    saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_075/3DPos/'
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
    
    for i in range(700, 1500):
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
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_withoutHip/3DPos/'
    saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_withoutHip_075/3DPos/'
    # saveDirPathIdx = './similarFeatVecIdx/leftSideKickStreamLinearMapping_FTTFFF/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftSideKick_withoutHip/3DPos/'
    # saveDirPathIdx = './similarFeatVecIdx/runSprintStreamLinearMapping_TFTTFT/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/runSprint_withoutHip/3DPos/'
    # saveDirPathIdx = './similarFeatVecIdx/walkInjuredStreamLinearMapping_TFTTFT/'
    # saveDirPath3DPos = 'DBPreprocFeatVec/walkInjured_withoutHip/3DPos/'
    
    DBPreproc3DPos = readDBEncodedMotionsFromFile(fullPositionsJointCount, saveDirPath3DPos)
    print(len(DBPreproc3DPos[2]))   # TODO: 發現feature vector的數量實際上比想像中少

    # 1.3
    # saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick/3DPos/'
    saveDirPath3DPos = 'DBPreprocFeatVec/leftFrontKick_075/3DPos/'
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
        time.sleep(0.05)

if __name__=='__main__':
    main()
    