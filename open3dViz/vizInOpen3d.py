import open3d as o3d
import open3d.visualization.gui as gui 
import numpy as np
import time
import sys
import json
import pandas as pd 
sys.path.append("../")
from testingStageViz import jsonToDf 
from positionAnalysis import jointsNames
from realTimeHandRotationCompute import jointsNames as handJointsNames

# 全身骨架的結構 
fullBodyBoneStrcuture = [
    [jointsNames.Hip, jointsNames.LeftUpperLeg], [jointsNames.Hip, jointsNames.RightUpperLeg], 
    [jointsNames.LeftUpperLeg, jointsNames.LeftLowerLeg], [jointsNames.RightUpperLeg, jointsNames.RightLowerLeg], 
    [jointsNames.LeftLowerLeg, jointsNames.LeftFoot], [jointsNames.RightLowerLeg, jointsNames.RightFoot], 
    [jointsNames.Hip, jointsNames.Spine], [jointsNames.Spine, jointsNames.Chest], [jointsNames.Chest, jointsNames.UpperChest], 
    [jointsNames.UpperChest, jointsNames.LeftUpperArm], [jointsNames.UpperChest, jointsNames.RightUpperArm], [jointsNames.UpperChest, jointsNames.Head], 
    [jointsNames.LeftUpperArm, jointsNames.LeftLowerArm], [jointsNames.RightUpperArm, jointsNames.RightLowerArm], 
    [jointsNames.LeftLowerArm, jointsNames.LeftHand], [jointsNames.RightLowerArm, jointsNames.RightHand], 
]

# 手部骨架結構 
handBoneStructure = [
    [handJointsNames.wrist, handJointsNames.thunmbCMC], [handJointsNames.wrist, handJointsNames.indexMCP], 
    [handJointsNames.wrist, handJointsNames.middleMCP], [handJointsNames.wrist, handJointsNames.ringMCP], 
    [handJointsNames.wrist, handJointsNames.pinkyMCP], 
    [handJointsNames.thunmbCMC, handJointsNames.thunmbMCP], [handJointsNames.thunmbMCP, handJointsNames.thunmbIP],
    [handJointsNames.thunmbIP, handJointsNames.thunmbTIP], 
    [handJointsNames.indexMCP, handJointsNames.indexPIP], [handJointsNames.indexPIP, handJointsNames.indexDIP],
    [handJointsNames.indexDIP, handJointsNames.indexTIP], 
    [handJointsNames.middleMCP, handJointsNames.middlePIP], [handJointsNames.middlePIP, handJointsNames.middleDIP],
    [handJointsNames.middleDIP, handJointsNames.middleTIP], 
    [handJointsNames.ringMCP, handJointsNames.ringPIP], [handJointsNames.ringPIP, handJointsNames.ringDIP],
    [handJointsNames.ringDIP, handJointsNames.ringTIP], 
    [handJointsNames.pinkyMCP, handJointsNames.pinkyPIP], [handJointsNames.pinkyPIP, handJointsNames.pinkyDIP],
    [handJointsNames.pinkyDIP, handJointsNames.pinkyTIP], 
]

# 目前顯示的frame index
curTimeInd = 0

def readAppliedRotPos(rotAppliedFilePath='../positionData/leftFrontKick_quat_directMapping.json', TPoseFilePath = ''):
    # 可以參考testingStageViz.py 
    with open(rotAppliedFilePath, 'r') as WFile:
        lowerBodyPosition=json.load(WFile)
    timeCount = len(lowerBodyPosition)
    jointsIndices = list(lowerBodyPosition[0]['data'].keys())
    lowerBodyPosition = [{'data': {int(i): lowerBodyPosition[t]['data'][i] for i in jointsIndices}} for t in range(timeCount)]
    lowerBodyDf = jsonToDf(lowerBodyPosition)
    concatDfs = pd.concat(lowerBodyDf.values(), axis=1)
    concatArr = concatDfs.values
    concatArr = concatArr.reshape((concatArr.shape[0], -1, 3))
    print('lower body joint indices:', jointsIndices)
    print('timeCount: ', timeCount)
    print('lower body array shape', concatArr.shape)
    # Hip校正為原點
    concatArr = concatArr - concatArr[:, -1:, :]

    # read T pose 
    # Ref: realTimeRotToAvatarPos.py 
    saveDirPath = '../positionData/fromDB/genericAvatar/'
    TPoseJson = None
    with open(saveDirPath+'TPose.json', 'r') as fileIn:
        TPoseJson = json.load(fileIn)['results']
    jointCount = len(TPoseJson[0]['data'])
    TPoseDfs = jsonToDf(TPoseJson)
    TPoseConcatDfs = pd.concat(TPoseDfs.values(), axis=1)
    TPoseArr = TPoseConcatDfs.values
    TPoseArr = TPoseArr.reshape((TPoseArr.shape[0], -1, 3))
    TPoseArr = TPoseArr[3:4, :, :]  # Just dont use the first frame, since it might be wrong 
    TPoseArr = np.repeat(TPoseArr, concatArr.shape[0], axis=0)
    print('T Pose joint count: ', jointCount)
    print('T Pose time count: ', TPoseArr.shape[0])
    print('T Pose array shape: ', TPoseArr.shape)
    # Hip校正回原點
    TPoseArr = TPoseArr - TPoseArr[:, 6:7, :]

    # combine T pose to lower body motions 
    retArr = np.concatenate((concatArr, TPoseArr[:, 7:, :]), axis=1)
    print('return array shape: ', retArr.shape)

    return retArr

def readSynthesisPos(synthesisFilePath = '../positionData/afterSynthesis/NoVelAccOverlap/leftFrontKick_quat_direct_075_EWMA.json'):
    # TODO 
    # read synthesis result
    # TODO: 好像會少一個點, 應該就是少hip, 應為預設他就是0, 需要把hip是0補回來 
    synthesisPos = None
    with open(synthesisFilePath, 'r') as WFile:
        synthesisPos=json.load(WFile)
    timeCount = len(synthesisPos)
    jointCount = len(synthesisPos[0]['data'])
    synthesisDfs = jsonToDf(synthesisPos)
    synthesisConcatDfs = pd.concat(synthesisDfs.values(), axis=1)
    synthesisArr = synthesisConcatDfs.values
    synthesisArr = synthesisArr.reshape((synthesisArr.shape[0], -1, 3))
    print('time count: ', timeCount)
    print('joint count: ', jointCount)

    # 補Hip是0回array當中
    _hip = np.zeros((synthesisArr.shape[0], 3))
    synthesisArr = np.insert(synthesisArr, 6, _hip, axis=1)
    print('output array shape: ', synthesisArr.shape)
    return synthesisArr

def readHandPos(
    handLandmarkFilePath = '../complexModel/newRecord/twoLegJump_rgb.json', 
    scale = [1, 1, 1], negate = [False, False, False]
):
    handPos = None
    with open(handLandmarkFilePath, 'r') as RFile:
        handPos = json.load(RFile)
    timeCount = len(handPos)
    jointCount = len(handPos[0]['data'])
    handDfs = jsonToDf(handPos)
    handConcatDfs = pd.concat(handDfs.values(), axis=1)
    handArr = handConcatDfs.values
    handArr = handArr.reshape((handArr.shape[0], -1, 3))
    print('time count: ', timeCount)
    print('joint count: ', jointCount)
    print('output array shape: ', handArr.shape)

    # 轉換成wrist在原點
    handArr = handArr - handArr[:, handJointsNames.wrist:handJointsNames.wrist+1, :]
    # 放大數值範圍
    for _axis, _scale in enumerate(scale):
        handArr[:, :, _axis] = handArr[:, :, _axis] * _scale
    # 反轉坐標軸
    for _axis, _neg in enumerate(negate):
        if _neg:
            handArr[:, :, _axis] = handArr[:, :, _axis] * -1
    return handArr

def readExampleAnimPos(filePath = ''):
    print('read example animation')
    # read example animation positions 
    synthesisPos = None
    with open(filePath, 'r') as WFile:
        synthesisPos=json.load(WFile)['results']
    timeCount = len(synthesisPos)
    jointCount = len(synthesisPos[0]['data'])
    synthesisDfs = jsonToDf(synthesisPos)
    synthesisConcatDfs = pd.concat(synthesisDfs.values(), axis=1)
    synthesisArr = synthesisConcatDfs.values
    synthesisArr = synthesisArr.reshape((synthesisArr.shape[0], -1, 3))

    # Hip校正回原點
    synthesisArr = synthesisArr - synthesisArr[:, 6:7, :]
    print('time count: ', timeCount)
    print('joint count: ', jointCount)
    print('output array shape: ', synthesisArr.shape)
    return synthesisArr

def vizMotions(jointData, jointHeirarchy, hipPos, axisJointInd, frameRate=0.05, return_imgs=False):
    '''
    Objective
        輸入joint data, 使用open 3d顯示骨架運動資訊
    Input 
        - jointData: list of 3 dimention array. 多組關節點的3d數值. time * number of joints * 3 
        - jointHeirarchy: list of lists. 關節點連結資訊. each element is a list with two numbers in it. 
            First number indicates parent joint's index, the other indicates child joint's index. 
        - hipPos: list of 1 dimention array. 每一組關節點的hip位置. XYZ values in the array.  
        - axisJointInd: list of list. 哪一些點需要在它上面繪製座標軸
        - frameRate: 更新率
    ''' 
    # 輸入joint data, 使用open 3d顯示骨架運動資訊 

    # create visualizer and window.
    # vis = o3d.visualization.Visualizer()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    # vis.create_window(height=480, width=640)
    vis.create_window(height=768, width=1024)
    opt = vis.get_render_option()
    opt.point_size = 10
    opt.line_width = 5

    # 註冊callback function, 讓按鍵觸發可以被偵測
    ## 印出當前顯示的frame index -> 按下空格鍵 
    vis.register_key_callback(ord(' '), printCurFrameIdxCallback)

    # create axis at origin 
    origin_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    vis.add_geometry(origin_axis)

    # 根據hipPos改變每一組關節點的3d positions資訊
    for i, _hipPos in enumerate(hipPos):
        jointData[i] = jointData[i] + _hipPos[np.newaxis, np.newaxis, :]

    # 在部分節點繪製坐標軸標示 (e.g. 下半身的upper leg and lower leg)
    axisFrameList = []
    axisFrameOriginPosList = []
    for i, _indList in enumerate(axisJointInd):
        _axis_frame_list = []
        _axis_frame_pos_list = []
        for _ind in _indList:
            _axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1, origin=jointData[i][0, _ind, :]
            )
            vis.add_geometry(_axis_frame)
            _axis_frame_list.append(_axis_frame)
            _axis_frame_pos_list.append(np.array(_axis_frame.vertices))
        axisFrameList.append(_axis_frame_list)
        axisFrameOriginPosList.append(_axis_frame_pos_list)
    
    # initial point clouds list 
    # initial line set list
    pcdList = []
    lineSetList = []
    for _jointData, _jointHeirarchy in zip(jointData, jointHeirarchy):
        _pcd = o3d.geometry.PointCloud()
        _pcd.points = o3d.utility.Vector3dVector(_jointData[0, :, :])
        pcdList.append(_pcd)

        colors = [[1, 0, 0] for i in range(len(_jointHeirarchy))]
        _line_set = o3d.geometry.LineSet()
        _line_set.points = o3d.utility.Vector3dVector(_jointData[0, :, :])
        _line_set.lines = o3d.utility.Vector2iVector(_jointHeirarchy)
        _line_set.colors = o3d.utility.Vector3dVector(colors)
        lineSetList.append(_line_set) 
        # print(_jointData[0, :, :])
        # print(np.asarray(_line_set.points).shape)
        # exit()

        vis.add_geometry(_pcd)
        vis.add_geometry(_line_set)
    
    # 將右腿的顏色改成與其他點不同, 並且把線段/骨頭顏色改為黑色
    # 右腿的三個點的index = {0, 1, 2}, 三個骨頭index = {0, 2, 4}
    # 假設前面兩個輸入的point cloud是full body point cloud 
    # bodyPointCloudInd = [0, 1]
    bodyPointCloudInd = [0]
    bodyBoneInd = [0, 2, 4]
    bodyJointInd = [0, 1, 2]
    for _pcdInd in bodyPointCloudInd:
        _colors = np.array([[0, 0, 0] for i in range(17)])
        _boneColors = np.array([[0, 0, 0] for i in range(16)])
        for i in bodyBoneInd:
            _boneColors[i, :] = [1, 0, 0] 
        lineSetList[_pcdInd].colors = o3d.utility.Vector3dVector(_boneColors)
        for i in bodyJointInd:
            _colors[i, :] = [1, 0, 0] 
        pcdList[_pcdInd].colors = o3d.utility.Vector3dVector(_colors)

    # 手的食指的點與線段也要修改顏色
    # 食指四個點index = {5, 6, 7, 8}, 三個骨頭的index = {8, 9, 10}
    # handPointCloudInd = [2]
    # handBoneInd = [8, 9, 10]
    # handJointInd = [5, 6, 7, 8]
    # _colors = np.array([[0, 0, 0] for i in range(21)])
    # _boneColors = np.array([[0, 0, 0] for i in range(20)])
    # for i in handBoneInd:
    #     _boneColors[i, :] = [1, 0, 0] 
    # for i in handJointInd:
    #     _colors[i, :] = [1, 0, 0] 
    # for i in handPointCloudInd:
    #     pcdList[i].colors = o3d.utility.Vector3dVector(_colors)
    #     lineSetList[i].colors = o3d.utility.Vector3dVector(_boneColors)

    # Trajectory改變顏色, 改成藍色
    trajectoryInd = [1]
    for _pcdInd in trajectoryInd:
        _colors = np.array([[0, 0, 1] for i in range(jointData[_pcdInd].shape[1])])
        pcdList[_pcdInd].colors = o3d.utility.Vector3dVector(_colors)
        
    
    # 紀錄visualize的影像畫面
    imgs = []

    # to add new points each dt secs.
    dt = frameRate
    previous_t = time.time()
    global curTimeInd
    curTimeInd = 0

    # run non-blocking visualization. 
    # To exit, press 'q' or click the 'x' of the window.
    keep_running = True
    while keep_running:
        
        if time.time() - previous_t > dt:

            # update joints
            # update bones
            # TODO update coordinate frame
            for i, _jointData in enumerate(jointData):
                pcdList[i].points = o3d.utility.Vector3dVector(_jointData[curTimeInd, :, :])
                vis.update_geometry(pcdList[i])

                lineSetList[i].points = o3d.utility.Vector3dVector(_jointData[curTimeInd, :, :])
                vis.update_geometry(lineSetList[i])

                for _j, _axisJointInd in enumerate(axisJointInd[i]):
                    axisFrameList[i][_j].vertices = \
                        o3d.utility.Vector3dVector(
                            axisFrameOriginPosList[i][_j] - _jointData[0, _axisJointInd:_axisJointInd+1, :] + _jointData[curTimeInd, _axisJointInd:_axisJointInd+1, :]
                        )
                    vis.update_geometry(axisFrameList[i][_j])

            previous_t = time.time()
            curTimeInd += 1
            if curTimeInd >= jointData[0].shape[0]:
                break
            if return_imgs:
                imgs.append(vis.capture_screen_float_buffer())
            

        keep_running = vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()

    # return 擷取的圖片
    if return_imgs:
        imgs = [np.asarray(i) for i in imgs]
        return imgs
    pass

def printCurFrameIdxCallback(vis):
    global curTimeInd
    print('cur frame index: ', curTimeInd)
    return True

def main():
    # create visualizer and window.
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=480, width=640)

    # create axis at origin 
    origin_axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis.add_geometry(origin_axis)

    # create plane by box mesh 
    # Ref: https://github.com/isl-org/Open3D/issues/3618 
    plane = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=0.01)
    vis.add_geometry(plane)

    # initialize pointcloud instance.
    pcd = o3d.geometry.PointCloud()
    # *optionally* add initial points
    points = np.random.rand(10, 3)
    pcd.points = o3d.utility.Vector3dVector(points)

    # initialize line set
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1],
              [0, 1, 1], [1, 1, 1]]).astype(float)
    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set)

    # include it in the visualizer before non-blocking visualization.
    vis.add_geometry(pcd)


    # to add new points each dt secs.
    dt = 0.05
    # number of points that will be added
    n_new = 10

    previous_t = time.time()

    # run non-blocking visualization. 
    # To exit, press 'q' or click the 'x' of the window.
    keep_running = True
    while keep_running:
        
        if time.time() - previous_t > dt:
            s = time.time()

            # Options (uncomment each to try them out):
            # 1) extend with ndarrays.
            pcd.points.extend(np.random.rand(n_new, 3))

            # 取代舊的points
            # print(np.asarray(pcd.points).shape[0])
            if np.asarray(pcd.points).shape[0] > 10:
                pcd.points = o3d.utility.Vector3dVector(np.random.rand(n_new, 3))

            # 線段當中某個點的x, y, z軸不斷增加
            points[0,0]+=0.001
            points[0,1]+=0.001
            points[0,2]+=0.001
            line_set.points = o3d.utility.Vector3dVector(points)
            # print(np.asarray(line_set.points).shape)
            
            # 2) extend with Vector3dVector instances.
            # pcd.points.extend(
            #     o3d.utility.Vector3dVector(np.random.rand(n_new, 3)))
            
            # 3) other iterables, e.g
            # pcd.points.extend(np.random.rand(n_new, 3).tolist())
            
            vis.update_geometry(pcd)
            vis.update_geometry(line_set)
            previous_t = time.time()
            

        keep_running = vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()

    pass

# 目標是展示全身動作, 並且也展示腳的trajectory 
# 正在考慮要不要實作 
if __name__=='__main01__':
    # example animation joints position隨時間變動資料, 透過重複相同資料延長播放時間
    exampleAnimMotion = readExampleAnimPos('../positionData/fromDB/genericAvatar/leftFrontKickPositionFullJointsWithHead_withoutHip_075_quat_direct_normalized.json')
    exampleAnimMotion = np.tile(exampleAnimMotion, (3, 1, 1))
    # 只有腳的部分隨時間變動資料, 重複複製原始frame數量的次數 
    frameCount = exampleAnimMotion.shape[0]
    leftFootTrajectory = exampleAnimMotion[:, 2, :]
    leftFootTrajectory = np.tile(leftFootTrajectory, (frameCount, 1, 1))

    # 修改成顯示單一frame 
    spFrameInd = 80
    # spFrameInd = 105
    repeatShowingTime = 10000
    exampleAnimMotion = np.repeat(
        exampleAnimMotion[spFrameInd:spFrameInd+1, :, :], 
        repeatShowingTime, 
        axis=0
    )
    leftFootTrajectory = np.repeat(
        leftFootTrajectory[spFrameInd:spFrameInd+1, :, :], 
        repeatShowingTime, 
        axis=0
    )

    vizMotions(
        [exampleAnimMotion, leftFootTrajectory], 
        [fullBodyBoneStrcuture, [[5, 10]]], 
        [np.array([0, 0, 0.5]), np.array([0, 0, 0.5])], 
        [[], [], []],
        0.05
    )
    pass

# 最初始的展示. 展示手的作與全身的動作
if __name__=='__main__':
    # main()
    appliedRotMotion = readAppliedRotPos('../positionData/runSprintAndFrontKick_3_2_5_quat_directMapping.json')
    # appliedRotMotion = readAppliedRotPos('../positionData/leftSideKick_quat_directMapping.json')
    # 因為synthesis motion會少前面10個frame, 所以applied rotation版本需要捨去前面10個frame
    appliedRotMotion = appliedRotMotion[1081:, :, :]
    appliedRotMotion = appliedRotMotion[10:, :, :]
    
    
    synthesisMotion = readSynthesisPos('../positionData/afterSynthesis/NoVelAccOverlap/runSprintAndFrontKick_3_2_5_quat_direct_EWMA.json')
    # exampleAnimMotion = readExampleAnimPos('../positionData/fromDB/genericAvatar/leftSideKickPositionFullJointsWithHead_withoutHip.json')
    # synthesisMotion = exampleAnimMotion
    synthesisMotion = synthesisMotion[1081:, :, :]

    fingerMotion = readHandPos('../complexModel/newRecord/runSprintAndFrontKick_3_2_5.json', scale=[3.5, 1.5, 7], negate=[True, True, True])
    # fingerMotion = fingerMotion[2600:3500, :, :]    # 側踢只有選部分區間的資料
    fingerMotion = fingerMotion[1081:, :, :]
    fingerMotion = fingerMotion[10:, :, :]
    capture_imgs = vizMotions(
        [appliedRotMotion, synthesisMotion, fingerMotion], 
        [fullBodyBoneStrcuture, fullBodyBoneStrcuture, handBoneStructure], 
        [np.array([0, 0, 0.5]), np.array([-1, 0, 0.5]), np.array([1, 0, 0.5])], 
        # [[], [jointsNames.LeftUpperLeg, jointsNames.LeftLowerLeg], [handJointsNames.indexMCP, handJointsNames.indexPIP]],
        [[], [], []],
        0.03, 
        True
    )

    print(capture_imgs[0].shape)
    # 輸出骨架運動過程的影片
    # import imageio
    # outputVideoFilePath = 'outputVideo.mp4'
    # imageio.mimwrite(outputVideoFilePath, capture_imgs, fps=25, quality=8)

    # 顯示單一個frame的資訊, 方便拍攝論文的展示圖片 
    # specificFrameInd = 57
    # # specificFrameInd = 67
    # repeatShowingTime = 10000
    # appliedRotMotion = np.repeat(
    #     appliedRotMotion[specificFrameInd:specificFrameInd+1, :, :], repeatShowingTime, axis=0
    # )
    # synthesisMotion = np.repeat(
    #     synthesisMotion[specificFrameInd:specificFrameInd+1, :, :], repeatShowingTime, axis=0
    # )
    # fingerMotion = np.repeat(
    #     fingerMotion[specificFrameInd:specificFrameInd+1, :, :], repeatShowingTime, axis=0
    # )
    # vizMotions(
    #     [appliedRotMotion, synthesisMotion, fingerMotion], 
    #     [fullBodyBoneStrcuture, fullBodyBoneStrcuture, handBoneStructure], 
    #     [np.array([-5, 0, 0.5]), np.array([-8, 0, 0.5]), np.array([5, 0, 0.5])], 
    #     # [[], [jointsNames.LeftUpperLeg, jointsNames.LeftLowerLeg], [handJointsNames.indexMCP, handJointsNames.indexPIP]],
    #     [[], [], []],
    #     0.05
    # )

    # 錄製只有全身與手的立正狀態對比, 使用front kick放鬆姿態作為例子. 
    # 需要繪製座標軸在特定的joint上方, 兩者需要距離近一些 
    # vizMotions(
    #     [synthesisMotion, fingerMotion], 
    #     [fullBodyBoneStrcuture, handBoneStructure], 
    #     [np.array([-0.5, 0, 0.5]), np.array([0.5, 0, 0.5])], 
    #     [[jointsNames.LeftUpperLeg, jointsNames.LeftLowerLeg], [handJointsNames.indexMCP, handJointsNames.indexPIP]],
    #     0.05
    # )