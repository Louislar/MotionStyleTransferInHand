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

def readSynthesisPos():
    # TODO 
    pass

def vizMotions(jointData, jointHeirarchy, hipPos, frameRate=0.05):
    '''
    Objective
        輸入joint data, 使用open 3d顯示骨架運動資訊
    Input 
        - jointData: list of 3 dimention array. 多組關節點的3d數值. time * number of joints * 3 
        - jointHeirarchy: list of lists. 關節點連結資訊. each element is a list with two numbers in it. 
            First number indicates parent joint's index, the other indicates child joint's index. 
        - hipPos: list of 1 dimention array. 每一組關節點的hip位置. XYZ values in the array.  
        - frameRate: 更新率
    ''' 
    # 輸入joint data, 使用open 3d顯示骨架運動資訊 

    # create visualizer and window.
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=480, width=640)
    opt = vis.get_render_option()
    opt.point_size = 10
    opt.line_width = 5

    # create axis at origin 
    origin_axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis.add_geometry(origin_axis)

    # TODO 根據hipPos改變每一組關節點的3d positions資訊
    for i, _hipPos in enumerate(hipPos):
        jointData[i] = jointData[i] + _hipPos[np.newaxis, np.newaxis, :]
    
    # initial point clouds list 
    # initial line set list
    pcdList = []
    lineSetList = []
    for _jointData in jointData:
        _pcd = o3d.geometry.PointCloud()
        _pcd.points = o3d.utility.Vector3dVector(_jointData[0, :, :])
        pcdList.append(_pcd)

        colors = [[1, 0, 0] for i in range(len(jointHeirarchy))]
        _line_set = o3d.geometry.LineSet()
        _line_set.points = o3d.utility.Vector3dVector(_jointData[0, :, :])
        _line_set.lines = o3d.utility.Vector2iVector(jointHeirarchy)
        _line_set.colors = o3d.utility.Vector3dVector(colors)
        lineSetList.append(_line_set) 
        # print(_jointData[0, :, :])
        # print(np.asarray(_line_set.points).shape)
        # exit()

        vis.add_geometry(_pcd)
        vis.add_geometry(_line_set)

    # to add new points each dt secs.
    dt = frameRate
    previous_t = time.time()
    curTimeInd = 0

    # run non-blocking visualization. 
    # To exit, press 'q' or click the 'x' of the window.
    keep_running = True
    while keep_running:
        
        if time.time() - previous_t > dt:

            # update joints
            # update bones
            for i, _jointData in enumerate(jointData):
                pcdList[i].points = o3d.utility.Vector3dVector(_jointData[curTimeInd, :, :])
                vis.update_geometry(pcdList[i])

                lineSetList[i].points = o3d.utility.Vector3dVector(_jointData[curTimeInd, :, :])
                vis.update_geometry(lineSetList[i])

            previous_t = time.time()
            curTimeInd += 1
            

        keep_running = vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()

    pass

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

if __name__=='__main__':
    # main()
    appliedRotMotion = readAppliedRotPos()
    vizMotions(
        [appliedRotMotion], 
        fullBodyBoneStrcuture, 
        [np.array([0, 0.5, 0.5])], 
        0.05
    )