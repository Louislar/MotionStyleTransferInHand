'''
Goal: 從MediaPipe抓取的手land mark轉換成rotation資訊, 
總共有6個rotation資訊, 
MCP兩個旋轉軸, PIP一個旋轉軸, 
index finger以及middle finger各有一個MCP and PIP
總共只會用到7個點, 但是MediaPipe會給21個點
'''

import numpy as np
import json
import enum
import time

class jointsNames(enum.IntEnum):
    wrist = 0
    thunmbCMC = 1
    thunmbMCP = 2
    thunmbIP = 3
    thunmbTIP = 4
    indexMCP = 5
    indexPIP = 6
    indexDIP = 7
    indexTIP = 8
    middleMCP = 9
    middlePIP = 10
    middleDIP = 11
    middleTIP = 12
    ringMCP = 13
    ringPIP = 14
    ringDIP = 15
    ringTIP = 16
    pinkyMCP = 17
    pinkyPIP = 18
    pinkyDIP = 19
    pinkyTIP = 20

usedJoints = [
    jointsNames.wrist, 
    jointsNames.indexMCP, jointsNames.indexPIP, jointsNames.indexDIP, 
    jointsNames.middleMCP, jointsNames.middlePIP, jointsNames.middleDIP
]

outputJointCount = 4

def visualizeHandModel():
    '''
    Goal: 顯示手部模型, 方便debug, 顯示部分joints以及vertex即可
    '''
    pass

def vectorProjOnPlane(v, n):
    '''
    Goal: TODO: project vector on plane
    Input:
    :v: Vector to be prjected, array with X,Y,Z values
    :n: Plane normal, array with X,Y,Z values
    '''
    n_norm = np.sqrt(sum(n**2))
    proj_of_v_on_n = (np.dot(v, n)/n_norm**2)*n
    return proj_of_v_on_n

def computeUsedVectors(positionData):
    '''
    Goal: 計算所需的向量數值, 
            indexWristToMCP, middleWristToMCP, 
            indexMCPToPIP, indexPIPToDIP, 
            middleMCPToPIP, middlePIPToDIP, 
            indexMCPNormal([9]-[5]), palmNormal(indexWristToMCP x indexMCPNormal), middlePalmNormal, 
            indexProjectToMCPNormal, indexProjectToPalmNormal, 
            middleProjectToMCPNormal, middleProjectToPalmNormal
    Input:
    :positionData: list of dicts, 單一時間點所有joints的XYZ 3d position資訊
    Output:
    :: list of arrays, 所有需要的vectors
    '''
    # 1. 得到所需的joint's XYZ 3D position資訊
    wristPos = np.array([
        positionData[jointsNames.wrist]['x'], positionData[jointsNames.wrist]['y'], positionData[jointsNames.wrist]['z']
    ])
    indexMCPPos = np.array([
        positionData[jointsNames.indexMCP]['x'], positionData[jointsNames.indexMCP]['y'], positionData[jointsNames.indexMCP]['z']
    ])
    indexPIPPos = np.array([
        positionData[jointsNames.indexPIP]['x'], positionData[jointsNames.indexPIP]['y'], positionData[jointsNames.indexPIP]['z']
    ])
    indexDIPPos = np.array([
        positionData[jointsNames.indexDIP]['x'], positionData[jointsNames.indexDIP]['y'], positionData[jointsNames.indexDIP]['z']
    ])
    middleMCPPos = np.array([
        positionData[jointsNames.middleMCP]['x'], positionData[jointsNames.middleMCP]['y'], positionData[jointsNames.middleMCP]['z']
    ])
    middlePIPPos = np.array([
        positionData[jointsNames.middlePIP]['x'], positionData[jointsNames.middlePIP]['y'], positionData[jointsNames.middlePIP]['z']
    ])
    middleDIPPos = np.array([
        positionData[jointsNames.middleDIP]['x'], positionData[jointsNames.middleDIP]['y'], positionData[jointsNames.middleDIP]['z']
    ])
    # 2. 向量數值計算
    indexWristToMCP = indexMCPPos - wristPos
    middleWristToMCP = middleMCPPos - wristPos
    indexMCPToPIP = indexPIPPos - indexMCPPos
    indexPIPToDIP = indexDIPPos - indexPIPPos
    middleMCPToPIP = middlePIPPos - middleMCPPos
    middlePIPToDIP = middleDIPPos - middlePIPPos
    indexMCPNormal = middleMCPPos - indexMCPPos
    indexMCPNormalNormalized = indexMCPNormal/np.linalg.norm(indexMCPNormal)
    # Note that numpy cross product is right-hand-rule
    palmNormal = np.cross(
        indexMCPNormalNormalized, # normalize the vector
        indexWristToMCP/np.linalg.norm(indexWristToMCP)
    )
    middlePalmNormal = np.cross(
        indexMCPNormalNormalized, 
        middleWristToMCP/np.linalg.norm(middleWristToMCP)
    )
    # Vector project to a plane: https://www.geeksforgeeks.org/vector-projection-using-python/
    indexProjectToMCPNormal = vectorProjOnPlane(indexMCPToPIP, indexMCPNormalNormalized)
    indexProjectToPalmNormal = vectorProjOnPlane(indexMCPToPIP, palmNormal)
    middleProjectToMCPNormal = vectorProjOnPlane(middleMCPToPIP, indexMCPNormalNormalized)
    middleProjectToPalmNormal = vectorProjOnPlane(middleMCPToPIP, middlePalmNormal)
    # print(indexProjectToMCPNormal)
    return [indexWristToMCP, middleWristToMCP, 
            indexMCPToPIP, indexPIPToDIP, 
            middleMCPToPIP, middlePIPToDIP, 
            indexMCPNormal, palmNormal, middlePalmNormal, 
            indexProjectToMCPNormal, indexProjectToPalmNormal, 
            middleProjectToMCPNormal, middleProjectToPalmNormal]

def angleBetweenTwoVecs(v1, v2, isSigned: bool=False, rotationDirection=None):
    '''
    Goal: compute the angle from vector1 to vector2
    Input: 
    :rotationDirection: The direction that decide the rotation's sign(right-hand-rule)
    '''
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)  # return angle in radian
    angleDegree = angle/ np.pi * 180
    if (isSigned) and (rotationDirection is not None):
        c = np.cross(v1, v2)
        # print('c: ', c)
        _dot_product = np.dot(c, rotationDirection)
        _angle = np.arccos(_dot_product)  # return angle in radian
        _angleDegree = _angle/ np.pi * 180
        # print('angle c and targetDirection: ', _angleDegree)
        if _angleDegree>90:
            angleDegree=-angleDegree
        # if c>0:
        #     angleDegree=-angleDegree
    return angleDegree

def computeUsedRotations(indexWristToMCP, middleWristToMCP, 
            indexMCPToPIP, indexPIPToDIP, 
            middleMCPToPIP, middlePIPToDIP, 
            indexMCPNormal, palmNormal, middlePalmNormal, 
            indexProjectToMCPNormal, indexProjectToPalmNormal, 
            middleProjectToMCPNormal, middleProjectToPalmNormal):
    '''
    Goal: 利用求得的vectors進一步計算所需的rotations
    Input:
    :usedVecs: list of arrays, [indexWristToMCP, middleWristToMCP, 
            indexMCPToPIP, indexPIPToDIP, 
            middleMCPToPIP, middlePIPToDIP, 
            indexMCPNormal, palmNormal, 
            indexProjectToMCPNormal, indexProjectToPalmNormal, 
            middleProjectToMCPNormal, middleProjectToPalmNormal]
    Output:
    :: list, rotation的計算結果
    '''
    # angle between teo vectors
    # ref: https://www.geeksforgeeks.org/how-to-find-the-angle-between-two-vectors/
    # ref: https://chadrick-kwag.net/get-rotation-angle-between-two-vectors/
    indexPIPAngle = angleBetweenTwoVecs(indexMCPToPIP, indexPIPToDIP)
    indexMCPAngle1 = angleBetweenTwoVecs(indexWristToMCP, indexProjectToMCPNormal, True, indexMCPNormal)
    indexMCPAngle2 = angleBetweenTwoVecs(indexWristToMCP, indexProjectToPalmNormal, True, palmNormal)

    middlePIPAngle = angleBetweenTwoVecs(middleMCPToPIP, middlePIPToDIP)
    middleMCPAngle1 = angleBetweenTwoVecs(middleWristToMCP, middleProjectToMCPNormal, True, indexMCPNormal)
    middleMCPAngle2 = angleBetweenTwoVecs(middleWristToMCP, middleProjectToPalmNormal, True, middlePalmNormal)

    return [indexPIPAngle, indexMCPAngle1, indexMCPAngle2, middlePIPAngle, middleMCPAngle1, middleMCPAngle2]

# For testing(plot the 3d vectors)
if __name__ == '__main01__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    origin = [0, 0, 0]
    x = [1, 2, 3]
    y = [4, 5, 6]
    xCy = np.cross(x, y)
    xyzZip = list(zip(*[origin+x, origin+y, origin+list(xCy)]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(xyzZip[0], xyzZip[1], xyzZip[2], xyzZip[3], xyzZip[4], xyzZip[5])
    ax.set_xlim([-1, 3])
    ax.set_ylim([-1, 3])
    ax.set_zlim([-1, 3])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

    pass

# For testing angle coputation
if __name__ == '__main01__':
    tmp1 = np.array([1,0])
    tmp2 = np.array([1,1])
    tmp3 = np.array([0,1])
    tmp4 = np.array([-1,1])
    tmp5 = np.array([-1,0])
    tmp6 = np.array([-1,-1])
    tmp7 = np.array([0,-1])
    tmp8 = np.array([1,-1])
    # print(angleBetweenTwoVecs(tmp1, tmp2, True))
    # print(angleBetweenTwoVecs(tmp1, tmp3, True))
    # print(angleBetweenTwoVecs(tmp1, tmp4, True))
    # print(angleBetweenTwoVecs(tmp1, tmp5, True))
    # print(angleBetweenTwoVecs(tmp1, tmp6, True))
    # print(angleBetweenTwoVecs(tmp1, tmp7, True))
    # print(angleBetweenTwoVecs(tmp1, tmp8, True))
    # right-hand-rule, index finger is X, middle finger is Y, thumb is Z
    # print(angleBetweenTwoVecs(np.array([1, 0, 0]), np.array([0, 1, 0]), True, np.array([0, 0, 1])))
    # print(angleBetweenTwoVecs(np.array([1, 0, 0]), np.array([0, 0, 1]), True, np.array([0, -1, 0])))
    # print(angleBetweenTwoVecs(np.array([1, 0, 0]), np.array([0, 0, 1]), True, np.array([0, 1, 0])))
    # print(angleBetweenTwoVecs(np.array([0, 0, 1]), np.array([1, 0, 0]), True, np.array([0, -1, 0])))
    # print(angleBetweenTwoVecs(np.array([0, 0, 1]), np.array([1, 0, 0]), True, np.array([0, 1, 0])))

if __name__ == '__main__':
    # 1. Read hand landmark data(only keep joints in used)
    # 1.1 make it a streaming data (already a streaming data)
    # 2. 計算向量
    #   2.1 手掌法向量
    # 3. 計算角度
    # 4. Store computed rotations

    # 1. 
    saveDirPath = 'complexModel/'
    handLMJson = None
    with open(saveDirPath+'frontKick.json', 'r') as fileOpen: 
        handLMJson=json.load(fileOpen)
    timeCount = len(handLMJson)
    print('time count: ', timeCount)
    for t in range(timeCount):
        # TODO: [暫緩]finish this. screen used joints' data
        pass
    # 2. 
    computedRotations = [None for t in range(timeCount)]
    rotComputeTimeLaps = np.zeros(timeCount)
    for t in range(timeCount):
        usedVecs = computeUsedVectors(handLMJson[t]['data'])
        targetRotations = computeUsedRotations(*usedVecs)
        computedRotations[t] = targetRotations
        rotComputeTimeLaps[t] = time.time()
        # break
    rotComputeCost = rotComputeTimeLaps[1:] - rotComputeTimeLaps[:-1]
    print('rotation map avg time: ', np.mean(rotComputeCost))
    print('rotation map time std: ', np.std(rotComputeCost))
    print('rotation map max time cost: ', np.max(rotComputeCost))
    print('rotation map min time cost: ', np.min(rotComputeCost))
    
    # 4. 
    # left upper leg(X, Z), left knee(X), right upper leg, right knee
    # X axis is the flexion, Z axis is the abduction
    rotComputeJsonData = [{'time': t, 'data': [{a: 0 for a in ['x', 'y', 'z']} for i in range(outputJointCount)]} for t in range(timeCount)]
    rotComputeRetSaveDirPath = 'HandRotationOuputFromHomePC/'
    for t in range(timeCount):
        rotComputeJsonData[t]['data'][0]['x'] = computedRotations[t][1]
        rotComputeJsonData[t]['data'][0]['z'] = computedRotations[t][2]
        rotComputeJsonData[t]['data'][1]['x'] = computedRotations[t][0]
        rotComputeJsonData[t]['data'][2]['x'] = computedRotations[t][4]
        rotComputeJsonData[t]['data'][2]['z'] = computedRotations[t][5]
        rotComputeJsonData[t]['data'][3]['x'] = computedRotations[t][3]
    # with open(rotComputeRetSaveDirPath+'leftFrontKickStream.json', 'w') as WFile: 
    #     json.dump(rotComputeJsonData, WFile)