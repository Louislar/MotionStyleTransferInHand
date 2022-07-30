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

def visualizeHandModel():
    '''
    Goal: 顯示手部模型, 方便debug, 顯示部分joints以及vertex即可
    '''
    pass

def vectorProjOnPlane():
    '''
    Goal: TODO: project vector on plane
    '''
    pass

def computeUsedVectors(positionData):
    '''
    Goal: 計算所需的向量數值, 
            indexWristToMCP, middleWristToMCP, 
            indexMCPToPIP, indexPIPToDIP, 
            middleMCPToPIP, middlePIPToDIP, 
            indexMCPNormal([9]-[5]), palmNormal(indexWristToMCP x indexMCPNormal)
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
    # Note that numpy cross product is right hand rule
    palmNormal = np.cross(
        indexMCPNormalNormalized, # normalize the vector
        indexWristToMCP/np.linalg.norm(indexWristToMCP)
    )
    # Vector project to a plane: https://www.geeksforgeeks.org/vector-projection-using-python/
    indexProjectToMCPNormal = None

    return 

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

if __name__ == '__main__':
    # 1. Read hand landmark data(only keep joints in used)
    # 1.1 make it a streaming data (already a streaming data)
    # 2. 計算向量
    #   2.1 手掌法向量
    # 4. 計算角度

    # 1. 
    saveDirPath = 'complexModel/'
    handLMJson = None
    with open(saveDirPath+'leftSideKick.json', 'r') as fileOpen: 
        handLMJson=json.load(fileOpen)
    timeCount = len(handLMJson)
    for t in range(timeCount):
        # TODO: finish this. screen used joints' data
        pass
    # 2. 
    computeUsedVectors(handLMJson, usedJoints)