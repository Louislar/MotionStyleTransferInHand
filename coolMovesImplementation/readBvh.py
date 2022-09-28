from genericpath import isdir
import os
import matplotlib.pyplot as plt 
from pymo.parsers import BVHParser
from pymo.viz_tools import *
from pymo.preprocessing import *

def extractDataFromMocapData(mocapDataDf, targetJoints):
    '''
    MocapData is a data structure defined in the pymo library. 
    I only want to extract the XYZ positions from some joints in the MocapData. 
    Easy access points is that the MocapData.values is a pd.DataFrame. 

    The desire joints: 
    1. left/right feet
    '''
    jointsDataDict = {_nm: None for _nm in targetJoints}
    for _nm in targetJoints:
        xPosColNM = _nm + '_Xposition'
        yPosColNM = _nm + '_Yposition'
        zPosColNM = _nm + '_Zposition'
        jointsDataDict[_nm] = mocapDataDf[[xPosColNM, yPosColNM, zPosColNM]].rename(columns={
            xPosColNM: 'x', yPosColNM: 'y', zPosColNM: 'z'
        })
    return jointsDataDict

def parseBvhDataToDf(bvhFilePath):
    '''
    Parse Bvh file into multiple DataFrames. 
    Each DataFrame represents a joint's 3d positions through time.
    '''
    # Parse bvh into MocapData defined in PyMo
    parser = BVHParser()
    parsed_data = parser.parse(bvhFilePath)
    # print_skel(parsed_data)
    mp = MocapParameterizer('position')

    positions = mp.fit_transform([parsed_data])

    # print(type(positions))
    # print(len(positions))
    # print(type(positions[0].values))
    # print(positions[0].values.shape)
    # print(positions[0].values.columns)
    # print(positions[0].skeleton.keys())

    # Parse data into 3d positions time series
    jointsPosData = extractDataFromMocapData(positions[0].values, list(positions[0].skeleton.keys()))
    # print(jointsPosData['Hips'].head(10))

    return jointsPosData

def findMocapDataBonePair(mocapData):
    bonePairs = []
    stack = [mocapData.root_name]
    while stack:
        joint = stack.pop()
        # print('%s- %s (%s)'%('| ', joint, mocapData.skeleton[joint]['parent']))
        bonePairs.append([joint, mocapData.skeleton[joint]['parent']])  # First element is a joint and second element is its parent joint
        for c in mocapData.skeleton[joint]['children']:
            stack.append(c)
    return bonePairs

# Output/obtain heirarchy of the skeleton
if __name__=='__main01__':
    filePath = 'data/swimming/126/126_01.bvh'
    parser = BVHParser()
    parsed_data = parser.parse(filePath)
    # print_skel(parsed_data)
    # Extract skeleton chain
    bonePairs = findMocapDataBonePair(parsed_data)
    # Store skeleton chain in DataFrame and save to csv
    skeletonDf = pd.DataFrame(
        {
            'joint':[_p[0] for _p in bonePairs], 
            'parent': [_p[1] for _p in bonePairs]
        }
    )
    # print(skeletonDf)
    skeletonDf.to_csv('data/skeleton.csv', index=False)

# Parse all the bvh files in a directory, store the parsed 3d informations in csv
if __name__=='__main__':

    # Parse file path
    # bvhDirPath = 'data/swimming/125/'
    # bvhDirPath = 'data/swimming/126/'
    # bvhDirPath = 'data/swimming/79/'
    bvhDirPath = 'data/swimming/80/'
    bvhFiles = os.listdir(bvhDirPath)
    bvhFilesPaths = [os.path.join(bvhDirPath, _file) for _file in bvhFiles]
    print('Read bvh directory path: ', bvhDirPath)
    print('Read bvh Files', bvhFilesPaths)

    ## Output path
    bvhDirPath = os.path.dirname(bvhDirPath)
    outputDirPath = bvhDirPath + '_parsed'
    bvhFilesWithoutExt = [os.path.splitext(_file)[0] for _file in bvhFiles]
    outputDirsPaths = [os.path.join(outputDirPath, _file) for _file in bvhFilesWithoutExt]
    print('Output directory path: ', outputDirPath)
    print('Output directories paths: ', outputDirsPaths)

    ## Check if output dir path exist. If not, create one.
    for _dirPath in outputDirsPaths:
        if os.path.isdir(_dirPath):
            print(_dirPath)
            print('exist')
        else:
            print(_dirPath)
            os.makedirs(_dirPath)
            print('not exist')

    # Loop for all the files
    for bvhFilePath, saveDirPath in zip(bvhFilesPaths, outputDirsPaths):
        # Parse bvh into multiple DataFrames
        jointsPosData = parseBvhDataToDf(bvhFilePath)

        # Save data in files
        for _jointNM in jointsPosData.keys():
            print('saved: ', os.path.join(saveDirPath, _jointNM+'.csv'))
            jointsPosData[_jointNM].to_csv(os.path.join(saveDirPath, _jointNM+'.csv'), index=False)    

    # plt.show()