import os
import matplotlib.pyplot as plt 
from pymo.parsers import BVHParser
from pymo.viz_tools import *
from pymo.preprocessing import *

def extractDataFromMocapData(mocapDataDf, targetJoints):
    '''
    TODO: 
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



bvhFilePath = 'data/swimming/125/125_02.bvh'
# TODO: 一次處理多個bvh檔案的轉檔
bvhFiles = []

if __name__=='__main__':

    # Parse file path
    bvhFileNM = os.path.basename(bvhFilePath)
    bvhFileNM_withoutEXT = os.path.splitext(bvhFileNM)[0]
    bvhDirPath = os.path.dirname(bvhFilePath)
    bvhDirParentPath = os.path.dirname(bvhDirPath)
    bvhDirNM = os.path.basename(bvhDirPath)
    print('Read bvh directory path: ', bvhDirPath)

    # Parse bvh
    parser = BVHParser()
    parsed_data = parser.parse(bvhFilePath)
    print_skel(parsed_data)
    mp = MocapParameterizer('position')

    positions = mp.fit_transform([parsed_data])

    print(type(positions))
    print(len(positions))
    print(type(positions[0].values))
    print(positions[0].values.shape)
    print(positions[0].values.columns)
    print(positions[0].skeleton.keys())

    # Parse data into 3d positions time series
    jointsPosData = extractDataFromMocapData(positions[0].values, list(positions[0].skeleton.keys()))
    print(jointsPosData['Hips'].head(10))

    # Save data in files
    saveDirPath = 'data/swimming/125_parsed/'
    saveDirPath = os.path.join(bvhDirParentPath, bvhDirNM+'_parsed')
    saveDirPath = os.path.join(saveDirPath, bvhFileNM_withoutEXT)
    print('Save directory path: ', saveDirPath)
    for _jointNM in jointsPosData.keys():
        print('saved: ', os.path.join(saveDirPath, _jointNM+'.csv'))
        jointsPosData[_jointNM].to_csv(os.path.join(saveDirPath, _jointNM+'.csv'), index=False)
        pass


    # draw_stickfigure(positions[0], frame=10)

    # plt.show()