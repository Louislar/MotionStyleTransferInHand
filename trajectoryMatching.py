'''
利用transformation matrix, 
將hand mapped trajectory以及body trajectory對齊
最後, 將對齊前後的body trajectory以及hand mapped trajectory畫出來
'''

import pandas as pd 
import numpy as np 
from scipy.spatial.transform import Rotation
import json 
import os 
from testingStageViz import jsonToDf
from realTimePositionSynthesis import readDBEncodedMotionsFromFile, fullPositionsJointCount

def main():
    # 1. read hand mapped positions
    # 2. read body positions
    # 3. construct transformation matrix
    ## ref: https://towardsdatascience.com/the-one-stop-guide-for-transformation-matrices-cea8f609bdb1
    # 3.1 apply transformation matrix to body trajectory
    # 4. visualize result 
    # (including before and after applying transformation trjectory and hand trajectory)

    handMappedPosDirPath = 'positionData/fromAfterMappingHand/'
    body3dPosDirPath = 'DBPreprocFeatVec/leftFrontKick_withoutHip_075/3DPos/'
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
    # TODO: 
    # rotation in x, y, z order
    rotationAngles = []
    # transition in x, y, z order 
    transitionValues = []
    

    pass

if __name__=='__main__':
    main()