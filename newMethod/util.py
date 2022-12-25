'''
各式雜項功能
'''

import numpy as np 
import matplotlib.pyplot as plt 
import json 
import sys
sys.path.append("../")
from rotationAnalysis import rotationJsonDataParser 

def readHandPerformance(filePath = '../HandRotationOuputFromHomePC/leftFrontKickStream.json'):
    '''
    讀取hand performance rotation data 
    '''
    handJointsRotations=None
    with open(filePath, 'r') as fileOpen: 
        rotationJson=json.load(fileOpen)
        handJointsRotations = rotationJsonDataParser({'results': rotationJson}, jointCount=4)    # For python output
    return handJointsRotations

def handPerformanceToMatrix():
    '''
    Make hand rotation data to a matrix with d * t dimension 
    每一行是所有維度的資料, 依序是joint 0 x, joint 0 y, joint 0 z, joint 1 x, .....
    總共的行數是多少個時間點 
    '''
    pass