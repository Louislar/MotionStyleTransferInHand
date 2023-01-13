'''
functions for visualization
'''

import numpy as np 
import matplotlib.pyplot as plt 
import json 
import sys
sys.path.append("../")
from rotationAnalysis import rotationJsonDataParser 

def main(rot): 

    jointInd = 0
    axisName = 'x'
    
    print(type(rot[jointInd][axisName]))
    print(len(rot[jointInd][axisName]))

    globMinInd = np.argmin(rot[jointInd][axisName])
    globMaxInd = np.argmax(rot[jointInd][axisName])
    arbitraryStartInd = 87
    arbitraryEndInd = 104
    print('global max: ', globMaxInd)
    print('global min: ', globMinInd)
    print('global max value: ', rot[jointInd][axisName][globMaxInd])
    print('global min value: ', rot[jointInd][axisName][globMinInd])
    print('arbitrary start value: ', rot[jointInd][axisName][arbitraryStartInd])
    print('arbitrary end value: ', rot[jointInd][axisName][arbitraryEndInd])

    # plot and show maximum and minimum positions
    plt.figure()
    plt.plot(range(len(rot[jointInd][axisName])), rot[jointInd][axisName])
    plt.plot(globMinInd, rot[jointInd][axisName][globMinInd], '.r', label='min')
    plt.plot(globMaxInd, rot[jointInd][axisName][globMaxInd], '.m', label='max')
    plt.plot(arbitraryStartInd, rot[jointInd][axisName][arbitraryStartInd], '.', label='start')
    plt.plot(arbitraryEndInd, rot[jointInd][axisName][arbitraryEndInd], '.', label='end')
    plt.legend()
    plt.show()
    pass

if __name__=='__main__':
    # read hand rotation 
    handRotationFilePath = '../bodyDBRotation/genericAvatar/quaternion/runInjured0.03_05_withHip.json'
    # handRotationFilePath = '../HandRotationOuputFromHomePC/runSprintStream.json'
    handJointsRotations=None
    with open(handRotationFilePath, 'r') as fileOpen: 
        rotationJson=json.load(fileOpen)
        handJointsRotations = rotationJsonDataParser(rotationJson, jointCount=4)    # For Unity output
        # handJointsRotations = rotationJsonDataParser({'results': rotationJson}, jointCount=4)    # For python output
    main(handJointsRotations)