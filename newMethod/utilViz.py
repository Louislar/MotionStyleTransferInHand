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
    
    print(type(rot[0]['x']))
    print(len(rot[0]['x']))

    globMinInd = np.argmin(rot[0]['x'])
    globMaxInd = np.argmax(rot[0]['x'])
    arbitraryStartInd = 1430
    arbitraryEndInd = 1530
    # plot and show maximum and minimum positions
    plt.figure()
    plt.plot(range(len(rot[0]['x'])), rot[0]['x'])
    plt.plot(globMinInd, rot[0]['x'][globMinInd], '.r', label='min')
    plt.plot(globMaxInd, rot[0]['x'][globMaxInd], '.m', label='max')
    plt.plot(arbitraryStartInd, rot[0]['x'][arbitraryStartInd], '.', label='start')
    plt.plot(arbitraryEndInd, rot[0]['x'][arbitraryEndInd], '.', label='end')
    plt.legend()
    plt.show()
    pass

if __name__=='__main__':
    # read hand rotation 
    handRotationFilePath = '../HandRotationOuputFromHomePC/leftFrontKickStream.json'
    handJointsRotations=None
    with open(handRotationFilePath, 'r') as fileOpen: 
        rotationJson=json.load(fileOpen)
        handJointsRotations = rotationJsonDataParser({'results': rotationJson}, jointCount=4)    # For python output
    main(handJointsRotations)