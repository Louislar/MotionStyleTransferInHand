'''
嘗試看看auto correlation以及cross correlation的效果
'''

import numpy as np 
import matplotlib.pyplot as plt 
from util import readHandPerformance, cropHandPerformance, handPerformanceToMatrix

cropInterval = [1430, 1530]
usedJointAxis = [[0, 'x'], [1, 'x']]

def main():
    # Read hand performance into a matrix and crop desire interval 
    data = readHandPerformance()
    cropData = cropHandPerformance(data, cropInterval[0], cropInterval[1])
    dataMat = handPerformanceToMatrix(cropData, usedJointAxis)    

    # Perform cross correlation by reference control motion 
    fullDataMat = handPerformanceToMatrix(data, usedJointAxis)
    crossCorr = np.correlate(fullDataMat.T[:, 0], dataMat.T[:, 0], mode='full') 
    crossCorr2 = np.correlate(fullDataMat.T[:, 1], dataMat.T[:, 1], mode='full') 
    # print(crossCorr)

    # plot cross correlation 
    plt.figure()
    plt.plot(crossCorr)
    plt.plot(crossCorr2)
    plt.show()

    pass

if __name__=='__main__':
    main()