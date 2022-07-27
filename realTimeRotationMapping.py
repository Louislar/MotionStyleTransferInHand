'''
Goal: 
1. Rotation mapping需要的bBSpline資訊儲存成檔案備用, 
    儲存的檔案會是sample很多點的結果
    這部分實作在原本的rotationAnalysis.py當中
2. 將stream rotation data做mapping的function
'''

import numpy as np 
import json
import pickle


if __name__=='__main01__':
    # TODO: Dynamiclly decide the streaming data 
    #       is in increasing segment or decreasing segment
    pass

if __name__=='__main__':
    # 1. Read in the pre compute BSpline parameter (from rotationAnalysis.py)
    saveDirPath = 'preprocBSpline/leftFrontKick/'
    usedJointIdx = [['x','z'], ['x'], ['x','z'], ['x']]
    BSplineParam = [
        [{aAxis: None for aAxis in usedJointIdx[aJoint]} for aJoint in range(len(usedJointIdx))], 
        [{aAxis: None for aAxis in usedJointIdx[aJoint]} for aJoint in range(len(usedJointIdx))]
    ]
    for aJoint in range(len(usedJointIdx)):
        for aAxis in usedJointIdx[aJoint]:
            for i in range(2):
                with open(saveDirPath+'{0}.pickle'.format(str(i)+'_'+aAxis+'_'+str(aJoint)), 'rb') as inPickle:
                    BSplineParam[i][aJoint][aAxis] = kdtree = pickle.load(inPickle)
    print(BSplineParam[0][0]['x'])
    # 2. Sample points from the pre compute BSpline parameter 
    # 3. Save the sample points