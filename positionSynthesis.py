import numpy as np
import pandas as pd
import json
import enum
from sklearn.metrics.pairwise import euclidean_distances
from positionAnalysis import positionDataPreproc, positionJsonDataParser, positionDataToPandasDf, setHipAsOrigin, rollingWindowSegRetrieve, jointsNames

'''
目標: 從輸入的DB motion當中找到與輸入motion最相似的motion, 
使用KNN尋找k個最相似的DB motions, 
最後需要將DB motion preprocessing後的結果儲存, 減少計算所需時間
'''

positionsJointCount = 7
rollingWinSize = 10
kSimilar = 5
augmentationRatio = [0.5, 0.7, 1, 1.3, 1.5]

def findKSimilarFeatureVectors(aJointDBFeatVecs, aJointMappedFeatVecs, k):
    '''
    計算與mapped feature vectors最相似的k個DB feature vectors, 這些feature vectors都代表相同joint卻又不同的window
    理論上DB feature數量是mapped feature vector的n倍, n是augment ratio的種類數量
    '''
    print(aJointMappedFeatVecs.shape)
    print(aJointDBFeatVecs.shape)
    l2BtwDBAndMapped = euclidean_distances(aJointMappedFeatVecs, aJointDBFeatVecs)  # 每個row都是與某個mapped feature vector與所有DB feature vectors的l2距離
    print(l2BtwDBAndMapped.shape)
    kSimilarL2Idx = np.argsort(l2BtwDBAndMapped, axis=1)
    kSimilarL2Dist = np.sort(l2BtwDBAndMapped, axis=1)
    return kSimilarL2Idx[:, :k], kSimilarL2Dist[:, :k]

def kSimilarFeatureVectorsBlending(allJointsFeatVecs, kSimilarFeatVecsIdx, kSimilarFeatVecsDists):
    '''
    將多個feature vectors做blending
    Input: 

    :allJointsFeatVecs: 所有joint所有time point的feature vectors
    :kSimilarFeatVecsIdx: k個相似的feature vectors的time point/index
    :kSimilarFeatVecsDists: 與k個相似的feature vectors的距離(作為weight使用)
    '''

    pass
    

if __name__=='__main__':
    # Read position data
    DBFileName = './positionData/fromDB/leftFrontKickPosition.json'
    AfterMappingFileName = \
        './positionData/fromAfterMappingHand/leftFrontKickCombinations/leftFrontKick(True, False, False, False, True, True).json'

    ## Read Position data in DB
    posDBDf = None
    AfterMapDf = None
    with open(DBFileName, 'r') as fileIn:
        jsonStr=json.load(fileIn)
        positionsDB = positionJsonDataParser(jsonStr, positionsJointCount)
        posDBDf = positionDataToPandasDf(positionsDB, positionsJointCount)
    with open(AfterMappingFileName, 'r') as fileIn:
        jsonStr=json.load(fileIn)
        positionsDB = positionJsonDataParser(jsonStr, positionsJointCount)
        AfterMapDf = positionDataToPandasDf(positionsDB, positionsJointCount)

    ## Preprocessing
    ## 現在每個joint都有自己的windows/segments組成的DataFrame，
    ## 維度為(windows數量, (XX...| YY...| ZZ...| speed_x...| speed_y...| speed_z...| acc_x...| acc_y...| acc_z...))
    ## windows數量為原始windows數量*(變化速度數量)
    DBPreproc = positionDataPreproc(posDBDf, positionsJointCount, rollingWinSize, True, augmentationRatio)
    AfterMapPreproc = positionDataPreproc(AfterMapDf, positionsJointCount, rollingWinSize, False, augmentationRatio, False)

    ## Find k similar motions depends on different leg's motions(left leg and right leg)
    ## 左右腿的比較可以結合knee, foot, displacement between upper leg and foot
    ## 可以再加上左右腿綜合的joint pairs displacement比較(optional)
    ## Thus, k_left and k_right motions is found in each time point
    ## upper leg 以及 hip的比對就不需要了，因為這兩種joint幾乎不會有位移(位移都是noise)
    ## 每個joint都先找出最相似的k個motions，在考慮如何結合多個joints找到的相似motions

    # 與每個mapped feature vec前k個相似的DB feature vectors
    # TODO: 對指定的多個joints尋找前k個相似的DB feature vectors
    jointsInUsedToSyhthesis = [
        jointsNames.LeftLowerLeg, jointsNames.LeftFoot, jointsNames.RightLowerLeg, jointsNames.RightFoot
    ]
    # multiJointsKSimilarDBIdx = [None for i in jointsInUsedToSyhthesis]
    # multiJointskSimilarDBDist = [None for i in jointsInUsedToSyhthesis]
    # for i in jointsInUsedToSyhthesis:
    #     kSimilarDBIdx, kSimilarDBDist = findKSimilarFeatureVectors(DBPreproc[i].values, AfterMapPreproc[i].values, kSimilar)
    #     multiJointsKSimilarDBIdx.append(kSimilarDBIdx)
    #     multiJointskSimilarDBDist.append(kSimilarDBDist)

    kSimilarDBIdx, kSimilarDBDist = findKSimilarFeatureVectors(DBPreproc[1].values, AfterMapPreproc[1].values, kSimilar)
    print(kSimilarDBIdx[-100:, :])
    print(kSimilarDBDist[-30:, :])
    print(kSimilarDBIdx.shape)

    # 需要決定要參考哪一些joints的motions，以及跨joint之間的blending weight該如何決定
    # TODO: 前k個相似的瞬時motion做blending
    # TODO: 這邊需要區分哪一個joint，因為不同joint會使用不同的blending策略
    # e.g. 左腳: 只使用左腳的前k個相似poses, 右膝: 只使用右腳的前k個相似poses, 左手: 使用所有joint得到的相似poses做blending
    jointsBlendingRef = {
        jointsNames.LeftFoot: [jointsNames.LeftFoot], 
        jointsNames.LeftLowerLeg: [jointsNames.LeftFoot]    # TODO: finish this dict(need weights of reference joints)
    }
    for aBlendingRef in jointsBlendingRef:
        mainJoint = aBlendingRef
        refJoints = jointsBlendingRef[mainJoint]
    kSimilarFeatureVectorsBlending(DBPreproc, kSimilarDBIdx, kSimilarDBDist)

    # TODO: 輸出blending完之後的整段motions
