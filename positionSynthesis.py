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

def kSimilarFeatureVectorsBlending(mainJointFeatVecs, kSimilarFeatVecsIdx, kSimilarFeatVecsDists):
    '''
    將多個feature vectors做blending
    Input: 

    TODO: 輸入的Feature vector不需要augment過(也不需要velocity, acceralation), 
        因為synthesis只需要目標位置的3D position(X, Y, Z)
    :mainJointFeatVecs: 目標要synthesis的joint, 在所有time point的feature vectors
    :kSimilarFeatVecsIdx: (某個reference joint的)所有時間點, 每一個時間點有k個相似的feature vectors的time point/index
    :kSimilarFeatVecsDists: (某個reference joint的)與k個相似的feature vectors的距離(作為weight使用)
    每個row都是一個time point, 內含前k個相似的DB poses
    '''
    print(mainJointFeatVecs.shape)
    print(kSimilarFeatVecsIdx.shape)
    print(kSimilarFeatVecsDists.shape)
    for t in range(kSimilarFeatVecsIdx.shape[0]):
        # 某個time point下, 最相似的k個3D positions
        kSimilarPositions = mainJointFeatVecs[kSimilarFeatVecsIdx[t, :], :] 
        print(kSimilarPositions)
        break
    
def augFeatVecToPos(aJointAugFeatVec):
    '''
    將augment後的某個joint的feature vector轉換為只有3D position的DataFrame

    Input: 
    :aJointAugFeatVec: 某個joint augment後的Feature vectors
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

    # TODO: Feature Vector轉換為原始的joint point(X, Y, Z)
    # 接下來在做motion synthesis時, 不會使用到速度與其他的augmentation
    # 轉換的方式為選取最後一個X, Y, Z數值，作為synthesis使用的數值

    print('after preproc: ', DBPreproc[1].shape)# (595, 81), 595=119*5
    print('before preproc: ', posDBDf.shape)#(601, 21)

    # 需要決定要參考哪一些joints的motions，以及跨joint之間的blending weight該如何決定
    # TODO: 前k個相似的瞬時motion做blending
    # TODO: 這邊需要區分哪一個joint，因為不同joint會使用不同的blending策略
    # e.g. 左腳: 只使用左腳的前k個相似poses, 右膝: 只使用右腳的前k個相似poses, 左手: 使用所有joint得到的相似poses做blending
    jointsBlendingRef = {
        jointsNames.LeftFoot: {jointsNames.LeftFoot: 1.0}, 
        jointsNames.LeftLowerLeg: {jointsNames.LeftFoot: 0.9, jointsNames.LeftLowerLeg: 0.1},    # TODO: finish this dict(need weights of reference joints)
        jointsNames.RightFoot: {jointsNames.RightFoot: 1.0}, 
        jointsNames.RightLowerLeg: {jointsNames.RightFoot: 0.9, jointsNames.RightLowerLeg: 0.1}
    }   # 第一層的key是main joint, 第二層的key是reference joints, 第二層value是reference joints之間的weight
    for aBlendingRef in jointsBlendingRef:
        mainJoint = aBlendingRef    # 要synthesis的joint
        refJoints = jointsBlendingRef[mainJoint].keys() # 做為參考找出k個similar vectors的joint(等同找到time point)
    kSimilarFeatureVectorsBlending(DBPreproc[2].values, kSimilarDBIdx, kSimilarDBDist)

    # TODO: 輸出blending完之後的整段motions
