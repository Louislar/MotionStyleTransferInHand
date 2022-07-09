import numpy as np
import pandas as pd
import json
import enum
from sklearn.metrics.pairwise import euclidean_distances
from positionAnalysis import positionDataPreproc, positionJsonDataParser, positionDataToPandasDf, setHipAsOrigin, rollingWindowSegRetrieve

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
    return kSimilarL2Idx[:, :k]
    

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
    ## TODO: 每個joint都先找出最相似的k個motions，在考慮如何結合多個joints找到的相似motions

    # 與每個mapped feature vec前k個相似的DB feature vectors
    kSimilarDBIdx = findKSimilarFeatureVectors(DBPreproc[2].values, AfterMapPreproc[2].values, kSimilar)
    print(kSimilarDBIdx[-100:, :])
    print(kSimilarDBIdx.shape)
