'''
專為testingStage設計的config class
'''
import json

class TestStageConfig():
    def __init__(self) -> None:
        self.handLandMarkFilePath = 'complexModel/frontKick.json'

        self.usedJointIdx = [['x','z'], ['x'], ['x','z'], ['x']]
        self.usedJointIdx1 = [(i,j) for i in range(len(self.usedJointIdx)) for j in self.usedJointIdx[i]]
        self.mappingCategory = 3 # 0: euler linear, 1: euler B-Spline, 2: quat linear, 3: quat B-Spline
        self.isLinearMapping = False
        self.quatIndex = [['x','y','z','w'], ['x','y','z','w'], ['x','y','z','w'], ['x','y','z','w']]
        self.mappingStrategy = [['x'], ['x'], ['z'], ['x']]
        self.negMappingStrategy = [['z'], ['x'], [], []]
        self.unusedJointAxis = [['y', 'z'], ['y', 'z'], ['y'], ['y', 'z']]    # 與mappingStrategy是互補的關係
        self.BSplineHandSPFilePath='rotationMappingQuaternionData/leftFrontKickBSpline/handNormMapSamplePts.pickle'
        self.BSplineBodySPFilePath='rotationMappingQuaternionData/leftFrontKickBSpline/bodyNormMapSamplePts.pickle'

        self.TPosePosDataFilePath = 'TPoseInfo/genericAvatar/' # From realTimeRotToAvatarPos.py
        self.DBMotionKDTreeFilePath = 'DBPreprocFeatVec/leftFrontKick_withHip_075_quat_BSpline_normalized/'  # From realTimePositionSynthesis.py
        self.DBMotion3DPosFilePath = 'DBPreprocFeatVec/leftFrontKick_075/3DPos/' # From realTimePositionSynthesis.py
        self.ksimilar = 5
        self.EWMAWeight = 0.7
        self.upperLegXAxisRotAdj = -30
        self.leftUpperLegZAxisRotAdj = -20

    def toJson(self, filePath):
        with open(filePath, 'w') as wFile:
            json.dump(self.__dict__, wFile)

    def fromJson(self, jsonFile): 
        for k in jsonFile:
            self.__dict__[k] = jsonFile[k]
if __name__=='__main__':
    config = TestStageConfig()
    config.toJson('testStageConfig/quatBSplineConfig.json')
    # print(config.__dict__)
    # print(TestStageConfig.__dict__)
    pass