'''
專為testingStage設計的config class
'''
import json

class TestStageConfig():
    def __initFrontKick__(self) -> None:
        self.handLandMarkFilePath = 'complexModel/frontKick.json'

        self.usedJointIdx = [['x','z'], ['x'], ['x','z'], ['x']]
        self.usedJointIdx1 = [(i,j) for i in range(len(self.usedJointIdx)) for j in self.usedJointIdx[i]]
        self.mappingCategory = 4 # 0: euler linear, 1: euler B-Spline, 2: quat linear, 3: quat B-Spline, 4: quat Direct mapping 
        self.isLinearMapping = False
        self.quatIndex = [['x','y','z','w'], ['x','y','z','w'], ['x','y','z','w'], ['x','y','z','w']]
        self.mappingStrategy = [['x'], ['x'], ['z'], ['x']]
        self.negMappingStrategy = [['z'], ['x'], [], []]
        self.unusedJointAxis = [['y', 'z'], ['y', 'z'], ['y'], ['y', 'z']]    # 與mappingStrategy是互補的關係
        self.handPerfAxisPair = {0: 'x', 1: 'x', 2: 'x', 3: 'x'}    # quat direct mapping會使用到 
        self.BSplineHandSPFilePath='rotationMappingQuaternionData/leftFrontKickBSpline/handNormMapSamplePts.pickle'
        self.BSplineBodySPFilePath='rotationMappingQuaternionData/leftFrontKickBSpline/bodyNormMapSamplePts.pickle'
        self.DBRefSeqFilePath = 'rotationMappingQuatDirectMappingData/leftFrontKick_body_ref.npy'    # quat direct mapping會使用到 
        self.handPerfRefSeqFilePath = 'rotationMappingQuatDirectMappingData/leftFrontKick_hand_ref.npy'    # quat direct mapping會使用到 

        self.ifUseVelAcc = False
        self.TPosePosDataFilePath = 'TPoseInfo/genericAvatar/' # From realTimeRotToAvatarPos.py
        self.DBMotionKDTreeFilePath = 'DBPreprocFeatVec/NoVelAccOverlap/leftFrontKickPositionFullJointsWithHead_withoutHip_075_quat_direct_normalized/'  # From realTimePositionSynthesis.py
        self.DBMotion3DPosFilePath = 'DBPreprocFeatVec/NoVelAccOverlap/leftFrontKick_withHip_075/3DPos/' # From realTimePositionSynthesis.py
        self.ksimilar = 5
        self.EWMAWeight = 0.7
        self.upperLegXAxisRotAdj = -30
        self.leftUpperLegZAxisRotAdj = -20

    def __init__(self) -> None:
        self.handLandMarkFilePath = 'complexModel/leftSideKick.json'

        self.usedJointIdx = [['x','z'], ['x'], ['x','z'], ['x']]
        self.usedJointIdx1 = [(i,j) for i in range(len(self.usedJointIdx)) for j in self.usedJointIdx[i]]
        self.mappingCategory = 4 # 0: euler linear, 1: euler B-Spline, 2: quat linear, 3: quat B-Spline, 4: quat Direct mapping
        self.quatIndex = [['x','y','z','w'], ['x','y','z','w'], ['x','y','z','w'], ['x','y','z','w']]
        self.mappingStrategy = [['z'], ['x'], ['z'], ['x']]
        self.negMappingStrategy = [['z'], ['x'], [], []]    # Only used in euler mapping functions 
        self.unusedJointAxis = [['x', 'y'], ['y', 'z'], ['x', 'y'], ['y', 'z']]    # 與mappingStrategy是互補的關係
        self.handPerfAxisPair = {0: 'z', 1: 'x', 2: 'z', 3: 'x'}    # quat direct mapping會使用到
        self.BSplineHandSPFilePath='rotationMappingQuaternionData/leftSideKickBSpline/handNormMapSamplePts.pickle'
        self.BSplineBodySPFilePath='rotationMappingQuaternionData/leftSideKickBSpline/bodyNormMapSamplePts.pickle'
        self.DBRefSeqFilePath = 'rotationMappingQuatDirectMappingData/leftSideKick_body_ref.npy'    # quat direct mapping會使用到 
        self.handPerfRefSeqFilePath = 'rotationMappingQuatDirectMappingData/leftSideKick_hand_ref.npy'    # quat direct mapping會使用到 

        self.ifUseVelAcc = False
        self.TPosePosDataFilePath = 'TPoseInfo/genericAvatar/' # From realTimeRotToAvatarPos.py
        self.DBMotionKDTreeFilePath = 'DBPreprocFeatVec/NoVelAccOverlap/leftSideKick_withoutHip_075/'  # From realTimePositionSynthesis.py
        self.DBMotion3DPosFilePath = 'DBPreprocFeatVec/NoVelAccOverlap/leftSideKick_withHip_075/3DPos/' # From realTimePositionSynthesis.py
        self.ksimilar = 5
        self.EWMAWeight = 0.7
        self.upperLegXAxisRotAdj = -30
        self.leftUpperLegZAxisRotAdj = -20

    def __initRunSprint__(self) -> None:
        self.handLandMarkFilePath = 'complexModel/runSprint.json'

        self.usedJointIdx = [['x','z'], ['x'], ['x','z'], ['x']]
        self.usedJointIdx1 = [(i,j) for i in range(len(self.usedJointIdx)) for j in self.usedJointIdx[i]]
        self.mappingCategory = 3 # 0: euler linear, 1: euler B-Spline, 2: quat linear, 3: quat B-Spline
        self.quatIndex = [['x','y','z','w'], ['x','y','z','w'], ['x','y','z','w'], ['x','y','z','w']]
        self.mappingStrategy = [['x'], ['x'], ['x'], ['x']]
        self.negMappingStrategy = [['z'], ['x'], [], []]    # Only used in euler mapping functions for rotation adjustment  
        self.unusedJointAxis = [['y', 'z'], ['y', 'z'], ['y', 'z'], ['y', 'z']]    # 與mappingStrategy是互補的關係
        self.BSplineHandSPFilePath='rotationMappingQuaternionData/runSprintBSpline/handNormMapSamplePts.pickle'
        self.BSplineBodySPFilePath='rotationMappingQuaternionData/runSprintBSpline/bodyNormMapSamplePts.pickle'

        self.TPosePosDataFilePath = 'TPoseInfo/genericAvatar/' # From realTimeRotToAvatarPos.py
        self.DBMotionKDTreeFilePath = 'DBPreprocFeatVec/runSprint_withHip_05_quat_BSpline_normalized/'  # From realTimePositionSynthesis.py
        self.DBMotion3DPosFilePath = 'DBPreprocFeatVec/runSprint_withHip_05/3DPos/' # From realTimePositionSynthesis.py
        self.ksimilar = 5
        self.EWMAWeight = 0.7
        self.upperLegXAxisRotAdj = -30
        self.leftUpperLegZAxisRotAdj = -20

    def __initHurdleJump__(self) -> None:
        self.handLandMarkFilePath = 'complexModel/newRecord/jumpHurdle_rgb.json'

        self.usedJointIdx = [['x','z'], ['x'], ['x','z'], ['x']]
        self.usedJointIdx1 = [(i,j) for i in range(len(self.usedJointIdx)) for j in self.usedJointIdx[i]]
        self.mappingCategory = 3 # 0: euler linear, 1: euler B-Spline, 2: quat linear, 3: quat B-Spline
        self.quatIndex = [['x','y','z','w'], ['x','y','z','w'], ['x','y','z','w'], ['x','y','z','w']]
        self.mappingStrategy = [['x'], ['x'], ['x'], ['x']]
        self.negMappingStrategy = [['z'], ['x'], [], []]    # Only used in euler mapping functions for rotation adjustment  
        self.unusedJointAxis = [['y', 'z'], ['y', 'z'], ['y', 'z'], ['y', 'z']]    # 與mappingStrategy是互補的關係
        self.BSplineHandSPFilePath='rotationMappingQuaternionData/hurdleJumpBSpline/handNormMapSamplePts.pickle'
        self.BSplineBodySPFilePath='rotationMappingQuaternionData/hurdleJumpBSpline/bodyNormMapSamplePts.pickle'

        self.TPosePosDataFilePath = 'TPoseInfo/genericAvatar/' # From realTimeRotToAvatarPos.py
        self.DBMotionKDTreeFilePath = 'DBPreprocFeatVec/hurdleJump_withoutHip_075_quat_BSpline_normalized/'  # From realTimePositionSynthesis.py
        self.DBMotion3DPosFilePath = 'DBPreprocFeatVec/hurdleJump_075/3DPos/' # From realTimePositionSynthesis.py
        self.ksimilar = 5
        self.EWMAWeight = 0.7
        self.upperLegXAxisRotAdj = -30
        self.leftUpperLegZAxisRotAdj = -20

    def __initWalkInjured__(self) -> None:
        self.handLandMarkFilePath = 'complexModel/newRecord/walkInjured_rgb_2022_9_12.json'

        self.usedJointIdx = [['x','z'], ['x'], ['x','z'], ['x']]
        self.usedJointIdx1 = [(i,j) for i in range(len(self.usedJointIdx)) for j in self.usedJointIdx[i]]
        self.mappingCategory = 3 # 0: euler linear, 1: euler B-Spline, 2: quat linear, 3: quat B-Spline
        self.quatIndex = [['x','y','z','w'], ['x','y','z','w'], ['x','y','z','w'], ['x','y','z','w']]
        self.mappingStrategy = [['x'], ['x'], ['x'], ['x']]
        self.negMappingStrategy = [['z'], ['x'], [], []]    # Only used in euler mapping functions for rotation adjustment  
        self.unusedJointAxis = [['y', 'z'], ['y', 'z'], ['y', 'z'], ['y', 'z']]    # 與mappingStrategy是互補的關係
        self.BSplineHandSPFilePath='rotationMappingQuaternionData/walkInjuredBSpline/handNormMapSamplePts.pickle'
        self.BSplineBodySPFilePath='rotationMappingQuaternionData/walkInjuredBSpline/bodyNormMapSamplePts.pickle'

        self.TPosePosDataFilePath = 'TPoseInfo/genericAvatar/' # From realTimeRotToAvatarPos.py
        self.DBMotionKDTreeFilePath = 'DBPreprocFeatVec/walkInjured_withHip_075_quat_BSpline_normalized/'  # From realTimePositionSynthesis.py
        self.DBMotion3DPosFilePath = 'DBPreprocFeatVec/walkInjured_withHip_075/3DPos/' # From realTimePositionSynthesis.py
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
    # config.toJson('testStageConfig/walkInjuredQuatBSplineConfig.json')
    # config.toJson('testStageConfig/hurdleJumpQuatBSplineConfig.json')
    # config.toJson('testStageConfig/runSprintQuatBSplineConfig.json')
    # config.toJson('testStageConfig/sideKickQuatBSplineConfig.json')
    # config.toJson('testStageConfig/frontKickQuatBSplineConfig.json')

    # config.toJson('testStageConfig/frontKickQuatDirectConfig.json')
    config.toJson('testStageConfig/sideKickQuatDirectConfig.json')
    # print(config.__dict__)
    # print(TestStageConfig.__dict__)
    pass