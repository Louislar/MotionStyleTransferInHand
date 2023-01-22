'''
分析計算耗時的統計資訊
1. mean
2. std
3. 10% low FPS
4. visualized
5. 輸出統計資料成csv檔案
'''

import pandas as pd 
import os 
from functools import reduce

def main():
    '''
    1. read in all the data 
    2. merge data into 4 groups that list in the thesis
    3. compute mean, std, 10% (10 percetile) high latency in each group across all actions
        同一步驟, 但是所有動作種類一起做統計
    4. compute statistics in each group of each action

    5. Store statistics in csv
    '''
    timeCostDataDirPath = 'timeConsume/'
    # 1. 
    actionDirList = [os.path.join(timeCostDataDirPath, i) for i in os.listdir(timeCostDataDirPath) if os.path.isdir(os.path.join(timeCostDataDirPath, i))]
    actionNmList = [os.path.basename(i) for i in actionDirList]
    timeCostNmList = []
    timeCostCSVFilePath = []
    actionTimeCost = {}
    for _actionNm, _actionDir in zip(actionNmList, actionDirList):
        _actionTimeCostFilePath = [os.path.join(_actionDir, i) for i in os.listdir(_actionDir)]
        timeCostNmList = [os.path.splitext(os.path.basename(i))[0] for i in _actionTimeCostFilePath]
        timeCostCSVFilePath.append(_actionTimeCostFilePath)
        _actionTimeCostFile = {_nm: pd.read_csv(i, index_col=None) for _nm, i in zip(timeCostNmList, _actionTimeCostFilePath)}
        actionTimeCost[_actionNm] = _actionTimeCostFile
        # print(_actionTimeCostFilePath)
        # print('=======')
    
    # Side kick的資料要特別處理, 因為之前有篩選部分片段 
    actionTimeCost['sideKick']['rotationCompute'] = actionTimeCost['sideKick']['rotationCompute'].iloc[2600:3500, :]
    actionTimeCost['sideKick']['rotationCompute'].reset_index(inplace=True, drop=True)

    # 2. 
    ## 合併變成5個主要步驟
    processTimeCost = {i: {} for i in actionNmList}
    ## Mediapipe, rotation compute and mapping, forward kinematic, searching for similar FV, pose blending
    for _actionNm, _actionData in actionTimeCost.items():
        ## merge rotation compute and rotation mapping
        rotationCompMap = pd.concat([_actionData['rotationCompute'], _actionData['rotationMapping']], axis=1)
        actionTimeCost[_actionNm]['rotationCompMap'] = rotationCompMap.reset_index(drop=True)
        # print(_actionNm)
        # print('rotationCompute before merge: ', _actionData['rotationCompute'].shape)
        # print('rotationMapping before merge: ', _actionData['rotationMapping'].shape)
        # print('after merge: ', rotationCompMap.shape)

        ## merge multiple columns and make Dataframe into series (single comlumn) 
        processTimeCost[_actionNm]['mediapipe'] = actionTimeCost[_actionNm]['mediapipe'].sum(axis=1)
        processTimeCost[_actionNm]['rotationCompMap'] = actionTimeCost[_actionNm]['rotationCompMap'].sum(axis=1)
        processTimeCost[_actionNm]['forwardKinematic'] = actionTimeCost[_actionNm]['forwardKinematic'].sum(axis=1)
        processTimeCost[_actionNm]['searchFV'] = actionTimeCost[_actionNm]['poseSynthesis'][['kSimilarSearch']].sum(axis=1)
        processTimeCost[_actionNm]['poseBlending'] = actionTimeCost[_actionNm]['poseSynthesis'][['poseBlending', 'EWMA']].sum(axis=1)

    # 3. 
    timeCostAcrossActions = {i: None for i in processTimeCost[actionNmList[0]].keys()}
    timeCostStatAcrossActions = {i: {} for i in processTimeCost[actionNmList[0]].keys()}
    for _processNm in timeCostAcrossActions.keys():
        # if _processNm == 'rotationCompMap':
        #     for v in processTimeCost.values():
        #         print(v[_processNm])
        _timeCostACrossActions = [v[_processNm] for v in processTimeCost.values()]
        timeCostAcrossActions[_processNm] = pd.concat(_timeCostACrossActions).reset_index(drop=True)

        # statistics
        timeCostStatAcrossActions[_processNm]['mean'] = timeCostAcrossActions[_processNm].mean()
        timeCostStatAcrossActions[_processNm]['std'] = timeCostAcrossActions[_processNm].std()
        timeCostStatAcrossActions[_processNm]['90Percentile'] = timeCostAcrossActions[_processNm].quantile(q=0.9)

        # print statistics 
        for _statNm in timeCostStatAcrossActions[_processNm].keys():
            print(f'{_processNm} {_statNm}: ', timeCostStatAcrossActions[_processNm][_statNm])
    ## To dataframe 
    timeCostStatAcrossActionsDf = pd.DataFrame(
        {k: v.values() for k, v in timeCostStatAcrossActions.items()}, 
        index=['mean', 'std', '90Percentile']
    )
    print(timeCostStatAcrossActionsDf.T)

    # 4. 各個action分開計算
    timeCostStatOfEachAction = {i: {j: {} for j in processTimeCost[actionNmList[0]].keys()} for i in processTimeCost.keys()}
    for _actionNm in timeCostStatOfEachAction.keys():
        for _processNm in timeCostStatOfEachAction[_actionNm].keys():
            timeCostStatOfEachAction[_actionNm][_processNm]['mean'] = processTimeCost[_actionNm][_processNm].mean()
            timeCostStatOfEachAction[_actionNm][_processNm]['std'] = processTimeCost[_actionNm][_processNm].std()
            timeCostStatOfEachAction[_actionNm][_processNm]['90Percentile'] = processTimeCost[_actionNm][_processNm].quantile(q=0.9)
    ## To DataFrame
    timeCostStatOfEachActionDfs = {
        _actionNm: pd.DataFrame(
            {k: v.values() for k, v in timeCostStatOfEachAction[_actionNm].items()}, 
            index=['mean', 'std', '90Percentile']
        ) for _actionNm in timeCostStatOfEachAction.keys()
    }
    print(timeCostStatOfEachActionDfs['runSprint'].T)
    
    
    # 5. 
    timeCostStatAcrossActionsDf.to_csv(
        os.path.join(timeCostDataDirPath, 'timeCostAcrossActions.csv')
    )
    for _actionNm in timeCostStatOfEachActionDfs.keys():
        timeCostStatOfEachActionDfs[_actionNm].to_csv(
            os.path.join(timeCostDataDirPath, f'timeCost_{_actionNm}.csv')
        )

if __name__=='__main__':
    main()
    pass