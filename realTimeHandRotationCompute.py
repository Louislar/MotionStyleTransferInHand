'''
Goal: 從MediaPipe抓取的手land mark轉換成rotation資訊, 
總共有6個rotation資訊, 
MCP兩個旋轉軸, PIP一個旋轉軸, 
index finger以及middle finger各有一個MCP and PIP
總共只會用到7個點, 但是MediaPipe會給21個點
'''

import numpy as np
import json
import enum

class jointsNames(enum.IntEnum):
    wrist = 0
    thunmbCMC = 1
    thunmbMCP = 2
    thunmbIP = 3
    thunmbTIP = 4
    indexMCP = 5
    indexPIP = 6
    indexDIP = 7
    indexTIP = 8
    middleMCP = 9
    middlePIP = 10
    middleDIP = 11
    middleTIP = 12
    ringMCP = 13
    ringPIP = 14
    ringDIP = 15
    ringTIP = 16
    pinkyMCP = 17
    pinkyPIP = 18
    pinkyDIP = 19
    pinkyTIP = 20

usedJoints = [
    jointsNames.wrist, 
    jointsNames.indexMCP, jointsNames.indexPIP, jointsNames.indexTIP, 
    jointsNames.middleMCP, jointsNames.middlePIP, jointsNames.middleTIP
]

if __name__ == '__main__':
    # 1. Read hand landmark data(only keep joints in used)
    # 1.1 make it a streaming data 
    # 2. 計算向量
    # 3. 手掌法向量
    # 4. 計算角度

    # 1. 
    saveDirPath = 'complexModel/'
    handLMJson = None
    with open(saveDirPath+'leftSideKick.json', 'r') as fileOpen: 
        handLMJson=json.load(fileOpen)
    timeCount = len(handLMJson)
    for t in range(timeCount):
        # TODO: finish this. screen used joints' data
        pass