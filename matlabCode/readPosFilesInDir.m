function [outputArg1, sortedVal, sortedIdx, fileArr] = readPosFilesInDir(dirIn)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
%   input arg dirIn can be "allDisplacement/leftFrontKick/displacementleftFrontKick*.csv"
% outputArg1 = inputArg1;
% outputArg2 = inputArg2;
% 
% Output: 
% :outputArg1: 所有檔案與DBMatrix的l2距離
% :sortedVal: outputArg1由小排到大的結果
% :sortedIdx: 依照檔案名稱作為index, 依照outputArg1數值由小排到大的結果
% :fileArr: 列出讀取的檔案順序
[filepath,name,ext] = fileparts(dirIn);
outputArg1 = dir(dirIn);
outputArg1 = struct2cell(outputArg1);
outputArg1 = string(outputArg1(1, :));
fileArr = outputArg1;

% Set kernel covariance parameter 
kerspec.sigma = 1;
kerspec.type = 'exp';

% Read DB file and compute feature of it
DBFile = readPosMatrix(fullfile(filepath, "displacementDBMatrix.csv"));
DBFeat = kercov(DBFile, kerspec);

% TODO: read in all the files
distanceRes = zeros(length(outputArg1), 1);
for i=1:length(outputArg1)
    readInFile = readPosMatrix(fullfile(filepath, outputArg1(i)));
    readInFeat = kercov(readInFile, kerspec);
    distanceRes(i) = motionDistance(readInFeat, DBFeat);
end
outputArg1 = distanceRes;
[sortedVal,sortedIdx] = sort(distanceRes);
end