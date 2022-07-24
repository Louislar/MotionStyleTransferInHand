function [outputArg2] = readPosMatrix(inputArg1)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%   outputArg2 = inputArg2;
outputArg2 = readtable(inputArg1);
outputArg2 = outputArg2.Variables.';
end

