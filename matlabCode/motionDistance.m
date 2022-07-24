function [outputArg1] = motionDistance(inputArg1,inputArg2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%   inputArg1 is the first feature matrix, 
%   inputArg2 is the second feature matrix
outputArg1 = norm(inputArg1-inputArg2, 1); % l1 norm/distance
outputArg1 = norm(inputArg1-inputArg2); % l2 norm/distance

end

