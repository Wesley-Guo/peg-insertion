clear all; clc
format long;
data = csvread('test_data_full.csv',1,1);

numStates = length(data(:, 1));
numStateComponents = length(data(1, :));
% add 5 intermediary steps, since we are only changing xyz and rx,ry,rz
% incrementally
numWithIntermediaryStates = (numStates - 1) * 5 + numStates; 
newStates = zeros(numWithIntermediaryStates, numStateComponents);
actionsTable = zeros(numWithIntermediaryStates-1, 1);

% initialize actionsTable w/ first row of data
idxNewStates = 1;
newStates(idxNewStates, :) = data(1, :);

for i = 1:numStates-1
    currState = data(i, :);
    nextState = data(i+1, :);
    intermediaryState = currState;
    for j = 1:numStateComponents
        delta = nextState(j) - currState(j);
        intermediaryState(j) = nextState(j);
        idxNewStates = idxNewStates + 1;
        newStates(idxNewStates, :) = intermediaryState;
        actionsTable(idxNewStates-1) = findActionOld(j, delta);
    end
end

