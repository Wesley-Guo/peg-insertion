clear all; clc
format long;

%raw_data = csvread('data_2023-12-02.16.48.48.csv',1,2);

numOfTraj = 11;

for k = 1:numOfTraj
    filename = strcat(int2str(k), '.csv');
    raw_data = csvread(filename, 1, 0);
    % extract only the pose data from the raw file
    data = raw_data(:, 3:8); 
    
    numStates = length(data(:, 1));
    numStateComponents = length(data(1, :));

    actionsTable = zeros(numStates - 1, numStateComponents);

    for i = 1:numStates - 1
        difference = data(i+1, :) - data(i, :);
        actionsTable(i, :) = findActionDeltaPhi(difference);
    end
    
    %%%%% create csv for gradient Q-learning

    numActionComponents = 6;
    numFullStateComponents = 9;
    numCol = (2 * numFullStateComponents) + numActionComponents + 1;
    finalCSV = zeros(numStates-1, numCol);

    clipping_idx = numStates-1;
    first = 1;
    for j = 1:numStates - 1
        finalCSV(j, 1:9) = raw_data(j, 3:11);
        if (first == 1) && (raw_data(j, 5) < 0)
            first = 0;
            clipping_idx = j;
        end
        finalCSV(j, 10:15) = actionsTable(j, :);
        reward = sum(abs(raw_data(j, 9:11)));
        reward = -.005*(reward)^3;
        finalCSV(j, 16) = reward; 
        finalCSV(j, 17:numCol) = raw_data(j+1, 3:11);
    end
    
    % add high reward for z-depth and clip all remaining states
    finalCSV(clipping_idx, 16) = 10000;
    outputCSV = zeros(clipping_idx, numCol);
    outputCSV = finalCSV(1:clipping_idx, 1:numCol);
    
    outputFilename = strcat('output', int2str(k), '.csv');
    
    writematrix(outputCSV, outputFilename);
    clear raw_data data actionsTable;

end



