clear all; clc
format long;

%raw_data = csvread('data_2023-12-02.16.48.48.csv',1,2);

numOfTraj = 1;

for i = 1:numOfTraj
    filename = strcat(int2str(i), '.csv');
    raw_data = csvread(filename, 1, 0);
    numStates = length(raw_data(:, 1));
    rotTable = raw_data(:, 6:14);
    eulerTable = zeros(numStates, 3);
    
    for m = 1:numStates
        rotV1 = rotTable(m, 1:3);
        rotV2 = rotTable(m, 4:6);
        rotV3 = rotTable(m, 7:9);
        rotM = [rotV1; rotV2; rotV3];
        eulerZYX = rotm2eul(rotM);
        eulerZYX = eulerZYX + pi;
        eulerTable(m, :) = eulerZYX;
    end
    
    sineTable = zeros(numStates, 3);
    
    for k = 1:numStates
        sineTable(k, :) = sin(eulerTable(k, :));
    end
    
    % extract only the pose data from the raw file
    numPose = 6;
    data = zeros(numStates, numPose);
    data(:, 1:3) = raw_data(:, 3:5);
    data(:, 4:6) = sineTable;

    actionsTable = zeros(numStates - 1, numPose);

    for l = 1:numStates - 1
        difference = data(l+1, :) - data(l, :);
        actionsTable(l, :) = findAction(difference);
    end
    
    %%%%% create csv for gradient Q-learning

    numActionComponents = 6;
    numFullStateComponents = 9;
    numCol = (2 * numFullStateComponents) + numActionComponents + 1;
    finalCSV = zeros(numStates-1, numCol);

    for m = 1:numStates - 1
        finalCSV(m, 1:3) = raw_data(m, 3:5); % position
        finalCSV(m, 4:6) = sineTable(m, :); % sin of euler
        finalCSV(m, 7:9) = raw_data(m, 18:20); % force data
        finalCSV(m, 10:15) = actionsTable(m, :);
        finalCSV(m, 16) = 0; % manually look through data to add reward values
        finalCSV(m, 1:3) = raw_data(m+1, 3:5);
        finalCSV(m, 4:6) = sineTable(m+1, :);
        finalCSV(m, 7:9) = raw_data(m+1, 18:20);
    end
    
    outputFilename = strcat('output', int2str(i), '.csv');
    
    writematrix(finalCSV, outputFilename);
    clear raw_data data actionsTable;

end



