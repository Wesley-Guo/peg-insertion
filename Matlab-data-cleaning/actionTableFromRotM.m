clear all; clc
format long;

numOfTraj = 23;

for i = 1:numOfTraj
    filename = strcat(int2str(i), '.csv');
    raw_data = csvread(filename, 1, 0);
    numStates = length(raw_data(:, 1));
    rotTable = raw_data(:, 18:26);
    eulerTable = zeros(numStates, 3);
    
    for m = 1:numStates
        rotV1 = rotTable(m, 1:3);
        rotV2 = rotTable(m, 4:6);
        rotV3 = rotTable(m, 7:9);
        rotM = [rotV1; rotV2; rotV3];
        eulerZYX = rotm2eul(rotM);
        eulerTable(m, :) = eulerZYX;
    end
    
    sineTable = sin(eulerTable);
    
    % extract only the pose data from the raw file
    % 3 for position and 3 for euler angles
    numPose = 6;
    data = zeros(numStates, numPose);
    data(:, 1:3) = raw_data(:, 3:5);
    data(:, 4:6) = sineTable;

    actionsTable = zeros(numStates - 1, numPose);

    for l = 1:numStates - 1
        difference = data(l+1, :) - data(l, :);
        actionsTable(l, :) = findActionFromRotM(difference);
    end
    
    %%%%% create csv for gradient Q-learning

    numActionComponents = 6;
    numFullStateComponents = 9;
    numCol = (2 * numFullStateComponents) + numActionComponents + 1;
    finalCSV = zeros(numStates-1, numCol);

    clipping_idx = numStates-1;
    first = 1;
    for m = 1:numStates - 1
        finalCSV(m, 1:3) = raw_data(m, 3:5); % position
        if (first == 1) && (raw_data(m, 5) < 0)
            first = 0;
            clipping_idx = m;
        end
        finalCSV(m, 4:6) = sineTable(m, :); % sin of euler
        finalCSV(m, 7:9) = raw_data(m, 9:11); % force data
        finalCSV(m, 10:15) = actionsTable(m, :);
        reward = sum(abs(raw_data(m, 9:11)));
        reward = -.005*(reward)^3;
        finalCSV(m, 16) = reward;
        finalCSV(m, 17:19) = raw_data(m+1, 3:5);
        finalCSV(m, 20:22) = sineTable(m+1, :);
        finalCSV(m, 23:25) = raw_data(m+1, 9:11);
    end
    
    finalCSV(clipping_idx, 16) = 10000;
    outputCSV = zeros(clipping_idx, numCol);
    outputCSV = finalCSV(1:clipping_idx, 1:numCol);
    
    outputFilename = strcat('output', int2str(i), '.csv');
    
    writematrix(outputCSV, outputFilename);
    clear raw_data data actionsTable sineTable;

end



