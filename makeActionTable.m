clear all; clc
format long;

%raw_data = csvread('data_2023-12-02.16.48.48.csv',1,2);

numOfTraj = 1;

for k = 1:numOfTraj
    filename = strcat(int2str(k), '.csv');
    raw_data = csvread(filename, 1, 0);
    % extract only the pose data from the raw file
    data = raw_data(:, [3:5 15:17]); 
    
    numStates = length(data(:, 1));
    numStateComponents = length(data(1, :));

    actionsTable = zeros(numStates - 1, numStateComponents);

    for i = 1:numStates - 1
        difference = data(i+1, :) - data(i, :);
        actionsTable(i, :) = findAction(difference);
    end
    
    %%%%% create csv for gradient Q-learning

    numActionComponents = 6;
    numFullStateComponents = 9;
    numCol = (2 * numFullStateComponents) + numActionComponents + 1;
    finalCSV = zeros(numStates-1, numCol);

    for j = 1:numStates - 1
        finalCSV(j, 1:9) = raw_data(j, [3:5 15:20]);
        finalCSV(j, 10:15) = actionsTable(j, :);
        finalCSV(j, 16) = 0; % manually look through data to add reward values
        finalCSV(j, 17:numCol) = raw_data(j+1,[3:5 15:20]);
    end
    
    outputFilename = strcat('output', int2str(k), '.csv');
    
    writematrix(finalCSV, outputFilename);
    clear raw_data data actionsTable;

end










