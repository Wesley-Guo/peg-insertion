clear all; clc
format long;

%raw_data = csvread('data_2023-12-02.16.48.48.csv',1,2);

numOfTraj = 23;

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
            max_rwd_idx = j;
            clipping_idx = j+10;
        end
        finalCSV(j, 10:15) = actionsTable(j, :);
        
        % add negative reward for high force
        f_reward = sum(abs(raw_data(j, 9:11)));
        if f_reward > 20
            f_reward = -.001*(f_reward)^2;
        else
            f_reward = 0;
        end
        % add positive reward for smaller pos errors
        x_err = data(j, 1);
        y_err = data(j, 2);
        z_err = data(j, 3);
        p_reward = 1/(50*sqrt(x_err^2 + y_err^2 + z_err^2));
        
        finalCSV(j, 16) = f_reward + p_reward; 
        finalCSV(j, 17:numCol) = raw_data(j+1, 3:11);
    end
    
    % add high reward for z-depth and clip all remaining states
    finalCSV(max_rwd_idx, 16) = 100;
    outputCSV = zeros(clipping_idx, numCol);
    outputCSV = finalCSV(1:clipping_idx, 1:numCol);
    
    outputFilename = strcat('output', int2str(k), '.csv');
    
    writematrix(outputCSV, outputFilename);
    clear raw_data data actionsTable;

end



