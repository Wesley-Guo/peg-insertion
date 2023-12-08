numOfTraj = 1;

for k = 1:numOfTraj
    filename = strcat(int2str(k), '.csv');
    raw_data = csvread(filename, 1, 0);
    
    numStates = length(raw_data(:, 1));
    rotTable = raw_data(:, 6:14);
    eulerTable = zeros(numStates, 3);
    
    for i = 1:numStates
        rotV1 = rotTable(i, 1:3);
        rotV2 = rotTable(i, 4:6);
        rotV3 = rotTable(i, 7:9);
        rotM = [rotV1; rotV2; rotV3];
        eulerZYX = rotm2eul(rotM);
        eulerZYX = eulerZYX + pi;
        eulerTable(i, :) = eulerZYX;
    end

    
end



