function [action] = findActionDeltaPhi(difference)
format long;
numActionComponents = 6;
action = zeros(numActionComponents,1);
posThreshold = 0.00005;
oriThreshold = .0002;

for i = 1:numActionComponents
    if i <= 3
        if abs(difference(i)) < posThreshold
            action(i) = 0;
        elseif difference < posThreshold
            action(i) = -posThreshold;
        else
            action(i) = posThreshold;
        end
    else
        if abs(difference(i)) < oriThreshold
            action(i) = 0;
        elseif difference < oriThreshold
            action(i) = -oriThreshold;
        else
            action(i) = oriThreshold;
        end
    end
end

end
