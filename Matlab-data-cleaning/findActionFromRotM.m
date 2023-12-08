function [action] = findActionFromRotM(difference)
format long;
numActionComponents = 6;
action = zeros(numActionComponents,1);
posThreshold = 0.00005;
oriThreshold = 0.0006;
oriDelta = .0002;

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
        elseif difference < oriDelta
            action(i) = -oriDelta;
        else
            action(i) = oriDelta;
        end
    end
end

end

