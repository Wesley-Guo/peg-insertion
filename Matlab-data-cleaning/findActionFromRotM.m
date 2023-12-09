function [action] = findActionFromRotM(difference)
format long;
numActionComponents = 6;
action = zeros(numActionComponents,1);
posThreshold = 0.00005;
oriThreshold = 0.0006;
oriDelta = .0002;

%find max pos delta
[max_pos, pos_idx] = max(abs(difference(1, 1:3)));
[max_ori, ori_idx] = max(abs(difference(1, 4:6)));
ori_idx = ori_idx+3;

if max_pos >= posThreshold
    if difference(pos_idx) >= posThreshold
        action(pos_idx) = posThreshold;
    else
        action(pos_idx) = -posThreshold;
    end
end

if max_ori >= oriThreshold
    if difference(ori_idx) >= oriThreshold
        action(ori_idx) = oriDelta;
    else
        action(ori_idx) = -oriDelta;
    end
end

end

