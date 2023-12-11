function [action] = findActionDeltaPhi(difference)
format long;
numActionComponents = 6;
action = zeros(numActionComponents,1);
posThreshold = 0.00001;
posDelta = .00005;
oriThreshold = .0002;

%find max pos delta

%weighted_z_err = difference(1, 3)*1.3;
%difference(1, 3) = weighted_z_err;
[max_pos, pos_idx] = max(abs(difference(1, 1:3)));
[max_ori, ori_idx] = max(abs(difference(1, 4:6)));
ori_idx = ori_idx+3;

if max_pos >= posThreshold
    if difference(pos_idx) >= posThreshold
        action(pos_idx) = posDelta;
    else
        action(pos_idx) = -posDelta;
    end
end

if max_ori >= oriThreshold
    if difference(ori_idx) >= oriThreshold
        action(ori_idx) = oriThreshold;
    else
        action(ori_idx) = -oriThreshold;
    end
end

end
