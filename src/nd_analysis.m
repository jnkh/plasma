%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is part of the
% Disruption Predictor Feature Developer tools.
% These scripts were developed to utilize
% Support Vector Machines to evaluate diagnostic
% signals as features for predicting disruptions
%
% Developer: Matthew Parsons, mparsons@pppl.gov
%
% nd_analysis.m - This script is called from test.m and
% contains the code for testing nondisruptive shots with
% the SVM predictor
% 
%
% Inputs:
% tt_dir/test/shot_num.txt
% 
% Outputs:
% none
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







% Load data from tt_dir/test/shot_num.txt

sizeA = [nfeat, Inf];

fid = fopen(strcat(tt_dir,'test/',shot_num,'.txt'),'r');
test_shot = fscanf(fid,feat_read,sizeA);
fclose(fid);



% Run SVM

[M1_ctresult, M1_dtresult] = predict(M1,test_shot(:,3:end)');
[M2_ctresult, M2_dtresult] = predict(M2,test_shot(:,2:end-1)');
[M3_ctresult, M3_dtresult] = predict(M3,test_shot(:,1:end-2)');

test_shot_m4 = [M1_dtresult(:,2),M2_dtresult(:,2),M3_dtresult(:,2)];
[M4_ctresult, M4_dtresult] = predict(M4,test_shot_m4);




% identify false alarms by time and position in layer_2 space
fa = find(M4_ctresult==1)';

if(size(fa,2) ~= 0)
    fa_temp = zeros(6,size(fa,2));
    fa_temp(1,:) = str2num(shot_num);
    fa_temp(2,:) = shot_end - (size(test_shot,2)-fa')*0.032;
    fa_temp(3,:) = shot_end;
    fa_temp(4,:) = test_shot_m4(fa,1)';
    fa_temp(5,:) = test_shot_m4(fa,2)';
    fa_temp(6,:) = test_shot_m4(fa,3)';
    false_alarms_nd = [false_alarms_nd fa_temp];
end



% give totals for correct/incorrect classifications
wrong_segs = size(fa,2);

segment_accuracy(2,1) = segment_accuracy(2,1) + size(M4_ctresult,1)-wrong_segs;
segment_accuracy(2,2) = segment_accuracy(2,2) + wrong_segs;

if(wrong_segs == 0)
    overall_accuracy(2,1) = overall_accuracy(2,1) + 1;
else
    overall_accuracy(2,2) = overall_accuracy(2,2) + 1;
end




