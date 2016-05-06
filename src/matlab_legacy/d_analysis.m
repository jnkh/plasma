%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is part of the
% Disruption Predictor Feature Developer tools.
% These scripts were developed to utilize
% Support Vector Machines to evaluate diagnostic
% signals as features for predicting disruptions
%
% Developer: Matthew Parsons, mparsons@pppl.gov
%
% d_analysis.m - This script is called from test.m and
% contains the code for testing disruptive shots with
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




% identify false (early) alarms by time and position in layer_2 space
fa = find(M4_ctresult==1);
fa = fa(find((shot_start + fa*0.032) < (shot_end - ewt)))';

if(size(fa,2) ~= 0)
    fa_temp = zeros(6,size(fa,2));
    fa_temp(1,:) = str2num(shot_num);
    fa_temp(2,:) = shot_end - (size(test_shot,2)-fa')*0.032;
    fa_temp(3,:) = shot_end;
    fa_temp(4,:) = test_shot_m4(fa,1)';
    fa_temp(5,:) = test_shot_m4(fa,2)';
    fa_temp(6,:) = test_shot_m4(fa,3)';
    false_alarms_d = [false_alarms_d fa_temp];
end



% identify last three time segments and relative stats
for k = 1:3
    final_three(1,k,j) = str2num(shot_num);
    final_three(2,k,j) = (size(M4_ctresult,1)-k+2)*0.032+(shot_start-shot_end);
    final_three(3,k,j) = test_shot_m4(end-k+1,1);
    final_three(4,k,j) = test_shot_m4(end-k+1,2);
    final_three(5,k,j) = test_shot_m4(end-k+1,3);
end



% identify earliest correct alarms by time
ca = find(M4_ctresult==1)';
ca = ca(find((shot_start + ca*0.032) > (shot_end - ewt)));
if(size(ca,2) ~= 0)
    correct_alarms(end+1) = -0.032*(size(test_shot,2)-ca(1));
    overall_accuracy(1,1) = overall_accuracy(1,1) + 1;
else
    overall_accuracy(1,2) = overall_accuracy(1,2) + 1;
    missed_disruptions = [missed_disruptions str2num(shot_num)];
end




% identify interval before disruption
ta = M4_ctresult(end-4:end);
times = [size(M4_ctresult,1)-4:size(M4_ctresult,1)];
times = -0.032*(size(test_shot,2)-times);
ta_temp = [times; double(ta')];
total_alarms = total_alarms + ta_temp;
           



% give totals for correct/incorrect classifications
wrong_segs = size(fa,2);

segment_accuracy(1,1) = segment_accuracy(1,1) + size(M4_ctresult,1)-wrong_segs;
segment_accuracy(1,2) = segment_accuracy(1,2) + wrong_segs;

