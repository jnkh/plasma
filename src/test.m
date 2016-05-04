%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is part of the
% Disruption Predictor Feature Developer tools.
% These scripts were developed to utilize
% Support Vector Machines to evaluate diagnostic
% signals as features for predicting disruptions
%
% Developer: Matthew Parsons, mparsons@pppl.gov
%
% test.m - This script is called from vars.m and
% contains the code for training and testing
% the SVM predictor, and outputting data to files
%
%
% Inputs:
% none
% 
% Outputs:
% out_dir/false_alarms_d.dat
% out_dir/false_alarms_nd.dat
% out_dir/missed_disruptions.dat
% out_dir/correct_alarms_binned.dat
% out_dir/final_three.dat
% out_dir/total_alarms.dat
% out_dir/run.out
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





% Set up variables for tracking performance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% False alarms (shot,time,tend,coordinates)
false_alarms_nd = [];
false_alarms_d = [];

% Correct alarms by earliest warning time later than early warning time
correct_alarms = [];

% Final three segments in a disruptive shot
final_three = zeros(5,3,num_d_test*cv_num);

% Total alarms at given time before disruption
total_alarms = zeros(2,5);

% Shot accuracy (d/nd,correct/incorrect)
overall_accuracy = zeros(2);

% Time segment accuracy (d/nd,correct/incorrect)
segment_accuracy = zeros(2);

% Missed disruptions
missed_disruptions = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






% Define other variables needed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Progress increment
tot_test = 0;
tot_test = tot_test + (num_d_train + num_nd_train)*(test_type==1|test_type==3);
tot_test = tot_test + (num_d_test + num_nd_test)*(test_type==2|test_type==3);
dp = 1/(tot_test*cv_num);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





% Get initialization stats
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tstart = datestr(now);
tsnow = now;
tstart = strcat(num2str(sprintf('%02.0f',hour(tsnow))),':');
tstart = strcat(tstart,num2str(sprintf('%02.0f',minute(tsnow))),':');
tstart = strcat(tstart,num2str(sprintf('%02.0f',second(tsnow))));

run_date = strcat(num2str(year(today)),'-',num2str(sprintf('%02d',month(today))));
run_date = strcat(run_date,'-',num2str(sprintf('%02d',day(today))));

system('echo Starting SVM testing');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





% Repeat the following section for cross-validation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:cv_num
    progress = (i-1)/cv_num;
    fprintf('%06.3f %% \r',100*progress);


    
    % Sample training data without replacement
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    rng(rng_dtrain+i-1);
    d_rand = uint16(randperm(num_d_train_set));
    d_train = sort(d_rand(1:num_d_train));

    rng(rng_ndtrain+i-1);
    nd_rand = uint16(randperm(num_nd_train_set*ndd_bal));
    nd_train = sort(nd_rand(1:num_nd_train*ndd_bal));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    

    
    % Train predictors and collect self-consistency stats
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    train_groups = int8([ones(num_d_train,1); zeros(num_nd_train*ndd_bal,1)]);
    
    %tic
    M1 = fitcsvm([dtrain_m1([d_train],:); ndtrain_m1([nd_train],:)],train_groups, 'KernelFunction', 'rbf', 'KernelScale', sqrt(1/gamma), 'BoxConstraint', C);
    %toc
    M2 = fitcsvm([dtrain_m2([d_train],:); ndtrain_m2([nd_train],:)],train_groups, 'KernelFunction', 'rbf', 'KernelScale', sqrt(1/gamma), 'BoxConstraint', C);
    %toc
    M3 = fitcsvm([dtrain_m3([d_train],:); ndtrain_m3([nd_train],:)],train_groups, 'KernelFunction', 'rbf', 'KernelScale', sqrt(1/gamma), 'BoxConstraint', C);
    %toc
    
    [M1_cresult, M1_dresult] = predict(M1,[dtrain_m1([d_train],:); ndtrain_m1([nd_train],:)]);
    [M2_cresult, M2_dresult] = predict(M2,[dtrain_m2([d_train],:); ndtrain_m2([nd_train],:)]);
    [M3_cresult, M3_dresult] = predict(M3,[dtrain_m3([d_train],:); ndtrain_m3([nd_train],:)]);
    
    M4 = fitcsvm([M1_dresult(:,2),M2_dresult(:,2),M3_dresult(:,2)],train_groups, 'KernelFunction', 'linear');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    


    % Sample testing set without replacement
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    rng(rng_dtest+i-1);
    d_rand = uint16(randperm(num_d_test_set));
    %d_test = sort(d_rand(end-num_d_test:end));
    d_test = uint16([1:num_d_test_set]);

    rng(rng_ndtest+i-1);
    nd_rand = uint16(randperm(num_nd_test_set));
    %nd_test = sort(nd_rand(end-num_nd_test:end));
    nd_test = uint16([1:num_nd_test_set]);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



    % Collect data and test model
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Complete testing with full training shots
    if(test_type == 1 | test_type == 3)
    
        % Test disruptive shots
        dnd = 1;
        for j = 1:num_d_train
        
            shot_num = num2str(sd_train(1,d_train(j)));
            shot_start = sd_train(2,d_train(j));
            shot_end = sd_train(3,d_train(j));
            run(strcat(dpfd_dir,'d_analysis.m'));
            
            progress = progress + dp;
            fprintf('%06.3f %% \r',100*progress)
        end
        
        % Test nondisruptive shots
        dnd = 0;
        for j = 1:num_nd_train
        
            shot_num = num2str(snd_train(1,nd_train(j)));
            shot_start = snd_train(2,nd_train(j));
            shot_end = snd_train(3,nd_train(j));
            run(strcat(dpfd_dir,'nd_analysis.m'));
            
            progress = progress + dp;
            fprintf('%06.3f %% \r',100*progress)
        end
        
    end

    % Complete testing with testing shots
    if(test_type == 2 | test_type == 3)
    
        % Test disruptive shots
        dnd = 1;
        for j = 1:num_d_test
            shot_num = num2str(sd_test(1,d_test(j)));
            shot_start = sd_test(2,d_test(j));
            shot_end = sd_test(3,d_test(j));
            run(strcat(dpfd_dir,'d_analysis.m'));
            
            progress = progress + dp;
            fprintf('%06.3f %% \r',100*progress)
        end
        
        % Test nondisruptive shots
        dnd = 0;
        for j = 1:num_nd_test

            shot_num = num2str(snd_test(1,nd_test(j)));
            shot_start = snd_test(2,nd_test(j));
            shot_end = snd_test(3,nd_test(j));
            run(strcat(dpfd_dir,'nd_analysis.m'));
            
            progress = progress + dp;
            fprintf('%06.3f %% \r',100*progress)
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



    % Collect stats on performance
    


end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of cross-validation section
system('echo Finished SVM testing');
system('echo Calculating stats');





% Calculate and print statistics on model performance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Print false alarms (shot, t, tend, m1, m2, m3)
fileID = fopen(strcat(out_dir,'/false_alarms_d.dat'),'w');
fprintf(fileID, '%d %f %f %f %f %f\n', false_alarms_d);
fclose(fileID);
fileID = fopen(strcat(out_dir,'/false_alarms_nd.dat'),'w');
fprintf(fileID, '%d %f %f %f %f %f\n', false_alarms_nd);
fclose(fileID);


% Print missed disruptions
fileID = fopen(strcat(out_dir,'/missed_disruptions.dat'),'w');
fprintf(fileID, '%d\n', missed_disruptions);
fclose(fileID);


% Print correct alarm stats
bins = max(abs(correct_alarms)/0.032);
b = zeros(1,bins+1);
for i = 0:bins
    b(1,i+1) = sum(correct_alarms < -0.032*(i-1));
end
b = b / (num_d_test*cv_num);
c = [0:-0.032:-bins*0.032];
ca_dat = [c;b];
fileID = fopen(strcat(out_dir,'/correct_alarms_binned.dat'),'w');
fprintf(fileID, '%f %f\n', ca_dat);
fclose(fileID);


% Print final three alarms per shot final_three = zeros(num_d_test*cv_num,3,5)
fileID = fopen(strcat(out_dir,'/final_three.dat'),'w');
fprintf(fileID,'%d %f %f %f %f %d %f %f %f %f %d %f %f %f %f\n',final_three);
fclose(fileID);


% Print total alarm stats
total_alarms = total_alarms/(num_d_test*cv_num);
%{
%binwidth = 0.032;
%[b, c] = hist(total_alarms(2,:),(max(total_alarms(1,:))-min(total_alarms(1,:))+1)/binwidth);
%ca_dat = [c;b];
%}
fileID = fopen(strcat(out_dir,'/total_alarms.dat'),'w');
fprintf(fileID, '%f %f\n', total_alarms);
fclose(fileID);


% Shot accuracy (# correct, # incorrect)
overall_accuracy(1,:) = overall_accuracy(1,:) / sum(overall_accuracy(1,:));
overall_accuracy(2,:) = overall_accuracy(2,:) / sum(overall_accuracy(2,:));
segment_accuracy(1,:) = segment_accuracy(1,:) / sum(segment_accuracy(1,:));
segment_accuracy(2,:) = segment_accuracy(2,:) / sum(segment_accuracy(2,:));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





% Get completion stats
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tenow = now;
tend = strcat(num2str(sprintf('%02.0f',hour(tenow))),':');
tend = strcat(tend,num2str(sprintf('%02.0f',minute(tenow))),':');
tend = strcat(tend,num2str(sprintf('%02.0f',second(tenow))));

tdiff = tenow - tsnow;
elapsed = strcat(num2str(sprintf('%02.0f',hour(tdiff))),':');
elapsed= strcat(elapsed,num2str(sprintf('%02.0f',minute(tdiff))),':');
elapsed = strcat(elapsed,num2str(sprintf('%06.3f',second(tdiff))));

fileID = fopen(strcat(out_dir,'/run.out'),'w');
fprintf(fileID, '%s %s %s %s\n', run_date, tstart, tend, elapsed);
fprintf(fileID, '%f %f\n', C, gamma);
fprintf(fileID, '%d %d %d %d\n', num_d_train, num_d_test, num_nd_train, num_nd_test);
fprintf(fileID, '%f %f\n', overall_accuracy(1,1), overall_accuracy(1,2));
fprintf(fileID, '%f %f\n', overall_accuracy(2,1), overall_accuracy(2,2));
fprintf(fileID, '%f %f\n', segment_accuracy(1,1), segment_accuracy(1,2));
fprintf(fileID, '%f %f\n', segment_accuracy(2,1), segment_accuracy(2,2));
fclose(fileID);
                               
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


