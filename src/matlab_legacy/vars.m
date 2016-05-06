%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is part of the
% Disruption Predictor Feature Developer tools.
% These scripts were developed to utilize
% Support Vector Machines to evaluate diagnostic
% signals as features for predicting disruptions
%
% Developer: Matthew Parsons, mparsons@pppl.gov
%
% vars.m - To run the SVM routines, this script
% should be copied to your working directory,
% edited to specify the desired variables,
% and run from Matlab (R2014a or newer)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





% COPY THIS FILE INTO THE WORKING DIRECTORY FOR THE CURRENT TRIAL
out_dir = pwd;
dpfd_dir = '/u/mparsons/DPFD/';

% Training/Testing data directory
tt_dir = '/p/datad/svm_results/test_06/t15/.FEATURES/';




% Load shot numbers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
shot_dir = '/p/datad/dpfd/shot_lists/';

formatSpec = '%d %f %f';
sizeA = [3, Inf];

% Read in shots to use
fileID = fopen(strcat(shot_dir,'disruptB_times_cf_ece.txt'),'r');
disruptB = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);

fileID = fopen(strcat(shot_dir,'disruptC_times_cf_ece.txt'),'r');
disruptC = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);

fileID = fopen(strcat(shot_dir,'nondisruptB_times_cf_ece.txt'),'r');
ndisruptB = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






% Define training and test sets to sample from
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Disruptive and nondisruptive sets for training and testing
sd_train = disruptC;
sd_test = disruptB;
snd_train = disruptC;
snd_test = ndisruptB;


% Sizes of sets
num_d_train_set = size(sd_train,2);
num_d_test_set = size(sd_test,2);
num_nd_train_set = size(snd_train,2);
num_nd_test_set = size(snd_test,2);


% Define sampling numbers for disruptive shots
num_d_train = num_d_train_set;
num_d_test = num_d_test_set;


% Define sampling numbers for nondisruptive shots
num_nd_train = num_d_train_set;
num_nd_test = num_d_test_set;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






% Variables for testing/training procedure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Test Type: (0) No Testing, (1) Training Set, (2) Testing Set, (3) Both Sets
test_type = 2;


% RBF kernal parameters
C = 10^(3);
gamma = 10^(1);


% Ratio between nd/d training samples
ndd_bal = 1;


% Number of features used
nfeat = 14;


% Identifiers for reading data
feat_read = '%f';
for i = 2:nfeat
    feat_read = strcat(feat_read,' %f');
end


% Early warning time (seconds)
ewt = 1.0;


% Number of models to generate for cross-validation
cv_num = 1;


% Seed rng
rng('shuffle');
srng = rng;
srng = 1;

% Decide random seeds for training and testing data
rng_dtrain = srng;
rng_ndtrain = srng;

% Use remainder of training shots for testing
rng_dtest = srng;
rng_ndtest = srng;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






% Load training data into train_disrupt, train_ndisrupt
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sizeA = [nfeat, Inf];

fid = fopen(strcat(tt_dir,'train/disrupt.txt'),'r');
train_disrupt = fscanf(fid,feat_read,sizeA);
fclose(fid);
train_disrupt = train_disrupt';

fid = fopen(strcat(tt_dir,'train/ndisrupt.txt'),'r');
train_ndisrupt = fscanf(fid,feat_read,sizeA);
fclose(fid);
train_ndisrupt = train_ndisrupt';

dtrain_seq = train_disrupt;
ndtrain_seq = train_ndisrupt;

dtrain_m1 = dtrain_seq(3:3:end,:);
dtrain_m2 = dtrain_seq(2:3:end,:);
dtrain_m3 = dtrain_seq(1:3:end,:);
ndtrain_m1 = ndtrain_seq(3:3:end,:);
ndtrain_m2 = ndtrain_seq(2:3:end,:);
ndtrain_m3 = ndtrain_seq(1:3:end,:);

clear disruptC disruptB ndisruptB train_disrupt train_ndisrupt;
clear dtrain_seq ndtrain_seq;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






system('echo Data Loaded');

run(strcat(dpfd_dir,'test.m'));

if(test_type > 0)
    run(strcat(dpfd_dir,'plots.m'));
end

system('echo Job Completed');

exit;

