%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is part of the
% Disruption Predictor Feature Developer tools.
% These scripts were developed to utilize
% Support Vector Machines to evaluate diagnostic
% signals as features for predicting disruptions
%
% Developer: Matthew Parsons, mparsons@pppl.gov
%
% setup.m - This script takes the specified
% features and compiles the data into a single
% file to be used by the SVM algorithms
% 
%
% Inputs:
% shot_dir/
% 
% Outputs:
% tt_dir/.FEATURES/train/disrupt.txt
% tt_dir/.FEATURES/train/ndisrupt.txt
% tt_dir/.FEATURES/test/features.txt
% tt_dir/.FEATURES/test/shots.txt
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




tt_dir = '/p/datad/svm_results/test_07/t01/';
[status,message,messageid] = mkdir(strcat(tt_dir,'.FEATURES'));
[status,message,messageid] = mkdir(strcat(tt_dir,'.FEATURES/train'));
[status,message,messageid] = mkdir(strcat(tt_dir,'.FEATURES/test'));



% Specify features to use
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1-2 Plasma Current (mean,stddev)
% 3-4 Mode Lock Amplitude (mean,stddev)
% 5-6 Plasma Internal Inductance (mean,stddev)
% 7-8 Density (mean,stddev)
% 9-10 Radiated Power (mean,stddev)
% 11-12 Total Input Power (mean,stddev)
% 13-14 d/dt Stored Diamagnetic Energy (mean,stddev)
% 15-40 Temperature Profile Information (mean,stddev)

include_feats = [1:14];

% 1 JET
% 2 NSTX

include_machines = [1 2];

run('feature_paths.m');

nmach = size(include_machines,2);
nfeat = size(feat_dir{1},1);

feat_write = '%f';
for i = 2:nfeat-1
    feat_write = strcat(feat_write,' %f');
end
feat_write = strcat(feat_write,' %f\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






% Load shot numbers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
shot_dir = '/p/datad/dpfd/shot_lists/';

formatSpec = '%d %f %f';
sizeA = [3, Inf];

fileID = fopen(strcat(shot_dir,'disruptC_times_cf_ece.txt'),'r');
disruptC = fscanf(fileID,formatSpec,sizeA);
disruptC = [disruptC; ones(1,size(disruptC,2))];
fclose(fileID);

fileID = fopen(strcat(shot_dir,'disruptB_times_cf_ece.txt'),'r');
disruptB = fscanf(fileID,formatSpec,sizeA);
disruptB = [disruptB; ones(1,size(disruptB,2))];
fclose(fileID);

fileID = fopen(strcat(shot_dir,'nondisruptB_times_cf_ece.txt'),'r');
ndisruptB = fscanf(fileID,formatSpec,sizeA);
ndisruptB = [ndisruptB; ones(1,size(ndisruptB,2))];
fclose(fileID);

fileID = fopen(strcat(shot_dir,'disrupt_nstx_times_cf.txt'),'r');
disruptNSTX = fscanf(fileID,formatSpec,sizeA);
disruptNSTX = [disruptNSTX; ones(1,size(disruptNSTX,2))*2];
fclose(fileID);

test_all = [disruptNSTX];
train_disrupt = [disruptC];
train_ndisrupt = [disruptC];

shots_all = [train_disrupt train_ndisrupt test_all];
[shots_unique, ia, ic] = unique(shots_all(1,:));
shots = shots_all(:,ia);

t0 = shots(2,:);
tf = shots(3,:);
ns = size(shots,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



d_samp = 3;
nd_samp = 3;

d_train = zeros(size(train_disrupt,2)*d_samp,nfeat);
nd_train = zeros(size(train_ndisrupt,2)*nd_samp,nfeat);




% For every shot generate feature vectors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:ns

    shot = num2str(shots(1,i));
    mach = shots(4,i);
    

    % Read all of the relevant signals into arrays
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    load_string = strcat(feat_path(mach,:),feat_dir{mach}(1,:),shot,'.txt');
    y = load(load_string);

    % Number of 32 ms time intervals in domain
    ni = floor( (tf(i)-t0(i)) / 0.032 );
    y = y(end-ni+1:end);
    
    feature_vec = zeros(ni,nfeat);
    feature_vec(:,1) = y;
    
    for j = 2:nfeat
        load_string = strcat(feat_path(mach,:),feat_dir{mach}(j,:),shot,'.txt');
        y = load(load_string);
        feature_vec(:,j) = y(end-ni+1:end);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    % Output testing data
    if(ismember(str2num(shot),test_all(1,:)))
        write_string = strcat(tt_dir,'.FEATURES/test/',shot,'.txt');
        fileID = fopen(write_string,'w');
        fprintf(fileID, feat_write, feature_vec');
        fclose(fileID);
    end
                
    % Save disruptive training data
    if(ismember(str2num(shot),train_disrupt(1,:)))
                
        d_train(d_samp*(i-1)+1:d_samp*(i-1)+d_samp,:) = feature_vec(end-d_samp+1:end,:);

    end
                
    % Save nondisruptive training data
    if(ismember(str2num(shot),train_ndisrupt(1,:)))
                
        nd_train(nd_samp*(i-1)+1:nd_samp*(i-1)+nd_samp,:) = feature_vec(end-30-nd_samp:end-31,:);

    end


    fprintf('%06.3f %% \r',100*(i-1)/ns);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                

% Output training data
                
write_string = strcat(tt_dir,'.FEATURES/train/disrupt.txt');
fileID = fopen(write_string,'w');
fprintf(fileID, feat_write, d_train');
fclose(fileID);
        
write_string = strcat(tt_dir,'.FEATURES/train/ndisrupt.txt');
fileID = fopen(write_string,'w');
fprintf(fileID, feat_write, nd_train');
fclose(fileID);

write_string = strcat(tt_dir,'.FEATURES/train/features.txt');
fileID = fopen(write_string,'w');
for i = 1:nmach
    for j = 1:nfeat
        fprintf(fileID,'%s\n',feat_dir{i}(j,:));
    end
end
fclose(fileID);

