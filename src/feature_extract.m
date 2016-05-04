%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is part of the
% Disruption Predictor Feature Developer tools.
% These scripts were developed to utilize
% Support Vector Machines to evaluate diagnostic
% signals as features for predicting disruptions
%
% Developer: Matthew Parsons, mparsons@pppl.gov
%
% feature_extract.m - This script extracts the
% defined features from specified signal data
% 
% 
% Inputs:
% shot_dir/read_shots
% 
% Outputs:
% ppf_path/feat_path/shots.txt
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





% Load shot numbers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
shot_dir = '/p/datad/dpfd/shot_lists/';
read_shots = 'disrupt_nstx_times_cf.txt';

formatSpec = '%d %f %f';
sizeA = [3, Inf];

fileID = fopen(strcat(shot_dir,read_shots),'r');
shots = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);

t0 = shots(2,:);
tf = shots(3,:);
ns = size(shots,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% Identify path to read data
sig_path = '/p/datad/signal_data/nstx/jpf/nbi/nb_p_inj/';
ppf_path = '/p/datad/signal_data/nstx/ppf/nbi/nb_p_inj/';


% Index of array to get stats for
n = 2;
nt = 2;
sizeA = [nt, Inf];
feat_form = '%f';
for i = 2:nt
    formatSpec = strcat(formatSpec,' %f');
end



% Specify feature type... 1:mean, 2:stddev
feat_n = 2;

switch feat_n
    case 1
        feat_path = 'mean/';
    case 2
        feat_path = 'stddev/';
end
[status,message,messageid] = mkdir(strcat(ppf_path,feat_path));


% Get normalization info after running sig_norm.m
fminmax = strcat(ppf_path,'minmax.txt');
minmax = load(fminmax);
smin = minmax(1);
smax = minmax(2);


% For every shot
for i = 1:ns


    % Read in data and normalize
    fileID = fopen(strcat(sig_path,num2str(shots(1,i)),'.txt'),'r');
    data = fscanf(fileID,feat_form,sizeA);
    fclose(fileID);
    data(n,:) = (data(n,:) - smin) / (smax - smin);
    data = data(:,data(1,:) < tf(i)+1);
    

    % Number of 32 ms time intervals in domain
    ni = floor( (tf(i)-t0(i)) / 0.032 );
    feat = zeros(1,ni);


    % Interpolate to normalize data to 1 ms sample rate
    ty = zeros(2,ni*32);
    ty(1,:) = [tf(i)-ni*0.032:0.001:tf(i)-0.001];
    ty(2,:) = interp1(data(1,:),data(n,:),ty(1,:));
    

    % For each interval
    for j = 1:ni

        % Calculate feature... 1:mean, 2:stddev

        switch feat_n
            case 1
                feat(j) = mean(ty(2,1+(j-1)*32:j*32));
            case 2
                trans = fft(ty(2,1+(j-1)*32:j*32));
                feat(j) = std(abs(trans(2:17)/32));
        end
    end

    % Print to ppf file
    fileID = fopen(strcat(ppf_path,feat_path,num2str(shots(1,i)),'.txt'),'w');
    fprintf(fileID,'%d\n',feat);
    fclose(fileID);

    fprintf('%06.3f %% \r',100*i/ns)
end




