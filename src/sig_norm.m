%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is part of the
% Disruption Predictor Feature Developer tools.
% These scripts were developed to utilize
% Support Vector Machines to evaluate diagnostic
% signals as features for predicting disruptions
%
% Developer: Matthew Parsons, mparsons@pppl.gov
%
% sig_norm.m - This script looks through all of 
% the shots to find the min/max values 
% for the specified signal
% 
%
% Inputs:
% shot_dir/read_shots
% sig_path_r/shot.txt
% 
% Outputs:
% sig_path_w/minmax.txt
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





% Load shot numbers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
shot_dir = '../data/shot_lists/';
read_shots = 'short_list_times_cf.txt';

formatSpec = '%d %f %f';
sizeA = [3, Inf];

fileID = fopen(strcat(shot_dir,read_shots),'r');
shots = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);

ns = size(shots,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





% Specify signal to normalize
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sig_path_r = '/p/datad/signal_data/jet/ppf/KK3/rtp/';
sig_path_w = '/p/datad/signal_data/jet/ppf/KK3/P350/';

% Index of array to get stats for
n = 14;
nt = 14;
sizeA = [nt, Inf];

formatSpec = '%f';
for i = 2:nt
    formatSpec = strcat(formatSpec,' %f');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



minmax = zeros(2,ns);




% For every shot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:ns

    % Read file
    fileID = fopen(strcat(sig_path_r,num2str(shots(1,i)),'.txt'),'r');
    data = fscanf(fileID,formatSpec,sizeA);
    fclose(fileID);

    % Ignore data outside time domain
    data = data(:,data(1,:)<shots(3,i));
    data = data(:,data(1,:)>shots(2,i));

    % Find min/max
    minmax(1,i) = min(data(n,:));
    minmax(2,i) = max(data(n,:));

    fprintf('%06.3f %% \r',100*i/ns)

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% Print min/max to file
fileID = fopen(strcat(sig_path_w,'minmax.txt'),'w');
fprintf(fileID,'%e %e\n', min(minmax(1,:)), max(minmax(2,:)));
fclose(fileID);

