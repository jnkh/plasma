%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is part of the
% Disruption Predictor Feature Developer tools.
% These scripts were developed to utilize
% Support Vector Machines to evaluate diagnostic
% signals as features for predicting disruptions
%
% Developer: Matthew Parsons, mparsons@pppl.gov
%
% current_filter.m - This script scans through shots
% to identify when their current first exceeds a
% specified threshold (default 750 kA)
% 
%
% Inputs:
% shot_dir/read_shots
% path/shot.txt
% 
% Outputs:
% shot_dir/write_shots
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





% Load shot numbers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
shot_dir = '../data/shot_lists/';
read_shots = 'short_list.txt';
write_shots = 'short_list_cf.txt';


current_threshold = 750000;


formatSpec = '%d %f %f';
sizeA = [3, Inf];

fileID = fopen(strcat(shot_dir,read_shots),'r');
shots = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);

ns = size(shots,2);

fileID = fopen(strcat(shot_dir,write_shots),'w');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Path to plasma current
path = '../data/signal_data/jet/jpf/da/c2-ipla/';
fileIDo = fopen(strcat(shot_dir,write_shots),'w');


formatSpec = '%f %f';
sizeA = [2,Inf];

% For every shot
for i = 1:ns

    % Read current
    fileID = fopen(strcat(path,num2str(shots(1,i)),'.txt'),'r');
    data = fscanf(fileID,formatSpec,sizeA);
    fclose(fileID);

    
    % Find first point where current is more than 750 kA (Ip has +/- direction)
    it = min(find(abs(data(2,:)) > current_threshold));
    

    % Print to out file
    fprintf(fileIDo,'%d %.4f %.4f\n', shots(1,i),data(1,it),shots(3,i));

    fprintf('%06.3f %% \r',100*i/ns)
    
end

fclose(fileIDo);



