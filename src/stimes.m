%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is part of the
% Disruption Predictor Feature Developer tools.
% These scripts were developed to utilize
% Support Vector Machines to evaluate diagnostic
% signals as features for predicting disruptions
%
% Developer: Matthew Parsons, mparsons@pppl.gov
%
% stimes.m - This script finds for each shot
% the earliest and latest times where
% all of the specified signals are active
% 
%
% Inputs:
% shot_dir/read_shots
% 
% Outputs:
% shot_dir/write_shots
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





% Load shot numbers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
shot_dir = '../data/shot_lists/';
read_shots = 'short_list.txt';
write_shots = 'short_list_times.txt';

DISRUPTIVE = 1;

if( DISRUPTIVE )
    formatSpec = '%d %f';
    sizeA = [2, Inf];
else
    formatSpec = '%d';
    sizeA = [1, Inf];
end

fileID = fopen(strcat(shot_dir,read_shots),'r');
shots = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);

ns = size(shots,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






% Identify paths to read data from
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1 Plasma Current
% 2 Mode Lock Amplitude
% 3 Plasma Internal Inductance
% 4 Density
% 5 Radiated Power
% 6 Total Input Power
% 7 d/dt Stored Diamagnetic Energy
% 8 Temperature Profile Information

include_sigs = [1:7];

% 1 JET
% 2 NSTX

include_machines = [1];


[sig_path,all_sig_dir,sig_dir] = get_signal_paths()

% run('signal_paths.m');

nsig = size(sig_dir{1},1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Set arrays to store times
all_data = zeros(3,ns);
all_data(1,:) = shots(1,:);


formatSpec = '%f %f';
sizeA = [2, Inf];



% For every shot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:ns

    t0max = 0;
    tfmin = 100.0;

    % For every signal
    for j = 1:nsig

        % Read in signal data for shot
        curr_shot_path = strcat(sig_path(1,:),sig_dir{1}(j,:),num2str(shots(1,i)),'.txt')
        fileID = fopen(curr_shot_path,'r');
        data = fscanf(fileID,formatSpec,sizeA);
        fclose(fileID);

        % Get min/max times
        t0max = max(t0max,data(1,1));
        tfmin = min(tfmin,data(1,end));

    end

    % If disruptive, set minmax to dtime
    if( DISRUPTIVE )
        tfmin = shots(2,i);
    end

    all_data(2,i) = t0max;
    all_data(3,i) = tfmin;
    
    fprintf('%06.3f %% \r',100*i/ns)

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% Output data
fileID = fopen(strcat(shot_dir,write_shots),'w');
fprintf(fileID,'%d %.4f %.4f\n', all_data);
fclose(fileID);


