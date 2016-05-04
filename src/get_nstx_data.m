%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is part of the
% Disruption Predictor Feature Developer tools.
% These scripts were developed to utilize
% Support Vector Machines to evaluate diagnostic
% signals as features for predicting disruptions
%
% Developer: Matthew Parsons, mparsons@pppl.gov
%
% get_nstx_data.m - This script accesses MDSplus to 
% download and save new signal data locally
% 
% 
% Inputs:
% shot_dir/read_shots
%
% Outputs:
% cname_out/shots.txt
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






% If running in "parallel"
proc = 1;
tproc = 1;

cname_out = [];

% Set up data paths
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p = '/p/datad/signal_data/nstx/jpf/';
tree = 'efit02';
node = 'wpdot';

nc = 1; % number of channels
for i = 1:nc
    cname_out = char(cname_out, strcat(p,tree,'/',node,'/'));
    [status,message,messageid] = mkdir(cname_out(i+1,:));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cname_out = cname_out(2:end,:);



% Load shot numbers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
shot_dir = '/p/datad/dpfd/shot_lists/';
read_shots = 'disrupt_nstx.txt';

formatSpec = '%d %f';
sizeA = [2, Inf];

fileID = fopen(strcat(shot_dir,read_shots),'r');
shots = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);

ns = size(shots,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




shots_proc = floor(ns/tproc);
nstart = (proc-1)*shots_proc + 1;
nstop = proc*shots_proc;
if(proc == tproc)
    nstop = ns;
end





mdsconnect('skylark.pppl.gov:8501::');
% For every shot, get and print data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = nstart:nstop
    for j = 1:nc

        mdsopen(tree,shots(1,i));
        mdsarg = strcat('_sig=\',node);
        y = mdsvalue(char(mdsarg));
        x = mdsvalue('dim_of(_sig)');
        mdsvalue('units(_sig)')
        purple
        fileID = fopen(strcat(cname_out(j,:),num2str(shots(1,i)),'.txt'),'w');
        if(size(y,1) > 0)
            fprintf(fileID, '%f %f\n', [x'; y']);
        else
            fprintf(fileID, '%s', ' ');
        end
        fclose(fileID);
                        
        [shots(1,i), j, nc*(i-1) + j, ns*nc]
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
