%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is part of the
% Disruption Predictor Feature Developer tools.
% These scripts were developed to utilize
% Support Vector Machines to evaluate diagnostic
% signals as features for predicting disruptions
%
% Developer: Matthew Parsons, mparsons@pppl.gov
%
% get_jet_data.m - This script accesses MDSplus to 
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

cname = [];
cname_out = [];

% Set up data paths
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p = 'data/signal_data/jet/';

%location of jet signals 
signals_dirs = {'jpf/da/c2-ipla',
'jpf/da/c2-loca',
'jpf/db/b5r-ptot>out',
'jpf/df/g1r-lid:003',
'jpf/gs/bl-li<s',
'jpf/gs/bl-fdwdt<s',
'jpf/gs/bl-ptot<s',
'jpf/gs/bl-wmhd<s'}





nc = 3; % number of channels
for i = 1:nc
    cname = char(cname, char(strcat(signals_dirs(i),'/')));
    cname_out = char(cname_out, char(strcat(p,signals_dirs(i),'/')));
    [status,message,messageid] = mkdir(cname_out(i+1,:));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cname = cname(2:end,:);
cname_out = cname_out(2:end,:);



% Load shot numbers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
shot_dir = '../data/shot_lists/';
read_shots = 'short_list.txt';

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





mdsconnect('mdsplus.jet.efda.org');
% For every shot, get and print data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = nstart:nstop
    for j = 1:nc
    
        mdsarg = strcat('_sig=jet(',cname(j,:),',',num2str(shots(1,i)),')');
        y = mdsvalue(char(mdsarg));
        x = mdsvalue('dim_of(_sig)');
        x
        y

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

