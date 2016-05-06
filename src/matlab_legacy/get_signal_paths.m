%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is part of the
% Disruption Predictor sigure Developer tools.
% These scripts were developed to utilize
% Support Vector Machines to evaluate diagnostic
% signals as sigures for predicting disruptions
%
% Developer: Matthew Parsons, mparsons@pppl.gov
%
% signal_paths.m - This script is called by stimes.m
% and gives the directory paths for all signals
% 
% 
% Inputs:
% none
% 
% Outputs:
% none
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [sig_path,all_sig_dir,sig_dir] = get_signal_paths(include_sigs,include_machines)

% 1 JET
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
jet_sig_path = '../data/signal_data/jet/jpf/';

% Plasma Current [A]
isig_dir = char('da/c2-ipla/');

% Mode Lock Amplitude [T]
isig_dir = char(isig_dir, 'da/c2-loca/');

% Plasma Internal Inductance [none]
isig_dir = char(isig_dir, 'gs/bl-li<s/');

% Density [m^-2]
isig_dir = char(isig_dir, 'df/g1r-lid:003/');

% Radiated Power [W]
isig_dir = char(isig_dir, 'db/b5r-ptot>out/');

% Total Input Power [W]
isig_dir = char(isig_dir, 'gs/bl-ptot<s/');

% Stored Diamagnetic Energy (time derivative) [W]
isig_dir = char(isig_dir, 'gs/bl-fdwdt<s/');

% Unkown??
isig_dir = char(isig_dir, 'gs/bl-wmhd<s/');

% ECE Profiles [K]
%for i = 1:13 % range [1,13]
%    isig_dir = char(isig_dir, strcat('KK3/P',sprintf('%03d',285+5*i),'/'));
%end

% Mode Lock / Toroidal Field [none]
%isig_dir = char(isig_dir, 'deriv_f/mla_torb0/');

% Greenwald Fraction [none]
%isig_dir = char(isig_dir, 'deriv_f/n_ngw/');

% Radiated Power / Total Input Power [none]
%isig_dir = char(isig_dir, 'deriv_f/prad_pin/');

jet_sig_dir = char('');

for i = include_sigs
    jet_sig_dir = char(jet_sig_dir, isig_dir(i,:));
end

jet_sig_dir(1,:) = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 2 NSTX
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nstx_sig_path = '../data/signal_data/nstx/jpf/';

% Plasma Current [A]
isig_dir = char('engineering/ip1/');

% Mode Lock Amplitude [T]
isig_dir = char(isig_dir, 'operations/rwmef_plas_n1_amp_br/');

% Plasma Internal Inductance [none]
isig_dir = char(isig_dir, 'efit02/li/');

% Density [cm^-2]
isig_dir = char(isig_dir, 'activespec/ts_ld/');

% Radiated Power [MW]
isig_dir = char(isig_dir, 'passivespec/bolom_totpwr/');

% Total Input Power [MW]
isig_dir = char(isig_dir, 'nbi/nb_p_inj/');

% Stored Diamagnetic Energy (time derivative) [W]
isig_dir = char(isig_dir, 'efit02/wpdot/');

nstx_sig_dir = char('');

for i = include_sigs
    nstx_sig_dir = char(nstx_sig_dir, isig_dir(i,:));
end

nstx_sig_dir(1,:) = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





% Collect paths from included machines
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

all_sig_path = char(jet_sig_path, nstx_sig_path);
sig_path = char('');
for i = include_machines
    sig_path = char(sig_path, all_sig_path(i,:));
end
sig_path(1,:) = [];

all_sig_dir = {jet_sig_dir; nstx_sig_dir};
sig_dir = {all_sig_dir{include_machines}};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


end