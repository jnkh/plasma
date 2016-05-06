%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is part of the
% Disruption Predictor Feature Developer tools.
% These scripts were developed to utilize
% Support Vector Machines to evaluate diagnostic
% signals as features for predicting disruptions
%
% Developer: Matthew Parsons, mparsons@pppl.gov
%
% feature_paths.m - This script is called by setup.m
% and gives the directory paths for all features
% 
% 
% Inputs:
% none
% 
% Outputs:
% none
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





% 1 JET
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
jet_feat_path = '/p/datad/signal_data/jet/ppf/';

% Plasma Current
ifeat_dir = char('da/c2_ipla/mean/');
ifeat_dir = char(ifeat_dir, 'da/c2_ipla/stddev/');

% Mode Lock Amplitude
ifeat_dir = char(ifeat_dir, 'da/c2_loca/mean/');
ifeat_dir = char(ifeat_dir, 'da/c2_loca/stddev/');

% Plasma Internal Inductance
ifeat_dir = char(ifeat_dir, 'gs/bl_li_s/mean/');
ifeat_dir = char(ifeat_dir, 'gs/bl_li_s/stddev/');

% Density
ifeat_dir = char(ifeat_dir, 'df/g1r_lid_003/mean/');
ifeat_dir = char(ifeat_dir, 'df/g1r_lid_003/stddev/');

% Radiated Power
ifeat_dir = char(ifeat_dir, 'db/b5r_ptot_out/mean/');
ifeat_dir = char(ifeat_dir, 'db/b5r_ptot_out/stddev/');

% Total Input Power
ifeat_dir = char(ifeat_dir, 'gs/bl_ptot_s/mean/');
ifeat_dir = char(ifeat_dir, 'gs/bl_ptot_s/stddev/');

% Stored Diamagnetic Energy (time derivative)
ifeat_dir = char(ifeat_dir, 'gs/bl_fdwdt_s/mean/');
ifeat_dir = char(ifeat_dir, 'gs/bl_fdwdt_s/stddev/');

% ECE Profiles
%for i = 1:13 % range [1,13]
%    ifeat_dir = char(ifeat_dir, strcat('KK3/P',sprintf('%03d',285+5*i),'/mean/'));
%    ifeat_dir = char(ifeat_dir, strcat('KK3/P',sprintf('%03d',285+5*i),'/stddev/'));
%end

% ECE Channel
%ifeat_dir = char(ifeat_dir, 'de/k3_b3_c03/mean/');
%ifeat_dir = char(ifeat_dir, 'de/k3_b3_c03/stddev/');

% Mode Lock / Toroidal Field
%ifeat_dir = char(ifeat_dir, 'deriv_f/mla_torb0/mean/');
%ifeat_dir = char(ifeat_dir, 'deriv_f/mla_torb0/stddev/');

% Greenwald Fraction
%ifeat_dir = char(ifeat_dir, 'deriv_f/n_ngw/mean/');
%ifeat_dir = char(ifeat_dir, 'deriv_f/n_ngw/stddev/');

% Radiated Power / Total Input Power
%ifeat_dir = char(ifeat_dir, 'deriv_f/prad_pin/mean/');
%ifeat_dir = char(ifeat_dir, 'deriv_f/prad_pin/stddev/');

jet_feat_dir = char('');

for i = include_feats
    jet_feat_dir = char(jet_feat_dir, ifeat_dir(i,:));
end

jet_feat_dir(1,:) = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





% 2 NSTX
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nstx_feat_path = '/p/datad/signal_data/nstx/ppf/';

% Plasma Current
ifeat_dir = char('engineering/ip1/mean/');
ifeat_dir = char(ifeat_dir, 'engineering/ip1/stddev/');

% Mode Lock Amplitude
ifeat_dir = char(ifeat_dir, 'operations/rwmef_plas_n1_amp_br/mean/');
ifeat_dir = char(ifeat_dir, 'operations/rwmef_plas_n1_amp_br/stddev/');

% Plasma Internal Inductance
ifeat_dir = char(ifeat_dir, 'efit02/li/mean/');
ifeat_dir = char(ifeat_dir, 'efit02/li/stddev/');

% Density
ifeat_dir = char(ifeat_dir, 'activespec/ts_ld/mean/');
ifeat_dir = char(ifeat_dir, 'activespec/ts_ld/stddev/');

% Radiated Power
ifeat_dir = char(ifeat_dir, 'passivespec/bolom_totpwr/mean/');
ifeat_dir = char(ifeat_dir, 'passivespec/bolom_totpwr/stddev/');

% Total Input Power
ifeat_dir = char(ifeat_dir, 'nbi/nb_p_inj/mean/');
ifeat_dir = char(ifeat_dir, 'nbi/nb_p_inj/stddev/');

% Stored Diamagnetic Energy (time derivative)
ifeat_dir = char(ifeat_dir, 'efit02/wpdot/mean/');
ifeat_dir = char(ifeat_dir, 'efit02/wpdot/stddev/');

nstx_feat_dir = char('');

for i = include_feats
    nstx_feat_dir = char(nstx_feat_dir, ifeat_dir(i,:));
end

nstx_feat_dir(1,:) = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





% Collect paths from included machines
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

all_feat_path = char(jet_feat_path, nstx_feat_path);
feat_path = char('');
for i = include_machines
    feat_path = char(feat_path, all_feat_path(i,:));
end
feat_path(1,:) = [];

all_feat_dir = {jet_feat_dir; nstx_feat_dir};
feat_dir = {all_feat_dir{include_machines}};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






