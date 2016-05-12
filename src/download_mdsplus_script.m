
%save_path = '/p/datad/jkatesha/data/signal_data/jet/';
save_path = '../data/signal_data/jet/';

% %location of jet signals 
signals_dirs = {'jpf/da/c2-ipla','jpf/da/c2-loca','jpf/db/b5r-ptot>out','jpf/df/g1r-lid:003','jpf/gs/bl-li<s','jpf/gs/bl-fdwdt<s','jpf/gs/bl-ptot<s','jpf/gs/bl-wmhd<s'}
% 
% Which shots to choose:
%shots_path = '/p/datad/jkatesha/data/shot_lists/all_shots.txt';
shots_path = '../data/shot_lists/short_list.txt';

download_jet_data(shots_path,save_path,signals_dirs);