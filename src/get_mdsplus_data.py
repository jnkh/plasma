'''
http://www.mdsplus.org/index.php?title=Documentation:Tutorial:RemoteAccess&open=76203664636339686324830207&page=Documentation%2FThe+MDSplus+tutorial%2FRemote+data+access+in+MDSplus
http://piscope.psfc.mit.edu/index.php/MDSplus_%26_python#Simple_example_of_reading_MDSplus_data
http://www.mdsplus.org/documentation/beginners/expressions.shtml
http://www.mdsplus.org/index.php?title=Documentation:Tutorial:MdsObjects&open=76203664636339686324830207&page=Documentation%2FThe+MDSplus+tutorial%2FThe+Object+Oriented+interface+of+MDSPlus
'''

from MDSplus import *

prepath = '/p/datad/jkatesha/data/'
shot_numbers_path = 'shot_lists/'
save_path = 'signal_data1'
mode = 'jet'


if mode == 'NSTX':
	shot_numbers_files = ['disrupt_nstx.txt'] 
	server_path = "skylark.pppl.gov:8501::"
	signal_paths = ['engineering/ip1/',
	'operations/rwmef_plas_n1_amp_br/',
	'efit02/li/',
	'activespec/ts_ld/',
	'passivespec/bolom_totpwr/',
	'nbi/nb_p_inj/',
	'efit02/wpdot/']

elif mode == 'jet':
	shot_numbers_files = ['CWall_clear.txt','CFC_unint.txt','BeWall_clear.txt','ILW_unint.txt'],
	server_path = 'mdsplus.jet.efda.org'

	signal_paths = ['jpf/da/c2-ipla',
	'jpf/da/c2-loca',
	'jpf/db/b5r-ptot>out',
	'jpf/df/g1r-lid:003',
	'jpf/gs/bl-li<s',
	'jpf/gs/bl-fdwdt<s',
	'jpf/gs/bl-ptot<s',
	'jpf/gs/bl-wmhd<s']
else:
	print('unkown machine. exiting')
	exit(1)

shot_numbers,_ = get_multiple_shots_and_disruption_times(shot_numbers_path,shot_numbers_files)

c = Connection(server_path)

for shot_num in shot_numbers:
	for signal_path in signal_paths:
		if mode == 'NSTX':
			tree,tag = get_tree_and_tag(signal_path)
			c.openTree(tree,shot_num)
			data = c.get(tag).data()
			time = c.get('dim_of('+tag+')').data()
		elif mode == 'jet':
			data = c.get('_sig=jet("{}/",{})'.format(signal_path,shot_num)).data()
			time = c.get('_sig=dim_of(jet("{}/",{}))'.format(signal_path,shot_num)).data()
	    data_two_column = vstack((time,data)).transpose()
            savetxt(prepath+save_path + '/' + signal_path,data_two_column,fmt = '%f %f')
	    print('.',end='')
    	print('saved shot {}'.format(shot_num))


def get_tree_and_tag(path):
	spl = path.split('/')
	tree = spl[0]
	tag = '\\' + spl[1]
	return tree,tag



######JET########