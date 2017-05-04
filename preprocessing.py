import numpy as np
import pickle
import ROOT as r
import os
from root_numpy import root2array, tree2array
from constants import base_features, base_path, gen_modes, gen_modes_merged, event_numbers, cross_sections, global_verbosity, \
    add_calculated_features
from warnings import warn
import numpy.lib.recfunctions as rcf
r.gROOT.LoadMacro("libs/cConstants_no_ext.cc")
r.gROOT.LoadMacro("libs/Discriminants_no_ext.cc")


calculated_features = {
'DVBF2j_ME': (r.DVBF2j_ME, ['p_JJVBF_SIG_ghv1_1_JHUGen_JECNominal', 'p_JJQCD_SIG_ghg2_1_JHUGen_JECNominal', 'ZZMass']),
'DVBF1j_ME' : (r.DVBF1j_ME, ['p_JVBF_SIG_ghv1_1_JHUGen_JECNominal', 'pAux_JVBF_SIG_ghv1_1_JHUGen_JECNominal',
                            'p_JQCD_SIG_ghg2_1_JHUGen_JECNominal', 'ZZMass'])}


def check_number_events():
    tmp = {'ggH': 0, 'VBFH': 0, 'WminusH': 0, 'WplusH': 0, 'ZH': 0, 'ttH': 0}
    for idx, tag in enumerate(gen_modes):
        rfile = r.TFile(base_path + tag + '125/ZZ4lAnalysis.root')
        tmp[tag] = rfile.Get('ZZTree/Counters').GetBinContent(40)
    if not tmp == event_numbers:
        print("Event numbers don't match, please modify constants.py")
        print(tmp)
    else:
        print("Event numbers match, nothing to change")


def append_field(structured_array, descr):
    if structured_array.dtype.fields is None:
        raise ValueError("`A' must be a structured numpy array")
    b = np.empty(structured_array.shape, dtype=structured_array.dtype.descr + descr)
    for name in structured_array.dtype.names:
        b[name] = structured_array[name]
    return b


def expand(array_of_tuples_1d):
    nb_columns = len(array_of_tuples_1d[0])
    nb_rows = np.ma.size(array_of_tuples_1d, 0)
    tmp = np.zeros((nb_rows, nb_columns))
    for i in range(nb_rows):
        for j in range(nb_columns):
            tmp[i, j] = array_of_tuples_1d[i][j]
    return tmp



def merge_data(directory):
    directory = 'saves/' + directory
    file_list = [directory + mode for mode in ['WplusH', 'WminusH', 'ZH']]

    training_set = np.loadtxt(file_list[0] + '_training.txt')
    test_set = np.loadtxt(file_list[0] + '_test.txt')
    weights_train = np.loadtxt(file_list[0] + '_weights_training.txt')
    weights_test = np.loadtxt(file_list[0] + '_weights_test.txt')
    # Rescale the events weights to match a common cross_section / event number equal to the one of WplusH
    for idx, filename in enumerate(file_list[1:]):
        temp_train = np.loadtxt(filename + '_training.txt')
        temp_test = np.loadtxt(filename + '_test.txt')
        temp_weights_train = np.loadtxt(filename + '_weights_training.txt')
        temp_weights_test = np.loadtxt(filename + '_weights_test.txt')
        temp_weights_train *= event_numbers['WplusH'] / event_numbers[filename.split('/')[-1]]
        temp_weights_test *= event_numbers['WplusH'] / event_numbers[filename.split('/')[-1]]

        temp_weights_train *= cross_sections[filename.split('/')[-1]] / cross_sections['WplusH']
        temp_weights_test *= cross_sections[filename.split('/')[-1]] / cross_sections['WplusH']

        training_set = np.concatenate((training_set, temp_train), axis=0)
        test_set = np.concatenate((test_set, temp_test), axis=0)
        weights_train = np.concatenate((weights_train, temp_weights_train), axis=0)
        weights_test = np.concatenate((weights_test, temp_weights_test), axis=0)

    np.savetxt(directory + 'VH_training.txt', training_set)
    np.savetxt(directory + 'VH_test.txt', test_set)
    np.savetxt(directory + 'VH_weights_training.txt', weights_train)
    np.savetxt(directory + 'VH_weights_test.txt', weights_test)
    print('Merged data successfully generated')


def prepare_data(directory='common/', additional_variables=None):
    if not add_calculated_features:
        directory = 'common_no_discr/'

    if not os.path.isdir('saves/' + directory):
        os.makedirs('saves/' + directory)
        print('Directory saves/' + directory + ' created')

    for mode in gen_modes:
        rfile = r.TFile(base_path + mode + '125/ZZ4lAnalysis.root')
        tree = rfile.Get('ZZTree/candTree')
        core_variables = base_features

        data_set = tree2array(tree, branches=core_variables, selection='ZZsel > 90 && 118 < ZZMass && ZZMass < 130')
        weight = tree2array(tree, branches='overallEventWeight', selection='ZZsel > 90 && 118 < ZZMass && ZZMass < 130')

        all_calculated_features = calculated_features
        nb_events = np.ma.size(data_set, 0)

        # Assume all additional variables are calculated ones, could probably work with some kind of root.get
        if additional_variables:
            all_calculated_features.update(additional_variables)
        if not add_calculated_features:
            all_calculated_features = None
        
        mask = True
        if all_calculated_features:
		new_features = [np.zeros(nb_events) for _ in range(len(all_calculated_features))]
		keys = []
		feature_idx = 0
		mask = np.ones(nb_events).astype(bool)
		for key, couple in all_calculated_features.iteritems():
		    keys.append(key)
		    plop = new_features[feature_idx]
		    feature_expression, vars_list = couple
		    for event_idx in range(nb_events):
			tmp = feature_expression(*data_set[vars_list][event_idx])
			if np.isnan(tmp) or np.isinf(tmp) or np.isneginf(tmp):
			    mask[event_idx] = False
			plop[event_idx] = tmp
		    new_features[feature_idx] = plop    
		    feature_idx += 1
		data_set = rcf.rec_append_fields(data_set, keys, new_features)
        
        if not np.all(mask):
            warn('At least one of the calculated features was Inf or NaN')
            data_set = data_set[mask]
            weight = weight[mask]

        np.savetxt('saves/' + directory + mode + '_training.txt', data_set[:nb_events // 2])
        np.savetxt('saves/' + directory + mode + '_test.txt', data_set[nb_events // 2:])
        np.savetxt('saves/' + directory + mode + '_weights_training.txt', weight[:nb_events // 2])
        np.savetxt('saves/' + directory + mode + '_weights_test.txt', weight[nb_events // 2:])
        print(mode + ' weights, training and test sets successfully stored in saves/' + directory)
    merge_data(directory)
