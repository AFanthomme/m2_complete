import numpy as np
import pickle
import ROOT as r
import os
from sklearn import preprocessing as pr
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


def frozen(*arg):
    raise AttributeError("This method has been removed")


def prepare_scalers():
    gen_modes_int = gen_modes_merged
    for directory in ['saves/common/', 'saves/common_no_discr/']:

        file_list = [directory + mode for mode in gen_modes_int]
        training_set = np.loadtxt(file_list[0] + '_training.txt')
        test_set = np.loadtxt(file_list[0] + '_test.txt')

        for idx, filename in enumerate(file_list[1:]):
            temp_train = np.loadtxt(filename + '_training.txt')
            temp_test = np.loadtxt(filename + '_test.txt')
            training_set = np.concatenate((training_set, temp_train), axis=0)
            test_set = np.concatenate((test_set, temp_test), axis=0)

        scaler = pr.StandardScaler()
        scaler.fit(training_set)
        scaler.fit = frozen
        scaler.fit_transform = frozen
        scaler.set_params = frozen

        with open(directory + 'scaler.txt', 'wb') as f:
            pickle.dump(scaler, f)


def make_scaled_datasets():
    fit_categories = gen_modes_merged
    for directory in ['saves/common/', 'saves/common_no_discr/']:

        with open(directory + 'scaler.txt', 'rb') as f:
            scaler = pickle.load(f)

        file_list = [directory + cat for cat in fit_categories]
        training_set = scaler.transform(np.loadtxt(file_list[0] + '_training.txt'))
        test_set = scaler.transform(np.loadtxt(file_list[0] + '_test.txt'))
        np.savetxt(file_list[0] + '_test_scaled.txt')
        training_labels = np.zeros(np.ma.size(training_set, 0))
        test_labels = np.zeros(np.ma.size(test_set, 0))

        for idx, filename in enumerate(file_list[1:]):
            temp_train = scaler.transform(np.loadtxt(filename + '_training.txt'))
            temp_test = scaler.transform(np.loadtxt(filename + '_test.txt'))
            training_set = np.concatenate((training_set, temp_train), axis=0)
            test_set = np.concatenate((test_set, temp_test), axis=0)
            np.savetxt(filename + '_test_scaled.txt', temp_test)
            training_labels = np.concatenate((training_labels, (idx + 1) * np.ones(np.ma.size(temp_train, 0))), axis=0)
            test_labels = np.concatenate((test_labels, (idx + 1) * np.ones(np.ma.size(temp_test, 0))), axis=0)

        np.savetxt(directory + 'full_training_set.txt', training_set)
        np.savetxt(directory + 'full_training_labels.txt', training_labels)
        np.savetxt(directory + 'full_test_set.txt', test_set)
        np.savetxt(directory + 'full_test_labels.txt', test_labels)



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
