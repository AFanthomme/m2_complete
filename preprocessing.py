import numpy as np
import numpy.lib.recfunctions as rcf
import ROOT as r
import pickle
import os
import logging
from sklearn import preprocessing as pr
from root_numpy import root2array, tree2array
from constants import base_features, base_path, production_modes, gen_modes_merged, event_numbers, cross_sections, \
    use_calculated_features, event_categories
from warnings import warn
from misc import frozen

r.gROOT.LoadMacro("libs/cConstants_no_ext.cc")
r.gROOT.LoadMacro("libs/Discriminants_no_ext.cc")


calculated_features = {
'DVBF2j_ME': (r.DVBF2j_ME, ['p_JJVBF_SIG_ghv1_1_JHUGen_JECNominal', 'p_JJQCD_SIG_ghg2_1_JHUGen_JECNominal', 'ZZMass']),
'DVBF1j_ME' : (r.DVBF1j_ME, ['p_JVBF_SIG_ghv1_1_JHUGen_JECNominal', 'pAux_JVBF_SIG_ghv1_1_JHUGen_JECNominal',
                            'p_JQCD_SIG_ghg2_1_JHUGen_JECNominal', 'ZZMass'])}


def read_root_files(directories=('saves/common/', 'saves/common_no_discr/')):
    for directory in directories:
        if not os.path.isdir(directory):
            os.makedirs(directory)
            logging.info('Directory ' + directory + ' created')

        for mode in production_modes:
            rfile = r.TFile(base_path + mode + '125/ZZ4lAnalysis.root')
            tree = rfile.Get('ZZTree/candTree')
            core_variables = base_features

            if mode not in ['WminusH', 'WplusH', 'ZH']:
                data_set = tree2array(tree, branches=core_variables, selection=
                            'ZZsel > 90 && 118 < ZZMass && ZZMass < 130')
                weights = tree2array(tree, branches='overallEventWeight', selection=
                            'ZZsel > 90 && 118 < ZZMass && ZZMass < 130')

                all_calculated_features = calculated_features
                nb_events = np.ma.size(data_set, 0)

                mask = np.ones(nb_events).astype(bool)
                if all_calculated_features:
                    new_features = [np.zeros(nb_events) for _ in range(len(all_calculated_features))]
                    keys = []
                    feature_idx = 0
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
                    weights = weights[mask]

                np.savetxt(directory + mode + '_training.txt', data_set[:nb_events // 2])
                np.savetxt(directory + mode + '_test.txt', data_set[nb_events // 2:])
                np.savetxt(directory + mode + '_weights_training.txt', weights[:nb_events // 2])
                np.savetxt(directory + mode + '_weights_test.txt', weights[nb_events // 2:])
                logging.info(mode + ' weights, training and test sets successfully stored in saves/' + directory)
            else:
                decay_criteria = {'_lept': ' && genExtInfo > 10', '_hadr': ' && genExtInfo < 10'}
                for decay in ['_lept', '_hadr']:
                    data_set = tree2array(tree, branches=core_variables, selection=
                            'ZZsel > 90 && 118 < ZZMass && ZZMass < 130' + decay_criteria[decay])
                    weights = tree2array(tree, branches='overallEventWeight', selection=
                            'ZZsel > 90 && 118 < ZZMass && ZZMass < 130' + decay_criteria[decay])

                    all_calculated_features = calculated_features
                    nb_events = np.ma.size(data_set, 0)

                    mask = np.ones(nb_events).astype(bool)
                    if all_calculated_features:
                        new_features = [np.zeros(nb_events) for _ in range(len(all_calculated_features))]
                        keys = []
                        feature_idx = 0
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
                        logging.warning('At least one of the calculated features was Inf or NaN')
                        data_set = data_set[mask]
                        weights = weights[mask]

                    np.savetxt(directory + mode + decay + '_training.txt', data_set[:nb_events // 2])
                    np.savetxt(directory + mode + decay + '_test.txt', data_set[nb_events // 2:])
                    np.savetxt(directory + mode + decay + '_weights_training.txt', weights[:nb_events // 2])
                    np.savetxt(directory + mode + decay + '_weights_test.txt', weights[nb_events // 2:])
                    logging.info(mode + decay + ' weights, training and test sets successfully stored in saves/' + directory)



def merge_vector_modes(directories=('saves/common/', 'saves/common_no_discr/')):
    for directory in directories:
        for decay in ['_lept', '_hadr']:
            file_list = [directory + mediator + decay for mediator in ['WplusH', 'WminusH', 'ZH']]

            training_set = np.loadtxt(file_list[0] + '_training.txt')
            test_set = np.loadtxt(file_list[0] + '_test.txt')
            weights_train = np.loadtxt(file_list[0] + '_weights_training.txt')
            weights_test = np.loadtxt(file_list[0] + '_weights_test.txt')
            # Rescale the events weights to match a common cross_section / event number equal to the ones of WplusH
            for idx, filename in enumerate(file_list[1:]):
                temp_train = np.loadtxt(filename + '_training.txt')
                temp_test = np.loadtxt(filename + '_test.txt')
                temp_weights_train = np.loadtxt(filename + '_weights_training.txt')
                temp_weights_test = np.loadtxt(filename + '_weights_test.txt')
                temp_weights_train *= event_numbers['WplusH'] / event_numbers[filename.split('/')[-1].split('_')[0]]
                temp_weights_test *= event_numbers['WplusH'] / event_numbers[filename.split('/')[-1].split('_')[0]]

                temp_weights_train *= cross_sections[filename.split('/')[-1].split('_')[0]] / cross_sections['WplusH']
                temp_weights_test *= cross_sections[filename.split('/')[-1].split('_')[0]] / cross_sections['WplusH']

                training_set = np.concatenate((training_set, temp_train), axis=0)
                test_set = np.concatenate((test_set, temp_test), axis=0)
                weights_train = np.concatenate((weights_train, temp_weights_train), axis=0)
                weights_test = np.concatenate((weights_test, temp_weights_test), axis=0)

            np.savetxt(directory + 'VH' + decay + '_training.txt', training_set)
            np.savetxt(directory + 'VH' + decay + '_test.txt', test_set)
            np.savetxt(directory + 'VH' + decay + '_weights_training.txt', weights_train)
            np.savetxt(directory + 'VH' + decay + '_weights_test.txt', weights_test)
    logging.info('Merged data successfully generated')


def prepare_scalers(directories=('saves/common/', 'saves/common_no_discr/')):
    gen_modes_int = event_categories #gen_modes_merged
    for directory in directories:

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
    fit_categories = event_categories #gen_modes_merged
    for directory in ['saves/common/', 'saves/common_no_discr/']:

        with open(directory + 'scaler.txt', 'rb') as f:
            scaler = pickle.load(f)

        file_list = [directory + cat for cat in fit_categories]
        training_set = scaler.transform(np.loadtxt(file_list[0] + '_training.txt'))
        test_set = scaler.transform(np.loadtxt(file_list[0] + '_test.txt'))
        np.savetxt(file_list[0] + '_test_scaled.txt', test_set)
        training_labels = np.zeros(np.ma.size(training_set, 0))
        test_labels = np.zeros(np.ma.size(test_set, 0))
        training_weights = np.loadtxt(file_list[0] + '_weights_training.txt') * \
                  cross_sections[fit_categories[0]] / event_numbers[fit_categories[0]]
        test_weights = np.loadtxt(file_list[0] + '_weights_test.txt') * \
                  cross_sections[fit_categories[0]] / event_numbers[fit_categories[0]]

        for idx, filename in enumerate(file_list[1:]):
            temp_train = scaler.transform(np.loadtxt(filename + '_training.txt'))
            temp_test = scaler.transform(np.loadtxt(filename + '_test.txt'))
            tmp_training_weights = np.loadtxt(filename + '_weights_training.txt') * \
                               cross_sections[filename.split('/')[-1]] / event_numbers[filename.split('/')[-1]]
            tmp_test_weights = np.loadtxt(filename + '_weights_test.txt') * \
                           cross_sections[filename.split('/')[-1]] / event_numbers[filename.split('/')[-1]]
            training_set = np.concatenate((training_set, temp_train), axis=0)
            test_set = np.concatenate((test_set, temp_test), axis=0)
            np.savetxt(filename + '_test_scaled.txt', temp_test)
            np.savetxt(filename + '_test_weights_scaled.txt', tmp_test_weights)
            training_labels = np.concatenate((training_labels, (idx + 1) * np.ones(np.ma.size(temp_train, 0))), axis=0)
            test_labels = np.concatenate((test_labels, (idx + 1) * np.ones(np.ma.size(temp_test, 0))), axis=0)
            training_weights = np.concatenate((training_weights, tmp_training_weights), axis=0)
            test_weights = np.concatenate((test_weights, tmp_test_weights), axis=0)

        np.savetxt(directory + 'full_training_set.txt', training_set)
        np.savetxt(directory + 'full_training_labels.txt', training_labels)
        np.savetxt(directory + 'full_training_weights.txt', training_weights)
        np.savetxt(directory + 'full_test_set.txt', test_set)
        np.savetxt(directory + 'full_test_labels.txt', test_labels)
        np.savetxt(directory + 'full_test_weights.txt', test_weights)

def clean_intermediate_files():
    for directory in ['saves/common/', 'saves/common_no_discr/']:
        files_list = os.listdir(directory)
        for file_name in files_list:
            if file_name.split('_')[0] != 'full':
                os.remove(directory + file_name)



def full_process():
    logging.info('\n Reading root files \n')
    read_root_files()
    logging.info('\n Merging vector modes \n')
    merge_vector_modes()
    logging.info('\n Preparing scalers \n')
    prepare_scalers()
    logging.info('\n Merging and scaling datasets \n')
    make_scaled_datasets()
    # logging.info('\n Removing all intermediate files \n')
    # clean_intermediate_files()

if __name__ == '__main__':
    full_process()


