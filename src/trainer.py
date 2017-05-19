import logging
import os
import pickle
import numpy as np
import src.constants as cst
from src.misc import frozen
from copy import deepcopy as copy

def model_training(model_name, verbose=cst.global_verbosity):
    models_dict = copy(cst.models_dict)
    analyser, model_weights = models_dict[model_name]

    directory, suffix = cst.dir_suff_dict[cst.features_set_selector]

    training_set = np.loadtxt(directory + 'full_training_set.dst')
    training_labels = np.loadtxt(directory + 'full_training_labels.lbl')

    debug = False
    if model_weights:
        weights = np.array([model_weights[int(cat)] for cat in training_labels])
        analyser.fit(training_set, training_labels, weights)
    elif model_name.split('_')[-1] == 'invfreq':
        weights = np.loadtxt(directory + 'full_training_weights.wgt')
        analyser.fit(training_set, training_labels, 1./weights)
    elif model_name.split('_')[-1] == 'purity':
        custom_weights = np.array([3., 0.5, 0.5,  0.5, 0.5, 0.5])
        weights = np.array([custom_weights[int(cat)] for cat in training_labels])
        analyser.fit(training_set, training_labels, weights)
    elif model_name.split('_')[-1] == 'content':
        custom_weights = np.array([1., 3., 0.5,  2., 0.5, 0.5])
        weights = np.array([custom_weights[int(cat)] for cat in training_labels])
        analyser.fit(training_set, training_labels, weights)
    else:
        analyser.fit(training_set, training_labels)
    if not debug:
        try:
            analyser.explore_thresholds()
            analyser.explore_history()
        except AttributeError:
            pass


    analyser.fit = frozen
    analyser.set_params = frozen

    if not os.path.isdir('saves_alt/' + model_name + suffix):
        os.makedirs('saves_alt/' + model_name + suffix)
    if verbose:
        print('Directory saves/' + model_name + suffix + ' successfully created.')

    with open('saves_alt/' + model_name + suffix + '/categorizer.pkl', mode='wb') as file:
        pickle.dump(analyser, file)

    # test_set = np.loadtxt(directory + 'full_test_set.dst')
    # test_labels = np.loadtxt(directory + 'full_test_labels.txt')
    # logging.info('Training score : ' + str(analyser.score(training_set, training_labels)))
    # logging.info('Generalization score : ' + str(analyser.score(test_set, test_labels)))


def generate_predictions(model_name, verbose=cst.global_verbosity, tolerance =0.):
    directory, suffix = cst.dir_suff_dict[cst.features_set_selector]
    scaled_dataset = np.loadtxt(directory + 'full_test_set.dst')
    background_dataset = np.loadtxt(directory + 'ZZTo4l.dst')
    with open('saves_alt/' + model_name + suffix + '/categorizer.pkl', mode='rb') as file:
        classifier = pickle.load(file)

    results = classifier.predict(scaled_dataset)
    probas = classifier.predict_proba(scaled_dataset)
    bkg_results = classifier.predict(background_dataset)

    # if tolerance:
    #     try:
    #         confidences = np.min(np.abs(classifier.decision_function(scaled_dataset)), 1) < tolerance
    #         results[confidences] = len(np.unique(results)) + 1
    #     except AttributeError:
    #         pass

    out_path = 'saves_alt/' + model_name + suffix
    np.savetxt(out_path + '/predictions.prd', results)
    np.savetxt(out_path + '/probas.prb', probas)
    np.savetxt(out_path + '/bkg_predictions.prd', bkg_results)

    if verbose:
        print(out_path + ' predictions successfully stored')

