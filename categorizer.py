import numpy as np
import pickle
import os
from misc import frozen
from constants import models_dict, global_verbosity, use_calculated_features


def model_training(model_name, verbose=global_verbosity):
    analyser = models_dict[model_name]

    if use_calculated_features:
        directory = 'saves/common/'
        suffix = ''
    else:
        directory = 'saves/common_no_discr/'
        suffix = '_no_discr'

    training_set = np.loadtxt(directory + 'full_training_set')
    training_labels = np.loadtxt(directory + 'full_training_labels')

    if model_name.split('_')[-1] == 'invfreq':
        weights = np.loadtxt('full_training_weights' + suffix)
        analyser.fit(training_set, training_labels, 1./weights)
    elif model_name.split('_')[-1] == 'purity':
        custom_weights = np.array([3., 0.5, 0.5, 0.5])
        weights = np.array([custom_weights[int(cat)] for cat in training_labels])
        analyser.fit(training_set, training_labels, weights)
    elif model_name.split('_')[-1] == 'content':
        custom_weights = np.array([1., 3., 2., 0.5])
        weights = np.array([custom_weights[int(cat)] for cat in training_labels])
        analyser.fit(training_set, training_labels, weights)
    else:
        analyser.fit(training_set, training_labels)

    analyser.fit = frozen
    analyser.set_params = frozen

    if not os.path.isdir('saves/' + model_name + suffix):
        os.makedirs('saves/' + model_name + suffix)
    if verbose:
        print('Directory saves/' + model_name + suffix + ' successfully created.')

    with open('saves/' + model_name + suffix + '/categorizer.txt', mode='wb') as file:
        pickle.dump(analyser, file)

    test_set = np.loadtxt(directory + 'full_test_set')
    test_labels = np.loadtxt(directory + 'full_training_labels')
    print('Training score : ' + str(analyser.score(training_set, training_labels)))
    print('Generalization score : ' + str(analyser.score(test_set, test_labels)))


def generate_predictions(model_name, tolerance=0., verbose=global_verbosity):
    if use_calculated_features:
        directory = 'saves/common/'
        suffix = ''
    else:
        directory = 'saves/common_no_discr/'
        suffix = '_no_discr'

    scaled_dataset = np.loadtxt(directory + 'test_set_scaled.txt')

    with open('saves/' + model_name + suffix + '/categorizer.txt', mode='rb') as file:
        classifier = pickle.load(file)

    results = classifier.predict(scaled_dataset)
    nb_categs = max(np.unique(results))

    if tolerance:
        try:
            confidences = np.min(np.abs(classifier.decision_function(scaled_dataset)), 1) < tolerance
            results[confidences] = nb_categs + 1
        except AttributeError:
            pass

    out_path = 'saves/' + model_name + suffix
    np.savetxt(out_path + 'predictions.txt', results)

    if verbose:
        print(out_path + ' predictions successfully stored')

