import numpy as np
import pickle
import os
from constants import models_dict, production_modes, gen_modes_merged, cross_sections, event_numbers, global_verbosity, \
    add_calculated_features


def model_training(model_name, use_merged_modes=True, verbose=global_verbosity):
    gen_modes_int = production_modes
    analyser = models_dict[model_name]

    if use_merged_modes:
        gen_modes_int = gen_modes_merged

    if add_calculated_features:
        directory = 'saves/common/'
        suffix = ''
    else:
        directory = 'saves/common_no_discr/'
        suffix = '_no_discr'

    training_set = np.loadtxt('dataset_train' + suffix)
    training_labels = np.loadtxt('labels_train' + suffix)
    weights = np.loadtxt('weights_train' + suffix)

    if model_name.split('_')[-1] == 'invfreq':
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
    filename = open('saves/' + model_name + suffix + '/categorizer.txt', mode='wb')
    pickle.dump(analyser, filename)
    filename.close()

    test_labels = np.loadtxt(directory + 'labels_test' + suffix)
    test_set = np.loadtxt(directory + 'dataset_test' + suffix)
    print('Training score : ' + str(analyser.score(training_set, training_labels)))
    print('Generalization score : ' + str(analyser.score(test_set, test_labels)))


def generate_predictions(filename, mode, classifier, out_dir='tmp/', tolerance=0., verbose=global_verbosity):
    if add_calculated_features:
        directory = 'saves/common/'
        suffix = ''
    else:
        directory = 'saves/common_no_discr/'
        suffix = '_no_discr'

    scaled_dataset = np.loadtxt(filename.split('.')[0] + 'scaled.txt')
    results = classifier.predict(scaled_dataset)
    nb_categs = max(np.unique(results))

    if tolerance:
        try:
            confidences = np.min(np.abs(classifier.decision_function(scaled_dataset)), 1) < tolerance
            results[confidences] = nb_categs + 1
        except AttributeError:
            pass

    out_path = 'saves/' + out_dir
    np.savetxt(out_path + mode + '_predictions.txt', results)

    if verbose:
        print(out_path + mode + ' predictions successfully stored')

