import numpy as np
from sklearn import preprocessing as pr
import pickle
import os
from constants import models_dict, gen_modes, gen_modes_merged, cross_sections, event_numbers, global_verbosity, \
    add_calculated_features


def frozen(*arg):
    raise AttributeError("This method has been removed")


def model_training(model_name, use_merged_modes=True, verbose=global_verbosity):
    gen_modes_int = gen_modes

    if use_merged_modes:
        gen_modes_int = gen_modes_merged

    if add_calculated_features:
        directory = 'saves/common/'
        suffix = ''
    else:
        directory = 'saves/common_no_discr/'
        suffix = '_no_discr'

    file_list = [directory + mode for mode in gen_modes_int]
    training_set = np.loadtxt(file_list[0] + '_training.txt')
    test_set = np.loadtxt(file_list[0] + '_test.txt')
    training_labels = np.zeros(np.ma.size(training_set, 0))
    test_labels = np.zeros(np.ma.size(test_set, 0))
    weights = np.loadtxt(file_list[0] + '_weights_training.txt') * \
              cross_sections[gen_modes_int[0]] / event_numbers[gen_modes_int[0]]

    for idx, filename in enumerate(file_list[1:]):
        temp_train = np.loadtxt(filename + '_training.txt')
        temp_test = np.loadtxt(filename + '_test.txt')
        temp_weights = np.loadtxt(filename + '_weights_training.txt') * \
                       cross_sections[gen_modes_int[idx]] / event_numbers[gen_modes_int[idx]]
        training_set = np.concatenate((training_set, temp_train), axis=0)
        test_set = np.concatenate((test_set, temp_test), axis=0)
        training_labels = np.concatenate((training_labels, (idx + 1) * np.ones(np.ma.size(temp_train, 0))), axis=0)
        test_labels = np.concatenate((test_labels, (idx + 1) * np.ones(np.ma.size(temp_test, 0))), axis=0)
        weights = np.concatenate((weights, temp_weights), axis=0)
    try:
        open(directory + 'scaler.txt', mode='rb')
    except IOError:
        scaler = pr.StandardScaler()
        scaler.fit(training_set)
        scaler.fit = frozen
        scaler.fit_transform = frozen
        scaler.set_params = frozen
        scaler_file = open(directory + 'scaler.txt', mode='wb')
        pickle.dump(scaler, scaler_file)
        scaler_file.close()
    
    with open(directory + 'scaler.txt', 'rb') as f:
        scaler = pickle.load(f)

    scaler.transform(training_set)
    scaler.transform(test_set)
    analyser = models_dict[model_name]

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

    print('Training score : ' + str(analyser.score(training_set, training_labels)))
    print('Generalization score : ' + str(analyser.score(test_set, test_labels)))


def generate_predictions(filename, mode, classifier, out_dir='tmp/', tolerance=0., verbose=global_verbosity):
    if add_calculated_features:
        directory = 'saves/common/'
        suffix = ''
    else:
        directory = 'saves/common_no_discr/'
        suffix = '_no_discr'

    scaler_file = open(directory + 'scaler.txt', mode='rb')
    scaler = pickle.load(scaler_file)
    unscaled_features = np.loadtxt(filename)
    scaled_dataset = scaler.transform(unscaled_features)
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

