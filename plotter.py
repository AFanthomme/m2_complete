import matplotlib.pyplot as p
import numpy as np
import pickle
import preprocessing as pr
import categorizer as ctg
import os

from shutil import rmtree
from constants import luminosity, cross_sections, production_modes, gen_modes_merged, event_numbers,\
    models_dict, global_verbosity, prompt_user, add_calculated_features
from copy import copy


def content_plot(model, tags=None, permutation=None, save=True, verbose=global_verbosity):
    """
    Use an instance of a sklearn model (custom ones possible as long as they're contained in a class with correctly 
    named attributes)
    
    :param model: model to study
    :param tolerance: minimal confidence in the result to get tagged
    :param tags: tags for the categories (categorizer predicts 1 => tag[1], etc..)
    :param permutation: to plot the categories in an order different than 0 at bottom, then 1, etc...
    :param save: save the output plot
    :param verbose: print the evolution messages
    :return: None but the plot
    """

    if add_calculated_features:
        suffix = '/'
    else:
        suffix = '_no_discr/'

    directory = model + suffix
    if not os.path.isfile('saves/' + directory + 'ggH_predictions.txt'):
        if verbose:
            print('Generating predictions')
        with open('saves/' + model + suffix + 'categorizer.txt', 'rb') as f:
            categorizer = pickle.load(f)
        for mode in production_modes:
            ctg.generate_predictions('saves/common' + suffix + mode + '_test.txt', mode, categorizer, out_dir=directory)

    nb_categories = max(len(np.unique(np.loadtxt('saves/' + directory + 'ggH_predictions.txt'))), 4)

    if not tags:
        tags_list = ['Category ' + str(i) for i in range(nb_categories)]
    else:
        tags_list = copy(tags)

    contents_table = np.zeros((nb_categories, len(production_modes)))
    ordering = range(nb_categories)

    if permutation:
        ordering = permutation

    for mod_idx, tag in enumerate(production_modes):
        weights_list = np.loadtxt('saves/common' + suffix + tag + '_weights_test.txt')
        predictions_list = np.loadtxt('saves/' + directory + tag + '_predictions.txt')
        if verbose:
            print(tag + 'predictions successfully retrieved')
        for event_idx, prediction in enumerate(predictions_list):
            contents_table[int(prediction), mod_idx] += weights_list[event_idx]

    for idx, gen_mode in enumerate(production_modes):
        contents_table[:, idx] *= cross_sections[gen_mode] * luminosity / event_numbers[gen_mode]

    fig = p.figure()
    p.title('Composition of the different categories for ' + model + ' classification', y=-0.12)
    ax = fig.add_subplot(111)
    color_array = ['b', 'g', 'r', 'brown', 'darkorange', 'm']

    for category in range(nb_categories):
        position = ordering[category]
        normalized_content = contents_table[category, :].astype('float') / np.sum(contents_table[category, :])
        tmp = 0.
        for gen_mode in range(len(production_modes)):
            if position == 1:
                ax.axhspan(position * 0.19 + 0.025, (position + 1) * 0.19 - 0.025, tmp,
                           tmp + normalized_content[gen_mode],
                           color=color_array[gen_mode], label=production_modes[gen_mode])
            else:
                ax.axhspan(position * 0.19 + 0.025, (position + 1) * 0.19 - 0.025, tmp,
                           tmp + normalized_content[gen_mode],
                           color=color_array[gen_mode])
            tmp += normalized_content[gen_mode]
        ax.text(0.01, (position + 0.5) * 0.19 - 0.025, tags_list[position] + ', ' +
                str(np.round(np.sum(contents_table[category, :]), 3)) + ' events', fontsize=15)
    ax.get_yaxis().set_visible(False)
    p.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, fontsize=11, mode="expand", borderaxespad=0.)

    if save:
        p.savefig('figs/tmp/' + model + suffix[:-1] + '.png')
    else:
        p.show()


if __name__ == "__main__":
    suff = ''
    if not add_calculated_features:
        suff = '_no_discr'
    for model_name in models_dict.keys():
        print('\n \n#######################################')
        print('#######################################\n')
        print('Studying model ' + model_name + '\n')
        if not os.path.isdir('figs/tmp'):
            os.makedirs('figs/tmp')

        try:
            open('saves/common' + suff + '/ttH_training.txt')
        except IOError:
            print('Common numpy saves not found, generating them')
            pr.prepare_data()

        try:
            open('saves/' + model_name + suff + '/categorizer.txt', 'rb')
            if prompt_user:
                if raw_input('Do you want to remove the files associated to classifier ' + model_name +
                                 ' and retrain it (y/n)? ') == 'y':
                    rmtree('saves/' + model_name + suff)
            else:
                continue 
            open('saves/' + model_name + suff + '/categorizer.txt', 'rb')
        except IOError:
            print('Training in progress')
            ctg.model_training(model_name)

    content_plot('adaboost_200_logreg_purity', tags=gen_modes_merged)




