import os
from copy import copy
from itertools import izip

import matplotlib.pyplot as p
import numpy as np
import src.trainer as ctg
from src.constants import dir_suff_dict

from src.constants import global_verbosity, features_set_selector, event_categories, luminosity


def content_plot(model_name, permutation=None, save=True, verbose=global_verbosity):
    """
    Use an instance of a sklearn model (custom ones possible as long as they're contained in a class with correctly 
    named attributes)
    
    :param model_name: model to study
    :param tolerance: minimal confidence in the result to get tagged
    :param tags: tags for the categories (categorizer predicts 1 => tag[1], etc..)
    :param permutation: to plot the categories in an order different than 0 at bottom, then 1, etc...
    :param save: save the output plot
    :param verbose: print the evolution messages
    :return: None but the plot
    """
    tags_list = copy(event_categories)

    no_care, suffix = dir_suff_dict[features_set_selector]
    suffix += '/'
    directory = 'saves/' + model_name + suffix
    if not os.path.isfile(directory + 'predictions.txt'):
        if verbose:
            print('Generating predictions')
        ctg.generate_predictions(model_name)

    true_categories = np.loadtxt('saves/common' + suffix + 'full_test_labels.txt')
    weights = np.loadtxt('saves/common' + suffix + 'full_test_weights.txt')
    predictions = np.loadtxt(directory + 'predictions.txt')

    nb_categories = len(event_categories)
    contents_table = np.zeros((nb_categories, nb_categories))

    for true_tag, predicted_tag, rescaled_weight in izip(true_categories, predictions, weights):
        contents_table[predicted_tag, true_tag] += rescaled_weight

    contents_table *= luminosity
    ordering = [nb_categories - 1 - i for i in range(nb_categories)]

    if permutation:
        ordering = permutation

    fig = p.figure()
    p.title('Content plot for ' + model_name, y=-0.12)
    ax = fig.add_subplot(111)
    color_array = ['b', 'g', 'r', 'brown', 'm']

    for category in range(nb_categories):
        position = ordering[category]
        normalized_content = contents_table[category, :].astype('float') / np.sum(contents_table[category, :])
        tmp = 0.
        for gen_mode in range(nb_categories):
            if position == 1:
                ax.axhspan(position * 0.19 + 0.025, (position + 1) * 0.19 - 0.025, tmp,
                           tmp + normalized_content[gen_mode],
                           color=color_array[gen_mode], label=event_categories[gen_mode])
            else:
                ax.axhspan(position * 0.19 + 0.025, (position + 1) * 0.19 - 0.025, tmp,
                           tmp + normalized_content[gen_mode],
                           color=color_array[gen_mode])
            tmp += normalized_content[gen_mode]
        ax.text(0.01, (position + 0.5) * 0.19 - 0.025, tags_list[category] + ', ' +
                str(np.round(np.sum(contents_table[category, :]), 3)) + ' events', fontsize=16, color='w')
    ax.get_yaxis().set_visible(False)
    p.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, fontsize=11, mode="expand", borderaxespad=0.)

    if save:
        p.savefig('figs/tmp/' + model_name + suffix[:-1] + '.png')
    else:
        p.show()





