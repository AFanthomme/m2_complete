import matplotlib.pyplot as p
import numpy as np
import preprocessing as pr
import categorizer as ctg
import logging
import os
from itertools import izip
from constants import models_dict, global_verbosity, use_calculated_features, event_categories, luminosity
from copy import copy


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

    if use_calculated_features:
        suffix = '/'
    else:
        suffix = '_no_discr/'

    directory = 'saves/' + model_name + suffix
    if not os.path.isfile(directory + 'predictions.txt'):
        if verbose:
            print('Generating predictions')
        ctg.generate_predictions(model_name)

    true_categories = np.loadtxt('saves/common' + suffix + 'full_test_labels.txt')
    weights = np.loadtxt('saves/common' + suffix + 'full_test_weights.txt')
    predictions = np.loadtxt(directory + 'predictions.txt')

    nb_categories = 5  # max(len(np.unique(np.loadtxt('saves/' + directory + 'ggH_predictions.txt'))), 5)
    contents_table = np.zeros((nb_categories, len(event_categories)))

    for true_tag, predicted_tag, rescaled_weight in izip(true_categories, predictions, weights):
        contents_table[predicted_tag, true_tag] += rescaled_weight

    contents_table *= luminosity
    ordering = range(nb_categories)
    if permutation:
        ordering = permutation

    fig = p.figure()
    p.title('Content plot for ' + model_name, y=-0.12)
    ax = fig.add_subplot(111)
    color_array = ['b', 'g', 'r', 'brown', 'darkorange', 'm']

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
        ax.text(0.01, (position + 0.5) * 0.19 - 0.025, tags_list[position] + ', ' +
                str(np.round(np.sum(contents_table[category, :]), 3)) + ' events', fontsize=15)
    ax.get_yaxis().set_visible(False)
    p.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, fontsize=11, mode="expand", borderaxespad=0.)

    if save:
        p.savefig('figs/tmp/' + model_name + suffix[:-1] + '.png')
    else:
        p.show()


if __name__ == "__main__":
    suff = ''
    if not use_calculated_features:
        suff = '_no_discr'

    if not os.path.isfile('saves/common_no_discr/full_test_set.txt'):
        pr.full_process()
    
    if not os.path.isdir('figs/tmp'):
        os.makedirs('figs/tmp')
    
    for model_name in models_dict.keys():
        logging.info('Studying model ' + model_name)
           
        try:
            open('saves/' + model_name + suff + '/categorizer.txt', 'rb')
        except IOError:
            logging.info('Training model ' + model_name)
            ctg.model_training(model_name)
        try:
            open('saves/' + model_name + suff + '/predictions.txt', 'rb')
        except IOError:
            logging.info('Generating predictions for ' + model_name + suff)
            ctg.generate_predictions(model_name)
   
        content_plot(model_name)
    logging.info('All models studied')



