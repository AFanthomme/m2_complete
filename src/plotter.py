import os
from copy import copy
from itertools import izip

import matplotlib.pyplot as p
import matplotlib.cm as cm
import numpy as np
import src.trainer as ctg
import src.constants as cst


def content_plot(model_name, permutation=None, save=True, verbose=cst.global_verbosity):
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
    tags_list = copy(cst.event_categories)

    no_care, suffix = cst.dir_suff_dict[cst.features_set_selector]
    suffix += '/'
    directory = 'saves_alt/' + model_name + suffix
    if not os.path.isfile(directory + 'predictions.prd'):
        if verbose:
            print('Generating predictions')
        ctg.generate_predictions(model_name)

    true_categories = np.loadtxt('saves_alt/common' + suffix + 'full_test_labels.lbl')
    weights = np.loadtxt('saves_alt/common' + suffix + 'full_test_weights.wgt')
    predictions = np.loadtxt(directory + 'predictions.prd')



    nb_categories = len(cst.event_categories)
    contents_table = np.zeros((nb_categories, nb_categories))

    for true_tag, predicted_tag, rescaled_weight in izip(true_categories, predictions, weights):
        contents_table[predicted_tag, true_tag] += rescaled_weight

    contents_table *= cst.luminosity
    correct = [contents_table[cat, cat] for cat in range(nb_categories)]
    incorrect = np.sum(contents_table[np.logical_not(np.identity(nb_categories, dtype=bool))], axis=1)

    bkg_predictions = np.loadtxt(directory + 'bkg_predictions.prd')
    bkg_weights = np.loadtxt('saves_alt/common' + suffix + 'ZZTo4l_weights.wgt')
    bkg_weights *= cst.cross_sections['ZZTo4l'] / cst.event_numbers['ZZTo4l']
    bkg = np.array([np.sum(bkg_weights[np.where(bkg_predictions == cat)]) for cat in range(nb_categories)])

    strengths = [1. / (1. + (bkg[cat] + incorrect[cat]) / correct[cat]) for cat in range(nb_categories)]
    np.savetxt(directory + 'strengths.txt')

    ordering = [nb_categories - 1 - i for i in range(nb_categories)]
    if permutation:
        ordering = permutation

    fig = p.figure()
    p.title('Content plot for ' + model_name, y=-0.12)
    ax = fig.add_subplot(111)
    # color_array = cm.rainbow(np.linspace(0, 1, nb_categories))
    color_array = ['b', 'g', 'r', 'brown', 'm', '0.75', 'c', 'b']

    for category in range(nb_categories):
        position = ordering[category]
        normalized_content = contents_table[category, :].astype('float') / np.sum(contents_table[category, :])
        tmp = 0.
        for gen_mode in range(nb_categories):
            if position == 1:
                ax.axhspan(position * 0.19 + 0.025, (position + 1) * 0.19 - 0.025, tmp,
                           tmp + normalized_content[gen_mode],
                           color=color_array[gen_mode], label=cst.event_categories[gen_mode])
            else:
                ax.axhspan(position * 0.19 + 0.025, (position + 1) * 0.19 - 0.025, tmp,
                           tmp + normalized_content[gen_mode],
                           color=color_array[gen_mode])
            tmp += normalized_content[gen_mode]
        ax.text(0.01, (position + 0.5) * 0.19 - 0.025, tags_list[category] + ', ' +
                str(np.round(np.sum(contents_table[category, :]), 2)) + r' events; $\mu = $' +
                str(np.round(strengths[category], 3)), fontsize=16, color='w')

    ax.get_yaxis().set_visible(False)
    p.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, fontsize=11, mode="expand", borderaxespad=0.)

    if save:
        if not os.path.isdir('saves_alt/figs'):
            os.makedirs('saves_alt/figs')
        p.savefig('saves_alt/figs/' + model_name + suffix[:-1] + '.png')
    else:
        p.show()


# def content_plot(model_name, permutation=None, save=True, verbose=cst.global_verbosity):
#     """
#     Use an instance of a sklearn model (custom ones possible as long as they're contained in a class with correctly
#     named attributes)
#
#     :param model_name: model to study
#     :param tolerance: minimal confidence in the result to get tagged
#     :param tags: tags for the categories (categorizer predicts 1 => tag[1], etc..)
#     :param permutation: to plot the categories in an order different than 0 at bottom, then 1, etc...
#     :param save: save the output plot
#     :param verbose: print the evolution messages
#     :return: None but the plot
#     """
#     tags_list = copy(cst.event_categories)
#
#     no_care, suffix = cst.dir_suff_dict[cst.features_set_selector]
#     suffix += '/'
#     directory = 'saves_alt/' + model_name + suffix
#     if not os.path.isfile(directory + 'predictions.prd'):
#         if verbose:
#             print('Generating predictions')
#         ctg.generate_predictions(model_name)
#
#     true_categories = np.loadtxt('saves_alt/common' + suffix + 'full_test_labels.lbl')
#     weights = np.loadtxt('saves_alt/common' + suffix + 'full_test_weights.wgt')
#     predictions = np.loadtxt(directory + 'predictions.prd')
#
#     nb_categories = len(cst.event_categories)
#     contents_table = np.zeros((nb_categories, nb_categories))
#
#     for true_tag, predicted_tag, rescaled_weight in izip(true_categories, predictions, weights):
#         contents_table[predicted_tag, true_tag] += rescaled_weight
#
#     contents_table *= cst.luminosity
#     ordering = [nb_categories - 1 - i for i in range(nb_categories)]
#
#     if permutation:
#         ordering = permutation
#
#     fig = p.figure()
#     p.title('Content plot for ' + model_name, y=-0.12)
#     ax = fig.add_subplot(111)
#     # color_array = cm.rainbow(np.linspace(0, 1, nb_categories))
#     color_array = ['b', 'g', 'r', 'brown', 'm', '0.75', 'c', 'b']
#
#     for category in range(nb_categories):
#         position = ordering[category]
#         normalized_content = contents_table[category, :].astype('float') / np.sum(contents_table[category, :])
#         tmp = 0.
#         for gen_mode in range(nb_categories):
#             if position == 1:
#                 ax.axhspan(position * 0.19 + 0.025, (position + 1) * 0.19 - 0.025, tmp,
#                            tmp + normalized_content[gen_mode],
#                            color=color_array[gen_mode], label=cst.event_categories[gen_mode])
#             else:
#                 ax.axhspan(position * 0.19 + 0.025, (position + 1) * 0.19 - 0.025, tmp,
#                            tmp + normalized_content[gen_mode],
#                            color=color_array[gen_mode])
#             tmp += normalized_content[gen_mode]
#         ax.text(0.01, (position + 0.5) * 0.19 - 0.025, tags_list[category] + ', ' +
#                 str(np.round(np.sum(contents_table[category, :]), 3)) + ' events', fontsize=16, color='w')
#     ax.get_yaxis().set_visible(False)
#     p.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, fontsize=11, mode="expand", borderaxespad=0.)
#
#     if save:
#         if not os.path.isdir('saves_alt/figs'):
#             os.makedirs('saves_alt/figs')
#         p.savefig('saves_alt/figs/' + model_name + suffix[:-1] + '.png')
#     else:
#         p.show()


def search_discrimination(model_name, mode=1, verbose=cst.global_verbosity):
    tags_list = copy(cst.event_categories)

    no_care, suffix = cst.dir_suff_dict[cst.features_set_selector]
    suffix += '/'
    directory = 'saves_alt/' + model_name + suffix
    if not os.path.isfile(directory + 'predictions.prd'):
        if verbose:
            print('Generating predictions')
        ctg.generate_predictions(model_name)

    test_set = np.loadtxt('saves_alt/common' + suffix + 'full_test_set.dst')
    true_categories = np.loadtxt('saves_alt/common' + suffix + 'full_test_labels.lbl')
    weights = np.loadtxt('saves_alt/common' + suffix + 'full_test_weights.wgt')

    predictions = np.loadtxt(directory + 'predictions.txt')

    nb_categories = len(cst.event_categories)

    for idx, true_cat, predicted_cat, rescaled_weight in enumerate(izip(true_categories, predictions, weights)):
        right_indices = []
        wrong_indices = []
        if predicted_cat == mode:
            if true_cat == mode:
                right_indices.append(idx)
            else:
                wrong_indices.append(idx)

    discriminants_list = []
    colors = ['g', 'r']
    labels = ['Correct', 'Incorrect']

    for discriminant in discriminants_list:
        my_list = [test_set[discriminant][idx_list] for idx_list in [right_indices, wrong_indices]]
        p.hist(my_list, 50, stacked=True, histtype='bar', color=colors, label=labels)
        p.title('Distribution of ' + discriminant + ' among events classified as ' + cst.event_categories[mode])
        p.savefig('saves_alt/hists/' + model_name + '_' + discriminant + '_' + suffix[:-1] + '.png')





