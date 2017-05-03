import os
import pickle
import ROOT as r
import numpy as np
import matplotlib.pyplot as p
from root_numpy import root2array, tree2array
from constants import base_path, gen_modes, base_features, cross_sections, event_numbers

luminosity = 35.9
r.gROOT.LoadMacro("libs/cConstants_no_ext.cc")
r.gROOT.LoadMacro("libs/Discriminants_no_ext.cc")
r.gROOT.LoadMacro("libs/Category_no_ext.cc")


def convert_array(np_array_of_arrays):
    plop = []
    for entry in np_array_of_arrays:
        if 'ndarray' in str(type(entry)):
            tmp = []
            for sub_entry in entry:
                tmp.append(sub_entry.item())
            plop.append(tuple(tmp))
        else:
            plop.append(entry.item())
    return tuple(plop)


def generate_predictions_legacy(features_list, production_tags, verbose=True):
    if not os.path.isdir('saves/legacy'):
        os.makedirs('saves/legacy')
        print('Directory saves/legacy successfully created')
    for idx, tag in enumerate(production_tags):
        rfile = r.TFile(base_path + tag + '125/ZZ4lAnalysis.root')
        tree = rfile.Get('ZZTree/candTree')
        data_set = tree2array(tree, branches=features_list, selection='ZZsel > 90 && 118 < ZZMass && ZZMass < 130')

        # Get the weights of events corresponding to the MC data in our file
        weights = tree2array(tree, branches='overallEventWeight', selection='ZZsel > 90 && 118 < ZZMass && ZZMass < 130')

        predictions = 6 * np.ones((len(data_set), 2))
        for event_idx, event in enumerate(data_set):
            temp = convert_array(data_set[features_list][event_idx]) + (True,)
            predictions[event_idx, :] = [r.categoryMor17(*temp), weights[event_idx]]
        np.savetxt('saves/legacy/' + tag + '_predictions.txt', predictions)

        if verbose:
            print('saves/legacy/' + tag + '.txt successfully stored')


def content_plot_legacy(verbose=False):
    """
    Each line is a line in the final plot, each column represents a production mode
    The last category is the "untagged" one
    """
    contents_table = np.zeros((7, 6))

    # This defines the position in the graph of the tag (and associated data)
    category_positions = [6, 5, 4, 2, 3, 0, 1]

    # These are the tags that will be written on the plot after reordering
    ordered_tags_list = ['ttH', 'VHMET', 'VHLept', 'VHHadr', 'VBF2j', 'VBF1j', 'Untagged']

    #if raw_input('Generate predictions (y/n)? ') == 'y':
    #    pr.generate_predictions_legacy(base_features, gen_modes)

    #with open('saves/legacy/event_numbers.txt', 'rb') as handle:
    #    event_numbers = pickle.load(handle)

    for mod_idx, tag in enumerate(gen_modes):
        predictions = np.loadtxt('saves/legacy/' + tag + '_predictions.txt')
        if verbose:
            print('saves/legacy/' + tag + '_predictions.txt successfully retrieved')
        for pred_idx in range(np.ma.size(predictions, 0)):
            contents_table[int(predictions[pred_idx, 0]), mod_idx] += predictions[pred_idx, 1]

    # Normalize each production mode (column) : divide by total number of events (not only in the selected region)
    # and multiply by the expected one which is the product of the luminosity by the cross section
    for idx, gen_mode in enumerate(gen_modes):
        contents_table[:, idx] *= cross_sections[gen_mode] * luminosity / event_numbers[gen_mode]

    fig = p.figure()
    p.title('Composition of the different categories for legacy classification', y=-0.12)
    ax = fig.add_subplot(111)
    color_array = ['b', 'g', 'r', 'brown', 'darkorange', 'm']

    for category in range(7):
        position = category_positions[category]
        normalized_content = contents_table[category, :].astype('float') / np.sum(contents_table[category, :])
        tmp = 0.
        for gen_mode in range(6):
            if position == 1:
                ax.axhspan(position * 0.19 + 0.025, (position + 1) * 0.19 - 0.025, tmp,
                           tmp + normalized_content[gen_mode],
                           color=color_array[gen_mode], label=gen_modes[gen_mode])
            else:
                ax.axhspan(position * 0.19 + 0.025, (position + 1) * 0.19 - 0.025, tmp,
                           tmp + normalized_content[gen_mode],
                           color=color_array[gen_mode])
            tmp += normalized_content[gen_mode]
        ax.text(0.01, (position + 0.5) * 0.19 - 0.025, ordered_tags_list[position] + ', ' +
                str(np.round(np.sum(contents_table[category, :]), 3)) + ' events', fontsize=15)
    ax.get_yaxis().set_visible(False)
    p.legend(bbox_to_anchor=(0., 1.02, 1., .102), fontsize=11, loc=3, ncol=6, borderaxespad=0., mode="expand")
    p.savefig('saves/legacy/legacy_content.png')
    p.show()

if __name__ == "__main__":
    try:
        content_plot_legacy()
    except IOError:
        generate_predictions_legacy(base_features, gen_modes)
        content_plot_legacy()
