import pickle
import numpy as np
from sklearn import preprocessing as pr
from constants import gen_modes_merged, event_numbers, cross_sections

def frozen(*arg):
    raise AttributeError("This method has been removed")


def expand(array_of_tuples_1d):
    nb_columns = len(array_of_tuples_1d[0])
    nb_rows = np.ma.size(array_of_tuples_1d, 0)
    tmp = np.zeros((nb_rows, nb_columns))
    for i in range(nb_rows):
        for j in range(nb_columns):
            tmp[i, j] = array_of_tuples_1d[i][j]
    return tmp


# def check_number_events():
#     tmp = {'ggH': 0, 'VBFH': 0, 'WminusH': 0, 'WplusH': 0, 'ZH': 0, 'ttH': 0}
#     for idx, tag in enumerate(gen_modes):
#         rfile = r.TFile(base_path + tag + '125/ZZ4lAnalysis.root')
#         tmp[tag] = rfile.Get('ZZTree/Counters').GetBinContent(40)
#     if not tmp == event_numbers:
#         print("Event numbers don't match, please modify constants.py")
#         print(tmp)
#     else:
#         print("Event numbers match, nothing to change")


def prepare_datasets():
    gen_modes_int = gen_modes_merged
    for add_calculated_features in [False, True]:

        if add_calculated_features:
            directory = 'saves/common/'
            suffix = ''
        else:
            directory = 'saves/common_no_discr/'
            suffix = '_no_discr'

        with open(directory + 'scaler.txt', 'rb') as f:
            scaler = pickle.load(f)

        file_list = [directory + mode for mode in gen_modes_int]
        training_set = scaler.transform(np.loadtxt(file_list[0] + '_training.txt'))
        test_set = scaler.transform(np.loadtxt(file_list[0] + '_test.txt'))
        training_labels = np.zeros(np.ma.size(training_set, 0))
        test_labels = np.zeros(np.ma.size(test_set, 0))
        weights_train = np.loadtxt(file_list[0] + '_weights_training.txt') * \
                  cross_sections[gen_modes_int[0]] / event_numbers[gen_modes_int[0]]
        weights_test = np.loadtxt(file_list[0] + '_weights_test.txt') * \
                        cross_sections[gen_modes_int[0]] / event_numbers[gen_modes_int[0]]

        for idx, filename in enumerate(file_list[1:]):
            temp_train = scaler.transform(np.loadtxt(filename + '_training.txt'))
            temp_test = scaler.transform(np.loadtxt(filename + '_test.txt'))
            temp_weights_train = np.loadtxt(filename + '_weights_training.txt') * \
                           cross_sections[gen_modes_int[idx]] / event_numbers[gen_modes_int[idx]]
            temp_weights_test = np.loadtxt(filename + '_weights_test.txt') * \
                                 cross_sections[gen_modes_int[idx]] / event_numbers[gen_modes_int[idx]]
            training_set = np.concatenate((training_set, temp_train), axis=0)
            test_set = np.concatenate((test_set, temp_test), axis=0)
            training_labels = np.concatenate((training_labels, (idx + 1) * np.ones(np.ma.size(temp_train, 0))), axis=0)
            test_labels = np.concatenate((test_labels, (idx + 1) * np.ones(np.ma.size(temp_test, 0))), axis=0)
            weights_train = np.concatenate((weights_train, temp_weights_train), axis=0)
            weights_test = np.concatenate((weights_test, temp_weights_test), axis=0)
            np.savetxt(filename + '_training_scaled.txt', temp_train)
            np.savetxt(filename + '_test_scaled.txt', temp_test)

        np.savetxt('plop/dataset_test' + suffix, test_set)
        np.savetxt('plop/labels_test' + suffix, test_labels)
        np.savetxt('plop/labels_train' + suffix, training_labels)
        np.savetxt('plop/weights_train' + suffix, weights_train)
        np.savetxt('plop/dataset_train' + suffix, training_set)
        np.savetxt('plop/weights_test' + suffix, weights_test)

# def main():
#     with open('saves/common/scaler.txt', 'rb') as f:
#         sc = pickle.load(f)
#     tmp = sc.transform(np.loadtxt('saves/common/ggH_test.txt'))
#     np.savetxt('ggH_test_scaled.txt', tmp)
#
#     with open('saves/common_no_discr/scaler.txt', 'rb') as f:
#         sc = pickle.load(f)
#     tmp2 = sc.transform(np.loadtxt('saves/common/ggH_test.txt'))
#     np.savetxt('ggH_test_scaled_no_discr.txt', tmp2)
#
# main()

def identify_final_state(Z1_flav, Z2_flav, merge_mixed_states=True):
    if Z1_flav == Z2_flav:
        if Z1_flav == -121:
            return 'fs4e'
        else:
            return 'fs4mu'
    else:
        if Z1_flav == -121 or merge_mixed_states:
            return 'fs2e2mu'
        else:
            return 'fs2mu2e'


# def histogram_compare(file_vbf="/data_CMS/cms/ochando/CJLSTReducedTree/170109/ggH125/ZZ4lAnalysis.root",
#                file_ggh='/data_CMS/cms/ochando/CJLSTReducedTree/170109/VBFH125/ZZ4lAnalysis.root',
#                quantity='ZZMass', nb_bins=100):
#
#     vbf_rfile = r.TFile(file_vbf)
#     vbf_tree = vbf_rfile.Get('ZZTree/candTree')
#     vbf_array = tree2array(vbf_tree, branches=quantity, selection='ZZsel > 90')
#     vbf_len = vbf_array.shape[0]
#
#     ggh_rfile = r.TFile(file_ggh)
#     ggh_tree = ggh_rfile.Get('ZZTree/candTree')
#     ggh_array = tree2array(ggh_tree, branches=quantity, selection='ZZsel > 90')
#
#     full_range = [min(min(vbf_array), min(ggh_array)), max(max(vbf_array), max(ggh_array))]
#     bins = np.linspace(full_range[0], full_range[1], nb_bins)
#
#     p.xlim(full_range)
#     p.hist(vbf_array, bins, alpha=0.5, normed=True, label='vbf events')
#     p.hist(ggh_array, bins, alpha=0.5, normed=True, label='ggh events')
#     p.legend(loc='upper right')
#     p.title('Comparison of the repartition of ' + quantity + ' between VBF and ggH events')
#     p.savefig('figs/comparison_' + quantity + '.png')
#     p.show()
#

# def check_category(kin_variables_array, useVHMETTagged=False, useQGTagging=False):
#     tmp = np.append(kin_variables_array, [useVHMETTagged, useQGTagging])
#     if r.categoryMor17(*tmp) in ['VBF2jTaggedMor17', 'VBF1jTaggedMor17']:
#         return True
#     else:
#         return False
#

# def ggh_vs_vbf(file_vbf="/data_CMS/cms/ochando/CJLSTReducedTree/170109/ggH125/ZZ4lAnalysis.root",
#                file_ggh='/data_CMS/cms/ochando/CJLSTReducedTree/170109/VBFH125/ZZ4lAnalysis.root',
#                third_component='plop', show_plot=True):
#
#
#     # kin_variables_list = ['ZZMass', 'ZZsel']
#
#     vbf_rfile = r.TFile(file_vbf)
#     vbf_tree = vbf_rfile.Get('ZZTree/candTree')
#     vbf_array = expand(tree2array(vbf_tree, branches=kin_variables_list, selection='ZZsel > 90'))
#     print(np.shape(vbf_array))
#     vbf_positions = []
#     nb_events_vbf = 0
#     for event in vbf_array:
#         # if check_category(event):
#         if np.random.uniform(0, 1) < 0.01:
#             dim1 = event[kin_variables_list.index('ZZMass')]
#             dim2 = event[kin_variables_list.index('ZZMass')]
#             dim3 = event[kin_variables_list.index('jetPhi')]
#             vbf_positions.append((dim1, dim2, dim3))
#             nb_events_vbf += 1
#     print('Events VBF :', )
#
#     ggh_rfile = r.TFile(file_ggh)
#     ggh_tree = ggh_rfile.Get('ZZTree/candTree')
#     ggh_array = tree2array(ggh_tree, branches=['ZZMass', 'ZZsel'], selection='ZZsel > 90')
#
#     ggh_positions = []
#     nb_events_ggh = 0
#     for event in tqdm(ggh_array):
#         if np.random.uniform(0, 1) < 0.01:
#             dim1 = event[0]
#             dim2 = event[1]
#             dim3 = 0
#             ggh_positions.append((dim1, dim2, dim3))
#             nb_events_ggh += 1
#     print('Events ggH :', nb_events_ggh)
#
#     ggh_filename = 'figs/ggh_'+str(third_component)+'.txt'
#     vbf_filename = 'figs/vbf_'+str(third_component)+'.txt'
#     np.savetxt(ggh_filename, ggh_positions)
#     np.savetxt(vbf_filename, vbf_positions)
#
#     if show_plot:
#         make_3D_plot(ggh_filename, vbf_filename)


# def test():
#     file_vbf = "/data_CMS/cms/ochando/CJLSTReducedTree/170109/ggH125/ZZ4lAnalysis.root"
#     kin_variables_list = ['nExtraLep', 'nExtraZ', 'nCleanedJetsPt30']
#
#     vbf_rfile = r.TFile(file_vbf)
#     vbf_tree = vbf_rfile.Get('ZZTree/candTree')
#     vbf_array = tree2array(vbf_tree, branches=kin_variables_list, selection='ZZsel > 90', stop=5)
#     vbf_array = expand(vbf_array)
#     print(vbf_array)