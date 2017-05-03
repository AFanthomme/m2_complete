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

kin_variables_list = ['nExtraLep', 'nExtraZ', 'nCleanedJetsPt30', 'nCleanedJetsPt30BTagged_bTagSF',
                      'JetQGLikelihood', 'p_JJQCD_SIG_ghg2_1_JHUGen_JECNominal',
                      'p_JQCD_SIG_ghg2_1_JHUGen_JECNominal',
                      'p_JJVBF_SIG_ghv1_1_JHUGen_JECNominal', 'p_JVBF_SIG_ghv1_1_JHUGen_JECNominal',
                      'pAux_JVBF_SIG_ghv1_1_JHUGen_JECNominal',
                      'p_HadWH_SIG_ghw1_1_JHUGen_JECNominal', 'p_HadZH_SIG_ghz1_1_JHUGen_JECNominal',
                      'JetPhi', 'ZZMass', 'PFMET']

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