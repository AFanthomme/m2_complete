"""
Define all constants needed for what we want to do, and the sklearn models to use
"""
import warnings
import os
import sys
import time
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
# test

global_verbosity = True  # if true, prints all confirmation messages, otherwise just the model and its scores.
ignore_warnings = False  # if true, all warnings will be ignored (use with caution)
prompt_user = False  # If false, cannot retrain an existing model, if true needs user to answer prompts to retrain
add_calculated_features = False 
redirect_output = True

if redirect_output:
    if not os.path.isdir('logs'):
        os.makedirs('logs')
    sys.stdout = open('logs/log_' + time.strftime("%d_%m_%H_%M"), 'w')
    sys.stderr = open('logs/log_' + time.strftime("%d_%m_%H_%M"), 'w')


if ignore_warnings:
    warnings.filterwarnings('ignore')

luminosity = 2 * 35.9   # (fb-1), factor 2 because only half of the initial data set used for evaluation

cross_sections = {'ggH': 13.41, 'VBFH': 1.044, 'WminusH': 0.147, 'WplusH': 0.232, 'ZH': 0.668, 'ttH': 0.393,
                  'VH': 0.232}

event_numbers = {'ZH': 376657.21875, 'WplusH': 252870.65625, 'WminusH': 168069.609375, 'ttH': 327699.28125,
                 'ggH': 999738.125, 'VBFH': 1885726.125, 'VH': 252870.65625}

gen_modes = ['ggH', 'VBFH', 'WminusH', 'WplusH', 'ZH', 'ttH']

gen_modes_merged = ['ggH', 'VBFH', 'ttH', 'VH']


base_path = '/data_CMS/cms/ochando/CJLSTReducedTree/170222/'


base_features = ['nExtraLep', 'nExtraZ', 'nCleanedJetsPt30', 'nCleanedJetsPt30BTagged_bTagSF',
                      'p_JJQCD_SIG_ghg2_1_JHUGen_JECNominal', 'p_JQCD_SIG_ghg2_1_JHUGen_JECNominal',
                      'p_JJVBF_SIG_ghv1_1_JHUGen_JECNominal', 'p_JVBF_SIG_ghv1_1_JHUGen_JECNominal',
                      'pAux_JVBF_SIG_ghv1_1_JHUGen_JECNominal', 'p_HadWH_SIG_ghw1_1_JHUGen_JECNominal',
                      'p_HadZH_SIG_ghz1_1_JHUGen_JECNominal', 'ZZMass', 'PFMET']


kin_variables_list_full = ['nExtraLep', 'nExtraZ', 'nCleanedJetsPt30', 'nCleanedJetsPt30BTagged_bTagSF',
                      'JetQGLikelihood', 'p_JJQCD_SIG_ghg2_1_JHUGen_JECNominal',
                      'p_JQCD_SIG_ghg2_1_JHUGen_JECNominal',
                      'p_JJVBF_SIG_ghv1_1_JHUGen_JECNominal', 'p_JVBF_SIG_ghv1_1_JHUGen_JECNominal',
                      'pAux_JVBF_SIG_ghv1_1_JHUGen_JECNominal',
                      'p_HadWH_SIG_ghw1_1_JHUGen_JECNominal', 'p_HadZH_SIG_ghz1_1_JHUGen_JECNominal',
                      'JetPhi', 'ZZMass', 'PFMET']





models_dict = {'logreg_newt_ovr_invfreq': LogisticRegression(solver='newton-cg', multi_class='ovr', n_jobs=6),
               'logreg_newt_ovr_noweight': LogisticRegression(solver='newton-cg', multi_class='ovr', n_jobs=6),
               'logreg_newt_ovr_purity': LogisticRegression(solver='newton-cg', multi_class='ovr', n_jobs=6),
               'logreg_newt_ovr_content': LogisticRegression(solver='newton-cg', multi_class='ovr', n_jobs=6),
               'logreg_liblin_ovr_noweight': LogisticRegression(solver='liblinear', multi_class='ovr', n_jobs=6),
               'logreg_sag_ovr_noweight': LogisticRegression(solver='sag', multi_class='ovr', n_jobs=6),
               #'gauss_none_ovr_noweight': GaussianProcessClassifier(n_jobs=2, verbose=True)
               'mlpc_adam_ovr_1_5_noweight': MLPClassifier(hidden_layer_sizes=(5,), verbose=global_verbosity), 
               'mlpc_adam_ovr_1_10_noweight': MLPClassifier(hidden_layer_sizes=(10,), verbose=global_verbosity), 
               'mlpc_adam_ovr_1_3_noweight': MLPClassifier(hidden_layer_sizes=(3,), verbose=global_verbosity),
               'mlpc_adam_ovr_2_5_5_noweight': MLPClassifier(hidden_layer_sizes=(5, 5), verbose=global_verbosity),
               'mlpc_adam_ovr_1_5_log_noweight': MLPClassifier(hidden_layer_sizes=(10,), activation='logistic',
                                                               verbose=global_verbosity),
               # These are too slow
                'svclin_newt_ovr_invfreq': BaggingClassifier(SVC(tol=.1, kernel='linear', decision_function_shape='ovr',
                                                                verbose=False), max_samples=0.01),
                'gauss_none_ovr_noweight': GaussianProcessClassifier(n_jobs=6, verbose=True),

                'bag_logreg_invfreq': BaggingClassifier(LogisticRegression(solver='newton-cg', multi_class='ovr')
                                        , max_samples=0.1, n_jobs=6),
                'bag_logreg_noweight': BaggingClassifier(LogisticRegression(solver='newton-cg', multi_class='ovr')
                                                                            , max_samples=0.1, n_jobs=6),
                'bag_tree_invfreq': BaggingClassifier(max_samples=0.1, n_jobs=6),
                'bag_tree_noweight': BaggingClassifier(max_samples=0.1, n_jobs=6),
                'forests_invfreq': RandomForestClassifier(n_jobs=6),
                'forests_noweight': RandomForestClassifier(n_jobs=6),
                'adaboost_tree_noweight': AdaBoostClassifier(),
                'adaboost_tree_invfreq': AdaBoostClassifier(),
                'adaboost_tree_purity': AdaBoostClassifier(),
                'adaboost_logreg_noweight': AdaBoostClassifier(LogisticRegression(solver='newton-cg',
                                                                                  multi_class='ovr', n_jobs=6)),
                'adaboost_logreg_invfreq': AdaBoostClassifier(LogisticRegression(solver='newton-cg',
                                                                                 multi_class='ovr', n_jobs=6)),
                'adaboost_logreg_purity': AdaBoostClassifier(LogisticRegression(solver='newton-cg',
                                                                                multi_class='ovr', n_jobs=6)),
               }


good_models = {'logreg_newt_ovr_invfreq': LogisticRegression(solver='newton-cg', multi_class='ovr', n_jobs=6),
               'logreg_newt_ovr_noweight': LogisticRegression(solver='newton-cg', multi_class='ovr', n_jobs=6),
               'logreg_newt_ovr_purity': LogisticRegression(solver='newton-cg', multi_class='ovr', n_jobs=6),
               'logreg_newt_ovr_content': LogisticRegression(solver='newton-cg', multi_class='ovr', n_jobs=6),
               }

