"""
Define all constants needed for what we want to do, and the sklearn models to use
"""
import warnings
import logging
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import BaggingClassifier, RandomForestClassifier,
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier

global_verbosity = False  # if true, prints all confirmation messages, otherwise just the model and its scores.
ignore_warnings = True  # if true, all warnings will be ignored (use with caution)
use_calculated_features = True

luminosity = 2 * 35.9   # (fb-1), factor 2 because only half of the initial data set used for evaluation

cross_sections = {'ggH': 13.41, 'VBFH': 1.044, 'WminusH': 0.147, 'WplusH': 0.232, 'ZH': 0.668, 'ttH': 0.393,
                  'VH': 0.232, 'VH_lept': 0.232, 'VH_hadr': 0.232}

event_numbers = {'ZH': 376657.21875, 'WplusH': 252870.65625, 'WminusH': 168069.609375, 'ttH': 327699.28125,
                 'ggH': 999738.125, 'VBFH': 1885726.125, 'VH': 252870.65625, 'VH_lept': 252870.65625,
                 'VH_hadr': 252870.65625}

production_modes = ['ggH', 'VBFH', 'WminusH', 'WplusH', 'ZH', 'ttH']

event_categories = ['ggH', 'VBFH', 'VH_lept', 'VH_hadr', 'ttH']

gen_modes_merged = ['ggH', 'VBFH', 'ttH', 'VH']


base_path = '/data_CMS/cms/ochando/CJLSTReducedTree/170222/'


base_features = [
                'nExtraLep', 'nExtraZ', 'nCleanedJetsPt30', 'nCleanedJetsPt30BTagged_bTagSF',
                'p_JJQCD_SIG_ghg2_1_JHUGen_JECNominal', 'p_JQCD_SIG_ghg2_1_JHUGen_JECNominal',
                'p_JJVBF_SIG_ghv1_1_JHUGen_JECNominal', 'p_JVBF_SIG_ghv1_1_JHUGen_JECNominal',
                'pAux_JVBF_SIG_ghv1_1_JHUGen_JECNominal', 'p_HadWH_SIG_ghw1_1_JHUGen_JECNominal',
                'p_HadZH_SIG_ghz1_1_JHUGen_JECNominal', 'ZZMass', 'PFMET'
                ]


kin_variables_list_full = \
                [
                'nExtraLep', 'nExtraZ', 'nCleanedJetsPt30', 'nCleanedJetsPt30BTagged_bTagSF',
                'JetQGLikelihood', 'p_JJQCD_SIG_ghg2_1_JHUGen_JECNominal',
                'p_JQCD_SIG_ghg2_1_JHUGen_JECNominal',
                'p_JJVBF_SIG_ghv1_1_JHUGen_JECNominal', 'p_JVBF_SIG_ghv1_1_JHUGen_JECNominal',
                'pAux_JVBF_SIG_ghv1_1_JHUGen_JECNominal',
                'p_HadWH_SIG_ghw1_1_JHUGen_JECNominal', 'p_HadZH_SIG_ghz1_1_JHUGen_JECNominal',
                'JetPhi', 'ZZMass', 'PFMET'
                ]


models_dict = {
        # 'logreg_newt_ovr_invfreq': LogisticRegression(solver='newton-cg', multi_class='ovr', n_jobs=8),
        # 'logreg_newt_ovr_noweight': LogisticRegression(solver='newton-cg', multi_class='ovr', n_jobs=8),
        # 'logreg_newt_ovr_purity': LogisticRegression(solver='newton-cg', multi_class='ovr', n_jobs=8),
        # 'logreg_newt_ovr_content': LogisticRegression(solver='newton-cg', multi_class='ovr', n_jobs=8),
        # 'adaboost_logreg_invfreq': AdaBoostClassifier(LogisticRegression(solver='newton-cg',
        #         multi_class='ovr', n_jobs=8)),
        # 'adaboost_logreg_purity': AdaBoostClassifier(LogisticRegression(solver='newton-cg',
        #         multi_class='ovr', n_jobs=8)),
        # 'adaboost_logreg_content': AdaBoostClassifier(LogisticRegression(solver='newton-cg',
        #         multi_class='ovr', n_jobs=8)),
        'adaboost_logreg_200_purity': AdaBoostClassifier(LogisticRegression(solver='newton-cg',
                multi_class='ovr', n_jobs=8), n_estimators=200),

        'adaboost_stumps_300_purity': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                            n_estimators=300),
        }

