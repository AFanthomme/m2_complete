"""
Define all constants needed for what we want to do, and the sklearn models to use
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from src.custom_classifiers import SelfThresholdingAdaClassifier

global_verbosity = 0
ignore_warnings = True

# Select one of the already defined modes. The selector is evaluated whenever it is needed, so that this value
# can easily be overriden from main.py
# To add new sets of features (either from file or calculated), add the corresponding file and suffix here then
# modify preprocessing.py
features_set_selector = 0

# dir_suff_dict = [('saves/common_nodiscr/', '_nodiscr'), ('saves/common_onlydiscr/', '_onlydiscr'),
#                  ('saves/common_full/', '_full')]
# production_modes = ['ggH', 'VBFH', 'WminusH', 'WplusH', 'ZH', 'ttH']
# event_categories = ['ggH', 'VBFH', 'VH_lept', 'VH_hadr', 'ttH']

dir_suff_dict = [('saves_alt/common_nodiscr/', '_nodiscr'), ('saves_alt/common_onlydiscr/', '_onlydiscr'),
                 ('saves_alt/common_full/', '_full')]


# These are the physical constants
luminosity = 2 * 35.9   # (fb-1), factor 2 because only half of the initial data set used for evaluation
cross_sections = {'ggH': 13.41, 'VBFH': 1.044, 'WminusH': 0.147, 'WplusH': 0.232, 'ZH': 0.668, 'ttH': 0.393,
                  'VH': 0.232, 'VH_lept': 0.232, 'VH_hadr': 0.232, 'bbH': 0.1347, 'tqH': 0}
event_numbers = {'ZH': 376657.21875, 'WplusH': 252870.65625, 'WminusH': 168069.609375, 'ttH': 327699.28125,
                 'ggH': 999738.125, 'VBFH': 1885726.125, 'VH': 252870.65625, 'VH_lept': 252870.65625,
                 'VH_hadr': 252870.65625, 'bbH':327699.28125, 'tqH': 0}


production_modes = ['ggH', 'VBFH', 'WminusH', 'WplusH', 'ZH', 'ttH', 'bbH']
event_categories = ['ggH', 'VBFH', 'VH_lept', 'VH_hadr', 'ttH', 'bbH']

base_features = [
                'nExtraLep', 'nExtraZ', 'nCleanedJetsPt30', 'nCleanedJetsPt30BTagged_bTagSF',
                'p_JJQCD_SIG_ghg2_1_JHUGen_JECNominal', 'p_JQCD_SIG_ghg2_1_JHUGen_JECNominal',
                'p_JJVBF_SIG_ghv1_1_JHUGen_JECNominal', 'p_JVBF_SIG_ghv1_1_JHUGen_JECNominal',
                'pAux_JVBF_SIG_ghv1_1_JHUGen_JECNominal', 'p_HadWH_SIG_ghw1_1_JHUGen_JECNominal',
                'p_HadZH_SIG_ghz1_1_JHUGen_JECNominal', 'ZZMass', 'PFMET'
                ]

likelihood_names = ['p_JJQCD_SIG_ghg2_1_JHUGen_JECNominal', 'p_JQCD_SIG_ghg2_1_JHUGen_JECNominal',
                'p_JJVBF_SIG_ghv1_1_JHUGen_JECNominal', 'p_JVBF_SIG_ghv1_1_JHUGen_JECNominal',
                'pAux_JVBF_SIG_ghv1_1_JHUGen_JECNominal', 'p_HadWH_SIG_ghw1_1_JHUGen_JECNominal',
                'p_HadZH_SIG_ghz1_1_JHUGen_JECNominal']

backgrounds = ['']

decision_stump = DecisionTreeClassifier(max_depth=1)
my_classifier = SelfThresholdingAdaClassifier(base_estimator=decision_stump, n_estimators=300)

models_dict = {
        # 'logreg_newt_ovr_invfreq': LogisticRegression(solver='newton-cg', multi_class='ovr', n_jobs=8),
        # 'adaboost_logreg_purity': AdaBoostClassifier(LogisticRegression(solver='newton-cg',
        #         multi_class='ovr', n_jobs=8)),
        # 'adaboost_logreg_content': AdaBoostClassifier(LogisticRegression(solver='newton-cg',
        #         multi_class='ovr', n_jobs=8)),
        # 'adaboost_logreg_200_purity': (AdaBoostClassifier(LogisticRegression(solver='newton-cg',
        #         multi_class='ovr', n_jobs=8), n_estimators=200), None),
        #
        # 'adaboost_stumps_300_purity': (AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
        #                     n_estimators=300), None),
        'adaboost_stumps_300_15_custom': (AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                            n_estimators=300), [1.5, 1., 1., 1., 1.]),
        'adaboost_stumps_300_20_custom': (AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                            n_estimators=300), [2, 1., 1., 1., 1.]),
        'adaboost_stumps_300_25_custom': (AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                                            n_estimators=300), [2.5, 1., 1., 1., 1.]),
        'adaboost_stumps_300_35_custom': (AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                                         n_estimators=300), [3.5, 1., 1., 1., 1.]),
        'adaboost_stumps_300_45_custom': (AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                                         n_estimators=300), [4.5, 1., 1., 1., 1.]),
        'adaboost_stumps_300_50_custom': (AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                                         n_estimators=300), [5., 1., 1., 1., 1.]),
        }


