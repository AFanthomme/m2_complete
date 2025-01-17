import logging
import warnings
import numpy as np
import pickle
import os
import src.preprocessing as pr
import src.trainer as ctg
import src.constants as cst
from src.plotter import content_plot
from src import tests

logging.basicConfig(filename='logs', format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO,
                    datefmt='%H:%M:%S')
logging.info('Logger initialized')

if cst.ignore_warnings:
    warnings.filterwarnings('ignore')


# def main():
#     model_name = 'adaboost_stumps_300_purity'
#     directory, suffix = dir_suff_dict[features_set_selector]
#     scaled_dataset = np.loadtxt(directory + 'full_test_set.txt')
#
#     with open('saves/' + model_name + suffix + '/categorizer.pkl', mode='rb') as file:
#         classifier = pickle.load(file)
#
#     out_path = 'saves/' + model_name + suffix
#     probas = classifier.predict_proba(scaled_dataset)
#     np.savetxt(out_path + '/probas.txt', probas)
#
# main()


if __name__ == "__main__":

    #pr.full_process((0, 2, 3, 4, 5, 6, 1))

    # if not (tests.common_saves_found() and tests.lengths_consistent()):
    #     pr.full_process()
    # if not (tests.common_saves_found() and tests.lengths_consistent()):
    #     raise UserWarning

    for plop in [1, 2, 3, 4, 5]:
        cst.features_set_selector = plop
        directory, suffix = cst.dir_suff_dict[cst.features_set_selector]
        for model_name in cst.models_dict.keys():
            logging.info('Studying model ' + model_name + suffix)
            try:
                open('saves_alt/' + model_name + suffix + '/categorizer.pkl', 'rb')
            except IOError:
                logging.info('Training model ' + model_name)
                ctg.model_training(model_name)
            try:
                open('saves_alt/' + model_name + suffix + '/predictions.txt', 'rb')
            except IOError:
                logging.info('Generating predictions for ' + model_name + suffix)
                ctg.generate_predictions(model_name)

            content_plot(model_name)
        logging.info('All models studied with features set ' + suffix)
