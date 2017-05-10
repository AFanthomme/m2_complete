import logging
import src.preprocessing as pr
import src.trainer as ctg
from src.constants import *
from src.plotter import content_plot
from src import tests

logging.basicConfig(filename='logs', format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO,
                    datefmt='%H:%M:%S')
logging.info('Logger initialized')

if ignore_warnings:
    warnings.filterwarnings('ignore')


if __name__ == "__main__":

    if features_set_selector == 0:
        directory = 'saves/common_nodiscr/'
        suffix = '_nodiscr'
    elif features_set_selector == 1:
        directory = 'saves/common_onlydiscr/'
        suffix = '_onlydiscr'
    elif features_set_selector == 2:
        directory = 'saves/common_full/'
        suffix = '_full'
    else:
        raise IOError

    if not (tests.common_saves_found() and tests.lengths_consistent()):
        pr.full_process()
    if not (tests.common_saves_found() and tests.lengths_consistent()):
        raise UserWarning

    if not os.path.isdir('figs/tmp'):
        os.makedirs('figs/tmp')

    for model_name in models_dict.keys():
        logging.info('Studying model ' + model_name + suffix)
        try:
            open('saves/' + model_name + suffix + '/categorizer.pkl', 'rb')
        except IOError:
            logging.info('Training model ' + model_name)
            ctg.model_training(model_name)
        try:
            open('saves/' + model_name + suffix + '/predictions.txt', 'rb')
        except IOError:
            logging.info('Generating predictions for ' + model_name + suffix)
            ctg.generate_predictions(model_name)

        content_plot(model_name)
    logging.info('All models studied with features set ' + suffix)
