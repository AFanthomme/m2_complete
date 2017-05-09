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
    suff = ''
    if not use_calculated_features:
        suff = '_no_discr'

    if not tests.common_saves_found():
        pr.full_process()

    if not os.path.isdir('figs/tmp'):
        os.makedirs('figs/tmp')

    for model_name in models_dict.keys():
        logging.info('Studying model ' + model_name)
        try:
            open('saves/' + model_name + suff + '/categorizer.pkl', 'rb')
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
