import os
import logging


def common_saves_found():
    directories = ['saves/common/', 'saves/common_no_discr/', 'saves/common_only_discr/']
    file_list = ['full_test_set.txt', 'full_training_set.txt']

    for directory in directories:
        for name in file_list:
            if not os.path.isfile(directory + name):
                logging.info(directory + name + ' not found')
                return False
    return True

