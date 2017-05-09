import os


def common_saves_found():
    directories = ['saves/common/', 'saves/common_no_discr/']
    file_list = ['full_test_set.txt, full_training_set.txt']

    for directory in directories:
        for file in file_list:
            if not os.path.isfile(directory + file):
                return False
    return True

