import numpy as np

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


