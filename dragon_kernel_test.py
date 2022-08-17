"""
dragon_kernel_test

This is an extraction of a kernel from SynthEco to demonstrate a
failure mode for Dragon
"""
import pandas as pd
import numpy as np
#import dragon
import multiprocessing as mp
import pickle as pkl

def main():
    # unpickle
    data = None

    with open("data.pkl","rb") as f:
        data = pkl.load(f)

    geo_codes_of_interest = data[0]
    fitting_vars = data[1]
    post_results = data[2]
    pums_freq_org = data[3]
    pums_hier = data[4]
    metadata_json = data[5]
    num_cores = 4

    s_argList = []

    with mp.Manager() as manager:
        for g,f_dict in post_results.items():
            s_argList.append([pums_hier, f_dict, g, fitting_vars, metadata_json, 0.0, 0.0001])

        arg_list = [tuple(x) for x in s_argList]
        with manager.Pool(num_cores) as pool:
            result_p = pool.map(_select_households_helper, arg_list)

def calculate_ordinal_distance(pums_val, tab_val, r, k):
    '''
    calculate_ordinal_distance

    calculates the distance between two variables if the range is ordinal
    i.e. the range has some progression and is not "categorical", such as it is a size, or an age range

    pums_val: the value of the fitted variable from pums
    tab_val: the summary table value
    r:  distance parameter
    k: fitting constant

    Returns:
        ordinal distance between the pums_val and tab_val
    '''
    return 1-abs((pums_val - tab_val)/r)**k

def calculate_categorical_distance(pums_val, tab_val, alpha):
    '''
    Document
    '''
    return alpha if pums_val == tab_val else 1.0-alpha

def _select_households(pums, fit_table, geo_code, fitting_vars,
                       metadata_json, alpha=0.0, k=0.001):
    try:
        mat_array = np.array([1.0 for i in range(0, fit_table.shape[0]*pums.shape[0])])
        distance_matrix = mat_array.reshape(fit_table.shape[0], pums.shape[0])
        c_o_d = calculate_ordinal_distance
        c_c_d = calculate_categorical_distance
        for var in fitting_vars:
            pums_ds = metadata_json[var]
            table_values = fit_table[var]
            table_values = table_values.astype(float)
            pums_values = pums[var]
            pums_values = pums_values.astype(float)
            # r is the difference between the maximum and minimum value of the fitting var
            if pums_ds['sample_type'] == "ordinal":
                r = int(pums_values.max()) - int(pums_values.min())
                for i in range(0, distance_matrix.shape[0]):
                    table_value = table_values[table_values.index[i]]
                    distance_tmp = np.array([c_o_d(x, table_value, r, k) for x in list(pums_values)])
                distance_matrix[i] = distance_matrix[i] * distance_tmp
            else:
                for i in range(0, distance_matrix.shape[0]):
                    table_value = table_values[table_values.index[i]]
                    distance_tmp = np.array([c_c_d(x, table_value, 0.0) for x in list(pums_values)])
                    distance_matrix[i] = distance_matrix[i] * distance_tmp

        distance_sums = distance_matrix.sum(axis=1)
        prob_matrix = np.apply_along_axis(lambda x: x/distance_sums, 0, distance_matrix)
        sample_inds = []
        for i in range(0, fit_table.shape[0]):
            t_i = list(fit_table.index)[i]
            n_samples = int(fit_table.loc[t_i, 'total'])
            prob_row = prob_matrix[i]
            inds_samp = pd.Series(pums.index)
            sample_inds = sample_inds + list(inds_samp.sample(n_samples, replace=True, weights=prob_row))

        return (geo_code, sample_inds)
    except Exception as e:
        SynthEcoError("There was a problem in parallel select_housholds:\n{}".format(e))

def _select_households_helper(args):
    #print("hello")
    return _select_households(*args)

class SynthEcoError(Exception):
    """
    SynthEcoError - Class that can be used as an exeption
    and prints to the proper logging for SynthEco
    """
    def __init__(self, msg=""):
        """
        Instant constructor

        Arguments:
            msg: string for message to give details about the error
        """
        self.msg = msg
        print("ERROR", msg)
        super().__init__(self.msg)

if __name__ == "__main__":
    #mp.set_start_method('dragon')
    main()
