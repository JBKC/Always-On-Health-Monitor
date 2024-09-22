'''
Initial file for pulling and normalising windowed PPG, accelerometer and activity data
'''

import pickle
import os
import numpy as np

def z_normalise(X):
    '''
    Z-normalises data for all windows, across each channel, using vectorisation
    :param X: of shape (n_windows, 4, 256)
    :return:
        X_norm: of shape (n_windows, 4, 256)
        ms (means): of shape (n_windows, 4)
        stds (standard deviations) of shape (n_windows, 4)
    '''

    # calculate mean and stdev for each channel in each window - creates shape (n_windows, 4)
    ms = np.mean(X, axis=2)
    stds = np.std(X, axis=2)

    # reshape ms and stds to allow broadcasting
    ms_reshaped = ms[:, :, np.newaxis]
    stds_reshaped = stds[:, :, np.newaxis]

    # Z-normalisation
    X_norm = (X - ms_reshaped) / np.where(stds_reshaped != 0, stds_reshaped, 1)

    return X_norm


def main():


    def load_dict(filename='ppg_dalia_dict'):

        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(root_dir, filename)

        with open(filepath, 'rb') as file:
            data_dict = pickle.load(file)
            print(f'Data dictionary loaded from {filename}')

            return data_dict

    data_dict = load_dict()
    z_normalise =






if __name__ == '__main__':
    main()


