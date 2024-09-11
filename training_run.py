'''
Main script for training temporal attention model
'''

import numpy as np
import pickle


def temporal_pairs(dict, sessions):
    '''
    Create temporal pairs between adjacent x_BVP points: x_t & y_t+1
    :param dict: dictionary of session data - each session shape (n_windows, n_channels, n_samples)
    :return:
    '''

    for s in sessions:

        print(dict[s]['bvp'].shape)



def main():

    def load_dict(filename='ppg_filt_dict'):

        with open(filename, 'rb') as file:
            data_dict = pickle.load(file)
            print(f'Data dictionary loaded from {filename}')

            return data_dict

    sessions = [f'S{i}' for i in range(1, 16)]

    # load dictionary
    dict = load_dict()

    temporal_pairs(dict, sessions)





if __name__ == '__main__':
    main()

