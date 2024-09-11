'''
Main script for training temporal attention model
'''

import numpy as np
import pickle


def temporal_pairs(dict):
    '''
    Create temporal pairs between adjacent x_BVP points
    :param dict:
    :return:
    '''

    print(dict['S1']['acc'])

def main():

    def load_dict(filename='ppg_filt_dict'):

        with open(filename, 'rb') as file:
            data_dict = pickle.load(file)
            print(f'Data dictionary loaded from {filename}')

            return data_dict

    # load dictionary
    dict = load_dict()


    temporal_pairs(dict)





if __name__ == '__main__':
    main()

