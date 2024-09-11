'''
Main script for training temporal attention model
'''

import numpy as np
import pickle


def main():

    def load_dict(filename='ppg_filt_dict'):

        with open(filename, 'rb') as file:
            data_dict = pickle.load(file)
            print(f'Data dictionary loaded from {filename}')

            return data_dict

    # load dictionary
    dict = load_dict()


if __name__ == '__main__':
    main()

