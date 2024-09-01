'''
Initial file for pulling and processing training data from PPG-DaLiA dataset
'''

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def load_data(session, data_dict):

    def unpack(data):
        # data = data.reshape(-1, 1)
        data = np.squeeze(data)
        return data

    # load relevant data into dictionary
    with open(f'ppg+dalia/{session}/{session}.pkl', 'rb') as file:

        print(f'loading {session}')
        data = pickle.load(file, encoding='latin1')

        data_dict[session]['ppg'] = unpack(data['signal']['wrist']['BVP'])
        data_dict[session]['acc'] = unpack(data['signal']['wrist']['ACC'])
        data_dict[session]['label'] = unpack(data['label'])        # ground truth EEG

    print(data_dict)


    return data_dict


def main():

    # create dictionary to hold all data
    sessions = [f'S{i}' for i in range(1,16)]
    data_dict = {f'{session}': {} for session in sessions}

    for session in sessions:
        data_dict = load_data(session, data_dict)



if __name__ == '__main__':
    main()
