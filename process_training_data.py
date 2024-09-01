'''
Initial file for pulling and processing training data from PPG-DaLiA dataset
'''

import pickle
import numpy as np
import motion_artifact_removal
import matplotlib.pyplot as plt

def save_data(session, data_dict):

    def unpack(data):
        # data = data.reshape(-1, 1)
        data = np.squeeze(data)
        return data

    # load relevant data into dictionary
    with open(f'ppg+dalia/{session}/{session}.pkl', 'rb') as file:

        print(f'saving {session}')
        data = pickle.load(file, encoding='latin1')

        data_dict[session]['ppg'] = unpack(data['signal']['wrist']['BVP'])
        data_dict[session]['acc'] = unpack(data['signal']['wrist']['ACC'])
        data_dict[session]['label'] = unpack(data['label'])        # ground truth EEG

        # plt.plot(data_dict[session]['ppg'])
        # plt.show()

    return data_dict

def window_data(data_dict):
    return


def main():

    def save_dict(filename='ppg_dalia_dict'):

        # create dictionary to hold all data
        sessions = [f'S{i}' for i in range(1, 16)]
        data_dict = {f'{session}': {} for session in sessions}

        # iterate over sessions
        for session in sessions:
            data_dict = save_data(session, data_dict)

        # save dictionary
        with open(filename, 'wb') as file:
            pickle.dump(data_dict, file)
        print(f'Data dictionary saved to {filename}')

        return

    def load_dict(filename='ppg_dalia_dict'):

        with open(filename, 'rb') as file:
            data_dict = pickle.load(file)
            print(f'Data dictionary loaded from {filename}')
            return data_dict

    # comment out save or load
    # save_dict()
    data_dict = load_dict()

    # window data
    data_dict = window_data(data_dict)

    # remove motion artifacts
    motion_artifact_removal.main(data_dict)

if __name__ == '__main__':
    main()
