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

def window_data(data_dict, sessions):
    '''
    Segment data into windows of 8 seconds with 2 second overlap
    :param data_dict: dictionary with all signals in arrays
    :param sessions: list of sessions
    :return: dictionary with windowed signals:
        ppg.shape = (window length, n_windows)
        acc.shape = (3, 256, n_windows) - to match Pytorch format
    '''

    fs = {
        'ppg': 64,
        'acc': 32
    }

    for session in sessions:
        for k, f in fs.items():

            window = 8 * f          # size of window
            step = 2 * f

            data = data_dict[session][k]
            len_signal = len(data)
            n_windows = int((len_signal - window) / step + 1)

            if len(data.shape) == 1:
                # PPG
                data_dict[session][k] = np.zeros((window, n_windows))
                for i in range(n_windows):
                    start = i * step
                    end = start + window
                    data_dict[session][k][:, i] = data[start:end]

            else:
                # accelerometer
                data_dict[session][k] = np.zeros((data.shape[-1], window, n_windows))
                for i in range(n_windows):
                    start = i * step
                    end = start + window
                    data_dict[session][k][:, :, i] = data[start:end, :].T

    print(data_dict['S1']['acc'].shape)

    return data_dict


def main():

    def save_dict(sessions, filename='ppg_dalia_dict'):

        # create dictionary to hold all data
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

    sessions = [f'S{i}' for i in range(1, 16)]

    # comment out save or load
    # save_dict(sessions)
    data_dict = load_dict()

    # window data
    data_dict = window_data(data_dict, sessions)

    # remove motion artifacts
    motion_artifact_removal.main(data_dict)

if __name__ == '__main__':
    main()
