'''
Generate adversarial dataset consisting of noise from original PGG signals + random ground labels
'''

import pickle
import scipy
from scipy import signal
from tqdm import tqdm
import numpy as np

def generate_noise(dict, sessions):
    '''
    Generate noise dataset by filtering original PPG Dalia data
    :param dict: original data dict, each session of shape (n_windows, n_channels, n_samples)
    '''

    fs = 32
    n_taps = 81  # number of filter coefficients
    tol = 2.5  # tolerance for bandstop around HR

    noise_dict = {f'{session}': {} for session in sessions}

    for s in sessions:

        print(f'Generating for Session {s}')

        X = dict[s]['ppg']          # shape (n_windows, n_channels, n_samples)
        y = dict[s]['label']        # shape (n_windows,)

        X_noise = np.zeros(X.shape)                                       # generate noise signal
        y_noise = np.random.uniform(low=20, high=300, size=y.shape)       # assign random ground truths

        noise_dict[s]['y_noise'] = y_noise


        # iterate over values in first column of X (n_windows)
        for i in tqdm(range(X.shape[0])):

            hr = y.flatten()[i]

            # ground HR
            low, high = ((hr - tol) / 60, (hr + tol) / 60)
            b = scipy.signal.firwin(n_taps, [low, high], fs=fs, pass_zero="bandpass")
            y1 = scipy.signal.filtfilt(b, 1, X[i, :, :])

            # 1st harmonic
            low, high = ((2*hr - tol) / 60, (2*hr + tol) / 60)
            b = scipy.signal.firwin(n_taps, [low, high], fs=fs, pass_zero="bandpass")
            y2 = scipy.signal.filtfilt(b, 1, X[i, :, :])

            # 2nd harmonic
            low, high = ((3*hr - tol) / 60, (3*hr + tol) / 60)
            b = scipy.signal.firwin(n_taps, [low, high], fs=fs, pass_zero="bandpass")
            y3 = scipy.signal.filtfilt(b, 1, X[i, :, :])

            X_noise[i, :, :] = X[i, :, :] - (y1 + y2 + y3)

        noise_dict[s]['x_noise'] = X_noise

    # save dictionary
    with open('noise_dict', 'wb') as file:
        pickle.dump(noise_dict, file)
    print(f'Data dictionary saved to noise_dict')

    return


def main():

    def load_dict(filename):

        with open(filename, 'rb') as file:
            data_dict = pickle.load(file)
            print(f'Data dictionary loaded from {filename}')

            return data_dict

    sessions = [f'S{i}' for i in range(1, 16)]

    # import original, unfiltered data dictionary
    data_dict = load_dict(filename='ppg_dalia_dict')

    generate_noise(data_dict, sessions)

    return



if __name__ == '__main__':
    main()