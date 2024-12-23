'''

'''

import numpy as np
from scipy.signal import butter, filtfilt

def butter_filter(signal, btype, lowcut=None, highcut=None, fs=32, order=5):
    """
    Applies Butterworth filter
    :param signal: input signal of shape (n_channels, n_samples)
    :return smoothed: smoothed signal of shape (n_channels, n_samples)
    """

    nyquist = 0.5 * fs

    if btype == 'bandpass':
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype=btype)
    elif btype == 'lowpass':
        high = highcut / nyquist
        b, a = butter(order, high, btype=btype)
    elif btype == 'highpass':
        low = lowcut / nyquist
        b, a = butter(order, low, btype=btype)

    # apply filter using filtfilt (zero-phase filtering)
    # filtered = np.array(filtfilt(b, a, signal))

    filtered = np.array([filtfilt(b, a, channel) for channel in signal])

    return filtered

def z_normalise(X):
    '''
    Z-normalises data for a single window, across each channel, using vectorisation
    :param X: of shape (1, 6, 256)
    :return: X_norm: of shape (1, 6, 256)
    '''

    # calculate mean and stdev for each channel in each window - creates shape (n_windows, 4)
    ms = np.mean(X, axis=2)
    stds = np.std(X, axis=2)

    # reshape ms and stds to allow broadcasting
    ms_reshaped = ms[:, :, np.newaxis]
    stds_reshaped = stds[:, :, np.newaxis]

    return (X - ms_reshaped) / np.where(stds_reshaped != 0, stds_reshaped, 1)


def main(snapshot, multi=None):
    '''
    Accepts 1 window of shape (256,4)
    Returns shape (1,6,256) = (n_windows, n_channels, n_samples)
    "multi" is a flag to determine if snapshot needs extra pre-processing
    '''

    # check for if multiple windows are inputted
    if multi==1:
        snapshot = snapshot[-256:, :]

    # transform into correct shape + filter
    ppg = np.expand_dims(snapshot[:,0].T,axis=0)
    acc = snapshot[:, 1:].T

    c = butter_filter(signal=ppg, btype='bandpass', lowcut=0.5, highcut=4)             # cardiac
    r = butter_filter(signal=ppg, btype='bandpass', lowcut=0.2, highcut=0.35)          # respiratory
    m = butter_filter(signal=ppg, btype='highpass', lowcut=4)                          # motion artifacts

    acc = butter_filter(signal=acc, btype='lowpass', highcut=10)

    # combine ppg and acc
    out = np.concatenate([c, r, m, acc], axis=0)
    out = np.expand_dims(out, axis=0)
    # print(out.shape)

    # z-normalise
    out = z_normalise(out)

    return out


if __name__ == '__main__':
    main()
