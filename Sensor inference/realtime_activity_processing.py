'''

'''

import numpy as np
from scipy.signal import butter, filtfilt

def butter_filter(signal, btype, lowcut=None, highcut=None, fs=32, order=5):
    """
    Applies Butterworth filter
    :param signal: input signal of shape (n_samples,)
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
    filtered = np.array(filtfilt(b, a, signal))

    return filtered

def z_normalise(X):
    '''
    Z-normalises data for all windows, across each channel, using vectorisation
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


def main(snapshot):
    '''
    Accepts 1 window of shape (256,4)
    Returns shape (1,6,256)
    '''

    # filter ppg data

    ppg = snapshot[:,0]

    c = butter_filter(signal=ppg, btype='bandpass', lowcut=0.5, highcut=4)             # cardiac
    r = butter_filter(signal=ppg, btype='bandpass', lowcut=0.2, highcut=0.35)          # respiratory
    m = butter_filter(signal=ppg, btype='highpass', lowcut=4)                          # motion artifacts

    ppg = np.stack([c,r,m])
    out = np.concatenate([ppg, snapshot[:, 1:].T], axis=0)

    out = np.expand_dims(out, axis=0)

    # z-normalise
    out = z_normalise(out)

    return out


if __name__ == '__main__':
    main()
