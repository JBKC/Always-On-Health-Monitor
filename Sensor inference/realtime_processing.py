'''
Contains artifact removal on a realtime stream of sensor data
'''

import pickle
import json
import numpy as np
from Heartrate_training_eval.accelerometer_cnn import AdaptiveLinearModel
import torch
import torch.optim as optim
import datetime
from scipy.signal import butter, filtfilt

def z_normalise(X):
    '''
    Z-normalises data for all windows, across each channel, using vectorisation
    :param X: of shape (2, 4, 256)
    :return:
        X_norm: of shape (2, 4, 256)
        ms (means): of shape (2, 4)
        stds (standard deviations) of shape (2, 4)
    '''

    # calculate mean and stdev for each channel in each window - creates shape (n_windows, 4, 1)
    ms = X.mean(axis=2, keepdims=True)
    stds = X.std(axis=2, keepdims=True)

    # Z-normalisation
    X_norm = (X - ms) / np.where(stds != 0, stds, 1)

    return X_norm, ms.squeeze(axis=2), stds.squeeze(axis=2)


def undo_normalisation(X_norm, ms, stds):
    '''
    Transform cleaned PPG signal back into original space following filtering
    :params:
        X_norm: of shape (2, 1, 256)
        ms (means): of shape (2, 4)
        stds (standard deviations) of shape (2, 4)
    :return:
        X: of shape (2, 4, 256)
    '''

    return (X_norm * stds[:, :, np.newaxis]) + ms[:, :, np.newaxis]

def butter_filter_torch(signal, btype, lowcut=None, highcut=None, fs=32, order=5):
    """
    Applies a Butterworth filter to a PyTorch tensor.
    :param signal: PyTorch tensor of shape (n_channels, n_samples)
    :param btype: Type of filter ('lowpass', 'highpass', 'bandpass')
    :param lowcut: Low cutoff frequency (for bandpass/highpass)
    :param highcut: High cutoff frequency (for bandpass/lowpass)
    :param fs: Sampling frequency
    :param order: Filter order
    :return: Filtered PyTorch tensor of shape (n_channels, n_samples)
    """

    # Compute Butterworth filter coefficients
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
    else:
        raise ValueError(f"Unsupported filter type: {btype}")

    # Convert coefficients to PyTorch tensors
    b = torch.tensor(b, dtype=torch.float32)
    a = torch.tensor(a, dtype=torch.float32)

    # Helper function for filtering
    def apply_filter(signal, b, a):
        """
        Applies IIR filtering to the signal using direct-form II transposed structure.
        """
        filtered = torch.zeros_like(signal)
        for i in range(len(b)):
            if i == 0:
                filtered[:, i:] += b[i] * signal[:, i:]
            else:
                filtered[:, i:] += b[i] * signal[:, :-i]
        for i in range(1, len(a)):
            filtered[:, i:] -= a[i] * filtered[:, :-i]
        return filtered

    # Apply filter forward and backward for zero-phase filtering
    filtered = apply_filter(signal, b, a)
    filtered = torch.flip(filtered, dims=[-1])
    filtered = apply_filter(filtered, b, a)
    filtered = torch.flip(filtered, dims=[-1])

    return filtered


def ma_removal(x):
    '''
    Remove motion artifacts from raw PPG data by first training an adaptive filter model, then performing inference
    :param x: single array of shape (n_samples, n_channels) = (320,4)
    :return x_bvp: cleaned signal array of shape (n_windows, n_channels, n_samples) = (2,1,256)
    '''

    # initialise CNN model
    n_epochs = 500
    model = AdaptiveLinearModel()
    optimizer = optim.SGD(model.parameters(), lr=1e-7, momentum=1e-2)

    # transform data into shape (n_windows, n_channels, n_samples) = (2, 4, 256)
    x = np.stack((x[64:, :].T, x[:256, :].T), axis=0)
    # z-normalisation
    x_in, ms, stds = z_normalise(x)

    # filter natively in Pytorch ####
    x_acc = butter_filter_torch(signal=torch.from_numpy(x_in[:, 1:, :]).float().unsqueeze(1), btype='lowpass', highcut=10, fs=32)
    x_ppg = butter_filter_torch(signal=torch.from_numpy(x_in[:, 0, :]).float(), btype='bandpass', lowcut=0.3, highcut=10, fs=32)

    print(x_acc.shape)                  # training data = acc = (2, 1, 3, 256)
    print(x_ppg.shape)                  # labels = ppg = (2, 256)

    x_ppg_json = x_ppg[0, :].tolist()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    with open(f'x_ppg_{timestamp}.json', 'a') as f:
        json.dump(x_ppg_json, f)
        f.write('\n')  # Separate each entry

    # x_acc = torch.from_numpy(np.expand_dims(x_acc, axis=1)).float()
    # x_ppg = torch.from_numpy(x_ppg).float()

    losses = []

    # training loop
    for epoch in range(n_epochs):

        # forward pass through CNN to get x_ma (motion artifact estimate)
        x_ma = model(x_acc)
        # compute loss against raw PPG data
        loss = model.adaptive_loss(y_true=x_ppg, y_pred=x_ma)
        # backprop
        optimizer.zero_grad()           # clear current gradients (avoid gradient accumulation)
        loss.backward()                 # calculate loss
        optimizer.step()                # update model parameters

        # print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')
        # losses.append(loss.item())

    # subtract the motion artifact estimate from raw signal to extract cleaned BVP
    with torch.no_grad():
        x_bvp = x_ppg - model(x_acc)

    # get BVP into original shape: (2, 1, 256) and denormalise
    x_bvp = undo_normalisation(torch.unsqueeze(x_bvp, dim=1).numpy(), ms, stds)

    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # with open(f'losses_{timestamp}.json', 'a') as f:
    #     json.dump(losses, f)
    #     f.write('\n')  # Separate each entry

    return np.expand_dims(x_bvp[:,0,:], axis=1)


def main(snapshot):
    '''
    Accepts 2 windows in form of 10-second snapshot of shape (320,4)
    '''

    return ma_removal(snapshot)

if __name__ == '__main__':
    main()



