'''
Contains artifact removal on a realtime stream of sensor data
'''

import pickle
import numpy as np
from Heartrate_training_eval.accelerometer_cnn import AdaptiveLinearModel
import torch
import torch.optim as optim

def z_normalise(X):
    '''
    Z-normalises data for all windows, across each channel, using vectorisation
    :param X: of shape (2, 4, 256)
    :return:
        X_norm: of shape (2, 4, 256)
        ms (means): of shape (2, 4)
        stds (standard deviations) of shape (2, 4)
    '''

    # calculate mean and stdev for each channel in each window - creates shape (n_windows, 4)
    ms = np.mean(X, axis=2)
    stds = np.std(X, axis=2)

    # reshape ms and stds to allow broadcasting
    ms_reshaped = ms[:, :, np.newaxis]
    stds_reshaped = stds[:, :, np.newaxis]

    # Z-normalisation
    X_norm = (X - ms_reshaped) / np.where(stds_reshaped != 0, stds_reshaped, 1)

    return X_norm, ms, stds

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

    ms_reshaped = ms[:, :, np.newaxis]
    stds_reshaped = stds[:, :, np.newaxis]

    return (X_norm * np.where(stds_reshaped != 0, stds_reshaped, 1)) + ms_reshaped

def ma_removal(x):
    '''
    Remove motion artifacts from raw PPG data by training on accelerometer_cnn
    :param x: single array of shape (n_samples, n_channels) = (320,4)
    :return x_bvp: cleaned signal array of shape (n_windows, n_channels, n_samples) = (2,1,256)
    '''

    # transform data into shape (n_windows, n_channels, n_samples) = (2, 4, 256)
    x = np.stack((x[64:, :].T, x[:256, :].T), axis=0)

    # initialise CNN model
    n_epochs = 1000
    model = AdaptiveLinearModel()
    optimizer = optim.SGD(model.parameters(), lr=1e-7, momentum=1e-2)

    # z-normalisation
    x_in, ms, stds = z_normalise(x)

    # change input shape for 2D conv model
    x_in = np.expand_dims(x_in, axis=1)
    x_in = torch.from_numpy(x_in).float()
    x_acc = x_in[:, :, 1:, :]                 # training data = acc = (2, 1, 3, 256)
    x_ppg = x_in[:, :, :1, :]                 # labels = ppg = (2, 1, 1, 256)

    # training loop
    for epoch in range(n_epochs):

        # forward pass through CNN to get x_ma (motion artifact estimate)
        x_ma = model(x_acc)
        # compute loss against raw PPG data
        loss = model.adaptive_loss(y_true=x_ppg, y_pred=x_ma)
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')

        # subtract the motion artifact estimate from raw signal to extract cleaned BVP
        with torch.no_grad():
            x_bvp = x_ppg[:, 0, 0, :] - x_ma

        # get BVP into original shape: (2, 1, 256)
        x_bvp = torch.unsqueeze(x_bvp, dim=1).numpy()

        # denormalise & take only BVP
        x_bvp = undo_normalisation(x_bvp, ms, stds)
        x_bvp = np.expand_dims(x_bvp[:,0,:], axis=1)

    return x_bvp


def main(snapshot):
    '''
    Accepts 2 windows in form of 10-second snapshot of shape (320,4)
    '''

    return ma_removal(snapshot)

if __name__ == '__main__':
    main()



