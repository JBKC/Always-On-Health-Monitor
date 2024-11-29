from kid_ppg.demo import demo_utils

X, y = demo_utils.load_demo_data()

#############

import matplotlib.pyplot as plt
import scipy
import numpy as np
from tqdm import tqdm  # Import the progress meter


Y = scipy.fft.fft(X, axis = -1)
Y = np.abs(Y)[..., :128]

t = np.arange(Y.shape[0]) / 2
xf = scipy.fft.fftfreq(256, 1/32)[:128] * 60

#############

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import tensorflow as tf
from kid_ppg.preprocessing import sample_wise_z_score_normalization, sample_wise_z_score_denormalization
from kid_ppg.adaptive_linear_model import AdaptiveFilteringModel

n_epochs = 1000

cur_activity_X, ms, stds = sample_wise_z_score_normalization(X.copy())

sgd = tf.keras.optimizers.SGD(
    learning_rate=1e-7,
    momentum=0.9,
    nesterov=True
)

# Set up your model
model = AdaptiveFilteringModel(
    local_optimizer=sgd,
    num_epochs_self_train=n_epochs
)

# Wrap the training logic with tqdm
for epoch in tqdm(range(n_epochs), desc="Training Progress", unit="epoch"):
    # Call the training step or method inside the loop
    X_filtered = model(cur_activity_X[..., None]).numpy()
    print(X_filtered.shape)

# Process the filtered output
X_filtered = X_filtered[:, None, :]
X_filtered = sample_wise_z_score_denormalization(X_filtered, ms, stds)
X_filtered = X_filtered[:, 0, :]


#############


from kid_ppg.kid_ppg import KID_PPG
from kid_ppg.preprocessing import create_temporal_pairs

X_filtered_temp, y_temp = create_temporal_pairs(X_filtered, y)
kid_ppg_model = KID_PPG()

hr_pred_m, hr_pred_std, hr_pred_p = kid_ppg_model.predict_threshold(X_filtered_temp, threshold = 10)