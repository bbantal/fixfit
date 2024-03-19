"""
Created on Feb 5 2024

@author: bbantal

"""
# %%

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import datetime
import h5py
from tqdm.auto import tqdm
import tensorflow as tf
import dill as pickle

print("TensorFlow version:", tf.__version__)

# %%
# =============================================================================
# Setup
# =============================================================================

# Project
date_time = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
VER = 5  # Version
DESC = ""

# Parameters
l1_reg = 0  # Regularization factor in bottleneck
l2_reg = 0  # Regularizatio factor (l2)
EPOCHS = 5000  # Epochs to train the model for
BATCH_SIZE = 256  # Batch size
K_VALS = np.array([1, 2, 3, 4, 5, 6])  # Increments to test for bottle neck dim, starting at 1
N_REPS = 10

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/"

# Create directory for models to be saved later
model_save_dir = OUTDIR + f"models/cr_big_ver-{VER}"
try:
    os.mkdir(model_save_dir)
except:
    pass

# Save script into variable, to be pickeld along with outputs
try:
    # Filepath to script
    script_filepath = os.path.realpath(__file__)

    # Open script file
    f = open(script_filepath, "r")

    # Convert script into string
    script = f.read()

    print("Saved script.")
except:
    print("Could not save script")

# Status
print(f"\nVERSION: {VER}\nDESC: {DESC}\n")
print("\n\nWARNING: None \n\n")

# %%
# =============================================================================
# Load data
# =============================================================================

# Filepath to data
data_filepath = SRCDIR + "big_training_data/"

# List of files
files = sorted(os.listdir(data_filepath))

# Collections for raw x and y data
x_data_raw_coll = []
y_data_raw_coll = []

# Iterate over all files
for file in tqdm(files, desc="files loaded"):

    # Skip problematic files
    if file in [f'test_sobol095{val}.mat' for val in range(13, 25)]:
        continue

    # Initialize dictionary for loaded data packet
    packet = {}

    # Load file
    f = h5py.File(data_filepath + file)

    # Itearte over items
    for k, v in f.items():

        # Unload
        packet[k] = np.array(v)

    # Append to collection
    x_data_raw_coll.append(packet["params"])
    y_data_raw_coll.append(packet["sol"])

# Convert collections to numpy arrays
x_data_raw = np.array(x_data_raw_coll)
y_data_raw = np.array(y_data_raw_coll)

# Status
print(f"Files loaded. Shapes: X: {x_data_raw.shape}, Y: {y_data_raw.shape}")

# %%
# =============================================================================
# Transform and prepare data
# =============================================================================

# Shuffle samples
# --------------

# Initiate rng
rng = np.random.default_rng(seed=42)

# Create indices for shuffling
indices = np.arange(x_data_raw.shape[0])

# Shuffle the indices
rng.shuffle(indices)

# Apply reordering to data
x_data_raw = x_data_raw[indices]
y_data_raw = y_data_raw[indices]

# Relax seed
np.random.seed(None)

# Scale x values
# ----------------

# # Inspect
# plt.hist(x_data_raw[:, 0])

# Fit scaler
x_scaler = preprocessing.MinMaxScaler()
x_scaler.fit(x_data_raw)

# Apply scaler
x_data_scaled = x_scaler.transform(x_data_raw)

# # Inspect
# plt.hist(x_data_scaled[:, 0])

# Scale y values
# ----------------
# # Inspect y samples before scaling
# plt.plot(y_data_raw[:100, :, 0].T)
# plt.plot(y_data_raw[:100, :, 1].T)

# Extract channels
channel1 = y_data_raw[:, :, 0]
channel2 = y_data_raw[:, :, 1]

# Extremes for scaling
y_min_ch1 =  min(channel1.flatten())
y_max_ch1 =  max(channel1.flatten())

# Scaling functions
y_scaler_ch1 = lambda x: (x - y_min_ch1) / (y_max_ch1 - y_min_ch1)

# Apply scaling
y_data_scaled_ch1 = y_scaler_ch1(channel1)

# # # Inspect
# plt.plot(y_data_scaled_ch1[:100, :].T);

# Downsampling
downsampling_factor = 5
y_data_scaled_ch1 = y_data_scaled_ch1[:, ::downsampling_factor]

# Train-test split
# ----------------

# Validation set
x_val, y_val_ch1, = x_data_scaled[-100:], y_data_scaled_ch1[-100:]

# Index to split at for train/test
sep_ind = int(x_data_scaled.shape[0]*8/10)

# Split x
x_train, x_test = np.split(x_data_scaled[:-100], [sep_ind], axis=0)

# Split channel 1
y_train_ch1, y_test_ch1 = np.split(y_data_scaled_ch1[:-100], [sep_ind], axis=0)

# Status
print(f"Data transformed. Shapes: X_train:{x_train.shape}, Y_train:{y_train_ch1.shape}")

# %%
# =============================================================================
# Neural network approximator
# =============================================================================

# Save history callback
class SaveHistory(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):

        # Get train loss
        current_train_loss = logs.get('mean_squared_error')

        # Get test loss
        current_test_loss = logs.get('val_mean_squared_error')

        # Append to train and test loss to history
        global history_loss
        history_loss["train"].append(current_train_loss)
        history_loss["test"].append(current_test_loss)

        # Update counter
        global ctr_tqdm
        ctr_tqdm.update(1)
        ctr_tqdm.set_postfix({
            "Train Loss": f"{current_train_loss:.4g}",
            "Test Loss": f"{current_test_loss:.4g}"
            })

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor=f"val_loss",
    min_delta=0,
    patience=200,
    verbose=1,
    mode="min",
    baseline=None,
    restore_best_weights=True,
)

# Iterations
# --------

# Collection of losses
loss_coll = []

# It over all k values (bottleneck dimensions)
for i, k in enumerate(K_VALS):

    # It over all replicates
    for r in range(N_REPS):


        # Dimensions
        input_dim = x_train.shape[1]
        output_dim1 = y_train_ch1.shape[1]
        output_dim2 = y_train_ch2.shape[1]

        # Layer properties
        reg_layer_props = {
            "activation": 'relu',
            "use_bias": True,
            "bias_initializer": 'random_normal',
            "activity_regularizer": tf.keras.regularizers.L2(l2_reg)
            }

        # Define model
        # -------------

        # Input layer
        input_layer = tf.keras.layers.Input(shape=(input_dim,), name='input_layer')

        # Common hidden layers
        hidden_layer1 = tf.keras.layers.Dense(50, name='hidden_1', **reg_layer_props)(input_layer)
        hidden_layer2 = tf.keras.layers.Dense(50, name='hidden_2', **reg_layer_props)(hidden_layer1)

        # Bottleneck
        bottleneck = tf.keras.layers.Dense(k, activation="linear", use_bias=False,
                kernel_regularizer=tf.keras.regularizers.L1(l1_reg), name='bottleneck')(hidden_layer2)

        # Branch 1
        branch1_layer1 = tf.keras.layers.Dense(150, name='branch_1_hidden_1', **reg_layer_props)(bottleneck)
        branch1_layer2 = tf.keras.layers.Dense(300, name='branch_1_hidden_2', **reg_layer_props)(branch1_layer1)
        branch1_layer3 = tf.keras.layers.Dense(300, name='branch_1_hidden_3', **reg_layer_props)(branch1_layer2)
        branch1_output_layer = tf.keras.layers.Dense(output_dim1, activation="linear", name='branch_1_output')(branch1_layer3)

        # Define the model by specifying the input and output layers
        model = tf.keras.models.Model(inputs=input_layer, outputs=[branch1_output_layer]) #, branch2_output_layer])

        # Loss function
        loss_func = tf.keras.losses.MeanSquaredError()

        # Compile model
        model.compile(
            optimizer='adam',
            loss={'branch_1_output': loss_func}, #, 'branch_2_output': loss_func},
            metrics=['MeanSquaredError']
            )

        # Train model
        # -----------

        # Initialize history
        history_loss = {"train": [], "test": []}

        # Initialize tqdm counter
        ctr_tqdm = tqdm(total=EPOCHS, position=0, leave=True)

        # Fit model
        model.fit(x_train, y_train_ch1, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0,
                callbacks=[SaveHistory(), early_stopping], validation_data=(x_test, y_test_ch1)); #, y_test_ch2]));

        # Close counter
        ctr_tqdm.close()

        # Save model
        model.save(model_save_dir + f"/k-{k}_r-{r}")

        # Add loss to collection
        loss_coll.append(history_loss)

        # Status
        print(f"Scenario: k={k}, r={r}")
        print(f"Final loss: {history_loss['test'][-1]:.2e}")
        print(f"Min loss: {min(history_loss['test']):.2e}")

# %%
# =============================================================================
# Save
# =============================================================================

# Output dictionary to fill with data
output_dict = {}

# Meta
output_dict["version"] = VER
output_dict["date_time"] = date_time
output_dict["description"] = DESC
output_dict["warning"] = ""

try:
    output_dict["script"] = script
except:
    pass

# Settings
output_dict["k_vals"] = K_VALS
output_dict["n_reps"] = N_REPS
output_dict["epochs"] = EPOCHS

# Data
output_dict["x_train"] = x_train
output_dict["y_train_ch1"] = y_train_ch1
output_dict["y_train_ch2"] = y_train_ch2
output_dict["x_test"] = x_test
output_dict["y_test_ch1"] = y_test_ch1
output_dict["y_test_ch2"] = y_test_ch2
output_dict["input_scaler"] = x_scaler
output_dict["output_scaler_ch1"] = y_scaler_ch1
output_dict["output_scaler_ch2"] = y_scaler_ch2
output_dict["ch1_minmax"] = [y_min_ch1, y_max_ch1]
output_dict["ch2_minmax"] = [y_min_ch2, y_max_ch2]

# Results
output_dict["loss_coll"] = loss_coll

# Pickle output object
output_fname = OUTDIR + f"cr_big_ver-{VER}.pickle"
with open(output_fname, 'wb') as f:
    pickle.dump(output_dict, f)
