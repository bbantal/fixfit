#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 11:46:30 2022

@author: botond


"""

import os
import pickle
import h5py
from tqdm.auto import tqdm
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

print("TensorFlow version:", tf.__version__)

# %%
# =============================================================================
# Setup
# =============================================================================

# Project
date_time = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
VER = 9  # Version
DESC = ""

# Parameters
l1_reg = 0  # Regularization factor in bottleneck
l2_reg = 0  # Regularization factor (l2)
EPOCHS = 5000  # Epochs to train the model for
BATCH_SIZE = 256  # Batch size
K_VALS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])  # Increments to test for bottle nick dim, starting at 1
N_REPS = 10  # Number of replicates

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/"

# Create directory for models to be saved later
model_save_dir = OUTDIR + f"models/cr_lb_ver-{VER}"
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

# Files
files = sorted(os.listdir(SRCDIR + "lb_training_data/"))

# Collections for raw x and y data
x_data_raw_coll = []
y_data_raw_coll = []

# Iterate over all files
for file in tqdm(files, desc="files loaded"):

    # Filepath
    filepath = SRCDIR + f"lb_training_data/{file}"

    # Initialize dictionary for loaded data packet
    packet = {}

    # Load file
    f = h5py.File(filepath)

    # Itearte over items
    for k, v in f.items():

        # Unload
        packet[k] = np.array(v)

    # Append to collection
    x_data_raw_coll.append(packet["params"])
    y_data_raw_coll.append(packet["cc"])

# Convert collections to numpy arrays
x_data_raw = np.array(x_data_raw_coll)
y_data_raw = np.array(y_data_raw_coll)

# Status
print(f"Files loaded. Shapes: X: {x_data_raw.shape}, Y: {y_data_raw.shape}")

# %%
# =============================================================================
# Transform and prepare data
# =============================================================================

# Function to convert outputs
def triu_to_full_matrix(y_flat):
    """
    This function converts a flattened upper triangle
    to its original full square matrix form
    """
    # Number of rows/columns of original square FC matrix
    a = int(np.sqrt(y_flat.shape[0]*2)) + 1

    # Build original square matrix
    y_full = np.zeros((a, a))

    # Populate lower triangle with flattened values
    y_full[np.triu_indices(a, k=1)] = y_flat

    # Populate upper triangle with flattened values
    y_full = y_full + y_full.T

    # Fill diagonal with  1
    np.fill_diagonal(y_full, 1)

    # Return
    return y_full

# Filter out high avg samples
means = np.mean(y_data_raw, axis=(1, 2))
filter_mean = means<.3

print(f"Samples retained: {filter_mean.sum()}/{len(filter_mean)}")

# Apply filter
x_data_filtered = x_data_raw[filter_mean]
y_data_filtered = y_data_raw[filter_mean]

# Take lower triangle
y_data_flattened = y_data_filtered[
    :,
    np.triu_indices_from(y_data_filtered[0], k=1)[0],
    np.triu_indices_from(y_data_filtered[0], k=1)[1]]

# Scale x within channels
x_scaler = MinMaxScaler()
x_data_scaled = x_scaler.fit_transform(x_data_filtered)

# Shuffle before splitting
order = np.arange(x_data_scaled.shape[0])
np.random.shuffle(order)

x_data_shuffled = x_data_scaled[order]
y_data_shuffled = y_data_flattened[order]

# Split into train/validation set
sep_ind = int(filter_mean.sum()*9/10)

x_train, x_val = np.split(x_data_scaled, [sep_ind], axis=0)
y_train, y_val = np.split(y_data_flattened, [sep_ind], axis=0)


# Status
print(f"Data transformed. Shapes: X:{x_train.shape}, Y:{y_train.shape}; "
      "(shown for training only)")

# %%
# =============================================================================
# Neural network approximator
# =============================================================================

# Callbacks
class SaveHistory(Callback):

    def on_epoch_end(self, epoch, logs=None):

        # Get train loss
        current_train_loss = logs.get('mean_squared_error')

        # Get val loss
        current_val_loss = logs.get('val_mean_squared_error')

        # Append to train and val loss to history
        global history_loss
        history_loss["train"].append(current_train_loss)
        history_loss["val"].append(current_val_loss)

        # Update counter
        global ctr_tqdm
        ctr_tqdm.update(1)
        ctr_tqdm.set_postfix({
            "Train Loss": f"{current_train_loss:.4g}",
            "Val Loss": f"{current_val_loss:.4g}"
            })


# Iterations
# --------

# Collection of losses
loss_coll = []

# It over all ks
for i, k in enumerate(K_VALS):

    # It over all reps
    for r in range(N_REPS):

        # Model
        # ----

        input_dim = x_train.shape[1]
        output_dim = y_train.shape[1]

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=200,
            verbose=0,
            mode="min",
            baseline=None,
            restore_best_weights=True,
        )

        # Regular props
        reg_layer_props = {
            "activation": 'relu',
            "use_bias": True,
            "bias_initializer": 'zeros',
            "activity_regularizer": tf.keras.regularizers.L2(l2_reg)
            }

        reg_layer_props2 = {
            "activation": 'relu',
            "use_bias": True,
            "bias_initializer": 'zeros',
            "activity_regularizer": tf.keras.regularizers.L2(l2_reg)
            }

        # Define model
        model = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=(input_dim)),
                tf.keras.layers.Dense(input_dim + 10, **reg_layer_props),
                tf.keras.layers.Dense(input_dim + 10, **reg_layer_props2),
                tf.keras.layers.Dense(k, activation="linear", use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.L1(l1_reg)),
                tf.keras.layers.Dense(output_dim + 10, **reg_layer_props2),
                tf.keras.layers.Dense(output_dim, activation='linear')
        ])


        # Loss function
        loss_func = tf.keras.losses.MeanSquaredError()

        # Compile model
        model.compile(
            optimizer='adam',
            loss=loss_func,
            metrics=['MeanSquaredError']
        )

        # Train model
        # -----

        # Initialize history
        history_loss = {"train": [], "val": []}

        # Initialize tqdm counter
        ctr_tqdm = tqdm(total=EPOCHS, position=0, leave=True)

        # Fit model
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0,
                    callbacks=[SaveHistory(), early_stopping], validation_data=(x_val, y_val));

        # Close counter
        ctr_tqdm.close()

        # Save model
        # model.save(model_save_dir + f"/k-{k}_r-{r}")

        # Add loss to collection
        loss_coll.append(history_loss)

        # Status
        print(f"Scenario: k={k}, r={r}")
        print(f"Final loss: {history_loss['val'][-1]:.2e}")
        print(f"Min loss: {min(history_loss['val']):.2e}")

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

# Observed data
output_dict["x_train"] = x_train
output_dict["y_train"] = y_train
output_dict["x_val"] = x_val
output_dict["y_val"] = y_val
output_dict["input_scaler"] = x_scaler
output_dict["output_scaler"] = None
output_dict["k_vals"] = K_VALS
output_dict["n_reps"] = N_REPS
output_dict["epochs"] = EPOCHS

# Results
output_dict["loss_coll"] = loss_coll

# Pickle output object
output_fname = OUTDIR + f"cr_lb_ver-{VER}.pickle"
with open(output_fname, 'wb') as f:
    pickle.dump(output_dict, f)