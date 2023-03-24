#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 00:13:22 2022

@author: botond

"""


import os
import pickle
from tqdm.auto import tqdm
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

print("TensorFlow version:", tf.__version__)

# %%
# =============================================================================
# Setup
# =============================================================================

# Project
date_time = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')

# Version
VER = 42  # Version
DESC = ""
K_VALS = np.array([1, 2, 3, 4])  # Increments to test for bottle nick dim, starting at 1
N_REPS = 10  # Number of replicates

# Model/training parameters
l1_reg = 0  # Regularization factor in bottleneck
l2_reg = 0  # Regularizatio factor (l2)
EPOCHS = 5000  # Epochs to train the model for
BATCH_SIZE = 256  # Batch size

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/"

# Create directory for models to be saved later
model_save_dir = OUTDIR + f"models/cr_tm2_ver-{VER}"
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

# %%
# =============================================================================
# Data generation process
# =============================================================================


def kepler_terms(pars):
    """
    This function produces planetary trajectory parameters ec and p
    given input parameters m1, m2, r0, and w0

    """

    # Extract values
    m1, m2, r0, w0 = pars

    # Compute intermediate terms
    L = m1 * r0**2 * w0  # Angular momentum
    E = 0.5*m1*(w0*r0)**2 - G*m1*m2/r0  # Total energy
    a = -G*m1*m2/(2*E)  # Major axis

    # Compute ellipse properties
    ec = np.sqrt(1 + 2*E*L**2/(m1*(G*m1*m2)**2))  # Eccentricity
    p = a*(1-ec**2)  # Semi-latus rectum

    # Return
    return ec, p


def generate_data_kepler(bounds, M, G):
    """
    Function used for generating data using kepler's equation.
    Maps input parameters (4) to theta-radius (polar) datapairs.
    There is a log transform around sampling to ensure equal representation
    of different scales if using wider bounds for input parameters.
    """
    # Generate raw input space
    # -------

    # Create grid of samples using sobol sampling
    sobol_sampler = stats.qmc.Sobol(d=len(bounds))
    grid = sobol_sampler.random_base2(m=M)

    # Extract values from bounds for vector operations
    b = np.array(list(bounds.values()))

    # Scale hypercube to parameter bounds
    input_data_raw = grid[:, :]*(b[None, :, 1]-b[None, :, 0]) + b[None, :, 0]

    # Clean input space based on eccentricity # (throw out non-elliptic trajectories)
    # -----

    # Unpack parameters
    m1, m2, r0, w0 = input_data_raw.T

    # Compute trajectory parameters
    ec, p = kepler_terms([m1, m2, r0, w0])

    # Compute filter based on eccentricity < 1
    input_filter = (ec<0.95) & (ec>0.7)

    # Apply filter to input data
    input_data = input_data_raw[input_filter]

    # Status
    print(f"Samples retained: {input_filter.sum()}/{2**M}")

    ## %%
    # Generate outputs based on inputs
    # ----

    # Number of theta increments
    n_theta_inc = 100

    # Generate theta grid (1d)
    theta = np.linspace(0, 2*np.pi, n_theta_inc)

    # Unpack parameters
    m1, m2, r0, w0 = input_data.T

    # Compute trajectory parameters
    ec, p = kepler_terms([m1, m2, r0, w0])

    # Compute r as a function of theta at given p and ec
    r = p[:, None]/(1+ec[:, None]*np.cos(theta[None, :]))  # Source: https://en.wikipedia.org/wiki/Kepler_orbit

    # Store r as output data
    output_data = r

    # Return
    return input_data, output_data

def data_scaler(input_data, output_data):
    """
    Scaling to make input and output values friendly for the autoencoder.
    Note that this type of scaling makes data specifici to this particular sample.
    If we scale like this we cannot combine different sets of samples anymore!
    It becomes conceptual at this point
    """

    # Scale data
    # -------

    # Scale inputs
    # input_scaler = MinMaxScaler()
    # input_data_scaled = input_scaler.fit_transform(input_data)

    input_scaler = None
    input_data_scaled = input_data.copy()


    # Scale outputs
    output_scaler = None
    output_data_log = np.log(output_data)

    vmin = output_data_log.min()
    vmax = output_data_log.max()

    output_data_scaled = (output_data_log - vmin)/(vmax-vmin)

    output_scaler_dict = {}
    output_scaler_dict["output_scaler"] = output_scaler
    output_scaler_dict["output_vmin"] = vmin
    output_scaler_dict["output_vmax"] = vmax
    output_scaler_dict["output_scaler_desc"] = "(log(x) - vmin)/(vmax - vmin)"

    # output_scaler = MinMaxScaler()
    # output_data_scaled = output_scaler.fit_transform(output_data.T).T

    # Inspect input and output spaces with respect to scaling
    plt.figure()
    for i in range(input_data_scaled.shape[1]):
        plt.subplot(1, input_data_scaled.shape[1], i+1)
        plt.title(list(bounds.keys())[i])
        plt.hist(input_data_scaled[:, i], edgecolor="black", linewidth=1, alpha=0.8)

    plt.tight_layout()

    plt.figure()
    plt.plot(output_data_scaled[:100, :].T);
    # plt.ylim([-1, 0])

    return input_data_scaled, output_data_scaled, input_scaler, output_scaler_dict

# %%
# Generate data
# -------

# Constants
G = 0.5  # m^3 kg^-1 day^-2

# Number of data points, as a power of two
M = 13

# Ranges for inputs
bounds = {
    "m1": [.1, 1],
    "m2": [.1, 1],
    "r0": [.1, 1],
    "w0": [.1, 1],
    }

# Generate
input_data, output_data = \
    generate_data_kepler(bounds, M, G)

# Transform data
# --------

# Scale samples
input_data_scaled, output_data_scaled, input_scaler, output_scaler_dict = \
    data_scaler(input_data, output_data)

# Shuffle samples
order = np.arange(input_data_scaled.shape[0])
np.random.shuffle(order)

input_data_shuffled = input_data_scaled[order]
output_data_shuffled = output_data_scaled[order]

# Index at which to split
sep_ind = int(input_data_shuffled.shape[0]*9/10)

# Perform train/val split
x_train, x_val = np.split(input_data_shuffled, [sep_ind], axis=0)
y_train, y_val = np.split(output_data_shuffled, [sep_ind], axis=0)


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
            "activation": 'tanh',
            "use_bias": True,
            "bias_initializer": 'zeros',
            "activity_regularizer": tf.keras.regularizers.L2(l2_reg)
            }

        # Define model
        model = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=(input_dim)),
                tf.keras.layers.Dense(input_dim + 10, **reg_layer_props),
                tf.keras.layers.Dense(input_dim + 10, **reg_layer_props),
                tf.keras.layers.Dense(k, activation="tanh", use_bias=False,#TODO
                        kernel_regularizer=tf.keras.regularizers.L1(l1_reg)),
                tf.keras.layers.Dense(output_dim + 10, **reg_layer_props),
                tf.keras.layers.Dense(output_dim + 10, **reg_layer_props),
                tf.keras.layers.Dense(output_dim, activation='linear')
        ])

        # Loss function
        loss_func = tf.keras.losses.MeanSquaredError()
        # weights = 1/output_train.mean(axis=0)
        # loss_func = lambda y_real, y_model: tf.keras.backend.mean(((y_real - y_model)*weights)**2)

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
        model.save(model_save_dir + f"/k-{k}_r-{r}")

        # Add loss to collection
        loss_coll.append(history_loss)

        # Status
        print(f"Final loss: {history_loss['val'][-1]:.2e}")

# %%
# =============================================================================
# Save
# =============================================================================

output_dict = {}

# Meta
output_dict["version"] = VER
output_dict["date_time"] = date_time
output_dict["description"] = DESC

try:
    output_dict["script"] = script
except:
    pass

# Observed data
output_dict["bounds"] = bounds
output_dict["G"] = G
output_dict["x_train"] = x_train
output_dict["y_train"] = y_train
output_dict["x_val"] = x_val
output_dict["y_val"] = y_val
output_dict["input_scaler"] = input_scaler
output_dict["output_scaler_dict"] = output_scaler_dict
output_dict["epochs"] = EPOCHS
output_dict["k_vals"] = K_VALS
output_dict["n_reps"] = N_REPS
output_dict["warning"] = ""

# Results
output_dict["loss_coll"] = loss_coll

# Pickle output object
output_fname = OUTDIR + f"cr_tm2_ver-{VER}.pickle"
with open(output_fname, 'wb') as f:
    pickle.dump(output_dict, f)