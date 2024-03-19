"""
Created on Feb 22 2024

@author: bbantal

"""
# %%

import os
import numpy as np
import pandas as pd
import dill as pickle
import datetime
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import tensorflow as tf
from SALib.analyze import hdmr
from scipy import optimize

import plotting_style

print("TensorFlow version:", tf.__version__)

# %%
# =============================================================================
# Setup
# =============================================================================

# Project
date_time = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/"

# %%
# Load pretrained network
# -------

# Version
model_name = "big"
ver = 5

# Filepath
fp = OUTDIR + f"cr_{model_name}_ver-{ver}.pickle"

# Open pickle
with open(fp, 'rb') as handle:
    output = pickle.load(handle)

# Description
print(output["description"])

# Load x/y data
x_train = output["x_train"]
y_train_ch1 = output["y_train_ch1"]

# Scaling functions
ch1_min, ch1_max = output["ch1_minmax"]
output_scaler_ch1 = lambda x: (x - ch1_min) / (ch1_max - ch1_min)
output_scaler_ch1_inverse = lambda x: x * (ch1_max - ch1_min) + ch1_min

# Loss history
loss_coll = output["loss_coll"]

# Load model
k = 4
r = 8
model_fp = OUTDIR + f"models/cr_{model_name}_ver-{ver}/k-{k}_r-{r}"
model = tf.keras.models.load_model(model_fp)


# %%
# =============================================================================
# Global sensitivity analysis (GSA)
# =============================================================================

# Number of input parameters
K_inp = 6

# Find bottleneck layer
k = 4 # Bottleneck dimension
bn_layer_ind = [model.layers[i].output_shape[1] for i in range(1, len(model.layers))].index(k) + 1

# Extract encoder from fitted model
encoder = tf.keras.models.Model(model.input, model.layers[bn_layer_ind].output)
# encoder.summary()

# Compute latent space using encoder
latent_space_nn = encoder(x_train).numpy()

# Define problem dict for GSA
problem = {
    'num_vars': x_train.shape[1],
    'names': [f'x{j}' for j in range(x_train.shape[1])],
    'bounds': [[x_train[:, j].min(), x_train[:, j].max()] \
               for j in range(x_train.shape[1])]
        }

# Collection for SA results
SA_res = []

# Repeat for all latent terms
for i in range(k):

    # Perform SA
    res = hdmr.analyze(problem, x_train,  latent_space_nn[:, i], print_to_console=True)

    # Append to results
    SA_res.append(res)

# %%
# Plot GSA results on heatmap
# -----------

# Extract and transform GSA results
DLDI = np.array([SA_res[i]["Sa"][:K_inp] for i in range(k)])

# Figure
plt.figure(figsize=(3.1, 3.4), dpi=300)

# Plot
plt.pcolormesh(DLDI.T, cmap="BuGn", vmax=0.8) #, vmax=0.5)

# Format
plt.gca().set_aspect("equal")
labels = ["C", r"S${_i}$", r"$p$", r"$\alpha$", r"$\gamma$", r"u$_{ext}$"]
plt.xticks(np.arange(0.5, k, 1), [f"L$_{i+1}$" for i in range(k)])
plt.yticks(np.arange(0.5, K_inp, 1), labels, rotation=0)
plt.gca().invert_yaxis()
plt.xlabel("Latent parameter")
plt.ylabel("Input parameter", labelpad=-8)

# Colorbar
cbar = plt.colorbar(shrink=1, aspect=25*0.7, label="Sensitivity")
cbar.set_ticks(np.arange(0, 0.9, 0.2))
# cbar.set_ticklabels([])

# Add spines
ax = plt.gca()
for sp in ['bottom', 'top', 'right', 'left']:
    ax.spines[sp].set_linewidth(1.4)
    ax.spines[sp].set_color("black")

plt.tight_layout()

# # Save fig
# plt.savefig(
#     OUTDIR + "figures/cr_big_heatmap.pdf",
#     transparent=True)

# %%
# =============================================================================
# Global optimization
# =============================================================================

#NOTE: run this section separately for both glucose_hc.csv and for glucose_obese_.csv.
# Make sure to pickle them both with different filenames before visualizing the results!

# Load data
# -------------

# Open data
data_glc_raw = pd \
    .read_csv(SRCDIR + "data/big_observed_data/glucose_hc.csv", header=None) \
    .set_axis(["time", "glucose"], axis=1)

# # Inspect raw data
# plt.figure()
# plt.plot(data_glc_raw["time"], data_glc_raw["glucose"])

# Resample data
# -------------

# Time increments to resample time to
downsampling_factor = 5
target_time_incs = np.linspace(0, y_train_ch1.shape[1], y_train_ch1.shape[1])*downsampling_factor

# Transform original time increments to target range
original_time_incs = (data_glc_raw["time"] - data_glc_raw["time"].min()) \
    * (target_time_incs.max()/(data_glc_raw["time"].max() - data_glc_raw["time"].min()))

# Resample data across time
data_glc_resampled = np.interp(target_time_incs, original_time_incs, data_glc_raw["glucose"])

# Rescale data
data_glc = output_scaler_ch1(data_glc_resampled)

# Inspect
plt.figure()
plt.plot(target_time_incs, y_train_ch1[0], label="simulated")
plt.plot(target_time_incs, data_glc, label="data")
plt.legend()

# %%
# Value ranges for optimization
# ------------------

# Colors
colors = ["teal", "orange", "indigo", "dodgerblue", "violet"]  

# Inspect distribution of values of latent parameters
plt.figure()
for i in range(latent_space_nn.shape[1]):
    plt.subplot(latent_space_nn.shape[1], 1, i+1)
    plt.hist(latent_space_nn[:, i], color=colors[i])
    plt.axvline(0, color='k')
plt.tight_layout()

# Define bounds
bounds_opt = np.array([
    [-1, 1.],
    [-1.25, 0.5],
    [-0.25, 1.],
    [-0.75, 0.75]
    ])

# Bounds on hypercube
bounds_kwg = [[0., 1.] for i in range(len(bounds_opt))]

# Objective function
# ----------

# Objective function
def compute_obj(pars_to_try, scale=True):
    """
    Function to minimize
    """

    # Convert proposed parameters to numpy array
    pars_to_try = np.array(pars_to_try)

    # Rescale parameters from cube
    pars_to_try_rescaled = pars_to_try*(bounds_opt[:, 1]-bounds_opt[:, 0]) + bounds_opt[:, 0]

    # Check if proposed values are within bounds
    if not all([pars_to_try[i] > bounds_kwg[i][0] and pars_to_try[i] < bounds_kwg[i][1] \
                for i in range(len(pars_to_try))]):
        obj_val = np.inf

    else:

        # Get trajectory based on input parameters
        data_model = decoder(tf.convert_to_tensor(
            np.array(pars_to_try_rescaled)[np.newaxis, :])).numpy()[0]


        # Get objective value (sum of squares)
        obj_val = np.sum((data_obs - data_model)**2)

    # Prevent nans being returned, return a large value instead
    if np.isnan(obj_val):
        obj_val = np.inf

    # Print
    # print(pars_to_try_rescaled, obj_val)

    # Save results
    global trials
    trials.append([pars_to_try_rescaled, obj_val])

    # Return
    return obj_val

# Preparations
# -----------

# Extract the decoder from the neural network
decoder = tf.keras.models.Model(model.layers[bn_layer_ind+1].input, model.output)

# Define observed data to fit
data_obs = data_glc

# %%
# Run fitting
# -------------------------------

# Trials to save intermediates
trials = []

# Run optimization
res = optimize.basinhopping(
    compute_obj,
    x0=[0.5 for i in range(len(bounds_opt))],
    stepsize=0.2,
    minimizer_kwargs={"method": "BFGS",}
    )

# Rescale results
pars_conv = res["x"]*(bounds_opt[:, 1]-bounds_opt[:, 0]) + bounds_opt[:, 0]

# Status
print(f"Optimiziation has finished. Number of evaluations: {len(trials)}")

# %%
# Pickle fitting results
# ------

# Dictionary for pickling
output_dict = {}

# Meta
output_dict["version"] = ver
output_dict["k"] = k
output_dict["rep"] = r

# Data
output_dict["time"] = target_time_incs
output_dict["data_obs_resampled"] = data_glc_resampled

# Optimization results
output_dict["res"] = res
output_dict["trials"] = trials
output_dict["bounds_opt"] = bounds_opt

# # Pickle output object
# extra = f"2"
# output_fname = OUTDIR + f"fitting/cr_big_fitting_latent_" \
#     f"ver-{ver}_k-{k}_r-{r}_{extra}.pickle"
# with open(output_fname, 'wb') as f:
#     pickle.dump(output_dict, f)

# %%
# Inspect results: combined
# -----------------

# Files to load
file_1 = f"fitting/cr_big_fitting_latent_ver-{ver}_k-{k}_r-{r}_1.pickle"
file_2 = f"fitting/cr_big_fitting_latent_ver-{ver}_k-{k}_r-{r}_2.pickle"

# Open files
with open(OUTDIR + file_1, 'rb') as handle:
    output_1 = pickle.load(handle)
with open(OUTDIR + file_2, 'rb') as handle:
    output_2 = pickle.load(handle)

# Fitted values
pars_conv_1 = output_1['res'].x*(bounds_opt[:, 1]-bounds_opt[:, 0]) + bounds_opt[:, 0]
pars_conv_2 = output_2['res'].x*(bounds_opt[:, 1]-bounds_opt[:, 0]) + bounds_opt[:, 0]

# Unpack
bounds_opt = output_1['bounds_opt']
target_time_incs = output_1['time']
data_glc_resampled_1 = output_1['data_obs_resampled']
data_glc_resampled_2 = output_2['data_obs_resampled']

# Rescale fitted latent parameters from cube
latent_pars_fitted_1 = output_1['res'].x*(bounds_opt[:, 1]-bounds_opt[:, 0]) + bounds_opt[:, 0]
latent_pars_fitted_2 = output_2['res'].x*(bounds_opt[:, 1]-bounds_opt[:, 0]) + bounds_opt[:, 0]

# Compute fitted time-series
data_fitted_1 = decoder(tf.convert_to_tensor(
            np.array(latent_pars_fitted_1)[np.newaxis, :])).numpy()[0]
data_fitted_2 = decoder(tf.convert_to_tensor(
            np.array(latent_pars_fitted_2)[np.newaxis, :])).numpy()[0]

# Rescale fitted time-series
data_fitted_retroscaled_1 = output_scaler_ch1_inverse(data_fitted_1)
data_fitted_retroscaled_2 = output_scaler_ch1_inverse(data_fitted_2)

# Inspect the time-series
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(4.0, 3.1), dpi=300)
plt.sca(ax1)
plt.title("Control")
plt.plot(target_time_incs, data_glc_resampled_1, label="data", color="teal", lw=1.0)
plt.plot(target_time_incs, data_fitted_retroscaled_1, label="fitted", color="crimson", lw=1.0)
plt.xlabel("Time [min]")
plt.ylabel("Glucose [mM]")
plt.ylim([4.3, 8.3])

plt.sca(ax2)
plt.title("Obese")
plt.plot(target_time_incs, data_glc_resampled_2, label="data", color="teal", lw=1.0)
plt.plot(target_time_incs, data_fitted_retroscaled_2, label="fitted", color="crimson", lw=1.0)
plt.xlabel("Time [min]")
plt.legend(fontsize=9)

# Splines
for sp in ['bottom', 'top', 'right', 'left']:
    ax1.spines[sp].set_linewidth(1.4)
    ax1.spines[sp].set_color("black")

for sp in ['bottom', 'top', 'right', 'left']:
    ax2.spines[sp].set_linewidth(1.4)
    ax2.spines[sp].set_color("black")

plt.tight_layout(pad=0.4)

# # Save
# plt.savefig(OUTDIR + f"figures/cr_{model_name}_fitting.pdf",
#             transparent=True)

# %%
# =============================================================================
# Misc plots
# =============================================================================

# %%
# SI: input parameter distritubions
# --------

# colors = ["dodgerblue", "teal", "gold", "crimson"]
colors = ["crimson"]*6
labels = ["C", r"S${_i}$", r"$p$", r"$\alpha$", r"$\gamma$", r"u$_{ext}$"]

plt.figure(figsize=(7.25, 4), dpi=300)

for i in range(x_train.shape[1]):

    plt.subplot(2, 3, i+1)
    plt.title("Parameter: " + labels[i])
    plt.hist(x_train[:, i], color=colors[i], edgecolor="black",
             linewidth=1.5, alpha=0.8, bins=np.arange(0.1, 1.1, 0.05))
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.xlim([0, 1])
    plt.ylim([0, 450])

    ax = plt.gca()
    for sp in ['bottom', 'top', 'right', 'left']:
        ax.spines[sp].set_linewidth(1.5)

plt.tight_layout()

# # Save fig
# plt.savefig(
#     OUTDIR + "figures/cr_big_x_train_dist.pdf",
#     transparent=True)

# %%
# SI: Output space data examples
# ----------
    
shades = np.linspace(0, 1, 20)
blue_shades = plt.cm.viridis(shades)

plt.figure(figsize=(3.625, 2.5), dpi=300)

for i in range(20):
    plt.plot(y_train_ch1[i, :].T, lw=1, color=blue_shades[i]);

plt.xlabel("Time")
plt.ylabel("Glucose") #, labelpad=-2)
# plt.xticks(np.array([0, 25, 50, 75, 100]), ["0", "0.5π", "π", "1.5π", "2π"])
plt.grid(color="lightgray")

ax = plt.gca()
for sp in ['bottom', 'top', 'right', 'left']:
    ax.spines[sp].set_linewidth(1.5)

plt.tight_layout()

# plt.savefig(
#     OUTDIR + "figures/cr_big_y_train_examples.pdf",
#     transparent=True)


# %%
# SI: Latent space distributions
# ----------

# colors = ["mediumslateblue", "orangered"]
colors = ["gold"]*4
labels = ["L$_{1}$", "L$_{2}$", "L$_{3}$", "L$_{4}$"]

plt.figure(figsize=(7.25, 5.5), dpi=300)

for i in range(latent_space_nn.shape[1]):

    plt.subplot(2, 2, i+1)
    plt.title("Parameter: " + labels[i])
    plt.hist(latent_space_nn[:, i], color=colors[i], edgecolor="black",
             linewidth=2, alpha=0.8, bins=np.arange(-2.5, 2.7, 0.2))
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.ylim([0, 1800])

    ax = plt.gca()
    for sp in ['bottom', 'top', 'right', 'left']:
        ax.spines[sp].set_linewidth(1.5)

plt.tight_layout()

# # Save fig
# plt.savefig(
#     OUTDIR + "figures/cr_big_latent_par_dist.pdf",
#     transparent=True)

# %%
# SI: model architecture
# ---------
tf.keras.utils.plot_model(
    model,
    to_file=OUTDIR + "figures/cr_big_model_architecture.png",
    dpi=300,
    show_shapes=True,
    show_dtype=False,
    show_layer_names=False,
    rankdir="LR",
    show_layer_activations=True
)
