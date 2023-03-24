#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 11:46:30 2022

@author: botond antal

"""

import os
import pickle
import datetime
import h5py
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from scipy import optimize
from SALib.analyze import hdmr

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
model_name = "lb"
ver = 8

# Filepath
fp = OUTDIR + f"cr_{model_name}_ver-{ver}.pickle"

# Open pickle
with open(fp, 'rb') as handle:
    output = pickle.load(handle)

# Description
print(output["description"])

# Load data
x_train = output["x_train"]
y_train = output["y_train"]
x_val = output["x_val"]
y_val = output["y_val"]


# Load model
k = 4
r = 1
model_fp = OUTDIR + f"models/cr_{model_name}_ver-{ver}/k-{k}_r-{r}"
model = keras.models.load_model(model_fp)


# %%
# =============================================================================
# Global sensitiviy analysis
# =============================================================================


# Find bottleneck layer
k = 4 # Bottleneck dimension
bn_layer_ind = [model.layers[i].input_shape[1] for i in range(len(model.layers))].index(k)

# Extract encoder
encoder = Model(model.input, model.layers[bn_layer_ind-1].output)

# Compute latent values
latent_pars = encoder(x_train).numpy()

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
    res = hdmr.analyze(problem, x_train,  latent_pars[:, i], print_to_console=True)

    # Append to results
    SA_res.append(res)

# %%
# Plot GSA results on heatmap
# ------

# Number of input parameters
K_inp = 11

# Extract and transform GSA results
DLDI = np.array([SA_res[i]["Sa"][:K_inp] for i in range(k)]) #TODO: Sa

# Figure
plt.figure(figsize=(7, 2.9), dpi=300)

# Plot
plt.pcolormesh(DLDI, cmap="BuGn", vmax=1.5) #, vmax=0.5)

# Format
plt.gca().set_aspect("equal")
labels = ["c", "$\delta$", "g$_{Ca}$", "V$_{Ca}$", "g$_{K}$", "V$_{K}$", "g$_{Na}$",
          "V$_{Na}$", "a$_{ee}$", "a$_{ei}$", "r$_{NMDA}$"]
plt.xticks(np.arange(0.5, K_inp, 1), labels, rotation=0)
plt.yticks(np.arange(0.5, k, 1), [f"L$_{i+1}$" for i in range(k)])
plt.gca().invert_yaxis()
plt.xlabel("Input parameter")
plt.ylabel("Latent parameter")

# Colorbar
cbar = plt.colorbar(shrink=0.75, aspect=20*0.6)
cbar.set_ticks([0, 0.5, 1.0, 1.5])
cbar.set_label("Sensitivity", loc="center")

# Add spines
ax = plt.gca()
for sp in ['bottom', 'top', 'right', 'left']:
    ax.spines[sp].set_linewidth(1.4)
    ax.spines[sp].set_color("black")

# Format
plt.tight_layout()

# # Save fig
# plt.savefig(
#     OUTDIR + "figures/cr_lb_heatmap.pdf",
#     transparent=True)

# %%
# =============================================================================
# Perturbation fitting experiment
# =============================================================================

# Load perturbation samples
# -----

# Files
files = ['run_A.mat', 'run_B.mat']

# Collections for raw x and y data
x_data_p_raw_coll = []
y_data_p_raw_coll = []

# Iterate over all files
for file in tqdm(files, desc="files loaded"):

    # Filepath
    filepath = SRCDIR + f"lb_perturbation_runs/{file}"

    # Initialize dictionary for loaded data packet
    packet = {}

    # Load file
    f = h5py.File(filepath)

    # Itearte over items
    for l, v in f.items():

        # Unload
        packet[l] = np.array(v)

    # Append to collection
    x_data_p_raw_coll.append(packet["params"])
    y_data_p_raw_coll.append(packet["cc"])

# Convert collections to numpy arrays
x_data_p_raw = np.array(x_data_p_raw_coll)
y_data_p_raw = np.array(y_data_p_raw_coll)

# Status
print(f"\nFiles loaded. Shapes: X: {x_data_p_raw.shape}, Y: {y_data_p_raw.shape}")

# Initiate dictionary for fitting results
solutions = {}

# %%
# Fit a distinct items
# ---------

# Run this and the next section below for ind=0 and ind=1 cases before proceeding
# to subsequent sections

# Pick index of current item
ind = 0

# Prepare output space
y_obs_raw = y_data_p_raw[ind]

# Observed y
y_obs = y_obs_raw[np.triu_indices_from(y_obs_raw, k=1)]

# Display distribution of values
plt.hist(y_obs, color="royalblue", edgecolor="black", linewidth=3, alpha=.9)

# Print mean
print(f"Mean: {y_obs.mean():.2f}")

# Compute ground truth
# ------

# Extract raw x data
x_real_raw = x_data_p_raw[ind]

# Scale x to standard scale
x_scaler = output["input_scaler"]
x_real = x_scaler.transform(x_real_raw.reshape([1, -1]))[0]

# Compute ground truth in latent space
l_target = encoder(x_real.reshape([1, -1])).numpy()[0]

# Determine bounds for fitting
# ------

# Get full latent space
latent_train = encoder(x_train).numpy()

# Show histogram to assess range
# Colors
colors = ["teal", "orange", "indigo", "dodgerblue", "violet"]

# Plot each latent par separately
for i in range(k):
    plt.figure()
    plt.hist(latent_train[:, i], color=colors[i], edgecolor="black", linewidth=3, alpha=0.6,
              bins=np.arange(-2, 2.2, .05), label=f"l{i+1}")
    plt.axvline(l_target[i], color="red")
    plt.legend()

# Determined bounds
bounds_opt = np.array([
    [-2, 2.],
    [-2, 2.],
    [-2, 2.],
    [-2, 2.]
    ])


# %%
# Optimization
# -------

# Objective function
def compute_obj(pars_to_try, scale=True):
    """
    Function to minimize
    """

    # Convert proposed parameters to numpy array
    pars_to_try = np.array(pars_to_try)

    # Rescale parameters from cube
    if scale:

        pars_to_try_rescaled = pars_to_try*(bounds_opt[:, 1]-bounds_opt[:, 0]) + bounds_opt[:, 0]

        # Check if proposed values are within bounds
        if not all([pars_to_try[i] > bounds_kwg[i][0] and pars_to_try[i] < bounds_kwg[i][1] \
                    for i in range(len(pars_to_try))]):
            obj_val = np.inf
            return obj_val
    else:
        pars_to_try_rescaled = pars_to_try



    # Get trajectory based on input parameters
    y_model = decoder(np.array(pars_to_try_rescaled)[np.newaxis, :]).numpy()[0]

    # Get objective value (sum of squares)
    obj_val = np.sum((y_obs - y_model)**2)

    # Prevent nans being returned, return a large value instead
    if np.isnan(obj_val):
        raise
        # obj_val = np.inf

    # Save results
    global trials
    trials.append([pars_to_try_rescaled, obj_val])

    # Return
    return obj_val

# Extract decoder
decoder = Model(model.layers[bn_layer_ind].input, model.output)


# Trials to save intermediates
trials = []

# Hypercube bounds
bounds_kwg = [[0., 1.] for i in range(len(bounds_opt))]

# Run optimization
res = optimize.basinhopping(
    compute_obj,
    x0=[0.5 for i in range(len(bounds_opt))],
    stepsize=0.3,
    minimizer_kwargs={"method": "BFGS",
        "bounds": bounds_kwg}
    )

# Rescale results
pars_conv = res["x"]*(bounds_opt[:, 1]-bounds_opt[:, 0]) + bounds_opt[:, 0]

# Status
print(f"\n######\nFitting has finished. Number of evaluations: {len(trials)}\n######\n")
print(f"Ground truth: {l_target}")
print(f"Found optimum: {pars_conv}")
print(f"Objective value at ground truth: {compute_obj(l_target, scale=False):.2e}")
print(f"Objective value: {res['fun']:.2e}")

# Add solution to collection
solutions[ind] = pars_conv.copy()

# %%
# Save as pickle
# ------

output_dict = {}

# Meta
output_dict["version"] = ver
output_dict["k"] = k
output_dict["rep"] = r

# Observed data
output_dict["res"] = res
output_dict["trials"] = trials
output_dict["bounds_opt"] = bounds_opt
output_dict["ind"] = ind
output_dict["x_data_p_raw"] = x_data_p_raw
output_dict["y_data_p_raw"] = y_data_p_raw

# Pickle output object
extra = f"par-V_Na_ind-{ind}"
output_fname = OUTDIR + f"fitting/cr_lb_fitting_latent_" \
    f"ver-{ver}_k-{k}_r-{r}_{extra}.pickle"
# with open(output_fname, 'wb') as f:
#     pickle.dump(output_dict, f)


# %%
# Visualize results
# ------
# (run after have run fitting for all items and solution dictionary has been populated)

# Convert results do DataFrame
df = pd \
    .DataFrame(solutions) \
    .stack() \
    .reset_index() \
    .set_axis(["latent", "condition", "value"], axis=1) \
    .pipe(lambda df: df.assign(**{"latent": "L" + (df["latent"]+1).map(str)})) \
    .sort_values(by="condition")

# Settings
mpl.rcParams['hatch.linewidth'] = 1.0
xlims = [-1.2, 1.2]

# Legend items
legend_items = []

# Open figure
plt.figure(figsize=(2.3, 3.92), dpi=300)

# Plot
for i in range(k):

    # Get current latent par's values
    row_item = df.query(f'latent == "L{i+1}"')["value"].to_numpy()

    # Make background rectangle stripe
    rect = mpl.patches.Rectangle([xlims[0], i-0.2], width=xlims[1]-xlims[0],
                                 height=0.4, facecolor="gainsboro", zorder=2)
    plt.gca().add_patch(rect)

    # Make top rectangle
    rect = mpl.patches.Rectangle([row_item[0], i-0.3], width=row_item[1]-row_item[0],
                                 height=0.6, facecolor="orangered", alpha=1,
                                 hatch="////", zorder=3, label="shift")
    plt.gca().add_patch(rect)
    legend_items.append(rect)

    # Make marker rectangle 1
    rect = mpl.patches.Rectangle([row_item[0]-0.065, i-0.4], width=0.13,
                                 height=0.8, facecolor="yellowgreen", zorder=4,
                                 label="condition 1")
                                 # label=r"$V_{Na}$=0.48")
    plt.gca().add_patch(rect)
    legend_items.append(rect)

    # Make marker rectangle 2
    rect = mpl.patches.Rectangle([row_item[1]-0.065, i-0.4], width=0.13,
                                 height=0.8, facecolor="teal", zorder=5,
                                 label="condition 2")
                                 # label=r"$V_{Na}$=0.54")
    plt.gca().add_patch(rect)
    legend_items.append(rect)

# Format
# plt.gca().set_xticks([-.5, .5], minor=True)
plt.gca().set_xticks([-1, -.5, 0, .5, 1], ["-1", "-0.5", "0", "0.5", "1"])
plt.grid(axis="x", zorder=1, which="both", linestyle="--")
plt.yticks(np.arange(k), [f"L$_{i+1}$" for i in range(k)], fontsize=10)
plt.xticks(fontsize=10)
plt.xlim(xlims)
plt.ylim([-.5, 3.5])
plt.gca().invert_yaxis()
plt.xlabel("Inferred value", fontsize=10)
plt.ylabel("Latent parameter", fontsize=10)
plt.legend(handles=[legend_items[1], legend_items[2], legend_items[0]], fontsize=7,
                    loc=4)

plt.tight_layout()

# Add spines
ax = plt.gca()
for sp in ['bottom', 'top', 'right', 'left']:
    ax.spines[sp].set_linewidth(1.4)
    ax.spines[sp].set_color("black")

# # Save
# plt.savefig(OUTDIR + f"figures/cr_{model_name}_perturbation_fits.pdf",
#             transparent=True)


# %%
# Plot FCs
# ----

# Figure
plt.figure(figsize=(2.5, 3.3), dpi=300)

# Subplot 1
plt.subplot(2, 1, 1)

# Plot
plt.pcolormesh(y_data_p_raw[0], cmap="seismic", vmin=-1, vmax=1) #, vmax=0.5)

# Format
plt.gca().set_aspect("equal")
plt.xticks(np.arange(0, 80, 10), rotation=45, fontsize=7) #, labelpad=7)
plt.xticks([], [])
plt.yticks([], [])
plt.title(r"Condition 1: $\bf{V_{Na}=0.48}$", fontsize=9)
plt.gca().invert_yaxis()

ax = plt.gca()
for sp in ['bottom', 'top', 'right', 'left']:
    ax.spines[sp].set_linewidth(3)
    ax.spines[sp].set_color("yellowgreen")

# Subplot 2
plt.subplot(2, 1, 2)

# Plot
plt.pcolormesh(y_data_p_raw[1], cmap="seismic", vmin=-1, vmax=1) #, vmax=0.5)

# Format
plt.gca().set_aspect("equal")
plt.xticks([], [])
plt.yticks([], [])
plt.gca().tick_params(axis='x', which='major', pad=2)
plt.title(r"Condition 2: $\bf{V_{Na}=0.54}$", fontsize=9)
plt.gca().invert_yaxis()

ax = plt.gca()
for sp in ['bottom', 'top', 'right', 'left']:
    ax.spines[sp].set_linewidth(3)
    ax.spines[sp].set_color("teal")

# Tightlayout
plt.tight_layout(pad=0.2)

# # Save
# plt.savefig(OUTDIR + f"figures/cr_{model_name}_perturbation_FCs.pdf",
#             transparent=True)

# # Colorbar [x]
# # ---

plt.figure(figsize=(1.55, 0.7), dpi=300)
ax = plt.gcf().add_axes([0.05, 0.80, 0.9, 0.1])
cb = mpl.colorbar.ColorbarBase(ax, norm=mpl.colors.Normalize(-1, 1),
                                orientation='horizontal', cmap='seismic',
                                label="correlation")
cb.set_ticks(np.arange(-1, 1.2, 0.5))
cb.ax.tick_params(labelsize=8)

# # Save
# plt.savefig(OUTDIR + "figures/FC_colorbar.pdf",
#             transparent=True)

# %%
# Single FC for illustration
# ---------

# Figure
plt.figure(figsize=(2.4, 2.4), dpi=300)

# Plot
plt.pcolormesh(y_data_p_raw[0], cmap="seismic", vmin=-1, vmax=1) #, vmax=0.5)

# Format
plt.gca().set_aspect("equal")
# plt.xticks([], [])
# plt.yticks([], [])
plt.xticks(np.arange(0, 80, 10), rotation=45, fontsize=7)
plt.yticks(np.arange(0, 80, 10), rotation=0, fontsize=7)
plt.gca().tick_params(axis='x', which='major', pad=2)
plt.gca().invert_yaxis()

ax = plt.gca()
for sp in ['bottom', 'top', 'right', 'left']:
    ax.spines[sp].set_linewidth(1.4)
    ax.spines[sp].set_color("black")

# # Save
# plt.savefig(OUTDIR + "figures/FC_example.pdf",
#             transparent=True)


# %%
# =============================================================================
# Misc plots
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

# %%
# SI: input parameter distritubions
# --------

# colors = ["dodgerblue", "teal", "gold", "crimson", "lightseagreen",
#           "aqua", "peru", "navy", "deeppink", "darkorange", "limegreen"]
colors = ["crimson"]*11
labels = ["c", "$\delta$", "g$_{Ca}$", "V$_{Ca}$", "g$_{K}$", "V$_{K}$", "g$_{Na}$",
          "V$_{Na}$", "a$_{ee}$", "a$_{ei}$", "r$_{NMDA}$"]


plt.figure(figsize=(7.25, 7), dpi=300)

for i in range(x_train.shape[1]):

    plt.subplot(3, 4, i+1)
    plt.title("Parameter: " + labels[i])
    plt.hist(x_train[:, i], color=colors[i], edgecolor="black",
             linewidth=1.5, alpha=0.8, bins=np.arange(0., 1.1, 0.1))
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.ylim([0, 600])

    ax = plt.gca()
    for sp in ['bottom', 'top', 'right', 'left']:
        ax.spines[sp].set_linewidth(1.5)

plt.tight_layout()

# # Save fig
# plt.savefig(
#     OUTDIR + "figures/cr_lb_x_train_dist.pdf",
#     transparent=True)

# %%
# SI: Output space data examples
# ----------

# Inds
plt.figure(figsize=(6.5, 4), dpi=300)

# Plot matrix
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.pcolormesh(triu_to_full_matrix((y_train[5+i])),
                   vmin=-1, vmax=1, cmap="seismic")
    plt.gca().set_aspect("equal")

    plt.xticks(np.arange(0, 80, 10), rotation=45, fontsize=9) #, labelpad=7)
    plt.yticks(np.arange(0, 80, 10), rotation=0, fontsize=9) #, labelpad=7)

    ax = plt.gca()
    for sp in ['bottom', 'top', 'right', 'left']:
        ax.spines[sp].set_linewidth(1.5)

plt.tight_layout()

# # Save
# plt.savefig(
#     OUTDIR + "figures/cr_lb_y_train_examples.pdf",
#     transparent=True)

# %%
# Colorbar
plt.figure(figsize=(1, 4), dpi=300)
ax = plt.gcf().add_axes([0.03, 0.03, 0.25, 0.9])
cb = mpl.colorbar.ColorbarBase(ax, norm=mpl.colors.Normalize(-1, 1),
                               orientation='vertical', cmap='seismic',
                               label="correlation")
cb.set_ticks(np.arange(-1, 1.2, 0.5))
cb.ax.tick_params(labelsize=8)

# plt.tight_layout()

# # Save
# plt.savefig(OUTDIR + "figures/FC_SI_colorbar_vertical.pdf",
#             transparent=True)

# %%
# SI: Latent space distributions
# ----------

# colors = ["mediumslateblue", "orangered", "deepskyblue", "limegreen"]
colors = ["gold"]*4
labels = ["L$_{1}$", "L$_{2}$", "L$_{3}$", "L$_{4}$"]

plt.figure(figsize=(7.25, 5.0), dpi=300)

for i in range(latent_pars.shape[1]):

    plt.subplot(2, 2, i+1)
    plt.title("Parameter: " + labels[i])
    plt.hist(latent_pars[:, i], color=colors[i], edgecolor="black",
             linewidth=1, alpha=0.8, bins=np.arange(-2, 2.1, 0.1))
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.ylim([0, 1100])

    ax = plt.gca()
    for sp in ['bottom', 'top', 'right', 'left']:
        ax.spines[sp].set_linewidth(1.5)


plt.tight_layout()

# # Save fig
# plt.savefig(
#     OUTDIR + "figures/cr_lb_latent_par_dist.pdf",
#     transparent=True)

# %%
# SI: model architecture
# ---------

tf.keras.utils.plot_model(
    model,
    to_file=OUTDIR + "figures/cr_lb_model_architecture.png",
    dpi=300,
    show_shapes=True,
    show_dtype=False,
    show_layer_names=False,
    rankdir="LR",
    show_layer_activations=True
)
