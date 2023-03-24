#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 00:13:22 2022

@author: botond antal

"""


import os
import pickle
import warnings
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from SALib.analyze import hdmr

import plotting_style

print("TensorFlow version:", tf.__version__)

warnings.filterwarnings('ignore')

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
model_name = "tm2"
ver = 35
k = 2
r = 3

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
G = output["G"]
bounds = output["bounds"]


# Load model
model_fp = OUTDIR + f"models/cr_{model_name}_ver-{ver}/k-{k}_r-{r}"
model = keras.models.load_model(model_fp)

# %%
# =============================================================================
# Global sensitivity analysis (GSA)
# =============================================================================

# Number of input parameters
K_inp = 4

# Find bottleneck layer
k = 2 # Bottleneck dimension
bn_layer_ind = [model.layers[i].input_shape[1] for i in range(len(model.layers))].index(k)
# (Bottleneck needs to be the first layer with k dimension!)

# Extract encoder from fitted model
encoder = Model(model.input, model.layers[bn_layer_ind-1].output)

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
plt.figure(figsize=(2.5, 2.8), dpi=300)

# Plot
plt.pcolormesh(DLDI.T, cmap="BuGn", vmax=1.5) #, vmax=0.5)

# Format
plt.gca().set_aspect("equal")
labels = ["m${_1}$", "m${_2}$", "r${_0}$", "w${_0}$"]
plt.xticks(np.arange(0.5, k, 1), [f"L$_{i+1}$" for i in range(k)])
plt.yticks(np.arange(0.5, K_inp, 1), labels, rotation=0)
plt.gca().invert_yaxis()
plt.xlabel("Latent parameter")
plt.ylabel("Input parameter")

# Colorbar
cbar = plt.colorbar(shrink=1, aspect=20*0.7, label="Sensitivity")
cbar.set_ticks(np.arange(0, 1.6, 0.5))
# cbar.set_ticklabels([])

# Add spines
ax = plt.gca()
for sp in ['bottom', 'top', 'right', 'left']:
    ax.spines[sp].set_linewidth(1.4)
    ax.spines[sp].set_color("black")

plt.tight_layout()

# # Save fig
# plt.savefig(
#     OUTDIR + "figures/cr_tm2_heatmap.pdf",
#     transparent=True)

# %%
# =============================================================================
# Global optimization
# =============================================================================

# Function to compute composite kepler terms
# ------

def kepler_terms(pars):
    """
    This function produces planetary trajectory parameters ec and p
    given input parameters m1, m2, r0, and w0

    """

    # Extract values
    m1, m2, r0, w0 = pars

    # Compute intermediate terms
    L = m1 * r0**2 * w0  # Angular momentum, source: https://en.wikipedia.org/wiki/Kepler_problem (towards the top)
    E = 0.5*m1*(w0*r0)**2 - G*m1*m2/r0  # Total energy; source: https://en.wikipedia.org/wiki/Specific_orbital_energy (top equation first part)
    a = -G*m1*m2/(2*E)  # Major axis; source: https://en.wikipedia.org/wiki/Specific_orbital_energy (top equation second part)

    # Compute ellipse properties
    ec = np.sqrt(1 + 2*E*L**2/(m1*(G*m1*m2)**2))  # Eccentricity; source: https://en.wikipedia.org/wiki/Kepler_problem (towards the bottom)
    p = a*(1-ec**2)  # Semi-latus rectum; source: https://en.wikipedia.org/wiki/Ellipse#Semi-latus_rectum (under semi-latus rectum)

    # Return
    return ec, p

# Determine vmin and vmax used for scaling
# -----

# Compute ec, p terms
ec, p = kepler_terms(np.concatenate([x_train, x_val]).T)

# Number of theta increments
n_theta_inc = 100

# Generate theta grid (1d)
theta = np.linspace(0, 2*np.pi, n_theta_inc)

# Compute r as a function of theta at given p and ec
r = p[:, None]/(1+ec[:, None]*np.cos(theta[None, :]))

# Compute vmin, vmax
vmin = np.log(r).min()
vmax = np.log(r).max()

# Model function
# -------

def get_trajectory(pars):
    """
    This function produces planetary trajectories (=output space) from input
    parameters m1, m2, r0, w0

    """
    # Make copy of pars
    pars_local = pars.copy()

    # Flip w0: TODO
    # pars_local[3] = 1/pars[3]

    # Compute trajectory parameters
    ec, p = kepler_terms(pars_local)

    # Drop if trajectory is not elliptical (characterized by ec)
    # m1, m2, r0, w0 = pars_local
    # ec_term = r0**3*w0**2/(G*m2) - 1
    # print(ec_term)
    # if (ec_term>1) | (ec_term<0.5):
    print(ec)
    if (ec>0.95): #| (ec<0.7):
        # Return nans in this case (of same shape as output)
        return np.full(100, np.nan)

    # Number of theta increments
    n_theta_inc = 100

    # Generate theta grid (1d)
    theta = np.linspace(0, 2*np.pi, n_theta_inc)

    # Compute r as a function of theta at given p and ec
    r = p/(1+ec*np.cos(theta))

    # Scale outputs with log and vmin/vmax to match original output space
    r_scaled = (np.log(r) - vmin)/(vmax-vmin)

    # Return
    return r_scaled


# %%
# ====================
# Optimization in native input space
# =====================

# Objective function
def compute_obj(pars_to_try):
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
        data_model = get_trajectory(pars_to_try_rescaled)

        # Get objective value (sum of squares)
        obj_val = np.sum((data_obs - data_model)**2)

    # If input pars are not interpetable
    if np.isnan(obj_val):
        obj_val = np.inf

    # Print
    # print(pars_to_try, pars_to_try_rescaled, obj_val)

    # Save results
    global trials
    trials.append([pars_to_try_rescaled, obj_val])

    # Return
    return obj_val

# Pick observed data to fit
ind = 1
data_obs = y_train[ind]

# Scaling
bounds_opt = np.array(
    list(bounds.values())
    )
bounds_kwg = [[0., 1.] for i in range(len(bounds_opt))]

# Trials to save intermediates
trials = []

# Run optimization
res = optimize.basinhopping(
    compute_obj,
    x0=[0.5 for i in range(len(bounds_opt))],
    stepsize=0.2,
    minimizer_kwargs={"method": "BFGS",
                      "bounds": bounds_kwg,
                      "tol": 1e-2}
    )

# Rescale results
pars_conv = res["x"]*(bounds_opt[:, 1]-bounds_opt[:, 0]) + bounds_opt[:, 0]

# Print results
print(x_train[ind], "\n", pars_conv)

# %%
# Inspect convergence of optimization
# -----

lw=2

# Status
print(f"Number of evaluations: {len(trials)}")
print(f"Objective value: {res['fun']:.1e}")

# Extract trials data
vals = np.array([trial[0] for trial in trials])
objs = np.log([trial[1] for trial in trials])

# Figure
plt.figure(figsize=(2.3, 2.8), dpi=300)

# Plot trials data
plt.scatter(vals[:, 1], vals[:, 3], c=objs, cmap="jet", s=2, vmin=-10, vmax=10)

# Add misc elements
plt.axhline(x_train[ind][3], color="k", linestyle="--", lw=1.)
plt.axvline(x_train[ind][1], color="k", linestyle="--", lw=1.)

# Format
plt.xticks(np.arange(0, 1.2, 0.2))
plt.yticks(np.arange(0, 1.2, 0.2))
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel("m$_{2}$ [kg]")
plt.ylabel("w$_{0}$ [day$^{-1}$]");

# Add spines
ax = plt.gca()
for sp in ['bottom', 'top', 'right', 'left']:
    ax.spines[sp].set_linewidth(1.4)
    ax.spines[sp].set_color("black")

plt.tight_layout()

# # Save fig
# plt.savefig(
#     OUTDIR + "figures/cr_tm2_fitting-input.pdf",
#     transparent=True)

# %%
# Save
# ------

output_dict = {}

# Meta
output_dict["version"] = ver
output_dict["version"] = k
output_dict["rep"] = r

# Observed data
output_dict["res"] = res
output_dict["trials"] = trials
output_dict["bounds_opt"] = bounds_opt
output_dict["ind"] = ind

# # Pickle output object
# output_fname = OUTDIR + f"fitting/cr_tm2_fitting_native_ver-{ver}_k-{k}_r-{r}.pickle"
# with open(output_fname, 'wb') as f:
#     pickle.dump(output_dict, f)


# %%
# ====================
# Optimization in latent space
# =====================
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

    else:
        pars_to_try_rescaled = pars_to_try

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

# Inspect distribution of values of latent parameters
plt.hist(latent_space_nn[:, 0])
plt.hist(latent_space_nn[:, 1])

# Get decoder
decoder = Model(model.layers[bn_layer_ind].input, model.output)

# Pick observed data to fit
data_obs = y_train[ind]

# Print ground truth input parameters
print(latent_space_nn[ind])

# Scaling
bounds_opt = np.array([
    [-0.2, 0.5],
    [-0.2, 0.5]
    ])

bounds_kwg = [[0., 1.] for i in range(len(bounds_opt))]

# Trials to save intermediates
trials = []

# Run optimization
res = optimize.basinhopping(
    compute_obj,
    x0=[0.5 for i in range(len(bounds_opt))],
    stepsize=0.2,
    minimizer_kwargs={"method": "BFGS",
        "bounds": bounds_kwg}
    )

# Rescale results
pars_conv = res["x"]*(bounds_opt[:, 1]-bounds_opt[:, 0]) + bounds_opt[:, 0]

# Print results
print(latent_space_nn[ind], "\n", pars_conv)

# %%
# Inspect convergence of optimization
# -----

# Status
print(f"Number of evaluations: {len(trials)}")
print(f"Ground truth: {latent_space_nn[ind]}")
print(f"Found optimum: {pars_conv}")
print(f"Objective value with ground truth: {compute_obj(latent_space_nn[ind], scale=False):.1e}")
print(f"Objective value: {res['fun']:.1e}")

# Extract trials data
vals = np.array([trial[0] for trial in trials])
objs = np.log([trial[1] for trial in trials])

# Figure
plt.figure(figsize=(2.9, 2.8), dpi=300)

# Plot trials data
plt.scatter(vals[:, 0], vals[:, 1], c=objs, cmap="jet", s=2, vmin=-10, vmax=10)

# Misc elements
plt.colorbar(label="Goodness of fit (log-scale)")

np.argwhere([objs == objs.min()])

plt.axhline(latent_space_nn[ind][1], color="k", linestyle="--", lw=1., zorder=1)
plt.axvline(latent_space_nn[ind][0], color="k", linestyle="--", lw=1., zorder=1)

plt.xlabel("L${_1}$")
plt.ylabel("L${_2}$", labelpad=-5)

plt.xticks(np.arange(-0.2, 0.5, 0.2))
plt.yticks(np.arange(-0.2, 0.5, 0.2))

# Add spines
ax = plt.gca()
for sp in ['bottom', 'top', 'right', 'left']:
    ax.spines[sp].set_linewidth(1.4)
    ax.spines[sp].set_color("black")

plt.tight_layout()

# # Save fig
# plt.savefig(
#     OUTDIR + "figures/cr_tm2_fitting-latent.pdf",
#     transparent=True)

# %%
# Save
# ------

output_dict = {}

# Meta
output_dict["version"] = ver
output_dict["version"] = k
output_dict["rep"] = r

# Observed data
output_dict["res"] = res
output_dict["trials"] = trials
output_dict["bounds_opt"] = bounds_opt
output_dict["ind"] = ind

# # Pickle output object
# output_fname = OUTDIR + f"fitting/cr_tm2_fitting_latent_ver-{ver}_k-{k}_r-{r}.pickle"
# with open(output_fname, 'wb') as f:
#     pickle.dump(output_dict, f)

# %%
# =============================================================================
# Misc plots
# =============================================================================

# Main: output space example
# -------

ind = 2

plt.figure(figsize=(2, 2), dpi=200)
pols = np.exp(y_train[ind])

plt.plot(pols, color="dodgerblue")

# Format
plt.xlabel("θ")
plt.ylabel("r(θ)")
plt.xticks(np.array([0, 50, 100]), ["0", "π", "2π"])

plt.tight_layout()

# # Save fig
# plt.savefig(
#     OUTDIR + "figures/cr_tm2_pol-example.pdf",
#     transparent=True)

# %%
# SI: input parameter distritubions
# --------

# colors = ["dodgerblue", "teal", "gold", "crimson"]
colors = ["crimson"]*4
labels = ["m$_{1}$", "m$_{2}$", "r$_{0}$", "w$_{0}$"]

plt.figure(figsize=(7.25, 4), dpi=300)

for i in range(x_train.shape[1]):

    plt.subplot(2, 2, i+1)
    plt.title("Parameter: " + labels[i])
    plt.hist(x_train[:, i], color=colors[i], edgecolor="black",
             linewidth=1.5, alpha=0.8, bins=np.arange(0.1, 1.1, 0.05))
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.xlim([0, 1])
    plt.ylim([0, 250])

    ax = plt.gca()
    for sp in ['bottom', 'top', 'right', 'left']:
        ax.spines[sp].set_linewidth(1.5)

plt.tight_layout()

# # Save fig
# plt.savefig(
#     OUTDIR + "figures/cr_tm2_x_train_dist.pdf",
#     transparent=True)

# %%
# SI: Output space data examples
# ----------

plt.figure(figsize=(3.625, 2.5), dpi=300)
plt.plot(y_train[:20, :].T, lw=1, color="dodgerblue");

# Format
plt.xlabel("θ")
plt.ylabel("r(θ) (scaled)")
plt.xticks(np.array([0, 25, 50, 75, 100]), ["0", "0.5π", "π", "1.5π", "2π"])
plt.grid(color="lightgray")

ax = plt.gca()
for sp in ['bottom', 'top', 'right', 'left']:
    ax.spines[sp].set_linewidth(1.5)

plt.tight_layout()

# plt.savefig(
#     OUTDIR + "figures/cr_tm2_y_train_examples.pdf",
#     transparent=True)

# %%
# SI: Latent space distributions
# ----------

# colors = ["mediumslateblue", "orangered"]
colors = ["gold"]*2
labels = ["L$_{1}$", "L$_{2}$"]

plt.figure(figsize=(7.25, 2.5), dpi=300)

for i in range(latent_space_nn.shape[1]):

    plt.subplot(1, 2, i+1)
    plt.title("Parameter: " + labels[i])
    plt.hist(latent_space_nn[:, i], color=colors[i], edgecolor="black",
             linewidth=2, alpha=0.8, bins=np.arange(-0.3, 0.55, 0.05))
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.ylim([0, 600])

    ax = plt.gca()
    for sp in ['bottom', 'top', 'right', 'left']:
        ax.spines[sp].set_linewidth(1.5)

plt.tight_layout()

# # Save fig
# plt.savefig(
#     OUTDIR + "figures/cr_tm2_latent_par_dist.pdf",
#     transparent=True)

# %%
# SI: model architecture
# ---------

tf.keras.utils.plot_model(
    model,
    to_file=OUTDIR + "figures/cr_tm2_model_architecture.png",
    dpi=300,
    show_shapes=True,
    show_dtype=False,
    show_layer_names=False,
    rankdir="LR",
    show_layer_activations=True
)

