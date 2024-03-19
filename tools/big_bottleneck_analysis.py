"""
Created on Feb 18 2024

@author: bbantal

"""

# %%

import os
import dill as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import plotting_style

# %%
# =============================================================================
# Setup
# =============================================================================

# Rcparams
plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = "DejaVu Sans"

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/"

# Version
model_name = "big"
ver = 5

# Filepath
fp = OUTDIR + f"cr_{model_name}_ver-{ver}.pickle"

# Open pickle
with open(fp, 'rb') as handle:
    output = pickle.load(handle)

print(output["description"])

kset = output["k_vals"]
rset = np.arange(output["n_reps"])
epochs = output["epochs"]

# %%
# =============================================================================
# Analysis
# =============================================================================

# Reshape results
# -----

# Get k values again
kset = output["k_vals"]

# Validation results
test_loss = np.array(
    [output["loss_coll"][i]["test"] for i in range(len(output["loss_coll"]))]
    ) \
    .reshape((len(kset), len(rset), -1))

# Get final losses and transform into dataframes
# --------
df_final_test = pd \
    .DataFrame(np.array([min(test_loss[i1, i2, 0]) \
          for i1 in range(test_loss.shape[0]) for i2 in range(test_loss.shape[1])]) \
        .reshape(len(kset), len(rset))) \
    .rename_axis("k").reset_index() \
    .melt(id_vars="k", var_name="rep", value_name="final loss") \
    .pipe(lambda df: df.assign(**{"k": df["k"]+1}))

# Remove k>6
kmax=6
kset = np.arange(1, kmax+1)
df_final_test = df_final_test.query(f"k < {kmax+1}")

# Log transform for plotting
df_final_test["final loss"] = np.log(df_final_test["final loss"])

# Rank replicates according to loss
df_final_test.sort_values(by=["k", "final loss"]).query("k==4")

# %%
# Plot bottleneck results
# ----

lw = 3
ms = 40

# Open figure
plt.figure(figsize=(3.05, 3.3), dpi=300)

# Plot
sns.lineplot(data=df_final_test, x="k", y="final loss", err_style="bars",
              err_kws={"capsize": 3*lw, "capthick": lw, "elinewidth": lw},
              marker="o", markersize=5*lw, ci="sd", lw=lw, zorder=2,
              color="teal", label=""); #"MSE (mean/sd)");

# Show individual points
df_final_val_wiggled = df_final_test.assign(**{"k": df_final_test["k"]+np.random.uniform(-.1, .1, df_final_test.shape[0])})
sns.scatterplot(data=df_final_val_wiggled, s=ms*2, x="k", y="final loss", color="orangered", zorder=3,
                label="", edgecolor="black", linewidth=lw/2);

# Plot chosen k
plt.axvline(4, c="black", linestyle="--", zorder=1, label="Identified\ncomplexity")

# Format
plt.legend(fontsize=9)
plt.title("")
plt.xlabel("Bottleneck dimension (k)")
plt.ylabel("Log mean squared validation error")
plt.xticks(kset)

# Add spines
ax = plt.gca()
for sp in ['bottom', 'top', 'right', 'left']:
    ax.spines[sp].set_linewidth(1.4)
    ax.spines[sp].set_color("black")

plt.tight_layout()

# # Save
# plt.savefig(OUTDIR + f"figures/cr_{model_name}_k.pdf",
#             transparent=True)

# %%
# Plot all learning curves
# --------

# Colors
n = len(kset)
cmap = plt.cm.viridis_r(np.linspace(0, 1, n))

# Open figure
plt.figure(figsize=(3.625, 3), dpi=300)

# Plot
for k in kset:
    for r in [3]:
        x = np.arange(len(train_loss[k-1, r, 0]))[::1]
        y = test_loss[k-1, r, 0][::1]
        plt.plot(x, y, color=cmap[k-1],
                 lw=1, label=f"k={k}" if r==3 else None,
                 alpha=1)

# Format
plt.legend(fontsize=7, title="Bottleneck\ndimension", title_fontsize=8, ncol=2)
plt.xlabel("Epoch #") #, fontsize=7)
plt.ylabel("Validation error (MSE)") #, fontsize=7)
plt.ylim([0e-5, 5e-4]);
plt.grid(color="lightgray")

plt.gca().yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.0e}"))

plt.tight_layout()

# # Save
# plt.savefig(OUTDIR + f"figures/cr_{model_name}_training_curves.pdf",
#             transparent=True)
