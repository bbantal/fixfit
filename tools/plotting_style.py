import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

plt.style.use("default")
#plt.style.use("ggplot")
#sns.set_style("whitegrid")

fs=1.2  # Fontsize
lw=2.0   # Linewidth

plot_pars = [fs, lw]

# Stylesheet
plt.rcParams['xtick.color'] = "black"
plt.rcParams['ytick.color'] = "black"
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['xtick.major.width'] = 0.5*lw
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['ytick.major.width'] = 0.5*lw
plt.rcParams['xtick.labelsize']=8*fs
plt.rcParams['ytick.labelsize']=8*fs
plt.rcParams['text.color'] = "black"
plt.rcParams['axes.labelcolor'] = "black"
plt.rcParams["font.weight"] = "regular"
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8*fs
plt.rcParams['axes.labelsize']=9*fs
plt.rcParams['axes.labelweight'] = "regular"
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 3
plt.rcParams['legend.fontsize'] = 9*fs
plt.rcParams['legend.title_fontsize'] = 9*fs
# plt.rcParams['text.latex.preamble'] = r'\math'
plt.rcParams['figure.titlesize'] = 10*fs
plt.rcParams['figure.titleweight'] = "regular"
plt.rcParams['axes.titlesize'] = 9*fs
plt.rcParams['axes.titleweight'] = "regular"
#plt.rcParams['axes.axisbelow'] = True