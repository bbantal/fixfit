# Overview:
FixFit is a tool for parameter compression to solve the inverse problem in models with redundant parameters.

This repository contains code for FixFit, structured in a way to reproduce all results in our relevant article where we describe FixFit in detail:
"FixFit: using parameter-compression to solve the inverse problem in overdetermined models" - B Antal, A Chesebro, H Strey, L Mujica-Parodi, C Weistuch

All necessary inputs (simulated data), intermediate outputs, and final outputs are included in this repository.

# Content:
- `data/`: simulated data used as inputs
- `results/`: intermediate outputs and figures. NOTE this folder contains large files (>100MB) and requires extra steps to download to your system! There are two options to download these files:
    - A) if you have Git LFS established on your system you can simply clone this whole repository
    - B) if you don't have Git LFS, you can download this folder through this link, simply copy its contents to the cloned repository: https://drive.google.com/drive/folders/1cRglXICpqj-D_3F1qvMaUWibLOF87VCU?usp=sharing
- `tools/`: code, see specifics below
- code and output files contain the prefixes "`tm2_`", "`big_`", and "`lb_`". These refer to the three model examples presented in the manuscript. "tm2" stands for the Kepler model, "big" for the beta-insulin-glucose regulation model, and "lb" for Larter-Breakspear brain network model.
- `tm2_neural_network.py`: performs the neural network approximation at various bottleneck dimensions for the Kepler model
- `tm2_bottleneck_analysis.py`: performs the analysis of validation error at various bottlenecks for the Kepler model to identify the underlying complexity of the model and the corresponding latent parameter representation
- `tm2_analysis.py`: performs downstream analyses at a selected latent parameter representation for the Kepler model. These include global sensitivity analysis with respect to input parameters and global fitting in input and latent parameter spaces.
- `big_simulate_training_data.jl`: simulates training data for the glucose regulation model
- `big_neural_network.py`: performs the neural network approximation at various bottleneck dimensions for the glucose regulation model
- `big_bottleneck_analysis.py`: performs the analysis of validation error at various bottlenecks dimensions for the glucose regulation model to identify the underlying complexity of the model and the corresponding latent parameter representation
- `big_analysis.py`: performs downstream analyses at a selected latent parameter representation for the glucose regulation model. These include global sensitivity analysis with respect to input parameters and global fitting in latent parameter space to real observed data
- `structural_identifiability_big.jl`: performs structural identifiability analysis for the glucose regulation model using a symbolic approach latent parameter space.
- `lb_simulations_fixfit`: contains scripts used to simulate training data for the Larter-Breakspear model example
- `lb_neural_network.py`: performs the neural network approximation at various bottleneck dimensions for the Larter-Breakspear model
- `lb_bottleneck_analysis.py`: performs the analysis of validation error at various bottlenecks dimensions for the Larter-Breakspear model to identify the underlying complexity of the model and the corresponding latent parameter representation
- `lb_analysis.py`: performs downstream analyses at a selected latent parameter representation for the Larter-Breakspear model. These include global sensitivity analysis with respect to input parameters and global fitting in 

# Instructions:
- To run the code, first create a new environment using pip or conda
- Run `install_dependencies.sh`, this will set up all dependencies for the code (for exact versions, see `environment.yml`)
- Each analysis step can be run separately as all intermediate inputs are provided by default
- If you wish to test the whole pipeline from end to end, you must uncomment all code snippets that are responsible for saving script outputs. By default no outputs are saved, the default intermediates are retained and used as inputs for subsequent steps
- It is recommended to run the code in an integrated development environment (IDE) which enables running code sections as separate cells and displaying figures interactively (spyder, vscode).
- Note that `tm2_neural_network.py`, `big_neural_network.py`, and `lb_neural_network.py` require a long time to run (on the scale of several hours). The subsequent scripts and analyses can be run within minutes. Therefore, as an alternative, one can skip the neural network training phase, use the already available training outputs and start at the stage of bottleneck analysis.
