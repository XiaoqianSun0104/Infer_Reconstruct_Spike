# Introduction
In this project, I translated a MATLAB implementation from Lütcke, Henry, et al. into Python, incorporating modifications to enhance usability, flexibility, and maintainability. The original MATLAB code served as a foundation for inferring spike trains from calcium imaging data and reconstructing calcium traces based on spkie dynamics, and my translation restructures it into an object-oriented design within a well-organized Python package.
The original MATLAB can be found in the supplementary material with the publication - https://doi.org/10.3389/fncir.2013.00201


## Overview

### Peeling 
- `peelParams(Object)` prepares necessary parameter space for peel original
- `peelReconstr(Object)` infers spkTrain from input dff traces

### Simulation
- `simuParams(Object)` prepares necessary parameter space for dff simulation
- `Simulation(Object)` inverse process of Peeling, convolving typical Ca2+ event to spkTimes and generate F/noisyF/downsampled dff


## useCase
In this folder, I provided some codes for exploring simulation & peeling
- `1.1-poissonSpikes_saveArray.ipynb` use `Simulation(Object)` to generate TIF 
- `1.2-poissonSpikes_CaImAn.ipynb` apply CaImAN on the generated TIF
- `1.3-poissonSpikes_peelInference.ipynb` apply Peeling algorithm on the simulated calcium trace to infer spikes
- `1.4-poissonSpikes_OASIS.ipynb` apply Oasis on dff to infer spikes and compare the result with the inference using peeling algorithm
- `1.5-peel_onRealData.ipynb` apply Peeling on real experimental data
- `1.6-spkTrain_2_ca2Traces.ipynb` reconstruct F/dff from spkTrain

## Data
Contains data for jupyter notebooks in useCase folder
