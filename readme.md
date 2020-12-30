# Cross Correlation of two Ohrnstein Uhlenbeck Processes with Delayed Noise
This repository contains code used to simulate an ensemble of two Ohrnstein Uhlenbeck processes which are connected by their noise.

- report - Contains latex code and the resulting pdf explainging the simulation and its results.
- src - Source Code of the Simulation


## Setup and Simulation Start
A working Python3 environment is assumed.

####  Dependency Installation
``pip3 install -r requirements.txt``

#### Simulation Start
Parameter set selection can be done in the function `calc_and_save()` in `src/main_multiple_runs.py`.

Start simulation:
```python3 src/main_multiple_runs.py```

#### Result Analysation and Plotting
The simulation results are saved in the folder `results`.
Plots are created manually in the Jupyter Notebooks in `src/notebooks`.

Start Jupyter lab to use notebooks:
```python3 jupyter lab```

Relevant notebooks are
- noise_validation.ipynb
- overview.ipynb
- overview_asymmetric_tau.ipynb
- overview_asymmetric_gamma.ipynb
