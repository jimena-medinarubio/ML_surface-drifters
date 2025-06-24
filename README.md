# ML-driven drifter data analysis

[![CCDS Project template](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)

Data-driven analysis of surface drifters' trajectories to infer the predominant forcing mechanisms that drive their transport & predict their paths.
Data: https://doi.org/10.5281/zenodo.14198921

Related publication: "Using surface drifters to characterise near-surface ocean dynamics in the southern North Sea: a data-driven approach"

Author: Jimena Medina Rubio (PhD Candidate)

## Project Organization

```
├── LICENSE            <- Open-source license
├── data
│   ├── external       <- hydrodynamic 
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- original drifter trajectories (available at https://zenodo.org/records/14198921) 
│
├── models             <- trained random forest & support vector regression models
│
│
├── references         <- variable dictionaries & labels for plotting
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
└── ml_driven_drifter_data_analysis   <- Source code for use in this project.
    │
    ├── Plotting
    │    ├──  domain_figures.py                        
    │    └──  flipping_index.py
    │                        
    ├──supplementary_material
    │    ├──  autocorrelation_times.py
    │    ├──  sensitivity_flipping_index.py                        
    │    └──  drifter_measurements_error.py
    │
    ├── preprocessing_drifter_data.py               <- data cleaning & formatting into xarray DataTree
    │
    ├── processing_drifter_data.py                  <- calculation of velocities, residual velocities & flipping index
    │
    ├── spectral_analysis_drifter_data.py           <- FFT & Morlet analysis of drifter velocities
    │
    ├── dataset.py                                  <- interpolation of hydrodynamic & atmospheric data to drifters' coordinates
    │
    ├── features.py                                 <- transformation of interpolated variables to construct feature matrix
    │
    ├── modeling                
        ├──  train.py                        <- functions to train models
        ├──  fitting_models                  <- execution of training functions for each model
        │    ├── linear_regression.py
        │    ├── RandomForest.py    
        │    ├── SVR.py
        │    ├── RandomForest_FI_fit.py
        │    └── RandomForest_tests.py
        │
        ├──  model_agnostics.py
        │    ├── PFI.py
        │    ├── calculate_ALE.py    
        │    ├── model_agnostics_plots.py
        │    └── ALE_plots.py
        │          
        ├──  prediction
        │    ├── prediction_functions.py
        │    ├── predict_linear_regression.py
        │    ├── prediction_ML.py    
        │    └── predict_linear_regression_sigmoid.py
        │
        ├──  statistics_prediction.py
        │  
        ├──  decomposing_currents.py
           
```

--------

