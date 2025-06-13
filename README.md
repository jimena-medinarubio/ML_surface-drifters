# ML-driven drifter data analysis <a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Data-driven analysis of surface drifters' trajectories to infer the predominant forcing mechanisms that drive their transport & predict their paths.

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
    ├── modeling**                
    │   
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── supplementary_material.py                <- **
```

--------

