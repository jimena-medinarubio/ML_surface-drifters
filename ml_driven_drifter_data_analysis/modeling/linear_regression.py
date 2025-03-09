#%%

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from haversine import haversine, Unit
import matplotlib.pyplot as plt
from plots import plot_trajs
from scipy.ndimage import gaussian_filter1d
import importlib
from importnb import Notebook
from pathlib import Path
import sys
from scipy.special import erfc
from sklearn.svm import SVR
import seaborn as sns
import pandas as pd
from config import DATA_DIR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from modeling.train import BlockingSpatioTemporalSeriesSplit

from features import create_feature_matrix
#%%
dt_features=xr.open_datatree(f'{DATA_DIR}/interim/interpolated_atm_ocean_datasets.nc')

#%%
def linear_regression(y, X):
    model = LinearRegression().fit(X, y)
    coeffs= model.coef_
    r2_total = model.score(X, y)

    print('slope', coeffs)

    return coeffs, r2_total

# %%
features=create_feature_matrix(dt_features, ['U', 'V', 'U10', 'V10', 'Ustokes', 'Vstokes', 'Tp', 'vx', 'vy'], waves=False, fc=False)

# %%

#1st-order approximation

Xu=features['vx']-features['U']-features['Ustokes']
Xv=features['vy']-features['V']-features['Vstokes']

yu=features['U10']
yv=features['V10']

coeff_u, r2u= linear_regression(Xu, yu.values.reshape(-1, 1))
coeff_v, r2v= linear_regression(Xv, yv.values.reshape(-1, 1))
# %%
#1st-order approximation: relative wind

Xu=features['vx']-features['U']-features['Ustokes']
Xv=features['vy']-features['V']-features['Vstokes']

yu=features['U10']-features['U']
yv=features['V10']-features['V']

coeff_u, r2u= linear_regression(Xu, yu.values.reshape(-1, 1))
coeff_v, r2v= linear_regression(Xv, yv.values.reshape(-1, 1))
# %%
