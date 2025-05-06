#%%
import numpy as np
import xarray as xr

import joblib
import pandas as pd
from pathlib import Path

import sys
sys.path.append("..")
from config import DATA_DIR,  PROJ_ROOT ,MODELS_DIR, FIGURES_DIR
from features import create_feature_matrix
from modeling.model_agnostics import plot_pfi
from modeling.train import TimeBlockSplit, RF_regression
from sklearn.model_selection import  KFold
#%%
PROJ_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJ_ROOT / "reports" / "figures"
dt_features=xr.open_datatree(f'{DATA_DIR}/interim/interpolated_atm_ocean_datasets.nc')
#%%
variables=['U', 'V', 'U10', 'V10', 'Ustokes', 'Vstokes', 'Hs_wind', 'Hs_swell', 'Hs_swell_secondary', 'Wave_dir',
           'Wave_dir_wind', 'Wave_dir_swell', 'Wave_dir_swell_secondary', 'Tp', ]
#'flipping_index_scaled'
y_variables=['vx', 'vy', 'vx_residual', 'vy_residual']

features=create_feature_matrix(dt_features, np.concatenate([variables, y_variables]) , waves=True, fc=True)

# %%
variables_analysis=['U_hp', 'U_lp', 'V_hp', 'V_lp','U10', 'V10', 'Ustokes', 'Vstokes', 'Hs_wind_x', 'Hs_wind_y',
                    'Hs_swell_x', 'Hs_swell_y', 'Hs_swell_secondary_x', 'Hs_swell_secondary_y', 'Tp_x', 'Tp_y']


total_autocorrelation_times = pd.read_csv(f'{PROJ_ROOT}/references/autocorrelation_Total.csv',delimiter = ',')
residual_autocorrelation_times = pd.read_csv(f'{PROJ_ROOT}/references/autocorrelation_Residual.csv',delimiter = ',')

#%%
ublock=TimeBlockSplit(block_duration=f'{str(total_autocorrelation_times['vx'].values[0])}h', n_splits=5)
vblock=TimeBlockSplit(block_duration=f'{total_autocorrelation_times['vy'].values[0]}h', n_splits=5)

ublock_residual=TimeBlockSplit(block_duration=f'{residual_autocorrelation_times['vx_residual'].values[0]}h', n_splits=5)
vblock_residual=TimeBlockSplit(block_duration=f'{residual_autocorrelation_times['vy_residual'].values[0]}h', n_splits=5)


output_path=PROJ_ROOT / "reports" / "figures" /'CrossValidation_RF'
#%%
#f'{output_path}/U_cv_FI.svg', model_name='Ud_FI'
umodel, ustats= RF_regression(features[variables_analysis], features['vx'], ublock, 'Zonal velocity', plot=True, calculate_permutation=True, output_path=f'{output_path}/U_cv.svg', model_name='Ud')
vmodel, vstats= RF_regression(features[variables_analysis], features['vy'], vblock, 'Meridional velocity', plot=True, calculate_permutation=True, output_path=f'{output_path}/V_cv.svg', model_name='Vd')
#%%
output_path=PROJ_ROOT / "reports" / "figures" /'CrossValidation_RF'
rumodel, rustats= RF_regression(features[variables_analysis], features['vx_residual'], ublock_residual, 'Zonal Residual', plot=True, calculate_permutation=True, output_path=f'{output_path}/Ur_cv.svg', model_name='rUd')
rvmodel, rvstats= RF_regression(features[variables_analysis], features['vy_residual'], vblock_residual, 'Meridional Residual', plot=True, calculate_permutation=True, output_path=f'{output_path}/Vr_cv.svg', model_name='rVd')
# %%
