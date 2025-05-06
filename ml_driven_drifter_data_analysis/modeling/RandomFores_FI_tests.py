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
from modeling.train import TimeBlockSplit, RF_regression, RF_regression_twostep
from sklearn.model_selection import  KFold
#%%
PROJ_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJ_ROOT / "reports" / "figures"
dt_features=xr.open_datatree(f'{DATA_DIR}/interim/interpolated_atm_ocean_datasets_depth.nc')
#%%

total_autocorrelation_times = pd.read_csv(f'{PROJ_ROOT}/references/autocorrelation_Total.csv',delimiter = ',')
residual_autocorrelation_times = pd.read_csv(f'{PROJ_ROOT}/references/autocorrelation_Residual.csv',delimiter = ',')
ublock=TimeBlockSplit(block_duration=f'{str(total_autocorrelation_times['vx'].values[0])}h', n_splits=5)
vblock=TimeBlockSplit(block_duration=f'{total_autocorrelation_times['vy'].values[0]}h', n_splits=5)

ublock_residual=TimeBlockSplit(block_duration=f'{residual_autocorrelation_times['vx_residual'].values[0]}h', n_splits=5)
vblock_residual=TimeBlockSplit(block_duration=f'{residual_autocorrelation_times['vy_residual'].values[0]}h', n_splits=5)


output_path=PROJ_ROOT / "reports" / "figures" /'CrossValidation_RF'

#%%
#TEST 1: RF INCLUDING FLIPPING INDEX

variables=['U', 'V', 'U10', 'V10', 'Ustokes', 'Vstokes', 'Hs_wind', 'Hs_swell', 'Hs_swell_secondary', 'Wave_dir','flipping_index_scaled',
           'Wave_dir_wind', 'Wave_dir_swell', 'Wave_dir_swell_secondary', 'Tp', ]
#'flipping_index_scaled'
y_variables=['vx', 'vy', 'vx_residual', 'vy_residual']

features=create_feature_matrix(dt_features, np.concatenate([variables, y_variables]) , waves=True, fc=True)

variables_analysis=['U_hp', 'U_lp', 'V_hp', 'V_lp','U10', 'V10', 'Ustokes', 'Vstokes', 'Hs_wind_x', 'Hs_wind_y','flipping_index_scaled',
                    'Hs_swell_x', 'Hs_swell_y', 'Hs_swell_secondary_x', 'Hs_swell_secondary_y', 'Tp_x', 'Tp_y']
output_path=PROJ_ROOT / "reports" / "figures" /'CrossValidation_RF'
umodel, ustats= RF_regression(features[variables_analysis], features['vx'], ublock, 'Zonal velocity', plot=True, calculate_permutation=True, output_path=f'{output_path}/U_cv_FI.svg', model_name='Ud_FI')
vmodel, vstats= RF_regression(features[variables_analysis], features['vy'], vblock, 'Meridional velocity', plot=True, calculate_permutation=True, output_path=f'{output_path}/V_cv_FI.svg', model_name='Vd_FI')

rumodel, rustats= RF_regression(features[variables_analysis], features['vx_residual'], ublock_residual, 'Zonal Residual', plot=True, calculate_permutation=True, output_path=f'{output_path}/Ur_cv_FI.svg', model_name='rUd_FI')
rvmodel, rvstats= RF_regression(features[variables_analysis], features['vy_residual'], vblock_residual, 'Meridional Residual', plot=True, calculate_permutation=True, output_path=f'{output_path}/Vr_cv_FI.svg', model_name='rVd_FI')
#%%
#a. FIT FLIPPING INDEX MODEL
block_FI=KFold(n_splits=5, shuffle=True, random_state=42)
output_path=PROJ_ROOT / "reports" / "figures" 
variables=['U', 'V', 'U10', 'V10', 'Ustokes', 'Vstokes', 'Hs_wind', 'Hs_swell', 'Hs_swell_secondary', 'Wave_dir',
           'Wave_dir_wind', 'Wave_dir_swell', 'Wave_dir_swell_secondary', 'Tp', ]
variables_analysis=['U_hp', 'U_lp', 'V_hp', 'V_lp','U10', 'V10', 'Ustokes', 'Vstokes', 'Hs_wind_x', 'Hs_wind_y',
                    'Hs_swell_x', 'Hs_swell_y', 'Hs_swell_secondary_x', 'Hs_swell_secondary_y', 'Tp_x', 'Tp_y']
y_variables=['flipping_index_scaled']
features=create_feature_matrix(dt_features, np.concatenate([variables, y_variables]) , waves=True, fc=True).dropna()

fimodel, fistats= RF_regression(features[variables_analysis], features['flipping_index_scaled'], block_FI, 'FI', plot=True, calculate_permutation=True, output_path=f'{output_path}/FI_fit_cv.svg', model_name='FI_fit')

#%%

#b. FIT FLIPPING INDEX MODEL USING TWO-STEP APPROACH
threshold=0.05
#create new binary variable for flipping index
for node in dt_features.leaves:
        node['flipping_index_binary'] = xr.where(node['flipping_index_scaled'] < threshold, 0, 1)
   
#%%

block_FI=KFold(n_splits=5, shuffle=True, random_state=42)
output_path=PROJ_ROOT / "reports" / "figures" 
variables=['U', 'V', 'U10', 'V10', 'Ustokes', 'Vstokes', 'Hs_wind', 'Hs_swell', 'Hs_swell_secondary', 'Wave_dir',
           'Wave_dir_wind', 'Wave_dir_swell', 'Wave_dir_swell_secondary', 'Tp', ]
variables_analysis=['U_hp', 'U_lp', 'V_hp', 'V_lp','U10', 'V10', 'Ustokes', 'Vstokes', 'Hs_wind_x', 'Hs_wind_y',
                    'Hs_swell_x', 'Hs_swell_y', 'Hs_swell_secondary_x', 'Hs_swell_secondary_y', 'Tp_x', 'Tp_y']
y_variables=['flipping_index_binary', 'flipping_index_scaled']
#1st step: binary classification
features=create_feature_matrix(dt_features, np.concatenate([variables, y_variables]) , waves=True, fc=True).dropna()
RF_regression_twostep(features[variables_analysis], features['flipping_index_binary'], features['flipping_index_scaled'], block_FI, 'FI', plot=True, calculate_permutation=True, output_path=f'{output_path}/FI_twostep_cv.svg', model_name='FI_twostep')

# %%
