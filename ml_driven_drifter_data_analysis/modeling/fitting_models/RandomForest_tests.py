#%%
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import sys
sys.path.append("..")
from config import INTERIM_DATA_DIR, FIGURES_DIR, REFS_DIR
from features import create_feature_matrix
from modeling.train import TimeBlockSplit, RF_regression

#%%
interpolated_data_file='interpolated_atm_ocean_datasets.nc'
total_autocorrelation_file= 'autocorrelation_Total.csv'
residual_autocorrelation_file= 'autocorrelation_Residual.csv'
#%%

dt_features=xr.open_datatree(f'{INTERIM_DATA_DIR}interpolated_atm_ocean_datasets.nc')

total_autocorrelation_times = pd.read_csv(f'{REFS_DIR}/{total_autocorrelation_file}',delimiter = ',')
residual_autocorrelation_times = pd.read_csv(f'{REFS_DIR}/{residual_autocorrelation_file}',delimiter = ',')

ublock=TimeBlockSplit(block_duration=f'{str(total_autocorrelation_times['vx'].values[0])}h', n_splits=5)
vblock=TimeBlockSplit(block_duration=f'{total_autocorrelation_times['vy'].values[0]}h', n_splits=5)

ublock_residual=TimeBlockSplit(block_duration=f'{residual_autocorrelation_times['vx_residual'].values[0]}h', n_splits=5)
vblock_residual=TimeBlockSplit(block_duration=f'{residual_autocorrelation_times['vy_residual'].values[0]}h', n_splits=5)
output_path=FIGURES_DIR /'CrossValidation_RF'

#%% TEST1: RF + including latitude and longitude as features

variables=['U', 'V', 'U10', 'V10', 'Ustokes', 'Vstokes', 'Hs_wind', 'Hs_swell', 'Hs_swell_secondary', 'Wave_dir', 'lat', 'lon',
           'Wave_dir_wind', 'Wave_dir_swell', 'Wave_dir_swell_secondary', 'Tp', ]
y_variables=['vx', 'vy', 'vx_residual', 'vy_residual']
features=create_feature_matrix(dt_features, np.concatenate([variables, y_variables]) , waves=True, fc=True)
variables_analysis=['U_hp', 'U_lp', 'V_hp', 'V_lp','U10', 'V10', 'Ustokes', 'Vstokes', 'Hs_wind_x', 'Hs_wind_y', 'lat', 'lon',
                    'Hs_swell_x', 'Hs_swell_y', 'Hs_swell_secondary_x', 'Hs_swell_secondary_y', 'Tp_x', 'Tp_y']

umodel, ustats= RF_regression(features[variables_analysis], features['vx'], ublock, 'Zonal velocity', plot=True, calculate_permutation=True, output_path=f'{output_path}/U_coords_cv.svg', model_name='Ud_coords')
vmodel, vstats= RF_regression(features[variables_analysis], features['vy'], vblock, 'Meridional velocity', plot=True, calculate_permutation=True, output_path=f'{output_path}/V_coords_cv.svg', model_name='Vd_coords')

#%%

#TEST2: RF + including water column depth as feature
variables=['U', 'V', 'U10', 'V10', 'Ustokes', 'Vstokes', 'Hs_wind', 'Hs_swell', 'Hs_swell_secondary', 'Wave_dir', 'z',
           'Wave_dir_wind', 'Wave_dir_swell', 'Wave_dir_swell_secondary', 'Tp', ]
y_variables=['vx', 'vy', 'vx_residual', 'vy_residual']
features=create_feature_matrix(dt_features, np.concatenate([variables, y_variables]) , waves=True, fc=True)
variables_analysis=['U_hp', 'U_lp', 'V_hp', 'V_lp','U10', 'V10', 'Ustokes', 'Vstokes', 'Hs_wind_x', 'Hs_wind_y', 'z',
                    'Hs_swell_x', 'Hs_swell_y', 'Hs_swell_secondary_x', 'Hs_swell_secondary_y', 'Tp_x', 'Tp_y']

umodel, ustats= RF_regression(features[variables_analysis], features['vx'], ublock, 'Zonal velocity', plot=True, calculate_permutation=True, output_path=f'{output_path}/U_depth_cv.svg', model_name='Ud_depth')
vmodel, vstats= RF_regression(features[variables_analysis], features['vy'], vblock, 'Meridional velocity', plot=True, calculate_permutation=True, output_path=f'{output_path}/V_depth_cv.svg', model_name='Vd_depth')

#%%
#TEST 3: RF INCLUDING FLIPPING INDEX

variables=['U', 'V', 'U10', 'V10', 'Ustokes', 'Vstokes', 'Hs_wind', 'Hs_swell', 'Hs_swell_secondary', 'Wave_dir','flipping_index_scaled',
           'Wave_dir_wind', 'Wave_dir_swell', 'Wave_dir_swell_secondary', 'Tp', ]
#'flipping_index_scaled'
y_variables=['vx', 'vy', 'vx_residual', 'vy_residual']

features=create_feature_matrix(dt_features, np.concatenate([variables, y_variables]) , waves=True, fc=True)

variables_analysis=['U_hp', 'U_lp', 'V_hp', 'V_lp','U10', 'V10', 'Ustokes', 'Vstokes', 'Hs_wind_x', 'Hs_wind_y','flipping_index_scaled',
                    'Hs_swell_x', 'Hs_swell_y', 'Hs_swell_secondary_x', 'Hs_swell_secondary_y', 'Tp_x', 'Tp_y']
output_path=FIGURES_DIR /'CrossValidation_RF'

#%%
umodel, ustats= RF_regression(features[variables_analysis], features['vx'], ublock, 'Zonal velocity', plot=True, calculate_permutation=True, output_path=f'{output_path}/U_cv_FI.svg', model_name='Ud_FI')
vmodel, vstats= RF_regression(features[variables_analysis], features['vy'], vblock, 'Meridional velocity', plot=True, calculate_permutation=True, output_path=f'{output_path}/V_cv_FI.svg', model_name='Vd_FI')

rumodel, rustats= RF_regression(features[variables_analysis], features['vx_residual'], ublock_residual, 'Zonal Residual', plot=True, calculate_permutation=True, output_path=f'{output_path}/Ur_cv_FI.svg', model_name='rUd_FI')
rvmodel, rvstats= RF_regression(features[variables_analysis], features['vy_residual'], vblock_residual, 'Meridional Residual', plot=True, calculate_permutation=True, output_path=f'{output_path}/Vr_cv_FI.svg', model_name='rVd_FI')
