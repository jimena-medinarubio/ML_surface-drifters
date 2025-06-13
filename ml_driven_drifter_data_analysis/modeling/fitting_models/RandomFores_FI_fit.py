#%%
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.model_selection import KFold
import sys
sys.path.append("..")
from config import INTERIM_DATA_DIR, FIGURES_DIR, REFS_DIR
from features import create_feature_matrix
from modeling.train import TimeBlockSplit, RF_regression_twostep


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
#%%

block_FI=KFold(n_splits=5, shuffle=True, random_state=42)

variables=['U', 'V', 'U10', 'V10', 'Ustokes', 'Vstokes', 'Hs_wind', 'Hs_swell', 'Hs_swell_secondary', 'Wave_dir',
           'Wave_dir_wind', 'Wave_dir_swell', 'Wave_dir_swell_secondary', 'Tp', ]
variables_analysis=['U_hp', 'U_lp', 'V_hp', 'V_lp','U10', 'V10', 'Ustokes', 'Vstokes', 'Hs_wind_x', 'Hs_wind_y',
                    'Hs_swell_x', 'Hs_swell_y', 'Hs_swell_secondary_x', 'Hs_swell_secondary_y', 'Tp_x', 'Tp_y']
y_variables=['flipping_index_scaled']
features=create_feature_matrix(dt_features, np.concatenate([variables, y_variables]) , waves=True, fc=True).dropna()
#%%
threshold=0.05
#create new binary variable for flipping index
for node in dt_features.leaves:
        node['flipping_index_binary'] = xr.where(node['flipping_index_scaled'] < threshold, 0, 1)
   
RF_regression_twostep(features[variables_analysis], features['flipping_index_binary'], features['flipping_index_scaled'], block_FI, 'FI', plot=True, calculate_permutation=True, output_path=f'{output_path}/FI_twostep_cv.svg', model_name='FI_twostep')

# %%
