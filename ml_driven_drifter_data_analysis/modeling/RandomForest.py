#%%
import numpy as np
import xarray as xr
from config import DATA_DIR,  PROJ_ROOT
import joblib
import pandas as pd
from pathlib import Path
from features import create_feature_matrix
from modeling.model_agnostics import plot_pfi
from modeling.train import TimeBlockSplit, RF_regression
#%%
PROJ_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJ_ROOT / "reports" / "figures"
dt_features=xr.open_datatree(f'{DATA_DIR}/interim/interpolated_atm_ocean_datasets.nc')
#%%
variables=['U', 'V', 'U10', 'V10', 'Ustokes', 'Vstokes', 'Hs_wind', 'Hs_swell', 'Hs_swell_secondary', 'Wave_dir',
           'Wave_dir_wind', 'Wave_dir_swell', 'Wave_dir_swell_secondary', 'Tp', 'flipping_index_scaled']
y_variables=['vx', 'vy', 'vx_residual', 'vy_residual']

features=create_feature_matrix(dt_features, np.concatenate([variables, y_variables]) , waves=True, fc=True)

# %%
variables_analysis=['U_hp', 'U_lp', 'V_hp', 'V_lp','U10', 'V10', 'Ustokes', 'Vstokes', 'Hs_wind_x', 'Hs_wind_y', 'flipping_index_scaled',
                    'Hs_swell_x', 'Hs_swell_y', 'Hs_swell_secondary_x', 'Hs_swell_secondary_y', 'Tp_x', 'Tp_y']

block=TimeBlockSplit(block_duration='25h', n_splits=5)
block_residual=TimeBlockSplit(block_duration='54h', n_splits=5)
output_path=PROJ_ROOT / "reports" / "figures" /'CrossValidation'
#%%

umodel, ustats= RF_regression(features[variables_analysis], features['vx'], block, 'Zonal velocity', plot=True, calculate_permutation=True, output_path=f'{output_path}/U_cv_FI.svg', model_name='Ud_FI')
vmodel, vstats= RF_regression(features[variables_analysis], features['vy'], block, 'Meridional velocity', plot=True, calculate_permutation=True, output_path=f'{output_path}/V_cv_FI.svg', model_name='Vd_FI')
#%%
output_path=PROJ_ROOT / "reports" / "figures" /'CrossValidation'
rumodel, rustats= RF_regression(features[variables_analysis], features['vx_residual'], block_residual, 'Zonal Residual', plot=True, calculate_permutation=True, output_path=f'{output_path}/Ur_cv.svg', model_name='rUd')
rvmodel, rvstats= RF_regression(features[variables_analysis], features['vy_residual'], block_residual, 'Meridional Residual', plot=True, calculate_permutation=True, output_path=f'{output_path}/Vr_cv.svg', model_name='rVd')

# %%
# %%
labels=pd.read_csv(f'{PROJ_ROOT}/references/variables_labels_FI.csv', delimiter=';', index_col=0)
output_path = FIGURES_DIR / 'pfi_rf_total_FI'
plot_pfi(ustats, vstats, labels, output_path, residual=False)
# %%

labels=pd.read_csv(f'{PROJ_ROOT}/references/variables_labels_residual.csv', delimiter=';', index_col=0)
output_path = FIGURES_DIR / 'pfi_rf_residual'
plot_pfi(rustats, rvstats, labels, output_path, residual=True)
# %%
