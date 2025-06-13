#%%
import sys
sys.path.append("..")
from config import DATA_DIR,  INTERIM_DATA_DIR ,MODELS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR, REFS_DIR
from features import create_feature_matrix
import pandas as pd
import pickle
import xarray as xr
import numpy as np
from ml_driven_drifter_data_analysis.modeling.model_agnostics.model_agnostics_plots import ale_plots_bootstrap_twin

 #%%
wave_variables_file= 'waves_dataset.csv'
interpolated_atm_ocean_file= 'interpolated_atm_ocean_datasets_depth.nc'

ML_model_settings={'RF_total': {'files':[f'{MODELS_DIR}/RandomForest/RF_Ud_models.pkl', f'{MODELS_DIR}/RandomForest/RF_Vd_models.pkl'], 
                                'u_variables': ['U10', 'U_hp'], 'v_variables': ['V10', 'V_hp'], 'path':['Total'],
                                'stats':[PROCESSED_DATA_DIR/ 'Statistics models'/'RandomForest'/'RF_Ud_stats', PROCESSED_DATA_DIR/ 'Statistics models'/'RandomForest'/'RF_Vd_stats'], 
                                'labels': pd.read_csv(f'{REFS_DIR}/variables_labels.csv', delimiter=';', index_col=0),
                                },
                      
                'RF_residual': {'files':[f"{MODELS_DIR}/RandomForest/RF_rUd_models.pkl", f"{MODELS_DIR}/RandomForest/RF_rVd_models.pkl"], 
                                'u_variables': ['U10', 'U_lp', 'Ustokes'], 'v_variables': ['V10', 'V_lp', 'Vstokes'], 'path':['residual'],
                                'stats':[PROCESSED_DATA_DIR/ 'Statistics models'/'RandomForest'/'RF_rUd_FI_stats', PROCESSED_DATA_DIR/ 'Statistics models'/'RandomForest'/'RF_rVd_FI_stats'], 
                                'labels': pd.read_csv(f'{REFS_DIR}/variables_labels_residual.csv', delimiter=';', index_col=0), 'title': 'Random forest model'}}
             
# %%

#%%
#total velocity
with open(f'{DATA_DIR}/processed/ALE/RF_Ud_ALE', "rb") as f:
    ale_u=pickle.load( f)

with open(f'{DATA_DIR}/processed/ALE/RF_Vd_ALE', "rb") as f:
    ale_v=pickle.load( f)

feature_pairs = { 'U_hp': 'V_hp',
    'U10': 'V10',}
variables=['U', 'V', 'U10', 'V10', 'Ustokes', 'Vstokes', 'Hs_wind', 'Hs_swell', 'Hs_swell_secondary', 'Wave_dir',
           'Wave_dir_wind', 'Wave_dir_swell', 'Wave_dir_swell_secondary', 'Tp', ]
y_variables=['vx', 'vy', 'vx_residual', 'vy_residual']

dt_features=xr.open_datatree(f'{DATA_DIR}/interim/interpolated_atm_ocean_datasets_depth.nc')

X=create_feature_matrix(dt_features, np.concatenate([variables, y_variables]) , waves=True, fc=True)

ale_plots_bootstrap_twin(X, feature_pairs, ale_u, ale_v, 
                            output_path=FIGURES_DIR/ 'ALE', output_name='Fig3', labels=ML_model_settings['RF_total']['labels'], x_names=['High-pass surface ocean currents', 'Wind'], y_name=['$U_d$', '$V_d$'])

# %%
feature_pairs = { 'U10': 'V10',}
with open(f'{DATA_DIR}/processed/ALE/RF_rUd_ALE', "rb") as f:
    rale_u=pickle.load( f)

with open(f'{DATA_DIR}/processed/ALE/RF_rVd_ALE', "rb") as f:
    rale_v=pickle.load( f)

ale_plots_bootstrap_twin(X, feature_pairs, rale_u, rale_v, 
                            output_path=FIGURES_DIR/ 'ALE', output_name='FigS5', labels=ML_model_settings['RF_residual']['labels'], x_names=['Wind'], y_name=[r'$\tilde{U}_d$', r'$\tilde{V}_d$'])

# %%
feature_pairs = { 'U_lp': 'V_lp',
    'Ustokes': 'Vstokes',  
}

ale_plots_bootstrap_twin(X, feature_pairs, rale_u, rale_v, 
                           output_path=FIGURES_DIR/ 'ALE', output_name='Fig5', labels=ML_model_settings['RF_residual']['labels'], x_names=['Low-pass surface ocean currents', 'Stokes drift'], y_name=[r'$\tilde{U}_d$', r'$\tilde{V}_d$'])

# %%
