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
from modeling.model_agnostics import single_ALE_plots, leave_one_out_ALE_plots, ale_plots_bootstrap, bootstrap_ale

#%%

ML_model_settings={'RF_total': {'files':[f'{MODELS_DIR}/RandomForest/RF_Ud_models.pkl', f'{MODELS_DIR}/RandomForest/RF_Vd_models.pkl'], 
                                'stats':[PROJ_ROOT/ 'data'/ 'processed'/ 'Statistics models'/'RandomForest'/'RF_Ud_stats', PROJ_ROOT/ 'data'/ 'processed'/ 'Statistics models'/'RandomForest'/'RF_Vd_stats'], 
                                'u_variables': ['U10', 'U_hp'], 'v_variables': ['V10', 'V_hp'], 'path':['Total'],
                                'labels': pd.read_csv(f'{PROJ_ROOT}/references/variables_labels.csv', delimiter=';', index_col=0)},
                      
                'RF_residual': {'files':[f"{MODELS_DIR}/RandomForest/RF_rUd_models.pkl", f"{MODELS_DIR}/RandomForest/RF_rVd_models.pkl"], 
                                'stats':[PROJ_ROOT/ 'data'/ 'processed'/ 'Statistics models'/'RandomForest'/'RF_rUd_FI_stats', PROJ_ROOT/ 'data'/ 'processed'/ 'Statistics models'/'RandomForest'/'RF_rVd_FI_stats'], 
                                'u_variables': ['U10', 'U_lp', 'Ustokes'], 'v_variables': ['V10', 'V_lp', 'Vstokes'], 'path':['residual'],
                                'labels': pd.read_csv(f'{PROJ_ROOT}/references/variables_labels_residual.csv', delimiter=';', index_col=0)},
             }
#%%
wave_variables=pd.read_csv(f'{PROJ_ROOT}/references/waves_dataset.csv')
variables=['U', 'V', 'U10', 'V10', 'Ustokes', 'Vstokes', 'Hs_wind', 'Hs_swell', 'Hs_swell_secondary', 'Wave_dir',
           'Wave_dir_wind', 'Wave_dir_swell', 'Wave_dir_swell_secondary', 'Tp', ]
y_variables=['vx', 'vy', 'vx_residual', 'vy_residual']
variables_analysis=['U_hp', 'U_lp', 'V_hp', 'V_lp','U10', 'V10', 'Ustokes', 'Vstokes', 'Hs_wind_x', 'Hs_wind_y', 
                    'Hs_swell_x', 'Hs_swell_y', 'Hs_swell_secondary_x', 'Hs_swell_secondary_y', 'Tp_x', 'Tp_y']

dt_features=xr.open_datatree(f'{DATA_DIR}/interim/interpolated_atm_ocean_datasets_depth.nc')

#%%

X=create_feature_matrix(dt_features, np.concatenate([variables, y_variables]) , waves=True, fc=True)
path=FIGURES_DIR/ 'ALE'

#%%

#sSINGLE ALE PLOTS
for model in ML_model_settings:
    for vars in ML_model_settings[model]['u_variables']:
        output_path = FIGURES_DIR / 'ALE'/ f'ale_{model}_{vars}'
        ustats=pd.read_csv(f'{ML_model_settings[model]['stats'][0]}.csv', delimiter=',')
        umodel=joblib.load(ML_model_settings[model]['files'][0])

        name=ML_model_settings[model]['labels'][vars].loc['label']
        units=ML_model_settings[model]['labels'][vars].loc['units']

        output_path=path/ ML_model_settings[model]['path'][0]/ f'{model}_{vars}'

        single_ALE_plots(umodel, X, '$U_d$', vars,  scale=None, name=f'{name} {units}', resolution=20, output_path=output_path)

    for vars in ML_model_settings[model]['v_variables']:
        vstats=pd.read_csv(f'{ML_model_settings[model]['stats'][1]}.csv', delimiter=',')
        vmodel=joblib.load(ML_model_settings[model]['files'][1])
        name=ML_model_settings[model]['labels'][vars].loc['label']
        units=ML_model_settings[model]['labels'][vars].loc['units']

        output_path=path/ ML_model_settings[model]['path'][0]/ f'{model}_{vars}'

        single_ALE_plots(vmodel, X, '$V_d$', vars,  scale=None, name=f'{name} {units}', resolution=20, output_path=output_path)
        
#%%
#LEAVE ONE OUT
vars='U_hp'
variables=['U', 'V', 'U10', 'V10', 'Ustokes', 'Vstokes', 'Hs_wind', 'Hs_swell', 'Hs_swell_secondary', 'Wave_dir',
           'Wave_dir_wind', 'Wave_dir_swell', 'Wave_dir_swell_secondary', 'Tp', ]

#montecarlo_ALE_plots(umodel, X[variables_analysis],  'rU_d', vars)
leave_one_out_ALE_plots(f'{DATA_DIR}/processed/prediction/', 'RF_Ud_models', '$U_d$', vars, variables, dt_features, scale=None, name=None, resolution=20,)

# %%

ale_u, ale_v=bootstrap_ale(X[variables_analysis], X['vx'], X['vy'],  ['U_hp', 'U10'], ['V_hp', 'V10'], n_bootstraps=100)
#%%
#%%
import pickle
with open(f'{DATA_DIR}/processed/ALE/RF_Ud_ALE', "rb") as f:
    ale_u=pickle.load( f)

with open(f'{DATA_DIR}/processed/ALE/RF_Vd_ALE', "rb") as f:
    ale_v=pickle.load( f)
#%%
ale_plots_bootstrap(ale_u, '$U_d$', output_path=FIGURES_DIR/ 'ALE'/'Total', labels=ML_model_settings['RF_total']['labels'])

ale_plots_bootstrap(ale_v, '$V_d$', output_path=FIGURES_DIR/ 'ALE'/'Total', labels=ML_model_settings['RF_total']['labels'])

# %%
ale_ru, ale_rv=bootstrap_ale(X[variables_analysis], X['vx_residual'], X['vy_residual'],  ['U_lp', 'U10', 'Ustokes'], ['V_lp', 'V10', 'Vstokes'])

#%%
import pickle
with open(f'{DATA_DIR}/processed/ALE/RF_rUd_ALE', "rb") as f:
    ale_ru=pickle.load( f)

with open(f'{DATA_DIR}/processed/ALE/RF_rVd_ALE', "rb") as f:
    ale_rv=pickle.load( f)
ale_plots_bootstrap(ale_ru, r'$\tilde{U_d}$', output_path=FIGURES_DIR/ 'ALE'/'Residual', labels=ML_model_settings['RF_total']['labels'])

ale_plots_bootstrap(ale_rv, r'$\tilde{V_d}$', output_path=FIGURES_DIR/ 'ALE'/'Residual', labels=ML_model_settings['RF_total']['labels'])

#%%
import pickle
with open(f'{DATA_DIR}/processed/ALE/RF_rUd_ALE', "wb") as f:
    pickle.dump(ale_ru, f)

with open(f'{DATA_DIR}/processed/ALE/RF_rVd_ALE', "wb") as f:
    pickle.dump(ale_rv, f)
# %%
