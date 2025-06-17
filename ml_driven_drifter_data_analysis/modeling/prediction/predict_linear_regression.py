#%%
import pandas as pd
import xarray as xr
import pickle
import sys
sys.path.append("..")
from config import DATA_DIR, REFS_DIR, EXTERNAL_DATA_DIR, INTERIM_DATA_DIR
from features import  create_feature_matrix
from dataset import create_fieldset, create_static_fieldset
from ml_driven_drifter_data_analysis.modeling.fitting_models.linear_regression import select_variables, linear_regression
from prediction_functions import recreate_trajs_LR

#%%
#open data
wave_file=f'{EXTERNAL_DATA_DIR}/waves-wind-swell.nc' 
wind_file=f'{EXTERNAL_DATA_DIR}/wind.nc' 
curr_file=f'{EXTERNAL_DATA_DIR}/extended_curr_surface.nc' 
curr_file_decomposed=f'{EXTERNAL_DATA_DIR}/decomposed_extended_curr_surface.nc' 
wave_variables=pd.read_csv(f'{REFS_DIR}/waves_dataset.csv')
bathymetry_file=f'{EXTERNAL_DATA_DIR}/bathymetry.nc' 

datatree = xr.open_datatree(f'{INTERIM_DATA_DIR}/processed_drifter_data.nc')
dt_features=xr.open_datatree(f'{INTERIM_DATA_DIR}/interpolated_atm_ocean_datasets.nc')
#%%
#create fields
wave_field= create_fieldset(wave_file, 'time', wave_variables.values[0], wave_variables.columns)
wind_field= create_fieldset(wind_file, 'valid_time', ['u10', 'v10'], ['U10', 'V10'])
curr_field= create_fieldset(curr_file, 'time', ['uo', 'vo'], ['U', 'V'])
curr_field_rf= create_fieldset(curr_file_decomposed, 'time', ['uo_lp', 'vo_lp', 'u_tides', 'v_tides'], ['U_lp', 'V_lp', 'U_hp', 'V_hp'])
bathymetry_field=create_static_fieldset(bathymetry_file, ['deptho'], ['z'])

fieldsets=[curr_field, wind_field, wave_field]
time_starts=[xr.open_dataset(curr_file)['time'][0], xr.open_dataset(wind_file)['valid_time'][0], xr.open_dataset(wave_file)['time'][0]]
variables=[['U', 'V'], ['U10', 'V10'], wave_variables.columns]

#%%
#prediction settings
dt=1 #minutes
days=60
#
LR_model_settings={'LR': False, 'LR_rw': True}
save_dir=f'{DATA_DIR}/processed/prediction'
#%%

for i, node in enumerate(datatree.leaves):
    drifter=node.name
    print(f'{i+1}/{len(datatree.leaves)}')

    drifter_dt= dt_features.copy()
    del drifter_dt[drifter]
    features=create_feature_matrix(drifter_dt, ['U', 'V', 'U10', 'V10', 'Ustokes', 'Vstokes', 'Tp', 'vx', 'vy'], waves=False, fc=False)
    #1st-order approximation

    for model in LR_model_settings:
        Xu, yu= select_variables(features, 'vx', 'U', relative_wind=LR_model_settings[model] )
        Xv, yv= select_variables(features, 'vy', 'V', relative_wind=LR_model_settings[model] )
        coeff_u, r2u= linear_regression(Xu, yu)
        coeff_v, r2v= linear_regression(Xv, yv)


        pred= recreate_trajs_LR(datatree[drifter], dt, days, coeff_u,  coeff_v,LR_model_settings[model], fieldsets, time_starts)
        with open(f"{save_dir}/{drifter}/{model}.pkl", "wb") as f:
            pickle.dump(pred, f)