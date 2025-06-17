#%%
import pandas as pd
import xarray as xr
import pickle
import joblib
import sys
from tqdm import tqdm 
sys.path.append("..")
from config import DATA_DIR, REFS_DIR, EXTERNAL_DATA_DIR, INTERIM_DATA_DIR, MODELS_DIR
from features import  create_feature_matrix
from dataset import create_fieldset, create_static_fieldset
from ml_driven_drifter_data_analysis.modeling.fitting_models.linear_regression import select_variables, linear_regression
from prediction_functions import recreate_trajs_ML

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
curr_field_rf= create_fieldset(curr_file_decomposed, 'time', ['uo_lp', 'vo_lp', 'u_tides', 'v_tides'], ['U_lp', 'V_lp', 'U_hp', 'V_hp'])
bathymetry_field=create_static_fieldset(bathymetry_file, ['deptho'], ['z'])

fieldsets=[curr_field_rf, wind_field, wave_field]
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

ML_model_settings={'RF': {'files':[f"{MODELS_DIR}/RandomForest/RF_Ud_models.pkl", f"{MODELS_DIR}/RandomForest/RF_Vd_models.pkl"], 'uscale':[None, None], 'vscale':[None, None],  'flipping': False},
                     'SVR': {'files':[f"{MODELS_DIR}/SVR/SVR_Ud_models.pkl", f"{MODELS_DIR}/SVR/SVR_Vd_models.pkl"], 'uscale':[f"{DATA_DIR}/processed/Statistics models/SVR/X_Ud_scaler.pkl", f"{DATA_DIR}/processed/Statistics models/SVR/y_Ud_scaler.pkl"], 
                                    'vscale':[f"{DATA_DIR}/processed/Statistics models/SVR/X_Vd_scaler.pkl", f"{DATA_DIR}/processed/Statistics models/SVR/y_Vd_scaler.pkl"], 'flipping': False }
}


variables_ML=[['U_lp', 'V_lp', 'U_hp', 'V_hp'], ['U10', 'V10'], wave_variables.columns]

fieldsets_ML=[curr_field_rf, wind_field, wave_field]
time_starts_ML=[xr.open_dataset(curr_file_decomposed)['time'][0], xr.open_dataset(wind_file)['valid_time'][0], xr.open_dataset(wave_file)['time'][0]]


fieldsets_ML_depth=[curr_field_rf, wind_field, wave_field, bathymetry_field]
time_starts_ML_depth=[xr.open_dataset(curr_file_decomposed)['time'][0], xr.open_dataset(wind_file)['valid_time'][0], xr.open_dataset(wave_file)['time'][0], None]
variables_ML_depth=[['U_lp', 'V_lp', 'U_hp', 'V_hp'], ['U10', 'V10'], wave_variables.columns, 'z']

#%%

for i, node in enumerate(datatree.leaves):
    drifter=node.name
    print(f'{i+1}/{len(datatree.leaves)} drifters')
    for model in tqdm(ML_model_settings, desc="Processing models", ncols=100):
        modelu = joblib.load(ML_model_settings[model]['files'][0])
        modelv = joblib.load(ML_model_settings[model]['files'][1])
        if ML_model_settings[model]['uscale'][0]!=None:
            Xuscaler, yuscaler = joblib.load(ML_model_settings[model]['uscale'][0]), joblib.load(ML_model_settings[model]['uscale'][1])
            Xvscaler, yvscaler = joblib.load(ML_model_settings[model]['vscale'][0]), joblib.load(ML_model_settings[model]['vscale'][1])
        else: 
            Xuscaler, yuscaler, Xvscaler, yvscaler=None, None, None, None

        if ML_model_settings[model]['flipping']==True:
            fi_model=joblib.load(ML_model_settings[model]['files'][2])
        else:
            fi_model=None
        if 'depth' in model:
            pred = recreate_trajs_ML(datatree[drifter], dt, days, fieldsets_ML_depth, time_starts_ML_depth, modelu, modelv, variables_ML_depth, fi_model )

        else:
            pred = recreate_trajs_ML(datatree[drifter], dt, days, fieldsets_ML, time_starts_ML, modelu, modelv, variables_ML, fi_model, [Xuscaler, yuscaler], [Xvscaler, yvscaler])

        with open(f"{save_dir}/{drifter}/{model}.pkl", "wb") as f:
            pickle.dump(pred, f)