#%%
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import xarray as xr
import sys
sys.path.append("..")
from config import MODELS_DIR
from features import transformations, create_feature_matrix
from dataset import create_fieldset, create_static_fieldset
from modeling.linear_regression import select_variables, linear_regression
import joblib

 #%%
PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "data"

#%%
datatree = xr.open_datatree(f'{DATA_DIR}/interim/processed_drifter_data.nc')

dt=3 #minutes
days=60
#upload data
wave_file=f'{DATA_DIR}/external/waves-wind-swell.nc' 
wind_file=f'{DATA_DIR}/external/wind.nc' 
curr_file=f'{DATA_DIR}/external/ocean_curr_extended.nc' 
curr_file_decomposed=f'{DATA_DIR}/external/decomposed_ocean_currents_extended.nc' 
wave_variables=pd.read_csv(f'{PROJ_ROOT}/references/waves_dataset.csv')
bathymetry_file=f'{DATA_DIR}/external/bathymetry.nc' 
#%%
#create fields
wave_field= create_fieldset(wave_file, 'time', wave_variables.values[0], wave_variables.columns)
wind_field= create_fieldset(wind_file, 'valid_time', ['u10', 'v10'], ['U10', 'V10'])
curr_field= create_fieldset(curr_file, 'time', ['uo', 'vo'], ['U', 'V'])
curr_field_rf= create_fieldset(curr_file_decomposed, 'time', ['uo_lp', 'vo_lp', 'u_tides', 'v_tides'], ['U_lp', 'V_lp', 'U_hp', 'V_hp'])
bathymetry_field=create_static_fieldset(bathymetry_file, ['deptho'], ['z'])

#%%

def advance_timestep_ML(x, y, zonal_vel, meridional_vel, dt=1*60, R=6371000):

    lon_rad = np.deg2rad(x)
    lat_rad = np.deg2rad(y)

    dx=zonal_vel*dt
    delta_lon_rad = dx / (R * np.cos(lat_rad))
    x_new=lon_rad+delta_lon_rad

    dy=meridional_vel*dt
    y_new = lat_rad + dy / R 

    return np.rad2deg(x_new)[0], np.rad2deg(y_new)[0]

def recreate_trajs_ML(ds, delta_t_minutes, trajs_days, fieldsets, initial_times, modelu, modelv,  variables, fi_class_model, fi_regression_model, uscale, vscale):

    """
    ds= xarray dataset for one drifter
    uscale=[Xcaler, yscaler]
    """

    lons=[ds['lon'][0].values]
    lats=[ds['lat'][0].values]
    time=[ds['time'][0].values]

    for i in tqdm(range(int(60/delta_t_minutes*24*trajs_days)),  desc="Processing models", ncols=100) :
        t=time[i]
       # print(f'{i}/{int(60/delta_t_minutes*24*trajs_days)}')
        
        zonal_vel, meridional_vel=interpolate_fieldsets_ML(lons[i], lats[i], t, fieldsets, initial_times, modelu, modelv, variables, fi_class_model, fi_regression_model, uscale[0], vscale[0])

        if uscale[0]!=None: 
            zonal_vel, meridional_vel=uscale[1].inverse_transform(zonal_vel.reshape(-1, 1)), vscale[1].inverse_transform(meridional_vel.reshape(-1, 1))
            zonal_vel=np.ravel(zonal_vel)
            meridional_vel=np.ravel(meridional_vel)

        x1, y1=advance_timestep_ML(lons[-1], lats[-1], zonal_vel,  meridional_vel, dt=delta_t_minutes*60)
        lons.append(x1)
        lats.append(y1)

        t_plus_5mins = t+ np.timedelta64(delta_t_minutes, 'm')
        # Format back to original format
        time.append(t_plus_5mins)

    saving_dict={}
    saving_dict[f'lon']=lons
    saving_dict[f'lat']=lats
    saving_dict[f'time']=time
    
    return saving_dict

def interpolate_fieldsets_ML(x, y, t, fieldsets, initial_times, modelu, modelv, variables, fi_model_class, fi_model_regression, uscale, vscale ):
    """
    fieldsets=[ocean, wind, waves]
    initial_times=[ocean, wind, waves]
    variables=[['U', 'V'], ['U10', 'V10'], wave_variables.columns]
    """

    interp_ds={}

    for f, fieldset in enumerate(fieldsets):
        if initial_times[f]!=None:
            for var in variables[f]:
                time=(t - initial_times[f].values).astype('timedelta64[s]').astype(int)
                a=getattr(fieldset, var).eval(time, 0, y, x, applyConversion=False) 
                interp_ds[var]=a
        else: #static field (e.g. bathymetry)
            for var in variables[f]:
                a=getattr(fieldset, var).eval(0, 0, y, x, applyConversion=False) 
                interp_ds[var]=[a]
    interp_ds['time']=[t]
    df = pd.DataFrame(interp_ds) # Convert dataset to DataFrame
    df=df.reset_index()
  #  print(df)
    df=transformations(df, waves=True, filter_currents=False)
    df=df.set_index('time')
    feature_matrix=df

    if 'obs' in feature_matrix.columns or 'trajectory' in feature_matrix.columns:
        feature_matrix = feature_matrix.drop(columns=['obs', 'trajectory'])
    feature_matrix=feature_matrix.drop(columns=['Hs', 'index'])
    feature_matrix.columns = [str(col) for col in feature_matrix.columns]

    if fi_model_class!=None:
        feature_matrix=feature_matrix[fi_model_class.feature_names_in_]
        flipping_index_binary=fi_model_class.predict(feature_matrix)
        feature_matrix['flipping_index_scaled']=flipping_index_binary
        
        if flipping_index_binary>0.05: #if flipping is predicted
            X_filtered = feature_matrix[fi_model_regression.feature_names_in_]
            flipping_index_total=fi_model_regression.predict(X_filtered)
            feature_matrix['flipping_index_scaled']=flipping_index_total
    
    if 'lat' in modelu.feature_names_in_:
        feature_matrix['lon'], feature_matrix['lat']=x, y       

    #reorder
    feature_matrix=feature_matrix[modelu.feature_names_in_]
    feature_matrixv=feature_matrix[modelv.feature_names_in_]

    if uscale!=None:
        u_feature_matrix=uscale.fit_transform(feature_matrix)
        v_feature_matrix=vscale.fit_transform(feature_matrixv)
        feature_matrix = pd.DataFrame(u_feature_matrix, columns=feature_matrix.columns, index=feature_matrix.index)
        feature_matrix = pd.DataFrame(v_feature_matrix, columns=feature_matrixv.columns, index=feature_matrixv.index)

    zonal_vel=modelu.predict(feature_matrix)
    meridional_vel=modelv.predict(feature_matrixv)

    return zonal_vel, meridional_vel
#%%
# MACHINE LEARNING MODEL
ML_model_settings={'RF_FI_twostep':
                   {'files':[f"{MODELS_DIR}/RandomForest/RF_Ud_FI_models.pkl", 
                             f"{MODELS_DIR}/RandomForest/RF_Vd_FI_models.pkl",],
                    'FI_models':[f"{MODELS_DIR}/RandomForest/RF_FI_twostep_class_models.pkl",
                                 f"{MODELS_DIR}/RandomForest/RF_FI_twostep_regression_models.pkl"], 
                    'uscale':[None, None], 
                    'vscale':[None, None], 
                    'flipping': True},
                   }


variables_ML=[['U_lp', 'V_lp', 'U_hp', 'V_hp'], ['U10', 'V10'], wave_variables.columns]

fieldsets_ML=[curr_field_rf, wind_field, wave_field]
time_starts_ML=[xr.open_dataset(curr_file_decomposed)['time'][0], xr.open_dataset(wind_file)['valid_time'][0], xr.open_dataset(wave_file)['time'][0]]


fieldsets_ML_depth=[curr_field_rf, wind_field, wave_field, bathymetry_field]
time_starts_ML_depth=[xr.open_dataset(curr_file_decomposed)['time'][0], xr.open_dataset(wind_file)['valid_time'][0], xr.open_dataset(wave_file)['time'][0], None]
variables_ML_depth=[['U_lp', 'V_lp', 'U_hp', 'V_hp'], ['U10', 'V10'], wave_variables.columns, 'z']

#%%

import pickle 
save_dir=f'{DATA_DIR}/processed/prediction'

for i, node in enumerate(datatree.leaves):
    drifter=node.name
    print(f'{i+1}/{len(datatree.leaves)} drifters')
    #for model in tqdm(ML_model_settings, desc="Processing models", ncols=100):
    for model in ML_model_settings:
        modelu = joblib.load(ML_model_settings[model]['files'][0])
        modelv = joblib.load(ML_model_settings[model]['files'][1])
        if ML_model_settings[model]['uscale'][0]!=None:
            Xuscaler, yuscaler = joblib.load(ML_model_settings[model]['uscale'][0]), joblib.load(ML_model_settings[model]['uscale'][1])
            Xvscaler, yvscaler = joblib.load(ML_model_settings[model]['vscale'][0]), joblib.load(ML_model_settings[model]['vscale'][1])
        else: 
            Xuscaler, yuscaler, Xvscaler, yvscaler=None, None, None, None

        if ML_model_settings[model]['flipping']==True:
            fi_model_class=joblib.load(ML_model_settings[model]['FI_models'][0])
            fi_model_regression=joblib.load(ML_model_settings[model]['FI_models'][1])
        else:
            fi_model_class, fi_model_regression=None, None
        if 'depth' in model:
            pred = recreate_trajs_ML(datatree[drifter], dt, days, fieldsets_ML_depth, time_starts_ML_depth, modelu, modelv, variables_ML_depth, fi_model_class, fi_model_regression )

        else:
            pred = recreate_trajs_ML(datatree[drifter], dt, days, fieldsets_ML, time_starts_ML, modelu, modelv, variables_ML, fi_model_class, fi_model_regression, [Xuscaler, yuscaler], [Xvscaler, yvscaler])

        with open(f"{save_dir}/{drifter}/{model}.pkl", "wb") as f:
            pickle.dump(pred, f)
# %%
dir=DATA_DIR/'processed'/'prediction'
pred_dict={}
#drifter.name:{} for drifter in dt_obs.leaves 
model_analysis=['RF', 'RF_FI', 'RF_FI_twostep']
for drifter in datatree.leaves[:3]:
    model_datasets={}
    for model in model_analysis:
        with open(f"{dir}/{drifter.name}/{model}.pkl", "rb") as f:
            data=pickle.load(f)
        model_datasets[model]=xr.Dataset({var: ("time", values) for var, values in data.items()})
    pred_dict[drifter.name]=xr.DataTree.from_dict(model_datasets)

# %%
for models in model_analysis:
    plt.plot(pred_dict['6480'][models]['lon'], pred_dict['6480'][models]['lat'], label=models)
plt.plot(datatree['6480']['lon'], datatree['6480']['lat'], label='observed', color='black', alpha=0.5)
plt.legend()
# %%
