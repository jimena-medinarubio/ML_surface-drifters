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
from ml_driven_drifter_data_analysis.modeling.fitting_models.linear_regression import select_variables, linear_regression
import joblib

#%%
PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "data"
#%%
def interpolate_fieldsets(x, y, component,  time, fieldsets, initial_times, names=['', '10', 'stokes'],):
    """
    fieldsets=[ocean, wind, waves]
    initial_times=[ocean, wind, waves]
    """

    interpolated_data={}

    for i, field in enumerate(fieldsets):
        time_index=(time-initial_times[i].values) / np.timedelta64(1, 's')
        interpolated_point=getattr(field, f'{component}{names[i]}').eval(time_index, 0, y, x, applyConversion=False,)
        interpolated_data[f'{component}{names[i]}']=interpolated_point
    return interpolated_data

def interpolate_all_components(x, y, time, fieldsets, initial_times):
    interpolated = {}
    for component in ['U', 'V']:
        data = interpolate_fieldsets(x, y, component, time, fieldsets, initial_times)
        interpolated.update(data)
    return interpolated

def traditional_formula(current, stokes, wind, wind_slip):
    return current + stokes +wind*wind_slip

def relative_wind_formula(current, stokes, wind, wind_slip):
    return current + stokes +(wind-current)*wind_slip

def advance_timestep_linear(x, y, U, Ustokes, U10, wind_slip_u, V, Vstokes, V10, wind_slip_v, dt=5*60, R=6371000, relative_wind=False):

    lon_rad = np.deg2rad(x)
    lat_rad = np.deg2rad(y)

    if relative_wind==True:
        zonal_vel=relative_wind_formula(U, Ustokes, U10, wind_slip_u)
        meridional_vel=relative_wind_formula(V, Vstokes, V10, wind_slip_v)
    else:
        zonal_vel = traditional_formula(U, Ustokes, U10,  wind_slip_u)
        meridional_vel = traditional_formula(V, Vstokes, V10, wind_slip_v)

    dx=zonal_vel*dt #meters
    delta_lon_rad = dx / (R * np.cos(lat_rad))
    x_new=lon_rad+delta_lon_rad
    dy=meridional_vel*dt
    y_new = lat_rad + dy / R 
    
    return np.rad2deg(x_new), np.rad2deg(y_new)

def recreate_trajs_LR(ds, delta_t_minutes, trajs_days, coeffs_u, coeffs_v, relative_wind, fieldsets, initial_times):

    """
    ds= xarray dataset for one drifter
    """

    lons=[ds['lon'][0].values]
    lats=[ds['lat'][0].values]
    time=[ds['time'][0].values]

    for i in tqdm(range(int(60/delta_t_minutes*24*trajs_days))):
        t=time[-1]
        interp = interpolate_all_components(lons[i], lats[i], t, fieldsets, initial_times)
        x1, y1 = advance_timestep_linear(
            lons[i], lats[i], interp['U'], interp['Ustokes'], interp['U10'], coeffs_u,
            interp['V'], interp['Vstokes'], interp['V10'], coeffs_v,
            dt=delta_t_minutes*60, relative_wind=relative_wind)
        
        lons.append(x1[0])
        lats.append(y1[0])

        t_plus_5mins = t+ np.timedelta64(delta_t_minutes, 'm')
        # Format back to original format
        time.append(t_plus_5mins)
        pass
    
    saving_dict={}
    saving_dict[f'lon']=lons
    saving_dict[f'lat']=lats
    saving_dict[f'time']=time
    
    return saving_dict

def interpolate_fieldsets_ML(x, y, t, fieldsets, initial_times, modelu, modelv, variables, fi_model, uscale, vscale ):
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

    if fi_model!=None:
        feature_matrix=feature_matrix[fi_model.feature_names_in_]
        flipping_index=fi_model.predict(feature_matrix)
        feature_matrix['flipping_index_scaled']=flipping_index
    
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

def advance_timestep_ML(x, y, zonal_vel, meridional_vel, dt=1*60, R=6371000):

    lon_rad = np.deg2rad(x)
    lat_rad = np.deg2rad(y)

    dx=zonal_vel*dt
    delta_lon_rad = dx / (R * np.cos(lat_rad))
    x_new=lon_rad+delta_lon_rad

    dy=meridional_vel*dt
    y_new = lat_rad + dy / R 

    return np.rad2deg(x_new)[0], np.rad2deg(y_new)[0]

def recreate_trajs_ML(ds, delta_t_minutes, trajs_days, fieldsets, initial_times, modelu, modelv,  variables, fi_model, uscale, vscale):

    """
    ds= xarray dataset for one drifter
    uscale=[Xcaler, yscaler]
    """

    lons=[ds['lon'][0].values]
    lats=[ds['lat'][0].values]
    time=[ds['time'][0].values]

    for i in tqdm(range(int(60/delta_t_minutes*24*trajs_days))) :
        t=time[i]
       # print(f'{i}/{int(60/delta_t_minutes*24*trajs_days)}')
        
        zonal_vel, meridional_vel=interpolate_fieldsets_ML(lons[i], lats[i], t, fieldsets, initial_times, modelu, modelv, variables, fi_model, uscale[0], vscale[0])

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

#%%

#upload data
wave_file=f'{DATA_DIR}/external/waves-wind-swell.nc' 
wind_file=f'{DATA_DIR}/external/wind.nc' 
curr_file=f'{DATA_DIR}/external/extended_curr_surface.nc' 
curr_file_decomposed=f'{DATA_DIR}/external/decomposed_extended_curr_surface.nc' 
wave_variables=pd.read_csv(f'{PROJ_ROOT}/references/waves_dataset.csv')
bathymetry_file=f'{DATA_DIR}/external/bathymetry.nc' 
#%%

#create fields
wave_field= create_fieldset(wave_file, 'time', wave_variables.values[0], wave_variables.columns)
wind_field= create_fieldset(wind_file, 'valid_time', ['u10', 'v10'], ['U10', 'V10'])
curr_field= create_fieldset(curr_file, 'time', ['uo', 'vo'], ['U', 'V'])
curr_field_rf= create_fieldset(curr_file_decomposed, 'time', ['uo_lp', 'vo_lp', 'u_tides', 'v_tides'], ['U_lp', 'V_lp', 'U_hp', 'V_hp'])
bathymetry_field=create_static_fieldset(bathymetry_file, ['deptho'], ['z'])

# %%
#prediction settings
datatree = xr.open_datatree(f'{DATA_DIR}/interim/processed_drifter_data.nc')

dt=1 #minutes
days=60
#%%
import pickle
#LINEAR REGRESSION SETTINGGS
fieldsets=[curr_field, wind_field, wave_field]
time_starts=[xr.open_dataset(curr_file)['time'][0], xr.open_dataset(wind_file)['valid_time'][0], xr.open_dataset(wave_file)['time'][0]]
variables=[['U', 'V'], ['U10', 'V10'], wave_variables.columns]
dt_features=xr.open_datatree(f'{DATA_DIR}/interim/interpolated_atm_ocean_datasets.nc')

LR_model_settings={'LR': False, 'LR_rw': True}
save_dir=f'{DATA_DIR}/processed/prediction'

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

#%%
import pickle
# MACHINE LEARNING MODEL
ML_model_settings={'RF': {'files':[f"{MODELS_DIR}/RandomForest/RF_Ud_models.pkl", f"{MODELS_DIR}/RandomForest/RF_Vd_models.pkl"], 'uscale':[None, None], 'vscale':[None, None],  'flipping': False},
                  #  'RF_FI':{'files':[f"{MODELS_DIR}/RandomForest/RF_Ud_FI_models.pkl", f"{MODELS_DIR}/RandomForest/RF_Vd_FI_models.pkl", f"{MODELS_DIR}/RandomForest/RF_FI_fit_models.pkl"], 'uscale':[None, None], 'vscale':[None, None], 'flipping': True},
                  #  'RF_coords': {'files':[f"{MODELS_DIR}/RandomForest/RF_Ud_coords_models.pkl", f"{MODELS_DIR}/RandomForest/RF_Vd_coords_models.pkl"], 'uscale':[None, None], 'vscale':[None, None], 'flipping': False},
                   # 'RF_depth': {'files':[f"{MODELS_DIR}/RandomForest/RF_Ud_depth_models.pkl", f"{MODELS_DIR}/RandomForest/RF_Vd_depth_models.pkl"], 'uscale':[None, None], 'vscale':[None, None], 'flipping': False},
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

save_dir=f'{DATA_DIR}/processed/prediction'

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
       # predicted_dict[drifter][model]=pred
# %%
