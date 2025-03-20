#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import xarray as xr
from config import MODELS_DIR
from features import transformations, create_feature_matrix
from dataset import create_fieldset
import pickle
from modeling.linear_regression import select_variables, linear_regression
import joblib

#%%
PROJ_ROOT = Path(__file__).resolve().parents[1]
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

def traditional_formula(current, stokes, wind, wind_slip):
    print('velocity', current+stokes+wind*wind_slip)
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

    print(zonal_vel, meridional_vel)

    dx=zonal_vel*dt #meters
    print(dx)
    delta_lon_rad = dx / (R * np.cos(lat_rad))
    x_new=lon_rad+delta_lon_rad
    print(np.rad2deg(x_new))

    dy=meridional_vel*dt
    print(dy)
    y_new = lat_rad + dy / R 
    print(np.rad2deg(y_new))

    return np.rad2deg(x_new), np.rad2deg(y_new)

def recreate_trajs_LR(ds, delta_t_minutes, trajs_days, coeffs_u, coeffs_v, relative_wind, fieldsets, initial_times):

    """
    ds= xarray dataset for one drifter
    """

    lons=[ds['lon'][0].values]
    lats=[ds['lat'][0].values]
    time=[ds['time'][0].values]

    for i in range(int(60/delta_t_minutes*24*trajs_days)):
        t=time[-1]
        zonal_data=interpolate_fieldsets(lons[i], lats[i], 'U', t, fieldsets, initial_times)
        meridional_data=interpolate_fieldsets(lons[i], lats[i], 'V',  t, fieldsets, initial_times )

        x1, y1=advance_timestep_linear(lons[i], lats[i], zonal_data['U'], zonal_data['Ustokes'],  zonal_data['U10'], coeffs_u,  meridional_data['V'], meridional_data['Vstokes'],  meridional_data['V10'], coeffs_v, dt=delta_t_minutes*60, relative_wind=relative_wind)
        lons.append(x1[0])
        lats.append(y1[0])

        t_plus_5mins = t+ np.timedelta64(delta_t_minutes, 'm')
        # Format back to original format
        time.append(t_plus_5mins)
    
    saving_dict={}
    saving_dict[f'lon']=lons
    saving_dict[f'lat']=lats
    saving_dict[f'ltime']=time
    
    return saving_dict

def interpolate_fieldsets_ML(x, y, t, fieldsets, initial_times, modelu, modelv, variables ):
    """
    fieldsets=[ocean, wind, waves]
    initial_times=[ocean, wind, waves]
    variables=[['U', 'V'], ['U10', 'V10'], wave_variables.columns]
    """

    interp_ds={}

    for f, fieldset in enumerate(fieldsets):
        for var in variables[f]:
            print(var)
            time=(t - initial_times[f].values).astype('timedelta64[s]').astype(int)
            #print(time)
            a=getattr(fieldset, var).eval(time, 0, y, x, applyConversion=False) 
            #print(a)
            interp_ds[var]=[a]
    interp_ds['time']=[t]
    df = pd.DataFrame(interp_ds) # Convert dataset to DataFrame
    df=df.reset_index()
    df=transformations(df, waves=True, filter_currents=False)
    df=df.set_index('time')
    feature_matrix=df

    if 'obs' in feature_matrix.columns or 'trajectory' in feature_matrix.columns:
        feature_matrix = feature_matrix.drop(columns=['obs', 'trajectory'])
    feature_matrix=feature_matrix.drop(columns=['Hs', 'index'])
    feature_matrix.columns = [str(col) for col in feature_matrix.columns]

    #reorder
    feature_matrix=feature_matrix[modelu.feature_names_in_]
    feature_matrixv=feature_matrix[modelv.feature_names_in_]

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

    return np.rad2deg(x_new), np.rad2deg(y_new)

def recreate_trajs_ML(ds, delta_t_minutes, trajs_days, fieldsets, initial_times, modelu, modelv, variables):

    """
    ds= xarray dataset for one drifter
    """

    lons=[ds['lon'][0].values]
    lats=[ds['lat'][0].values]
    time=[ds['time'][0].values]

    for i in range(int(60/delta_t_minutes*24*trajs_days)):
        t=time[i]
        zonal_vel, meridional_vel=interpolate_fieldsets_ML(lons[i], lats[i], t, fieldsets, initial_times, modelu, modelv, variables)
        x1, y1=advance_timestep_ML(lons[-1], lats[-1], zonal_vel,  meridional_vel, dt=delta_t_minutes*60)
        lons.append(x1[0])
        lats.append(y1[0])

        t_plus_5mins = t+ np.timedelta64(delta_t_minutes, 'm')
        # Format back to original format
        time.append(t_plus_5mins)

    saving_dict={}
    saving_dict[f'lon']=lons
    saving_dict[f'lat']=lats
    saving_dict[f'ltime']=time
    
    return saving_dict

#%%

#upload data
wave_file=f'{DATA_DIR}/external/waves-wind-swell.nc' 
wind_file=f'{DATA_DIR}/external/wind.nc' 
curr_file=f'{DATA_DIR}/external/ocean_currents.nc' 
curr_file_decomposed=f'{DATA_DIR}/external/decomposed_ocean_currents.nc' 
wave_variables=pd.read_csv(f'{PROJ_ROOT}/references/waves_dataset.csv')

#%%

#create fields
wave_field= create_fieldset(wave_file, 'time', wave_variables.values[0], wave_variables.columns)
wind_field= create_fieldset(wind_file, 'valid_time', ['u10', 'v10'], ['U10', 'V10'])
curr_field= create_fieldset(curr_file, 'time', ['uo', 'vo'], ['U', 'V'])
curr_field_rf= create_fieldset(curr_file_decomposed, 'time', ['uo_lp', 'vo_lp', 'u_tides', 'v_tides'], ['U_lp', 'V_lp', 'U_hp', 'V_hp'])

# %%

datatree = xr.open_datatree(f'{DATA_DIR}/interim/processed_drifter_data.nc')
predicted_dict={drifter.name:{} for drifter in datatree.leaves}
drifter='6480'

dt=3 #minutes
days=65

#%%

# MACHINE LEARNING MODEL

modelu = joblib.load(f"{MODELS_DIR}/RandomForest/RF_Ud_models.pkl")
modelv= joblib.load(f"{MODELS_DIR}/RandomForest/RF_Vd_models.pkl")

modelu_fi = joblib.load(f"{MODELS_DIR}/RandomForest/RF_Ud_FI_models.pkl")
modelv_fi= joblib.load(f"{MODELS_DIR}/RandomForest/RF_Vd_FI_models.pkl")

variables_ML=[['U_lp', 'V_lp', 'U_hp', 'V_hp'], ['U10', 'V10'], wave_variables.columns]

fieldsets_ML=[curr_field_rf, wind_field, wave_field]
time_starts_ML=[xr.open_dataset(curr_file_decomposed)['time'][0], xr.open_dataset(wind_file)['valid_time'][0], xr.open_dataset(wave_file)['time'][0]]

predicted_dict[drifter]['RF'] = recreate_trajs_ML(datatree[drifter], dt, days, fieldsets_ML, time_starts_ML, modelu, modelv, variables_ML)
#%%
#variables_fi=np.concatenate([variables, 'flipping_index_scaled'])
#predicted_dict[drifter]['RF_fi'] = recreate_trajs_ML(datatree[drifter], 3, 3, fieldsets, time_starts, modelu_fi, modelv_fi, variables_fi)


# %%

#LINEAR REGRESSION
fieldsets=[curr_field, wind_field, wave_field]
time_starts=[xr.open_dataset(curr_file)['time'][0], xr.open_dataset(wind_file)['valid_time'][0], xr.open_dataset(wave_file)['time'][0]]
variables=[['U', 'V'], ['U10', 'V10'], wave_variables.columns]

dt_features=xr.open_datatree(f'{DATA_DIR}/interim/interpolated_atm_ocean_datasets.nc')
features=create_feature_matrix(dt_features, ['U', 'V', 'U10', 'V10', 'Ustokes', 'Vstokes', 'Tp', 'vx', 'vy'], waves=False, fc=False)
#1st-order approximation
Xu, yu= select_variables(features, 'vx', 'U', relative_wind=False )
Xv, yv= select_variables(features, 'vy', 'V', relative_wind=False )
coeff_u, r2u= linear_regression(Xu, yu.values.reshape(-1, 1))
coeff_v, r2v= linear_regression(Xv, yv.values.reshape(-1, 1))
predicted_dict[drifter]['LR'] = recreate_trajs_LR(datatree[drifter], dt, days, coeff_u,  coeff_v, False, fieldsets, time_starts)

#relative wind
Xu, yu= select_variables(features, 'vx', 'U', relative_wind=True )
Xv, yv= select_variables(features, 'vy', 'V', relative_wind=True )
coeff_u, r2u= linear_regression(Xu, yu.values.reshape(-1, 1))
coeff_v, r2v= linear_regression(Xv, yv.values.reshape(-1, 1))
predicted_dict[drifter]['LR_rw'] = recreate_trajs_LR(datatree[drifter], dt, days, coeff_u,  coeff_v, True, fieldsets, time_starts)

# %%
plt.scatter(predicted_dict['6480']['RF']['lon'], predicted_dict['6480']['RF']['lat'])
# %%
