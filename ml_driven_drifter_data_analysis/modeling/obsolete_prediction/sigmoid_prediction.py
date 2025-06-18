#%%
import numpy as np
from tqdm import tqdm
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit
import pickle
from scipy.special import lambertw
import sys
sys.path.append("..")
from config import DATA_DIR
from features import create_feature_matrix
from dataset import create_fieldset, create_static_fieldset
from ml_driven_drifter_data_analysis.modeling.fitting_models.linear_regression import select_variables
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

def alternative_sigmoid(current, stokes, wind_sigmoid):
    return current + stokes + wind_sigmoid

# Sigmoid function
def sigmoid(x, L, x0, k, e):
    return L / (1 + np.exp(-k * (x - x0)))+e

# Fit sigmoid to data
def fit_sigmoid(X, y):
    X = np.array(X)
    y = np.array(y)
    # Initial parameter guess: [L, x0, k]
    p0 = [max(y), np.median(X), 1, 0]
    # Fit the curve
    popt, _ = curve_fit(sigmoid, X, y, p0, maxfev=10000)
    
    return popt  # [L, x0, k]


def advance_timestep_sigmoid(x, y, U, Ustokes, U10, sigmoid_coeffs, V, Vstokes, V10, wind_slip_v, dt=5*60, R=6371000, ):

    lon_rad = np.deg2rad(x)
    lat_rad = np.deg2rad(y)

    zonal_vel = alternative_sigmoid(U, Ustokes, sigmoid(U10, *sigmoid_coeffs[0:4]))
    a=traditional_formula(U, Ustokes, U10, 0.015)

    meridional_vel = traditional_formula(V, Vstokes, V10, wind_slip_v)
    
    dx=np.multiply(np.array(zonal_vel), dt) #meters
    delta_lon_rad = dx / (R * np.cos(lat_rad))
    x_new=lon_rad+delta_lon_rad
    dy=meridional_vel*dt
    y_new = lat_rad + dy / R 

    return [np.rad2deg(x_new)], np.rad2deg(y_new)

def recreate_trajs_LR_sigmoid(ds, delta_t_minutes, trajs_days, sigmoid_coeffs, coeffs_v,  fieldsets, initial_times):

    """
    ds= xarray dataset for one drifter
    """

    lons=[ds['lon'][0].values]
    lats=[ds['lat'][0].values]
    time=[ds['time'][0].values]

    for i in tqdm(range(int(60/delta_t_minutes*24*trajs_days))):
        t=time[-1]
        interp = interpolate_all_components(lons[i], lats[i], t, fieldsets, initial_times)
        x1, y1 = advance_timestep_sigmoid(
            lons[i], lats[i], interp['U'], interp['Ustokes'], interp['U10'], sigmoid_coeffs,
            interp['V'], interp['Vstokes'], interp['V10'], coeffs_v,
            dt=delta_t_minutes*60, )
        
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

def advance_timestep_linear(x, y, U, Ustokes, U10, wind_slip_u, V, Vstokes, V10, wind_slip_v, dt=5*60, R=6371000, relative_wind=False):

    lon_rad = np.deg2rad(x)
    lat_rad = np.deg2rad(y)

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
#%%
#SIGMOID TEST


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

# %%
#prediction settings
datatree = xr.open_datatree(f'{DATA_DIR}/interim/processed_drifter_data.nc')

dt=3 #minutes
days=51
#%%

#LINEAR REGRESSION SETTINGGS
fieldsets=[curr_field, wind_field, wave_field]
time_starts=[xr.open_dataset(curr_file)['time'][0], xr.open_dataset(wind_file)['valid_time'][0], xr.open_dataset(wave_file)['time'][0]]
variables=[['U', 'V'], ['U10', 'V10'], wave_variables.columns]
dt_features=xr.open_datatree(f'{DATA_DIR}/interim/interpolated_atm_ocean_datasets_depth.nc')

save_dir=f'{DATA_DIR}/processed/prediction'

for i, node in enumerate(datatree.leaves):
    drifter=node.name
    print(f'{i+1}/{len(datatree.leaves)}')

    drifter_dt= dt_features.copy()
    del drifter_dt[drifter]
    features=create_feature_matrix(drifter_dt, ['U', 'V', 'U10', 'V10', 'Ustokes', 'Vstokes', 'Tp', 'vx', 'vy'], waves=False, fc=False)
    #1st-order approximation

    Xu, yu= select_variables(features, 'vx', 'U', relative_wind=False )
    Xv, yv= select_variables(features, 'vy', 'V', relative_wind=False )

    popt_u= fit_sigmoid(Xu, yu)
    coeff_v, r2v= linear_regression(Xv, yv)
    #coeff_u, r2u= linear_regression(Xu, yu)

    #pred= recreate_trajs_LR(datatree[drifter], dt, days, coeff_u,  coeff_v, False, fieldsets, time_starts)

    pred= recreate_trajs_LR_sigmoid(datatree[drifter], dt, days, popt_u,  coeff_v, fieldsets, time_starts)
    with open(f"{save_dir}/{drifter}/sigmoid.pkl", "wb") as f:
           pickle.dump(pred, f)

        

# %%
