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
    Interpolates the given component (U or V) for the specified x, y coordinates and time across multiple fieldsets.
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
    """
    Interation over the fieldsets to interpolate both U and V components.
    """
    interpolated = {}
    for component in ['U', 'V']:
        data = interpolate_fieldsets(x, y, component, time, fieldsets, initial_times)
        interpolated.update(data)
    return interpolated

def traditional_formula(current, stokes, wind, wind_drag_coeff):
    """
    linear regression formula for the drifter velocity as a linear combination of currents, stokes & wind
    """
    return current + stokes +wind*wind_drag_coeff

def alternative_sigmoid(current, stokes, wind_sigmoid):
    """
    alternative regression formula inclusing
    """
    return current + stokes + wind_sigmoid

# Sigmoid function
def sigmoid(x, L, x0, k, e):
    return L / (1 + np.exp(-k * (x - x0))) + e

from sklearn.preprocessing import StandardScaler
# Fit sigmoid to data

def rescale_sigmoid_params(popt, scaler_X, scaler_y):
    L_std, x0_std, k_std, e_std = popt

    # Reverse standardization
    L = L_std * scaler_y.scale_[0]
    e = e_std * scaler_y.scale_[0] + scaler_y.mean_[0]
    
    # x0 and k require a change of variable
    x0 = x0_std * scaler_X.scale_[0] + scaler_X.mean_[0]
    k = k_std / scaler_X.scale_[0]
    
    return L, x0, k, e

def fit_sigmoid(X, y,):
    X = np.array(X)
    y = np.array(y)

    scaler_X = StandardScaler()
    X_standardized = scaler_X.fit_transform(X.reshape(-1, 1)).ravel()
    # Standardize y (if needed, for example, if target variable has a different scale)
    scaler_y = StandardScaler()
    y_standardized = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    # Initial parameter guess: [L, x0, k]
    p0 = [max(y), np.median(y), 1, 0]
    # Fit the curve
    popt, pcov= curve_fit(sigmoid, X_standardized, y_standardized, p0, bounds = ([0, min(X), -np.inf, -np.inf], [np.inf, max(X), np.inf, np.inf]))
    
    return rescale_sigmoid_params(popt, scaler_X, scaler_y)  # [L, x0, k, e]

def charnock_function(wind, a, b, g=9.81, k=0.4, zstar=0.0144, zprime=10):

    #wind=u10+v10*1j
    root=np.sqrt(zprime*g/zstar)
    x=-wind*k/(2*root)

    return a*root*lambertw(x).real+b

def fit_charnock(X, y):
    X = np.array(X)
    y = np.array(y)

    p0 = [1, 0]
    # Fit the curve
    popt, _ = curve_fit(charnock_function, X, y, p0, maxfev=10000)
    
    return popt  # [a, b]


def advance_timestep_sigmoid(x, y, U, Ustokes, U10, sigmoid_coeffs, V, Vstokes, V10, sigmoid_coeffs_v, dt=5*60, R=6371000,):

    lon_rad = np.deg2rad(x)
    lat_rad = np.deg2rad(y)

    zonal_vel = alternative_sigmoid(U, Ustokes, sigmoid(U10, *sigmoid_coeffs))
   # a=traditional_formula(U, Ustokes, U10, 0.015)
    #
    meridional_vel = alternative_sigmoid(V, Vstokes, sigmoid(V10, *sigmoid_coeffs_v))
   # meridional_vel = traditional_formula(V, Vstokes, V10, sigmoid_coeffs_v)
    
    dx=np.multiply(np.array(zonal_vel), dt) #meters
    delta_lon_rad = dx / (R * np.cos(lat_rad))
    x_new=lon_rad+delta_lon_rad
    dy=meridional_vel*dt
    y_new = lat_rad + dy / R 

    return [np.rad2deg(x_new)], [np.rad2deg(y_new)]

def advance_timestep_charnock(x, y, U, Ustokes, U10, coeffs_u, V, Vstokes, V10, coeffs_v, dt=5*60, R=6371000, ):

    lon_rad = np.deg2rad(x)
    lat_rad = np.deg2rad(y)

    zonal_vel = alternative_sigmoid(U, Ustokes, charnock_function(U10, *coeffs_u))

    meridional_vel = alternative_sigmoid(V, Vstokes, charnock_function(V10, *coeffs_v))
    
    dx=np.multiply(np.array(zonal_vel), dt) #meters
    delta_lon_rad = dx / (R * np.cos(lat_rad))
    x_new=lon_rad+delta_lon_rad
    dy=meridional_vel*dt
    y_new = lat_rad + dy / R 

    return [np.rad2deg(x_new)], [np.rad2deg(y_new)]

def recreate_trajs_LR_tests(ds, delta_t_minutes, trajs_days, coeffs_u, coeffs_v,  fieldsets, initial_times, function='sigmoid'):

    """
    ds= xarray dataset for one drifter
    """

    lons=[ds['lon'][0].values]
    lats=[ds['lat'][0].values]
    time=[ds['time'][0].values]

    for i in tqdm(range(int(60/delta_t_minutes*24*trajs_days))):
        t=time[-1]
        interp = interpolate_all_components(lons[i], lats[i], t, fieldsets, initial_times)

        if function=='sigmoid':
            x1, y1 = advance_timestep_sigmoid(
                lons[i], lats[i], interp['U'], interp['Ustokes'], interp['U10'], coeffs_u,
                interp['V'], interp['Vstokes'], interp['V10'], coeffs_v,
                dt=delta_t_minutes*60,)
        elif function=='charnock':
            x1, y1 = advance_timestep_charnock(
                lons[i], lats[i], interp['U'], interp['Ustokes'], interp['U10'], coeffs_u,
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

#%%
#SIGMOID TEST


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

#LINEAR REGRESSION SETTINGGS: SIGMOID
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

    popt_u = np.array(fit_sigmoid(Xu, yu))
    # coeff_v, r2v= linear_regression(Xv, yv)
    popt_v = np.array(fit_sigmoid(Xv, yv))
    # popt_v, r2u= linear_regression(Xv, yv)

    pred= recreate_trajs_LR_tests(datatree[drifter], dt, days, popt_u,  popt_v, fieldsets, time_starts, function='sigmoid')

    with open(f"{save_dir}/{drifter}/sigmoid.pkl", "wb") as f:
            pickle.dump(pred, f)

        

# %%
dt=1 #minutes
days=60
#LINEAR REGRESSION SETTINGGS: CHARNOCK
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

    coeffs_u= fit_charnock(Xu, yu )
    coeffs_v= fit_charnock(Xv, yv)

    pred= recreate_trajs_LR_tests(datatree[drifter], dt, days, coeffs_u,  coeffs_v, fieldsets, time_starts, function='charnock')
    with open(f"{save_dir}/{drifter}/charnock.pkl", "wb") as f:
           pickle.dump(pred, f)

# %%
save_dir=f'{DATA_DIR}/processed/prediction'

with open(f"{save_dir}/6480/sigmoid.pkl", "rb") as f:
           pred=pickle.load(f)

# %%
