#%%
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.optimize import curve_fit
import sys
sys.path.append("..")
from features import transformations
from sklearn.preprocessing import StandardScaler
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

def traditional_formula(current, stokes, wind, wind_slip):
    return current + stokes +wind*wind_slip

def alternative_sigmoid(current, stokes, wind_sigmoid):
    return current + stokes + wind_sigmoid

def relative_wind_formula(current, stokes, wind, wind_slip):
    return current + stokes +(wind-current)*wind_slip

# Sigmoid function
def sigmoid(x, L, x0, k, e):
    return L / (1 + np.exp(-k * (x - x0))) + e

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
    
    return rescale_sigmoid_params(popt, scaler_X, scaler_y) 


def advance_timestep_sigmoid(x, y, U, Ustokes, U10, sigmoid_coeffs, V, Vstokes, V10, sigmoid_coeffs_v, dt=5*60, R=6371000,):

    lon_rad = np.deg2rad(x)
    lat_rad = np.deg2rad(y)

    zonal_vel = alternative_sigmoid(U, Ustokes, sigmoid(U10, *sigmoid_coeffs))  #
    meridional_vel = alternative_sigmoid(V, Vstokes, sigmoid(V10, *sigmoid_coeffs_v)) 

    dx=np.multiply(np.array(zonal_vel), dt) #meters
    delta_lon_rad = dx / (R * np.cos(lat_rad))
    x_new=lon_rad+delta_lon_rad
    dy=meridional_vel*dt
    y_new = lat_rad + dy / R 

    return [np.rad2deg(x_new)], [np.rad2deg(y_new)]

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

def advance_timestep_ML(x, y, zonal_vel, meridional_vel, dt=1*60, R=6371000):

    lon_rad = np.deg2rad(x)
    lat_rad = np.deg2rad(y)

    dx=zonal_vel*dt
    delta_lon_rad = dx / (R * np.cos(lat_rad))
    x_new=lon_rad+delta_lon_rad

    dy=meridional_vel*dt
    y_new = lat_rad + dy / R 

    return np.rad2deg(x_new)[0], np.rad2deg(y_new)[0]

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