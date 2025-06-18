#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import xarray as xr
import sys
sys.path.append("..")
from config import MODELS_DIR
from features import transformations, create_feature_matrix
from dataset import create_fieldset
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

        plt.scatter(x1, y1)
    
    saving_dict={}
    saving_dict[f'lon']=lons
    saving_dict[f'lat']=lats
    saving_dict[f'time']=time
    
    return saving_dict

def interpolate_fieldsets_ML(x, y, t, fieldsets, initial_times, modelu, modelv, variables, fi_model ):
    """
    fieldsets=[ocean, wind, waves]
    initial_times=[ocean, wind, waves]
    variables=[['U', 'V'], ['U10', 'V10'], wave_variables.columns]
    """

    interp_ds={}

    for f, fieldset in enumerate(fieldsets):
        for var in variables[f]:
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

    if fi_model!=None:
        feature_matrix=feature_matrix[fi_model.feature_names_in_]
        flipping_index=fi_model.predict(feature_matrix)
        feature_matrix['flipping_index_scaled']=flipping_index
    
    if 'lat' in modelu.feature_names_in_:
        feature_matrix['lon'], feature_matrix['lat']=x, y       

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

def recreate_trajs_ML(ds, delta_t_minutes, trajs_days, fieldsets, initial_times, modelu, modelv,  variables, fi_model):

    """
    ds= xarray dataset for one drifter
    """

    lons=[ds['lon'][0].values]
    lats=[ds['lat'][0].values]
    time=[ds['time'][0].values]

    for i in range(int(60/delta_t_minutes*24*trajs_days)):
        t=time[i]
       # print(f'{i}/{int(60/delta_t_minutes*24*trajs_days)}')
        zonal_vel, meridional_vel=interpolate_fieldsets_ML(lons[i], lats[i], t, fieldsets, initial_times, modelu, modelv, variables, fi_model)
        x1, y1=advance_timestep_ML(lons[-1], lats[-1], zonal_vel,  meridional_vel, dt=delta_t_minutes*60)
        lons.append(x1[0])
        lats.append(y1[0])

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
curr_file=f'{DATA_DIR}/external/ocean_curr_extended.nc' 
curr_file_decomposed=f'{DATA_DIR}/external/decomposed_ocean_currents_extended.nc' 
wave_variables=pd.read_csv(f'{PROJ_ROOT}/references/waves_dataset.csv')

#%%

#create fields
wave_field= create_fieldset(wave_file, 'time', wave_variables.values[0], wave_variables.columns)
wind_field= create_fieldset(wind_file, 'valid_time', ['u10', 'v10'], ['U10', 'V10'])
curr_field= create_fieldset(curr_file, 'time', ['uo', 'vo'], ['U', 'V'])
curr_field_rf= create_fieldset(curr_file_decomposed, 'time', ['uo_lp', 'vo_lp', 'u_tides', 'v_tides'], ['U_lp', 'V_lp', 'U_hp', 'V_hp'])

# %%
#prediction settings
datatree = xr.open_datatree(f'{DATA_DIR}/interim/processed_drifter_data.nc')
predicted_dict={drifter.name:{} for drifter in datatree.leaves}

dt=3 #minutes
days=60
#%%

#LINEAR REGRESSION SETTINGGS
fieldsets=[curr_field, wind_field, wave_field]
time_starts=[xr.open_dataset(curr_file)['time'][0], xr.open_dataset(wind_file)['valid_time'][0], xr.open_dataset(wave_file)['time'][0]]
variables=[['U', 'V'], ['U10', 'V10'], wave_variables.columns]
dt_features=xr.open_datatree(f'{DATA_DIR}/interim/interpolated_atm_ocean_datasets.nc')

LR_model_settings={'LR': False, 'LR_rw': True}

for i, node in enumerate(datatree.leaves):
    drifter=node.name
    print(f'{i+1}/{len(datatree.leaves)}')

    drifter_dt=dt_features.drop_groups(drifter)
    features=create_feature_matrix(drifter_dt, ['U', 'V', 'U10', 'V10', 'Ustokes', 'Vstokes', 'Tp', 'vx', 'vy'], waves=False, fc=False)
    #1st-order approximation

    for model in LR_model_settings:
        Xu, yu= select_variables(features, 'vx', 'U', relative_wind=LR_model_settings[model] )
        Xv, yv= select_variables(features, 'vy', 'V', relative_wind=LR_model_settings[model] )
        coeff_u, r2u= linear_regression(Xu, yu)
        coeff_v, r2v= linear_regression(Xv, yv)
        predicted_dict[drifter][model] = recreate_trajs_LR(datatree[drifter], dt, days, coeff_u,  coeff_v, False, fieldsets, time_starts)

#%%
# MACHINE LEARNING MODEL
ML_model_settings={'RF': {'files':[f"{MODELS_DIR}/RandomForest/RF_Ud_models.pkl", f"{MODELS_DIR}/RandomForest/RF_Vd_models.pkl"], 'scaling':False, 'flipping': False},
                    'RF_FI':{'files':[f"{MODELS_DIR}/RandomForest/RF_Ud_FI_models.pkl", f"{MODELS_DIR}/RandomForest/RF_Vd_FI_models.pkl", f"{MODELS_DIR}/RandomForest/RF_FI_fit_models.pkl"], 'scaling':False, 'flipping': True},
                    'RF_coords': {'files':[f"{MODELS_DIR}/RandomForest/RF_Ud_coords_models.pkl", f"{MODELS_DIR}/RandomForest/RF_Vd_coords_models.pkl"], 'scaling':False, 'flipping': False},
}
                    # 'SVR': {'files':[f"{MODELS_DIR}/SVR/SVR_Ud_models.pkl", f"{MODELS_DIR}/SVR/SVR_Vd_models.pkl"], 'scaling':True}, 'flipping': False }

variables_ML=[['U_lp', 'V_lp', 'U_hp', 'V_hp'], ['U10', 'V10'], wave_variables.columns]

fieldsets_ML=[curr_field_rf, wind_field, wave_field]
time_starts_ML=[xr.open_dataset(curr_file_decomposed)['time'][0], xr.open_dataset(wind_file)['valid_time'][0], xr.open_dataset(wave_file)['time'][0]]

#%%

days=30
for i, node in enumerate(datatree.leaves):
    if i==6:
        drifter=node.name
        print(f'{i+1}/{len(datatree.leaves)}')
        for model in ML_model_settings:
            print(model)
            modelu = joblib.load(ML_model_settings[model]['files'][0])
            modelv = joblib.load(ML_model_settings[model]['files'][1])
            if ML_model_settings[model]['flipping']==True:
                fi_model=joblib.load(ML_model_settings[model]['files'][2])
            else:
                fi_model=None

            predicted_dict[drifter][model] = recreate_trajs_ML(datatree[drifter], dt, days, fieldsets_ML, time_starts_ML, modelu, modelv, variables_ML, fi_model )

# %%
#save data
dt_dict = {}

for drifter, models in predicted_dict.items():
    model_datasets = {}

    for model, variables in models.items():
        ds = xr.Dataset({var: ("time", values) for var, values in variables.items()})
        model_datasets[model] = ds  # Store dataset per model

    dt_dict[drifter] = xr.DataTree.from_dict(model_datasets)  # Convert models into DataTree

# Create final DataTree
datatree_pred = xr.DataTree.from_dict(dt_dict)

datatree_pred.to_netcdf(f'{PROJ_ROOT}/data/interim/predicted_trajectories_rf_lr.nc')  

# %%
drifter='2400'
for model in predicted_dict[drifter]:
    plt.scatter(predicted_dict[drifter][model]['lon'], predicted_dict[drifter][model]['lat'], label=model, s=2)
plt.plot(datatree[drifter]['lon'], datatree[drifter]['lat'], label='obs', color='black')
plt.ylim(53.5, 54.5)
plt.legend()
plt.show()
# %%
