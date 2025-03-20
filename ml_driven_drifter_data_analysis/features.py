#%%
from pathlib import Path
from tqdm import tqdm
import xarray as xr
import numpy as np
import pandas as pd
from config import DATA_DIR
from scipy.special import erfc

#%%
dt_features=xr.open_datatree(f'{DATA_DIR}/interim/interpolated_atm_ocean_datasets.nc')

# %%

def calculate_stokes_drift_depth(u, Tp, z, beta=1, gamma=5.97):
        def calculate_k(u, tp, beta):
            return abs(u)/(2*abs(tp))*(1-2*beta/3)
        k=calculate_k(u, Tp, gamma)
        expo=np.exp(2*k*z)
        a= beta*np.sqrt(-2*k*np.pi*z )*erfc( np.sqrt(-2*k*z) )
        return u*(expo-a)

def transformations(df, filter_currents=True, waves=True):

    for var in ['Ustokes', 'Vstokes']:
        if var in df.columns:  # Check if the variable exists
            df[var] = calculate_stokes_drift_depth(df[var], df['Tp'], 0.5)


    wave_directions=['Wave_dir_wind', 'Wave_dir_swell', 'Wave_dir_swell_secondary', 'Wave_dir']
    if waves==True:
        for i, var in enumerate(['Hs_wind', 'Hs_swell', 'Hs_swell_secondary', 'Tp']):
            if var in df.columns:
                angle=(np.array(df[wave_directions[i]])+180) %360 
                df[f'{var}_x']=df[var]*np.sin(np.radians(angle))
                df[f'{var}_y']=df[var]*np.cos(np.radians(angle))
                df=df.drop(columns=[var, wave_directions[i]])
    
    if filter_currents:
        for var in ['U', 'V']:
            df = df.set_index('time')
            df[f'{var}_lp'] = df[var].rolling(window='24.83h', min_periods=1, center=True).mean()
            df[f'{var}_hp'] = df[var]-df[f'{var}_lp']
            df=df.drop(columns=[var])
            df = df.reset_index(drop=False)

    return df

def create_feature_matrix(dt_features, selected_vars, waves=True, fc=True):
    dfs = []
    for drifter_id, node in dt_features.children.items():
        ds_subset = node.ds[selected_vars]  # Select only chosen variables
        if 'time' not in ds_subset.coords:
            ds_subset['time'] = node.ds['time']  # Add time if not present in the subset
        
        df = ds_subset.to_dataframe() # Convert dataset to DataFrame
        df["drifter_id"] = drifter_id  # Add drifter ID
        
        df=df.reset_index()
        df=transformations(df, waves=waves, filter_currents=fc)

        df=df.set_index('time')

        dfs.append(df)

    # Combine all drifters into one DataFrame
    feature_matrix = pd.concat(dfs, ignore_index=False)

    if 'obs' in feature_matrix.columns or 'trajectory' in feature_matrix.columns:
        feature_matrix = feature_matrix.drop(columns=['obs', 'trajectory'])
    
    feature_matrix.columns = [str(col) for col in feature_matrix.columns]

    return feature_matrix

# %%
#example
feature_matrix=create_feature_matrix(dt_features, ['U10', 'V', 'U', 'Ustokes', 'Tp', 'Wave_dir'])

# %%
print(feature_matrix)
# %%
