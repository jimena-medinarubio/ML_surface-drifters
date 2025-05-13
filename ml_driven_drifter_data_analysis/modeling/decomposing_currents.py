#%%
import xarray as xr
import numpy as np
from config import DATA_DIR
#%%
curr_file=f'{DATA_DIR}/external/extended_curr_surface.nc' 


oc=xr.open_dataset(curr_file)

# Define the rolling mean function for low-pass filtering
def low_pass_moving_window(data, dt, time_window_hours):

    
    time_window = int(time_window_hours * 60 / (dt / np.timedelta64(1, 'm')))  # Time window in terms of time steps

    
    # Compute the rolling mean (low-pass component)
    low_pass = data.rolling(time=time_window).mean()

    # Calculate the tidal component (original - low-pass)
    tidal = data - low_pass
    
    return low_pass, tidal
# Decompose uo into low-pass and tidal components
oc['uo_lp'], oc['u_tides'] = low_pass_moving_window(oc['uo'], dt=np.timedelta64(15, 'm'),time_window_hours=24.83)  # assuming 1 minute spacing time_window='24.83h')

# Decompose vo into low-pass and tidal components
oc['vo_lp'], oc['v_tides'] = low_pass_moving_window(oc['vo'],  dt=np.timedelta64(15, 'm') , time_window_hours=24.83)
new_curr_files=f'{DATA_DIR}/external/decomposed_extended_curr_surface.nc'
oc.to_netcdf(new_curr_files)
# %%
