#%%

import xarray as xr
import pandas as pd
from pathlib import Path
from parcels import FieldSet, Geographic, GeographicPolar
from processing_drifter_data import select_frequency_interval

#%%
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"

datatree=xr.open_datatree(f'{DATA_DIR}/interim/processed_drifter_data.nc')

wave_file=f'{DATA_DIR}/external/waves-wind-swell.nc' 
wind_file=f'{DATA_DIR}/external/wind.nc' 
curr_file=f'{DATA_DIR}/external/ocean_currents.nc' 

wave_variables=pd.read_csv(f'{PROJ_ROOT}/references/waves_dataset.csv')

#%#

def create_fieldset(file, time_variable, vars, vars_names):
    filename= {vars_names[i]: file for i in range(len(vars_names))}
    variables = {vars_names[i]: vars[i] for i in range(len(vars_names))}
    dimensions={ vars_names[i]: {'time': time_variable, 'lat': 'latitude', 'lon':'longitude'} for i in range(len(vars_names))}

    fieldset = FieldSet.from_netcdf(filename, variables, dimensions, deferred_load=False,  mesh="spherical")

    
    for var in vars_names:
        if 'U' in var:
            getattr(fieldset, var).units = GeographicPolar()
        elif 'V' in var:
            getattr(fieldset, var).units =Geographic()

    return fieldset

#%%
def interpolate_datasets( fieldsets, datatree, time_starts, variables):

    mutable_ds={node.name: [] for node in datatree.leaves}

    for node in datatree.leaves:
        ds=node.ds
        index=select_frequency_interval(datatree[node.name]['time'], 1800)

        # Apply resampling to this part
        ds_finet= ds.isel(time=slice(0, index))  
        new_time = pd.date_range(ds_finet.time[0].values, ds_finet.time[-1].values, freq="30min")
        # Interpolate to match new timestamps
        ds_finet_resampled = ds_finet.interp(time=new_time)
        ds_coarset= ds.isel(time=slice(index, None))  
        ds_combined = xr.concat([ ds_finet_resampled,  ds_coarset], dim="time")
        
        for f, fieldset in enumerate(fieldsets):
            time=(ds_combined['time'] - time_starts[f]).astype('timedelta64[s]').astype(int)
            for var in variables[f]:
                a=[getattr(fieldset, var).eval(time.values[i], 0, ds_combined['lat'].values[i], ds_combined['lon'].values[i], applyConversion=False)
                    for i in range(len(time))] 
                ds_combined[var]=xr.DataArray(a, dims='time', coords={'time': ds_combined['time']})

        mutable_ds[node.name]=ds_combined
    
    return xr.DataTree.from_dict(mutable_ds)
#%%

# Only execute this if running directly (not on import)
if __name__ == "__main__":
    PROJ_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJ_ROOT / "data"

    datatree = xr.open_datatree(f'{DATA_DIR}/interim/processed_drifter_data.nc')

    wave_file = f'{DATA_DIR}/external/waves-wind-swell.nc' 
    wind_file = f'{DATA_DIR}/external/wind.nc' 
    curr_file = f'{DATA_DIR}/external/ocean_currents.nc' 

    wave_variables = pd.read_csv(f'{PROJ_ROOT}/references/waves_dataset.csv')

    wave_field = create_fieldset(wave_file, 'time', wave_variables.values[0], wave_variables.columns)
    wind_field = create_fieldset(wind_file, 'valid_time', ['u10', 'v10'], ['U10', 'V10'])
    curr_field = create_fieldset(curr_file, 'time', ['uo', 'vo'], ['U', 'V'])

    fieldsets = [curr_field, wind_field, wave_field]
    time_starts = [
        xr.open_dataset(curr_file)['time'][0],
        xr.open_dataset(wind_file)['valid_time'][0],
        xr.open_dataset(wave_file)['time'][0]
    ]
    variables = [['U', 'V'], ['U10', 'V10'], wave_variables.columns]

    dt_interpolated = interpolate_datasets(fieldsets, datatree, time_starts, variables)

    dt_interpolated.to_netcdf(f'{PROJ_ROOT}/data/interim/interpolated_atm_ocean_datasets.nc') 
# %%
