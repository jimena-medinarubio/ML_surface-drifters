#%%
import xarray as xr
import pandas as pd
from pathlib import Path
from parcels import FieldSet, Geographic, GeographicPolar

#%%
import sys
sys.path.append("..")
PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "data"
sys.path.append(str(PROJ_ROOT))  # Add project root to sys.path
from processing_drifter_data import select_frequency_interval

#%%
#specify the files to be used
wave_file=f'{DATA_DIR}/external/waves-wind-swell.nc' 
wind_file=f'{DATA_DIR}/external/wind.nc' 
curr_file=f'{DATA_DIR}/external/ocean_currents.nc' 
bathymetry_file=f'{DATA_DIR}/external/bathymetry.nc' 

#drifters data
datatree_file= f'{DATA_DIR}/interim/processed_drifter_data.nc'
wave_variables=f'{PROJ_ROOT}/references/waves_dataset.csv'
saving_file=f'{PROJ_ROOT}/data/interim/interpolated_atm_ocean_datasets_depth.nc'

#%%

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

def create_static_fieldset(file, vars, vars_names):
    filename= {vars_names[i]: file for i in range(len(vars_names))}
    variables = {vars_names[i]: vars[i] for i in range(len(vars_names))}
    dimensions={ vars_names[i]: {'lat': 'latitude', 'lon':'longitude'} for i in range(len(vars_names))}

    fieldset = FieldSet.from_netcdf(filename, variables, dimensions, deferred_load=False,  mesh="spherical")
    return fieldset

#%%
def interpolate_datasets( fieldsets, datatree, time_starts, variables):

    mutable_ds={node.name: [] for node in datatree.leaves}

    for node in datatree.leaves:
        ds=node.ds
        index=select_frequency_interval(datatree[node.name]['time'], 5*60) #index when sampling frequency stops being 30 mins

        # Apply resampling to this part
        ds_finet= ds.isel(time=slice(0, index))  
        new_time = pd.date_range(ds_finet.time[0].values, ds_finet.time[-1].values, freq="30min")
        # Interpolate to match new timestamps
        ds_finet_resampled = ds_finet.interp(time=new_time)

        ds_coarset= ds.isel(time=slice(index, None))  
        ds_combined = xr.concat([ ds_finet_resampled,  ds_coarset], dim="time")
        
        for f, fieldset in enumerate(fieldsets):
            if time_starts[f]!=None:
                time=(ds_combined['time'] - time_starts[f]).astype('timedelta64[s]').astype(int)
                for var in variables[f]:
                    a=[getattr(fieldset, var).eval(time.values[i], 0, ds_combined['lat'].values[i], ds_combined['lon'].values[i], applyConversion=False)
                        for i in range(len(time))] 
                    ds_combined[var]=xr.DataArray(a, dims='time', coords={'time': ds_combined['time']})
            else: #bathymetry
                for var in variables[f]:
                    a=[getattr(fieldset, var).eval(0, 0, ds_combined['lat'].values[i], ds_combined['lon'].values[i], applyConversion=False)
                        for i in range(len(time))] 
                    ds_combined[var]=xr.DataArray(a, dims='time', coords={'time': ds_combined['time']})

        mutable_ds[node.name]=ds_combined
    
    return xr.DataTree.from_dict(mutable_ds)
#%%

# Only execute this if running directly (not on import)
if __name__ == "__main__":
    datatree=xr.open_datatree(datatree_file)
    wave_variables=pd.read_csv(wave_variables)

    #create parcels
    wave_field = create_fieldset(wave_file, 'time', wave_variables.values[0], wave_variables.columns)
    wind_field = create_fieldset(wind_file, 'valid_time', ['u10', 'v10'], ['U10', 'V10'])
    curr_field = create_fieldset(curr_file, 'time', ['uo', 'vo'], ['U', 'V'])
    bathymetry_field = create_static_fieldset(bathymetry_file, ['deptho'], ['z'])
    fieldsets = [curr_field, wind_field, wave_field, bathymetry_field]

    #specify starting times
    time_starts = [
        xr.open_dataset(curr_file)['time'][0],
        xr.open_dataset(wind_file)['valid_time'][0],
        xr.open_dataset(wave_file)['time'][0], None
    ]
    #specify variables to be interpolated
    variables = [['U', 'V'], ['U10', 'V10'], wave_variables.columns, 'z']

    #interpolate data
    dt_interpolated = interpolate_datasets(fieldsets, datatree, time_starts, variables)

    # Save the interpolated dataset
    dt_interpolated.to_netcdf(saving_file) 
# %%
