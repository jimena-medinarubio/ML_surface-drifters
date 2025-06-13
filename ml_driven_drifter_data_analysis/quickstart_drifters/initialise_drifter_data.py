#%%
import json
import xarray as xr
import numpy as np
import pandas as pd
#%%
with open('path/to/file/data.json', 'r') as file:
    data = json.load(file)
# %%
#define drifter ids (numbered from 1 to 12)
labels={list(data.keys())[i]: i+1 for i in range(len(data.keys()))}

#%%
# Create dictionary assuming each drifter has variables: Time, Longitude, Latitude

drifter_datasets_dict = {}

for drifter_id, drifter_data in data.items():
    # Ensure time is datetime
    time = pd.to_datetime(drifter_data["Time"])
    
    # Create dataset for this drifter
    ds = xr.Dataset(
        {
            "lon": ("obs", drifter_data["Longitude"]),
            "lat": ("obs", drifter_data["Latitude"]),
            "time": ("obs", time),
            #add more variables if available in drifter data
        },
        coords={
            "obs": np.arange(0, len(time), 1)
        }
    )
    new_label=labels[drifter_id]
    drifter_datasets_dict[new_label] = ds

#%%
#OPTION A: CREATE SINGLE NETCDF FILE
# Combine all drifters along a new dimension: 'drifter'
ds = xr.concat(list(drifter_datasets_dict.values()), dim="trajectory")
ds = ds.assign_coords(trajectory=list(drifter_datasets_dict.keys()))

# %%
# OPTION B: CREATE DATATREE

# Convert keys to strings instead of floats
drifter_datasets_dict_str_keys = {str(k): v for k, v in drifter_datasets_dict.items()}

dt=xr.DataTree.from_dict(drifter_datasets_dict_str_keys)
# %%
