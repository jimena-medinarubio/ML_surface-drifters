#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import xarray as xr
from haversine import haversine, Unit
import sys
import cartopy.crs as ccrs
import seaborn as sns
sys.path.append("..")

from config import MODELS_DIR, DATA_DIR
from features import transformations, create_feature_matrix
from dataset import create_fieldset
import tqdm
from modeling.linear_regression import select_variables, linear_regression
import joblib
import pickle
#%%
PROJ_ROOT = Path(__file__).resolve().parents[2]
#dt_pred=xr.open_datatree(f'{PROJ_ROOT}/data/interim/predicted_trajectories_rf_lr.nc')
dt_obs=xr.open_datatree(f'{DATA_DIR}/interim/interpolated_atm_ocean_datasets_depth.nc')
dir=DATA_DIR/'processed'/'prediction'
#%%

pred_dict={}
#drifter.name:{} for drifter in dt_obs.leaves 
model_analysis=['RF', 'RF_FI', 'RF_FI_twostep','RF_residual', 'RF_coords', 'RF_depth', 'SVR', 'LR', 'LR_rw', 'sigmoid', 'charnock']

for drifter in dt_obs.leaves:
    model_datasets={}
    
    for model in model_analysis:
        with open(f"{dir}/{drifter.name}/{model}.pkl", "rb") as f:
            data=pickle.load(f)
        model_datasets[model]=xr.Dataset({var: ("time", values) for var, values in data.items()})
    pred_dict[drifter.name]=xr.DataTree.from_dict(model_datasets)

dt_pred=xr.DataTree.from_dict(pred_dict)

#%%

#dt_pred.to_netcdf(f'{DATA_DIR}/interim/predicted_trajectories.nc')

#%%


def MCSD(drifter_pred, drifter_obs, residual=False):
    
    obs_positions=np.vstack( (drifter_obs['lat'].values, drifter_obs['lon'].values)).T
    pred_positions=np.vstack( (drifter_pred['lat'].values, drifter_pred['lon'].values )).T
     
    if residual:
        time_drifter = pd.DatetimeIndex(drifter_obs['time'].values)
        drifter_df = pd.DataFrame({var:drifter_obs[var].values for var in drifter_obs.variables}, index=time_drifter)
        
        lat_residual=drifter_df['lat'].rolling(window='24.83h', center=True).mean()
        lon_residual=drifter_df['lon'].rolling(window='24.83h', center=True).mean()
        obs_positions=np.vstack( (lat_residual.values, lon_residual.values)).T

    distances = [haversine((obs[0], obs[1]), (sim[0], sim[1]), unit=Unit.KILOMETERS) for obs, sim in zip(obs_positions, pred_positions)]
   
    # Compute MCSD (Mean Cumulative Separation Distance)
    MCSD = np.sum(distances)/len(drifter_obs['time']) #km

    return MCSD, distances


def LiuWeisberg_ss(drifter_pred_block, drifter_obs_block, n=1):
    obs_positions = np.vstack((drifter_obs_block['lat'].values, drifter_obs_block['lon'].values)).T
    pred_positions = np.vstack((drifter_pred_block['lat'].values, drifter_pred_block['lon'].values)).T
    
    distances = [haversine((obs[0], obs[1]), (sim[0], sim[1]), unit=Unit.KILOMETERS) for obs, sim in zip(obs_positions, pred_positions)]

    traj_lengths = [haversine((obs[0], obs[1]), (sim[0], sim[1]), unit=Unit.KILOMETERS) for obs, sim in zip(obs_positions[1:], obs_positions[:-1])]
    
    cumulative_length = np.cumsum(traj_lengths)
    
    s = np.sum(distances) / np.sum(cumulative_length)
    LWS = 1 - s if s <= n else 0

    return LWS

def compute_LWS_timeseries(drifter_pred, drifter_obs,  n=1):
    # Ensure both DataFrames align
    time_coord = drifter_obs['time'].values[0]
    start_time = pd.to_datetime(time_coord)

    lws_series = []

    for i in range(1, len(drifter_pred['time']) - 1):
        print(start_time, drifter_pred['time'][i].values)
        #for i in tqdm.tqdm(range(len(drifter_pred['time']) - 1)):
        time = drifter_pred['time'][i].values

        obs_block = drifter_obs.sel(time=slice(start_time, time))
        pred_block = drifter_pred.sel(time=slice(start_time, time))
        
        if obs_block['time'].size > 1:
            lws = LiuWeisberg_ss(pred_block, obs_block, n=n)
            lws_series.append((time, lws))
        pass

    lws_df = pd.DataFrame(lws_series, columns=['time', 'LWS']).set_index('time')
    return lws_df

def plot_violin(trajectories,  colors, property='MCSD', ylabel='D [km]', labels=None, order=None):

    if order==None:
        data = {model: [] for model in trajectories['0510'].keys() if model != 'obs'}  # Dictionary to store MCSD values for each model
    else:
        data={model:[] for model in order}
        
    for drifter in trajectories:
        for model in data:
            data[model].append(trajectories[drifter][model][property].values)
    
    # Convert to list of lists format
    values = [np.array(data[model]).ravel() for model in data]
    print(values)
    # Create box plot
    plt.figure(figsize=(9, 6))
    sns.violinplot(data=values, width=0.6, palette=colors,  )

    # Customize plot
    if labels==None:
        plt.xticks(ticks=range(len(data)), labels=[model for model in data], fontsize=14, rotation=45)  # Set model names as x-axis labels
    else:
        plt.xticks(ticks=range(len(data)), labels=labels, fontsize=14, rotation=45, ha='right')  # Set model names as x-axis labels
    plt.ylabel(ylabel, fontsize=14)
    plt.yticks(fontsize=14,)

    # Show plot
    plt.show()

from scipy.stats import iqr
def plot_points_and_avg(trajectories, colors, property='MCSD', ylabel='Mean Cumulative Separation Distance [km]', labels=None, order=None, name=None):
    if order is None:
        data = {model: [] for model in trajectories['0510'].keys() if model != 'obs'}  # Dictionary to store MCSD values for each model
    else:
        data = {model: [] for model in order}
        
    for drifter in trajectories:
        for model in data:
            data[model].append(trajectories[drifter][model][property].values)
    
    # Convert to list of lists format
    values = [np.array(data[model]).ravel() for model in data]
    
    # Create a plot
    height=len(data)
    plt.figure(figsize=(10, height))
    
    # Plot points for each model
    for i, model in enumerate(data):
        sns.scatterplot(x=values[i] , y=[i] * len(values[i]), color=colors[i],  s=50, alpha=0.7)
        
        # Calculate and plot the average for each model
        avg_value = np.median(values[i])
        plt.plot([avg_value, avg_value], [i - 0.2, i + 0.2], color=colors[i], lw=3)  # Horizontal line showing the average
        plt.scatter( avg_value, i, color=colors[i], marker='o', s=100, edgecolor='black', zorder=5)  # Marker for average
        print(avg_value,iqr(values[i]))

    # Customize plot
    if labels is None:
        plt.yticks(ticks=range(len(data)), labels=[model for model in data], fontsize=14, rotation=45)  # Set model names as x-axis labels
    else:
        plt.yticks(ticks=range(len(data)), labels=labels, fontsize=14, )  # Set model names as x-axis labels
    
    plt.scatter([], [], color='white', marker='o', s=150, edgecolor='black', zorder=5, label='Median')  # Marker for average
  #  plt.scatter([], [], color='black', marker='o', s=100, edgecolor='black', zorder=5, label='Single drifter data')  # Marker for average

    plt.xlabel(ylabel, fontsize=14)
    plt.xticks(fontsize=14)
    if property=='MCSD':
        plt.xlim(0, np.max([np.max(values[i]) for i in range(len(data))]) + 10)
    plt.legend( fontsize=14, loc='best')

    plt.savefig(f'{PROJ_ROOT}/reports/figures/prediction/{property}_{name}.svg', dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()

def plot_prediction(drifter, labels, dt_obs, dt_pred, name=None):

    fig, ax = plt.subplots(figsize=(14, 12), subplot_kw={'projection': ccrs.PlateCarree()})


    for model_name in labels.keys():
        model=dt_pred[drifter][model_name]
        plt.plot(model['lon'], model['lat'], label=labels[model_name][0], color=labels[model_name][1], linewidth=2)
        ax.scatter(model['lon'].values[0],model['lat'].values[0], marker='*', s=240, color=labels[model_name][1], edgecolors='black' , zorder=3)
            # Plot the final position with a square marker
        ax.scatter(model['lon'].values[-1], model['lat'].values[-1], marker='^', s=240, color=labels[model_name][1], edgecolors='black', zorder=3)


    plt.plot(dt_obs[drifter]['lon'], dt_obs[drifter]['lat'], label='Observations', color='black', linestyle='--')
    ax.scatter(dt_obs[drifter]['lon'].values[0],dt_obs[drifter]['lat'].values[0], marker='*', s=240, color='black', edgecolors='black' , zorder=3)
            # Plot the final position with a square marker
    ax.scatter(dt_obs[drifter]['lon'].values[-1], dt_obs[drifter]['lat'].values[-1], marker='^', s=240, color='black', edgecolors='black', zorder=3)

    ax.add_feature(cfeature.LAND, edgecolor='w', facecolor='lightgrey')  # Specify edgecolor and facecolor

    ax.add_feature(cfeature.COASTLINE, color='k')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    # Set the labels for latitude and longitude
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    # Set tick label font size
    ax.tick_params(axis='both', labelsize=12)  # Change label size for both x and y axes

    # Add gridlines
    gridlines=ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--')
    gridlines.xlabel_style = {'size': 14}  # Font size for longitude labels
    gridlines.ylabel_style = {'size': 14}  # Font size for latitude labels
    gridlines.right_labels=False
    gridlines.top_labels=False
    plt.legend(fontsize=14, loc='best')

    plt.xlim(3, 9)
    plt.ylim(53, 56)
    fig.patch.set_alpha(0.0)         # Transparent figure background
    plt.savefig(f'{PROJ_ROOT}/reports/figures/prediction/trajs_{name}.svg', dpi=300, bbox_inches='tight', )

    # Show plot
    plt.show()

def plot_background(drifter, labels, dt_obs, dt_pred, name=None):

    fig, ax = plt.subplots(figsize=(14, 12), subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(cfeature.LAND, edgecolor='w', facecolor='lightgrey')  # Specify edgecolor and facecolor

    ax.add_feature(cfeature.COASTLINE, color='k')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    # Set the labels for latitude and longitude
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    # Set tick label font size
    ax.tick_params(axis='both', labelsize=12)  # Change label size for both x and y axes

    # Add gridlines
    gridlines=ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--')
    gridlines.xlabel_style = {'size': 14}  # Font size for longitude labels
    gridlines.ylabel_style = {'size': 14}  # Font size for latitude labels
    gridlines.right_labels=False
    gridlines.top_labels=False
    plt.legend(fontsize=14, loc='best')

    plt.xlim(3, 9)
    plt.ylim(53, 56)
    fig.patch.set_alpha(0.0)         # Transparent figure background
    plt.savefig(f'{PROJ_ROOT}/reports/figures/prediction/trajs_background.png', dpi=300, bbox_inches='tight', )

    # Show plot
    plt.show()


def willmott_skill_score(lons, lats, lons2_resampled, lats2_resampled):
    """
    Compute Willmott Skill Score (WMS) for longitude and latitude separately.
    
    Parameters:
    lons (array-like): Observed longitudes.
    lats (array-like): Observed latitudes.
    lons2_resampled (array-like): Simulated longitudes.
    lats2_resampled (array-like): Simulated latitudes.
    
    Returns:
    tuple: Willmott Skill Score for longitude and latitude.
    """
    # Calculate mean of observed and simulated positions
    mean_lon_obs = np.mean(lons)
    mean_lat_obs = np.mean(lats)

    
    # Calculate deviations from the mean for observed and simulated positions
    P_star_lon = lons2_resampled - mean_lon_obs
    O_star_lon = lons - mean_lon_obs
    P_star_lat = lats2_resampled - mean_lat_obs
    O_star_lat = lats - mean_lat_obs
    
    # Calculate the numerator for WMS (sum of squared differences)
    numerator_lon = np.sum((lons2_resampled - lons) ** 2)
    numerator_lat = np.sum((lats2_resampled - lats) ** 2)
    
    # Calculate the denominator for WMS (sum of squared absolute deviations)
    denominator_lon = np.sum((np.abs(P_star_lon) + np.abs(O_star_lon)) ** 2)    
    denominator_lat = np.sum((np.abs(P_star_lat) + np.abs(O_star_lat)) ** 2)   
    
    # Compute WMS for longitude and latitude
    WMS_lon = 1 - (numerator_lon / denominator_lon)
    WMS_lat = 1 - (numerator_lat / denominator_lat)

    
    return WMS_lon, WMS_lat


from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.patches as patches

def D_timeseries(labels, dt_pred, dt_obs, name=None, xlim=None):
    
    # Define common time base (in days since release)
    common_days = np.linspace(0, 66,len(dt_obs['6480']['time']))  # or adjust depending on your simulation length and resolution

    plt.figure(figsize=(10, 5))
    # Loop over all models

    models=[]
    lines=[]
    for model in labels.keys():
        aligned_D = []

        for drifter in dt_pred:
            
            da = dt_pred[drifter][model]['D']
            da = da.rename({'t': 'time'})

            valid_obs_times = dt_obs[drifter]['time'].where(dt_obs[drifter]['time'] <= dt_pred[drifter][model].time.max().values, drop=True)

            da = da.assign_coords({'time': valid_obs_times})

            time_since_release = (valid_obs_times - valid_obs_times[0]) / np.timedelta64(1, 'D')
            da = da.assign_coords(days_since_release=("time", time_since_release.data))
            
            # Interpolate onto common time base
            da = da.swap_dims({'time': 'days_since_release'})
            da_interp = da.interp(days_since_release=common_days)

            aligned_D.append(da_interp)

            plt.plot(time_since_release, da, alpha=0.1, color=labels[model][1], linewidth=2)
            

        if aligned_D:
            # Stack into single array
            D_all = xr.concat(aligned_D, dim='drifter')

            # Compute average
            D_mean = D_all.median(dim='drifter')

            # Plot
            plt.plot(common_days, D_mean, label=labels[model][0], color=labels[model][1], linewidth=2)
            models.append(D_mean)
            lines.append(aligned_D)
    plt.legend(fontsize=14, loc='upper right')

    plt.xlabel('Time since release [days]', fontsize=14)
    plt.ylabel('Cumulative Separation Distance [km]', fontsize=14)
    plt.ylim(0, 150)
    plt.xlim(0, xlim)

    zoom_xlim = [0, 7]  # Define the x-limits for the zoom window (adjust these values based on your data)

    zoom_ylim = [0, 30]  # Define the y-limits for the zoom window (adjust these values based on your data)

    rect = patches.Rectangle((zoom_xlim[0], zoom_ylim[0]), zoom_xlim[1] - zoom_xlim[0], zoom_ylim[1] - zoom_ylim[0],
                            linewidth=1, edgecolor='red', facecolor='none', linestyle='--', label='Zoom-in Area')
    plt.gca().add_patch(rect)
    # Create inset axes for the zoomed-in area
    axins = inset_axes(plt.gca(), width="30%", height="30%", loc='upper left', borderpad=5)  # Increase padding from edge)  # Position the zoom window
    # Set zoom area limits

    # Plot the zoomed-in region
    for i, model in enumerate(labels.keys()):
        axins.plot(common_days, models[i], color=labels[model][1], linewidth=2,)
        for line in lines[i]:
            axins.plot(common_days, line, alpha=0.1, color=labels[model][1], linewidth=2)

    axins.set_xlim(zoom_xlim)
    axins.set_ylim(zoom_ylim)
    axins.set_ylabel('CSD [km]', fontsize=12)
    axins.set_xlabel('Time [days]', fontsize=12)

    axins.tick_params(axis='both', labelsize=12) 
    mark_inset(plt.gca(), axins, loc1=2, loc2=4, fc="none", ec="0.4")
    #plt.xlim(0, 7)
    plt.savefig(f'{PROJ_ROOT}/reports/figures/prediction/MCSD_timseries_zoom_{name}.svg', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    return models
#%%

#calculate stats
for drifter in dt_pred:
    for model in dt_pred[drifter]:
        if 'residual' in model:
            res=True
        else:
            res=False

        valid_obs_times = dt_obs[drifter]['time'].where(dt_obs[drifter]['time'] <= dt_pred[drifter][model].time.max().values, drop=True)

        # Select predictions at the closest times
        predictions = dt_pred[drifter][model].sel(time=valid_obs_times, method='nearest')
    
       
        cumd, d=MCSD(predictions, dt_obs[drifter], residual=res)
        ss=LiuWeisberg_ss(predictions, dt_obs[drifter],)
        dt_pred[drifter][model]['MCSD']=xr.DataArray(cumd, dims=[])
        dt_pred[drifter][model]['ss']=xr.DataArray(ss, dims=[])
        dt_pred[drifter][model]['D']=xr.DataArray(d, dims=['t'])

# %%
labels=[ 'Support vector regression','Random forest', 'Linear regression',]
time=dt_pred['0510']['RF']['time'].values
plot_points_and_avg(dt_pred, [ '#9ab9f9','#009988',  '#DDAA33', ], 'MCSD', order=['SVR', 'RF',  'LR',], labels=labels, name='models')
# %%
labels=['Original', '+Flipping Index',  '+Lat & Lon', '+Bathymetry'][::-1]
plot_points_and_avg(dt_pred, [ '#009988', 'grey',  '#AA4499', '#332288',  ][::-1], 'MCSD', order=[ 'RF', 'RF_FI_twostep', 'RF_coords', 'RF_depth', ][::-1], labels=labels, name='RF_tests')
# %%
labels=['Charnock parametrisation', 'Sigmoid function: wind', 'Linear function: relative wind', 'Linear function: wind' ]
plot_points_and_avg(dt_pred, ['#6A041D', '#4BC6B9', '#EE3377', '#DDAA33', ], 'MCSD', order=['charnock', 'sigmoid', 'LR_rw', 'LR',], labels=labels, name='LR_tests_charnock')
#%%
import cartopy.feature as cfeature
import cartopy.crs as ccrs

drifter='3510'
labels={'LR': ['Linear regression, MCSD=27km','#DDAA33'],
        'RF': ['Random forest: MCSD=11km','#009988'] , 
         'SVR': ['Support vector regression: MCSD: 14km', '#9ab9f9']}

plot_prediction(drifter, labels, dt_obs, dt_pred, name='models_poster')

#%%
labels={'LR': ['Linear regression','#DDAA33'],
        'RF': ['Random forest','#009988'] , 
         'SVR': ['Support vector regression', '#9ab9f9']}

plot_prediction(drifter, labels, dt_obs, dt_pred, name='models')

#%%

labels={'RF': ['Total velocity','#009988'] , 
        'RF_FI_twostep': ['Total velocity, +Flipping Index','#CC3311'] , 
        'RF_coords': ['Total velocity, +lat & lon','#33BBEE'] , 
        'RF_depth': ['Total velocity, +bathymetry','#332288'] ,
         'RF_residual': [ 'Residual velocity','#AA4499'] ,  }

plot_prediction(drifter, labels, dt_obs, dt_pred, name='RF_tests')


#%%

drifter='0510'
labels={'LR': ['Linear function: wind','#DDAA33'],
        'LR_rw': ['Linear function: relative wind','#C44775'] , 
          'sigmoid': ['Sigmoid function: wind','#4BC6B9'],
           'charnock': ['Charnock param: wind','red'] }
plot_prediction(drifter, labels, dt_obs, dt_pred, name='LR_tests')


# %%
#   ss
labels={'LR': ['Linear Regression','#DDAA33'],
        'RF': ['Random Forest','#009988'] , 
         'SVR': ['Support Vector Regression', '#0077BB']}

lws_timeseries={model:[] for model in labels.keys()}
for i, model in enumerate(labels.keys()):
    lws_df = compute_LWS_timeseries(dt_pred[drifter][model].sel(time=dt_obs[drifter]['time'], method='nearest'), dt_obs[drifter], n=1)
    lws_timeseries[model]=lws_df
    
# %
# %%
model='LR'
i=660
end_time=dt_obs[drifter]['time'][i]
plt.plot(dt_pred[drifter][model]['lon'].sel(time=slice(dt_pred[drifter][model]['time'][0], end_time)), dt_pred[drifter][model]['lat'].sel(time=slice(dt_pred[drifter][model]['time'][0], end_time)), label=labels[model][0], color=labels[model][1], linewidth=2)
plt.plot(dt_obs[drifter]['lon'].isel(time=slice(0, i)), dt_obs[drifter]['lat'].isel(time=slice(0, i)),  linewidth=2)

# %%
model='LR'
palette = [
            "#CC6677",  # muted red
            "#332288",  # deep blue
            "#888888",  # medium gray (new)
            "#DDCC77",  # mustard
            "#FFAABB",
            "#117733",  # dark green
            "#882255",  # burgundy
            "#44AA99",  # turquoise
            "#999933",  # olive
            "#AA4499",  # magenta
            "#661100",  # dark brown (new)
            "#EE8866",
        ]
for i, node in enumerate(dt_obs.leaves):
    drifter=node.name
    plt.plot(dt_obs[drifter]['time'], dt_pred[drifter][model]['D'], color=palette[i] )

plt.xlabel('Time since release [days]', fontsize=14)
plt.ylabel('Separation Distance [km]', fontsize=14)
plt.title('Linear regression', fontsize=16)
# %%

labels={'LR': ['Linear regression','#DDAA33'],
        'RF': ['Random forest','#009988'] , 
         'SVR': ['Support vector regression', '#9ab9f9']}

mod=D_timeseries(labels, dt_pred, dt_obs, name='models')
#%%
labels={'LR': ['Linear function: wind','#DDAA33'],
        'LR_rw': ['Linear function: relative wind','#C44775'] , 
          'sigmoid': ['Sigmoid function: wind','#4BC6B9'] }
lrmodels=D_timeseries(labels, dt_pred, dt_obs, name='lr_tests', xlim=50)

# %%
labels=[ 'Support vector regression','Random forest', 'Linear regression',]
time=dt_pred['0510']['RF']['time'].values
plot_points_and_avg(dt_pred, [ '#0077BB','#009988',  '#DDAA33', ], 'wss_lon', ylabel='Longitude Willmott score', order=['SVR', 'RF',  'LR',], labels=labels, name='models_wss_lon')

# %%
plot_points_and_avg(dt_pred, [ '#0077BB','#009988',  '#DDAA33', ], 'wss_lat', ylabel='Latitude Willmott score', order=['SVR', 'RF',  'LR',], labels=labels, name='models_wss_lat')

#%%
labels={'LR': ['Linear regression','#DDAA33'],
        'RF': ['Random forest','#009988'] , 
         'SVR': ['Support vector regression', '#0077BB']}
plot_points_and_avg(dt_pred, [ '#0077BB','#009988',  '#DDAA33', ], 'ss', order=['SVR', 'RF',  'LR',], labels=labels, ylabel='Liu-Weisberg skill score',name='modelss')


# %%
labels={'RF': ['Original','#009988'] , 
       # 'RF_FI': ['Total velocity, +Flipping Index',] , 
         'RF_FI_twostep': ['+Flipping Index','grey'] , 
        'RF_coords': ['+Lat & lon','#AA4499'] , 
        'RF_depth': ['+Bathymetry','#332288'] ,}
       #  'RF_residual': [ 'Residual velocity','#AA4499'] ,  }

mod=D_timeseries(labels, dt_pred, dt_obs, name='test_RF')

# %%

# %%

labels=[ 'Support vector regression','Random forest', 'Linear regression',]
time=dt_pred['0510']['RF']['time'].values
plot_points_and_avg(dt_pred, [ '#9ab9f9','#009988',  '#DDAA33', ], 'ss', order=['SVR', 'RF',  'LR',], labels=labels, name='models_ss', ylabel='Liu-Weisberg skill score')

# %%
