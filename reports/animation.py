#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import xarray as xr
import sys
import seaborn as sns
import cmocean

sys.path.append("..")
from config import MODELS_DIR, DATA_DIR
import joblib
import pickle
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
#%%

PROJ_ROOT = Path(__file__).resolve().parents[2]
dt_pred=xr.open_datatree(f'{PROJ_ROOT}/data/interim/predicted_trajectories.nc')
dt_obs=xr.open_datatree(f'{DATA_DIR}/interim/interpolated_atm_ocean_datasets_depth.nc')
bathymetry=xr.open_dataset(f'{DATA_DIR}/external/bathymetry.nc')
# %%


def animate_prediction(bathymetry_field, drifter, model_name, label, color, dt_obs, dt_pred, name=None, save=True, i=200, step_minutes=30, dutch=False):
    """
    Animate the trajectory of one model vs. observations using regular time steps.
    
    Parameters:
    - drifter: str, ID of the drifter
    - model_name: str, key for the model to animate
    - label: str, label for legend
    - color: str, color for the model trajectory
    - dt_obs: dict, observed data
    - dt_pred: dict, predicted data for all models
    - name: str, optional filename to save
    - save: bool, whether to save the gif
    - i: int, interval between frames in milliseconds
    - step_minutes: int, regular time step in minutes for animation
    """
    obs = dt_obs[drifter]
    model = dt_pred[drifter][model_name]

    # Create regular time steps
    t_start = obs['time'][0].values
    t_end = obs['time'][-1].values
    step = np.timedelta64(step_minutes, 'm')
    anim_times = np.arange(t_start, t_end, step)

    # Find nearest indices in obs/model to each animation time
    obs_times = obs['time'].values
    model_times = model['time'].values
    obs_indices = np.searchsorted(obs_times, anim_times)
    model_indices = np.searchsorted(model_times, anim_times)
    obs_indices = np.clip(obs_indices, 0, len(obs_times) - 1)
    model_indices = np.clip(model_indices, 0, len(model_times) - 1)

    fig, ax = plt.subplots(figsize=(14, 12), subplot_kw={'projection': ccrs.PlateCarree()})

    df = {'deptho': bathymetry_field['deptho'].values}
    # Replace NaN values with 0
    df['deptho'][np.isnan(df['deptho'])] = 0

    plt.pcolormesh(bathymetry_field['longitude'], bathymetry_field['latitude'], df['deptho'], cmap=cmocean.cm.ice.reversed(),
                   vmax=80, zorder=1, transform=ccrs.PlateCarree(), alpha=0.95)
    cbar = plt.colorbar(label='Depth [m]', ax=ax, shrink=0.48)
    cbar.ax.tick_params(labelsize=12)  # Adjust tick label size

    if dutch:
        cbar.set_label('Diepte [m]', fontsize=14)
    else:
        cbar.set_label('Depth [m]', fontsize=14)  # Adjust label font size

    ax.add_feature(cfeature.LAND, edgecolor='w', facecolor='lightgrey', zorder=2)
    ax.add_feature(cfeature.COASTLINE, color='k', zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=2)
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.set_xlim(3, 9)
    ax.set_ylim(53, 56)

    gridlines = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--')
    gridlines.xlabel_style = {'size': 14}
    gridlines.ylabel_style = {'size': 14}
    gridlines.right_labels = False
    gridlines.top_labels = False
    gridlines.xlines=None
    gridlines.ylines=None

    if dutch:
        obs_line, = ax.plot([], [], color='black', label='Observaties')
    else:
        obs_line, = ax.plot([], [], color='black', label='Observations')
    obs_star = ax.scatter([], [], marker='*', s=240, color='black', edgecolors='black', zorder=3)

    model_line, = ax.plot([], [], color=color, linewidth=2, linestyle='--', label=label)
    model_star = ax.scatter([], [], marker='*', s=240, color=color, edgecolors='black', zorder=3)

    title=ax.set_title(f'Time: {pd.to_datetime(anim_times[0]).strftime("%Y-%m-%d %H:%M")}', fontsize=16)

    plt.legend(fontsize=14, loc='best')

    def init():
        model_line.set_data([], [])
        obs_line.set_data([], [])
        title.set_text('')
        return [model_line, obs_line, title]

    def update(frame):
        i_obs = obs_indices[frame]
        i_model = model_indices[frame]

        model_line.set_data(model['lon'][:i_model+1], model['lat'][:i_model+1])
        model_star.set_offsets([[model['lon'][i_model].values, model['lat'][i_model].values]])

        obs_line.set_data(obs['lon'][:i_obs+1], obs['lat'][:i_obs+1])
        obs_star.set_offsets([[obs['lon'][i_obs].values, obs['lat'][i_obs].values]])

        # Update title with current timestamp
        current_time = pd.to_datetime(anim_times[frame])
        if dutch:
            title.set_text(f'Tijd: {current_time.strftime("%Y-%m-%d %H:%M")}')
        else:
            title.set_text(f'Time: {current_time.strftime("%Y-%m-%d %H:%M")}')

        return [model_line, model_star, obs_line, obs_star, title]

    ani = animation.FuncAnimation(fig, update, frames=len(anim_times), init_func=init,
                                  blit=True, repeat=False, interval=i)

    if save:
        ani.save(f'{PROJ_ROOT}/reports/figures/prediction/trajs_{name}.gif', dpi=300, writer='pillow')

    plt.show()
# %%

drifter='2400'
animate_prediction(bathymetry, drifter, 'RF', 'Machine Learning model', "red", dt_obs.sel(time=slice(dt_pred[drifter]['RF']['time'][0], dt_pred[drifter]['RF']['time'][-1].values)), dt_pred.sel(time=dt_obs[drifter]['time'], method='nearest'), name='animation', save=True, i=100, step_minutes=6*60)
#%%

animate_prediction(bathymetry, drifter, 'RF', 'Machine Learning model', "red", dt_obs.sel(time=slice(dt_pred[drifter]['RF']['time'][0], dt_pred[drifter]['RF']['time'][-1].values)), dt_pred.sel(time=dt_obs[drifter]['time'], method='nearest'), name='animation_dutch', save=True, i=100, step_minutes=6*60, dutch=True)


# %%

def animate_prediction_two_drifters(bathymetry_field, drifter1, drifter2, model_name1,
                                    color1, color2,
                                    dt_obs, dt_pred, name=None, save=True, i=200, step_minutes=30):
    """
    Animate the trajectory of two drifters (observed vs predicted) using regular time steps.
    """
    model_name2=model_name1
    obs1 = dt_obs[drifter1]
    model1 = dt_pred[drifter1][model_name1]
    obs2 = dt_obs[drifter2]
    model2 = dt_pred[drifter2][model_name2]

    # Create common animation time range based on intersection
    t_start = max(obs1['time'][0].values, obs2['time'][0].values)
    t_end = min(obs1['time'][-1].values, obs2['time'][-1].values)
    step = np.timedelta64(step_minutes, 'm')
    anim_times = np.arange(t_start, t_end, step)

    obs1_indices = np.searchsorted(obs1['time'].values, anim_times)
    model1_indices = np.searchsorted(model1['time'].values, anim_times)
    obs2_indices = np.searchsorted(obs2['time'].values, anim_times)
    model2_indices = np.searchsorted(model2['time'].values, anim_times)

    obs1_indices = np.clip(obs1_indices, 0, len(obs1['time']) - 1)
    model1_indices = np.clip(model1_indices, 0, len(model1['time']) - 1)
    obs2_indices = np.clip(obs2_indices, 0, len(obs2['time']) - 1)
    model2_indices = np.clip(model2_indices, 0, len(model2['time']) - 1)

    fig, ax = plt.subplots(figsize=(14, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Plot background
    df = {'deptho': bathymetry_field['deptho'].values}
    df['deptho'][np.isnan(df['deptho'])] = 0
    plt.pcolormesh(bathymetry_field['longitude'], bathymetry_field['latitude'], df['deptho'],
                   cmap=cmocean.cm.ice.reversed(), vmax=80, zorder=1,
                   transform=ccrs.PlateCarree(), alpha=0.95)
    
    ax.add_feature(cfeature.LAND, edgecolor='w', facecolor='lightgrey', zorder=2)
    ax.set_xlim(3, 9)
    ax.set_ylim(53, 56)
    ax.set_xticks(np.arange(3, 10, 1), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(53, 56, 0.5), crs=ccrs.PlateCarree())
    ax.tick_params(labelbottom=False, labelleft=False)  # hide labels

    # Plot lines & markers
    #REFERENCES
    ax.plot([], [], color='black', label=f'Observations')
    ax.plot([], [], color='black', linestyle='--', label='ML model')

    obs1_line, = ax.plot([], [], color=color1,)
    model1_line, = ax.plot([], [], color=color1, linestyle='--',)

    obs1_star = ax.scatter([], [], color=color1, marker='*',edgecolors='black', s=240, zorder=3)
    model1_star = ax.scatter([], [], edgecolor=color1, facecolor='none', marker='*', s=240,  zorder=3,)

    obs2_line, = ax.plot([], [], color=color2)
    model2_line, = ax.plot([], [], color=color2, linestyle='--', )
    obs2_star = ax.scatter([], [], color=color2, marker='*', s=240, edgecolors='black', zorder=3)
    model2_star = ax.scatter([], [],  marker='*', s=240, edgecolors=color2, zorder=3, facecolor='none')

    title = ax.set_title('', fontsize=16)
    plt.legend(fontsize=12, loc='best')

    def init():
        for artist in [obs1_line, model1_line, obs2_line, model2_line]:
            artist.set_data([], [])
        title.set_text('')
        return [obs1_line, model1_line, obs2_line, model2_line, obs1_star, model1_star, obs2_star, model2_star, title]

    def update(frame):
        i1o, i1m = obs1_indices[frame], model1_indices[frame]
        i2o, i2m = obs2_indices[frame], model2_indices[frame]

        obs1_line.set_data(obs1['lon'][:i1o+1], obs1['lat'][:i1o+1])
        model1_line.set_data(model1['lon'][:i1m+1], model1['lat'][:i1m+1])
        obs1_star.set_offsets([[obs1['lon'][i1o], obs1['lat'][i1o]]])
        model1_star.set_offsets([[model1['lon'][i1m], model1['lat'][i1m]]])

        obs2_line.set_data(obs2['lon'][:i2o+1], obs2['lat'][:i2o+1])
        model2_line.set_data(model2['lon'][:i2m+1], model2['lat'][:i2m+1])
        obs2_star.set_offsets([[obs2['lon'][i2o], obs2['lat'][i2o]]])
        model2_star.set_offsets([[model2['lon'][i2m], model2['lat'][i2m]]])

        time_now = pd.to_datetime(anim_times[frame])
        title.set_text(f'Time: {time_now.strftime("%Y-%m-%d %H:%M")}')
        return [obs1_line, model1_line, obs1_star, model1_star,
                obs2_line, model2_line, obs2_star, model2_star, title]

    ani = animation.FuncAnimation(fig, update, frames=len(anim_times), init_func=init,
                                  blit=True, repeat=False, interval=i)

    if save:
        filename = f"{name or f'{drifter1}_{drifter2}'}_comparison.gif"
        ani.save(f'{PROJ_ROOT}/reports/figures/prediction/{filename}', dpi=300, writer='pillow')

    plt.show()
# %%
animate_prediction_two_drifters(bathymetry, '6530', '5490', 'RF', '#D55E00', '#CC79A7', dt_obs.sel(time=slice(dt_pred[drifter]['RF']['time'][0], dt_pred[drifter]['RF']['time'][-1].values)), dt_pred.sel(time=dt_obs[drifter]['time'], method='nearest'), name='animation', save=True, i=100, step_minutes=6*60)

# %%
