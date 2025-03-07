#%%

import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.patches as patches
import seaborn as sns

from processing_drifter_data import flipping_index_continuous, select_frequency_interval
#%%
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"

datatree=xr.open_datatree(f'{DATA_DIR}/interim/processed_drifter_data.nc')
# %%

def extract_variable(tree, var_name):
    return {node: data.get(var_name, None) for node, data in tree.items()}

delta_times=np.arange(1, 8, 1)
dict_flipping_index={str(delta_time):[] for delta_time in delta_times}

for delta_time in delta_times:
    dt=flipping_index_continuous(datatree, delta_time, sampling_frequencies=[300, 1800])
    dict_flipping_index[str(delta_time)]=extract_variable(dt, 'flipping_index_scaled')

# %%
example_drifter='0510'

def sensitivity_analysis_fi_plot(example_drifter, datatree, dict_flipping_index, delta_times, output_path=None, output_format='svg', name_fig='sensitivity_analysis_fi'):
    # Create the main plot
    plt.figure(figsize=(12, 7))

    # Color map for different lines
    color_map = [ '#A6CEE3', '#FF7F00',  'teal', '#E41A1C',  '#F781BF',  '#4DAF4A',  'royalblue', ]

    # Plot the main lines
    for idx, t in enumerate(delta_times):
        time= (datatree[example_drifter]['time'] - datatree[example_drifter]['time'][0]).astype('timedelta64[s]').astype(int)
        plt.plot(time.values/3600/24, dict_flipping_index[str(t)][example_drifter], label=f'{t} h', linewidth=2, color=color_map[idx], alpha=0.8)

    plt.legend(fontsize=14, loc='upper left')
    # Labels and formatting
    plt.xlabel('Time since release [days]', fontsize=14)
    plt.ylabel('Flipping Index', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Inset zoom: choose a region of interest (e.g., a peak area)
    # Adjust the zoom window to the region where the peaks are (example range)
    zoom_xlim = [5.5, 7.75]  # Define the x-limits for the zoom window (adjust these values based on your data)

    zoom_ylim = [0.13, 0.28]  # Define the y-limits for the zoom window (adjust these values based on your data)

    rect = patches.Rectangle((zoom_xlim[0], zoom_ylim[0]), zoom_xlim[1] - zoom_xlim[0], zoom_ylim[1] - zoom_ylim[0],
                            linewidth=1, edgecolor='black', facecolor='none', linestyle='--', label='Zoom-in Area')
    plt.gca().add_patch(rect)
    # Create inset axes for the zoomed-in area
    axins = inset_axes(plt.gca(), width="30%", height="30%", loc='upper right')  # Position the zoom window

    # Plot the zoomed-in region
    for idx, t in enumerate(delta_times):
        time= (datatree[example_drifter]['time'] - datatree[example_drifter]['time'][0]).astype('timedelta64[s]').astype(int)
        axins.plot(time.values/3600/24, dict_flipping_index[str(t)][example_drifter], linewidth=2, color=color_map[idx])

    # Set zoom area limits
    axins.set_xlim(zoom_xlim)
    axins.set_ylim(zoom_ylim)

    axins.tick_params(axis='both', labelsize=12) 

    # Draw a rectangle showing the zoom area on the main plot
    mark_inset(plt.gca(), axins, loc1=2, loc2=4, fc="none", ec="0.5")

    if output_path is None:
            PROJ_ROOT = Path(__file__).resolve().parents[1]
            FIGURES_DIR = PROJ_ROOT / "reports" / "figures"
            output_path = FIGURES_DIR / f'{name_fig}.{output_format}'

    plt.savefig(output_path, dpi=300)

    # Display the plot
    plt.show()
# %%
sensitivity_analysis_fi_plot(example_drifter, datatree, dict_flipping_index, delta_times)
# %%
