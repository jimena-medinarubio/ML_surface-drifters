#%%
from pathlib import Path
import typer
from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import seaborn as sns

from ml_driven_drifter_data_analysis.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()

def plot_trajs(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    dt: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
        # Create a figure and axes with a cartopy projection
    fig, ax = plt.subplots(figsize=(14, 12), subplot_kw={'projection': ccrs.PlateCarree()})

    bright_palette = sns.color_palette("bright", 10)
    extra_colors = [(204/255, 153/255, 255/255),   # Bright Orange
                    (0, 153/255, 153/255)]   # Bright Teal

    # Extend the original palette by appending the new colors
    palette = bright_palette + extra_colors
    b='black'
    # Add land and ocean features
    land_feature = cfeature.LAND  # Reference to LAND feature
    ax.add_feature(land_feature, edgecolor='w', facecolor='lightgrey')  # Specify edgecolor and facecolor

    ax.add_feature(cfeature.COASTLINE, color='k')
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Iterate through each drifter and plot its trajectory
    for idx, drifter_id in enumerate(dt.leaves):
        # Plot the trajectory with a unique color
        ax.plot(dt[drifter_id].lons, dt[drifter_id].lats, color=palette[idx], linewidth=1, zorder=2,)
        # Plot the initial position with a star marker
        ax.scatter(dt[drifter_id].lons.values[0], dt[drifter_id].lats.values[0], marker='*', s=240, color=palette[idx], edgecolors=b , zorder=3)
        # Plot the final position with a square marker
        ax.scatter(dt[drifter_id].lons.values[-1], dt[drifter_id].lats.values[-1], marker='^', s=240, color=palette[idx], edgecolors=b, zorder=3)

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
    ax.plot([], [], marker='*', markersize=14, color=b, label='Start Position', )  # Star for start
    ax.plot([], [], marker='^', markersize=14, color=b, label='End Position',)    # Square for end

    # Set the legend
    plt.legend(fontsize=14, loc='best')

    plt.xlim(3, 9)
    plt.ylim(53, 56)

    # Show the plot
    plt.tight_layout()
    plt.show()
    plt.savefig(output_path, dpi=300)

if __name__ == "__main__":
    app()
