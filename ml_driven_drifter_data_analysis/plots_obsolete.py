#%%
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import seaborn as sns
from config import FIGURES_DIR, PROCESSED_DATA_DIR


#%%

def plot_trajs(dt, name_fig='plot', output_path=None, palette=None, output_format='svg'):
        # Create a figure and axes with a cartopy projection
    fig, ax = plt.subplots(figsize=(14, 12), subplot_kw={'projection': ccrs.PlateCarree()})

    if palette is None:
        bright_palette = sns.color_palette("bright", 10)
        extra_colors = [(204/255, 153/255, 255/255),   # Bright Orange
                        (0, 153/255, 153/255)]   # Bright Teal

        # Extend the original palette by appending the new colors
        palette_array = bright_palette + extra_colors
        palette={drifter.name: palette_array[idx] for idx, drifter in enumerate(dt.leaves)}
    b='black'
    # Add land and ocean features
    land_feature = cfeature.LAND  # Reference to LAND feature
    ax.add_feature(land_feature, edgecolor='w', facecolor='lightgrey')  # Specify edgecolor and facecolor

    ax.add_feature(cfeature.COASTLINE, color='k')
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Iterate through each drifter and plot its trajectory
    for idx, node in enumerate(dt.leaves):
        drifter=node.ds
        # Plot the trajectory with a unique color
        ax.plot(drifter['lon'], drifter['lat'], color=palette[node.name], linewidth=1, zorder=2,)
        # Plot the initial position with a star marker
        ax.scatter(drifter['lon'].values[0],drifter['lat'].values[0], marker='*', s=240, color=palette[node.name], edgecolors=b , zorder=3)
        # Plot the final position with a square marker
        ax.scatter(drifter['lon'].values[-1], drifter['lat'].values[-1], marker='^', s=240, color=palette[node.name], edgecolors=b, zorder=3)

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
   

    if output_path is None:
        PROJ_ROOT = Path(__file__).resolve().parents[1]
        FIGURES_DIR = PROJ_ROOT / "reports" / "figures"
        output_path = FIGURES_DIR / f'{name_fig}.{output_format}'
    plt.savefig(output_path, dpi=300)

    plt.show()

# %%



def plot_trajs_bathymetry_poster(
    dt, 
    german_bight_bathymetry,  
    europe_bathymetry=None,   
    name_fig='plot', 
    output_path=None, 
    palette=None, 
    output_format='svg', ext=[-105, 40, -35, 70]
):
    fig = plt.figure(figsize=(23.4, 16.5))
    ax_inset = fig.add_subplot(111, projection=ccrs.PlateCarree())

    # Main map region (German Bight)
    ax_inset.set_extent([3, 9, 53, 56], crs=ccrs.PlateCarree())

    gb_bathy = german_bight_bathymetry['deptho'].values
    gb_bathy[np.isnan(gb_bathy)] = 0
    mesh = ax_inset.pcolormesh(
        german_bight_bathymetry['longitude'], 
        german_bight_bathymetry['latitude'], 
        gb_bathy, 
        cmap=cmocean.cm.ice_r,
        transform=ccrs.PlateCarree(),
        zorder=1
    )
   # ax_inset.add_feature(cfeature.LAND, color='lightgray', zorder=3)
    #ax_inset.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=4)
    from cartopy.feature import NaturalEarthFeature

    land = NaturalEarthFeature(
        'physical', 'land', '10m',
        edgecolor='face',
        facecolor='lightgray'
    )
    coastline = NaturalEarthFeature(
        'physical', 'coastline', '10m',
        edgecolor='black',
        facecolor='none'
    )

    ax_inset.add_feature(land, zorder=3)
    ax_inset.add_feature(coastline, linewidth=0.5, zorder=4)

    if palette is None:
        palette = [
            "#CC6677", "#332288", "#888888", "#DDCC77", "#FFAABB",
            "#117733", "#882255", "#44AA99", "#999933", "#AA4499", "#661100", "#EE8866",
        ]

    for idx, node in enumerate(dt.leaves):
        drifter = node.ds
        ax_inset.plot(
            drifter['lon'], drifter['lat'], 
            color=palette[idx % len(palette)],
            linewidth=1.5, 
            zorder=4
        )
        ax_inset.scatter(
            drifter['lon'].values[0], drifter['lat'].values[0], 
            marker='*', s=300, 
            color= palette[idx % len(palette)],
            edgecolor='black', zorder=5,
        )
        ax_inset.scatter(
            drifter['lon'].values[-1], drifter['lat'].values[-1], 
            marker='^', s=300, 
            color= palette[idx % len(palette)],
            edgecolor='black', zorder=5,
        )
    
    if output_path is None:
        PROJ_ROOT = Path(__file__).resolve().parents[2]
        FIGURES_DIR = PROJ_ROOT / "reports" / "figures"
        output_path = FIGURES_DIR / f'{name_fig}_trajs.{output_format}'
        plt.savefig(output_path, dpi=300, format=f'{output_format}', bbox_inches='tight')

    plt.show()
    #
    fig = plt.figure(figsize=(10, 6))
    ax_robinson = fig.add_subplot(111, projection=ccrs.Orthographic(central_longitude=6, central_latitude=54.5))

    # Plot the world map with grey land and blue ocean
    ax_robinson.add_feature(cfeature.LAND, color='lightgrey', zorder=1)
    ax_robinson.add_feature(cfeature.OCEAN, color='lightblue', zorder=2)
    ax_robinson.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)

    # Save or show
    output_path = FIGURES_DIR / f'{name_fig}_globe.svg'
    plt.savefig(output_path, dpi=300, format=f'{output_format}', bbox_inches='tight')
    plt.show()

#%%
def plot_europe_with_region(name_fig, output_path=None):
    # Create a map of Europe
    fig, ax = plt.subplots(figsize=(5, 6), subplot_kw={'projection': ccrs.Robinson()})
    ax.set_extent([-10, 30, 35, 70], crs=ccrs.PlateCarree())  # Extent for Europe

    # Add land and coastlines
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgrey', zorder=1)
    ax.add_feature(cfeature.COASTLINE, zorder=2)

    # Draw the delimited region
    region_x = [3, 9, 9, 3, 3]  # Longitude coordinates of the square
    region_y = [53, 53, 56, 56, 53]  # Latitude coordinates of the square
    ax.plot(region_x, region_y, color='red', linewidth=2, transform=ccrs.PlateCarree(), zorder=3)
   # ax.text(6, 54.5, 'Study Area', color='red', transform=ccrs.PlateCarree(),
           # ha='center', fontsize=10, zorder=4)

    # Add gridlines with labels
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linewidth=0.5)

    # Adjust gridline labels
    gl.xlabel_style = {'size': 12}  # Set font size for longitude labels
    gl.ylabel_style = {'size': 12}  # Set font size for latitude labels

    # Optionally adjust gridline appearance
    gl.xlines = False  # Remove vertical gridlines
    gl.ylines = False  # Remove horizontal gridlines

    # You can also choose whether to show the gridline labels at the top or right
    gl.top_labels = False
    gl.right_labels = False

    ax.tick_params(axis='both', labelsize=12)  # Change label size for both x and y axes
    if output_path is None:
        PROJ_ROOT = Path(__file__).resolve().parents[1]
        FIGURES_DIR = PROJ_ROOT / "reports" / "figures"
        output_path = FIGURES_DIR / f'{name_fig}.svg'
    plt.savefig(output_path, dpi=300)

    plt.show()
#%%


def plot_bathymetry(bathymetry_field, name_fig, output_path=None):
    df = {'deptho': bathymetry_field['deptho'].values}
    # Replace NaN values with 0
    df['deptho'][np.isnan(df['deptho'])] = 0

    fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    plt.pcolormesh(bathymetry_field['longitude'], bathymetry_field['latitude'], df['deptho'], cmap=cmocean.cm.gray , 
                   vmax=80, zorder=1, transform=ccrs.PlateCarree() )
    cbar = plt.colorbar(label='Depth [m]', ax=ax, shrink=0.4)
    cbar.ax.tick_params(labelsize=12)  # Adjust tick label size
    cbar.set_label('Depth [m]', fontsize=14)  # Adjust label font size

    # Add land and ocean features
    land_feature = cfeature.LAND  # Reference to LAND feature
    ax.add_feature(land_feature, edgecolor='w', facecolor='lightgrey', zorder=2)  # Specify edgecolor and facecolor

    ax.add_feature(cfeature.COASTLINE, color='k', zorder=3)
    ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=4)


    # Set the tick locations explicitly for longitude and latitude axes
    ax.set_xticks(np.arange(4, 9, 1))
    ax.set_yticks(np.arange(53.5, 56, 0.5))
    ax.set_xticklabels([f'{i}°E' for i in np.arange(4, 9, 1)])  # Longitude ticks
    ax.set_yticklabels([f'{i}°N' for i in np.arange(53.5, 56, 0.5)])  # Latitude ticks
    ax.tick_params(axis='both', labelsize=12)  # Change label size for both x and y axes

    plt.xlim(3, 9)
    plt.ylim(53, 56)
    # Show the plot
    plt.tight_layout()

    if output_path is None:
        PROJ_ROOT = Path(__file__).resolve().parents[1]
        FIGURES_DIR = PROJ_ROOT / "reports" / "figures"
        output_path = FIGURES_DIR / f'{name_fig}.png'
    plt.savefig(output_path, dpi=300)
    plt.show()

#%%

def plot_trajs_bathymetry(dt, bathymetry_field, name_fig='plot', output_path=None, palette=None, output_format='svg'):
    # Create a figure and axes with a cartopy projection
    fig, ax = plt.subplots(figsize=(15, 18), subplot_kw={'projection': ccrs.PlateCarree()})

    if palette is None:
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

    b = 'black'

    df = {'deptho': bathymetry_field['deptho'].values}
    # Replace NaN values with 0
    df['deptho'][np.isnan(df['deptho'])] = 0

    plt.pcolormesh(bathymetry_field['longitude'], bathymetry_field['latitude'], df['deptho'], cmap=cmocean.cm.ice.reversed(),
                   vmax=80, zorder=1, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(label='Depth [m]', ax=ax, shrink=0.4)
    cbar.ax.tick_params(labelsize=12)  # Adjust tick label size
    cbar.set_label('Depth [m]', fontsize=14)  # Adjust label font size

    # Add land and ocean features
    land_feature = cfeature.LAND  # Reference to LAND feature
    ax.add_feature(land_feature, edgecolor='grey', facecolor='lightgrey', zorder=2)  # Specify edgecolor and facecolor

    ax.add_feature(cfeature.COASTLINE, color='grey', zorder=2)

    # Iterate through each drifter and plot its trajectory
    for idx, node in enumerate(dt.leaves):
        drifter = node.ds
        # Plot the trajectory with a unique color
        ax.plot(drifter['lon'], drifter['lat'], color=palette[idx], linewidth=1.2, zorder=3)
        # Plot the initial position with a star marker
        ax.scatter(drifter['lon'].values[0], drifter['lat'].values[0], marker='*', s=240, color=palette[idx], edgecolors=b, zorder=4)
        # Plot the final position with a square marker
        ax.scatter(drifter['lon'].values[-1], drifter['lat'].values[-1], marker='^', s=240, color=palette[idx], edgecolors=b, zorder=4)

    # Set the labels for latitude and longitude
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)  # Change label size for both x and y axes

    # Add gridlines
    gridlines = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree())
    gridlines.xlabel_style = {'size': 14}  # Font size for longitude labels
    gridlines.ylabel_style = {'size': 14}  # Font size for latitude labels
    gridlines.right_labels = False
    gridlines.xlines = None
    gridlines.ylines = None
    gridlines.top_labels = False
    ax.plot([], [], marker='*', markersize=14, color=b, label='Start Position', )  # Star for start
    ax.plot([], [], marker='^', markersize=14, color=b, label='End Position',)    # Square for end

    # Set the legend
    plt.legend(fontsize=14, loc='best')

    plt.xlim(3, 9)
    plt.ylim(53, 56)

    # Show the plot
  #  plt.tight_layout()

    if output_path is None:
        PROJ_ROOT = Path(__file__).resolve().parents[2]
        FIGURES_DIR = PROJ_ROOT / "reports" / "figures"
        output_path = FIGURES_DIR / f'{name_fig}.{output_format}'
        plt.savefig(output_path, dpi=300, format=f'{output_format}', bbox_inches='tight')

