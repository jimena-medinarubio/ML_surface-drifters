#%%
from pathlib import Path
from tqdm import tqdm
import xarray as xr
import numpy as np
import pandas as pd
from scipy.special import erfc
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import cmocean
#%%
import sys
sys.path.append("..")
from config import INTERIM_DATA_DIR, PROJ_ROOT, REFS_DIR

#%%
interpolated_data_file=f'{INTERIM_DATA_DIR}/interpolated_atm_ocean_datasets.nc'
labels_file=f'{REFS_DIR}/variables_labels_residual.csv'

# %%
def calculate_stokes_drift_depth(u, Tp, z, beta=1, gamma=5.97):

    """
    calculate stokes drift at depth z using deep water wave theory
    """

    def calculate_k(u, tp, beta):
        return abs(u)/(2*abs(tp))*(1-2*beta/3)
    k=calculate_k(u, Tp, gamma)
    expo=np.exp(2*k*z)
    a= beta*np.sqrt(-2*k*np.pi*z )*erfc( np.sqrt(-2*k*z) )
    return u*(expo-a)

def transformations(df, filter_currents=True, waves=True, period='24.83h'):
    """
    transformations to be applied to the DataFrame:
    - calculate Stokes drift at depth z=0.5m for zonal and meridional components
    - calculate wave components in x and y directions
    - filter currents using a low-pass filter with a window of 24.83 hours
    """

    for var in ['Ustokes', 'Vstokes']:
        if var in df.columns:  # check if the variable exists
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
            df[f'{var}_lp'] = df[var].rolling(window=period, min_periods=1, center=True).mean()
            df[f'{var}_hp'] = df[var]-df[f'{var}_lp']
            df=df.drop(columns=[var])
            df = df.reset_index(drop=False)

    return df

def create_feature_matrix(dt_features, selected_vars, waves=True, fc=True):
    """
    create feature matrix of interpolated datasets at drifter location
    """

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

    # combine all drifters into one DataFrame
    feature_matrix = pd.concat(dfs, ignore_index=False)

    #remove obs & trajectory vars from matrix
    if 'obs' in feature_matrix.columns:
        feature_matrix = feature_matrix.drop(columns=['obs'])
    if 'trajectory' in feature_matrix.columns:
        feature_matrix = feature_matrix.drop(columns=['trajectory'])
    
    feature_matrix.columns = [str(col) for col in feature_matrix.columns]

    return feature_matrix

def plot_matrix(X_drifter, labels):
    
    """
    Plot a matrix with correlation plots for each pair of features in the DataFrame X_drifter.
    """
    n_features = X_drifter.shape[1]
    feature_names = X_drifter.columns[::-1]

    fig, axes = plt.subplots(n_features, n_features, figsize=(20, 20))

    hist_thermal = None
    hist_dense = None

    for i in range(n_features):
        ni = feature_names[i]
        for j in range(n_features):
            nj = feature_names[j]
            ax = axes[j, i]

            if i == j:
                ax.axis("off")

            elif i < j:
                spearman_corr, p_value = spearmanr(X_drifter[nj], X_drifter[ni])

                # if high correlation, use thermal colormap, else use dense colormap
                if spearman_corr > 0.8:
                    hist = ax.hist2d(
                        X_drifter[ni], X_drifter[nj],
                        bins=20, cmap=cmocean.cm.thermal, cmin=1, cmax=800
                    )
                    if hist_thermal is None:
                        hist_thermal = hist
                    ax.text(
                        0.46, 0.95, np.round(spearman_corr, 2), fontsize=14,
                        ha='right', va='top', transform=ax.transAxes, color="red")
                else:
                    hist = ax.hist2d(
                        X_drifter[ni], X_drifter[nj],
                        bins=20, cmap=cmocean.cm.dense, cmin=1, cmax=800
                    )
                    if hist_dense is None:
                        hist_dense = hist

                min_val_x = X_drifter[ni].min()
                max_val_x = X_drifter[ni].max()
                min_val_y = X_drifter[nj].min()
                max_val_y = X_drifter[nj].max()
                ax.plot([min_val_x, max_val_x], [min_val_y, max_val_y],
                        color='red', linewidth=1, linestyle='--')

            else:
                ax.axis("off")

            if i < n_features - 1:
                ax.set_xticks([])
            if j > 0:
                ax.set_yticks([])

    # add feature labels
    for i, name in enumerate(feature_names):
        label = labels[name]['label']
        axes[-1, i].set_xlabel(label, fontsize=16, rotation=45, ha='center')
        axes[i, 0].set_ylabel(label, fontsize=16, rotation=45, ha='right', va='center')

    fig.tight_layout()

    # add colorbars for both colormaps (high & low correlation)
    if hist_thermal:
        cbar_ax = fig.add_axes([0.91, 0.55, 0.015, 0.3])  # [left, bottom, width, height]
        cbar=fig.colorbar(hist_thermal[3], cax=cbar_ax, )
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label(label="Number of data points (high correlation)", size=14)

    if hist_dense:
        cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.3])
        cbar2=fig.colorbar(hist_dense[3], cax=cbar_ax, label="Number of data points (low correlation)",)
        cbar2.ax.tick_params(labelsize=14)
        cbar2.set_label(label="Number of data points (low correlation)", size=14)

    plt.savefig(f'{PROJ_ROOT}/reports/figures/feature_matrix.svg', dpi=300, bbox_inches='tight')
    plt.show()

#%%
if __name__ == "__main__":
    dt_features=xr.open_datatree(interpolated_data_file)
    labels=pd.read_csv(labels_file, delimiter=';', index_col=0)

    #specify variables to include in the feature matrix
    variables=['U', 'V', 'U10', 'V10', 'Ustokes', 'Vstokes', 'Hs_wind', 'Hs_swell', 'Hs_swell_secondary', 'Wave_dir',
           'Wave_dir_wind', 'Wave_dir_swell', 'Wave_dir_swell_secondary', 'Tp', ]
    
    #create feature matrix
    feature_matrix_total=create_feature_matrix(dt_features, variables, fc=True)
#%%
if __name__ == "__main__":
    #plot correlation matrix of all combination of features
   plot_matrix(feature_matrix_total.drop(columns='drifter_id'), labels)
