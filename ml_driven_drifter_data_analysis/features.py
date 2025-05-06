#%%
from pathlib import Path
from tqdm import tqdm
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from scipy.stats import spearmanr
from config import DATA_DIR, PROJ_ROOT
from scipy.special import erfc

#%%
if __name__ == "__main__":
    dt_features=xr.open_datatree(f'{DATA_DIR}/interim/interpolated_atm_ocean_datasets.nc')
    labels=pd.read_csv(f'{PROJ_ROOT}/references/variables_labels_residual.csv', delimiter=';', index_col=0)
# %%

def calculate_stokes_drift_depth(u, Tp, z, beta=1, gamma=5.97):
        def calculate_k(u, tp, beta):
            return abs(u)/(2*abs(tp))*(1-2*beta/3)
        k=calculate_k(u, Tp, gamma)
        expo=np.exp(2*k*z)
        a= beta*np.sqrt(-2*k*np.pi*z )*erfc( np.sqrt(-2*k*z) )
        return u*(expo-a)

def transformations(df, filter_currents=True, waves=True):

    for var in ['Ustokes', 'Vstokes']:
        if var in df.columns:  # Check if the variable exists
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
            df[f'{var}_lp'] = df[var].rolling(window='24.83h', min_periods=1, center=True).mean()
            df[f'{var}_hp'] = df[var]-df[f'{var}_lp']
            df=df.drop(columns=[var])
            df = df.reset_index(drop=False)

    return df

def create_feature_matrix(dt_features, selected_vars, waves=True, fc=True):
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

    # Combine all drifters into one DataFrame
    feature_matrix = pd.concat(dfs, ignore_index=False)

    if 'obs' in feature_matrix.columns:
        feature_matrix = feature_matrix.drop(columns=['obs'])
    if 'trajectory' in feature_matrix.columns:
        feature_matrix = feature_matrix.drop(columns=['trajectory'])
    
    feature_matrix.columns = [str(col) for col in feature_matrix.columns]

    return feature_matrix

def plot_matrix(X_drifter, labels):
    n_features = X_drifter.shape[1]  # Number of features
    feature_names = X_drifter.columns[::-1]  # Example feature names
    from scipy.stats import spearmanr
    import cmocean
    # Set up the figure size based on the number of features
    fig, axes = plt.subplots(n_features, n_features, figsize=(20, 20))

    # Loop over rows and columns to create scatter plots
    for i in range(n_features):
        ni=feature_names[i]
        for j in range(n_features):
            nj=feature_names[j]
            ax = axes[j, i]
        
            if i == j:
                # Eliminate the diagonal
                ax.axis("off")

            elif i<j:
                spearman_corr, p_value = spearmanr(X_drifter[nj], X_drifter[ni])

                if spearman_corr > 0.8:
                    # Density plot for strong correlations
                    hist = ax.hist2d(
                        
                        X_drifter[ni], X_drifter[nj], 
                        bins=20, cmap=cmocean.cm.thermal, cmin=1, )
                    ax.text(
                        0.4, 0.95, np.round(spearman_corr, 2), fontsize=12,
                        ha='right', va='top', transform=ax.transAxes, color="red")
                    print(nj, ni)
                else:
                    # Density plot for weak correlations
                    hist = ax.hist2d(
                    
                        X_drifter[ni],  X_drifter[nj], 
                        bins=20, cmap=cmocean.cm.dense, cmin=1, )
                min_val_x = X_drifter[ni].min()
                max_val_x = X_drifter[ni].max()

                min_val_y=X_drifter[nj].min()
                max_val_y=X_drifter[nj].max()
                ax.plot([min_val_x, max_val_x], [min_val_y, max_val_y], color='red', linewidth=1, linestyle='--')

            else:
                # Hide the upper triangle (and optionally diagonal)
                ax.axis("off")  # Turn off axis completely

            # Remove ticks for clarity
            if i < n_features - 1:
                ax.set_xticks([])
            if j > 0:
                ax.set_yticks([])

    # Add global labels for x and y axes
    for i, name in enumerate(feature_names):
        # Add feature names to the bottom x-axis
        feature=labels[name]['label']
        axes[-1, i].set_xlabel(feature, fontsize=12, rotation=45, ha='right')
        # Add feature names to the left y-axis
        axes[i, 0].set_ylabel(feature, fontsize=12, rotation=0, ha='right', va='center')
    # Add space between plots and adjust layout
    fig.tight_layout()
    plt.savefig(f'{PROJ_ROOT}/reports/figures/feature_matrix.svg', dpi=300, bbox_inches='tight')
    plt.show()

# %%
#example
if __name__ == "__main__":
    variables=['U', 'V', 'U10', 'V10', 'Ustokes', 'Vstokes', 'Hs_wind', 'Hs_swell', 'Hs_swell_secondary', 'Wave_dir',
           'Wave_dir_wind', 'Wave_dir_swell', 'Wave_dir_swell_secondary', 'Tp', ]
#
    feature_matrix_total=create_feature_matrix(dt_features, variables, fc=True)
    #plot_matrix(feature_matrix_total.drop(columns='drifter_id'), labels)

# %%
