#%%
import sys
sys.path.append("..")
from config import DATA_DIR,  PROJ_ROOT ,MODELS_DIR, FIGURES_DIR
from features import create_feature_matrix
import pandas as pd
import pickle
import xarray as xr

import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1
    #%%

def ale_plots_bootstrap_poster(feature_pairs, ale_resultsu, ale_resultsv, y_name, x_names, output_path=None, labels=None, colors={'u': '#1D4E89', 'v': '#E94F37'}):

        for i, [u_feat, v_feat] in enumerate(feature_pairs.items()):
            print(u_feat)
            def prepare_vals(ale_results, feat):
                x_vals_arr = []
                folds_vals_arr = []
                for fold in range(len(ale_results[feat]['x'])):
                    x_vals_arr.append(ale_results[feat]['x'][fold])
                    folds_vals_arr.append(ale_results[feat]['eff'][fold])
                all_x = np.sort(np.unique(np.concatenate(x_vals_arr)))
                folds_vals = [
                    interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')(all_x)
                    for x, y in zip(x_vals_arr, folds_vals_arr)
                ]
                mean_vals = np.mean(folds_vals, axis=0)
                ci_lower = np.percentile(folds_vals, 2.5, axis=0)
                ci_upper = np.percentile(folds_vals, 97.5, axis=0)
                return all_x, mean_vals, ci_lower, ci_upper

            x_u, mean_u, ci_u_l, ci_u_u = prepare_vals(ale_resultsu, u_feat)
            x_v, mean_v, ci_v_l, ci_v_u = prepare_vals(ale_resultsv, v_feat)

            fig, ax = plt.subplots(figsize=(6, 4))
            feature_display_name = labels[u_feat].loc['label']
            feature_dims = labels[u_feat].loc['units']

            feature_display_name_v= labels[v_feat].loc['label']

            ax.plot(x_u, mean_u, color=colors['u'], label=feature_display_name , linewidth=2.5)
            ax.fill_between(x_u, ci_u_l, ci_u_u, color=colors['u'], alpha=0.2)

            ax.plot(x_v, mean_v, color=colors['v'], label=feature_display_name_v, linewidth=2.5)
            ax.fill_between(x_v, ci_v_l, ci_v_u, color=colors['v'], alpha=0.2)

            ax.set_ylabel(fr'Effect on {y_name} (centered) $[m\,s^{{-1}}]$', fontsize=14)
            ax.set_xlabel(f'{x_names[i]} [{feature_dims}]', fontsize=14)
            ax.tick_params(axis='both', labelsize=14)
            ax.legend(fontsize=12)

            plt.savefig(f'{output_path}/{u_feat}_uv_ale.svg', bbox_inches='tight')
            plt.show()

        #%%
feature_pairs = { 'U_hp': 'V_hp',
    'U10': 'V10',
   
}
ML_model_settings={'RF_total': {'files':[f'{MODELS_DIR}/RandomForest/RF_Ud_models.pkl', f'{MODELS_DIR}/RandomForest/RF_Vd_models.pkl'], 
                                'stats':[PROJ_ROOT/ 'data'/ 'processed'/ 'Statistics models'/'RandomForest'/'RF_Ud_stats', PROJ_ROOT/ 'data'/ 'processed'/ 'Statistics models'/'RandomForest'/'RF_Vd_stats'], 
                                'u_variables': ['U10', 'U_hp'], 'v_variables': ['V10', 'V_hp'], 'path':['Total'],
                                'labels': pd.read_csv(f'{PROJ_ROOT}/references/variables_labels.csv', delimiter=';', index_col=0)},
                      
                'RF_residual': {'files':[f"{MODELS_DIR}/RandomForest/RF_rUd_models.pkl", f"{MODELS_DIR}/RandomForest/RF_rVd_models.pkl"], 
                                'stats':[PROJ_ROOT/ 'data'/ 'processed'/ 'Statistics models'/'RandomForest'/'RF_rUd_FI_stats', PROJ_ROOT/ 'data'/ 'processed'/ 'Statistics models'/'RandomForest'/'RF_rVd_FI_stats'], 
                                'u_variables': ['U10', 'U_lp', 'Ustokes'], 'v_variables': ['V10', 'V_lp', 'Vstokes'], 'path':['residual'],
                                'labels': pd.read_csv(f'{PROJ_ROOT}/references/variables_labels_residual.csv', delimiter=';', index_col=0)},
             }

# %%

def ale_plots_bootstrap_twin(X, feature_pairs, ale_resultsu, ale_resultsv, y_name, x_names, output_path=None, labels=None, colors={'u': '#1D4E89', 'v': '#E94F37'}):

    for i, (u_feat, v_feat) in enumerate(feature_pairs.items()):
        print(u_feat)

        def prepare_vals(ale_results, feat):
            x_vals_arr = []
            folds_vals_arr = []
            for fold in range(len(ale_results[feat]['x'])):
                x_vals_arr.append(ale_results[feat]['x'][fold])
                folds_vals_arr.append(ale_results[feat]['eff'][fold])
            all_x = np.sort(np.unique(np.concatenate(x_vals_arr)))
            folds_vals = [
                interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')(all_x)
                for x, y in zip(x_vals_arr, folds_vals_arr)
            ]
            mean_vals = np.mean(folds_vals, axis=0)
            ci_lower = np.percentile(folds_vals, 2.5, axis=0)
            ci_upper = np.percentile(folds_vals, 97.5, axis=0)
            return all_x, mean_vals, ci_lower, ci_upper

        x_u, mean_u, ci_u_l, ci_u_u = prepare_vals(ale_resultsu, u_feat)
        x_v, mean_v, ci_v_l, ci_v_u = prepare_vals(ale_resultsv, v_feat)

        fig, (ax_top, ax_u) = plt.subplots(
            nrows=2,
            figsize=(7, 6),  # taller figure for better layout
            sharex=True,
            gridspec_kw={'height_ratios': [1, 2.5]}  # Top is 1x, bottom is 2x taller
        )
        ax_v = ax_u.twinx()

       
        yminu, ymaxu = np.min(mean_u), np.max(mean_u)
        yminv, ymaxv = np.min(mean_v), np.max(mean_v)
        # Make sure zero is included and both axes are symmetric around it if desired
        yabsu = max(abs(yminu), abs(ymaxu))
        yabsv = max(abs(yminv), abs(ymaxv))

        # Set same y-limits on both axes
        ax_u.set_ylim(-yabsu, yabsu)
        ax_v.set_ylim(-yabsv, yabsv)

        # Labels
        feature_display_name_u = labels[u_feat].loc['label']
        feature_display_name_v = labels[v_feat].loc['label']
        feature_units = labels[u_feat].loc['units']  # assumes same x for both

        # Plot U (zonal) on left y-axis
        ax_u.plot(x_u, mean_u, color=colors['u'], label=feature_display_name_u, linewidth=2.5)
        ax_u.fill_between(x_u, ci_u_l, ci_u_u, color=colors['u'], alpha=0.2)
        ax_u.set_ylabel(fr'Eeffect on {y_name[0]} (centred) $[m\,s^{{-1}}]$', fontsize=14, color=colors['u'])
        ax_u.tick_params(axis='y', labelcolor=colors['u'])
        
        yminu, ymaxu = np.min(x_u), np.max(x_u)
        yminv, ymaxv = np.min(x_v), np.max(x_v)
        xmin=min(yminu, yminv)
        xmax=max(ymaxu, ymaxv)
        x_axis=np.linspace(xmin, xmax, 100 )
        ax_u.plot(x_axis, np.zeros_like(x_axis), color='black', linestyle='--', linewidth=1)  # Zero line

        # Plot V (meridional) on right y-axis
        ax_v.plot(x_v, mean_v, color=colors['v'], label=feature_display_name_v, linewidth=2.5)
        ax_v.fill_between(x_v, ci_v_l, ci_v_u, color=colors['v'], alpha=0.2)
        ax_v.set_ylabel(fr'Effect on {y_name[1]} (centred) $[m\,s^{{-1}}]$', fontsize=14, color=colors['v'])
        ax_v.tick_params(axis='y', labelcolor=colors['v'])

        # X-axis
        ax_u.set_xlabel(f'{x_names[i]} [{feature_units}]', fontsize=14)
        ax_u.tick_params(axis='x', labelsize=14)
        ax_u.tick_params(axis='y', labelsize=14)
        ax_v.tick_params(axis='y', labelsize=14)
        # Combine legends from both axes
        lines_u, labels_u = ax_u.get_legend_handles_labels()
        lines_v, labels_v = ax_v.get_legend_handles_labels()
        ax_u.legend(lines_u + lines_v, labels_u + labels_v, fontsize=14, loc='best')

        # Add your top subplot (e.g., an auxiliary plot)
       
        bin_edges_u = np.histogram_bin_edges(X[u_feat], bins='scott')
        bin_edges_v = np.histogram_bin_edges(X[v_feat], bins='scott')
        ax_top.hist(X[u_feat], bins=bin_edges_u, color=colors['u'], alpha=0.4)
        ax_top.hist(X[v_feat], bins=bin_edges_v, color=colors['v'], alpha=0.4)
        
        ax_top.set_ylabel('# obs', fontsize=14)
        
        ax_top.tick_params(axis='y', labelsize=12)  # Increases font size of axis scalars

        fig.tight_layout()
        plt.savefig(f'{output_path}/{u_feat}_twin_ale.svg', bbox_inches='tight')
        plt.show()
#%%
with open(f'{DATA_DIR}/processed/ALE/RF_Ud_ALE', "rb") as f:
    ale_u=pickle.load( f)

with open(f'{DATA_DIR}/processed/ALE/RF_Vd_ALE', "rb") as f:
    ale_v=pickle.load( f)

feature_pairs = { 'U_hp': 'V_hp',
    'U10': 'V10',
   
}
variables=['U', 'V', 'U10', 'V10', 'Ustokes', 'Vstokes', 'Hs_wind', 'Hs_swell', 'Hs_swell_secondary', 'Wave_dir',
           'Wave_dir_wind', 'Wave_dir_swell', 'Wave_dir_swell_secondary', 'Tp', ]
y_variables=['vx', 'vy', 'vx_residual', 'vy_residual']

dt_features=xr.open_datatree(f'{DATA_DIR}/interim/interpolated_atm_ocean_datasets_depth.nc')

X=create_feature_matrix(dt_features, np.concatenate([variables, y_variables]) , waves=True, fc=True)

ale_plots_bootstrap_twin(X, feature_pairs, ale_u, ale_v, 
                            output_path=FIGURES_DIR/ 'ALE'/'Total', labels=ML_model_settings['RF_total']['labels'], x_names=['High-pass surface ocean currents', 'Wind'], y_name=['$U_d$', '$V_d$'])

# %%
feature_pairs = {     'U10': 'V10',
   
}
with open(f'{DATA_DIR}/processed/ALE/RF_rUd_ALE', "rb") as f:
    ale_u=pickle.load( f)

with open(f'{DATA_DIR}/processed/ALE/RF_rVd_ALE', "rb") as f:
    ale_v=pickle.load( f)

ale_plots_bootstrap_twin(X, feature_pairs, ale_u, ale_v, 
                            output_path=FIGURES_DIR/ 'ALE'/'Residual', labels=ML_model_settings['RF_residual']['labels'], x_names=['Wind'], y_name=[r'$\tilde{U}_d$', r'$\tilde{V}_d$'])

# %%
feature_pairs = { 'U_lp': 'V_lp',
    'Ustokes': 'Vstokes',
   
}
with open(f'{DATA_DIR}/processed/ALE/RF_rUd_ALE', "rb") as f:
    ale_u=pickle.load( f)

with open(f'{DATA_DIR}/processed/ALE/RF_rVd_ALE', "rb") as f:
    ale_v=pickle.load( f)

ale_plots_bootstrap_twin(X, feature_pairs, ale_u, ale_v, 
                            output_path=FIGURES_DIR/ 'ALE'/'Residual', labels=ML_model_settings['RF_residual']['labels'], x_names=['Low-pass surface ocean currents', 'Stokes drift'], y_name=[r'$\tilde{U}_d$', r'$\tilde{V}_d$'])

# %%
