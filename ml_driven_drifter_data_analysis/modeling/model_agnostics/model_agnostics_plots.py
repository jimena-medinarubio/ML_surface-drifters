#%%
import numpy as np
import matplotlib.pyplot as plt
from PyALE import ale
from matplotlib.patches import Patch
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor
import tqdm
from scipy.interpolate import interp1d
#%%
def plot_single_pfi(importances_u, stats_u, labels, name, output_path, output_format='svg', bar_width=0.5, color='#8E5572'):

    fig, ax=plt.subplots(figsize=(10, 4))

    # Extract ordered variable names and categories
    
    correct_order = np.argsort(np.abs(importances_u.mean(axis=0)).sort_values(ascending=False))
    # Reorder hydro_vars correctly
    ordered_vars = correct_order.keys()
    print(ordered_vars)
    categories = labels.loc['category', ordered_vars]
    
    category_colors = {"wind": "#FBD1A2", "ocean": '#00B2CA', "waves": "#7DCFB6", 'fi': '#8E5572'}

    ## Bar plot data
    x_offsets = np.arange(len(ordered_vars)) * 1.5
    data_U, errors_U,  = [], [],
    
    for var in ordered_vars:
        data_U.append(np.mean(importances_u[var]))
        errors_U.append(np.std(importances_u[var]))
    
    # Background shading by category
    for i, var in enumerate(ordered_vars):
        plt.axvspan(i * 1.5 - bar_width, i * 1.5 + bar_width, color=category_colors[categories[var]], alpha=0.35)
    

    labelu=rf'{name}: $R^2={np.round(np.mean(stats_u['R2']), 2)}$, RMSE={np.round(np.mean(stats_u['RMSE']), 2)}'

    # Plot bars
    plt.bar(x_offsets , np.abs(data_U), width=bar_width, color=color, alpha=1, label=labelu)

    # Labels and formatting
    plt.ylabel(r'RMSE increase [$m s^{-1}$]', fontsize=14)
    plt.xlabel('Features', fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(x_offsets, labels.loc['label', ordered_vars], rotation=45, ha='center', fontsize=16)
    
    plt.legend(loc='upper right', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_path}.{output_format}', dpi=300)


# %%

def plot_pfi_shadow(importances_u, importances_v, stats_u, stats_v, labels, output_path, 
             residual=True, output_format='svg', bar_width=0.5, model_name='RF'):
    """
    Plots the Permutation Feature Importance (PFI) as a bar chart for two velocity components (U and V),
    with optional residual-based labeling and background shading for different feature groups.
    
    Parameters:
    - importances_u, importances_v: Dicts or DataFrames with feature importances for U and V.
    - stats_u, stats_v: Dicts or DataFrames with statistical metrics (R², RMSE).
    - labels: DataFrame containing feature names and categories (e.g., wind, ocean, waves).
    - output_path: Path to save the figure.
    - residual: Whether the plotted values correspond to residual components.
    - output_format: Output file format (e.g., 'svg', 'png').
    """

    fig, ax=plt.subplots(figsize=(10, 4))

    # Extract ordered variable names and categories
    desired_order = np.array(list(labels.loc['order'].astype(int).values) ) # Assuming the 3rd row defines the correct order
    correct_order = np.argsort(desired_order)
    hydro_vars = labels.columns
    # Reorder hydro_vars correctly
    ordered_vars = np.array(hydro_vars)[correct_order]
   
    categories = labels.loc['category', ordered_vars]
    
    category_colors = {"wind": "#FBD1A2", "ocean": '#00B2CA', "waves": "#7DCFB6", 'fi': '#8E5572'}
    category_names = {"wind": "Wind", "ocean": 'Ocean currents', "waves": "Waves", 'fi': 'Flipping Index'}

    ## Bar plot data
    x_offsets = np.arange(len(ordered_vars)) * 1.5
    data_U, errors_U, data_V, errors_V = [], [], [], []
    
    for var in ordered_vars:
        data_U.append(np.mean(importances_u[var]))
        errors_U.append(np.std(importances_u[var]))
        data_V.append(np.mean(importances_v[var]))
        errors_V.append(np.std(importances_v[var]))
    
    # Background shading by category
    cats_used=np.unique([categories[var] for var in ordered_vars])
    print(cats_used)
    for i, var in enumerate(ordered_vars):
        plt.axvspan(i * 1.5 - bar_width, i * 1.5 + bar_width, color=category_colors[categories[var]], alpha=0.35)
    
    if residual==True:
        labelu=rf'$\tilde{{U}}_{{d}}$: $R^2={np.mean(stats_u['R2']):.2f}$, RMSE={np.mean(stats_u['RMSE']):.2f}, MAE={np.mean(stats_u['ME']):.2f}'
        labelv=rf'$\tilde{{V}}_{{d}}$: $R^2={np.mean(stats_v['R2']):.2f}$, RMSE={np.mean(stats_v['RMSE']):.2f}, MAE={np.mean(stats_v['ME']):.2f}'
    else:
        labelu=f'$U_{{d}}$: $R^2={np.mean(stats_u['R2']):.2f}$, RMSE={np.mean(stats_u['RMSE']):.2f}, MAE={np.mean(stats_u['ME']):.2f}'
        labelv=rf'$V_{{d}}$: $R^2={np.mean(stats_v['R2']):.2f}$, RMSE={np.mean(stats_v['RMSE']):.2f}, MAE={np.mean(stats_v['ME']):.2f}'
    
    # Plot bars
    plt.bar(x_offsets - bar_width / 2, np.abs(data_U), width=bar_width, color='#1D4E89', alpha=1, label=labelu)
    plt.bar(x_offsets + bar_width / 2, np.abs(data_V), width=bar_width, color='#E94F37', alpha=1, label=labelv)
    
    # Labels and formatting
    plt.ylabel(r'RMSE increase [$m s^{-1}$]', fontsize=14)
    plt.xlabel('Features', fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(x_offsets, labels.loc['label', ordered_vars], rotation=45, ha='center', fontsize=16)
    

    plt.title(model_name, fontsize=16)
    plt.xlim(x_offsets[0] - bar_width-0.3, x_offsets[-1] + bar_width+0.3)
    plt.tight_layout()


    category_patches = [Patch(facecolor=category_colors[color], edgecolor='none', alpha=0.35, label=category_names[color]) 
                    for color in category_colors.keys() if color in cats_used]
    
    # First legend for the bars
    bar_legend = plt.legend(loc='upper right', fontsize=14)

    # Add background shading legend outside the plot
    if model_name=='Random forest model':
        category_legend = plt.legend(handles=category_patches, 
                                    loc='center left', 
                                    bbox_to_anchor=(1.02, 0.8),
                                    fontsize=13, title_fontsize=14)

    # Add both legends
    plt.gca().add_artist(bar_legend)
    plt.savefig(f'{output_path}.{output_format}', dpi=300)
    plt.show()

# %%
def bootstrap_ale(X_features, yu, yv, ale_featuresu, ale_featuresv, n_bootstraps=100, ntrees=100):
    # Initialize ALE results storage as lists to accumulate bootstrap samples
    ale_resultsu = {feature: {'eff': [], 'x': []} for feature in ale_featuresu}
    ale_resultsv = {feature: {'eff': [], 'x': []} for feature in ale_featuresv}

    for _ in tqdm.tqdm(range(n_bootstraps)):
        # Bootstrap resampling
        X_resampled, y_resampled = resample(X_features, yu)
        X_resampledv, y_resampledv = resample(X_features, yv)

        # Train model on bootstrap sample
        model_ale = RandomForestRegressor(
            random_state=42, n_estimators=ntrees, oob_score=True, bootstrap=True) 
        model_alev = RandomForestRegressor(
            random_state=42, n_estimators=ntrees, oob_score=True, bootstrap=True)            
        model_ale.fit(X_resampled, y_resampled)
        model_alev.fit(X_resampledv, y_resampledv)

        # Calculate ALE for each feature
        for feature in ale_featuresu:
            ale_exp = ale(X=X_resampled, model=model_ale, feature=[feature], grid_size=20, include_CI=False, plot=False, feature_type="continuous")
            
            # Append ALE results for each bootstrap
            ale_resultsu[feature]['eff'].append(ale_exp['eff'].values)
            # Save x-values (should be the same for all bootstraps if grid_size is fixed)
            ale_resultsu[feature]['x'].append(np.array(ale_exp.index.values))
        
        for feature in ale_featuresv:
            ale_exp = ale(X=X_resampledv, model=model_alev, feature=[feature], grid_size=20, include_CI=False, plot=False, feature_type="continuous")
            
            # Append ALE results for each bootstrap
            ale_resultsv[feature]['eff'].append(ale_exp['eff'].values)
            # Save x-values (should be the same for all bootstraps if grid_size is fixed)
            ale_resultsv[feature]['x'].append(np.array(ale_exp.index.values))

    return ale_resultsu, ale_resultsv

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

        if output_name ==None:
            output_name = f'{u_feat}_twin_ale'
            
        plt.savefig(f'{output_path}/{output_name}', bbox_inches='tight')
        plt.show()
#%%