
#%%
import numpy as np
import matplotlib.pyplot as plt
from PyALE import ale
import joblib
from features import create_feature_matrix
import tqdm
from scipy.interpolate import interp1d

#%
def plot_pfi(importances_u, importances_v, stats_u, stats_v, labels, output_path, residual=True, output_format='svg', ):

    plt.figure(figsize=(10, 5))
    # Prepare data for bar plot
    desired_order = labels.iloc[2].values  # Assuming the first row holds the desired order

    # Prepare data for bar plot
    data_U = []  # Data for U
    data_V = []  # Data for V
    x_coords_U = []  # X positions for U bars
    x_coords_V = []  # X positions for V bars
    errors_U = []  # Errors for U
    errors_V = []  # Errors for V

    bar_width = 0.4  # Adjust the bar width
    hydro_vars=labels.columns
    x_offsets = np.arange(len(hydro_vars)) * 1.5  # X positions for the features
    # Reorder the hydro_vars according to the desired order
    hydro_vars_ordered = []
    for i in range(len(hydro_vars)):
        index=np.where(desired_order == str(i))[0]
        hydro_vars_ordered.append(hydro_vars[index].to_list())
    hydro_vars_ordered=np.ravel(hydro_vars_ordered)

    for i, vars in enumerate(hydro_vars_ordered):
        # Data for U
        values_U = importances_u[vars]  # All values for the selected experiment
        data_U.append(np.mean(values_U))
        errors_U.append(np.std(values_U))
        x_coords_U.append(x_offsets[i] - bar_width / 2)  # Shift left by half-bar width

        # Data for V
        values_V = importances_v[vars]  # All values for the selected experiment  
        data_V.append(np.mean(values_V))
        errors_V.append(np.std(values_V))
        x_coords_V.append(x_offsets[i] + bar_width / 2)  # Shift right by half-bar width

    # Create the bar plot
    if residual==True:
        labelu=rf'$\tilde{{U}}_{{d}}$: $R^2={np.mean(stats_u['R2']):.2f}, RMSE={np.mean(stats_u['RMSE']):.2f}'
        labelv=rf'$\tilde{{V}}_{{d}}$: $R^2={np.mean(stats_v['R2']):.2f}, RMSE={np.mean(stats_v['RMSE']):.2f}'
    else:
        labelu=f'$U_{{d}}$: $R^2={np.mean(stats_u['R2']):.2f}, RMSE={np.mean(stats_u['RMSE']):.2f}'
        labelv=rf'$V_{{d}}$: $R^2={np.mean(stats_v['R2']):.2f}, RMSE={np.mean(stats_v['RMSE']):.2f}'

    plt.bar(
        x_coords_U,
        abs(np.array(data_U)),
        width=bar_width,
        color='#1D4E89',
        capsize=5,
        alpha=0.8,
        label=labelu
    
       )
    plt.bar(
        x_coords_V,
        abs(np.array(data_V)),
       
        width=bar_width,
        color='#E94F37',
        capsize=5,
        alpha=0.8,
        label=labelv
    )
    # Labels and legend
    plt.ylabel(fr'RMSE increase [$m s^{{-1}}$]', fontsize=14)
    plt.xlabel('Features', fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(x_offsets, [labels[i][0] for i in hydro_vars_ordered ], rotation=45, ha='center', fontsize=15)
    plt.legend(loc='upper right', fontsize=14 )
    plt.tight_layout()
    plt.savefig(f'{output_path}.{output_format}', dpi=300)


    plt.show()
#%%

def plot_pfi_shadow_vertical(importances_u, importances_v, stats_u, stats_v, labels, output_path, 
             residual=True, output_format='svg', bar_width=0.5, title=' ', ylabel=True):
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
    if ylabel==True:
        plt.figure(figsize=(7, 10))
    else:
        plt.figure(figsize=(6, 10))

    # Extract ordered variable names and categories
    desired_order = np.array(list(labels.loc['order'].astype(int).values) ) # Assuming the 3rd row defines the correct order
    correct_order = np.argsort(desired_order)
    hydro_vars = labels.columns
    # Reorder hydro_vars correctly
    ordered_vars = np.array(hydro_vars)[correct_order]
   
    categories = labels.loc['category', ordered_vars]
    
    category_colors = {"wind": "#87c147", "ocean": '#62c1d8', "waves": "#a098e6", 'fi': '#eb988c'}

    ## Bar plot data
    x_offsets = np.arange(len(ordered_vars)) * 1.5
    data_U, errors_U, data_V, errors_V = [], [], [], []
    
    for var in ordered_vars:
        data_U.append(np.mean(importances_u[var]))
        errors_U.append(np.std(importances_u[var]))
        data_V.append(np.mean(importances_v[var]))
        errors_V.append(np.std(importances_v[var]))
    
    # Background shading by category
    for i, var in enumerate(ordered_vars[::-1]):
        plt.axhspan( i * 1.5 + bar_width, i * 1.5 - bar_width, color=category_colors[categories[var]], alpha=0.35)
    
    if residual==True:
        labelu=rf'$\tilde{{U}}_{{d}}$: $R^2={np.mean(stats_u['R2']):.2f}$, RMSE={np.mean(stats_u['RMSE']):.2f}'
        labelv=rf'$\tilde{{V}}_{{d}}$: $R^2={np.mean(stats_v['R2']):.2f}$, RMSE={np.mean(stats_v['RMSE']):.2f}'
    else:
        labelu=f'$U_{{d}}$: $R^2={np.mean(stats_u['R2']):.2f}$, RMSE={np.mean(stats_u['RMSE']):.2f}'
        labelv=rf'$V_{{d}}$: $R^2={np.mean(stats_v['R2']):.2f}$, RMSE={np.mean(stats_v['RMSE']):.2f}'
    
    # Plot bars
    plt.barh(  x_offsets - bar_width / 2,np.abs(data_U)[::-1], height=bar_width, color='#1F77B4', alpha=1, label=labelu, )
    plt.barh( x_offsets + bar_width / 2, np.abs(data_V)[::-1], height=bar_width, color='#E41A1C', alpha=1, label=labelv,)
    
    # Labels and formatting
    plt.xlabel(r'RMSE increase [$m s^{-1}$]', fontsize=14)

    plt.xticks(fontsize=14)

    if ylabel==True:
        plt.ylabel('Features', fontsize=14)
        plt.yticks(x_offsets, labels.loc['label', ordered_vars[::-1]],  fontsize=14)
    else:
        plt.ylabel(' ', fontsize=14)
        plt.yticks(x_offsets, np.repeat(' ', len(x_offsets)),  fontsize=14)

    plt.legend(loc='lower right', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{output_path}.{output_format}', dpi=300)
    plt.show()

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
def ale_plots_bootstrap(ale_resultsu, y_name, output_path=None, labels=None):
    ale_plots={}
    # Plot bars
    if 'U' in y_name:
        c='#1D4E89'
    else:
        c='#E94F37'

    for i, feature in enumerate(ale_resultsu.keys()):
        ale_plots[feature]={}
        folds_vals_arr=[]
        x_vals_arr=[]
        for folds in range(len(ale_resultsu[feature]['x'])):
            x_vals_arr.append(ale_resultsu[feature]['x'][folds])
            folds_vals_arr.append(ale_resultsu[feature]['eff'][folds])
        
        
        # Create a union of all x-values
        all_x = np.sort(np.unique(np.concatenate(x_vals_arr)))
        # Interpolate all ALE curves onto the shared x-grid
        folds_vals = []
        for x_vals, y_vals in zip(x_vals_arr, folds_vals_arr):
            f = interp1d(x_vals, y_vals, kind='linear', bounds_error=False, fill_value='extrapolate')
            folds_vals.append(f(all_x))


        # Compute statistics
        mean_vals = np.mean(folds_vals, axis=0)
        std_vals = np.std(folds_vals, axis=0, ddof=1)  # ddof=1 for sample std
        ci_lower = np.percentile(folds_vals, 2.5, axis=0)
        ci_upper = np.percentile(folds_vals, 97.5, axis=0)

        # Store results
        ale_plots[feature]['x'] = all_x
        ale_plots[feature]['mean_y'] = mean_vals
        ale_plots[feature]['std_y'] = std_vals
        ale_plots[feature]['ci_lower'] = ci_lower
        ale_plots[feature]['ci_upper'] = ci_upper

        fig, ax = plt.subplots(figsize=(6, 4))
        feature_display_name = labels[feature].loc['label']
        feature_dims = labels[feature].loc['units']
        
        plt.plot(all_x, mean_vals, color=c, linewidth=3)
        plt.fill_between(all_x, ci_lower, ci_upper, color=c, alpha=0.2)
        ax.set_ylabel(fr'Effect on {y_name} (centered) $[m s^{{-1}}$]', fontsize=14)
        ax.set_xlabel(f'{feature_display_name} [{feature_dims}]', fontsize=14)
        #axs_ale[i].legend(fontsize=8)
        ax.tick_params(axis='both', labelsize=14)  # Increases font size of axis scalars
        plt.savefig(f'{output_path}/{feature}_bootstrap.svg', bbox_inches='tight') 
        plt.show()
#%%

def single_ALE_plots(model, X, y_name, feature,  scale=None, name=None, resolution=20, output_format='svg', output_path=None):

    if 'U' in y_name:
        c='#1D4E89'
    else:
        c='#E94F37'   

    fig, ax = plt.subplots(figsize=(6, 4))
    
    # ALE plot (dummy data, replace with your ALE calculation)
    x=X[model.feature_names_in_]
    ale_eff = ale(
        X=x, model=model, feature=[feature], grid_size=resolution,
        include_CI=False, C=0.95, feature_type="continuous", plot=False
    )
    
    x_values = np.array(ale_eff.index.values)
    y_values = ale_eff['eff'].values

    if scale!=None:
        ax.plot(x_values, scale.inverse_transform(y_values.reshape(-1, 1)), label=f'ALE', color=c)
 
    else:    
        ax.plot(x_values, y_values, label=f'ALE', color=c)

    ax.set_ylabel(f'Effect on {y_name} (centered) [m/s]', fontsize=14)
    ax.set_xlabel(f'{name}', fontsize=14)
    #axs_ale[i].legend(fontsize=8)
    ax.tick_params(axis='both', labelsize=14)  # Increases font size of axis scalars

    plt.savefig(f'{output_path}_ALE.{output_format}', dpi=300)
    plt.show()



    fig, ax2 = plt.subplots(figsize=(6, 1.5))

    # Histogram
    bin_edges = np.histogram_bin_edges(x[feature], bins='scott')
    ax2.hist(x[feature], bins=bin_edges, color=c, alpha=0.6)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_ylabel('# obs', fontsize=14)
    ax2.set_xticklabels([]) 
    ax2.tick_params(axis='y', labelsize=14)  # Increases font size of axis scalars

    # Ensure layout works without overlap
    plt.tight_layout()
    plt.savefig(f'{output_path}_histogram.{output_format}', dpi=300)
    
#%%

def leave_one_out_ALE_plots(model_dir, model_name, y_name, feature, Xfeatures, dt, scale=None, name=None, resolution=20, output_format='svg', output_path=None):

    if 'U' in y_name:
        c='#1F77B4'
    else:
        c='#E41A1C'    

    fig, ax = plt.subplots(figsize=(6, 4))

    x_values={}
    y_values={}

    for node in tqdm.tqdm(dt.leaves):
        drifter=node.name
        drifter_dt= dt.copy()
        del drifter_dt[drifter]

        model=joblib.load(f'{model_dir}/{drifter}/{model_name}.pkl')

        X=create_feature_matrix(drifter_dt, Xfeatures, waves=True, fc=True)
    
        # ALE plot (dummy data, replace with your ALE calculation)
        x=X[model.feature_names_in_]
        ale_eff = ale(
            X=x, model=model, feature=[feature], grid_size=resolution,
            include_CI=False, C=0.95, feature_type="continuous", plot=False
        )
        
        x_values[drifter] = np.array(ale_eff.index.values)
        y_values[drifter] = ale_eff['eff'].values

        pass
    
    # Create a union of all x-values

    print(len(x_values['6480']), len(y_values['6480']))
    
    all_x = np.sort(np.unique(np.concatenate(list(x_values.values()))))
    # Interpolate all ALE curves onto the shared x-grid
    folds_vals = []
    for i, drifter in enumerate(x_values):
        x_vals=x_values[drifter]
        y_vals=y_values[drifter]

        f = interp1d(x_vals, y_vals, kind='linear', bounds_error=False, fill_value='extrapolate')
        folds_vals.append(f(all_x))


    # Compute statistics
    mean_vals = np.mean(folds_vals, axis=0)
    ci_lower = np.percentile(folds_vals, 2.5, axis=0)
    ci_upper = np.percentile(folds_vals, 97.5, axis=0)
    
    plt.plot(all_x, mean_vals, color=c)
    plt.fill_between(all_x, ci_lower, ci_upper, color=c, alpha=0.2)
    

    ax.set_ylabel(f'Effect on {y_name} (centered) [m/s]', fontsize=14)
    ax.set_xlabel(f'{name}', fontsize=14)
    #axs_ale[i].legend(fontsize=8)
    ax.tick_params(axis='both', labelsize=14)  # Increases font size of axis scalars

   # plt.savefig(f'{output_path}_ALE.{output_format}', dpi=300)
    plt.show()


    fig, ax2 = plt.subplots(figsize=(6, 1.5))

    # Histogram
    bin_edges = np.histogram_bin_edges(x[feature], bins='scott')
    ax2.hist(x[feature], bins=bin_edges, color=c, alpha=0.6)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_ylabel('# obs', fontsize=14)
    ax2.set_xticklabels([]) 
    ax2.tick_params(axis='y', labelsize=14)  # Increases font size of axis scalars

    # Ensure layout works without overlap
    plt.tight_layout()
   # plt.savefig(f'{output_path}_histogram.{output_format}', dpi=300)
