#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#%%

def plot_pfi(stats_u, stats_v, labels, output_path, residual=True, output_format='svg', ):

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

    importances_u = stats_u['permutation_importance']
    importances_v = stats_v['permutation_importance']

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
        labelu=rf'$\tilde{{U}}_{{d}}$: $R^2={np.round(np.mean(stats_u['R2']), 2)}$, RMSE={np.round(np.mean(stats_u['RMSE']), 2)}'
        labelv=rf'$\tilde{{V}}_{{d}}$: $R^2={np.round(np.mean(stats_v['R2']), 2)}$, RMSE={np.round(np.mean(stats_v['RMSE']), 2)}'
    else:
        labelu=f'$U_{{d}}$: $R^2={np.round(np.mean(stats_u['R2']), 2)}$, RMSE={np.round(np.mean(stats_u['RMSE']), 2)}'
        labelv=rf'$V_{{d}}$: $R^2={np.round(np.mean(stats_v['R2']), 2)}$, RMSE={np.round(np.mean(stats_v['RMSE']), 2)}'

    plt.bar(
        x_coords_U,
        abs(np.array(data_U)),
        width=bar_width,
        color='#1F77B4',
        capsize=5,
        alpha=0.8,
        label=labelu
    
       )
    plt.bar(
        x_coords_V,
        abs(np.array(data_V)),
       
        width=bar_width,
        color='#E41A1C',
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


# %%
