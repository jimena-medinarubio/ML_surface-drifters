#%%
import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import pandas as pd
sys.path.append("..")
from config import FIGURES_DIR
from statsmodels.tsa.stattools import acf
from processing_drifter_data import select_frequency_interval
from features import create_feature_matrix
#%%
PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "data"

datatree=xr.open_datatree(f'{DATA_DIR}/interim/processed_drifter_data.nc')
# %%
X=create_feature_matrix(datatree, ['vx', 'vy', 'vx_residual', 'vy_residual', 'v'], waves=False, fc=False)

#%%
def autocorrelation_time(feature, datatree, X, sampling_frequencies, feature_name, max_lag=50, threshold = 1 / np.e, file_format='svg', output_path=None, filename='test'):
    
    autocorr_array=[ [] for elem in feature]
    tau=[ [] for elem in feature]
    
    for ds in datatree.leaves:
        
        start_idx = select_frequency_interval(ds, sampling_frequencies[0]) #sampling freqs in secs
        #end_idx=-1
        end_idx = select_frequency_interval(ds, sampling_frequencies[1])

        time_start, time_end=ds.isel(time=start_idx).time, ds.isel(time=end_idx).time
       

        for i, elem in enumerate(feature):
            
            x=X[X['drifter_id'] == ds.name].loc[time_start.values:time_end.values][elem]
            #x=X[X['drifter_id'] == ds.name][elem]
            
            autocorr=acf(x, nlags=max_lag)
            
            # Find the first lag where ACF is below the threshold
            autocorr_time = np.argmax(autocorr < threshold)/(sampling_frequencies[1]/3600) #hours

            autocorr_array[i].append(autocorr)
            tau[i].append(autocorr_time)
    colors=['blue', '#E41A1C']
#'#1F77B4'
    for i, elem in enumerate(feature):
        print(f"Autocorrelation time (τ): {np.mean(tau[i])}, {feature[i]}")
        mean=np.mean(autocorr_array[i], axis=0)
        std=np.std(autocorr_array[i], axis=0)
        plt.plot(np.arange(0, max_lag+1)/(sampling_frequencies[1]/3600), mean, color=colors[i], label=f'{feature_name[i]}')
        plt.fill_between(np.arange(0, max_lag+1)/(sampling_frequencies[1]/3600), mean-std, mean+std , color=colors[i], alpha=0.2)
        
    
    plt.plot(np.arange(0, max_lag+1)/(sampling_frequencies[1]/3600), np.repeat(threshold,len(mean)), label=rf'$1/e$', linestyle='--', color='forestgreen')    
    plt.legend()
    plt.xlabel('Time [h]')
    plt.ylabel(f'Autocorrelation')

    if output_path is not None:
        plt.savefig(f'{output_path}/Autocorrelation_{filename}.{file_format}')
        df = pd.DataFrame([np.mean(tau, axis=1)], columns=feature)
        file_output_path=PROJ_ROOT/ 'references'
        df.to_csv(f'{file_output_path}/autocorrelation_{filename}.csv', index=False) 
        plt.show

    return tau, autocorr

# %%

output=FIGURES_DIR
tau, autocorr=autocorrelation_time(['vx', 'vy'], datatree, X, [5*60, 30*60], [r'$U_d$', r'$V_d$'], output_path=output, filename='Total')
# %%
tau, autocorr=autocorrelation_time(['vx_residual', 'vy_residual'], datatree, X, [5*60, 30*60],  [r'$\tilde{U}_d$', r'$\tilde{V}_d$'], max_lag=100, output_path=output, filename='Residual')

# %%
