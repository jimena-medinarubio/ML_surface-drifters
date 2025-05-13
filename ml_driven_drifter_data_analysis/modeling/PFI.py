#%%
import numpy as np
import xarray as xr

import joblib
import pandas as pd
from pathlib import Path

import sys
sys.path.append("..")
from config import DATA_DIR,  PROJ_ROOT ,MODELS_DIR, FIGURES_DIR
from features import create_feature_matrix
from modeling.model_agnostics import plot_pfi_shadow, plot_pfi_shadow_vertical, plot_single_pfi
from modeling.train import TimeBlockSplit, RF_regression
from sklearn.model_selection import  KFold
#%%

ML_model_settings={'RF_total': {'files':[f"{MODELS_DIR}/RandomForest/RF_Ud_models.pkl", f"{MODELS_DIR}/RandomForest/RF_Vd_models.pkl"], 
                                'stats':[PROJ_ROOT/ 'data'/ 'processed'/ 'Statistics models'/'RandomForest'/'RF_Ud_stats', PROJ_ROOT/ 'data'/ 'processed'/ 'Statistics models'/'RandomForest'/'RF_Vd_stats'], 
                                'pfi': [PROJ_ROOT/ 'data'/ 'processed'/ 'PFI'/'RandomForest'/'RF_Ud_pfi', PROJ_ROOT/ 'data'/ 'processed'/ 'PFI'/'RandomForest'/'RF_Vd_pfi'], 
                                'labels': pd.read_csv(f'{PROJ_ROOT}/references/variables_labels.csv', delimiter=';', index_col=0),
                                'title': 'Random forest model'},
                
                'RF_FI_total': {'files':[f"{MODELS_DIR}/RandomForest/RF_Ud_FI_models.pkl", f"{MODELS_DIR}/RandomForest/RF_Vd_FI_models.pkl", f"{MODELS_DIR}/RandomForest/RF_FI_fit_models.pkl"], 
                                'stats':[PROJ_ROOT/ 'data'/ 'processed'/ 'Statistics models'/'RandomForest'/'RF_Ud_FI_stats', PROJ_ROOT/ 'data'/ 'processed'/ 'Statistics models'/'RandomForest'/'RF_Vd_FI_stats'], 
                                'pfi': [PROJ_ROOT/ 'data'/ 'processed'/ 'PFI'/'RandomForest'/'RF_Ud_FI_pfi', PROJ_ROOT/ 'data'/ 'processed'/ 'PFI'/'RandomForest'/'RF_Vd_FI_pfi'], 
                                'labels': pd.read_csv(f'{PROJ_ROOT}/references/variables_labels_FI.csv', delimiter=';', index_col=0), 'title': 'Random forest model'},
                
                'RF_residual': {'files':[f"{MODELS_DIR}/RandomForest/RF_rUd_models.pkl", f"{MODELS_DIR}/RandomForest/RF_rVd_models.pkl"], 
                                'stats':[PROJ_ROOT/ 'data'/ 'processed'/ 'Statistics models'/'RandomForest'/'RF_rUd_FI_stats', PROJ_ROOT/ 'data'/ 'processed'/ 'Statistics models'/'RandomForest'/'RF_rVd_FI_stats'], 
                                'pfi': [PROJ_ROOT/ 'data'/ 'processed'/ 'PFI'/'RandomForest'/'RF_rUd_FI_pfi', PROJ_ROOT/ 'data'/ 'processed'/ 'PFI'/'RandomForest'/'RF_rVd_FI_pfi'], 
                                'labels': pd.read_csv(f'{PROJ_ROOT}/references/variables_labels_residual.csv', delimiter=';', index_col=0), 'title': 'Random forest model'},
              #  'RF_FI_total': {'files':[f"{MODELS_DIR}/RandomForest/RF_rUd_FI_models.pkl", f"{MODELS_DIR}/RandomForest/RF_rVd_FI_models.pkl", f"{MODELS_DIR}/RandomForest/RF_FI_fit_models.pkl"], 
               #                 'stats':[PROJ_ROOT/ 'data'/ 'processed'/ 'Statistics models'/'RandomForest'/'RF_Ud_FI_stats'], 
                #                'labels': [pd.read_csv(f'{PROJ_ROOT}/references/variables_labels_FI.csv', delimiter=';', index_col=0)]}
             #   'SVR_total': {'files':[f"{MODELS_DIR}/RandomForest/RF_Ud_SVR_models.pkl", f"{MODELS_DIR}/RandomForest/RF_Vd_SVR_models.pkl"],  } 
             'SVR_total': {'files':[f"{MODELS_DIR}/SVR/SVR_Ud_models.pkl", f"{MODELS_DIR}/SVR/SVR_Vd_models.pkl"], 
                                'stats':[PROJ_ROOT/ 'data'/ 'processed'/ 'Statistics models'/'SVR'/'SVR_Ud_stats', PROJ_ROOT/ 'data'/ 'processed'/ 'Statistics models'/'SVR'/'SVR_Vd_stats'], 
                                'pfi': [PROJ_ROOT/ 'data'/ 'processed'/ 'PFI'/'SVR'/'SVR_Ud_pfi', PROJ_ROOT/ 'data'/ 'processed'/ 'PFI'/'SVR'/'SVR_Vd_pfi'], 
                                'labels': pd.read_csv(f'{PROJ_ROOT}/references/variables_labels.csv', delimiter=';', index_col=0), 'title': 'Support vector regression model'},
             'SVR_resdiual': {'files':[f"{MODELS_DIR}/SVR/SVR_rUd_models.pkl", f"{MODELS_DIR}/SVR/SVR_rVd_models.pkl"], 
                                'stats':[PROJ_ROOT/ 'data'/ 'processed'/ 'Statistics models'/'SVR'/'SVR_rUd_stats', PROJ_ROOT/ 'data'/ 'processed'/ 'Statistics models'/'SVR'/'SVR_rVd_stats'], 
                                'pfi': [PROJ_ROOT/ 'data'/ 'processed'/ 'PFI'/'SVR'/'SVR_rUd_pfi', PROJ_ROOT/ 'data'/ 'processed'/ 'PFI'/'SVR'/'SVR_rVd_pfi'], 
                                'labels': pd.read_csv(f'{PROJ_ROOT}/references/variables_labels_residual.csv', delimiter=';', index_col=0), 'title': 'Support vector regression model'},
}

#%%

# %%

for model in ML_model_settings:
    output_path = FIGURES_DIR / 'PFI'/ f'pfi_{model}'
    ustats=pd.read_csv(f'{ML_model_settings[model]['stats'][0]}.csv', delimiter=',')
    upfi=pd.read_csv(f'{ML_model_settings[model]['pfi'][0]}.csv', delimiter=',')
    vstats=pd.read_csv(f'{ML_model_settings[model]['stats'][1]}.csv', delimiter=',')
    vpfi=pd.read_csv(f'{ML_model_settings[model]['pfi'][1]}.csv', delimiter=',')
    labels=ML_model_settings[model]['labels']


    plot_pfi_shadow(upfi, vpfi, ustats, vstats, labels, output_path, residual=('residual' in model), bar_width=0.6, model_name=ML_model_settings[model]['title'] )
# %%
labels=pd.read_csv(f'{PROJ_ROOT}/references/variables_labels.csv', delimiter=';', index_col=0)
importances_fi= pd.read_csv(f'{PROJ_ROOT}/data/processed/PFI/RandomForest/RF_FI_fit_pfi.csv', delimiter=',')
stats_fi= pd.read_csv(f'{PROJ_ROOT}/data/processed/Statistics models/RandomForest/RF_FI_fit_stats.csv', delimiter=',')
name='FI'

plot_single_pfi(importances_fi, stats_fi, labels, name, FIGURES_DIR / 'PFI'/ f'pfi_FI', output_format='svg', color='#817F82')

# %%
labels=pd.read_csv(f'{PROJ_ROOT}/references/variables_labels.csv', delimiter=';', index_col=0)
importances_fi= pd.read_csv(f'{PROJ_ROOT}/data/processed/PFI/RandomForest/RF_FI_twostep_regression_pfi.csv', delimiter=',')
stats_fi= pd.read_csv(f'{PROJ_ROOT}/data/processed/Statistics models/RandomForest/RF_FI_twostep_regression_stats.csv', delimiter=',')
name='FI regression'

plot_single_pfi(importances_fi, stats_fi, labels, name, FIGURES_DIR / 'PFI'/ f'pfi_FI_twostep', output_format='svg', color='#817F82')

# %%
labels=pd.read_csv(f'{PROJ_ROOT}/references/variables_labels.csv', delimiter=';', index_col=0)
importances_fi= pd.read_csv(f'{PROJ_ROOT}/data/processed/PFI/RandomForest/RF_FI_twostep_class_pfi.csv', delimiter=',')
stats_fi= pd.read_csv(f'{PROJ_ROOT}/data/processed/Statistics models/RandomForest/RF_FI_twostep_class_stats.csv', delimiter=',')
name='FI classification'

plot_single_pfi(importances_fi, stats_fi, labels, name, FIGURES_DIR / 'PFI'/ f'pfi_FI_twostep_class', output_format='svg', color='#817F82' )

# %%
