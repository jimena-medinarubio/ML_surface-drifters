#%%
import pandas as pd
import sys
sys.path.append("..")
from config import REFS_DIR, PROJ_ROOT , FIGURES_DIR, PROCESSED_DATA_DIR
from modeling.model_agnostics import plot_pfi_shadow,  plot_single_pfi
#%%

ML_model_settings={'RF_total': {'stats':[PROCESSED_DATA_DIR/ 'Statistics models'/'RandomForest'/'RF_Ud_stats', PROCESSED_DATA_DIR/ 'Statistics models'/'RandomForest'/'RF_Vd_stats'], 
                                'pfi': [PROCESSED_DATA_DIR/ 'PFI'/'RandomForest'/'RF_Ud_pfi', PROCESSED_DATA_DIR/ 'PFI'/'RandomForest'/'RF_Vd_pfi'], 
                                'labels': pd.read_csv(f'{REFS_DIR}/variables_labels.csv', delimiter=';', index_col=0),
                                'title': 'Random forest model'},
                
                'RF_residual': {'stats':[PROCESSED_DATA_DIR/ 'Statistics models'/'RandomForest'/'RF_rUd_FI_stats', PROCESSED_DATA_DIR/ 'Statistics models'/'RandomForest'/'RF_rVd_FI_stats'], 
                                'pfi': [PROCESSED_DATA_DIR/ 'PFI'/'RandomForest'/'RF_rUd_FI_pfi', PROCESSED_DATA_DIR/ 'PFI'/'RandomForest'/'RF_rVd_FI_pfi'], 
                                'labels': pd.read_csv(f'{REFS_DIR}/variables_labels_residual.csv', delimiter=';', index_col=0), 'title': 'Random forest model'},
             
             'SVR_total': {'stats':[PROJ_ROOT/ 'data'/ 'processed'/ 'Statistics models'/'SVR'/'SVR_Ud_stats', PROCESSED_DATA_DIR/ 'Statistics models'/'SVR'/'SVR_Vd_stats'], 
                                'pfi': [PROJ_ROOT/ 'data'/ 'processed'/ 'PFI'/'SVR'/'SVR_Ud_pfi', PROCESSED_DATA_DIR/ 'PFI'/'SVR'/'SVR_Vd_pfi'], 
                                'labels': pd.read_csv(f'{REFS_DIR}/variables_labels.csv', delimiter=';', index_col=0), 'title': 'Support vector regression model'},
             'SVR_residual': {'stats':[PROCESSED_DATA_DIR/ 'Statistics models'/'SVR'/'SVR_rUd_stats',PROCESSED_DATA_DIR/ 'Statistics models'/'SVR'/'SVR_rVd_stats'], 
                                'pfi': [PROCESSED_DATA_DIR/ 'PFI'/'SVR'/'SVR_rUd_pfi', PROCESSED_DATA_DIR/ 'PFI'/'SVR'/'SVR_rVd_pfi'], 
                                'labels': pd.read_csv(f'{REFS_DIR}/variables_labels_residual.csv', delimiter=';', index_col=0), 'title': 'Support vector regression model'},
            
            'RF_FI_classification': {'stats':[f'{PROJ_ROOT}/data/processed/Statistics models/RandomForest/RF_FI_twostep_class_stats.csv'], 
                                'pfi': [f'{PROJ_ROOT}/data/processed/PFI/RandomForest/RF_FI_twostep_class_pfi.csv.csv'], 
                                'labels': pd.read_csv(f'{REFS_DIR}/variables_labels.csv', delimiter=';', index_col=0), },

            'RF_FI_classification': {'stats':[f'{PROJ_ROOT}/data/processed/Statistics models/RandomForest/RF_FI_twostep_regression_stats.csv'], 
                                'pfi': [f'{PROJ_ROOT}/data/processed/PFI/RandomForest/RF_FI_twostep_regression_pfi.csv.csv'], 
                                'labels': pd.read_csv(f'{REFS_DIR}/variables_labels.csv', delimiter=';', index_col=0), },
}

# %%

#permutation feature importance bar plots for all models
for model in ML_model_settings:
    output_path = FIGURES_DIR / 'PFI'/ f'pfi_{model}'

    if 'FI' in model: #two-step model
        labels=ML_model_settings[model]['labels']
        stats_fi=pd.read_csv(f'{ML_model_settings[model]['stats'][0]}.csv', delimiter=',')
        importances_fi= pd.read_csv(f'{ML_model_settings[model]['pfi'][0]}.csv', delimiter=',')
        plot_single_pfi(importances_fi, stats_fi, labels,  model, FIGURES_DIR / 'PFI'/ f'pfi_FI_twostep', output_format='svg', color='#817F82')

    else:
        ustats=pd.read_csv(f'{ML_model_settings[model]['stats'][0]}.csv', delimiter=',')
        upfi=pd.read_csv(f'{ML_model_settings[model]['pfi'][0]}.csv', delimiter=',')
        vstats=pd.read_csv(f'{ML_model_settings[model]['stats'][1]}.csv', delimiter=',')
        vpfi=pd.read_csv(f'{ML_model_settings[model]['pfi'][1]}.csv', delimiter=',')
        labels=ML_model_settings[model]['labels']

        plot_pfi_shadow(upfi, vpfi, ustats, vstats, labels, output_path, residual=('residual' in model), bar_width=0.6, model_name=ML_model_settings[model]['title'] )
# %%
