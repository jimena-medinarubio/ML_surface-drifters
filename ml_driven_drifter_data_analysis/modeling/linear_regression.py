#%%

import xarray as xr
import numpy as np
from config import DATA_DIR
from sklearn.linear_model import LinearRegression
from features import create_feature_matrix
#%%
dt_features=xr.open_datatree(f'{DATA_DIR}/interim/interpolated_atm_ocean_datasets.nc')

#%%
def linear_regression(y, X):
    model = LinearRegression().fit(X, y)
    coeffs= model.coef_
    r2_total = model.score(X, y)
    return coeffs, r2_total

def select_variables(features, velocity_component='vx', dir='U', relative_wind=False):

    if relative_wind==False:
        X= features[velocity_component]-features[dir]-features[f'{dir}stokes']
        y= features[f'{dir}10']
    
    else:
        X= features[velocity_component]-features[dir]-features[f'{dir}stokes']
        y= features[f'{dir}10']-features[dir]
    
    return X, y

def calculate_uncertainties_fit(features_matrix, relative_wind=False):
    drifter_list=np.unique(features_matrix['drifter_id'])

    coeffsu=[]
    coeffsv=[]
    
    for i, drifter in enumerate(drifter_list):
        features_drifter=features_matrix[features_matrix['drifter_id']==list(drifter_list).pop(i)]
        Xu, yu= select_variables(features_drifter, 'vx', 'U', relative_wind=relative_wind )
        Xv, yv= select_variables(features_drifter, 'vy', 'V', relative_wind=relative_wind)
        coeff_u, r2u= linear_regression(Xu, yu.values.reshape(-1, 1))
        coeff_v, r2v= linear_regression(Xv, yv.values.reshape(-1, 1))
        coeffsu.append(coeff_u)
        coeffsv.append(coeff_v)

    print('u:', np.mean(coeffsu), np.std(coeffsu))
    print('v:', np.mean(coeffsv), np.std(coeffsv))

    return coeffsu, coeffsv

# %%
features=create_feature_matrix(dt_features, ['U', 'V', 'U10', 'V10', 'Ustokes', 'Vstokes', 'Tp', 'vx', 'vy'], waves=False, fc=False)

# %%

#1st-order approximation
Xu, yu= select_variables(features, 'vx', 'U', relative_wind=False )
Xv, yv= select_variables(features, 'vy', 'V', relative_wind=False )

coeff_u, r2u= linear_regression(Xu, yu.values.reshape(-1, 1))
coeff_v, r2v= linear_regression(Xv, yv.values.reshape(-1, 1))

coeffsu, coeffsv= calculate_uncertainties_fit(features, False)
# %%
#1st-order approximation: relative wind

Xu, yu= select_variables(features, 'vx', 'U', relative_wind=True)
Xv, yv= select_variables(features, 'vy', 'V', relative_wind=True)

coeff_u, r2u= linear_regression(Xu, yu.values.reshape(-1, 1))
coeff_v, r2v= linear_regression(Xv, yv.values.reshape(-1, 1))

coeffsu, coeffsv= calculate_uncertainties_fit(features, True)
# %%

# %%
