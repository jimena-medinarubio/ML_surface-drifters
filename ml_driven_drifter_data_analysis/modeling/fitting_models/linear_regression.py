#%%

import xarray as xr
import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit

#%%
sys.path.append("..")
from config import DATA_DIR
from features import create_feature_matrix

interpolated_data_file=f'{DATA_DIR}/interim/interpolated_atm_ocean_datasets_depth.nc'

#%%
def select_variables(features, velocity_component='vx', dir='U', relative_wind=False):

    if relative_wind==False:
        y= features[velocity_component]-features[dir]-features[f'{dir}stokes']
        X= features[f'{dir}10']
    
    else:
        y= features[velocity_component]-features[dir]-features[f'{dir}stokes']
        X= features[f'{dir}10']-features[dir]
    
    return X, y

def calculate_stats(y, y_pred):
    print('R2:', r2_score(y, y_pred))
    print('RMSE:', root_mean_squared_error(y, y_pred))
    print('MAE:', np.mean(abs(y - y_pred)))

def linear_regression(X, y):
    """
    linear regression fit of feature matrix & target variable (velocity component)
    """
    model = LinearRegression()

    #standardize X
    scaler_X = StandardScaler()
    X_standardized = scaler_X.fit_transform(X.values.reshape(-1, 1))

    #standardize y 
    scaler_y = StandardScaler()
    y_standardized = scaler_y.fit_transform(y.values.reshape(-1, 1))

    #fit model
    model.fit(X_standardized, y_standardized)

    #de-standardize the slope
    beta_standardized = model.coef_[0]  # Coefficient after standardization
    beta_de_standardized = beta_standardized * (scaler_y.scale_/scaler_X.scale_ )

    # predicted values and de-standardized y
    y_pred = scaler_y.inverse_transform(model.predict(X_standardized))
    
    calculate_stats(y.values, y_pred)

    return beta_de_standardized, r2_score(y, y_pred)

def calculate_uncertainties_fit(features_matrix, relative_wind=False, scalar=False):

    """
    option B: calculate uncertainties of the linear regression fit by leaving out one drifter at a time
    """
    drifter_list=np.unique(features_matrix['drifter_id'])

    coeffsu=[]
    coeffsv=[]
    
    for i, drifter in enumerate(drifter_list):
        features_drifter=features_matrix[features_matrix['drifter_id'] != drifter]
        
        Xu, yu= select_variables(features_drifter, 'vx', 'U', relative_wind=relative_wind )
        Xv, yv= select_variables(features_drifter, 'vy', 'V', relative_wind=relative_wind)

        Xu=Xu.dropna()
        Xv=Xv.dropna()

        if scalar==False:
            coeff_u, r2u= linear_regression(Xu, yu)
            coeff_v, r2v= linear_regression(Xv, yv)
            coeffsu.append(coeff_u)
            coeffsv.append(coeff_v)
        
        else:
            coeff_u, r2u= linear_regression(pd.concat([Xu, Xv]), pd.concat([yu, yv]))
            coeffsu.append(coeff_u)
    
    print('u:', np.mean(coeffsu), np.std(coeffsu),)
    if len(coeffsv)>1:
        print('v:', np.mean(coeffsv), np.std(coeffsv))

    return coeffsu, coeffsv

# sigmoid function fit
def sigmoid(x, L, x0, k, e):
    return L / (1 + np.exp(-k * (x - x0)))+e

#rescaling output parameters after fitting sigmoid function
def rescale_sigmoid_params(popt, scaler_X, scaler_y):
    L_std, x0_std, k_std, e_std = popt

    # Reverse standardization
    L = L_std * scaler_y.scale_[0]
    e = e_std * scaler_y.scale_[0] + scaler_y.mean_[0]
    
    # x0 and k require a change of variable
    x0 = x0_std * scaler_X.scale_[0] + scaler_X.mean_[0]
    k = k_std / scaler_X.scale_[0]
    
    return L, x0, k, e

def fit_sigmoid(X, y,):
    X = np.array(X)
    y = np.array(y)

    scaler_X = StandardScaler()
    X_standardized = scaler_X.fit_transform(X.reshape(-1, 1)).ravel()
    # Standardize y (if needed, for example, if target variable has a different scale)
    scaler_y = StandardScaler()
    y_standardized = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    # Initial parameter guess: [L, x0, k]
    p0 = [max(y), np.median(y), 1, 0]
    # Fit the curve
    popt, pcov= curve_fit(sigmoid, X_standardized, y_standardized, p0, bounds = ([0, min(X_standardized), -np.inf, -np.inf], [np.inf, max(X_standardized), np.inf, np.inf]))
    pop_non, pcov= curve_fit(sigmoid, X, y, p0, bounds = ([0, min(X), -np.inf, -np.inf], [np.inf, max(X), np.inf, np.inf]))
    
    return rescale_sigmoid_params(popt, scaler_X, scaler_y)  # [L, x0, k, e]

# %%

# physics-based linear regression model

if __name__ == "__main__":
    dt_features=xr.open_datatree(interpolated_data_file)
    features=create_feature_matrix(dt_features, ['U', 'V', 'U10', 'V10', 'Ustokes', 'Vstokes', 'Tp', 'vx', 'vy'], waves=False, fc=False)

    Xu, yu= select_variables(features, 'vx', 'U', relative_wind=False )
    Xv, yv= select_variables(features, 'vy', 'V', relative_wind=False )
    coeff_u, r2u= linear_regression(Xu, yu)
    print(coeff_u)

    coeff_v, r2v= linear_regression(Xv, yv)
    print(coeff_v)
 #   coeffsu, coeffsv= calculate_uncertainties_fit(features, False)

#%%
# physics-based linear regression model with relative wind
if __name__ == "__main__":
    Xu_rw, yu_rw= select_variables(features, 'vx', 'U', relative_wind=True)
    Xv_rw, yv_rw= select_variables(features, 'vy', 'V', relative_wind=True)

    coeff_u, r2u= linear_regression(Xu_rw, yu_rw)
    print(coeff_u)

    coeff_v, r2v= linear_regression(Xv_rw, yv_rw)
    print(coeff_v)

# %%

if __name__ == "__main__":
    Xu, yu= select_variables(features, 'vx', 'U', relative_wind=False )
    params=fit_sigmoid(Xu, yu)
    yupred=sigmoid(Xu, *params)
    calculate_stats(yu, yupred)
    print(params)
   
    Xv, yv= select_variables(features, 'vy', 'V', relative_wind=False )
    params=fit_sigmoid(Xv, yv)
    yupred=sigmoid(Xv, *params)
    calculate_stats(yv, yupred)
    print(params)


# %%

#testing Charnock function fit
from scipy.special import lambertw

def charnock_function(wind, a, b, g=9.81, k=0.4, zstar=0.0144, zprime=10):

    #wind=u10+v10*1j
    root=np.sqrt(zprime*g/zstar)
    x=-wind*k/(2*root)

    return a*root*lambertw(x).real+b

def fit_charnock(X, y):
    X = np.array(X)
    y = np.array(y)

    p0 = [1, 0]
    # Fit the curve
    popt, pcov = curve_fit(charnock_function, X, y, p0, maxfev=10000)
    
    return popt , np.sqrt(np.diag(pcov))
# %%
