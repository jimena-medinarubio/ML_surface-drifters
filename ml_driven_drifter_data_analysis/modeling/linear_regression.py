#%%

import xarray as xr
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from config import DATA_DIR
from sklearn.linear_model import LinearRegression
from features import create_feature_matrix
from sklearn.metrics import r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso

#%%
dt_features=xr.open_datatree(f'{DATA_DIR}/interim/interpolated_atm_ocean_datasets_depth.nc',)

#%%

# Example for Ridge Regression

def linear_regression(X, y):
    model = LinearRegression()
    #model=LinearRegression()
   # model=Lasso(alpha=0.1)

    scaler_X = StandardScaler()
    X_standardized = scaler_X.fit_transform(X.values.reshape(-1, 1))

    # Standardize y (if needed, for example, if target variable has a different scale)
    scaler_y = StandardScaler()
    y_standardized = scaler_y.fit_transform(y.values.reshape(-1, 1))


    model.fit(X_standardized, y_standardized)

    # Step 3: De-standardize the coefficient
    # Multiply the coefficient by the ratio of std of original X and y
    beta_standardized = model.coef_[0]  # Coefficient after standardization
    beta_de_standardized = beta_standardized * (scaler_y.scale_/scaler_X.scale_ )
# Predicted values and de-standardized y
    y_pred = model.predict(X_standardized) * scaler_y.scale_ + np.mean(y)
    
    # Calculate R^2 and RMSE
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(root_mean_squared_error(y, y_pred))

    # Calculate the residuals
    residuals = y.values - y_pred

    # Calculate the variance of the residuals
    residual_variance = np.var(residuals)

    # Compute the covariance matrix (this is for a simple linear regression with one predictor)
    X_design_matrix = np.c_[np.ones(X_standardized.shape[0]), X_standardized]
    covariance_matrix = residual_variance * np.linalg.inv(X_design_matrix.T @ X_design_matrix)

    # Standard error of the coefficient (just for the single coefficient in simple linear regression)
    standard_error = np.sqrt(covariance_matrix[1, 1])

    return beta_de_standardized, [standard_error, r2, rmse]

def select_variables(features, velocity_component='vx', dir='U', relative_wind=False):

    if relative_wind==False:
        y= features[velocity_component]-features[dir]-features[f'{dir}stokes']
        X= features[f'{dir}10']
    
    else:
        y= features[velocity_component]-features[dir]-features[f'{dir}stokes']
        X= features[f'{dir}10']-features[dir]
    
    return X, y

def calculate_uncertainties_fit(features_matrix, relative_wind=False, scalar=False):
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

# %%


features=create_feature_matrix(dt_features, ['U', 'V', 'U10', 'V10', 'Ustokes', 'Vstokes', 'Tp', 'vx', 'vy'], waves=False, fc=False)

# %%

if __name__ == "__main__":
    #1st-order approximation
    Xu, yu= select_variables(features, 'vx', 'U', relative_wind=False )
    Xv, yv= select_variables(features, 'vy', 'V', relative_wind=False )
    coeff_u, r2u= linear_regression(Xu, yu)
    coeff_v, r2v= linear_regression(Xv, yv)
 #   coeffsu, coeffsv= calculate_uncertainties_fit(features, False)

  #  coeffs_total, r2_total= linear_regression(pd.concat([Xu, Xv]), pd.concat([yu, yv]))
   # coeffsu, coeffsv= calculate_uncertainties_fit(features, False, scalar=True)

    #1st-order approximation: relative wind
#%%
if __name__ == "__main__":
    Xu_rw, yu_rw= select_variables(features, 'vx', 'U', relative_wind=True)
    Xv_rw, yv_rw= select_variables(features, 'vy', 'V', relative_wind=True)

    coeff_u, r2u= linear_regression(Xu_rw, yu_rw)
    coeff_v, r2v= linear_regression(Xv_rw, yv_rw)

    coeffsu, coeffsv= calculate_uncertainties_fit(features, True)

   # coeffs_total, r2_total= linear_regression(pd.concat([Xu_rw, Xv_rw]), pd.concat([yu_rw, yv_rw]))
    ##coeffsu, coeffsv= calculate_uncertainties_fit(features, True, scalar=True)
# %%
from scipy.optimize import curve_fit

# Sigmoid f1unction
def sigmoid(x, L, x0, k, e):
    return L / (1 + np.exp(-k * (x - x0)))+e

# Fit sigmoid to data
def fit_sigmoid(X, y):
    X = np.array(X)
    y = np.array(y)
    # Initial parameter guess: [L, x0, k]
    p0 = [max(y), np.median(X), 1, 0]
    # Fit the curve
    popt, pcov= curve_fit(sigmoid, X, y, p0, maxfev=10000)
    
    return popt, np.sqrt(np.diag(pcov))  # [L, x0, k, e]

if __name__ == "__main__":
    Xu, yu= select_variables(features, 'vx', 'U', relative_wind=False )
   
    params, error=fit_sigmoid(Xu, yu)
    print(params)
    print(error)



# %%
