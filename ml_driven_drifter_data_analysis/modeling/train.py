#%%
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from config import MODELS_DIR, PROCESSED_DATA_DIR, DATA_DIR,  PROJ_ROOT
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import  GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import clone
import joblib
#%%

class TimeBlockSplit(BaseCrossValidator):
    def __init__(self, block_duration, n_splits=5):
        """
        Parameters:
        - block_duration: str or pd.Timedelta, duration of each time block (e.g., '1D' for 1 day).
        - n_splits: int, number of splits/folds for cross-validation.
        """
        self.block_duration = pd.Timedelta(block_duration)
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and testing sets.

        Parameters:
        - X: pd.DataFrame, feature data with a datetime index.
        - y: pd.Series or pd.DataFrame, target data (optional).
        - groups: ignored, included for compatibility with sklearn.

        Yields:
        - train_indices: np.array, indices for the training set.
        - test_indices: np.array, indices for the testing set.
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex.")

        # Create time blocks
        start_time = X.index.min()
        end_time = X.index.max()
        blocks = []

        while start_time < end_time:
            block_end = start_time + self.block_duration
            block_indices = X.index[(X.index >= start_time) & (X.index < block_end)]
            if not block_indices.empty:
                blocks.append(block_indices)
            start_time = block_end

        # Shuffle blocks for randomness
        rng = np.random.default_rng()
        rng.shuffle(blocks)

        fold_size = len(blocks) // self.n_splits
        for i in range(self.n_splits):
            # Test blocks for this fold
            test_blocks = blocks[i * fold_size:(i + 1) * fold_size]
            # Train blocks are all other blocks
            train_blocks = blocks[:i * fold_size] + blocks[(i + 1) * fold_size:]

            # Flatten block indices into train and test indices
            train_indices = np.where(X.index.isin(np.concatenate(train_blocks)))[0]
            test_indices = np.where(X.index.isin(np.concatenate(test_blocks)))[0]

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splits."""
        return self.n_splits
#%%

def RF_regression(X, y, block, y_vars_name=None, ntrees=100, plot=False, calculate_permutation=False, output_path=None, model_name='test'):
    
    model=RandomForestRegressor(random_state=42, n_estimators=ntrees, oob_score=True, 
                                bootstrap=True)
    
    if plot==True:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(np.arange(np.min(y), np.max(y), 0.05), 
                np.arange(np.min(y), np.max(y), 0.05), 
                color='red', label='1:1', linestyle='--')

    colors = sns.color_palette('bright').as_hex()

    stats={'RMSE':[], 'ME':[], 'R2':[]}

    for i, (train_idx, test_idx) in enumerate(block.split(X, y)):
        print(f'Outer loop: {i+1}')
        
        X_train, X_test= X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        X_train.columns = X_train.columns.astype(str)
        model.fit(X_train, y_train)

        y_test_pred = model.predict(X_test)
        test_r2=r2_score(y_test, y_test_pred)

        stats['R2'].append(test_r2)
        stats['RMSE'].append(np.sqrt(mean_squared_error(y_test, y_test_pred)))
        stats['ME'].append(mean_absolute_error(y_test, y_test_pred))

        if plot==True:
            sns.scatterplot(ax=ax, x=y_test, y=y_test_pred, color=colors[i], label=fr'$f_{i+1}$, $R^2$={np.round(test_r2, 2)} ')

    rmse=np.round(np.mean(stats["RMSE"]), 2)
    me=np.round(np.mean(stats["ME"]), 2)
    r2=[np.round(np.mean(stats["R2"]), 2), np.round(np.std(stats["R2"]), 2)]
    
    if plot==True:
        if 'Zonal Residual' in y_vars_name:
            l=r'$\tilde{U}_d$'
        elif 'Meridional Residual' in y_vars_name:
            l=r'$\tilde{V}_d$'
        elif 'Zonal velocity' in y_vars_name:
            l='$U_d$'
        elif 'Meridional velocity' in y_vars_name:
            l='$V_d$'
        ax.set_xlabel(rf'Observed {l} [$m \, s^{{-1}}$]', fontsize=14)
        ax.set_ylabel(rf'Predicted {l} [$m \, s^{{-1}}$]', fontsize=14)
        ax.legend(loc='lower right', fontsize=12)
        ax.set_title(rf'ME={me}, RMSE: {rmse}, $R^2$={r2[0]} $\pm$ {r2[1]}', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)

        if output_path is not None:
            plt.savefig(output_path, dpi=300)
        plt.show()
    
    #FIT MODEL TO ALL DATA
    model=RandomForestRegressor(random_state=42, n_estimators=ntrees, oob_score=True, 
                                bootstrap=True)
    model.fit(X,y)

    if calculate_permutation==True:
        def rmse(y_true, y_pred):
            return np.sqrt(mean_squared_error(y_true, y_pred)) 
        mse_scorer = make_scorer(rmse)
        results = permutation_importance(model, X, y , n_repeats=10, random_state=42, scoring=mse_scorer)
        stats['permutation_importance']=pd.DataFrame(results.importances.T, columns=X.columns)
    else:
        stats['permutation_importance']=[]
    
    oob_score = model.oob_score_
    stats['oob_score']=oob_score

    save_stats([stats], [f'RF_{model_name}'], PROJ_ROOT/ 'data'/ 'processed'/ 'Statistics models'/'RandomForest' )
    save_models([model], [f'RF_{model_name}'], PROJ_ROOT/ 'models'/ 'RandomForest' )
    save_permutation([stats], [f'RF_{model_name}'], PROJ_ROOT/ 'data'/ 'processed'/ 'PFI'/'RandomForest' )
    

    return model, stats
#%%

def SVR_regression(X, y, block, y_vars_name, kernel='rbf', plot=False, calculate_permutation=False, grid_search=False, param_grid=None, params=None):
    
    
    if plot==True:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(np.arange(np.min(y), np.max(y), 0.05), 
                np.arange(np.min(y), np.max(y), 0.05), 
                color='red', label='1:1', linestyle='--')

    colors = sns.color_palette('bright').as_hex()

    sorted_indices = X.index.argsort()
    X = X.sort_index()  # Sort X by its datetime index
    y = np.array(y)[sorted_indices]

    X_scaler = StandardScaler()
    Xs = X_scaler.fit_transform(X)
    X = pd.DataFrame(Xs, columns=X.columns, index=X.index)
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    stats={'RMSE':[], 'ME':[], 'R2':[]}

    if grid_search==True:
        grid_search = GridSearchCV(
                estimator=SVR(),
                param_grid=param_grid,
                cv=3,  # Inner cross-validation for GridSearch
                scoring='neg_mean_squared_error',  # Adjust scoring as needed
                verbose=1
            )
        grid_search.fit(X, y)

            # Get the best model
        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    else:
        best_model= SVR(kernel=kernel, C=params['C'], gamma=params['gamma'], epsilon=params['epsilon'])

    for i, (train_idx, test_idx) in enumerate(block.split(X, y)):
        model_cv=best_model.copy()

        print(f'Outer loop: {i+1}')
        X_train, X_test= X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model_cv.fit(X_train, y_train)

        y_test_pred = model_cv.predict(X_test)
        y_test_pred_og=y_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
        y_test_og=y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        test_r2=r2_score(y_test_og, y_test_pred_og)
        stats['R2'].append(test_r2)
        stats['RMSE'].append(np.sqrt(mean_squared_error(y_test_og, y_test_pred_og)))
        stats['R2'].append(mean_absolute_error(y_test_og, y_test_pred_og))

        if plot==True:
            sns.scatterplot(ax=ax, x=y_test_og, y=y_test_pred_og, color=colors[i], label=fr'$f_{i+1}$, $R^2$={np.round(test_r2, 2)} ')

    rmse=np.round(np.mean(stats["RMSE"]), 3)
    me=np.round(np.mean(stats["ME"]), 3)
    r2=[np.round(np.mean(stats["R2"]), 3), np.round(np.std(stats["R2"]), 3)]
    
    if plot==True:
        if 'Zonal Residual' in y_vars_name:
            l=r'$\tilde{U}_d$'
        elif 'Meridional Residual' in y_vars_name:
            l=r'$\tilde{V}_d$'
        elif 'Residual' not in y_vars_name and 'Zonal' in y_vars_name:
            l='$U_d$'
        elif 'velocity' not in y_vars_name and 'Meridional' in y_vars_name:
            l='$V_d$'
        ax.set_xlabel(rf'Observed {l} [$m \, s^{{-1}}$]', fontsize=14)
        ax.set_ylabel(rf'Predicted {l} [$m \, s^{{-1}}$]', fontsize=14)
        ax.legend(loc='lower right', fontsize=12)
        ax.set_title(rf'ME={me}, RMSE: {rmse}, $R^2$={r2[0]} $\pm$ {r2[1]}', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        plt.show()
    
    #FIT MODEL TO ALL DATA
    best_model.fit(X,y)

    if calculate_permutation==True:
        def rmse(y_true, y_pred):
            return np.sqrt(mean_squared_error(y_true, y_pred)) 
        mse_scorer = make_scorer(rmse)
        results = permutation_importance(best_model, X, y , n_repeats=10, random_state=42, scoring=mse_scorer)
        stats['permutation_importance']=results.importances.T
    else:
        stats['permutation_importance']=[]
    
    
    return best_model, stats, X_scaler, y_scaler

#%%
def save_stats(stats_array, names, output_dir):
    for i, elem in enumerate(stats_array):
        filtered_elem = {k: v for k, v in elem.items() if k != 'permutation_importance' and k!='oob_score'}
        df = pd.DataFrame(filtered_elem)
        # Save as CSV
        df.to_csv(f"{output_dir}/{names[i]}_stats.csv", index=False)

def save_permutation(stats_array, names, output_dir):
    for i, elem in enumerate(stats_array):
        df = pd.DataFrame(elem['permutation_importance'])
        # Save as CSV
        df.to_csv(f"{output_dir}/{names[i]}_pfi.csv", index=False)

def save_models(stats_array, names, output_dir):
    for i, elem in enumerate(stats_array):
        joblib.dump(elem, f"{output_dir}/{names[i]}_models.pkl")

# %%
