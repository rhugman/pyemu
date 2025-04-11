from __future__ import print_function, division
import os
import copy
import shutil
from datetime import datetime
import warnings
from .pyemu_warnings import PyemuWarning
import numpy as np
import pandas as pd
from pyemu.en import ObservationEnsemble
from pyemu.mat.mat_handler import Matrix, Jco, Cov
from pyemu.pst.pst_handler import Pst
from .logger import Logger
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
#from scipy.stats import norm

class Emulator:
    """
    Base class for emulators. This class is not intended to be used directly.
    Instead, use one of the subclasses: DSI, GPR, etc. #TODO
    """

    def __init__(self,verbose=False):
        """
        Initialize the Emulator class.

        Args:
        ----------
        """

        self.logger = Logger(verbose)
        self.log = self.logger.log


    #TODO: do we want to handle all data preprocessing/transforms here???


    class DSI:
        """
        Class for the DSI emulator.
        """

        def __init__(self, 
                     pst=None,
                     sim_ensemble=None,
                     normal_score_transform=False,
                     nst_extrapolate=False,
                     energy_threshold=1.0,
                     log_transform=False,verbose=False):
            """
            Initialize the DSI emulator.

            Parameters
            ----------
            pst : Pst, optional
                A Pst object. If provided, the emulator will be initialized with the
                information from the Pst object.
            sim_ensemble : ObservationEnsemble, optional
                An ensemble of simulated observations. If provided, the emulator will
                be initialized with the information from the ensemble.
            normal_score_transform : bool, optional
                If True, the emulator will apply a normal score transformation to the
                simulated observations.
            energy_threshold : float, optional 
                The energy threshold for the SVD. Default is 1.0, no truncation.
            log_transform : bool or list, optional
                If True, the emulator will apply a log transformation to all the simulated
                observations. If list of observation column names, will be applied to these. Default is False.
            """

            

            super().__init__()
            self.logger = Logger(verbose)
            self.log = self.logger.log

            self.pst = pst
            self.__org_sim_ensemble = sim_ensemble
            self.data = sim_ensemble
            self.data_transformed = None
            self.feature_scaler = None
            self.energy_threshold = energy_threshold
            if log_transform is True:
                self.log_transform = sim_ensemble.columns.tolist()
            else:
                self.log_transform = log_transform
            assert isinstance(self.log_transform, (bool, list)), "log_transform must be a boolean or a list of column names"
            self.nst_extrapolate=nst_extrapolate
            
            

            
        def apply_feature_transforms(self,log_transform=None,nst_extrapolate=None,normal_score_transform=None):

            if log_transform is None:
                log_transform = self.log_transform
            if log_transform is True:
                log_transform = self.data.columns.tolist()
            
            if normal_score_transform is None:
                normal_score_transform = self.normal_score_transform
            if nst_extrapolate is None:
                nst_extrapolate = self.nst_extrapolate

            self.logger.statement("applying feature transforms")
            df = self.data.copy()
            if isinstance(df, ObservationEnsemble):
                df = df._df
            ft = AutobotsAssemble(df)

            #log transform
            if log_transform != False:
                self.logger.statement("applying log transform")
                ft.apply("log10", columns=log_transform)
                #log_transformed = ft.df.copy()
            #normal score transform
            if normal_score_transform:
                self.logger.statement("applying normal score transform")
                ft.apply("normal_score",
                         columns=ft.df.columns.tolist(),
                         quadratic_extrapolation=nst_extrapolate)
                #normal_score_transformed = ft.df.copy()
            

            #autoencoder
            #TODO

        

            #update self
            self.feature_transformer = ft
            self.data_transformed = ft.df.copy()
            return
        
        def compute_projection_matrix(self,energy_threshold=None):
            self.logger.statement("normalizing data")
            # normalize the data by subtracting the mean and dividing by the standard deviation
            X = self.data_transformed.copy()
            deviations = X - X.mean()
            z = deviations / np.sqrt(float(X.shape[0] - 1))
            if isinstance(z, pd.DataFrame):
                z = z.values

            self.logger.statement("undertaking SVD")
            u, s, v = np.linalg.svd(z, full_matrices=False)
            us = np.dot(v.T, np.diag(s))
            if energy_threshold is None:
                energy_threshold = self.energy_threshold
            if energy_threshold<1.0:
                self.logger.statement("applying energy truncation")
                # compute the cumulative energy of the singular values
                cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
                print(cumulative_energy)
                # find the number of components needed to reach the energy threshold
                num_components = np.argmax(cumulative_energy >= energy_threshold) + 1
                # keep only the first num_components singular values and vectors
                us = us[:, :num_components]
                s = s[:num_components]
                u = u[:, :num_components]
                print(f"Truncated from {len(s)} to {num_components} components while retaining {energy_threshold*100:.1f}% of variance")
                if num_components<=1:
                    print(f"Warning: only {num_components} component retained, you may need to check the data")
            
            self.logger.statement("calculating us matrix")
           

            # store components needed for forward run
            # store mean vector
            self.ovals = self.data_transformed.mean(axis=0)
            # store proj matrix and singular values
            self.pmat = us
            self.s = s
            return
        


        def forward_run(self,pvals):
           
            pmat = self.pmat
            ovals = self.ovals

            sim_vals = ovals + np.dot(pmat,pvals)

            # inverse transforms...
            
            ft = self.feature_transformer
            sim_vals = ft.inverse_on_external_df(sim_vals, columns=self.data_transformed.columns.tolist())

            self.sim_vals = sim_vals

            return sim_vals
        
        def check_for_pdc():
            #TODO
            return
            






class AutobotsAssemble:
    """
    Class for transforming features in a DataFrame.
    This class allows for applying and inverting various transformations
    such as log transformations, standard scaling, etc.

    # Sample usage
    df = pd.DataFrame({
        "a": [1, 10, 100],
        "b": [0.1, 0.5, 1.0]
    })

    ft = AutobotsAssemble(df)
    ft.apply("log10", columns=["a", "b"])
    log_transformed = ft.df.copy()
    ft.apply("standard", columns=["a"])
    standard_transformed = ft.df.copy()
    
    ft.inverse(columns=["a", "b"])
    inverted_df = ft.df.copy()
    
    """

    _transform_funcs = {}
    _inverse_funcs = {}
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._transforms = {}
        self._shared_z_scores = None  # For reusing z-scores across columns

    @classmethod
    def register_transform(cls, name):
        def decorator(func):
            cls._transform_funcs[name] = func
            return func
        return decorator

    @classmethod
    def register_inverse(cls, name):
        def decorator(func):
            cls._inverse_funcs[name] = func
            return func
        return decorator

    def apply(self, transform_name, columns, **kwargs):
        if transform_name not in self._transform_funcs:
            raise ValueError(f"Transform '{transform_name}' not registered.")
        for col in columns:
            func = self._transform_funcs[transform_name]
            params = func(self, col, **kwargs)
            if col not in self._transforms:
                self._transforms[col] = []
            self._transforms[col].append({"type": transform_name, "params": params})

    def inverse(self, columns=None):
        columns = columns or list(self._transforms.keys())
        for col in columns:
            if col not in self._transforms:
                continue
            for step in reversed(self._transforms[col]):
                inv_func = self._inverse_funcs.get(step["type"])
                if inv_func:
                    inv_func(self, col, **step["params"])

    def inverse_on_external_df(self, df, columns=None):
        out = df.copy()
        columns = columns or self._transforms.keys()
        for col in columns:
            if col not in self._transforms:
                continue
            for step in reversed(self._transforms[col]):
                inv_func = self._inverse_funcs.get(step["type"])
                if inv_func:
                    # Temporarily patch self.df to use the external df
                    original_df = self.df
                    self.df = out
                    inv_func(self, col, **step["params"])
                    out = self.df
                    self.df = original_df
        return out


@AutobotsAssemble.register_transform("log10")
def _log10(self, col):
    min_val = self.df[col].min()
    shift = -min_val + 1e-6 if min_val <= 0 else 0
    self.df[col] = np.log10(self.df[col] + shift)
    return {"shift": shift}

@AutobotsAssemble.register_inverse("log10")
def _inv_log10(self, col, shift):
    self.df[col] = (10 ** self.df[col]) - shift

@AutobotsAssemble.register_transform("standard")
def _standard_scale(self, col):
    mean, std = self.df[col].mean(), self.df[col].std()
    self.df[col] = (self.df[col] - mean) / std
    return {"mean": mean, "std": std}

@AutobotsAssemble.register_inverse("standard")
def _inv_standard_scale(self, col, mean, std):
    self.df[col] = self.df[col] * std + mean


class NormalScoreTransformer:
    def __init__(self, tol=1e-7, max_samples=1000000):
        self.tol = tol
        self.max_samples = max_samples
        

    def randrealgen_optimized(self, nreal):
        rval = np.zeros(nreal)
        nsamp = 0
        numsort = (nreal + 1) // 2 if nreal % 2 == 0 else nreal // 2

        while nsamp < self.max_samples:
            nsamp += 1
            work1 = np.random.normal(size=nreal)
            work1.sort()

            if nsamp > 1:
                previous_mean = rval[:numsort] / (nsamp - 1)
                rval[:numsort] += work1[:numsort]
                current_mean = rval[:numsort] / nsamp
                max_diff = np.max(np.abs(current_mean - previous_mean))

                if max_diff <= self.tol:
                    break
            else:
                rval[:numsort] = work1[:numsort]

        rval[:numsort] /= nsamp
        rval[numsort:] = -rval[:numsort][::-1] if nreal % 2 == 0 else np.concatenate(([-rval[numsort]], -rval[:numsort][::-1]))
        return rval




@AutobotsAssemble.register_transform("normal_score")
def _normal_score(self, col, tol=1e-7, max_samples=1000000,quadratic_extrapolation=False):
    x = self.df[col].values
    sorted_vals = np.sort(x)
    sorted_vals = _moving_average_with_endpoints(sorted_vals)

    if self._shared_z_scores is None or len(self._shared_z_scores) != len(sorted_vals):
        nst = NormalScoreTransformer(tol=tol, max_samples=max_samples)
        self._shared_z_scores = nst.randrealgen_optimized(len(sorted_vals))

    self.df[col] = np.interp(x, sorted_vals, self._shared_z_scores)
    return {
        "z_scores": self._shared_z_scores.tolist(),
        "originals": sorted_vals.tolist(),
        "quadratic_extrapolation": quadratic_extrapolation,
    }

@AutobotsAssemble.register_inverse("normal_score")
def _inv_normal_score(self, col, z_scores, originals, quadratic_extrapolation=False):
    z_scores = np.array(z_scores)
    originals = np.array(originals)
    z_vals = self.df[col]
    if isinstance(z_vals, pd.Series):
        z_vals = z_vals.values
    interpolated = np.interp(z_vals, z_scores, originals)

    if quadratic_extrapolation:
        low_mask = z_vals < z_scores.min()
        high_mask = z_vals > z_scores.max()

        if low_mask.any():
            coeffs_low = np.polyfit(z_scores[:3], originals[:3], deg=2)
            interpolated[low_mask] = np.polyval(coeffs_low, z_vals[low_mask])

        if high_mask.any():
            coeffs_high = np.polyfit(z_scores[-3:], originals[-3:], deg=2)
            interpolated[high_mask] = np.polyval(coeffs_high, z_vals[high_mask])

    self.df[col] = interpolated


def _moving_average_with_endpoints(y_values):
    # apply smoothing as per DSI2; window sizes are arbitrary...                
    window_size=3   
    if y_values.shape[0]>40:
        window_size=5                    
    if y_values.shape[0]>90:
        window_size=7
    if y_values.shape[0]>200:
        window_size=9     

    # Ensure the window size is odd
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    # Calculate half-window size
    half_window = window_size // 2
    # Initialize the output array
    smoothed_y = np.zeros_like(y_values)
    # Handle the endpoints
    for i in range(0,half_window):
        # Start
        smoothed_y[i] = np.mean(y_values[:i + half_window ])
    for i in range(1,half_window+1):
        # End
        smoothed_y[-i] = np.mean(y_values[::-1][:i + half_window +1])
    # Handle the middle part with full window
    for i in range(half_window, len(y_values) - half_window):
        smoothed_y[i] = np.mean(y_values[i - half_window:i + half_window])
    #Enforce endpoints
    smoothed_y[0] = y_values[0]
    smoothed_y[-1] = y_values[-1]
    # Ensure uniqueness by adding small increments if values are duplicated
    #NOTE: this is a hack to ensure uniqueness in the normal score transform
    for i in range(1, len(smoothed_y)):
        if smoothed_y[i] <= smoothed_y[i - 1]:
            smoothed_y[i] = smoothed_y[i - 1] + 1e-16

    return smoothed_y


#TODO parse into the AutobotsAssemble class
class RowWiseMinMaxScaler:
    def __init__(self, feature_range=(-1, 1), groups=None, fit_groups=None):
        """
        Parameters:
        feature_range: tuple (min, max) to scale into.
        groups: dict mapping group names to lists of column names to be scaled (the entire timeseries for that group).
        fit_groups: dict mapping group names to lists of column names (a subset of the above) used to compute the row‐wise min and max.
                    If not provided, defaults to groups.
        """
        assert isinstance(fit_groups,dict), "fit_groups must be a dictionary or None"
        assert isinstance(groups, dict), "groups must be a dictionary"
        
        self.feature_range = feature_range
        self.groups = groups
        self.fit_groups = fit_groups if fit_groups is not None else groups
        self._row_params = {}  # will store per–row (min, max) for each group on the last transform call

    def fit(self, X):
        # For row–wise scaling, nothing needs to be learned globally.
        return self

    def transform(self, X):
        # X is a pandas DataFrame.
        f_min, f_max = self.feature_range
        X_scaled = X.copy()
        self._row_params = {}  # reset stored row parameters

        for group_name, group_cols in self.groups.items():
            # Determine which columns to use for computing min/max for each row.
            fit_cols = self.fit_groups.get(group_name, group_cols)
            # Compute row–wise min and max using the fit columns.
            row_min = X[fit_cols].min(axis=1)
            row_max = X[fit_cols].max(axis=1)
            # Avoid division by zero: if a row is constant over fit_cols, set its range to 1.
            row_range = row_max - row_min
            row_range[row_range == 0] = 1
            # Store these parameters for inverse transformation.
            self._row_params[group_name] = (row_min, row_max)
            # For all columns in the group, subtract the row's min and divide by the row's range.
            # Using broadcasting with .sub(..., axis=0) and .div(..., axis=0)
            group_data = X[group_cols]
            group_std = group_data.sub(row_min, axis=0).div(row_range, axis=0)
            # Scale to the desired feature range.
            X_scaled[group_cols] = group_std * (f_max - f_min) + f_min

        return X_scaled

    def inverse_transform(self, X_scaled):
        f_min, f_max = self.feature_range
        X_orig = X_scaled.copy()
        if not self._row_params:
            raise ValueError("No stored row parameters. Make sure to call transform before inverse_transform.")
        for group_name, group_cols in self.groups.items():
            if all(col not in X_scaled.columns for col in group_cols):
                continue
            row_min, row_max = self._row_params[group_name]
            row_range = row_max - row_min
            row_range[row_range == 0] = 1
            group_data = X_scaled[group_cols]
            # Inverse scaling: first convert from feature_range to [0, 1]
            group_std = (group_data - f_min) / (f_max - f_min)
            # Then recover original values
            X_orig[group_cols] = group_std.mul(row_range, axis=0).add(row_min, axis=0)
            
        return X_orig