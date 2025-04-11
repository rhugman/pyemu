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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
                     energy_threshold=0.9999,
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
                The energy threshold for the SVD. Default is 0.9999.
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

            
            

            
        def apply_feature_transforms(self,log_transform=None):
            #TODO
            if log_transform is None:
                log_transform = self.log_transform
            if log_transform is True:
                log_transform = self.data.columns.tolist()
            
            df = self.data.copy()
            if isinstance(df, ObservationEnsemble):
                df = df._df
            ft = FeatureTransformer(df)

            #log transform
            if log_transform != False:
                ft.apply("log10", columns=log_transform)
                log_transformed = ft.df.copy()
            #normal score transform
            #TODO
            

            #autoencoder
            #TODO


            #update self
            self.feature_transfomer = ft
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
                print(f"Truncated from {len(s)} to {num_components} components while retaining {self.energy_threshold*100:.1f}% of variance")
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
            #TODO
            
            pmat = self.pmat
            ovals = self.ovals

            sim_vals = ovals + np.dot(pmat,pvals)

            #TODO: inverse transforms...
            
            ft = self.feature_transfomer
            ft.inverse_on_external_df(sim_vals)
            sim_vals = ft.inverse(columns=["a", "b"])

            self.sim_vals = sim_vals

            return sim_vals
        
        def check_for_pdc():
            #TODO
            return
            


class FeatureTransform:
    """
    Class for feature transforms.
    """
    def __init__(self, data=None):
        """
        Initialize the FeatureTransform class.

        Parameters
        ----------
        #TODO
        """

    class RowWiseMinMaxScaler:
        def __init__(self, feature_range=(-1, 1), groups=None, fit_groups=None):
            """
            Parameters:
            feature_range: tuple (min, max) to scale into.
            groups: dict mapping group names to lists of column names to be scaled (the entire timeseries for that group).
            fit_groups: dict mapping group names to lists of column names (a subset of the above) used to compute the row‐wise min and max.
                        If not provided, defaults to groups.
            """
            super().__init__()
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
        
    class NormalScoreTransform:
        """
        Class for normal score transformation.
        """

        from scipy.stats import norm, rankdata
        from scipy.interpolate import interp1d

        def __init__(self, pst=None, sim_ensemble=None):
            super().__init__()
            self.pst = pst
            self.sim_ensemble = sim_ensemble
            self._norm_score = None      
       

        def transform(data):
            data = np.asarray(data)
            ranks = rankdata(data, method='average')
            cdf = ranks / (len(data) + 1)
            nscores = norm.ppf(cdf)
            return nscores

        def inverse_transform(data, nscores, tail_fraction=0.05):
            data = np.asarray(data)
            sorted_data = np.sort(data)
            n = len(data)

            # Empirical CDF → normal scores
            cdf_vals = (np.arange(1, n + 1)) / (n + 1)
            norm_scores = norm.ppf(cdf_vals)

            # Linear interpolator (within range)
            interp = interp1d(norm_scores, sorted_data, kind='linear',
                            bounds_error=False, fill_value=np.nan)

            # Fit quadratics to tails
            k = max(3, int(tail_fraction * n))  # at least 3 points

            # Lower tail (quadratic fit)
            coef_lo = np.polyfit(norm_scores[:k], sorted_data[:k], deg=2)
            poly_lo = np.poly1d(coef_lo)

            # Upper tail (quadratic fit)
            coef_hi = np.polyfit(norm_scores[-k:], sorted_data[-k:], deg=2)
            poly_hi = np.poly1d(coef_hi)

            # Evaluate inverse with extrapolation
            result = interp(nscores)
            
            # Apply quadratic extrapolation where needed
            result = np.where(nscores < norm_scores[0], poly_lo(nscores), result)
            result = np.where(nscores > norm_scores[-1], poly_hi(nscores), result)

            return result


    class LogTransformer:
        """
        Class for log transformation.
        """

        def __init__(self, df: pd.DataFrame):
            self.df = df.copy()
            self._log10_columns = {}

        def log10_transform(self, columns, shift=1e-6):
            for col in columns:
                if col in self.df.columns:
                    self.df[col] = np.log10(self.df[col] + shift)
                    self._log10_columns[col] = {"shift": shift}
                else:
                    raise ValueError(f"Column '{col}' not in DataFrame")

        def inverse_log10_transform(self):
            for col, meta in self._log10_columns.items():
                shift = meta["shift"]
                self.df[col] = (10 ** self.df[col]) - shift



class FeatureTransformer:
    """
    Class for transforming features in a DataFrame.
    This class allows for applying and inverting various transformations
    such as log transformations, standard scaling, etc.

    # Sample usage
    df = pd.DataFrame({
        "a": [1, 10, 100],
        "b": [0.1, 0.5, 1.0]
    })

    ft = FeatureTransformer(df)
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


@FeatureTransformer.register_transform("log10")
def _log10(self, col, epsilon=1e-6):
    min_val = self.df[col].min()
    shift = -min_val + epsilon if min_val <= 0 else epsilon
    self.df[col] = np.log10(self.df[col] + shift)
    return {"shift": shift}


@FeatureTransformer.register_inverse("log10")
def _inv_log10(self, col, shift):
    self.df[col] = (10 ** self.df[col]) - shift

@FeatureTransformer.register_transform("standard")
def _standard_scale(self, col):
    mean, std = self.df[col].mean(), self.df[col].std()
    self.df[col] = (self.df[col] - mean) / std
    return {"mean": mean, "std": std}

@FeatureTransformer.register_inverse("standard")
def _inv_standard_scale(self, col, mean, std):
    self.df[col] = self.df[col] * std + mean
