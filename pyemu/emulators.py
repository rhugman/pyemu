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



class Emulator:
    """
    Base class for emulators. This class is not intended to be used directly.
    Instead, use one of the subclasses: DSI, GPR, etc. #TODO
    """

    def __init__(self,pst=None,sim_ensemble=None,verbose=False):
        """
        Initialize the Emulator class.

        Args:
        ----------
        """

        self.logger = Logger(verbose)
        self.log = self.logger.log

    class DSI:
        """
        Class for the DSI emulator.
        """

        def __init__(self, pst=None,
                     sim_ensemble=None,
                     normal_score_transform=False,
                     energy_threshold=0.9999):
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
            """
            super().__init__(pst=pst, sim_ensemble=sim_ensemble)
            self.pst = pst
            self.__org_sim_ensemble = sim_ensemble
            self.data = sim_ensemble
            self.energy_threshold = energy_threshold
            from sklearn.decomposition import PCA

            
            def apply_feature_transforms():
                #TODO
                #standardize
                #normal score transform
                #log transform
                #autoencoder
                return
            
            def _apply_pca(self):
                self.pca = PCA(n_components=self.energy_threshold)
                self.__data_pca = self.pca.fit_transform(self.sim_ensemble)
                return
            
            def _build_emulator(self):

                #standardize data TODO

                # autoconcder or PCA
                _apply_pca(self)
                #TODO

                return


            def forward_run():
                #TODO
                return
            
            def check_for_pdc():
                #TODO
                return
            


class FeatureTransform:
    """
    Class for feature transforms.
    """
    def __init__(self, pst=None, sim_ensemble=None):
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