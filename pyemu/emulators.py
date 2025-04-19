from __future__ import print_function, division
import os
import copy
import shutil
from datetime import datetime
import warnings
from .pyemu_warnings import PyemuWarning
import numpy as np
import pandas as pd
from pyemu.en import ObservationEnsemble,ParameterEnsemble
from pyemu.mat.mat_handler import Matrix, Jco, Cov
from pyemu.pst.pst_handler import Pst
from .logger import Logger
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
#from scipy.stats import norm
import inspect
from pyemu.utils.helpers import dsi_forward_run, series_to_insfile
import pickle
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import layers, models
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

class Emulator:
    """
    Base class for emulators.
    
    This class defines the common interface for all emulator implementations
    and provides shared functionality used by multiple emulator types.
    
    Parameters
    ----------
    verbose : bool, optional
        If True, enable verbose logging. Default is True.
    """

    def __init__(self, verbose=True):
        """
        Initialize the Emulator base class.

        Parameters
        ----------
        verbose : bool, optional
            If True, enable verbose logging. Default is True.
        """
        self.logger = Logger(verbose)
        self.log = self.logger.log
        self.fitted = False
        self.data = None
        self.data_transformed = None
        self.feature_scaler = None
        self.energy_threshold = 1.0
        self.feature_transformer = None

    def fit(self, X, y=None):
        """
        Fit the emulator to training data.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Input features for training.
        y : pandas.DataFrame or None, optional
            Target values for training if separate from X.
            
        Returns
        -------
        self : Emulator
            Returns self for method chaining.
        """
        raise NotImplementedError("Subclasses must implement fit method")

    def predict(self, X):
        """
        Generate predictions using the fitted emulator.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Input data to generate predictions for.
            
        Returns
        -------
        pandas.DataFrame or pandas.Series
            Predictions for the input data.
        """
        if not self.fitted:
            raise ValueError("Emulator must be fitted before prediction")
        raise NotImplementedError("Subclasses must implement predict method")

    def prepare_training_data(self, data=None):
        """
        Prepare and transform training data for model fitting.
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            Raw training data. If None, uses self.data.
            
        Returns
        -------
        tuple
            Processed data ready for model fitting.
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data provided and no data stored in the emulator")
            data = self.data
        
        # Common preprocessing logic could go here
        return data
        
    def apply_feature_transforms(self, data=None, transforms=None):
        """
        Apply feature transformations to data with customizable transformer sequence.
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            Data to transform. If None, uses self.data.
        transforms : list of dict, optional
            List of transformation specifications. Each dict should have:
            - 'type': str - Type of transformation (e.g., 'log10', 'normal_score')
            - 'columns': list - Columns to apply the transformation to (optional)
            - Additional kwargs specific to the transformer
            If None, no transformations are applied.
            
        Returns
        -------
        pandas.DataFrame
            Transformed data.
        
        Examples
        --------
        # Using the transforms parameter:
        emulator.apply_feature_transforms(
            transforms=[
                {'type': 'log10', 'columns': ['flow', 'heads']},
                {'type': 'normal_score', 'columns': None, 'quadratic_extrapolation': True}
            ]
        )
        """
        if data is None:
            data = self.data
            
        if data is None:
            raise ValueError("No data provided and no data stored in the emulator")
                
        self.logger.statement("applying feature transforms")
        ft = AutobotsAssemble(data.copy())
        
        # Process the transforms parameter if provided
        if transforms:
            for transform in transforms:
                transform_type = transform.get('type')
                columns = transform.get('columns')
                # Extract transformer-specific kwargs
                kwargs = {k: v for k, v in transform.items() 
                        if k not in ('type', 'columns')}
                
                self.logger.statement(f"applying {transform_type} transform")
                ft.apply(transform_type, columns=columns, **kwargs)
        
        transformed_data = ft.df.copy()
        self.feature_transformer = ft
        self.data_transformed = transformed_data
            
        return transformed_data

    def save(self, filename):
        """
        Save the fitted emulator to a file.
        
        Parameters
        ----------
        filename : str
            Path to save the emulator.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filename):
        """
        Load a fitted emulator from a file.
        
        Parameters
        ----------
        filename : str
            Path to the saved emulator file.
            
        Returns
        -------
        Emulator
            The loaded emulator instance.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

class LDFA(Emulator):
    """
    Class for the Learning-based pattern-data-driven forecast approach from Kim et al (2025).
    
    This emulator uses neural networks to learn the relationships between inputs 
    and forecast outputs, with dimensionality reduction via PCA.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The training data with input and forecast columns.
    input_cols : list
        List of column names to use as inputs.
    groups : dict
        Dictionary mapping group names to lists of column names. Used for row-wise min-max scaling.
    fit_groups : dict
        Dictionary mapping group names to lists of column names used to fit the scaling.
    forecast_names : list, optional
        List of column names to forecast. If None, all columns in data will be used.
    energy_threshold : float, optional
        Energy threshold for the PCA. Default is 1.0.
    seed : int, optional
        Random seed for reproducibility. Default is None.
    early_stop : bool, optional
        Whether to use early stopping during training. Default is True.
    apply_std_scaler : bool, optional
        Whether to apply standard scaling before min-max scaling. Default is False.
    verbose : bool, optional
        If True, enable verbose logging. Default is True.
    """

    def __init__(self,
                 data,
                 input_cols,
                 groups,
                 fit_groups,
                 forecast_names=None,
                 energy_threshold=1.0,
                 seed=None,
                 early_stop=True,
                 apply_std_scaler=False,
                 verbose=True):
        """
        Initialize the Learning-based pattern-data-driven NN emulator.

        Parameters
        ----------
        data : pandas.DataFrame
            The training data with input and forecast columns.
        input_cols : list
            List of column names to use as inputs.
        groups : dict
            Dictionary mapping group names to lists of column names. Used for row-wise min-max scaling.
        fit_groups : dict
            Dictionary mapping group names to lists of column names used to fit the scaling.
        forecast_names : list, optional
            List of column names to forecast. If None, all columns in data will be used.
        energy_threshold : float, optional
            Energy threshold for the PCA. Default is 1.0.
        seed : int, optional
            Random seed for reproducibility. Default is None.
        early_stop : bool, optional
            Whether to use early stopping during training. Default is True.
        apply_std_scaler : bool, optional
            Whether to apply standard scaling before min-max scaling. Default is False.
        verbose : bool, optional
            If True, enable verbose logging. Default is True.
        """
        super().__init__(verbose=verbose)

        self.seed = seed
        self.data = data
        self.input_cols = input_cols
        self.groups = groups
        self.fit_groups = fit_groups
        
        if forecast_names is None:
            forecast_names = data.columns
        self.forecast_names = forecast_names
        
        self.energy_threshold = energy_threshold
        
        # Configure early stopping
        self.early_stop = None
        if early_stop:
            self.early_stop = EarlyStopping(
                monitor='val_loss', 
                patience=20, 
                restore_best_weights=True
            )
            
        self.apply_std_scaler = apply_std_scaler
        self.noise_model = None
        self.model = None
        self.train_data = None
        self.test_data = None
        
    def prepare_training_data(self, data=None, test_size=0.2):
        """
        Prepare the training data for model fitting.
        
        This method:
        1. Splits the data into training and test sets
        2. Applies standard scaling if requested
        3. Applies row-wise min-max scaling
        4. Performs PCA dimensionality reduction
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            Data to prepare. If None, uses self.data. Default is None.
        test_size : float, optional
            Fraction of data to use for testing. Default is 0.2.
            
        Returns
        -------
        dict
            Dictionary containing prepared data components:
            - X_train: Input training data after transformation and PCA
            - y_train: Target training data after transformation and PCA
            - X_test: Input testing data after transformation and PCA
            - y_test: Target testing data after transformation and PCA
        """
        if data is None:
            data = self.data
            
        if data is None:
            raise ValueError("No data provided and no data stored in the emulator")
            
        # Split the data into training and test sets
        train, test = train_test_split(
            data, 
            test_size=test_size, 
            random_state=self.seed
        )
        
        self.logger.statement("preparing training data: data split complete")
        
        # Store for later use
        self.train_data = train.copy()
        self.test_data = test.copy()
        
        transforms = []
        
        # Apply standard scaling if requested
        if self.apply_std_scaler:
            self.logger.statement("applying standard scaling")
            transforms.append({
                'type': 'standard_scaler',
                'columns': None  # Apply to all columns
            })
            
        # Apply row-wise min-max scaling
        self.logger.statement("applying row-wise min-max scaling")
        # Store the row-wise min-max scaler for use in prediction
        self.rowwise_mm_scaler = RowWiseMinMaxScaler(
            feature_range=(-1, 1),
            groups=self.groups,
            fit_groups=self.fit_groups
        )
        
        # Apply transformations using the base class method
        if transforms:
            train_transformed = self.apply_feature_transforms(train, transforms)
            test_transformed = self.feature_transformer.transform(test)
        else:
            train_transformed = train
            test_transformed = test
        
        # Apply row-wise min-max scaling directly (not through the pipeline)
        self.rowwise_mm_scaler.fit(train_transformed)
        train_scaled = self.rowwise_mm_scaler.transform(train_transformed)
        test_scaled = self.rowwise_mm_scaler.transform(test_transformed)
        
        self.logger.statement("row-wise min-max scaling complete")
        
        # Split datasets into input (X) and target (y) variables
        X_train = train_scaled.loc[:, self.input_cols].copy()
        y_train = train_scaled.loc[:, self.forecast_names].copy()
        
        X_test = test_scaled.loc[:, self.input_cols].copy()
        y_test = test_scaled.loc[:, self.forecast_names].copy()
        
        # Apply PCA to reduce the dimensionality of the data
        self.logger.statement("applying PCA dimensionality reduction")
        self.pcaX = PCA(n_components=self.energy_threshold)
        self.pcay = PCA(n_components=self.energy_threshold)
        
        self.X = self.pcaX.fit_transform(X_train)
        self.y = self.pcay.fit_transform(y_train)
        
        self.X_test = self.pcaX.transform(X_test)
        self.y_test = self.pcay.transform(y_test)
        
        self.logger.statement("PCA dimensionality reduction complete")
        
        return {
            'X_train': self.X,
            'y_train': self.y,
            'X_test': self.X_test,
            'y_test': self.y_test
        }
        
    def _build_model(self, params=None, prob=False):
        """
        Build a neural network model with the specified parameters.
        
        Parameters
        ----------
        params : dict or pandas.Series, optional
            Dictionary with model parameters including:
            - activation: Activation function to use
            - hidden_units: List of units in each hidden layer
            - dropout_rate: Rate of dropout for regularization
            - learning_rate: Learning rate for optimizer
            If None, uses default parameters. Default is None.
        prob : bool, optional
            Whether to build a probabilistic model. Default is False.
            
        Returns
        -------
        tensorflow.keras.Model
            The compiled Keras model.
        """
        if params is None:
            params = {
                'activation': 'relu', 
                'hidden_units': None, 
                'dropout_rate': 0.0,
                'learning_rate': 0.01
            }
        
        if isinstance(params, pd.Series):
            params = params.to_dict()

        activation = params['activation']
        hidden_units = params['hidden_units']
        dropout_rate = params['dropout_rate']

        input_dim = self.X.shape[1]
        output_dim = self.y.shape[1]
        
        # Set seed for reproducibility
        if self.seed is not None:
            tf.keras.utils.set_random_seed(self.seed)
            
        # Create the model architecture
        inputs = tf.keras.Input(shape=(input_dim,))
        x = inputs
        
        # Add hidden layers
        if hidden_units is None:
            hidden_units = [2 * input_dim]
            
        for units in hidden_units:
            x = layers.Dense(units, activation=activation)(x)
            x = layers.Dropout(rate=dropout_rate)(x)

        # Output layer
        outputs = layers.Dense(output_dim)(x)
        
        # Define loss function
        loss_fn = 'mean_squared_error'
        
        # Create and compile the model
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss=loss_fn,
            metrics=['mae', 'accuracy']
        )
        
        return model

    def create_model(self, params=None):
        """
        Create and store the main model.
        
        Parameters
        ----------
        params : dict, optional
            Dictionary of model parameters. Default is None.
            
        Returns
        -------
        self : LDFA
            The emulator instance with model created.
        """
        self.model = self._build_model(params)
        return self

    def add_noise_model(self, params=None):
        """
        Add a noise model to capture residuals.
        
        Parameters
        ----------
        params : dict, optional
            Dictionary of model parameters for the noise model. Default is None.
            
        Returns
        -------
        self : LDFA
            The emulator instance with noise model added.
        """
        self.noise_model = self._build_model(params)

        # Get residuals from main model predictions
        pred = self.model.predict(self.X)
        res = self.y - pred

        # Get test residuals
        pred_test = self.model.predict(self.X_test)
        res_test = self.y_test - pred_test

        # Train the noise model on residuals
        self.logger.statement("training noise model on residuals")
        history = self.noise_model.fit(
            self.X, res, 
            epochs=200, 
            batch_size=32,
            validation_data=(self.X_test, res_test),
            callbacks=[self.early_stop] if self.early_stop else None,
            verbose=1
        )
        
        self.noise_history = history
        return self

    def fit(self, epochs=200, batch_size=32, X=None, y=None, prepare_data=True):
        """
        Fit the model to the training data.
        
        Parameters
        ----------
        epochs : int, optional
            Number of training epochs. Default is 200.
        batch_size : int, optional
            Batch size for training. Default is 32.
        X : pandas.DataFrame, optional
            Input data for training. If None and prepare_data is True,
            will run prepare_training_data(). Default is None.
        y : pandas.DataFrame, optional
            Not used directly but included for API consistency. Default is None.
        prepare_data : bool, optional
            Whether to prepare training data if not already done. Default is True.
            
        Returns
        -------
        self : LDFA
            The fitted emulator.
        """
        if prepare_data and (X is None or self.X is None):
            self.prepare_training_data()
            
        if self.model is None:
            self.create_model()
            
        X_train = self.X if X is None else X
        y_train = self.y
        X_test = self.X_test
        y_test = self.y_test

        self.logger.statement(f"fitting model: {epochs} epochs, batch size {batch_size}")
        history = self.model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[self.early_stop] if self.early_stop else None,
            verbose=1
        )
        
        self.history = history
        self.fitted = True
        return self

    def predict(self, data):
        """
        Generate predictions for new data.
        
        Parameters
        ----------
        data : pandas.DataFrame
            New data to generate predictions for.
            
        Returns
        -------
        pandas.DataFrame
            Predictions for the input data.
        """
        if not self.fitted:
            raise ValueError("Emulator must be fitted before prediction")
        
        if self.model is None:
            raise ValueError("No model has been created. Call create_model() first")
            
        self.logger.statement("generating predictions from fitted model")
            
        # Make a copy of the input data to avoid modifying the original
        truth = data.copy()
        predictions = truth.copy()
        predictions[:] = np.nan
        
        # STEP 1: Apply the same sequence of transformations used during training
        self.logger.statement("applying transformations to input data")
        
        # Apply standard scaling if it was used during training
        if self.apply_std_scaler and self.feature_transformer:
            # Use the transformer pipeline for consistent transformation
            truth_transformed = self.feature_transformer.transform(truth)
        else:
            truth_transformed = truth.copy()
        
        # Apply row-wise min-max scaling
        # We need to fit a new scaler on the truth data
        forecast_rowwise_mm_scaler = RowWiseMinMaxScaler(
            feature_range=(-1, 1),
            groups=self.groups,
            fit_groups=self.fit_groups
        )
        forecast_rowwise_mm_scaler.fit(truth_transformed)
        truth_scaled = forecast_rowwise_mm_scaler.transform(truth_transformed)
        
        # Extract input columns and apply PCA transformation
        X_truth = truth_scaled.loc[:, self.input_cols].copy()
        y_truth = truth_scaled.loc[:, self.forecast_names].copy()
        
        # Apply PCA transform
        truth_pca = self.pcaX.transform(X_truth.values.reshape(1, -1))
        
        # STEP 2: Run model prediction
        self.logger.statement("running model prediction")
        pred_pca = self.model.predict(truth_pca)
        
        # Add noise prediction if available
        if self.noise_model is not None:
            self.logger.statement("adding noise model prediction")
            noise = self.noise_model.predict(truth_pca)
            pred_pca = pred_pca + noise
        
        # STEP 3: Apply inverse transformations in REVERSE order of the original transformations
        self.logger.statement("performing inverse transformations")
        
        # 1. First inverse the PCA transform (was the last transform applied)
        pred_scaled = pd.DataFrame(
            self.pcay.inverse_transform(pred_pca),
            columns=y_truth.columns, 
            index=y_truth.index
        )
        
        # 2. Then inverse the row-wise min-max scaling (applied before PCA)
        pred_transformed = forecast_rowwise_mm_scaler.inverse_transform(pred_scaled)
        
        # Assign predictions to output
        predictions.loc[:, self.forecast_names] = pred_transformed.loc[:, self.forecast_names]
        
        # 3. Finally, inverse the standard scaling if it was applied (was the first transform)
        if self.apply_std_scaler and self.feature_transformer:
            predictions = self.feature_transformer.inverse_transform(predictions)
        
        return predictions

class DSI(Emulator):
    """
    Data Space Inversion emulator class.
    
    This class implements the Data Space Inversion emulation method, which uses
    SVD-based techniques for generating emulation outputs.
    
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
    nst_extrapolate : bool, optional
        If True, the normal score transform will use quadratic extrapolation.
    energy_threshold : float, optional 
        The energy threshold for the SVD. Default is 1.0, no truncation.
    log_transform : bool or list, optional
        If True, the emulator will apply a log transformation to all the simulated
        observations. If list of observation column names, will be applied to these. Default is False.
    verbose : bool, optional
        If True, enable verbose logging. Default is False.
    """

    def __init__(self, 
                pst=None,
                sim_ensemble=None,
                normal_score_transform=False,
                nst_extrapolate=False,
                energy_threshold=1.0,
                log_transform=False,
                verbose=False):
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
        verbose : bool, optional
            If True, enable verbose logging. Default is False.
        """

        super().__init__(verbose=verbose)

        self.__org_observation_data = pst.observation_data.copy() if pst is not None else None
        self.__org_parameter_data = pst.parameter_data.copy() if pst is not None else None
        #self.__org_control_data = pst.control_data.copy() #breaks pickling
        if isinstance(sim_ensemble, ObservationEnsemble):
            sim_ensemble = sim_ensemble._df.copy()
        self.__org_sim_ensemble = sim_ensemble.copy() if sim_ensemble is not None else None
        self.data = sim_ensemble.copy() if sim_ensemble is not None else None
        self.data_transformed = None
        self.feature_scaler = None
        self.energy_threshold = energy_threshold
        if log_transform is True and sim_ensemble is not None:
            self.log_transform = sim_ensemble.columns.tolist()
        else:
            self.log_transform = log_transform
        assert isinstance(self.log_transform, (bool, list)), "log_transform must be a boolean or a list of column names"
        self.normal_score_transform = normal_score_transform
        self.nst_extrapolate = nst_extrapolate
        self.fitted = False
        
    def prepare_training_data(self, data=None):
        """
        Prepare training data by applying transformations and computing the projection matrix.
        
        This method follows these steps:
        1. Apply feature transformations (log transform, normal score transform)
        2. Compute projection matrix using SVD
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            Data to prepare. If None, uses self.data.
            
        Returns
        -------
        pandas.DataFrame
            The prepared data.
        """
        if data is None:
            data = self.data
            
        if data is None:
            raise ValueError("No data provided and no data stored in the emulator")
            
        # Apply transformations
        transforms = []
        
        # Add log transform if configured
        if self.log_transform:
            log_columns = data.columns.tolist() if self.log_transform is True else self.log_transform
            transforms.append({
                'type': 'log10',
                'columns': log_columns
            })
            
        # Add normal score transform if configured
        if self.normal_score_transform:
            transforms.append({
                'type': 'normal_score',
                'columns': None,  # Apply to all columns
                'quadratic_extrapolation': self.nst_extrapolate
            })
            
        # Apply transformations using the base class method
        if transforms:
            self.data_transformed = self.apply_feature_transforms(data, transforms)
        else:
            self.data_transformed = data.copy()
            
        return self.data_transformed
        
    def compute_projection_matrix(self, energy_threshold=None):
        """
        Compute the projection matrix using SVD.
        
        Parameters
        ----------
        energy_threshold : float, optional
            Energy threshold for truncation. Default is None, which uses the threshold from initialization.
            
        Returns
        -------
        None
        """
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
    
    def fit(self, X=None, y=None):
        """
        Fit the emulator to training data.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Input data to fit the emulator on.
        y : None
            Not used, present for API consistency.
            
        Returns
        -------
        self : DSI
            The fitted emulator.
        """
        if X is not None:
            self.data = X
        self.prepare_training_data()
        # Compute projection matrix
        self.compute_projection_matrix()
        self.fitted = True
        return self
    
    def predict(self, pvals):
        """
        Generate predictions from the emulator.
        
        Parameters
        ----------
        pvals : numpy.ndarray or pandas.Series
            Parameter values for prediction.
            
        Returns
        -------
        pandas.Series
            Predicted observation values.
        """
        if not self.fitted:
            raise ValueError("Emulator must be fitted before prediction")
            
        if isinstance(pvals, pd.Series):
            pvals = pvals.values.flatten()
        assert pvals.shape[0] == self.s.shape[0], "pvals must be the same length as the number of singular values"
        assert pvals.shape[0] == self.pmat.shape[1], "pvals must be the same length as the number of singular values"
        pmat = self.pmat
        ovals = self.ovals
        sim_vals = ovals + np.dot(pmat,pvals)
        ft = self.feature_transformer
        sim_vals = ft.inverse(sim_vals)
        sim_vals.index.name = 'obsnme'
        sim_vals.name = "obsval"
        self.sim_vals = sim_vals
        return sim_vals
    
    def check_for_pdc(self):
        """Check for Parameter Data Components."""
        #TODO
        return
        
    def prepare_pestpp(self, t_d=None, observation_data=None):
        """
        Prepare PEST++ control files for the emulator.
        
        Parameters
        ----------
        t_d : str, optional
            Template directory path. Must be provided.
        observation_data : pandas.DataFrame, optional
            Observation data to use. If None, uses the data from initialization.
            
        Returns
        -------
        Pst
            PEST++ control file object.
        """
        
        assert t_d is not None, "template directory must be provided"
        self.template_dir = t_d

        if os.path.exists(t_d):
            shutil.rmtree(t_d)
        os.makedirs(t_d)
        self.logger.statement("creating template directory {0}".format(t_d))

        self.logger.log("creating tpl files")
        dsi_in_file = os.path.join(t_d, "dsi_pars.csv")
        dsi_tpl_file = dsi_in_file + ".tpl"
        ftpl = open(dsi_tpl_file, 'w')
        fin = open(dsi_in_file, 'w')
        ftpl.write("ptf ~\n")
        fin.write("parnme,parval1\n")
        ftpl.write("parnme,parval1\n")
        npar = self.s.shape[0]
        assert npar>0, "no parameters found in the DSI emulator"
        dsi_pnames = []
        for i in range(npar):
            pname = "dsi_par{0:04d}".format(i)
            dsi_pnames.append(pname)
            fin.write("{0},0.0\n".format(pname))
            ftpl.write("{0},~   {0}   ~\n".format(pname, pname))
        fin.close()
        ftpl.close()
        self.logger.log("creating tpl files")

        # run once to get the dsi_pars.csv file
        pvals = np.zeros_like(self.s)
        sim_vals = self.predict(pvals)
        
        self.logger.log("creating ins file")
        out_file = os.path.join(t_d,"dsi_sim_vals.csv")
        sim_vals.to_csv(out_file,index=True)
              
        ins_file = out_file + ".ins"
        sdf = pd.read_csv(out_file,index_col=0)
        with open(ins_file,'w') as f:
            f.write("pif ~\n")
            f.write("l1\n")
            for oname in sdf.index.values:
                f.write("l1 ~,~ !{0}!\n".format(oname))
        self.logger.log("creating ins file")

        self.logger.log("creating Pst")
        pst = Pst.from_io_files([dsi_tpl_file],[dsi_in_file],[ins_file],[out_file],pst_path=".")

        par = pst.parameter_data
        dsi_pars = par.loc[par.parnme.str.startswith("dsi_par"),"parnme"]
        par.loc[dsi_pars,"parval1"] = 0
        par.loc[dsi_pars,"parubnd"] = 10.0
        par.loc[dsi_pars,"parlbnd"] = -10.0
        par.loc[dsi_pars,"partrans"] = "none"
        with open(os.path.join(t_d,"dsi.unc"),'w') as f:
            f.write("START STANDARD_DEVIATION\n")
            for p in dsi_pars:
                f.write("{0} 1.0\n".format(p))
            f.write("END STANDARD_DEVIATION")
        pst.pestpp_options['parcov'] = "dsi.unc"

        obs = pst.observation_data

        if observation_data is None:
            observation_data = self.__org_observation_data
        assert isinstance(observation_data, pd.DataFrame), "observation_data must be a pandas DataFrame"
        for col in observation_data.columns:
            obs.loc[sim_vals.index,col] = observation_data.loc[:,col]

        # check if any observations are missing
        missing_obs = list(set(obs.index) - set(observation_data.index))
        assert len(missing_obs) == 0, "missing observations: {0}".format(missing_obs)

        pst.control_data.noptmax = 0
        pst.model_command = "python forward_run.py"
        self.logger.log("creating Pst")


        function_source = inspect.getsource(dsi_forward_run)
        with open(os.path.join(t_d,"forward_run.py"),'w') as file:
            file.write(function_source)
            file.write("\n\n")
            file.write("if __name__ == \"__main__\":\n")
            file.write(f"    {function_source.split('(')[0].split('def ')[1]}()\n")
        self.logger.log("creating Pst")

        pst.pestpp_options["save_binary"] = True
        pst.pestpp_options["overdue_giveup_fac"] = 1e30
        pst.pestpp_options["overdue_giveup_minutes"] = 1e30
        pst.pestpp_options["panther_agent_freeze_on_fail"] = True
        pst.pestpp_options["ies_no_noise"] = False
        pst.pestpp_options["ies_subset_size"] = -10 # the more the merrier
        #pst.pestpp_options["ies_bad_phi_sigma"] = 2.0
        #pst.pestpp_options["save_binary"] = True

        pst.write(os.path.join(t_d,"dsi.pst"),version=2)
        self.logger.statement("saved pst to {0}".format(os.path.join(t_d,"dsi.pst")))
        
        #self.pst_dsi = pst #breaks pickling #TODO: add save/load methods to Emulator class
        with open(os.path.join(t_d,"dsi.pickle"),"wb") as f:
            pickle.dump(self,f)
        return pst
        
    def prepare_dsivc(self, decvar_names, t_d=None, pst=None, oe=None, track_stack=False, dsi_args=None, percentiles=[0.25,0.75,0.5], mou_population_size=None):
        """
        Prepare Data Space Inversion Variability Calculation (DSIVC) control files.
        
        Parameters
        ----------
        decvar_names : list or str
            Names of decision variables.
        t_d : str, optional
            Template directory path.
        pst : Pst, optional
            PST control file object.
        oe : ObservationEnsemble, optional
            Observation ensemble.
        track_stack : bool, optional
            Whether to track the stack. Default is False.
        dsi_args : dict, optional
            Arguments for DSI.
        percentiles : list, optional
            Percentiles to calculate. Default is [0.25, 0.75, 0.5].
        mou_population_size : int, optional
            Population size for multi-objective optimization.
            
        Returns
        -------
        Pst
            PEST++ control file object for DSIVC.
        """
        # check that percentiles is a list or array of floats between 0 and 1.
        assert isinstance(percentiles, (list, np.ndarray)), "percentiles must be a list or array of floats"
        assert all([isinstance(i, (float, int)) for i in percentiles]), "percentiles must be a list or array of floats"
        assert all([0 <= i <= 1 for i in percentiles]), "percentiles must be between 0 and 1"
        # ensure that pecentiles are unique
        percentiles = np.unique(percentiles)


        #track dsivc args for forward run
        self.dsivc_args = {"percentiles":percentiles,
                        "decvar_names":decvar_names,
                            "track_stack":track_stack,
                        }

        if t_d is None:
            self.logger.statement("using existing DSI template dir...")
            t_d = self.template_dir
        self.logger.statement(f"using {t_d} as template directory...")
        assert os.path.exists(t_d), f"template directory {t_d} does not exist"

        if pst is None:
            self.logger.statement("no pst provided...")
            self.logger.statement("using dsi.pst in DSI template dir...")
            assert os.path.exists(os.path.join(t_d,"dsi.pst")), f"dsi.pst not found in {t_d}"
            pst = Pst(os.path.join(t_d,"dsi.pst"))
        if oe is None:
            self.logger.statement("no posterior DSI observation ensemble provided, using dsi.3.obs.jcb in DSI template dir...")
            self.logger.statement(f"using dsi.{dsi_args['noptmax']}.obs.jcb in DSI template dir...")
            assert os.path.exists(os.path.join(t_d,f"dsi.{dsi_args['noptmax']}.obs.jcb")), f"dsi.{dsi_args['noptmax']}.obs.jcb not found in {t_d}"
            oe = ObservationEnsemble.from_binary(pst,os.path.join(t_d,f"dsi.{dsi_args['noptmax']}.obs.jcb"))
        else:
            assert isinstance(oe, ObservationEnsemble), "oe must be an ObservationEnsemble"

        #check if decvar_names str
        if isinstance(decvar_names, str):
            decvar_names = [decvar_names]
        # chekc htat decvars are in the oe columns
        missing = [col for col in decvar_names if col not in oe.columns]
        assert len(missing) == 0, f"The following decvars are missing from the DSI obs ensemble: {missing}"
        # chekc htat decvars are in the pst observation data
        missing = [col for col in decvar_names if col not in pst.obs_names]
        assert len(missing) == 0, f"The following decvars are missing from the DSI pst control file: {missing}"


        # handle DSI args
        default_dsi_args =  {"noptmax":pst.control_data.noptmax,
                            "decvar_weight":1.0,
                            #"decvar_phi_factor":0.5,
                            "num_pyworkers":1,
                            }
        # ensure it's a dict
        if dsi_args is None:
            dsi_args = {}
        elif not isinstance(dsi_args, dict):
            raise TypeError("Expected a dictionary for 'options'")
        # merge with defaults (user values override defaults)
        dsi_args = {**default_dsi_args, **dsi_args}


        out_files = []

        self.logger.statement(f"preparing stack stats observations...")
        assert isinstance(oe, ObservationEnsemble), "oe must be an ObservationEnsemble"
        if oe.index.name is None:
            id_vars="index"
        else:
            id_vars=oe.index.name
        stack_stats = oe._df.describe(percentiles=percentiles).reset_index().melt(id_vars=id_vars)
        stack_stats.rename(columns={"value":"obsval","index":"stat"},inplace=True)
        stack_stats['obsnme'] = stack_stats.apply(lambda x: x.variable+"_stat:"+x.stat,axis=1)
        stack_stats.set_index("obsnme",inplace=True)
        stack_stats = stack_stats.obsval
        self.logger.statement(f"stack osb recorded to dsi.stack_stats.csv...")
        out_file = os.path.join(t_d,"dsi.stack_stats.csv")
        out_files.append(out_file)
        stack_stats.to_csv(out_file,float_format="%.6e")
        series_to_insfile(out_file,ins_file=None)


        if track_stack:
            self.logger.statement(f"including {oe.values.flatten().shape[0]} stack observations...")

            stack = oe._df.reset_index().melt(id_vars=id_vars)
            stack.rename(columns={"value":"obsval"},inplace=True)
            stack['obsnme'] = stack.apply(lambda x: x.variable+"_real:"+x.index,axis=1)
            stack.set_index("obsnme",inplace=True)
            stack = stack.obsval
            out_file = os.path.join(t_d,"dsi.stack.csv")
            out_files.append(out_file)
            stack.to_csv(out_file,float_format="%.6e")
            series_to_insfile(out_file,ins_file=None)



        self.logger.statement(f"prepare DSIVC template files...")
        dsi_in_file = os.path.join(t_d, "dsivc_pars.csv")
        dsi_tpl_file = dsi_in_file + ".tpl"
        ftpl = open(dsi_tpl_file, 'w')
        fin = open(dsi_in_file, 'w')
        ftpl.write("ptf ~\n")
        fin.write("parnme,parval1\n")
        ftpl.write("parnme,parval1\n")
        for pname in decvar_names:
            val = oe._df.loc[:,pname].mean()
            fin.write(f"{pname},{val:.6e}\n")
            ftpl.write(f"{pname},~   {pname}   ~\n")
        fin.close()
        ftpl.close()

        
        self.logger.statement(f"building DSIVC control file...")
        pst_dsivc = Pst.from_io_files([dsi_tpl_file],[dsi_in_file],[i+".ins" for i in out_files],out_files,pst_path=".")

        self.logger.statement(f"setting dec var bounds...")
        par = pst_dsivc.parameter_data
        # set all parameters fixed
        par.loc[:,"partrans"] = "fixed"
        # constrain decvar pars to training data bounds
        par.loc[decvar_names,"pargp"] = "decvars"
        par.loc[decvar_names,"partrans"] = "none"
        par.loc[decvar_names,"parubnd"] = self.data.loc[:,decvar_names].max()
        par.loc[decvar_names,"parlbnd"] = self.data.loc[:,decvar_names].min()
        
        self.logger.statement(f"zero-weighting observation data...")
        # prepemtpively set obs weights 0.0
        obs = pst_dsivc.observation_data
        obs.loc[:,"weight"] = 0.0

        self.logger.statement(f"getting obs metadata from DSI observation_data...")
        obsorg = pst.observation_data.copy()
        columns = [i for i in obsorg.columns if i !='obsnme']
        for o in obsorg.obsnme.values:
            obs.loc[obs.obsnme.str.startswith(o), columns] = obsorg.loc[obsorg.obsnme==o, columns].values

        obs.loc[stack_stats.index,"obgnme"] = "stack_stats"
        #obs.loc[stack.index,"obgnme"] = "stack"

        self.logger.statement(f"building dsivc_forward_run.py...")
        pst_dsivc.model_command = "python dsivc_forward_run.py"
        from pyemu.utils.helpers import dsivc_forward_run
        function_source = inspect.getsource(dsivc_forward_run)
        with open(os.path.join(t_d,"dsivc_forward_run.py"),'w') as file:
            file.write(function_source)
            file.write("\n\n")
            file.write("if __name__ == \"__main__\":\n")
            file.write(f"    {function_source.split('(')[0].split('def ')[1]}()\n")

        self.logger.statement(f"preparing nominal initial population...")
        if mou_population_size is None:
            # set the population size to 2 * number of decision variables
            # this is a good rule of thumb for MOU
            mou_population_size = 2 * len(decvar_names)
        # these should generally be twice the number of decision variables
        if mou_population_size < 2 * len(decvar_names):
            self.logger.statement(f"mou population is less than 2x number of decision variables, this may be too small...")
        # sample 160 sets of decision variables from a unform distribution
        dvpop = ParameterEnsemble.from_uniform_draw(pst_dsivc,num_reals=mou_population_size)
        # record to external file for PESTPP-MOU
        dvpop.to_binary(os.path.join(t_d,"initial_dvpop.jcb"))
        # tell PESTPP-MOU about the new file
        pst_dsivc.pestpp_options["mou_dv_population_file"] = 'initial_dvpop.jcb'


        # some additional PESTPP-MOU options:
        pst_dsivc.pestpp_options["mou_population_size"] = mou_population_size #twice the number of decision variables
        pst_dsivc.pestpp_options["mou_save_population_every"] = 1 # save lots of files! 
        
        pst_dsivc.control_data.noptmax = 0 #just for a test run
        pst_dsivc.write(os.path.join(t_d,"dsivc.pst"),version=2)  

        # updating the DSI pst control file
        self.logger.statement(f"updating DSI pst control file...")
        self.logger.statement("overwriting dsi.pst file...")
        pst.observation_data.loc[decvar_names, "weight"] = dsi_args["decvar_weight"]
        pst.control_data.noptmax = dsi_args["noptmax"]
        pst.write(os.path.join(t_d,"dsi.pst"), version=2)
        
        
        self.logger.statement("overwriting dsi.pickle file...")
        # re-pickle dsi to track dsivc args
        with open(os.path.join(t_d,"dsi.pickle"),"wb") as f:
            pickle.dump(self,f)

        self.logger.statement("DSIVC control files created...the user still needs to specify objectives...")
        return pst_dsivc


class BaseTransformer:
    """Base class for all transformers providing a consistent interface."""

    def fit(self, X):
        """Learn parameters from data if needed."""
        return self

    def transform(self, X):
        """Apply transformation to X."""
        raise NotImplementedError

    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """Inverse transform X back to original space."""
        raise NotImplementedError

class Log10Transformer(BaseTransformer):
    """Apply log10 transformation."""

    def __init__(self):
        self.shifts = {}

    def transform(self, X):
        result = X.copy()
        for col in X.columns:
            min_val = X[col].min()
            shift = -min_val + 1e-6 if min_val <= 0 else 0
            self.shifts[col] = shift
            result[col] = np.log10(X[col] + shift)
        return result

    def inverse_transform(self, X):
        result = X.copy()
        for col in X.columns:
            shift = self.shifts.get(col, 0)
            result[col] = (10 ** X[col]) - shift
        return result

class RowWiseMinMaxScaler(BaseTransformer):
    """Scale each row of a DataFrame to a specified range.
    
    Parameters
    ----------
    feature_range : tuple (min, max), default=(-1, 1)
        The range to scale features into.
    groups : dict or None, default=None
        Dict mapping group names to lists of column names to be scaled together (entire timeseries for that group).
        If None, all columns will be treated as a single group.
    fit_groups : dict or None, default=None
        Dict mapping group names to lists of column names (subset of groups) used to compute row-wise min and max.
        If None, defaults to using the same columns as in groups.
    """

    def __init__(self, feature_range=(-1, 1), groups=None, fit_groups=None):
        self.feature_range = feature_range
        self.groups = groups
        self.fit_groups = fit_groups if fit_groups is not None else groups
        self.row_params = {}  # Will store per-row (min, max) for each group

    def fit(self, X):
        """Compute row-wise min and max for each group.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame to fit the scaler on.
            
        Returns
        -------
        self : object
            Returns self.
        """
        # If groups not specified, treat all columns as one group
        if self.groups is None:
            self.groups = {"all": X.columns.tolist()}
            
        if self.fit_groups is None:
            self.fit_groups = self.groups.copy()
        
        # Calculate and store row-wise min and max for each group
        self.row_params = {}
        for group_name, group_cols in self.groups.items():
            # Determine which columns to use for computing min/max for each row
            fit_cols = self.fit_groups.get(group_name, group_cols)
            # Keep only columns that exist in the DataFrame
            fit_cols = [col for col in fit_cols if col in X.columns]
            if not fit_cols:
                continue
                
            # Compute row-wise min and max using the fit columns
            row_min = X[fit_cols].min(axis=1)
            row_max = X[fit_cols].max(axis=1)
            self.row_params[group_name] = (row_min, row_max)
        
        return self

    def transform(self, X):
        """Scale each row of data to the specified range.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame to transform.
            
        Returns
        -------
        pandas.DataFrame
            The transformed DataFrame.
        """
        result = X.copy()
        f_min, f_max = self.feature_range
        
        # Auto-fit if not already fitted or if groups weren't specified
        if not self.row_params or self.groups is None:
            self.fit(X)
        
        # Transform each group
        for group_name, group_cols in self.groups.items():
            # Keep only columns that exist in the DataFrame
            valid_cols = [col for col in group_cols if col in X.columns]
            if not valid_cols:
                continue
                
            # Get the min and max for each row in this group
            row_min, row_max = self.row_params[group_name]
            
            # Calculate the row range, avoiding division by zero
            row_range = row_max - row_min
            row_range[row_range == 0] = 1.0  # Set to 1 where range is 0
            
            # For all columns in the group, scale using the row-wise parameters
            group_data = X[valid_cols]
            # First scale to [0, 1]
            group_std = group_data.sub(row_min, axis=0).div(row_range, axis=0)
            # Then scale to the desired feature range
            result[valid_cols] = group_std * (f_max - f_min) + f_min
        
        return result

    def inverse_transform(self, X):
        """Inverse transform data back to the original scale.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame to inverse transform.
            
        Returns
        -------
        pandas.DataFrame
            The inverse-transformed DataFrame.
        """
        if not self.row_params:
            raise ValueError("This RowWiseMinMaxScaler instance is not fitted yet. "
                            "Call 'fit' before using this method.")
        
        result = X.copy()
        f_min, f_max = self.feature_range
        
        # Inverse transform each group
        for group_name, group_cols in self.groups.items():
            # Keep only columns that exist in the DataFrame
            valid_cols = [col for col in group_cols if col in X.columns]
            if not valid_cols:
                continue
                
            # Get the min and max for each row in this group
            row_min, row_max = self.row_params[group_name]
            row_range = row_max - row_min
            row_range[row_range == 0] = 1.0  # Avoid division by zero
            
            # Get the scaled data for this group
            group_data = X[valid_cols]
            
            # First convert from feature_range to [0, 1]
            group_std = (group_data - f_min) / (f_max - f_min)
            
            # Then recover original values
            result[valid_cols] = group_std.mul(row_range, axis=0).add(row_min, axis=0)
        
        return result

class TransformerPipeline:
    """Apply a sequence of transformers in order."""

    def __init__(self):
        self.transformers = []
        self.fitted = False

    def add(self, transformer, columns=None):
        """Add a transformer to the pipeline, optionally for specific columns."""
        self.transformers.append((transformer, columns))
        return self

    def fit(self, X):
        """Fit all transformers in the pipeline."""
        for transformer, columns in self.transformers:
            cols_to_transform = columns if columns is not None else X.columns
            sub_X = X[cols_to_transform]
            transformer.fit(sub_X)
        self.fitted = True
        return self

    def transform(self, X):
        """Transform data using all transformers in the pipeline.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame to transform.
            
        Returns
        -------
        pandas.DataFrame
            The transformed DataFrame.
        """
        result = X.copy()
        for transformer, columns in self.transformers:
            cols_to_transform = columns if columns is not None else X.columns
            # Only use columns that exist in the input data
            valid_cols = [col for col in cols_to_transform if col in X.columns]
            if not valid_cols:
                continue
            sub_X = result[valid_cols]
            result[valid_cols] = transformer.transform(sub_X)
        return result

    def fit_transform(self, X):
        """Fit all transformers and transform data in one operation."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        """Apply inverse transformations in reverse order.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame to inverse transform.
            
        Returns
        -------
        pandas.DataFrame
            The inverse-transformed DataFrame.
        """
        
        if isinstance(X, pd.Series):
            result = X.copy().to_frame().T
        else:
            result = X.copy()
        # Need to reverse the order of transformers for inverse
        for transformer, columns in reversed(self.transformers):
            cols_to_transform = columns if columns is not None else result.columns
            # Only use columns that exist in the input data
            valid_cols = [col for col in cols_to_transform if col in result.columns]
            if not valid_cols:
                print("invalid cols")
                continue
            sub_X = result[valid_cols].copy()  # Create a copy to avoid reference issues
            inverted = transformer.inverse_transform(sub_X)
            result.loc[:, valid_cols] = inverted  # Use loc for proper assignment
        if isinstance(X, pd.Series):
            result = result.iloc[0]
        return result

class AutobotsAssemble:
    """Class for transforming features in a DataFrame using a pipeline approach."""

    def __init__(self, df=None):
        self.df = df.copy() if df is not None else None
        self.pipeline = TransformerPipeline()

    def apply(self, transform_type, columns=None, **kwargs):
        """Apply a transformation to specified columns."""
        transformer = self._create_transformer(transform_type, **kwargs)
        if columns is None:
            columns = list(self.df.columns)  # Convert to list to avoid pandas index issues
        
        # Fit transformer to data if needed
        if hasattr(transformer, 'fit') and callable(transformer.fit):
            if self.df is not None:
                df_subset = self.df[columns]
                transformer.fit(df_subset)
        
        # Add to pipeline
        self.pipeline.add(transformer, columns)
        
        # Apply transformation to current df if available
        if self.df is not None:
            # Use transform directly to ensure correct application
            df_subset = self.df[columns].copy()
            transformed = transformer.transform(df_subset)
            self.df[columns] = transformed
            
        return self

    def transform(self, df):
        """Transform an external DataFrame using the pipeline.
        
        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to transform.
            
        Returns
        -------
        pandas.DataFrame
            The transformed DataFrame.
        """
        if self.pipeline.transformers:
            return self.pipeline.transform(df)
        return df.copy()

    def inverse(self, df=None):
        """Apply inverse transformations in reverse order."""
        to_transform = df if df is not None else self.df
        result = self.pipeline.inverse_transform(to_transform)
        if df is None:
            self.df = result
        return result

    def inverse_on_external_df(self, df, columns=None):
        """Apply inverse transformations to an external DataFrame.
        
        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to inverse transform.
        columns : list, optional
            Specific columns to inverse transform. If None, all columns are processed.
            
        Returns
        -------
        pandas.DataFrame
            The inverse-transformed DataFrame.
        """
        to_transform = df.copy()
        if columns is not None:
            # Ensure we only process specified columns
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
            
        return self.pipeline.inverse_transform(to_transform)

    def _create_transformer(self, transform_type, **kwargs):
        """Factory method to create appropriate transformer."""
        if transform_type == "log10":
            return Log10Transformer()
        elif transform_type == "normal_score":
            return NormalScoreTransformer(**kwargs)
        elif transform_type == "row_wise_minmax":
            return RowWiseMinMaxScaler()
        elif transform_type == "standard_scaler":
            return StandardScalerTransformer()
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")

class NormalScoreTransformer(BaseTransformer):
    """A transformer for normal score transformation."""

    def __init__(self, tol=1e-7, max_samples=1000000, quadratic_extrapolation=False):
        self.tol = tol
        self.max_samples = max_samples
        self.quadratic_extrapolation = quadratic_extrapolation
        self.column_parameters = {}
        self.shared_z_scores = {}

    def fit(self, X):
        """Fit the transformer to the data."""
        for col in X.columns:
            values = X[col].values
            sorted_vals = np.sort(values)
            smoothed_vals = self._moving_average_with_endpoints(sorted_vals)

            n_points = len(smoothed_vals)
            if n_points not in self.shared_z_scores:
                self.shared_z_scores[n_points] = self._randrealgen_optimized(n_points)

            z_scores = self.shared_z_scores[n_points]
            
            self.column_parameters[col] = {
                'z_scores': z_scores,
                'originals': smoothed_vals,
            }
        return self
        
    def transform(self, X):
        """Transform the data using normal score transformation.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame to transform.
            
        Returns
        -------
        pandas.DataFrame
            The transformed DataFrame with normal scores.
        """
        result = X.copy()
        for col in X.columns:
            params = self.column_parameters.get(col, {})
            z_scores = params.get('z_scores', [])
            originals = params.get('originals', [])
            
            if len(z_scores) == 0 or len(originals) == 0:
                continue
                
            values = X[col].values
            
            # Handle values outside the original range
            min_orig, max_orig = np.min(originals), np.max(originals)
            min_z, max_z = np.min(z_scores), np.max(z_scores)
            
            # For values within range, use interpolation
            within_range = (values >= min_orig) & (values <= max_orig)
            if within_range.any():
                result.loc[within_range, col] = np.interp(
                    values[within_range], originals, z_scores
                )
                
            # For values outside range, use extrapolation if enabled or clamp to bounds
            below_min = values < min_orig
            above_max = values > max_orig
            
            if below_min.any():
                if self.quadratic_extrapolation:
                    # Use linear extrapolation below minimum
                    slope = (z_scores[1] - z_scores[0]) / (originals[1] - originals[0])
                    result.loc[below_min, col] = min_z + slope * (values[below_min] - min_orig)
                else:
                    # Otherwise clamp to minimum z-score
                    result.loc[below_min, col] = min_z
                    
            if above_max.any():
                if self.quadratic_extrapolation:
                    # Use linear extrapolation above maximum
                    slope = (z_scores[-1] - z_scores[-2]) / (originals[-1] - originals[-2])
                    result.loc[above_max, col] = max_z + slope * (values[above_max] - max_orig)
                else:
                    # Otherwise clamp to maximum z-score
                    result.loc[above_max, col] = max_z
            
        return result

    def inverse_transform(self, X):
        """Inverse transform data back to original space.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame with transformed data to inverse transform.
            
        Returns
        -------
        pandas.DataFrame
            The inverse-transformed DataFrame.
        """
        result = X.copy()
        for col in X.columns:
            params = self.column_parameters.get(col, {})
            z_scores = params.get('z_scores', [])
            originals = params.get('originals', [])
            if len(z_scores) == 0 or len(originals) == 0:
                continue

            # Get values to inverse transform
            values = X[col].values
            min_z, max_z = np.min(z_scores), np.max(z_scores)
            min_orig, max_orig = np.min(originals), np.max(originals)

            # For values within the z-score range, use interpolation
            within_range = (values >= min_z) & (values <= max_z)
            if within_range.any():
                result.loc[within_range, col] = np.interp(values[within_range], z_scores, originals)
            
            # For values outside the z-score range, use extrapolation if enabled
            below_min = values < min_z
            above_max = values > max_z
            
            if below_min.any():
                if self.quadratic_extrapolation:
                    # Use linear extrapolation below minimum z-score
                    slope = (originals[1] - originals[0]) / (z_scores[1] - z_scores[0])
                    intercept = originals[0] - slope * z_scores[0]
                    result.loc[below_min, col] = slope * values[below_min] + intercept
                else:
                    # Otherwise clamp to minimum original value
                    result.loc[below_min, col] = min_orig
                    
            if above_max.any():
                if self.quadratic_extrapolation:
                    # Use linear extrapolation above maximum z-score
                    slope = (originals[-1] - originals[-2]) / (z_scores[-1] - z_scores[-2])
                    intercept = originals[-1] - slope * z_scores[-1]
                    result.loc[above_max, col] = slope * values[above_max] + intercept
                else:
                    # Otherwise clamp to maximum original value
                    result.loc[above_max, col] = max_orig

        return result

    def _randrealgen_optimized(self, nreal):
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

    def _moving_average_with_endpoints(self, y_values):
        """Apply a moving average smoothing to an array while preserving endpoints."""
        window_size = 3
        if y_values.shape[0] > 40:
            window_size = 5
        if y_values.shape[0] > 90:
            window_size = 7
        if y_values.shape[0] > 200:
            window_size = 9

        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        half_window = window_size // 2
        smoothed_y = np.zeros_like(y_values)

        # Handle start points correctly
        for i in range(0, half_window):
            smoothed_y[i] = np.mean(y_values[:i + half_window + 1])
        
        # Handle end points correctly 
        for i in range(1, half_window + 1):
            smoothed_y[-i] = np.mean(y_values[-(i + half_window):])
        
        # Middle points
        for i in range(half_window, len(y_values) - half_window):
            smoothed_y[i] = np.mean(y_values[i - half_window:i + half_window + 1])

        # Preserve original endpoints exactly
        smoothed_y[0] = y_values[0]
        smoothed_y[-1] = y_values[-1]
        
        # Ensure monotonicity
        for i in range(1, len(smoothed_y)):
            if smoothed_y[i] <= smoothed_y[i - 1]:
                smoothed_y[i] = smoothed_y[i - 1] + 1e-16

        return smoothed_y

class StandardScalerTransformer(BaseTransformer):
    """Apply standard scaling (zero mean, unit variance) to data."""

    def __init__(self):
        self.means = {}
        self.stds = {}

    def fit(self, X):
        """Compute mean and standard deviation for each feature."""
        for col in X.columns:
            self.means[col] = X[col].mean()
            self.stds[col] = X[col].std()
            if self.stds[col] == 0:
                self.stds[col] = 1.0  # Avoid division by zero
        return self

    def transform(self, X):
        """Transform the data using mean and std from fit."""
        result = X.copy()
        for col in X.columns:
            if col in self.means:
                mean = self.means[col]
                std = self.stds[col]
                result[col] = (X[col] - mean) / std
        return result

    def inverse_transform(self, X):
        """Inverse transform data back to original scale."""
        result = X.copy()
        for col in X.columns:
            if col in self.means:
                mean = self.means[col]
                std = self.stds[col]
                result[col] = (X[col] * std) + mean
        return result





