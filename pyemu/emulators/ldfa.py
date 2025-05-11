"""
Learning-based pattern-data-driven forecast approach (LDFA) emulator implementation.

Note: This module requires TensorFlow. Install with:
    pip install pyemu[emulators-ldfa] or pip install tensorflow
"""
from __future__ import print_function, division
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Import tensorflow lazily to allow importing this module even if tf is not installed
# The actual import will occur when the LDFA class is instantiated
def _import_tensorflow():
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        raise ImportError(
            "The LDFA emulator requires TensorFlow, which is not installed. "
            "Install it with 'pip install pyemu[emulators-ldfa]' or 'pip install tensorflow'."
        )

from .base import Emulator
from .transformers import RowWiseMinMaxScaler

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
        # Import tensorflow here to avoid import error if not installed
        # This will raise a helpful error message if tensorflow is not available
        self.tf = _import_tensorflow()
        
        # Make Keras components available as attributes
        self.layers = self.tf.keras.layers
        self.models = self.tf.keras.models
        self.EarlyStopping = self.tf.keras.callbacks.EarlyStopping
        
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
            self.early_stop = self.EarlyStopping(
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