"""
Learning-based pattern-data-driven forecast approach (LDFA) emulator implementation.
"""
from __future__ import print_function, division
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from .base import Emulator
from .transformers import RowWiseMinMaxScaler

# Define PyTorch model class
class LDFAModel(nn.Module):
    """
    PyTorch implementation of the LDFA neural network model.
    """
    def __init__(self, input_dim, output_dim, hidden_units=None, activation='relu', dropout_rate=0.0):
        super().__init__()
        
        if hidden_units is None:
            hidden_units = [2 * input_dim]
            
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers
        for units in hidden_units:
            layers.append(nn.Linear(prev_dim, units))
            
            # Apply activation function
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())  # Default to ReLU
                
            # Apply dropout if specified
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
                
            prev_dim = units
            
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass through the network"""
        return self.model(x)

# Early stopping handler
class EarlyStopping:
    """
    Early stopping to terminate training when validation loss doesn't improve.
    """
    def __init__(self, patience=20, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_model = {key: val.cpu().clone() for key, val in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_model is not None:
                    model.load_state_dict(self.best_model)

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

        
        # Apply transformations using the base class method
        if transforms:
            train_transformed = self.apply_feature_transforms(train, transforms)
            test_transformed = self.feature_transformer.transform(test)
        else:
            train_transformed = train
            test_transformed = test
        
        # Apply row-wise min-max scaling directly (not through the pipeline)
        self.rowwise_mm_scalers ={
            "train": RowWiseMinMaxScaler(
                        feature_range=(-1, 1),
                        groups=self.groups,
                        fit_groups=self.fit_groups )
        }
        
        self.rowwise_mm_scalers["train"].fit(train_transformed)
        train_scaled = self.rowwise_mm_scalers["train"].transform(train_transformed)


        self.rowwise_mm_scalers["test"] =  RowWiseMinMaxScaler(
                        feature_range=(-1, 1),
                        groups=self.groups,
                        fit_groups=self.fit_groups )
        self.rowwise_mm_scalers["train"].fit(test_transformed)
        test_scaled = self.rowwise_mm_scalers["test"].transform(test_transformed)

        
        self.logger.statement("row-wise min-max scaling complete")
        
        # Split datasets into input (X) and target (y) variables
        X_train = train_scaled.loc[:, self.input_cols].copy()
        y_train = train_scaled.loc[:, self.forecast_names].copy()
        
        X_test = test_scaled.loc[:, self.input_cols].copy()
        y_test = test_scaled.loc[:, self.forecast_names].copy()
        
        # Apply PCA to reduce the dimensionality of the data
        self.logger.statement("applying PCA dimensionality reduction")
        self.pcaX = PCA()#n_components=X_test.shape[1])
        self.pcay = PCA()#n_components=y_test.shape[1])

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
        LDFAModel
            The PyTorch model instance.
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
        
        # Create the model architecture
        model = LDFAModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_units=hidden_units,
            activation=activation,
            dropout_rate=dropout_rate
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
        # Create the noise model with the same architecture as the main model
        self.noise_model = self._build_model(params)
        
        # Get residuals from main model predictions
        with torch.no_grad():
            # Convert input data to tensors
            X_tensor = torch.tensor(self.X, dtype=torch.float32)
            y_tensor = torch.tensor(self.y, dtype=torch.float32)
            
            # Get predictions from main model
            self.model.eval()
            pred_tensor = self.model(X_tensor)
            
            # Calculate residuals
            res_tensor = y_tensor - pred_tensor
            
            # Get test residuals
            X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(self.y_test, dtype=torch.float32)
            pred_test_tensor = self.model(X_test_tensor)
            res_test_tensor = y_test_tensor - pred_test_tensor

        # Train the noise model on residuals
        self.logger.statement("training noise model on residuals")
        
        # Create DataLoader for batch processing
        train_dataset = TensorDataset(X_tensor, res_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True
        )
        
        # Setup optimizer and loss function
        optimizer = optim.Adam(self.noise_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # Setup early stopping
        early_stop = None
        if self.early_stop:
            early_stop = EarlyStopping(
                patience=20, 
                restore_best_weights=True
            )
        
        # Set model to training mode
        self.noise_model.train()
        
        # Training loop
        for epoch in range(200):
            epoch_loss = 0.0
            
            # Train on batches
            for batch_X, batch_res in train_loader:
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.noise_model(batch_X)
                loss = criterion(outputs, batch_res)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch_X)
            
            # Calculate average loss
            epoch_loss /= len(X_tensor)
            
            # Validation
            with torch.no_grad():
                self.noise_model.eval()
                val_outputs = self.noise_model(X_test_tensor)
                val_loss = criterion(val_outputs, res_test_tensor).item()
                self.noise_model.train()
            
            # Log progress
            if  (epoch + 1) % 20 == 0:
                self.logger.statement(
                    f"Noise model - Epoch {epoch+1}/200 - loss: {epoch_loss:.6f} - val_loss: {val_loss:.6f}"
                )
            
            # Early stopping check
            if early_stop:
                early_stop(val_loss, self.noise_model)
                if early_stop.early_stop:
                    self.logger.statement(f"Early stopping noise model at epoch {epoch+1}")
                    break
        
        # Set noise model to evaluation mode for inference
        self.noise_model.eval()
        
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
        
        # Set model to training mode
        self.model.train()
        
        # Convert numpy arrays to PyTorch tensors
        X_train = torch.tensor(self.X if X is None else X, dtype=torch.float32)
        y_train = torch.tensor(self.y, dtype=torch.float32)
        X_test = torch.tensor(self.X_test, dtype=torch.float32)
        y_test = torch.tensor(self.y_test, dtype=torch.float32)
        
        # Create DataLoader for batch processing
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Create optimizer and loss function
        self.logger.statement(f"fitting model: {epochs} epochs, batch size {batch_size}")
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # Track training history (like Keras' History object)
        history = {
            'loss': [],
            'val_loss': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Training
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch_X)
            
            # Calculate average loss for the epoch
            epoch_loss /= len(X_train)
            history['loss'].append(epoch_loss)
            
            # Validation
            with torch.no_grad():
                self.model.eval()  # Set model to evaluation mode
                val_outputs = self.model(X_test)
                val_loss = criterion(val_outputs, y_test).item()
                history['val_loss'].append(val_loss)
                self.model.train()  # Set model back to training mode
            
            # Log progress
            if  (epoch + 1) % 10 == 0:
                self.logger.statement(
                    f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.6f} - val_loss: {val_loss:.6f}"
                )
            
            # Early stopping check
            if self.early_stop:
                self.early_stop(val_loss, self.model)
                if self.early_stop.early_stop:
                    self.logger.statement(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Set model to evaluation mode for inference
        self.model.eval()
        self.fitted = True
        self.history = history
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
        
        # Set model to evaluation mode
        self.model.eval()
        if self.noise_model is not None:
            self.noise_model.eval()
            
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
        truth_pca = self.pcaX.transform(X_truth.values)
        
        # STEP 2: Run model prediction with torch.no_grad() for efficient inference
        self.logger.statement("running model prediction")
        with torch.no_grad():
            # Convert numpy array to PyTorch tensor
            truth_tensor = torch.tensor(truth_pca, dtype=torch.float32)
            
            # Get model prediction
            pred_tensor = self.model(truth_tensor)
            
            # Add noise prediction if available
            if self.noise_model is not None:
                self.logger.statement("adding noise model prediction")
                noise_tensor = self.noise_model(truth_tensor)
                pred_tensor = pred_tensor + noise_tensor
                
            # Convert PyTorch tensor back to numpy array
            pred_pca = pred_tensor.cpu().numpy()
        
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