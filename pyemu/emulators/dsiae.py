"""
Data Space Inversion AutoEncoder (DSIAE) emulator implementation.

Note: This module requires TensorFlow. Install with:
    pip install pyemu[emulators-dsiae] or pip install tensorflow
"""
from __future__ import print_function, division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import datetime
import copy

from .base import Emulator

class Encoder(nn.Module):
    """
    Encoder component of the DSIAE Variational Autoencoder.
    
    This module encodes data from different modalities (scalar, time series, spatial)
    into a latent space representation.
    
    Parameters
    ----------
    input_shapes : dict
        Dictionary containing the shapes of different input types:
        {
            'scalar': (num_scalar_features,),
            'timeseries': [(ts1_seq_len, ts1_features), (ts2_seq_len, ts2_features), ...],
            'spatial': [(spatial1_height, spatial1_width, spatial1_channels), ...]
        }
    latent_dim : int
        Dimension of the latent space.
    """
    def __init__(self, input_shapes, latent_dim=8):
        super().__init__()
        
        self.input_shapes = input_shapes
        self.latent_dim = latent_dim
        
        # Branch processing networks
        self.scalar_net = None
        self.timeseries_nets = nn.ModuleList() if 'timeseries' in input_shapes else None
        self.spatial_nets = nn.ModuleList() if 'spatial' in input_shapes else None
        
        # Dimensions for each branch output
        self.scalar_dim = 0
        self.timeseries_dim = 0
        self.spatial_dim = 0
        
        # Build the scalar branch
        if 'scalar' in input_shapes and input_shapes['scalar'][0] > 0:
            scalar_input_dim = input_shapes['scalar'][0]
            self.scalar_dim = min(128, scalar_input_dim * 2)
            
            self.scalar_net = nn.Sequential(
                nn.Linear(scalar_input_dim, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, self.scalar_dim),
                nn.LeakyReLU(0.2),
            )
        
        # Build time series branches using LSTMs instead of CNNs
        if 'timeseries' in input_shapes and input_shapes['timeseries']:
            for i, shape in enumerate(input_shapes['timeseries']):
                seq_len, features = shape
                hidden_dim = (2*features)//1  # Output dimension for each time series
                
                # Use bidirectional LSTM that can handle any sequence length
                lstm = nn.LSTM(
                    input_size=features, 
                    hidden_size=hidden_dim, 
                    num_layers=3,
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.1
                )
                
                # Create adapter to extract features from LSTM output
                adapter = LSTMOutputAdapter(hidden_dim * 2)  # *2 for bidirectional
                
                ts_net = nn.ModuleList([lstm, adapter])
                self.timeseries_nets.append(ts_net)
                self.timeseries_dim += hidden_dim * 2  # *2 for bidirectional LSTM
         
        # Build spatial branches if needed
        if 'spatial' in input_shapes and input_shapes['spatial']:
            for i, shape in enumerate(input_shapes['spatial']):
                h, w, c = shape
                
                # For small images/grids, use simpler networks
                if h * w <= 100:  # e.g., 10x10 or smaller
                    spatial_net = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(h * w * c, 64),
                        nn.LeakyReLU(0.2),
                        nn.Linear(64, 32),
                        nn.LeakyReLU(0.2)
                    )
                # For larger spatial data, use CNNs
                else:
                    spatial_net = nn.Sequential(
                        nn.Conv2d(c, 16, kernel_size=3, padding=1),
                        nn.LeakyReLU(0.2),
                        nn.MaxPool2d(2),
                        nn.Conv2d(16, 32, kernel_size=3, padding=1),
                        nn.LeakyReLU(0.2),
                        nn.MaxPool2d(2),
                        nn.Flatten(),
                        nn.Linear(32 * (h // 4) * (w // 4), 32),
                        nn.LeakyReLU(0.2)
                    )
                    
                self.spatial_nets.append(spatial_net)
                self.spatial_dim += 32  # Add output dimension for each spatial branch
        
        # Calculate total feature dimension after all branches
        self.total_features = self.scalar_dim + self.timeseries_dim + self.spatial_dim
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.total_features, 64),
            nn.LeakyReLU(0.2)
        )
        
        # VAE bottleneck outputs
        self.z_mean = nn.Linear(64, latent_dim)
        self.z_log_var = nn.Linear(64, latent_dim)
        
    def reparameterize(self, z_mean, z_log_var):
        """
        Reparameterization trick to sample from N(z_mean, z_var) while allowing backprop.
        
        Parameters
        ----------
        z_mean : torch.Tensor
            Mean of the latent Gaussian.
        z_log_var : torch.Tensor
            Log variance of the latent Gaussian.
            
        Returns
        -------

        torch.Tensor
            Sampled latent vector.
        """
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        z = z_mean + eps * std
        return z
    
    def forward(self, inputs):
        """
        Forward pass through the encoder.
        
        Parameters
        ----------

        inputs : list of torch.Tensor
            List of input tensors for each data modality.
            
        Returns
        -------

        tuple
            (z_mean, z_log_var, z) - latent space representations:
            z_mean: mean values of the latent space
            z_log_var: log variance values of the latent space
            z: sampled latent vectors
        """
        features = []
        input_idx = 0
        
        # Process scalar inputs
        if self.scalar_net is not None:
            scalar_features = self.scalar_net(inputs[input_idx])
            features.append(scalar_features)
            input_idx += 1
        
        # Process time series inputs with LSTM
        if self.timeseries_nets is not None:
            for i, ts_net in enumerate(self.timeseries_nets):
                ts_input = inputs[input_idx]  # Already in shape [batch, seq_len, features]
                
                # Process through LSTM and adapter
                lstm, adapter = ts_net
                output, (h_n, c_n) = lstm(ts_input)
                ts_features = adapter(output, h_n)
                
                features.append(ts_features)
                input_idx += 1
        
        # Process spatial inputs
        if self.spatial_nets is not None:
            for i, spatial_net in enumerate(self.spatial_nets):
                spatial_input = inputs[input_idx]
                # Reshape if using 2D CNNs
                if isinstance(spatial_net[0], nn.Conv2d):
                    # Input shape should be [batch, channels, height, width]
                    spatial_input = spatial_input.permute(0, 3, 1, 2)
                spatial_features = spatial_net(spatial_input)
                features.append(spatial_features)
                input_idx += 1
        
        # Concatenate all features
        x = torch.cat(features, dim=1)
        
        # Apply fusion layer
        x = self.fusion(x)
        
        # VAE outputs
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.reparameterize(z_mean, z_log_var)
        
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    """
    Decoder component of the DSIAE Variational Autoencoder.
    
    This module decodes latent space representations back to the original data spaces,
    including scalar, time series, and spatial data.
    
    Parameters
    ----------

    latent_dim : int
        Dimension of the latent space.
    output_shapes : dict
        Dictionary containing the shapes of different output types:
        {
            'scalar': (num_scalar_features,),
            'timeseries': [(ts1_seq_len, ts1_features), (ts2_seq_len, ts2_features), ...],
            'spatial': [(spatial1_height, spatial1_width, spatial1_channels), ...]
        }
    """
    def __init__(self, latent_dim, output_shapes):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_shapes = output_shapes
        
        # Latent to hidden expansion
        self.latent_expansion = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2)
        )
        
        # Branch processors
        self.scalar_net = None
        self.timeseries_nets = nn.ModuleList() if 'timeseries' in output_shapes else None
        self.spatial_nets = nn.ModuleList() if 'spatial' in output_shapes else None
        
        # Create output branches
        
        # Scalar branch
        if 'scalar' in output_shapes and output_shapes['scalar'][0] > 0:
            scalar_output_dim = output_shapes['scalar'][0]
            self.scalar_net = nn.Sequential(
                nn.Linear(64, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, scalar_output_dim)
            )
        
        # Time series branches using LSTMs
        if 'timeseries' in output_shapes and output_shapes['timeseries']:
            for i, shape in enumerate(output_shapes['timeseries']):
                seq_len, features = shape
                hidden_dim = (2*features)//1  # Output dimension for each time series
                
                # Sequential network for time series generation
                ts_net = nn.Sequential(
                    # First expand latent to features for each timestep
                    nn.Linear(64, hidden_dim * seq_len),
                    nn.LeakyReLU(0.2),
                    TimeSeriesReshaper(seq_len, hidden_dim),
                    
                    # Then use LSTM to generate the output sequence
                    nn.LSTM(
                        input_size=hidden_dim,
                        hidden_size=hidden_dim,
                        num_layers=3,
                        batch_first=True,
                        dropout=0.1
                    ),
                    
                    # Final projection to target dimensions
                    TimeSeriesOutputProjector(hidden_dim, features, seq_len)
                )
                
                self.timeseries_nets.append(ts_net)
                
        # Spatial branches
        if 'spatial' in output_shapes and output_shapes['spatial']:
            for i, shape in enumerate(output_shapes['spatial']):
                h, w, c = shape
                
                # For small images/grids, use simpler networks
                if h * w <= 100:  # e.g., 10x10 or smaller
                    spatial_net = nn.Sequential(
                        nn.Linear(64, 64),
                        nn.LeakyReLU(0.2),
                        nn.Linear(64, h * w * c)
                    )
                # For larger spatial data, use transposed convolutions
                else:
                    # Calculate intermediate dimensions
                    h_small = h // 4
                    w_small = w // 4
                    
                    spatial_net = nn.Sequential(
                        nn.Linear(64, 32 * h_small * w_small),
                        nn.LeakyReLU(0.2),
                        nn.Unflatten(1, (32, h_small, w_small)),
                        nn.Upsample(scale_factor=2),
                        nn.Conv2d(32, 16, kernel_size=3, padding=1),
                        nn.LeakyReLU(0.2),
                        nn.Upsample(scale_factor=2),
                        nn.Conv2d(16, c, kernel_size=3, padding=1)
                    )
                    
                self.spatial_nets.append(spatial_net)
                
    def forward(self, z):
        """
        Forward pass through the decoder.
        
        Parameters
        ----------

        z : torch.Tensor
            Latent space representation.
            
        Returns
        -------

        list
            List of decoded outputs for each data type.
        """
        # Expand latent vector to hidden representation
        x = self.latent_expansion(z)
        
        outputs = []
        
        # Decode scalar outputs if needed
        if self.scalar_net is not None:
            scalar_output = self.scalar_net(x)
            outputs.append(scalar_output)
        
        # Decode time series outputs with LSTM
        if self.timeseries_nets is not None:
            for i, ts_net in enumerate(self.timeseries_nets):
                # Process through the sequential model
                ts_output = ts_net(x)
                outputs.append(ts_output)
        
        # Decode spatial outputs
        if self.spatial_nets is not None:
            for i, spatial_net in enumerate(self.spatial_nets):
                spatial_output = spatial_net(x)
                
                # Reshape output if using transposed convolutions
                if len(spatial_output.shape) == 4:  # [batch, channels, height, width]
                    # Change to [batch, height, width, channels]
                    spatial_output = spatial_output.permute(0, 2, 3, 1)
                else:
                    # Reshape to [batch, height, width, channels]
                    h, w, c = self.output_shapes['spatial'][i]
                    spatial_output = spatial_output.view(-1, h, w, c)
                    
                outputs.append(spatial_output)
        
        return outputs


class VariationalAutoEncoder(nn.Module):
    """
    Variational Autoencoder for DSIAE.
    
    This combines the encoder and decoder components into a complete VAE model.
    
    Parameters
    ----------

    input_shapes : dict
        Dictionary containing the shapes of different input types.
    output_shapes : dict
        Dictionary containing the shapes of different output types.
    latent_dim : int
        Dimension of the latent space. Default is 8.
    """
    def __init__(self, input_shapes, output_shapes, latent_dim=8):
        super().__init__()
        
        self.encoder = Encoder(input_shapes, latent_dim)
        self.decoder = Decoder(latent_dim, output_shapes)
        
    def forward(self, inputs):
        """
        Forward pass through the complete autoencoder.
        
        Parameters
        ----------

        inputs : list of torch.Tensor
            List of input tensors for each data modality.
            
        Returns
        -------

        tuple
            (outputs, z_mean, z_log_var, z) where:
            outputs: list of decoded outputs for each data type
            z_mean: mean values of the latent space
            z_log_var: log variance values of the latent space
            z: sampled latent vectors
        """
        # Encode inputs to latent space
        z_mean, z_log_var, z = self.encoder(inputs)
        
        # Decode latent vectors to outputs
        outputs = self.decoder(z)
        
        return outputs, z_mean, z_log_var, z


class DSIAE(Emulator):
    """
    Data Space Inversion AutoEncoder (DSIAE) Emulator for emulating the relationship between 
    parameter space and observation space.
    
    DSIAE extends the DSI approach by replacing PCA with a neural network-based 
    variational autoencoder for more powerful nonlinear dimensionality reduction.
    
    Parameters
    ----------
    sim_obs : pandas.DataFrame
        Simulation observations data with columns as observation names and rows as realizations.
    pars : pandas.DataFrame, optional
        Parameters data with columns as parameter names and rows as realizations.
        This is only needed for prediction, not for training the autoencoder.
    obs_names : list, optional
        List of observation names to include in the emulator. If None, all observations are used.
    par_names : list, optional
        List of parameter names to include in the emulator. If None, all parameters are used.
    data_type_map : dict, optional
        Dictionary specifying how to organize observation columns by data type. Must have the structure:
        {
            'scalar': [column names],  # All scalar columns
            'timeseries': {
                'group1': {
                    'columns': [column names],  # Ordered list of column names
                    'coordinates': [t1, t2, ...],  # Corresponding time coordinates 
                    'frequency': '1D',  # Optional for uniform time steps
                    'irregular': False  # Optional flag for irregular time series
                },
                'group2': {
                    'columns': [...],
                    'coordinates': [...],
                    ...
                },
                ...
            },
            'spatial': {
                'group1': {
                    'columns': [column names],  # Ordered list of column names
                    'coordinates': [(x1,y1,z1), (x2,y2,z2), ...],  # Corresponding spatial coordinates
                    'dimensions': (nx, ny, nz),  # Optional grid dimensions
                    'spacing': (dx, dy, dz)  # Optional grid spacing
                },
                'group2': {
                    'columns': [...],
                    'coordinates': [...],
                    ...
                },
                ...
            }
        }
        
        For both timeseries and spatial data, the coordinates list must be the same length as the columns list.
        Time series coordinates can be numeric values representing time steps or points.
        Spatial coordinates should be tuples of 2D (x,y) or 3D (x,y,z) coordinates.
        
        If None, all observations are treated as scalars.
    latent_dim : int, optional
        Dimension of the latent space. Default is 8.
    learning_rate : float, optional
        Learning rate for the optimizer. Default is 0.001.
    verbose : bool, optional
        If True, enable verbose logging. Default is True.
    device : str, optional
        Device to use for training. Default is 'auto', which will use GPU if available.
    transforms : list of dict, optional
        List of transformations to apply to the data before encoding/decoding. Each dict should have:
        - 'type': str - Type of transformation (e.g., 'log10', 'normal_score')
        - 'columns': list - Columns to apply the transformation to (optional)
        - Additional kwargs specific to the transformer
        If None, no transformations are applied.
    """

    def __init__(self, 
                sim_obs=None, 
                pars=None, 
                obs_names=None, 
                par_names=None, 
                data_type_map=None, 
                latent_dim=8, 
                learning_rate=0.001, 
                verbose=True,
                device='auto',
                transforms=None):
        """
        Initialize the DSIAE emulator.

        Parameters
        ----------
        sim_obs : pandas.DataFrame
            Simulation observations data with columns as observation names and rows as realizations.
        pars : pandas.DataFrame, optional
            Parameters data with columns as parameter names and rows as realizations.
            This is only needed for prediction, not for training the autoencoder.
        obs_names : list, optional
            List of observation names to include in the emulator. If None, all observations are used.
        par_names : list, optional
            List of parameter names to include in the emulator. If None, all parameters are used.
        data_type_map : dict, optional
            Dictionary specifying how to organize observation columns by data type.
        latent_dim : int, optional
            Dimension of the latent space. Default is 8.
        learning_rate : float, optional
            Learning rate for the optimizer. Default is 0.001.
        verbose : bool, optional
            If True, enable verbose logging. Default is True.
        device : str, optional
            Device to use for training. Default is 'auto', which will use GPU if available.
        transforms : list of dict, optional
            List of transformations to apply to the data before encoding/decoding. Each dict should have:
            - 'type': str - Type of transformation (e.g., 'log10', 'normal_score')
            - 'columns': list - Columns to apply the transformation to (optional)
            - Additional kwargs specific to the transformer
            If None, no transformations are applied.
        """
        super().__init__(verbose=verbose)
        
        # Store input data
        self.sim_obs = sim_obs
        self.pars = pars  # Optional - only needed for prediction, not for model training
        
        # Set observation and parameter names
        if obs_names is None and sim_obs is not None:
            obs_names = sim_obs.columns.tolist()
        self.obs_names = obs_names
        
        if par_names is None and pars is not None:
            par_names = pars.columns.tolist()
        self.par_names = par_names
        
        # Data organization
        self.data_type_map = data_type_map or {}  # Default to empty dict if None
        self.organized_obs = None
        
        # Model parameters
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        
        # Data transformations configuration
        self.transforms = transforms or []
        
        # Select device (CPU or GPU)
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger.statement(f"Using device: {self.device}")
        
        # State variables
        self.encoder = None
        self.decoder = None
        self.vae = None
        self.obs_vec = None  # Mean of observations
        self.par_vec = None  # Mean of parameters (only used if parameters provided)
        self.sim_vals = None  # Transformed training data
        self.kdtree = None   # KDTree for parameter space (created only when needed for prediction)
        self.min_max = {}    # Parameter bounds (created only when needed for prediction)
        self.weights = None
        self.use_localizer = False
        self.fitted = False
        
        # Training history
        self.training_history = {
            'total_loss': [],
            'reconstruction_loss': [],
            'kl_loss': [],
            'val_total_loss': [],
            'val_reconstruction_loss': [],
            'val_kl_loss': []
        }
        
        # If observation data is provided at initialization, prepare it
        if sim_obs is not None:
            self.prepare_data(sim_obs, pars, obs_names, par_names)
        
    def prepare_data(self, sim_obs, pars=None, obs_names=None, par_names=None):
        """
        Prepare data for training the DSIAE emulator.
        
        Parameters
        ----------
        sim_obs : pandas.DataFrame
            Simulation observations data with columns as observation names and rows as realizations.
        pars : pandas.DataFrame, optional
            Parameters data with columns as parameter names and rows as realizations.
            This is only needed for prediction, not for training the autoencoder.
        obs_names : list, optional
            List of observation names to include in the emulator.
        par_names : list, optional
            List of parameter names to include in the emulator.
            
        Returns
        -------
        None
        """
        # Store original data
        self.sim_obs = sim_obs
        self.pars = pars  # May be None if only training
        
        # Update observation names if provided
        if obs_names is not None:
            self.obs_names = obs_names
        elif self.obs_names is None:
            self.obs_names = sim_obs.columns.tolist()
            
        # Update parameter names if both pars and par_names are provided
        if pars is not None:
            if par_names is not None:
                self.par_names = par_names
            elif self.par_names is None:
                self.par_names = pars.columns.tolist()
        
        # Filter observation data to specified columns
        sim_obs_filtered = sim_obs[self.obs_names].copy()
        
        # Store parameter information if provided
        if pars is not None and self.par_names is not None:
            # Filter parameter data to specified columns
            pars_filtered = pars[self.par_names].copy()
            
            # Store min/max values for parameters for later checks
            self.min_max = {}
            for col in pars_filtered.columns:
                self.min_max[col] = (pars_filtered[col].min(), pars_filtered[col].max())
                
            # Compute mean parameter vector
            self.par_vec = pars_filtered.mean()
            
            # Note: KDTree is created only when needed for prediction, not here
        
        # Apply transformations if configured
        if self.transforms:
            self.logger.statement("Applying data transformations using AutobotsAssemble")
            
            # Import AutobotsAssemble here to avoid circular imports
            from .transformers import AutobotsAssemble
            
            # Create an AutobotsAssemble instance with the filtered data
            transformer = AutobotsAssemble(sim_obs_filtered)
            
            # Apply the configured transformations
            for transform in self.transforms:
                transform_type = transform.get('type')
                columns = transform.get('columns')
                
                # Extract transformer-specific kwargs
                kwargs = {k: v for k, v in transform.items() 
                         if k not in ('type', 'columns')}
                
                self.logger.statement(f"Applying {transform_type} transformation")
                transformer.apply(transform_type, columns=columns, **kwargs)
            
            # Store the transformer for inverse operations later
            self.feature_transformer = transformer
            
            # Get the transformed data
            sim_obs_transformed = transformer.df.copy()
            
            self.logger.statement("Transformations applied successfully")
        else:
            # No transformations applied
            sim_obs_transformed = sim_obs_filtered.copy()
            self.feature_transformer = None
        
        # Compute mean observation vector from transformed data
        self.obs_vec = sim_obs_transformed.mean()
        
        # Center observation data
        obs_centered = sim_obs_transformed - self.obs_vec.values
        
        # Store transformed data
        self.sim_vals = sim_obs_transformed
        
        # Organize data by type
        self.organized_obs = self._organize_data_types(obs_centered)
        
        self.logger.statement("Data prepared for training")
        
    def _organize_data_types(self, df):
        """
        Organize observation data into different types based on data_type_map.
        
        This method transforms a flat DataFrame into a structured dictionary
        according to the data types specified in data_type_map.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing observation data (rows are realizations)
        
        Returns
        -------
        dict
            Dictionary with keys for different data types:
            {
                'input_shapes': {
                    'scalar': (num_scalar_features,),
                    'timeseries': [(length1, features1), (length2, features2), ...],
                    'spatial': [(height1, width1, channels1), ...],
                },
                'output_shapes': Same structure as input_shapes,
                'tensor_inputs': [tensor1, tensor2, ...],  # PyTorch tensors ready for model
                'column_groups': [(data_type, group_name, columns), ...], # Column organization for reconstruction
            }
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        # Create default data_type_map if not provided
        if not self.data_type_map:
            self.data_type_map = {'scalar': df.columns.tolist()}
        
        # Initialize result dictionary
        result = {
            'input_shapes': {
                'scalar': (0,)
            },
            'output_shapes': {
                'scalar': (0,)
            },
            'tensor_inputs': [],
            'column_groups': []
        }
        
        # Process scalar data
        if 'scalar' in self.data_type_map and self.data_type_map['scalar']:
            scalar_cols = self.data_type_map['scalar']
            if scalar_cols and len(scalar_cols) > 0:
                # Check if all columns exist
                missing = [col for col in scalar_cols if col not in df.columns]
                if missing:
                    raise ValueError(f"Missing scalar columns in dataframe: {missing}")
                    
                # Extract scalar data
                scalar_data = df[scalar_cols].values
                
                # Update shapes
                result['input_shapes']['scalar'] = (scalar_data.shape[1],)
                result['output_shapes']['scalar'] = (scalar_data.shape[1],)
                
                # Convert to tensor
                scalar_tensor = torch.tensor(scalar_data, dtype=torch.float32, device=self.device)
                result['tensor_inputs'].append(scalar_tensor)
                
                # Store column group
                result['column_groups'].append(('scalar', 'scalar', scalar_cols))
                
                self.logger.statement(f"Prepared {len(scalar_cols)} scalar variables")
        
        # Process time series data
        if 'timeseries' in self.data_type_map and self.data_type_map['timeseries']:
            ts_data = self.data_type_map['timeseries']
            
            result['input_shapes']['timeseries'] = []
            result['output_shapes']['timeseries'] = []
            
            for group_name, group_info in ts_data.items():
                ts_cols = group_info.get('columns', [])
                ts_coords = group_info.get('coordinates', [])
                
                # Ensure we have coordinates and columns
                if not ts_cols or not ts_coords:
                    self.logger.warn(f"Skipping time series group '{group_name}': missing columns or coordinates")
                    continue
                    
                if len(ts_cols) != len(ts_coords):
                    raise ValueError(f"Time series group '{group_name}' has mismatched columns and coordinates lengths")
                
                # Check if all columns exist
                missing = [col for col in ts_cols if col not in df.columns]
                if missing:
                    raise ValueError(f"Missing time series columns in dataframe for group '{group_name}': {missing}")
                    
                # Extract time series data
                ts_values = df[ts_cols].values
                num_samples = ts_values.shape[0]
                
                # Determine shape (samples, time_steps, features)
                # For regular time series, features=1 (single feature at each time step)
                # For multivariate time series with the same time base, features>1
                
                # Default assumption: each column is a separate time step for a single feature
                time_steps = len(ts_cols)
                features = 1
                
                # Reshape to match expected input format for 1D convolutions: (samples, time_steps, features)
                ts_shaped = ts_values.reshape((num_samples, time_steps, features))
                
                # Update shapes
                result['input_shapes']['timeseries'].append((time_steps, features))
                result['output_shapes']['timeseries'].append((time_steps, features))
                
                # Convert to tensor
                ts_tensor = torch.tensor(ts_shaped, dtype=torch.float32, device=self.device)
                result['tensor_inputs'].append(ts_tensor)
                
                # Store column group
                result['column_groups'].append(('timeseries', group_name, ts_cols))
                
                self.logger.statement(f"Prepared time series group '{group_name}' with {time_steps} time steps")
        
        # Process spatial data
        if 'spatial' in self.data_type_map and self.data_type_map['spatial']:
            spatial_data = self.data_type_map['spatial']
            
            result['input_shapes']['spatial'] = []
            result['output_shapes']['spatial'] = []
            
            for group_name, group_info in spatial_data.items():
                sp_cols = group_info.get('columns', [])
                sp_coords = group_info.get('coordinates', [])
                dimensions = group_info.get('dimensions', None)
                
                # Ensure we have coordinates and columns
                if not sp_cols or not sp_coords:
                    self.logger.warn(f"Skipping spatial group '{group_name}': missing columns or coordinates")
                    continue
                    
                if len(sp_cols) != len(sp_coords):
                    raise ValueError(f"Spatial group '{group_name}' has mismatched columns and coordinates lengths")
                
                # Check if all columns exist
                missing = [col for col in sp_cols if col not in df.columns]
                if missing:
                    raise ValueError(f"Missing spatial columns in dataframe for group '{group_name}': {missing}")
                    
                # Extract spatial data
                spatial_values = df[sp_cols].values
                num_samples = spatial_values.shape[0]
                
                # Determine grid dimensions if not provided
                if dimensions is None:
                    # Try to infer dimensions from coordinates
                    coords = np.array(sp_coords)
                    
                    if coords.ndim > 1 and coords.shape[1] >= 2:
                        # 2D or 3D coordinates
                        x_unique = np.unique(coords[:, 0])
                        y_unique = np.unique(coords[:, 1])
                        
                        width = len(x_unique)
                        height = len(y_unique)
                        channels = 1  # Default to 1 channel
                        
                        if coords.shape[1] >= 3:
                            # 3D coordinates might have a z dimension or channel dimension
                            z_unique = np.unique(coords[:, 2])
                            if len(z_unique) > 1:
                                # Multiple z values - treat as 3D grid
                                depth = len(z_unique)
                                dimensions = (height, width, depth, 1)  # (h, w, d, c)
                            else:
                                # Single z value or it represents a channel - treat as 2D grid with channels
                                dimensions = (height, width, 1)  # (h, w, c)
                        else:
                            dimensions = (height, width, 1)  # (h, w, c)
                    else:
                        # Unable to infer dimensions, use a 1D representation
                        self.logger.warn(f"Unable to infer spatial dimensions for group '{group_name}', using flat representation")
                        dimensions = (len(sp_cols), 1, 1)  # (flat_size, 1, 1)
                
                # Ensure we have a 3-tuple for dimensions (h, w, c)
                if len(dimensions) == 2:
                    dimensions = (dimensions[0], dimensions[1], 1)  # Add channel dimension
                elif len(dimensions) == 4:
                    # 3D grid with channels - flatten depth and height for 2D representation
                    h, w, d, c = dimensions
                    dimensions = (h, w * d, c)
                    self.logger.warn(f"Converting 3D grid to 2D representation for group '{group_name}'")
                
                # Reshape spatial data to grid format
                h, w, c = dimensions
                grid_size = h * w * c
                
                if grid_size != len(sp_cols):
                    self.logger.warn(
                        f"Spatial group '{group_name}' dimension mismatch: {dimensions} requires {grid_size} values but {len(sp_cols)} provided"
                    )
                    # Try to adjust dimensions
                    if c == 1:
                        # Adjust width to make dimensions match
                        w = len(sp_cols) // (h * c)
                        dimensions = (h, w, c)
                        self.logger.warn(f"Adjusted dimensions to {dimensions}")
                    else:
                        # Fall back to flat representation
                        dimensions = (1, len(sp_cols), 1)
                        self.logger.warn(f"Falling back to flat representation {dimensions}")
                
                # Reshape values to match grid dimensions
                h, w, c = dimensions
                try:
                    spatial_shaped = spatial_values.reshape((num_samples, h, w, c))
                except ValueError:
                    self.logger.error(
                        f"Failed to reshape spatial data for group '{group_name}'. "
                        f"Cannot reshape array of size {spatial_values.size} into shape ({num_samples}, {h}, {w}, {c})"
                    )
                    # Fall back to flat representation
                    spatial_shaped = spatial_values.reshape((num_samples, 1, len(sp_cols), 1))
                    dimensions = (1, len(sp_cols), 1)
                    self.logger.warn(f"Falling back to flat representation {dimensions}")
                
                # Update shapes
                result['input_shapes']['spatial'].append(dimensions)
                result['output_shapes']['spatial'].append(dimensions)
                
                # Convert to tensor
                spatial_tensor = torch.tensor(spatial_shaped, dtype=torch.float32, device=self.device)
                result['tensor_inputs'].append(spatial_tensor)
                
                # Store column group
                result['column_groups'].append(('spatial', group_name, sp_cols))
                
                self.logger.statement(f"Prepared spatial group '{group_name}' with shape {dimensions}")
        
        return result

    def _build_kdtree(self):
        """
        Build KDTree for parameter space lookup.
        This is only needed for prediction, not for training the autoencoder.
        
        Returns
        -------
        bool
            True if KDTree was built successfully, False otherwise.
        """
        # Check if we have parameter data
        if self.pars is None or self.par_names is None:
            self.logger.statement("No parameter data available to build KDTree")
            return False
            
        # Filter parameter data to specified columns
        pars_filtered = self.pars[self.par_names].copy()
        
        # Create KDTree for parameter space
        self.kdtree = cKDTree(pars_filtered.values)
        self.logger.statement("KDTree built for parameter space")
        
        return True

    def _build_model(self):
        """
        Build the variational autoencoder model based on the organized data.
        
        This method creates the encoder and decoder components of the VAE based on
        the data_type_map and organized observation data.
        
        Returns
        -------
        None
        """
        if self.organized_obs is None:
            raise ValueError("No organized observation data available. Call prepare_data first.")
            
        # Get input and output shapes from organized data
        input_shapes = self.organized_obs['input_shapes']
        output_shapes = self.organized_obs['output_shapes']
        
        # Create encoder and decoder
        self.logger.statement(f"Building encoder with latent dim {self.latent_dim}")
        self.encoder = Encoder(input_shapes, latent_dim=self.latent_dim)
        
        self.logger.statement("Building decoder")
        self.decoder = Decoder(self.latent_dim, output_shapes)
        
        # Create complete VAE model
        self.vae = VariationalAutoEncoder(input_shapes, output_shapes, latent_dim=self.latent_dim)
        
        # Move models to the specified device
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.vae.to(self.device)
        
        # Set up optimizer
        self.optimizer = optim.Adam(self.vae.parameters(), lr=self.learning_rate)
        
        self.logger.statement("Model built successfully")
        return

    def predict(self, par_vals=None):
        """
        Predict observations for a set of parameter values.
        
        This method uses the nearest neighbor approach from the parameter space
        to map to the latent space, then decodes to generate predictions.
        
        Parameters
        ----------
        par_vals : pandas.DataFrame or numpy.ndarray, optional
            Parameter values to predict for. Must have the same column names/order as training parameters.
            If None, the parameter data provided during initialization or preparation will be used.
            
        Returns
        -------
        pandas.DataFrame
            Predicted observations.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
            
        # Check if we have parameter data
        if par_vals is None:
            # If no parameter values are provided, use the training parameters
            if self.pars is None:
                raise ValueError("No parameter values provided for prediction and no training parameters available")
            par_vals = self.pars
            
        # Convert to DataFrame if ndarray
        if isinstance(par_vals, np.ndarray):
            if self.par_names is None:
                raise ValueError("Parameter names must be provided when passing array-like parameter values")
                
            assert par_vals.shape[1] == len(self.par_names), "Input data must have same dimensions as training parameters"
            par_vals = pd.DataFrame(par_vals, columns=self.par_names)
            
        # Ensure all required columns are present
        missing = [col for col in self.par_names if col not in par_vals.columns]
        if missing:
            raise ValueError(f"Missing columns in input data: {missing}")
            
        # Check parameter bounds
        for col in self.par_names:
            if col in self.min_max:
                min_val, max_val = self.min_max[col]
                if par_vals[col].min() < min_val or par_vals[col].max() > max_val:
                    self.logger.warn(f"Parameter {col} has values outside training range. Extrapolation may be unreliable.")
                    
        # Build KDTree if not already built
        if self.kdtree is None:
            success = self._build_kdtree()
            if not success:
                raise ValueError("Failed to build KDTree for prediction. Parameter data must be provided.")
                
        # Query the KDTree for nearest neighbors
        distances, indices = self.kdtree.query(par_vals[self.par_names].values, k=1)
        
        # Get the encoder inputs for the nearest neighbors
        neighbor_obs = self.sim_obs.iloc[indices]
        
        # Encode to latent space and decode to get predictions
        latent_mean,latent_logvar,latent_sample= self.encode(neighbor_obs)
        predictions = self.decode(latent_sample)
        
        return predictions

    def fit(self, 
            epochs=100, 
            batch_size=32, 
            beta=1.0, 
            beta_annealing=True, 
            validation_split=0.2, 
            early_stopping=True, 
            patience=10, 
            return_history=False):
        """
        Train the variational autoencoder on the provided data.
        
        Parameters
        ----------
        epochs : int, optional
            Number of training epochs. Default is 100.
        batch_size : int, optional
            Batch size for training. Default is 32.
        beta : float, optional
            Weight for KL divergence loss term. Default is 1.0.
        beta_annealing : bool, optional
            Whether to use KL divergence annealing during training. Default is True.
        validation_split : float, optional
            Fraction of data to use for validation. Default is 0.2.
        early_stopping : bool, optional
            Whether to use early stopping during training. Default is True.
        patience : int, optional
            Number of epochs to wait for improvement before stopping. Default is 10.
        return_history : bool, optional
            Whether to return the training history. Default is False.
            
        Returns
        -------
        dict or self
            If return_history is True, returns a dictionary containing training history.
            Otherwise returns self.
        """
        if self.organized_obs is None:
            raise ValueError("No organized observation data available. Call prepare_data first.")

        if self.encoder is None or self.decoder is None or self.vae is None:
            self._build_model()

        # Get training data
        train_data = self.organized_obs['tensor_inputs']
        
        # Create PyTorch Dataset and DataLoader
        train_dataset = torch.utils.data.TensorDataset(*train_data)
        
        # Split into training and validation
        dataset_size = len(train_dataset)
        val_size = int(validation_split * dataset_size)
        train_size = dataset_size - val_size
        
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=batch_size, shuffle=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=batch_size, shuffle=False
        )
        
        self.logger.statement(f"Training on {train_size} samples, validating on {val_size} samples")
        
        # Set up early stopping
        if early_stopping:
            best_val_loss = float('inf')
            best_model_state = None
            counter = 0
        
        # Training loop
        for epoch in range(epochs):
            # Set up KL annealing if enabled
            current_beta = beta
            if beta_annealing:
                # Gradually increase beta from 0 to target value over first half of epochs
                current_beta = beta * min(1.0, epoch / (epochs / 2))
            
            # Training phase
            self.vae.train()
            train_total_loss = 0
            train_recon_loss = 0
            train_kl_loss = 0
            
            for batch_idx, batch_data in enumerate(train_loader):
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs, z_mean, z_log_var, z = self.vae(batch_data)
                
                # Calculate reconstruction loss
                recon_loss = 0
                for i, (output, target) in enumerate(zip(outputs, batch_data)):
                    # Handle mismatched shapes by ensuring shapes match before calculating loss
                    if output.shape != target.shape:
                        # Adjust output shape to match target shape if they're compatible
                        # This usually happens when batched tensors have different sequence lengths
                        if output.numel() == target.numel():
                            output = output.view(target.shape)
                        else:
                            # If number of elements don't match, use a safe fallback
                            self.logger.warn(
                                f"Shape mismatch in tensor {i}: output {output.shape} vs target {target.shape}. "
                                f"Using element-wise loss on shared dimensions."
                            )
                            # Find minimum dimensions
                            min_shape = [min(s1, s2) for s1, s2 in zip(output.shape, target.shape)]
                            
                            # Create slices for each dimension
                            slices_output = tuple(slice(0, s) for s in min_shape)
                            slices_target = tuple(slice(0, s) for s in min_shape)
                            
                            # Use matching portions of tensors
                            output_slice = output[slices_output]
                            target_slice = target[slices_target]
                            
                            # Calculate loss on matching portions
                            recon_loss += F.mse_loss(output_slice, target_slice, reduction='sum')
                            continue
                            
                    # Use mean squared error for reconstruction loss
                    recon_loss += F.mse_loss(output, target, reduction='sum')
                
                # Calculate KL divergence loss
                kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
                
                # Total loss with beta weighting for KL term
                loss = recon_loss + current_beta * kl_loss
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                # Accumulate losses
                train_total_loss += loss.item()
                train_recon_loss += recon_loss.item()
                train_kl_loss += kl_loss.item()
            
            # Normalize losses by dataset size
            train_total_loss /= train_size
            train_recon_loss /= train_size
            train_kl_loss /= train_size
            
            # Validation phase
            self.vae.eval()
            val_total_loss = 0
            val_recon_loss = 0
            val_kl_loss = 0
            
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(val_loader):
                    # Forward pass
                    outputs, z_mean, z_log_var, z = self.vae(batch_data)
                    
                    # Calculate reconstruction loss
                    recon_loss = 0
                    for i, (output, target) in enumerate(zip(outputs, batch_data)):
                        # Handle mismatched shapes the same way as in training
                        if output.shape != target.shape:
                            # Adjust output shape to match target shape if they're compatible
                            if output.numel() == target.numel():
                                output = output.view(target.shape)
                            else:
                                # Find minimum dimensions
                                min_shape = [min(s1, s2) for s1, s2 in zip(output.shape, target.shape)]
                                
                                # Create slices for each dimension
                                slices_output = tuple(slice(0, s) for s in min_shape)
                                slices_target = tuple(slice(0, s) for s in min_shape)
                                
                                # Use matching portions of tensors
                                output_slice = output[slices_output]
                                target_slice = target[slices_target]
                                
                                # Calculate loss on matching portions
                                recon_loss += F.mse_loss(output_slice, target_slice, reduction='sum')
                                continue
                        
                        recon_loss += F.mse_loss(output, target, reduction='sum')
                    
                    # Calculate KL divergence loss
                    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
                    
                    # Total loss with beta weighting for KL term
                    loss = recon_loss + current_beta * kl_loss
                    
                    # Accumulate losses
                    val_total_loss += loss.item()
                    val_recon_loss += recon_loss.item()
                    val_kl_loss += kl_loss.item()
                
                # Normalize losses by validation set size
                val_total_loss /= val_size
                val_recon_loss /= val_size
                val_kl_loss /= val_size
            
            # Log progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                self.logger.statement(
                    f"Epoch {epoch+1}/{epochs}, Beta: {current_beta:.4f}, "
                    f"Train: Loss={train_total_loss:.4f}, Recon={train_recon_loss:.4f}, KL={train_kl_loss:.4f}, "
                    f"Val: Loss={val_total_loss:.4f}, Recon={val_recon_loss:.4f}, KL={val_kl_loss:.4f}"
                )
            
            # Early stopping check
            if early_stopping:
                if val_total_loss < best_val_loss:
                    best_val_loss = val_total_loss
                    best_model_state = copy.deepcopy(self.vae.state_dict())
                    counter = 0
                else:
                    counter += 1
                    
                if counter >= patience:
                    self.logger.statement(f"Early stopping at epoch {epoch+1}")
                    # Restore best model
                    self.vae.load_state_dict(best_model_state)
                    break
            
            # Record history
            self.training_history['total_loss'].append(train_total_loss)
            self.training_history['reconstruction_loss'].append(train_recon_loss)
            self.training_history['kl_loss'].append(train_kl_loss)
            self.training_history['val_total_loss'].append(val_total_loss)
            self.training_history['val_reconstruction_loss'].append(val_recon_loss)
            self.training_history['val_kl_loss'].append(val_kl_loss)
        
        self.fitted = True
        
        if return_history:
            return self.training_history
        else:
            return self

    def encode(self, obs_data):
        """
        Encode observation data into the latent space.
        
        Parameters
        ----------
        obs_data : pandas.DataFrame
            Observation data to encode. Must have the same columns as the training data.
            
        Returns
        -------
        numpy.ndarray
            Latent space representation of the input data.
        """
        if self.vae is None:
            raise ValueError("Model has not been built yet. Call _build_model() first.")
        
        if not isinstance(obs_data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        # Check that all columns are present
        missing = [col for col in self.obs_names if col not in obs_data.columns]
        if missing:
            raise ValueError(f"Missing columns in observation data: {missing}")
        
        # Apply transformations if configured using the feature_transformer
        if hasattr(self, 'feature_transformer') and self.feature_transformer is not None:
            self.logger.statement("Applying transformations before encoding")
            # Use the transformer to transform the input data but only for the columns we need
            transformed_obs = self.feature_transformer.transform(obs_data[self.obs_names])
        else:
            transformed_obs = obs_data[self.obs_names].copy()
        
        # Center data using the training mean
        if self.obs_vec is None:
            raise ValueError("Mean observation vector is not available. Train the model first.")
            
        obs_centered = transformed_obs - self.obs_vec.values
        
        # Organize data types
        organized_data = self._organize_data_types(obs_centered)
        
        # Get tensor inputs
        tensor_inputs = organized_data['tensor_inputs']
        
        # Set model to evaluation mode
        self.vae.eval()
        self.encoder.eval()
        
        # Encode to latent space
        with torch.no_grad():
            z_mean, z_log_var, z = self.encoder(tensor_inputs)
            
        # Return as numpy array
        return z_mean.cpu().numpy(), z_log_var.cpu().numpy(), z.cpu().numpy()
        
    def decode(self, latent_vectors):
        """
        Decode latent space vectors back to observation space.
        
        Parameters
        ----------
        latent_vectors : numpy.ndarray or torch.Tensor
            Latent space vectors to decode.
            
        Returns
        -------
        pandas.DataFrame
            Reconstructed observations in the original data space.
        """
        if self.vae is None:
            raise ValueError("Model has not been built yet. Call _build_model() first.")
            
        # Convert to tensor if needed
        if isinstance(latent_vectors, np.ndarray):
            z = torch.tensor(latent_vectors, dtype=torch.float32, device=self.device)
        else:
            z = latent_vectors
            
        # Set model to evaluation mode
        self.vae.eval()
        self.decoder.eval()
        
        # Decode from latent space
        with torch.no_grad():
            outputs = self.decoder(z)
            
        # Initialize DataFrame for results
        result = pd.DataFrame(index=range(len(z)), columns=self.obs_names)
        
        # Process outputs and organize back into DataFrame
        col_idx = 0
        for i, (data_type, group_name, columns) in enumerate(self.organized_obs['column_groups']):
            output_tensor = outputs[i]
            
            # Reshape tensor based on data type
            if data_type == 'scalar':
                # Scalar data is already in the right shape
                output_values = output_tensor.cpu().numpy()
                for j, col in enumerate(columns):
                    result[col] = output_values[:, j]
                    
            elif data_type == 'timeseries':
                # Time series data needs to be flattened to columns
                output_values = output_tensor.cpu().numpy()
                num_samples = output_values.shape[0]
                time_steps = output_values.shape[1]
                features = output_values.shape[2] if output_values.ndim > 2 else 1
                
                # Check if we have the right number of elements
                expected_cols = len(columns)
                actual_cols = time_steps * features if output_values.ndim > 2 else time_steps
                
                if actual_cols != expected_cols:
                    self.logger.warn(
                        f"Time series dimension mismatch for group '{group_name}': "
                        f"expected {expected_cols} columns but got {actual_cols} values"
                    )
                
                # Reshape based on features dimension
                if features == 1 or output_values.ndim == 2:
                    # Single feature per time step case
                    output_flat = output_values.reshape((num_samples, -1))
                    # Ensure we have the right number of columns (truncate or pad if necessary)
                    if output_flat.shape[1] != expected_cols:
                        if output_flat.shape[1] > expected_cols:
                            output_flat = output_flat[:, :expected_cols]
                        else:
                            # Pad with zeros
                            pad_width = expected_cols - output_flat.shape[1]
                            output_flat = np.pad(output_flat, ((0, 0), (0, pad_width)))
                else:
                    # Multiple features case
                    output_flat = output_values.reshape((num_samples, -1))
                    if output_flat.shape[1] != expected_cols:
                        if output_flat.shape[1] > expected_cols:
                            output_flat = output_flat[:, :expected_cols]
                        else:
                            # Pad with zeros
                            pad_width = expected_cols - output_flat.shape[1]
                            output_flat = np.pad(output_flat, ((0, 0), (0, pad_width)))
                
                # Assign to DataFrame
                for j, col in enumerate(columns):
                    if j < output_flat.shape[1]:
                        result[col] = output_flat[:, j]
                        
            elif data_type == 'spatial':
                # Spatial data needs to be flattened to columns
                output_values = output_tensor.cpu().numpy()
                num_samples = output_values.shape[0]
                
                # Flatten spatial dimensions
                output_flat = output_values.reshape((num_samples, -1))
                
                # The number of flattened elements should match the number of columns
                if output_flat.shape[1] != len(columns):
                    self.logger.warn(
                        f"Mismatch between flattened spatial output ({output_flat.shape[1]}) "
                        f"and number of columns ({len(columns)})"
                    )
                    # Try to adjust if possible
                    if output_flat.shape[1] > len(columns):
                        output_flat = output_flat[:, :len(columns)]
                    else:
                        # Pad with zeros
                        pad_width = len(columns) - output_flat.shape[1]
                        output_flat = np.pad(output_flat, ((0, 0), (0, pad_width)))
                
                for j, col in enumerate(columns):
                    result[col] = output_flat[:, j]
        
        # Un-center data using the training mean
        result += self.obs_vec.values
        
        # Apply inverse transformations if configured
        if hasattr(self, 'feature_transformer') and self.feature_transformer is not None:
            self.logger.statement("Applying inverse transformations after decoding")
            # Use the transformer to inverse transform the result
            result = self.feature_transformer.inverse_on_external_df(result)
            
        return result
    
    def generate(self, n_samples=1, std_dev=1.0):
        """
        Generate new samples by sampling from the latent space.
        
        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Default is 1.
        std_dev : float, optional
            Standard deviation of the normal distribution to sample from.
            Controls the diversity of generated samples. Default is 1.0.
            
        Returns
        -------
        pandas.DataFrame
            Generated observation samples.
        """
        if self.vae is None:
            raise ValueError("Model has not been built yet. Call _build_model() first.")
            
        # Sample from normal distribution
        z = torch.randn(n_samples, self.latent_dim, device=self.device) * std_dev
        
        # Decode sampled latent vectors
        return self.decode(z)
    
    def configure_transformers(self, transforms):
        """
        Configure data transformation options for the emulator after initialization.
        
        This method allows updating transformation options without recreating the emulator,
        which is useful when you need to modify transformation settings for different scenarios.
        
        Parameters
        ----------
        transforms : list of dict
            List of transformations to apply to the data. Each dictionary should have:
            - 'type': str - Type of transformation (e.g., 'log10', 'normal_score')
            - 'columns': list - Columns to apply the transformation to (optional)
            - Additional kwargs specific to the transformer
            
        Returns
        -------
        self : DSIAE
            Returns self for method chaining.
        """
        self.transforms = transforms
        
        # If we already have data, reapply the transformations
        if self.sim_obs is not None:
            self.prepare_data(self.sim_obs, self.pars, self.obs_names, self.par_names)
            
            # Need to rebuild the model if it was already built
            if self.fitted:
                self._build_model()
                self.logger.statement("Model rebuilt to accommodate new transformations")
            
        return self









    import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Custom LSTM helper classes
class LSTMOutputAdapter(nn.Module):
    """Helper module to process LSTM outputs for encoder"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
    
    def forward(self, output, hidden):
        # Use the final time step output for each sequence
        return output[:, -1, :]

class TimeSeriesReshaper(nn.Module):
    """Reshape flat tensor to sequence for LSTM input"""
    def __init__(self, seq_len, hidden_dim):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        # Reshape from [batch, seq_len*hidden] to [batch, seq_len, hidden]
        return x.view(-1, self.seq_len, self.hidden_dim)

class TimeSeriesOutputProjector(nn.Module):
    """Project LSTM output to target sequence dimension"""
    def __init__(self, hidden_dim, out_features, seq_len):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, out_features)
        self.seq_len = seq_len
        self.out_features = out_features
    
    def forward(self, lstm_output):
        # lstm_output is tuple of (outputs, (h_n, c_n))
        # Get outputs which is [batch, seq_len, hidden]
        # Project to target feature dimension
        return self.proj(lstm_output[0])  # Shape: [batch, seq_len, out_features]
# Define LSTM-based Encoder to replace CNN encoder
class EncoderWithLSTM(nn.Module):
    """Encoder component that uses LSTM for time series data."""
    def __init__(self, input_shapes, latent_dim=8):
        super().__init__()
        
        self.input_shapes = input_shapes
        self.latent_dim = latent_dim
        
        # Branch processing networks
        self.scalar_net = None
        self.timeseries_nets = nn.ModuleList() if 'timeseries' in input_shapes else None
        self.spatial_nets = nn.ModuleList() if 'spatial' in input_shapes else None
        
        # Dimensions for each branch output
        self.scalar_dim = 0
        self.timeseries_dim = 0
        self.spatial_dim = 0
        
        # Build the scalar branch (unchanged)
        if 'scalar' in input_shapes and input_shapes['scalar'][0] > 0:
            scalar_input_dim = input_shapes['scalar'][0]
            self.scalar_dim = max(32, min(128, scalar_input_dim * 2))
            
            self.scalar_net = nn.Sequential(
                nn.Linear(scalar_input_dim, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, self.scalar_dim),
                nn.LeakyReLU(0.2),
            )
        
        # Build time series branches using LSTMs instead of CNNs
        if 'timeseries' in input_shapes and input_shapes['timeseries']:
            for i, shape in enumerate(input_shapes['timeseries']):
                seq_len, features = shape
                hidden_dim = 32  # Output dimension for each time series
                
                # Use bidirectional LSTM that can handle any sequence length
                lstm = nn.LSTM(
                    input_size=features, 
                    hidden_size=hidden_dim, 
                    num_layers=2,
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.1
                )
                
                # Create adapter to extract features from LSTM output
                adapter = LSTMOutputAdapter(hidden_dim * 2)  # *2 for bidirectional
                
                ts_net = nn.ModuleList([lstm, adapter])
                self.timeseries_nets.append(ts_net)
                self.timeseries_dim += hidden_dim * 2  # *2 for bidirectional LSTM
        
        # Build spatial branches (unchanged)
        if 'spatial' in input_shapes and input_shapes['spatial']:
            for i, shape in enumerate(input_shapes['spatial']):
                h, w, c = shape
                
                # For small images/grids, use simpler networks
                if h * w <= 100:  
                    spatial_net = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(h * w * c, 64),
                        nn.LeakyReLU(0.2),
                        nn.Linear(64, 32),
                        nn.LeakyReLU(0.2)
                    )
                else:
                    spatial_net = nn.Sequential(
                        nn.Conv2d(c, 16, kernel_size=3, padding=1),
                        nn.LeakyReLU(0.2),
                        nn.MaxPool2d(2),
                        nn.Conv2d(16, 32, kernel_size=3, padding=1),
                        nn.LeakyReLU(0.2),
                        nn.MaxPool2d(2),
                        nn.Flatten(),
                        nn.Linear(32 * (h // 4) * (w // 4), 32),
                        nn.LeakyReLU(0.2)
                    )
                    
                self.spatial_nets.append(spatial_net)
                self.spatial_dim += 32
        
        # Calculate total feature dimension after all branches
        self.total_features = self.scalar_dim + self.timeseries_dim + self.spatial_dim
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.total_features, 64),
            nn.LeakyReLU(0.2)
        )
        
        # VAE bottleneck outputs
        self.z_mean = nn.Linear(64, latent_dim)
        self.z_log_var = nn.Linear(64, latent_dim)
    
    def reparameterize(self, z_mean, z_log_var):
        """Reparameterization trick (unchanged)"""
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        z = z_mean + eps * std
        return z
    
    def forward(self, inputs):
        """Forward pass through the encoder"""
        features = []
        input_idx = 0
        
        # Process scalar inputs (unchanged)
        if self.scalar_net is not None:
            scalar_features = self.scalar_net(inputs[input_idx])
            features.append(scalar_features)
            input_idx += 1
        
        # Process time series inputs with LSTM
        if self.timeseries_nets is not None:
            for i, ts_net in enumerate(self.timeseries_nets):
                ts_input = inputs[input_idx]  # Already in shape [batch, seq_len, features]
                
                # Process through LSTM and adapter
                lstm, adapter = ts_net
                output, (h_n, c_n) = lstm(ts_input)
                ts_features = adapter(output, h_n)
                
                features.append(ts_features)
                input_idx += 1
        
        # Process spatial inputs (unchanged)
        if self.spatial_nets is not None:
            for i, spatial_net in enumerate(self.spatial_nets):
                spatial_input = inputs[input_idx]
                # Reshape if using 2D CNNs
                if isinstance(spatial_net[0], nn.Conv2d):
                    # Input shape should be [batch, channels, height, width]
                    spatial_input = spatial_input.permute(0, 3, 1, 2)
                spatial_features = spatial_net(spatial_input)
                features.append(spatial_features)
                input_idx += 1
        
        # Concatenate all features
        x = torch.cat(features, dim=1)
        
        # Apply fusion layer
        x = self.fusion(x)
        
        # VAE outputs
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.reparameterize(z_mean, z_log_var)
        
        return z_mean, z_log_var, z
# Define LSTM-based Decoder to replace CNN decoder
class DecoderWithLSTM(nn.Module):
    """Decoder component that uses LSTM for time series data."""
    def __init__(self, latent_dim, output_shapes):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_shapes = output_shapes
        
        # Latent to hidden expansion (unchanged)
        self.latent_expansion = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2)
        )
        
        # Branch processors
        self.scalar_net = None
        self.timeseries_nets = nn.ModuleList() if 'timeseries' in output_shapes else None
        self.spatial_nets = nn.ModuleList() if 'spatial' in output_shapes else None
        
        # Scalar branch (unchanged)
        if 'scalar' in output_shapes and output_shapes['scalar'][0] > 0:
            scalar_output_dim = output_shapes['scalar'][0]
            self.scalar_net = nn.Sequential(
                nn.Linear(64, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, scalar_output_dim)
            )
        
        # Time series branches using LSTMs
        if 'timeseries' in output_shapes and output_shapes['timeseries']:
            for i, shape in enumerate(output_shapes['timeseries']):
                seq_len, features = shape
                hidden_dim = 32
                
                # Sequential network for time series generation
                ts_net = nn.Sequential(
                    # First expand latent to features for each timestep
                    nn.Linear(64, hidden_dim * seq_len),
                    nn.LeakyReLU(0.2),
                    TimeSeriesReshaper(seq_len, hidden_dim),
                    
                    # Then use LSTM to generate the output sequence
                    nn.LSTM(
                        input_size=hidden_dim,
                        hidden_size=hidden_dim,
                        num_layers=2,
                        batch_first=True,
                        dropout=0.1
                    ),
                    
                    # Final projection to target dimensions
                    TimeSeriesOutputProjector(hidden_dim, features, seq_len)
                )
                
                self.timeseries_nets.append(ts_net)
        
        # Spatial branches (unchanged)
        if 'spatial' in output_shapes and output_shapes['spatial']:
            for i, shape in enumerate(output_shapes['spatial']):
                h, w, c = shape
                
                # For small images/grids, use simpler networks
                if h * w <= 100:  # e.g., 10x10 or smaller
                    spatial_net = nn.Sequential(
                        nn.Linear(64, 64),
                        nn.LeakyReLU(0.2),
                        nn.Linear(64, h * w * c)
                    )
                # For larger spatial data, use transposed convolutions
                else:
                    # Calculate intermediate dimensions
                    h_small = h // 4
                    w_small = w // 4
                    
                    spatial_net = nn.Sequential(
                        nn.Linear(64, 32 * h_small * w_small),
                        nn.LeakyReLU(0.2),
                        nn.Unflatten(1, (32, h_small, w_small)),
                        nn.Upsample(scale_factor=2),
                        nn.Conv2d(32, 16, kernel_size=3, padding=1),
                        nn.LeakyReLU(0.2),
                        nn.Upsample(scale_factor=2),
                        nn.Conv2d(16, c, kernel_size=3, padding=1)
                    )
                    
                self.spatial_nets.append(spatial_net)
                
    def forward(self, z):
        """Forward pass through the decoder"""
        # Expand latent vector to hidden representation
        x = self.latent_expansion(z)
        
        outputs = []
        
        # Decode scalar outputs (unchanged)
        if self.scalar_net is not None:
            scalar_output = self.scalar_net(x)
            outputs.append(scalar_output)
        
        # Decode time series outputs with LSTM
        if self.timeseries_nets is not None:
            for i, ts_net in enumerate(self.timeseries_nets):
                # Process through the sequential model
                ts_output = ts_net(x)
                outputs.append(ts_output)
        
        # Decode spatial outputs (unchanged)
        if self.spatial_nets is not None:
            for i, spatial_net in enumerate(self.spatial_nets):
                spatial_output = spatial_net(x)
                
                # Reshape output if using transposed convolutions
                if len(spatial_output.shape) == 4:  # [batch, channels, height, width]
                    # Change to [batch, height, width, channels]
                    spatial_output = spatial_output.permute(0, 2, 3, 1)
                else:
                    # Reshape to [batch, height, width, channels]
                    h, w, c = self.output_shapes['spatial'][i]
                    spatial_output = spatial_output.view(-1, h, w, c)
                    
                outputs.append(spatial_output)
        
        return outputs
# Create a full VAE with the LSTM-based encoder and decoder
class VariationalAutoEncoderWithLSTM(nn.Module):
    """Variational Autoencoder with LSTM for time series data."""
    def __init__(self, input_shapes, output_shapes, latent_dim=8):
        super().__init__()
        
        self.encoder = EncoderWithLSTM(input_shapes, latent_dim)
        self.decoder = DecoderWithLSTM(latent_dim, output_shapes)
        
    def forward(self, inputs):
        """Forward pass through the complete autoencoder"""
        # Encode inputs to latent space
        z_mean, z_log_var, z = self.encoder(inputs)
        
        # Decode latent vectors to outputs
        outputs = self.decoder(z)
        
        return outputs, z_mean, z_log_var, z
    

# Custom DSIAE class that uses LSTM for time series
class DSIAE_LSTM(DSIAE):
    """DSIAE emulator that uses recurrent neural networks for time series data."""
    
    def _build_model(self):
        """Override _build_model to use LSTM-based architecture"""
        if self.organized_obs is None:
            raise ValueError("No organized observation data available. Call prepare_data first.")
            
        # Get input and output shapes from organized data
        input_shapes = self.organized_obs['input_shapes']
        output_shapes = self.organized_obs['output_shapes']
        
        # Create encoder and decoder with LSTM components
        self.logger.statement(f"Building LSTM encoder with latent dim {self.latent_dim}")
        self.encoder = EncoderWithLSTM(input_shapes, latent_dim=self.latent_dim)
        
        self.logger.statement("Building LSTM decoder")
        self.decoder = DecoderWithLSTM(self.latent_dim, output_shapes)
        
        # Create complete VAE model
        self.vae = VariationalAutoEncoderWithLSTM(input_shapes, output_shapes, latent_dim=self.latent_dim)
        
        # Move models to the specified device
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.vae.to(self.device)
        
        # Set up optimizer
        self.optimizer = optim.Adam(self.vae.parameters(), lr=self.learning_rate)
        
        self.logger.statement("LSTM-based model built successfully")
        return