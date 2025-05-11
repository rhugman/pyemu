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
import os
import datetime

# Import tensorflow lazily to allow importing this module even if tf is not installed
# The actual import will occur when the DSIAE class is instantiated
def _import_tensorflow():
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        raise ImportError(
            "The DSIAE emulator requires TensorFlow, which is not installed. "
            "Install it with 'pip install pyemu[emulators-dsiae]' or 'pip install tensorflow'."
        )

from .base import Emulator

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
    pars : pandas.DataFrame
        Parameters data with columns as parameter names and rows as realizations.
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
    """

    def __init__(self, sim_obs=None, pars=None, obs_names=None, par_names=None, 
                 data_type_map=None, latent_dim=8, learning_rate=0.001, verbose=True):
        """
        Initialize the DSIAE emulator.

        Parameters
        ----------
        sim_obs : pandas.DataFrame
            Simulation observations data with columns as observation names and rows as realizations.
        pars : pandas.DataFrame
            Parameters data with columns as parameter names and rows as realizations.
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
        """
        # Import tensorflow here to avoid import error if not installed
        # This will raise a helpful error message if tensorflow is not available
        self.tf = _import_tensorflow()
        
        super().__init__(verbose=verbose)
        
        # Store input data
        self.sim_obs = sim_obs
        self.pars = pars
        
        # Set observation and parameter names
        if obs_names is None and sim_obs is not None:
            obs_names = sim_obs.columns.tolist()
        self.obs_names = obs_names
        
        if par_names is None and pars is not None:
            par_names = pars.columns.tolist()
        self.par_names = par_names
        
        # Data organization
        self.data_type_map = data_type_map or {}  # Default to empty dict if None
        
        # Model parameters
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        
        # State variables
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.obs_vec = None  # Mean of observations
        self.par_vec = None  # Mean of parameters
        self.kdtree = None
        self.min_max = {}
        self.weights = None
        self.use_localizer = False
        self.fitted = False
        
    def _organize_data_types(self, data):
        """
        Organize data into different types based on data_type_map.
        
        The data_type_map must be provided in the following structure:
        {
            'scalar': [column names],  # All scalar columns
            'timeseries': {
                'group1': {
                    'columns': [column names],  # Ordered list of column names
                    'coordinates': [t1, t2, ...],  # Corresponding time coordinates 
                    'frequency': '1D',  # Optional for uniform time steps
                    'irregular': False  # Optional flag for irregular time series
                },
                'group2': {...},
                ...
            },
            'spatial': {
                'group1': {
                    'columns': [column names],  # Ordered list of column names
                    'coordinates': [(x1,y1,z1), (x2,y2,z2), ...],  # Corresponding spatial coordinates
                    'dimensions': (nx, ny, nz),  # Optional grid dimensions
                    'spacing': (dx, dy, dz)  # Optional grid spacing
                },
                'group2': {...},
                ...
            }
        }
        
        For both timeseries and spatial data, the coordinates list must be the same length as the columns list.
        Time series coordinates can be numeric values representing time steps or points.
        Spatial coordinates should be tuples of 2D (x,y) or 3D (x,y,z) coordinates.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Data to organize.
            
        Returns
        -------
        dict
            Dictionary with the following structure:
            {
                'scalar': pandas.DataFrame,  # All scalar columns
                'timeseries': {
                    'group1': {
                        'data': pandas.DataFrame,  # Columns for first time series
                        'coordinates': [t1, t2, ...],  # Corresponding time coordinates
                        'metadata': {}  # Additional metadata like frequency, etc.
                    },
                    'group2': {...},
                    ...
                },
                'spatial': {
                    'group1': {
                        'data': pandas.DataFrame,  # Columns for first spatial dataset
                        'coordinates': [(x1,y1,z1), (x2,y2,z2), ...],  # Corresponding spatial coordinates
                        'metadata': {}  # Additional metadata like dimensions, spacing, etc.
                    },
                    'group2': {...},
                    ...
                }
            }
        """
        if not self.data_type_map:
            # If no data type map is provided, treat all as scalar
            self.logger.statement("No data_type_map provided, treating all columns as scalar")
            return {'scalar': data, 'timeseries': {}, 'spatial': {}}
        
        # Initialize result structure
        result = {
            'scalar': pd.DataFrame(index=data.index),
            'timeseries': {},
            'spatial': {}
        }
        
        # Keep track of processed columns to identify any missing ones
        processed_columns = set()
        
        # Process scalar columns
        if 'scalar' in self.data_type_map and self.data_type_map['scalar']:
            scalar_cols = [col for col in self.data_type_map['scalar'] if col in data.columns]
            if scalar_cols:
                result['scalar'] = data[scalar_cols].copy()
                processed_columns.update(scalar_cols)
            else:
                self.logger.statement("No valid scalar columns found in data_type_map")
        
        # Process time series data with new structure
        if 'timeseries' in self.data_type_map and self.data_type_map['timeseries']:
            for group, group_info in self.data_type_map['timeseries'].items():
                # Check if the group_info is a dict (new format) or list/array (old format)
                if isinstance(group_info, dict):
                    # New format with coordinates
                    if 'columns' not in group_info:
                        self.logger.warn(f"No 'columns' field found in timeseries group '{group}'")
                        continue

                    # Get columns and check which ones exist in the data
                    columns = group_info['columns']
                    valid_cols = [col for col in columns if col in data.columns]
                    
                    if not valid_cols:
                        self.logger.warn(f"No valid columns found for timeseries group '{group}'")
                        continue
                        
                    # Extract coordinates if available
                    coordinates = group_info.get('coordinates', None)
                    
                    if coordinates is not None:
                        # Validate coordinates length matches columns length
                        if len(coordinates) != len(valid_cols):
                            self.logger.warn(
                                f"Length of coordinates ({len(coordinates)}) "
                                f"doesn't match length of valid columns ({len(valid_cols)}) "
                                f"for timeseries group '{group}'"
                            )
                            # Use only the coordinates that match valid columns
                            coordinates = coordinates[:len(valid_cols)]
                    
                    # Extract additional metadata
                    metadata = {
                        k: v for k, v in group_info.items() 
                        if k not in ('columns', 'coordinates')
                    }
                    
                    # Store data, coordinates and metadata
                    result['timeseries'][group] = {
                        'data': data[valid_cols].copy(),
                        'coordinates': coordinates,
                        'metadata': metadata
                    }
                    
                else:
                    # Old format (list of columns)
                    columns = group_info
                    valid_cols = [col for col in columns if col in data.columns]
                    
                    if not valid_cols:
                        self.logger.warn(f"No valid columns found for timeseries group '{group}'")
                        continue
                        
                    # Store data with no coordinates or metadata
                    result['timeseries'][group] = {
                        'data': data[valid_cols].copy(),
                        'coordinates': None,
                        'metadata': {}
                    }
                    
                # Mark columns as processed
                processed_columns.update(valid_cols)
        
        # Process spatial data with new structure
        if 'spatial' in self.data_type_map and self.data_type_map['spatial']:
            for group, group_info in self.data_type_map['spatial'].items():
                # Check if the group_info is a dict (new format) or list/array (old format)
                if isinstance(group_info, dict):
                    # New format with coordinates
                    if 'columns' not in group_info:
                        self.logger.warn(f"No 'columns' field found in spatial group '{group}'")
                        continue
                        
                    # Get columns and check which ones exist in the data
                    columns = group_info['columns']
                    valid_cols = [col for col in columns if col in data.columns]
                    
                    if not valid_cols:
                        self.logger.warn(f"No valid columns found for spatial group '{group}'")
                        continue
                        
                    # Extract coordinates if available
                    coordinates = group_info.get('coordinates', None)
                    
                    if coordinates is not None:
                        # Validate coordinates length matches columns length
                        if len(coordinates) != len(valid_cols):
                            self.logger.warn(
                                f"Length of coordinates ({len(coordinates)}) "
                                f"doesn't match length of valid columns ({len(valid_cols)}) "
                                f"for spatial group '{group}'"
                            )
                            # Use only the coordinates that match valid columns
                            coordinates = coordinates[:len(valid_cols)]
                    
                    # Extract additional metadata
                    metadata = {
                        k: v for k, v in group_info.items() 
                        if k not in ('columns', 'coordinates')
                    }
                    
                    # Store data, coordinates and metadata
                    result['spatial'][group] = {
                        'data': data[valid_cols].copy(),
                        'coordinates': coordinates,
                        'metadata': metadata
                    }
                    
                else:
                    # Old format (list of columns)
                    columns = group_info
                    valid_cols = [col for col in columns if col in data.columns]
                    
                    if not valid_cols:
                        self.logger.warn(f"No valid columns found for spatial group '{group}'")
                        continue
                        
                    # Store data with no coordinates or metadata
                    result['spatial'][group] = {
                        'data': data[valid_cols].copy(),
                        'coordinates': None,
                        'metadata': {}
                    }
                    
                # Mark columns as processed
                processed_columns.update(valid_cols)
                
        # Check if any columns weren't assigned to a data type
        missing_cols = set(data.columns) - processed_columns
        if missing_cols:
            self.logger.warn(f"The following columns were not assigned to any data type and will be treated as scalar: {missing_cols}")
            if result['scalar'].empty:
                result['scalar'] = data[list(missing_cols)].copy()
            else:
                for col in missing_cols:
                    result['scalar'][col] = data[col]
        
        return result

    def predict(self, pars, nsamples=0, return_variance=False):
        """
        Predict observations from parameter values using the DSIAE emulator.
        
        Parameters
        ----------
        pars : pandas.DataFrame or pandas.Series
            Parameter values to predict from.
        nsamples : int, optional
            Number of samples to draw for stochastic simulation. If 0, returns the mean prediction.
        return_variance : bool, optional
            If True, also return variance of the predictions. Default is False.
            
        Returns
        -------
        pandas.DataFrame
            Predicted observations for the input parameters.
        pandas.DataFrame, optional
            Variance of the predictions, only if return_variance is True.
        """
        if not self.fitted:
            raise ValueError("Emulator must be fitted before prediction")
        
        # Handle both DataFrame and Series inputs
        if isinstance(pars, pd.Series):
            pars = pars.to_frame().T
        
        # Ensure we have the right parameters
        if self.par_names is not None:
            missing_pars = set(self.par_names) - set(pars.columns)
            if missing_pars:
                raise ValueError(f"Missing parameters: {missing_pars}")
            pars = pars[self.par_names]
            
        # Check if parameters are within training range
        for par_name in pars.columns:
            if par_name in self.min_max:
                min_val, max_val = self.min_max[par_name]
                if (pars[par_name] < min_val).any() or (pars[par_name] > max_val).any():
                    self.logger.warn(f"Parameter {par_name} includes values outside the training range")
        
        # Find nearest neighbors in parameter space
        distances, indices = self.kdtree.query(pars.values, k=5)  # Get 5 nearest neighbors
        
        # Get observations for nearest neighbors
        nearest_obs = self.sim_vals.iloc[indices[:, 0]].copy()
        nearest_obs.index = pars.index  # Match indices with input parameters
        
        # Calculate deviations from mean observation vector for nearest neighbors
        nearest_obs_dev = nearest_obs - self.obs_vec.values
        
        # Organize nearest observations by data type
        organized_nearest_obs = self._organize_data_types(nearest_obs_dev)
        
        # Prepare inputs for the autoencoder (encoder)
        encoder_inputs = []
        
        # Add scalar input if needed
        if 'scalar' in organized_nearest_obs and not organized_nearest_obs['scalar'].empty:
            encoder_inputs.append(organized_nearest_obs['scalar'].values)
            
        # Add time series inputs with coordinates
        if 'timeseries' in organized_nearest_obs:
            for group, group_data in organized_nearest_obs['timeseries'].items():
                encoder_inputs.append(group_data['data'].values)
                
        # Add spatial inputs with coordinates
        if 'spatial' in organized_nearest_obs:
            for group, group_data in organized_nearest_obs['spatial'].items():
                encoder_inputs.append(group_data['data'].values)
        
        # Encode to get the latent representation
        self.logger.statement("Encoding nearest neighbor observations to latent space")
        latent_mean, latent_log_var, latent_sample = self.encoder.predict(
            encoder_inputs, 
            verbose=0
        )
        
        # If requesting samples, generate multiple latent codes
        if nsamples > 0:
            self.logger.statement(f"Generating {nsamples} samples from latent space")
            latent_samples = []
            
            # Generate samples from the latent distribution
            for _ in range(nsamples):
                # Sample from the latent space using mean and log variance
                epsilon = np.random.normal(size=latent_mean.shape)
                z_sample = latent_mean + np.exp(0.5 * latent_log_var) * epsilon
                latent_samples.append(z_sample)
                
            # Decode each sample
            decoded_samples = []
            for z_sample in latent_samples:
                decoder_outputs = self.decoder.predict(z_sample, verbose=0)
                
                # Combine outputs into a single DataFrame
                decoded_df = self._combine_decoder_outputs(decoder_outputs)
                
                # Add the mean observations to get the final predictions
                pred_sample = decoded_df + self.obs_vec.values
                
                decoded_samples.append(pred_sample)
                
            # Return either a single DataFrame or a list, depending on number of samples
            if len(decoded_samples) == 1:
                return decoded_samples[0]
            else:
                return pd.concat(decoded_samples, axis=0)
        
        # For normal prediction without sampling, use the mean latent representation
        self.logger.statement("Decoding from latent space to observation space")
        decoder_outputs = self.decoder.predict(latent_mean, verbose=0)
        
        # Combine the decoder outputs into a single DataFrame
        decoded_df = self._combine_decoder_outputs(decoder_outputs)
        
        # Add the mean observations to get the final predictions
        pred = decoded_df + self.obs_vec.values
        
        if return_variance:
            # If variance requested, compute variance based on latent space uncertainty
            self.logger.statement("Computing prediction variance")
            # Sample multiple times from the latent distribution and compute variance
            n_var_samples = 10
            var_samples = []
            
            for _ in range(n_var_samples):
                # Sample from the latent space
                epsilon = np.random.normal(size=latent_mean.shape)
                z_sample = latent_mean + np.exp(0.5 * latent_log_var) * epsilon
                
                # Decode this sample
                var_decoder_outputs = self.decoder.predict(z_sample, verbose=0)
                var_decoded_df = self._combine_decoder_outputs(var_decoder_outputs)
                var_pred = var_decoded_df + self.obs_vec.values
                
                var_samples.append(var_pred)
                
            # Compute variance across samples
            var_stack = np.stack([df.values for df in var_samples], axis=0)
            variance = pd.DataFrame(
                np.var(var_stack, axis=0),
                index=pred.index,
                columns=pred.columns
            )
            
            return pred, variance
        else:
            return pred
            
    def _combine_decoder_outputs(self, decoder_outputs):
        """
        Combine multiple decoder outputs into a single DataFrame.
        
        Parameters
        ----------
        decoder_outputs : list
            List of numpy arrays from the decoder model outputs.
            
        Returns
        -------
        pandas.DataFrame
            Combined DataFrame of all decoder outputs.
        """
        combined_columns = []
        combined_data = None
        output_idx = 0
        
        # Handle scalar outputs
        if 'scalar' in self.organized_obs and not self.organized_obs['scalar'].empty:
            scalar_data = decoder_outputs[output_idx]
            scalar_cols = self.organized_obs['scalar'].columns
            
            if combined_data is None:
                combined_data = scalar_data
                combined_columns.extend(scalar_cols)
            else:
                # This branch shouldn't normally execute for the first output
                combined_data = np.concatenate((combined_data, scalar_data), axis=1)
                combined_columns.extend(scalar_cols)
                
            output_idx += 1
            
        # Handle time series outputs with coordinates
        if 'timeseries' in self.organized_obs:
            for group, group_data in self.organized_obs['timeseries'].items():
                ts_data = decoder_outputs[output_idx]
                ts_cols = group_data['data'].columns
                
                if combined_data is None:
                    combined_data = ts_data
                    combined_columns.extend(ts_cols)
                else:
                    combined_data = np.concatenate((combined_data, ts_data), axis=1)
                    combined_columns.extend(ts_cols)
                    
                output_idx += 1
                
        # Handle spatial outputs with coordinates
        if 'spatial' in self.organized_obs:
            for group, group_data in self.organized_obs['spatial'].items():
                spatial_data = decoder_outputs[output_idx]
                spatial_cols = group_data['data'].columns
                
                if combined_data is None:
                    combined_data = spatial_data
                    combined_columns.extend(spatial_cols)
                else:
                    combined_data = np.concatenate((combined_data, spatial_data), axis=1)
                    combined_columns.extend(spatial_cols)
                    
                output_idx += 1
                
        # Create DataFrame with the right column order
        result = pd.DataFrame(
            combined_data,
            columns=combined_columns
        )
        
        # Ensure output columns match the expected order
        if self.obs_names:
            if not all(col in result.columns for col in self.obs_names):
                missing = [col for col in self.obs_names if col not in result.columns]
                self.logger.warn(f"Some observation columns are missing in decoder output: {missing}")
            
            # Only include columns that actually exist in the result
            valid_obs_names = [col for col in self.obs_names if col in result.columns]
            result = result[valid_obs_names]
        
        return result

    def encode(self, data):
        """
        Map data to latent space representation.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Data to encode. Must contain columns matching those used during training.
            
        Returns
        -------
        tuple
            (z_mean, z_log_var, z_sample) - latent space representations:
            z_mean: mean values of the latent space
            z_log_var: log variance values of the latent space
            z_sample: sampled latent vectors
        """
        if not self.fitted or self.encoder is None:
            raise ValueError("Emulator must be fitted before encoding")
            
        # Center the data using the mean observation vector
        data_centered = data - self.obs_vec.values
        
        # Organize data by type
        organized_data = self._organize_data_types(data_centered)
        
        # Prepare encoder inputs
        encoder_inputs = []
        
        # Add scalar input
        if 'scalar' in organized_data and not organized_data['scalar'].empty:
            encoder_inputs.append(organized_data['scalar'].values)
        
        # Add time series inputs
        if 'timeseries' in organized_data:
            for group, group_data in organized_data['timeseries'].items():
                encoder_inputs.append(group_data['data'].values)
        
        # Add spatial inputs
        if 'spatial' in organized_data:
            for group, group_data in organized_data['spatial'].items():
                encoder_inputs.append(group_data['data'].values)
        
        # Encode to latent space
        z_mean, z_log_var, z_sample = self.encoder.predict(encoder_inputs, verbose=0)
        
        return z_mean, z_log_var, z_sample
        
    def decode(self, latent_vectors):
        """
        Map from latent space back to data space.
        
        Parameters
        ----------
        latent_vectors : numpy.ndarray
            Latent space vectors to decode.
            
        Returns
        -------
        pandas.DataFrame
            Decoded data with original column names.
        """
        if not self.fitted or self.decoder is None:
            raise ValueError("Emulator must be fitted before decoding")
            
        # Generate predictions from latent vectors
        decoder_outputs = self.decoder.predict(latent_vectors, verbose=0)
        
        # Combine outputs into a single DataFrame
        decoded_df = self._combine_decoder_outputs(decoder_outputs)
        
        # Add the mean observations to get the final values
        result = decoded_df + self.obs_vec.values
        
        return result
        
    def plot_latent_space(self, n_samples=1000, dimensions=(0, 1), figsize=(10, 8)):
        """
        Visualize the latent space of the autoencoder by projecting to 2D.
        
        Parameters
        ----------
        n_samples : int, optional
            Number of samples to use for visualization. Default is 1000.
        dimensions : tuple, optional
            The latent space dimensions to plot (as x, y). Default is (0, 1).
        figsize : tuple, optional
            Figure size. Default is (10, 8).
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        if not self.fitted or self.encoder is None:
            raise ValueError("Emulator must be fitted before plotting latent space")
            
        # Limit number of samples to available data
        n_samples = min(n_samples, len(self.sim_vals))
        
        # Get a sample of data
        sample_indices = np.random.choice(len(self.sim_vals), n_samples, replace=False)
        sample_data = self.sim_vals.iloc[sample_indices]
        
        # Encode to latent space
        z_mean, z_log_var, z = self.encode(sample_data)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        scatter = ax.scatter(
            z_mean[:, dimensions[0]], 
            z_mean[:, dimensions[1]], 
            alpha=0.7, 
            s=30
        )
        
        # Add axis labels
        ax.set_title('Latent Space Visualization')
        ax.set_xlabel(f'Latent Dimension {dimensions[0]}')
        ax.set_ylabel(f'Latent Dimension {dimensions[1]}')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add uncertainty ellipses for a subset of points
        if n_samples > 50:
            subset_size = min(50, n_samples)
            subset_indices = np.random.choice(n_samples, subset_size, replace=False)
            
            for idx in subset_indices:
                # Get latent mean and variance for this point
                center = z_mean[idx, dimensions]
                # Convert log variance to standard deviation
                std = np.exp(0.5 * z_log_var[idx, dimensions])
                
                # Create an ellipse to represent uncertainty
                ellipse = plt.matplotlib.patches.Ellipse(
                    xy=center,
                    width=std[0]*2, height=std[1]*2,
                    alpha=0.2, color='red'
                )
                ax.add_patch(ellipse)
        
        self.logger.statement("Plotting latent space dimensions "
                             f"{dimensions[0]} vs {dimensions[1]}")
        return fig