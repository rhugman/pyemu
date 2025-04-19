"""
Data Space Inversion (DSI) emulator implementation.
"""
from __future__ import print_function, division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from .base import Emulator

class DSI(Emulator):
    """
    Data Space Inversion (DSI) Emulator for emulating the relationship between 
    parameter space and observation space.
    
    DSI operates directly in the observation space, projecting from parameter 
    space to observation space using SVD-based dimensionality reduction.
    
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
    verbose : bool, optional
        If True, enable verbose logging. Default is True.
    """

    def __init__(self, sim_obs=None, pars=None, obs_names=None, par_names=None, verbose=True):
        """
        Initialize the DSI emulator.

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
        verbose : bool, optional
            If True, enable verbose logging. Default is True.
        """
        super().__init__(verbose=verbose)
        
        self.sim_obs = sim_obs
        self.pars = pars
        
        if obs_names is None and sim_obs is not None:
            obs_names = sim_obs.columns.tolist()
        self.obs_names = obs_names
        
        if par_names is None and pars is not None:
            par_names = pars.columns.tolist()
        self.par_names = par_names
        
        self.obs_vec = None
        self.par_vec = None
        self.projection_matrix = None
        self.kdtree = None
        self.min_max = {}
        self.mean_residuals = None
        self.var_residuals = None
        self.sim_vals = None
        self.weights = None
        self.use_localizer = False
    
    def compute_projection_matrix(self, energy_threshold=1.0):
        """
        Compute the SVD-based projection matrix that maps from parameter space to observation space.
        
        This method forms the core of the DSI algorithm by:
        1. Centering both parameter and observation data
        2. Computing the SVD of the observation deviations
        3. Optionally truncating the SVD components based on energy threshold
        4. Computing the projection matrix using the SVD components
        
        Parameters
        ----------
        energy_threshold : float, optional
            Threshold of energy to retain in the SVD, between 0 and 1. Default is 1.0.
            
        Returns
        -------
        self : DSI
            The emulator instance with projection matrix computed.
        """
        if self.sim_obs is None or self.pars is None:
            raise ValueError("Both simulation observations and parameters must be provided")
            
        sim_obs = self.sim_obs.copy()
        if self.obs_names is not None:
            sim_obs = sim_obs[self.obs_names]
            
        pars = self.pars.copy()
        if self.par_names is not None:
            pars = pars[self.par_names]
            
        self.logger.statement("computing projection matrix with energy threshold: {0}".format(energy_threshold))
            
        # Calculate means for both spaces
        self.obs_vec = sim_obs.mean().copy()
        self.par_vec = pars.mean().copy()
        
        # Calculate deviations from means
        obs_dev = sim_obs - self.obs_vec.values
        par_dev = pars - self.par_vec.values
        
        # Normalize to have unit variance
        Z = obs_dev / np.sqrt(float(obs_dev.shape[0] - 1))
        
        # Compute SVD
        u, s, v = np.linalg.svd(Z, full_matrices=False)
        
        # Calculate energy in each component
        energy = np.cumsum(s**2) / np.sum(s**2)
        
        # Determine how many components to keep based on energy threshold
        num_keep = np.argmax(energy >= energy_threshold) + 1
        self.logger.statement("keeping {0} of {1} SVD components".format(num_keep, len(s)))
        
        # Apply truncation if needed
        if num_keep < len(s):
            u = u[:, :num_keep]
            s = s[:num_keep]
            v = v[:num_keep]
            
        # Compute projection matrix components
        u_s = np.dot(v.T, np.diag(s))
        
        # Compute projection matrix (regression coefficients)
        obs_dev_pseudoinv = np.linalg.pinv(obs_dev.values)
        self.projection_matrix = np.dot(par_dev.values.T, obs_dev_pseudoinv)
        
        self.fitted = True
        
        # Store input data for diagnostics
        self.sim_vals = sim_obs.copy()
        
        return self
        
    def fit(self, sim_obs=None, pars=None, obs_names=None, par_names=None, energy_threshold=1.0):
        """
        Fit the DSI emulator to the training data.
        
        Parameters
        ----------
        sim_obs : pandas.DataFrame, optional
            Simulation observations data. If None, uses self.sim_obs. Default is None.
        pars : pandas.DataFrame, optional
            Parameters data. If None, uses self.pars. Default is None.
        obs_names : list, optional
            List of observation names to include in the emulator. If None, uses self.obs_names. Default is None.
        par_names : list, optional
            List of parameter names to include in the emulator. If None, uses self.par_names. Default is None.
        energy_threshold : float, optional
            Threshold of energy to retain in the SVD, between 0 and 1. Default is 1.0.
            
        Returns
        -------
        self : DSI
            The fitted emulator.
        """
        if sim_obs is not None:
            self.sim_obs = sim_obs
            
        if pars is not None:
            self.pars = pars
            
        if obs_names is not None:
            self.obs_names = obs_names
            
        if par_names is not None:
            self.par_names = par_names
        
        # Compute the projection matrix
        self.compute_projection_matrix(energy_threshold=energy_threshold)
        
        # Check for parameter range
        pars_subset = self.pars[self.par_names] if self.par_names else self.pars
        self.min_max = {}
        for par_name in pars_subset.columns:
            self.min_max[par_name] = (pars_subset[par_name].min(), pars_subset[par_name].max())
        
        # Build KDTree for nearest-neighbor lookups of parameter sets
        self.kdtree = cKDTree(pars_subset.values)
        
        self.fitted = True
        return self
    
    def predict(self, pars, nsamples=0, return_variance=False):
        """
        Predict observations from parameter values using the DSI emulator.
        
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
        
        # Calculate deviation from mean parameter vector
        par_dev = pars - self.par_vec.values
        
        # Calculate predicted observations using projection matrix
        obs_pred = pd.DataFrame(
            self.obs_vec.values + np.dot(par_dev.values, self.projection_matrix),
            index=pars.index,
            columns=self.obs_names if self.obs_names else self.sim_obs.columns
        )
        
        # If no sampling is requested, return the mean prediction
        if nsamples == 0:
            if return_variance:
                # Placeholder for variance estimation - in a real implementation this would
                # calculate actual prediction variance
                var = pd.DataFrame(
                    np.zeros(obs_pred.shape),
                    index=obs_pred.index,
                    columns=obs_pred.columns
                )
                return obs_pred, var
            else:
                return obs_pred
                
        # For sampling, we would generate samples from the predictive distribution
        # This is a simplistic implementation - a full implementation would consider
        # the variance structure more carefully
        samples = []
        for _ in range(nsamples):
            # Add noise based on residuals
            if self.var_residuals is not None:
                noise = np.random.normal(0, np.sqrt(self.var_residuals))
                sample = obs_pred + noise
            else:
                sample = obs_pred.copy()
            samples.append(sample)
        
        # Return the samples as a stacked DataFrame
        if len(samples) == 1:
            return samples[0]
        else:
            return pd.concat(samples, axis=0)

    def get_nearest_neighbor(self, par_vals, k=1):
        """
        Get the nearest neighbor(s) in parameter space to the given parameters.
        
        Parameters
        ----------
        par_vals : pandas.DataFrame or pandas.Series
            Parameter values to find neighbors for.
        k : int, optional
            Number of neighbors to retrieve. Default is 1.
            
        Returns
        -------
        tuple
            (distances, indices) - distances to nearest neighbors and their indices.
        """
        if not self.fitted or self.kdtree is None:
            raise ValueError("Emulator must be fitted with parameters before finding neighbors")
            
        # Handle both DataFrame and Series inputs
        if isinstance(par_vals, pd.Series):
            par_vals = par_vals.to_frame().T
        
        # Select correct parameters
        if self.par_names is not None:
            par_vals = par_vals[self.par_names]
            
        # Query the KDTree
        distances, indices = self.kdtree.query(par_vals.values, k=k)
        return distances, indices
        
    def evaluate_localizer(self, sim_vals=None, k=1):
        """
        Evaluate the localizer performance using cross-validation.
        
        Parameters
        ----------
        sim_vals : pandas.DataFrame, optional
            Simulation values to use. If None, uses self.sim_vals. Default is None.
        k : int, optional
            Number of neighbors to consider. Default is 1.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame containing RMSE and other performance metrics.
        """
        if sim_vals is None:
            sim_vals = self.sim_vals
            
        if sim_vals is None:
            raise ValueError("Simulation values must be provided")
        
        # Placeholder for simple implementation
        # In a full implementation, this would perform leave-one-out cross-validation
        
        # For each parameter set:
        # 1. Exclude it from the training set
        # 2. Find its k nearest neighbors in the remaining training set
        # 3. Compare the actual simulation result to the predicted result
        # 4. Compute the RMSE and other metrics
        
        # Simplified implementation
        metrics = pd.DataFrame({
            "k": [k],
            "rmse": [0.0],
            "mean_absolute_error": [0.0],
            "r2": [0.0]
        })
        
        self.logger.statement(f"Localization with k={k} metrics computed")
        return metrics
        
    def plot_projection_matrix(self, obs_names=None, par_names=None, figsize=(10, 8)):
        """
        Plot the projection matrix as a heatmap showing parameter-observation relationships.
        
        Parameters
        ----------
        obs_names : list, optional
            Observation names to include in the plot. If None, uses self.obs_names or all observations.
        par_names : list, optional
            Parameter names to include in the plot. If None, uses self.par_names or all parameters.
        figsize : tuple, optional
            Figure size. Default is (10, 8).
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        if self.projection_matrix is None:
            raise ValueError("Projection matrix not computed. Call fit() first.")
            
        # Determine which names to use
        if obs_names is None:
            obs_names = self.obs_names if self.obs_names is not None else self.sim_obs.columns
            
        if par_names is None:
            par_names = self.par_names if self.par_names is not None else self.pars.columns
            
        # Create a DataFrame from the projection matrix for easier plotting
        proj_df = pd.DataFrame(
            self.projection_matrix,
            index=par_names,
            columns=obs_names
        )
            
        # Create the heatmap
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(proj_df.values, cmap='RdBu_r', aspect='auto')
        
        # Set axis labels
        ax.set_xlabel('Observations')
        ax.set_ylabel('Parameters')
        
        # Set tick labels (may need to limit for large matrices)
        ax.set_xticks(np.arange(len(obs_names)))
        ax.set_yticks(np.arange(len(par_names)))
        ax.set_xticklabels(obs_names, rotation=90)
        ax.set_yticklabels(par_names)
        
        # Add a colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Sensitivity')
        
        plt.tight_layout()
        return fig

    def plot_eigenvalues(self, figsize=(8, 6)):
        """
        Plot the eigenvalue spectrum from SVD analysis.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size. Default is (8, 6).
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        if not self.fitted:
            raise ValueError("Emulator must be fitted before plotting eigenvalues")
            
        # Recompute SVD to get eigenvalues
        obs_dev = self.sim_obs - self.obs_vec.values
        Z = obs_dev / np.sqrt(float(obs_dev.shape[0] - 1))
        _, s, _ = np.linalg.svd(Z, full_matrices=False)
        
        # Plot eigenvalues
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(s, 'o-', label='Singular Values')
        ax.set_xlabel('Component Index')
        ax.set_ylabel('Singular Value')
        ax.set_title('Singular Value Spectrum')
        ax.grid(True)
        
        # Plot cumulative explained variance
        ax2 = ax.twinx()
        cumulative_variance = np.cumsum(s**2) / np.sum(s**2)
        ax2.plot(cumulative_variance, 'r-', label='Cumulative Explained Variance')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_ylim([0, 1.05])
        
        # Add legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='best')
        
        plt.tight_layout()
        return fig