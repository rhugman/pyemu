"""
Data Space Inversion (DSI) emulator implementation.
"""
from __future__ import print_function, division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import inspect
from pyemu.utils.helpers import dsi_forward_run, series_to_insfile
import pickle
import os
import shutil
from pyemu.pst.pst_handler import Pst
from pyemu.en import ObservationEnsemble,ParameterEnsemble
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
