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

class Emulator:
    """
    Base class for emulators.
    
    This class serves as the foundation for various emulator implementations such as
    Data Space Inversion (DSI), Gaussian Process Regression (GPR), etc.
    It provides common functionality and interfaces for all emulator types.
    
    Parameters
    ----------
    verbose : bool, optional
        If True, enable verbose logging. Default is False.
    """

    def __init__(self,verbose=False):
        """
        Initialize the Emulator class.

        Parameters
        ----------
        verbose : bool, optional
            If True, enable verbose logging. Default is False.
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

            self.__org_observation_data = pst.observation_data.copy()
            self.__org_parameter_data = pst.parameter_data.copy()
            #self.__org_control_data = pst.control_data.copy() #breaks pickling
            if isinstance(sim_ensemble, ObservationEnsemble):
                sim_ensemble = sim_ensemble._df.copy()
            self.__org_sim_ensemble = sim_ensemble.copy()
            self.data = sim_ensemble.copy()
            self.data_transformed = None
            self.feature_scaler = None
            self.energy_threshold = energy_threshold
            if log_transform is True:
                self.log_transform = sim_ensemble.columns.tolist()
            else:
                self.log_transform = log_transform
            assert isinstance(self.log_transform, (bool, list)), "log_transform must be a boolean or a list of column names"
            self.normal_score_transform = normal_score_transform
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
        


        def predict(self,pvals):
            if isinstance(pvals, pd.Series):
                pvals = pvals.values.flatten()
            assert pvals.shape[0] == self.s.shape[0], "pvals must be the same length as the number of singular values"
            assert pvals.shape[0] == self.pmat.shape[1], "pvals must be the same length as the number of singular values"
            pmat = self.pmat
            ovals = self.ovals
            sim_vals = ovals + np.dot(pmat,pvals)
            ft = self.feature_transformer
            sim_vals = ft.inverse_on_external_df(sim_vals, columns=self.data_transformed.columns.tolist())
            sim_vals.index.name = 'obsnme'
            sim_vals.name = "obsval"
            self.sim_vals = sim_vals
            return sim_vals
        
        def check_for_pdc():
            #TODO
            return
            

        def prepare_pestpp(self,t_d=None,observation_data=None):
            
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
                file.write(f"    {function_source.split("(")[0].split("def ")[1]}()\n")
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

        def prepare_dsivc(self,decvar_names,t_d=None,pst=None,oe=None,track_stack=False,dsi_args=None,percentiles=[0.25,0.75,0.5],mou_population_size=None):

    
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
                self.logger.warning("using existing DSI template dir...")
                t_d = self.template_dir
            self.logger.statement(f"using {t_d} as template directory...")
            assert os.path.exists(t_d), f"template directory {t_d} does not exist"

            if pst is None:
                self.logger.statement("no pst provided...")
                self.logger.warning("using dsi.pst in DSI template dir...")
                assert os.path.exists(os.path.join(t_d,"dsi.pst")), f"dsi.pst not found in {t_d}"
                pst = Pst(os.path.join(t_d,"dsi.pst"))
            if oe is None:
                self.logger.statement("no posterior DSI observation ensemble provided, using dsi.3.obs.jcb in DSI template dir...")
                self.logger.warning(f"using dsi.{dsi_args['noptmax']}.obs.jcb in DSI template dir...")
                assert os.path.exists(os.path.join(t_d,f"dsi.{dsi_args['noptmax']}.obs.jcb")), f"dsi.{dsi_args['noptmax']}.obs.jcb not found in {t_d}"
                oe = ObservationEnsemble.from_binary(pst,os.path.join(t_d,f"dsi.{dsi_args['noptmax']}.obs.jcb"))
            else:
                assert isinstance(oe, ObservationEnsemble), "oe must be an ObservationEnsemble"

            #check if decvar_names str
            if isinstance(decvars, str):
                decvars = [decvars]
            # chekc htat decvars are in the oe columns
            missing = [col for col in decvars if col not in oe.columns]
            assert len(missing) == 0, f"The following decvars are missing from the DSI obs ensemble: {missing}"
            # chekc htat decvars are in the pst observation data
            missing = [col for col in decvars if col not in pst.obs_names]
            assert len(missing) == 0, f"The following decvars are missing from the DSI pst control file: {missing}"


            # handle DSI args
            default_dsi_args =  {"noptmax":pst.control_data.noptmax,
                                "decvar_weight":1.0,
                                #"decvar_phi_factor":0.5,
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
                file.write(f"    {function_source.split("(")[0].split("def ")[1]}()\n")

            self.logger.statement(f"preparing nominal initial population...")
            if mou_population_size is None:
                # set the population size to 2 * number of decision variables
                # this is a good rule of thumb for MOU
                mou_population_size = 2 * len(decvar_names)
            # these should generally be twice the number of decision variables
            if mou_population_size < 2 * len(decvar_names):
                self.logger.warning(f"mou population is less than 2x number of decision variables, this may be too small...")
            # sample 160 sets of decision variables from a unform distribution
            dvpop = ParameterEnsemble.from_uniform_draw(pst,num_reals=mou_population_size)
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
            self.logger.warning("overwriting dsi.pst file...")
            pst.observation_data.loc[decvar_names, "weight"] = dsi_args["decvar_weight"]
            pst.control_data.noptmax = dsi_args["noptmax"]
            pst.write(os.path.join(t_d,"dsi.pst"), version=2)
            
            
            self.logger.warning("overwriting dsi.pickle file...")
            # re-pickle dsi to track dsivc args
            with open(os.path.join(t_d,"dsi.pickle"),"wb") as f:
                pickle.dump(self,f)

            self.logger.warning("DSIVC control files created...the user still needs to specify objectives...")
            return pst_dsivc


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
    """
    A transformer for normal score transformation.
    
    This class implements the normal score transformation algorithm,
    which transforms arbitrary distributions to normal distributions
    by matching quantiles.
    
    Parameters
    ----------
    tol : float, optional
        Convergence tolerance for the normal score estimation.
        Default is 1e-7.
    max_samples : int, optional
        Maximum number of samples to use for estimating the normal distribution.
        Default is 1,000,000.
    """
    def __init__(self, tol=1e-7, max_samples=1000000):
        self.tol = tol
        self.max_samples = max_samples
        

    def randrealgen_optimized(self, nreal):
        """
        Generate an optimized set of normal score quantiles.
        
        This method implements an optimized algorithm to generate normal
        score quantiles through Monte Carlo simulation.
        
        Parameters
        ----------
        nreal : int
            Number of quantiles to generate.
            
        Returns
        -------
        ndarray
            Array of normal score quantiles sorted in ascending order.
        """
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
def _normal_score(self, col, tol=1e-7, max_samples=1000000, quadratic_extrapolation=False):
    """
    Apply normal score transformation to a column in the DataFrame.
    
    This transformation maps the empirical distribution of data to a standard normal
    distribution. The transformation is monotonic and preserves the rank order
    of the original data.
    
    Parameters
    ----------
    col : str
        Column name to apply the transformation to.
    tol : float, optional
        Tolerance for convergence of the empirical distribution. Default is 1e-7.
    max_samples : int, optional
        Maximum number of samples to use for estimating the empirical distribution.
        Default is 1,000,000.
    quadratic_extrapolation : bool, optional
        If True, enables quadratic extrapolation for the inverse transformation.
        Default is False.
        
    Returns
    -------
    dict
        Parameters needed for the inverse transformation:
        - z_scores: The standard normal quantiles
        - originals: The original sorted values
        - quadratic_extrapolation: Whether to use quadratic extrapolation
    """
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
    """
    Inverse transform for the normal score transformation.
    
    Parameters
    ----------
    col : str
        Column name to apply the inverse transformation to.
    z_scores : array-like
        The Z-scores used for the transformation.
    originals : array-like
        The original values corresponding to the Z-scores.
    quadratic_extrapolation : bool, optional
        If True, use quadratic extrapolation for values outside the range
        of the original data. Default is False.
        
    Returns
    -------
    None
        The transformation is applied in-place to self.df[col].
    """
    z_scores = np.array(z_scores)
    originals = np.array(originals)
    z_vals = self.df[col]
    if isinstance(z_vals, pd.Series):
        z_vals = z_vals.values
    interpolated = np.interp(z_vals, z_scores, originals)
    assert np.all(np.isfinite(interpolated)), "Interpolation resulted in NaN values"
    assert np.all(np.isfinite(z_vals)), "Input values contain NaN values"
    assert interpolated.shape == z_vals.shape, "Interpolated values do not match input shape"

    if quadratic_extrapolation:
        low_mask = z_vals < z_scores.min()
        high_mask = z_vals > z_scores.max()

        if low_mask.any():
            coeffs_low = np.polyfit(z_scores[:3], originals[:3], deg=2)
            interpolated = np.atleast_1d(interpolated)
            interpolated[low_mask] = np.polyval(coeffs_low, z_vals[low_mask])

        if high_mask.any():
            coeffs_high = np.polyfit(z_scores[-3:], originals[-3:], deg=2)
            interpolated = np.atleast_1d(interpolated)
            interpolated[high_mask] = np.polyval(coeffs_high, z_vals[high_mask])

    self.df[col] = interpolated


def _moving_average_with_endpoints(y_values):
    """
    Apply a moving average smoothing to an array while preserving endpoints.
    
    This function applies a sliding window average to smooth data while
    preserving the endpoints. The window size increases with the size of
    the input array. Uniqueness is enforced to ensure proper operation with
    normal score transforms.
    
    Parameters
    ----------
    y_values : array-like
        The input values to be smoothed.
        
    Returns
    -------
    smoothed_y : ndarray
        The smoothed values with preserved endpoints and enforced uniqueness.
    """
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
        Row-wise min-max scaler for time series or grouped data.
        
        This scaler normalizes data on a row-by-row basis, with separate scaling
        for different feature groups. This is useful for time series data where
        each row might represent a different entity with its own scale.
        
        Parameters
        ----------
        feature_range : tuple, optional
            The range to scale features to. Default is (-1, 1).
        groups : dict
            Dictionary mapping group names to lists of column names to be scaled.
            Each group will be scaled independently.
        fit_groups : dict, optional
            Dictionary mapping group names to lists of column names (a subset of groups)
            used to compute the row-wise min and max. If not provided, defaults to groups.
        """
        assert isinstance(fit_groups,dict), "fit_groups must be a dictionary or None"
        assert isinstance(groups, dict), "groups must be a dictionary"
        
        self.feature_range = feature_range
        self.groups = groups
        self.fit_groups = fit_groups if fit_groups is not None else groups
        self._row_params = {}  # will store per–row (min, max) for each group on the last transform call

    def fit(self, X):
        """
        Fit the scaler to the data.
        
        For row-wise scaling, nothing needs to be learned globally.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The data to fit the scaler on.
            
        Returns
        -------
        self
            The fitted scaler.
        """
        # For row–wise scaling, nothing needs to be learned globally.
        return self

    def transform(self, X):
        """
        Transform the data using row-wise min-max scaling.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The data to transform.
            
        Returns
        -------
        pandas.DataFrame
            The transformed data.
        """
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
        """
        Inverse transform the scaled data back to the original scale.
        
        Parameters
        ----------
        X_scaled : pandas.DataFrame
            The scaled data to inverse transform.
            
        Returns
        -------
        pandas.DataFrame
            The inverse transformed data in the original scale.
            
        Raises
        ------
        ValueError
            If transform hasn't been called previously to store the row parameters.
        """
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





