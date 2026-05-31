from __future__ import print_function, division
import pyemu
import os
import copy
import shutil
from datetime import datetime
import warnings
from .pyemu_warnings import PyemuWarning
import numpy as np
import pandas as pd
from pyemu.en import ObservationEnsemble, ParameterEnsemble
from pyemu.mat.mat_handler import Matrix, Jco, Cov
from pyemu.pst.pst_handler import Pst
from pyemu.utils.os_utils import _istextfile,run
from .logger import Logger


def _islistlike(v):
    """True if `v` is a non-string iterable of names (list/tuple/array/Index/Series)."""
    return isinstance(v, (list, tuple, set, np.ndarray, pd.Index, pd.Series))


class EnDS(object):
    """Ensemble Data Space Analysis using the approach of He et al (2018)
    (https://doi.org/10.2118/182609-PA)

    Args:
        pst (varies): something that can be cast into a `pyemu.Pst`.  Can be an `str` for a
            filename or an existing `pyemu.Pst`.
        sim_ensemble (varies): something that can be cast into a `pyemu.ObservationEnsemble`.  Can be
            an `str` for a  filename or `pd.DataFrame` or an existing `pyemu.ObservationEnsemble`.  This is
            the source of the `predictions` and of the observations used for data-worth analysis.
        par_ensemble (varies, optional): something that can be cast into a `pyemu.ParameterEnsemble`.  Can be
            an `str` for a filename or `pd.DataFrame` or an existing `pyemu.ParameterEnsemble`.  Required only
            for parameter-importance analysis (`EnDS.get_parameter_importance_moments()`).  Parameters with
            `partrans=="log"` in `pst.parameter_data` are log10-transformed unless `apply_log_transform` is
            `False`.  Must share realization (index) labels with `sim_ensemble`.
        noise_ensemble (varies): something that can be cast into a `pyemu.ObservationEnsemble` that is the
            obs+noise realizations.  If not passed, a noise ensemble is generated using either `obs_cov` or the
            information in `pst` (i.e. weights or standard deviations)
        obscov (varies, optional): observation noise covariance matrix.  If `str`, a filename is assumed and
            the noise covariance matrix is loaded from a file using
            the file extension (".jcb"/".jco" for binary, ".cov"/".mat" for PEST-style ASCII matrix,
            or ".unc" for uncertainty files).  If `None`, the noise covariance matrix is
            constructed from the observation weights (and optionally "standard_deviation")
            .  Can also be a `pyemu.Cov` instance.  Loaded lazily - only required for observation
            (data-worth) conditioning, not for parameter-importance analysis.
        predictions (enumerable of `str`): the names of the entries in `pst.observation_data` (and in
            `sim_ensemble`) whose posterior moments are reported.  If `None`, `pst.forecast_names` is used.
        apply_log_transform (`bool`): if `True` (default), log10-transform `partrans=="log"` parameters in
            `par_ensemble`.  Set `False` if the parameter ensemble is already in the desired space.
        verbose (`bool`): controls screen output.  If `str`, a filename is assumed and
                and log file is written.


    Example::

        #assumes "my.pst" exists
        ends = pyemu.EnDS(pst="my.pst",sim_ensemble="my.0.obs.csv",predictions=["fore1","fore2"])
        ends.get_posterior_prediction_moments() #similar to Schur-style data worth


    """

    def __init__(
        self,
        pst=None,
        sim_ensemble=None,
        par_ensemble=None,
        noise_ensemble=None,
        obscov=None,
        predictions=None,
        apply_log_transform=True,
        verbose=False,
    ):
        self.logger = Logger(verbose)
        self.log = self.logger.log
        self.pst_arg = pst
        self.apply_log_transform = bool(apply_log_transform)

        # private attributes - access is through @decorated functions
        self.__pst = None
        self.__obscov = None
        self.__sim_en = None
        self.__par_en = None
        self.__noise_ensemble = None

        # the pst is needed to load the ensembles, so load it first
        if pst is None:
            raise Exception("pst is required for EnDS")
        self.__load_pst()

        self.log("pre-loading base components")
        # the observation ensemble is always required - it is the source of the
        # predictions (and of the observations for data-worth analysis)
        self.sim_ensemble_arg = sim_ensemble
        if sim_ensemble is not None:
            self.__sim_en = self.__load_ensemble(self.sim_ensemble_arg, ObservationEnsemble)
        if self.sim_ensemble is None:
            raise Exception("sim_ensemble is required for EnDS")

        # the parameter ensemble is optional - only needed for parameter-importance analysis
        self.par_ensemble_arg = par_ensemble
        if par_ensemble is not None:
            self.__par_en = self.__load_par_ensemble(self.par_ensemble_arg)

        self.noise_ensemble_arg = noise_ensemble
        if noise_ensemble is not None:
            self.__noise_ensemble = self.__load_ensemble(self.noise_ensemble_arg, ObservationEnsemble)

        # obscov is loaded lazily (see the `obscov` property): it is only required for
        # observation-conditioning runs, not for parameter-importance analysis
        self.obscov_arg = obscov
        if obscov is not None:
            self.__load_obscov()

        self.predictions = predictions
        if predictions is None and self.pst is not None:
            if self.pst.forecast_names is not None:
                self.predictions = self.pst.forecast_names
        if self.predictions is None:
            raise Exception("predictions are required for EnDS")
        if isinstance(self.predictions,list):
            self.predictions = [p.strip().lower() for p in self.predictions]

        self.log("pre-loading base components")

    def __fromfile(self, filename, astype=None):
        """a private method to deduce and load a filename into a matrix object.
        Uses extension: 'jco' or 'jcb': binary, 'mat','vec' or 'cov': ASCII,
        'unc': pest uncertainty file.

        """
        assert os.path.exists(filename), (
            "LinearAnalysis.__fromfile(): " + "file not found:" + filename
        )
        ext = filename.split(".")[-1].lower()
        if ext in ["jco", "jcb"]:
            self.log("loading jcb format: " + filename)
            if astype is None:
                astype = Jco
            m = astype.from_binary(filename)
            self.log("loading jcb format: " + filename)
        elif ext in ["mat", "vec"]:
            self.log("loading ascii format: " + filename)
            if astype is None:
                astype = Matrix
            m = astype.from_ascii(filename)
            self.log("loading ascii format: " + filename)
        elif ext in ["cov"]:
            self.log("loading cov format: " + filename)
            if astype is None:
                astype = Cov
            if _istextfile(filename):
                m = astype.from_ascii(filename)
            else:
                m = astype.from_binary(filename)
            self.log("loading cov format: " + filename)
        elif ext in ["unc"]:
            self.log("loading unc file format: " + filename)
            if astype is None:
                astype = Cov
            m = astype.from_uncfile(filename)
            self.log("loading unc file format: " + filename)
        elif ext in ["csv"]:
            self.log("loading csv format: " + filename)
            if astype is None:
                astype = ObservationEnsemble
            m = astype.from_csv(self.pst,filename=filename)
            self.log("loading csv format: " + filename)
        else:
            raise Exception(
                "EnDS.__fromfile(): unrecognized"
                + " filename extension:"
                + str(ext)
            )
        return m

    def __load_pst(self):
        """private method set the pst attribute"""
        if self.pst_arg is None:
            return None
        if isinstance(self.pst_arg, Pst):
            self.__pst = self.pst_arg
            return self.pst
        else:
            try:
                self.log("loading pst: " + str(self.pst_arg))
                self.__pst = Pst(self.pst_arg)
                self.log("loading pst: " + str(self.pst_arg))
                return self.pst
            except Exception as e:
                raise Exception(
                    "EnDS.__load_pst(): error loading"
                    + " pest control from argument: "
                    + str(self.pst_arg)
                    + "\n->"
                    + str(e)
                )

    def __ensemble_fromfile(self, filename, astype):
        """private method to load an ensemble (`ObservationEnsemble` or
        `ParameterEnsemble`) from a file, deducing the format from the extension.
        Unlike `__fromfile`, the `pst` is passed to the classmethods, which the
        ensemble constructors require."""
        assert os.path.exists(filename), (
            "EnDS.__ensemble_fromfile(): file not found: " + filename
        )
        ext = filename.split(".")[-1].lower()
        if ext in ["jcb", "jco", "bin"]:
            self.log("loading binary ensemble: " + filename)
            ensemble = astype.from_binary(self.pst, filename)
            self.log("loading binary ensemble: " + filename)
        elif ext in ["csv"]:
            self.log("loading csv ensemble: " + filename)
            ensemble = astype.from_csv(self.pst, filename=filename)
            self.log("loading csv ensemble: " + filename)
        else:
            raise Exception(
                "EnDS.__ensemble_fromfile(): unrecognized ensemble "
                + "filename extension: " + str(ext)
            )
        return ensemble

    def __load_ensemble(self, arg, astype):
        """private method to build an ensemble of type `astype`
        (`ObservationEnsemble` or `ParameterEnsemble`) from a file, dataframe,
        matrix or existing ensemble object.  Always works on a copy."""
        if arg is None:
            return None
        if isinstance(arg, astype):
            ensemble = arg.copy()
        elif isinstance(arg, str):
            ensemble = self.__ensemble_fromfile(arg, astype=astype)
        elif isinstance(arg, Matrix):
            ensemble = astype(pst=self.pst, df=arg.to_dataframe())
        elif isinstance(arg, pd.DataFrame):
            ensemble = astype(pst=self.pst, df=arg.copy())
        else:
            raise Exception(
                "EnDS.__load_ensemble(): arg must "
                + "be a matrix object, dataframe, "
                + astype.__name__ + ", or a file name: "
                + str(arg)
            )
        return ensemble

    def __load_par_ensemble(self, arg):
        """private method to build a `ParameterEnsemble` and apply the log10
        transform to `partrans=="log"` parameters (unless `apply_log_transform`
        is `False`).  Always works on a copy so the caller's object is never
        mutated.  Raw dataframe/file inputs are assumed to be in arithmetic
        (untransformed) space."""
        pe = self.__load_ensemble(arg, ParameterEnsemble)
        if pe is None:
            return None
        if self.apply_log_transform:
            # ParameterEnsemble.transform() is driven by the istransformed flag
            # and log10-transforms only partrans=="log" parameters, in place on
            # our copy.  it is a no-op if the ensemble is already transformed.
            pe.transform()
        return pe



    def __load_obscov(self):
        """private method to set the obscov attribute from:
        a pest control file (observation weights)
        a pst object
        a matrix object
        an uncert file
        an ascii matrix file
        """
        # if the obscov arg is None, but the pst arg is not None,
        # reset and load from obs weights
        self.log("loading obscov")
        if not self.obscov_arg:
            if self.pst_arg:
                self.obscov_arg = self.pst_arg
            else:
                raise Exception(
                    "linear_analysis.__load_obscov(): " + "obscov_arg is None"
                )
        if isinstance(self.obscov_arg, Matrix):
            self.__obscov = self.obscov_arg
            return

        elif isinstance(self.obscov_arg, str):
            if self.obscov_arg.lower().endswith(".pst"):
                self.__obscov = Cov.from_obsweights(self.obscov_arg)
            else:
                self.__obscov = self.__fromfile(self.obscov_arg, astype=Cov)
        elif isinstance(self.obscov_arg, Pst):
            self.__obscov = Cov.from_observation_data(self.obscov_arg)
        else:
            raise Exception(
                "EnDS.__load_obscov(): "
                + "obscov_arg must be a "
                + "matrix object or a file name: "
                + str(self.obscov_arg)
            )
        self.log("loading obscov")



    # these property decorators help keep from loading potentially
    # unneeded items until they are called
    # returns a reference - cheap, but can be dangerous

    @property
    def sim_ensemble(self):
        """the observation (simulation) ensemble - source of predictions and observations

        Returns:
            `pyemu.ObservationEnsemble`: the loaded observation ensemble
        """
        return self.__sim_en

    @property
    def par_ensemble(self):
        """the parameter ensemble used for parameter-importance analysis

        Returns:
            `pyemu.ParameterEnsemble`: the loaded (and optionally log-transformed)
            parameter ensemble, or `None` if none was passed
        """
        return self.__par_en

    @property
    def obscov(self):
        """get the observation noise covariance matrix attribute

        Returns:
            `pyemu.Cov`: a reference to the `LinearAnalysis.obscov` attribute

        """
        if not self.__obscov:
            self.__load_obscov()
        return self.__obscov


    @property
    def pst(self):
        """the pst attribute

        Returns:
            `pyemu.Pst`: the pst attribute

        """
        if self.__pst is None and self.pst_arg is None:
            raise Exception(
                "linear_analysis.pst: can't access self.pst:"
                + "no pest control argument passed"
            )
        elif self.__pst:
            return self.__pst
        else:
            self.__load_pst()
            return self.__pst

    def reset_pst(self, arg):
        """reset the EnDS.pst attribute

        Args:
            arg (`str` or `pyemu.Pst`): the value to assign to the pst attribute

        """
        self.logger.statement("resetting pst")
        self.__pst = None
        self.pst_arg = arg

    def reset_obscov(self, arg=None):
        """reset the obscov attribute to None

        Args:
            arg (`str` or `pyemu.Matrix`): the value to assign to the obscov
                attribute.  If None, the private __obscov attribute is cleared
                but not reset
        """
        self.logger.statement("resetting obscov")
        self.__obscov = None
        if arg is not None:
            self.obscov_arg = arg

    def get_posterior_prediction_convergence_summary(self,num_realization_sequence,num_replicate_sequence,
                                             obslist_dict=None):
        """repeatedly run `EnDS.get_predictive_posterior_moments() with less than all the possible
        realizations to evaluate whether the uncertainty estimates have converged

        Args:
            num_realization_sequence (`[int']): the sequence of realizations to test.
            num_replicate_sequence (`[int]`): The number of replicates of randomly selected realizations to test
                for each `num_realization_sequence` value.  For example, if num_realization_sequence is [10,100,1000]
                and num_replicated_sequence is [4,5,6], then `EnDS.get_predictive posterior_moments()` is called 4
                times using 10 randomly selected realizations (new realizations selected 4 times), 5 times using
                100 randomly selected realizations, and then 6 times using 1000 randomly selected realizations.
            obslist_dict (`dict`, optional): a nested dictionary-list of groups of observations
                to pass to `EnDS.get_predictive_posterior_moments()`.

        Returns:
             `dict`: a dictionary of num_reals: `pd.DataFrame` pairs, where the dataframe is the mean
                predictive standard deviation results from calling `EnDS.get_predictive_posterior_moments()` for the
                desired number of replicates.

         Example::

            ends = pyemu.EnDS(pst="my.pst",sim_ensemble="my.0.obs.csv",predictions=["predhead","predflux"])
            obslist_dict = {"hds":["head1","head2"],"flux":["flux1","flux2"]}
            num_reals_seq = [10,20,30,100,1000] # assuming there are 1000 reals in "my.0.obs.csv"]
            num_reps_seq = [5,5,5,5,5]
            mean_dfs = sc.get_posterior_prediction_convergence_summary(num_reals_seq,num_reps_seq,
                obslist_dict=obslist_dict)

        """
        real_idx = np.arange(self.sim_ensemble.shape[0],dtype=int)
        results = {}
        for nreals,nreps in zip(num_realization_sequence,num_replicate_sequence):
            rep_results = []
            print("-->testing ",nreals)
            for rep in range(nreps):
                rreals = pyemu.en.rng.choice(real_idx,nreals,False)
                sim_ensemble = self.sim_ensemble.iloc[rreals,:].copy()
                _,dfstd,_ = self.get_posterior_prediction_moments(obslist_dict=obslist_dict,
                                                                 sim_ensemble=sim_ensemble,
                                                                 include_first_moment=False)
                rep_results.append(dfstd)
            results[nreals] = rep_results

        means = {}
        for nreals,dfs in results.items():
            mn = dfs[0]
            for df in dfs[1:]:
                mn += df
            mn /= len(dfs)
            means[nreals] = mn

        return means

    def get_parameter_importance_convergence_summary(self, num_realization_sequence,
                                                      num_replicate_sequence, parlist_dict=None,
                                                      eigthresh=1.0e-5, noise_cov=None):
        """repeatedly run `EnDS.get_parameter_importance_moments()` with less than all the
        possible realizations to evaluate whether the importance (uncertainty) estimates
        have converged with respect to ensemble size.  The parameter-conditioning analog
        of `EnDS.get_posterior_prediction_convergence_summary()`.

        Args:
            num_realization_sequence (`[int]`): the sequence of realization counts to test.
            num_replicate_sequence (`[int]`): the number of replicates of randomly selected
                realizations to test for each `num_realization_sequence` value.
            parlist_dict (`dict`, optional): a nested dictionary-list of groups of parameters
                to pass to `EnDS.get_parameter_importance_moments()`.
            eigthresh (`float`): truncation ratio for the truncated-SVD pseudo-inverse of
                each parameter data block.  Default is 1.0e-5
            noise_cov (`float`, `pyemu.Cov` or `str`, optional): regularization passed to
                `EnDS.get_parameter_importance_moments()`.  Default is `None` (exact).

        Returns:
            `dict`: a dictionary of num_reals: `pd.DataFrame` pairs, where the dataframe is
                the mean predictive standard deviation results across the replicates.

        Example::

            ends = pyemu.EnDS(pst="my.pst",sim_ensemble="my.0.obs.csv",
                              par_ensemble="my.0.par.csv",predictions=["predhead","predflux"])
            num_reals_seq = [10,20,30,100]
            num_reps_seq = [5,5,5,5]
            mean_dfs = ends.get_parameter_importance_convergence_summary(num_reals_seq,num_reps_seq)

        """
        if self.par_ensemble is None:
            raise Exception(
                "EnDS.get_parameter_importance_convergence_summary(): a par_ensemble is "
                "required - pass one to the EnDS constructor"
            )
        # align par and obs ensembles once, then subsample from the shared realizations
        par_df = self.par_ensemble._df
        sim_df = self.sim_ensemble._df
        common = par_df.index.intersection(sim_df.index)
        if len(common) == 0:
            raise Exception(
                "EnDS.get_parameter_importance_convergence_summary(): par_ensemble and "
                "sim_ensemble share no realization (index) labels"
            )
        real_idx = np.array(list(common))
        results = {}
        for nreals, nreps in zip(num_realization_sequence, num_replicate_sequence):
            rep_results = []
            print("-->testing ", nreals)
            for rep in range(nreps):
                rreals = pyemu.en.rng.choice(real_idx, nreals, False)
                # pass plain (already log-transformed) frames so no re-transform happens
                par_sub = par_df.loc[rreals].copy()
                sim_sub = sim_df.loc[rreals].copy()
                _, dfstd, _ = self.get_parameter_importance_moments(
                    parlist_dict=parlist_dict, par_ensemble=par_sub, sim_ensemble=sim_sub,
                    include_first_moment=False, eigthresh=eigthresh, noise_cov=noise_cov)
                rep_results.append(dfstd)
            results[nreals] = rep_results

        means = {}
        for nreals, dfs in results.items():
            mn = dfs[0]
            for df in dfs[1:]:
                mn += df
            mn /= len(dfs)
            means[nreals] = mn

        return means


    def get_posterior_prediction_moments(self, obslist_dict=None,sim_ensemble=None,
                                         include_first_moment=True,eigthresh=1.0e-5):
        """A dataworth method to analyze the posterior (expected) mean and uncertainty as a result of conditioning with
         some additional observations not used in the conditioning of the current ensemble results.

        Args:
            obslist_dict (`dict`, optional): a nested dictionary-list of groups of observations
                that are to be treated as gained/collected.  key values become
                row labels in returned dataframe. If `None`, then every nonzero-weighted
                observation is tested sequentially. Default is `None`
            sim_ensemble (`pyemu.ObservationEnsemble`): the simulation results ensemble to use.
                If `None`, `self.sim_ensemble` is used.  Default is `None`

            include_first_moment (`bool`): flag to include calculations of the predictive first moments.
                This can slow things down,so if not needed, better to skip.  Default is `True`
            eigthresh (`float`): truncation ratio for the truncated-SVD pseudo-inverse of each
                observation data block.  Default is 1.0e-5

        Returns:
            tuple containing

            - **dict**: dictionary of first-moment dataframes. Keys are `obslist_dict` keys.  If `include_first_moment`
                is `False`, this is an empty dict.
            - **pd.DataFrame**: prediction standard deviation summary
            - **pd.DataFrame**: percent prediction standard deviation reduction summary


        Example::

            ends = pyemu.EnDS(pst="my.pst",sim_ensemble="my.0.obs.csv",predictions=["predhead","predflux"])
            obslist_dict = {"hds":["head1","head2"],"flux":["flux1","flux2"]}
            mean_dfs,dfstd,dfpercent = ends.get_posterior_prediction_moments(obslist_dict=obslist_dict)

        """

        if obslist_dict is not None:
            if type(obslist_dict) == list:
                obslist_dict = dict(zip(obslist_dict, obslist_dict))
            obs = self.pst.observation_data
            onames = []
            [onames.extend(names) for gp,names in obslist_dict.items()]
            oobs = obs.loc[onames,:]
            zobs = oobs.loc[oobs.weight==0,"obsnme"].tolist()

            if len(zobs) > 0:
                raise Exception(
                    "Observations in obslist_dict must have "
                    + "nonzero weight. The following observations "
                    + "violate that condition: "
                    + ",".join(zobs)
                )
        else:
            obslist_dict = dict(zip(self.pst.nnz_obs_names, self.pst.nnz_obs_names))
            onames = self.pst.nnz_obs_names

        if "posterior" not in obslist_dict:
            obslist_dict["posterior"] = onames

        # normalize group values to lists (the default {name: name} mapping has
        # scalar string values)
        obslist_dict = {g: (list(v) if _islistlike(v) else [v])
                        for g, v in obslist_dict.items()}

        if sim_ensemble is None:
            sim_ensemble = self.sim_ensemble
        sim_df = sim_ensemble._df if hasattr(sim_ensemble, "_df") else sim_ensemble

        # union of all conditioning observation names plus the predictions, deduped
        # (a plain copy - never mutate pst.nnz_obs_names in place)
        cond_names = []
        for group in obslist_dict:
            cond_names.extend(obslist_dict[group])
        names = list(dict.fromkeys(list(cond_names) + list(self.predictions)))
        data_df = sim_df.loc[:, names]

        # observation conditioning adds the observation noise covariance to each data block
        return self._conditioning_moments(data_df, obslist_dict, self.predictions,
                                          noise_cov=self.obscov,
                                          include_first_moment=include_first_moment,
                                          eigthresh=eigthresh)

    def _conditioning_moments(self, data_df, cond_dict, predictions, noise_cov=None,
                              include_first_moment=True, eigthresh=1.0e-5):
        """core ensemble Schur computation shared by the observation-conditioning
        (data-worth) and parameter-conditioning (importance) entry points.

        Args:
            data_df (`pd.DataFrame`): a realizations-by-names frame containing every
                conditioning name referenced in `cond_dict` plus all `predictions`,
                sharing a single realization (row) index.
            cond_dict (`dict`): {group_name: [conditioning names]} mapping.
            predictions (`[str]`): prediction names (columns of `data_df`) whose
                posterior moments are reported.
            noise_cov (`pyemu.Cov` or `float`, optional): regularization/noise added to each
                conditioning data block.  A `pyemu.Cov` is added directly (e.g. observation
                noise or a prior parameter covariance).  A scalar is treated as relative
                diagonal inflation - a fraction of each conditioning quantity's own variance
                is added to the diagonal.  If `None`, conditioning is exact.
            include_first_moment (`bool`): whether to compute per-realization first moments.
            eigthresh (`float`): truncation ratio for the truncated-SVD pseudo-inverse
                of each conditioning data block.  Keeps the computation robust to
                rank-deficient/collinear blocks (an empirical cov from n realizations
                has rank <= n-1).

        Returns:
            (mean_dfs, dfstd, dfper) - as documented on the public methods.
        """
        names = list(data_df.columns)
        nreal = data_df.shape[0]
        self.logger.log("getting deviations")
        oe = (data_df - data_df.mean(axis=0)) / np.sqrt(float(nreal - 1))
        self.logger.log("getting deviations")
        self.logger.log("forming cov matrix")
        cmat = np.dot(oe.values.transpose(), oe.values)
        cdf = pd.DataFrame(cmat, index=names, columns=names)
        self.logger.log("forming cov matrix")

        var_data = {}
        prior_var = data_df.loc[:, predictions].std() ** 2
        prior_mean = data_df.loc[:, predictions].mean()
        var_data["prior"] = prior_var
        groups = list(cond_dict.keys())
        groups.sort()

        mean_dfs = {}
        for group in groups:
            self.logger.log("processing " + group)
            cnames = list(cond_dict[group])
            self.logger.log("extracting blocks from cov matrix")
            dd = cdf.loc[cnames, cnames].values.copy()
            ccov = cdf.loc[cnames, predictions].values.copy()
            self.logger.log("extracting blocks from cov matrix")

            if noise_cov is not None:
                self.logger.log("adding noise cov to data block")
                if np.isscalar(noise_cov):
                    # relative diagonal inflation: add a fraction of each conditioning
                    # quantity's own variance (scale-free Tikhonov regularization)
                    dd[np.diag_indices_from(dd)] += float(noise_cov) * np.diag(dd)
                else:
                    # an explicit (e.g. observation noise or prior) covariance block
                    dd += noise_cov.get(cnames, cnames).x
                self.logger.log("adding noise cov to data block")

            # truncated-SVD pseudo-inverse - robust to rank-deficient/collinear blocks
            self.logger.log("inverting data cov block")
            dd = Matrix(x=dd, row_names=cnames, col_names=cnames).pseudo_inv(eigthresh=eigthresh).x
            self.logger.log("inverting data cov block")

            pt_var = []
            pt_mean = {}
            if include_first_moment:
                self.logger.log("preping first moment pieces")
                cmean = data_df.loc[:, cnames].mean().values
                reals = data_df.index.values
                innovation_vecs = {real: data_df.loc[real, cnames].values - cmean
                                   for real in data_df.index}
                self.logger.log("preping first moment pieces")

            neg_preds = []
            for i, p in enumerate(predictions):
                self.logger.log("calc second moment for " + p)
                ccov_vec = ccov[:, i]
                first_term = np.dot(ccov_vec.transpose(), dd)
                schur = np.dot(first_term, ccov_vec)
                post_var = prior_var.iloc[i] - schur
                if post_var < 0:
                    neg_preds.append(p)
                pt_var.append(post_var)
                self.logger.log("calc second moment for " + p)
                if include_first_moment:
                    self.logger.log("calc first moment values for " + p)
                    mean_vals = []
                    prmn = prior_mean[p]
                    for real in reals:
                        mn = prmn + np.dot(first_term, innovation_vecs[real])
                        mean_vals.append(mn)
                    pt_mean[p] = np.array(mean_vals)
                    self.logger.log("calc first moment values for " + p)
            if len(neg_preds) > 0:
                # a negative posterior variance (-> NaN std) signals a rank-deficient
                # or ill-conditioned conditioning block (group larger than the ensemble
                # can resolve, or strongly collinear).  the resulting std/percent values
                # will be NaN for these predictions.
                self.logger.warn(
                    "negative posterior variance for group '{0}' and prediction(s) {1}; "
                    "the conditioning block is rank-deficient/ill-conditioned (group of {2} "
                    "names vs {3} realizations) - consider more realizations, a larger "
                    "eigthresh, or smaller/less-collinear groups".format(
                        group, ",".join(neg_preds), len(cnames), nreal)
                )
            if include_first_moment:
                mean_df = pd.DataFrame(pt_mean, index=reals)
                mean_dfs[group] = mean_df
            var_data[group] = pt_var
            self.logger.log("processing " + group)

        dfstd = pd.DataFrame(var_data, index=predictions).apply(np.sqrt).T
        dfper = dfstd.copy()
        prior_std = prior_var.apply(np.sqrt)
        for p in predictions:
            dfper.loc[groups, p] = 100 * (1 - (dfstd.loc[groups, p].values / prior_std.loc[p]))
        dfper = dfper.loc[groups, predictions]

        return mean_dfs, dfstd, dfper

    def get_parameter_importance_moments(self, parlist_dict=None, par_ensemble=None,
                                         sim_ensemble=None, include_first_moment=False,
                                         eigthresh=1.0e-5, noise_cov=None):
        """A parameter-importance method to analyze the posterior (expected) mean and
        uncertainty of the predictions as a result of conditioning on (i.e. coming to
        know) one or more parameters.  This is the same ensemble Schur math as
        `EnDS.get_posterior_prediction_moments()`, but the conditioning quantities are
        parameters rather than observations.

        Args:
            parlist_dict (`dict`, optional): a nested dictionary-list of groups of
                parameters whose importance is to be evaluated.  Keys become row labels
                in the returned dataframes.  If `None`, every adjustable parameter
                (`partrans` not "fixed" or "tied") is tested sequentially.  Default is `None`
            par_ensemble (`pyemu.ParameterEnsemble`, optional): the parameter ensemble to
                use.  If `None`, `self.par_ensemble` is used.  Must share realization
                (index) labels with `sim_ensemble`.  When `None`, the constructor's
                (already log-transformed) ensemble is used.
            sim_ensemble (`pyemu.ObservationEnsemble`, optional): the simulation results
                ensemble - the source of the predictions.  If `None`, `self.sim_ensemble`
                is used.
            include_first_moment (`bool`): flag to include calculation of the predictive
                first moments.  Defaults to `False` (importance work usually needs only
                the variance reduction, and the first moment is the slow path).
            eigthresh (`float`): truncation ratio for the truncated-SVD pseudo-inverse of
                each parameter data block.  Default is 1.0e-5
            noise_cov (`float`, `pyemu.Cov` or `str`, optional): regularization added to each
                parameter data block before inversion (inexact conditioning).  A scalar is
                relative diagonal inflation - a fraction of each parameter's own (log-space)
                variance is added to the diagonal (e.g. `0.05` inflates by 5%).  A `pyemu.Cov`
                (or a filename to load one from) is added directly and may carry parameter
                correlation - it must be in the same (log) space as the ensemble; see
                `pyemu.Cov.from_parameter_data`.  If `None` (default), conditioning is exact.

        Returns:
            tuple containing

            - **dict**: dictionary of first-moment dataframes. Keys are `parlist_dict`
                keys.  If `include_first_moment` is `False`, this is an empty dict.
            - **pd.DataFrame**: prediction standard deviation summary
            - **pd.DataFrame**: percent prediction standard deviation reduction summary

        Note:
            Conditioning on parameters is *exact* by default - no noise term is added to
            the parameter data block.  Because an empirical covariance from `n`
            realizations has rank <= `n-1`, multi-parameter groups (e.g. the injected
            "all" group) are rank-deficient and are inverted with a truncated-SVD
            pseudo-inverse controlled by `eigthresh`.  A rank-deficient/ill-conditioned
            block can yield a negative posterior variance (reported as NaN std); if that
            happens, use more realizations, a larger `eigthresh`, smaller/less-collinear
            groups, or supply `noise_cov` to regularize.

        Example::

            ends = pyemu.EnDS(pst="my.pst",sim_ensemble="my.0.obs.csv",
                              par_ensemble="my.0.par.csv",predictions=["predhead","predflux"])
            parlist_dict = {"hk":["hk1","hk2"],"rch":["rch1","rch2"]}
            mean_dfs,dfstd,dfpercent = ends.get_parameter_importance_moments(parlist_dict=parlist_dict)

            # if a large/collinear group overshoots (negative posterior variance -> NaN std),
            # regularize.  a scalar inflates each parameter's variance by a fraction (here 5%) -
            # the quick, scale-free knob:
            _,dfstd,_ = ends.get_parameter_importance_moments(parlist_dict=parlist_dict,noise_cov=0.05)

            # when the trouble is parameter *correlation* (pilot points, multipliers), a scaled
            # prior covariance regularizes the off-diagonals too.  Cov.from_parameter_data is in
            # the same log space as the (log-transformed) ensemble:
            reg = 0.1 * pyemu.Cov.from_parameter_data(ends.pst)   # 10% of the prior, as "noise"
            _,dfstd,_ = ends.get_parameter_importance_moments(parlist_dict=parlist_dict,noise_cov=reg)

        """
        if par_ensemble is None:
            par_ensemble = self.par_ensemble
        if par_ensemble is None:
            raise Exception(
                "EnDS.get_parameter_importance_moments(): a par_ensemble is required - "
                "pass one here or to the EnDS constructor"
            )
        if sim_ensemble is None:
            sim_ensemble = self.sim_ensemble

        # resolve the optional regularization: scalar (relative diagonal inflation) is
        # passed through to the helper as-is; a Cov is used directly; a filename is loaded
        if noise_cov is not None and not np.isscalar(noise_cov):
            if isinstance(noise_cov, str):
                noise_cov = self.__fromfile(noise_cov, astype=Cov)
            elif not isinstance(noise_cov, Matrix):
                raise Exception(
                    "EnDS.get_parameter_importance_moments(): noise_cov must be a scalar, "
                    "a pyemu.Cov/Matrix, or a filename - not " + str(type(noise_cov))
                )
        elif np.isscalar(noise_cov) and noise_cov < 0:
            raise Exception(
                "EnDS.get_parameter_importance_moments(): scalar noise_cov "
                "(relative inflation) must be non-negative"
            )

        par = self.pst.parameter_data
        adj_names = par.loc[~par.partrans.isin(["fixed", "tied"]), "parnme"].tolist()
        adj_set = set(adj_names)

        if parlist_dict is not None:
            if type(parlist_dict) == list:
                parlist_dict = dict(zip(parlist_dict, parlist_dict))
            # normalize group values to lists up front so validation and the
            # union below never iterate the characters of a scalar name
            parlist_dict = {g: (list(v) if _islistlike(v) else [v])
                            for g, v in parlist_dict.items()}
            bad = []
            for gp, pnames in parlist_dict.items():
                bad.extend([n for n in pnames if n not in adj_set])
            if len(bad) > 0:
                raise Exception(
                    "Parameters in parlist_dict must be adjustable (partrans not "
                    "'fixed' or 'tied'). The following violate that condition: "
                    + ",".join(bad)
                )
        else:
            parlist_dict = dict(zip(adj_names, adj_names))

        if "all" not in parlist_dict:
            parlist_dict["all"] = adj_names

        # normalize group values to lists (the default {name: name} mapping has
        # scalar string values)
        parlist_dict = {g: (list(v) if _islistlike(v) else [v])
                        for g, v in parlist_dict.items()}

        # union of all conditioning parameter names, deduped
        cond_names = []
        for group in parlist_dict:
            cond_names.extend(parlist_dict[group])
        cond_names = list(dict.fromkeys(cond_names))

        # guard against name collisions between conditioning parameters and predictions:
        # the shared cov matrix is indexed by name, so a collision would silently
        # produce duplicate columns and wrong results
        collide = set(cond_names).intersection(set(self.predictions))
        if len(collide) > 0:
            raise Exception(
                "EnDS.get_parameter_importance_moments(): the following names appear as "
                "both a conditioning parameter and a prediction, which is not allowed: "
                + ",".join(sorted(collide))
            )

        par_df = par_ensemble._df if hasattr(par_ensemble, "_df") else par_ensemble
        sim_df = sim_ensemble._df if hasattr(sim_ensemble, "_df") else sim_ensemble
        cond_df = par_df.loc[:, cond_names]
        pred_df = sim_df.loc[:, self.predictions]

        # align the parameter and observation ensembles on shared realization labels
        common = cond_df.index.intersection(pred_df.index)
        if len(common) == 0:
            raise Exception(
                "EnDS.get_parameter_importance_moments(): par_ensemble and sim_ensemble "
                "share no realization (index) labels"
            )
        ndrop = max(cond_df.shape[0], pred_df.shape[0]) - len(common)
        if ndrop > 0:
            self.logger.warn(
                "parameter/observation ensembles aligned on {0} shared realizations; "
                "{1} realization(s) dropped".format(len(common), ndrop)
            )
        data_df = pd.concat([cond_df.loc[common], pred_df.loc[common]], axis=1)

        # exact conditioning by default (noise_cov is None); a scalar or Cov regularizes
        return self._conditioning_moments(data_df, parlist_dict, self.predictions,
                                          noise_cov=noise_cov,
                                          include_first_moment=include_first_moment,
                                          eigthresh=eigthresh)
