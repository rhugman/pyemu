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

        def __init__(self, pst=None, sim_ensemble=None):
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

            
            def apply_feature_transforms():
                #TODO
                #standardize
                #normal score transform
                #log transform
                #autoencoder
                return
            
            def apply_pca():
                #TODO
                return
            
            def forward_run():
                #TODO
                return