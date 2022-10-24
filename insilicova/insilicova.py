# -*- coding: utf-8 -*-

"""
insilicova.insilicova
-------------------

This module contains the class for the InSilicoVA algorithm.
"""

from dataclasses import dataclass
from typing import Union
from vacheck.datacheck5 import datacheck5
import warnings
import os
import pandas as pd
import numpy as np


class InSilicoVAException(Exception):
    """ Base exception for package"""


class InSilicoVA:
    """
    Implementation of InSilicoVA algorithm.

    """

    def __init__(self,
                 data: pd.DataFrame,
                 data_type: str = "WHO2016",
                 sci: Union[None, pd.DataFrame] = None,
                 is_numeric: bool = False,
                 update_cond_prob: bool = True,
                 keep_probbase_level: bool = True,
                 cond_prob: Union[None, np.ndarray] = None,
                 cond_prob_num: Union[None, np.ndarray] = None,
                 datacheck: bool = True,
                 datacheck_missing: bool = True,
                 warning_write: bool = False,
                 directory: Union[None, str] = None,
                 external_sep: bool = True,
                 n_sim: int = 4000,
                 thin: int = 10,
                 burnin: int = 2000,
                 auto_length: bool = True,
                 conv_csmf: float = 0.02,
                 jump_scale: float = 0.1,
                 levels_prior: Union[None, np.ndarray] = None,
                 levels_strength: int = 1,
                 trunc_min: float = 0.0001,
                 trunc_max: float = 0.9999,
                 subpop: Union[None, str] = None,
                 java_option: str = "-Xmx1g",
                 seed: int = 1,
                 phy_code: Union[None, pd.DataFrame] = None,
                 phy_cat: Union[None, pd.DataFrame] = None,
                 phy_unknown: Union[None, str] = None,
                 phy_external: Union[None, str] = None,
                 phy_debias: Union[None, dict] = None,
                 exclude_impossible_causes: str = "subset2",
                 impossible_combination: Union[None, str] = None,
                 no_is_missing: bool = False,
                 indiv_ci: Union[None, float] = None,
                 groupcode: bool = False):

        # assign all the arguments as instance attributes
        vars(self).update(locals())
        self._check_args()

    def _check_args(self):
        if not self.datacheck and self.data_type == "WHO2016":
            warnings.warn("Data check is turned off. Please be very careful "
                          "with this, because some indicators needs to be " 
                          "negated in the data check steps (i.e., having " 
                          "symptom = Yes turned into not having symptom = No "
                          "). Failing to properly negate all such symptoms " 
                          "will lead to erroneous inference.",
                          UserWarning)
        if self.warning_write:
            try:
                os.makedirs(self.directory, exist_ok=True)
            except (PermissionError, OSError) as exc:
                raise InSilicoVAException(
                    f"InSilicoVA unable to create {self.directory}:\n{exc}")

    @staticmethod
    def _interva_table(standard: bool = True,
                       min_level: Union[None, float] = None,
                       table_num_dev: Union[None, np.ndarray] = None) -> np.ndarray:
        """Function to return the interVA conversion table for alphabetic
        levels also change the smallest value from 0 to user input.

        :param standard: if True, only need min and use interVA standard values
        :type standard: bool
        :param min_level: minimum level to replace 0 in interVA standard
        :type min_level: None or float
        :param table_num_dev: customized values
        :type table_num_dev: numpy.ndarray of floats
        :returns: numeric values for mapping of InterVA probbase levels (grades)
        to probabilities
        :rtype: numpy.ndarray
        """

        if standard:
            if min_level is None:
                raise InSilicoVAException(
                    "Minimum level (min_level) not specified")
            tab_min = np.array([1, 0.8, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005,
                                0.002, 0.001, 0.0005, 0.0001, 0.00001,
                                min_level])
            return tab_min
        else:
            if table_num_dev is None:
                raise InSilicoVAException(
                    "Numerical level table (table_num_dev) not specified")
            new_tab = table_num_dev.copy()
            new_tab = new_tab[::-1]
            if table_num_dev.min() == 0:
                new_tab[-1] = new_tab[-2]/2
                return new_tab
            else:
                return new_tab
