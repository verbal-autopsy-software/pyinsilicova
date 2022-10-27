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
    def _interva_table(
            standard: bool = True,
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

    def _scale_vec_inter(
            self,
            vector: np.ndarray,
            scale: Union[None, float] = None,
            scale_max: Union[None, float] = None) -> Union[None, np.ndarray]:
        """Function to map ordered input vector to InterVA alphabetic scale.
        Note: to avoid 0, change the last element in InterVA table to 0.000001.

        :param vector: vector to be scaled
        :type vector: np.ndarray
        :param scale: sum of the vector after scaling
        :type scale: float
        :param scale_max: max of the vector after scaling
        :type scale_max: float
        :returns: rescaled vector
        :rtype: numpy.ndarray
        """
        dist = self._interva_table(standard=True, min_level=0.000001)
        if len(dist) != len(dist):
            raise InSilicoVAException(
                "Dimension of probbase prior is incorrect.")
        out_vector = dist[vector.argsort()]
        if scale:
            return scale * (out_vector / out_vector.sum())
        if scale_max:
            return (scale_max * out_vector) / out_vector.max()

    def _change_inter(
            self,
            x: np.ndarray,
            order: bool = False,
            standard: bool = True,
            table_dev: Union[None, np.ndarray] = None,
            table_num_dev: Union[None, np.ndarray] = None) -> np.ndarray:
        """Translate alphabetic array into numeric array (potentially ordered)

        :param x: alphabetic array
        :type x: numpy.ndarray
        :param order: whether to change the array into an ordered array
        :type order: bool
        :param standard: whether to use the standard table
        :type standard: bool
        :param table_dev: new table of level names used (high to low)
        :type table_dev: numpy.ndarray
        :param table_num_dev: new table of numerical values corresponding to
        table_dev
        :type table_num_dev: numpy.ndarray
        :return: numeric array that is a mapping of alphabetic array
        :rtype: numpy.ndarray
        """

        y = np.zeros(x.shape)
        inter_table = self._interva_table(standard=standard,
                                          table_num_dev=table_num_dev,
                                          min_level=0)
        if table_dev:
            if len(table_dev) != len(table_num_dev):
                raise InSilicoVAException(
                    "table_dev and table_num_dev have different lengths"
                )
            for i in range(len(table_dev)):
                y[x == table_dev[i]] = inter_table[i]
        else:
            y[x == "I"] = inter_table[0]
            y[x == "A+"] = inter_table[1]
            y[x == "A"] = inter_table[2]
            y[x == "A-"] = inter_table[3]
            y[x == "B+"] = inter_table[4]
            y[x == "B"] = inter_table[5]
            y[x == "B-"] = inter_table[6]
            y[x == "B -"] = inter_table[6]
            y[x == "C+"] = inter_table[7]
            y[x == "C"] = inter_table[8]
            y[x == "C-"] = inter_table[9]
            y[x == "D+"] = inter_table[10]
            y[x == "D"] = inter_table[11]
            y[x == "D-"] = inter_table[12]
            y[x == "E"] = inter_table[13]
            y[x == "N"] = inter_table[14]
            y[x == ""] = inter_table[14]

        if order:
            for i in range(len(inter_table)):
                y_new = y.copy()
                y_new[y == inter_table[i]] = i + 1
                y = y_new
        return y

    def _cond_initiate(self):
        pass
