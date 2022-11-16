# -*- coding: utf-8 -*-

"""
insilicova.insilicova
-------------------

This module contains the class for the InSilicoVA algorithm.
"""

from insilicova.exceptions import ArgumentException, DataException
from insilicova.utils import get_vadata
from typing import Union, List
from vacheck.datacheck5 import datacheck5
import warnings
import os
import pandas as pd
import numpy as np
from collections import defaultdict


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
                 subpop: Union[None, list, str, pd.Series] = None,
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

        self.data = data.copy()
        self.data_type = data_type
        self.sci = sci
        self.is_numeric = is_numeric
        self.update_cond_prob = update_cond_prob
        self.keep_probbase_level = keep_probbase_level
        self.cond_prob = cond_prob
        self.cond_prob_num = cond_prob_num
        self.datacheck = datacheck
        self.datacheck_missing = datacheck_missing
        self.warning_write = warning_write
        self.directory = directory
        self.external_sep = external_sep
        self.n_sim = n_sim
        self.thin = thin
        self.burnin = burnin
        self.auto_length = auto_length
        self.conv_csmf = conv_csmf
        self.jump_scale = jump_scale
        self.levels_prior = levels_prior
        self.levels_strength = levels_strength
        self.trunc_min = trunc_min
        self.trunc_max = trunc_max
        self.subpop = subpop
        self.java_option = java_option
        self.seed = seed
        self.phy_code = phy_code
        self.phy_cat = phy_cat
        self.phy_unknown = phy_unknown
        self.phy_external = phy_external
        self.phy_debias = phy_debias
        self.exclude_impossible_causes = exclude_impossible_causes
        self.impossible_combination = impossible_combination
        self.no_is_missing = no_is_missing
        self.indiv_ci = indiv_ci
        self.groupcode = groupcode
        self.error_log = defaultdict(list)
        self.vacheck_log = {"first_pass": [], "second_pass": []}
        self.causetext = None
        self.probbase = None

        self.original_data = data.copy()
        if isinstance(subpop, str):
            self.original_subpop = subpop
        elif subpop:
            self.original_subpop = subpop.copy()
        else:
            self.original_subpop = None
        self.subpop_order_list = None
        # initialize probbase
        self.prob_orig = None
        # attributes for external causes
        self.ext_sub = None
        self.ext_id = None
        self.ext_cod = None
        self.ext_csmf = None
        self.ext_prob = None
        self.negate = None
        # attributes for future use? (unused in R code)
        # self.probbase_by_symp_dev
        self.method = "normal"
        self.alpha_scale = None
        # self.n_level_dev = None
        # if self.n_level_dev is not None:
        #     self.n_level = self.n_level_dev
        self.n_level = 15

        self._change_data_coding()
        self._check_args()
        self._initialize_data_dependencies()
        self._setup_subpop()

    def _change_data_coding(self):
        if self.data_type == "WHO2016":
            col_names = list(self.data)[1:]  # ignore ID
            for i in col_names:
                self.data[i] = self.data[i].astype("string")
                if self.subpop is not None and i in self.subpop:
                    continue
                self.data[i].fillna(".", inplace=True)
                if "" in self.data[i].values:
                    raise DataException(
                        "Wrong format: WHO 2016 input uses 'N' to denote "
                        "absence instead of ''.  Please change your coding"
                        "first."
                    )
                nomiss_tmp = self.data[i].isin(["Y", "y", "N", "n"]).copy()
                self.data.loc[~nomiss_tmp, i] = "."
        if self.no_is_missing:
            self.data.replace("", ".", inplace=True)

    def _check_args(self):
        # column names converted to lowercase on instantiation
        if self.data_type == "WHO2016" and "i183o" in list(self.data):
            warnings.warn("Due to the inconsistent names in the early version "
                          "of InterVA5, the indicator 'i183o' has been " 
                          "renamed as 'i183a'.")
            self.data.rename(columns={"i183o": "i183a"}, inplace=True)

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
                raise ArgumentException(
                    f"InSilicoVA unable to create {self.directory}:\n{exc}")
        if self.jump_scale is None:
            raise ArgumentException("No jump scale provided for Normal prior")
        if None in [self.n_sim, self.thin, self.burnin]:
            raise ArgumentException(
                "Length of chain/thinning/burn-in not specified")
        # Not tested yet
        # if self.keep_probbase_level and self.probbase_by_symp_dev:
        #     raise ArgumentException(
        #         "")

    def _initialize_data_dependencies(self):
        """Set up probbase SCI and cause list. """
        if self.data_type == "WHO2012":
            self.probbase = get_vadata("probbaseV3", verbose=False)
            self.probbase.replace({"sk_les": "skin_les"}, inplace=True)
            self.causetext = get_vadata("causetext", verbose=False)
        elif self.data_type == "WHO2016":
            if self.sci is None:
                self.probbase = get_vadata("probbaseV5", verbose=False)
                probbase_version = self.probbase.iat[0, 2]
                print(f"Using Probbase version: {probbase_version}\n")
            else:
                if self.sci.shape != (354, 87) or not isinstance(self.sci, pd.DataFrame):
                    raise ArgumentException(
                        "Error: invalid sci (must be data frame with "
                        "354 rows and 87 columns.\n")
                self.probbase = self.sci.copy()
            self.causetext = get_vadata("causetextV5", verbose=False)

        self.probbase.fillna(".", inplace=True)
        self.probbase = self.probbase.to_numpy(dtype=str)
        self.prob_orig = self.probbase[1:, 20:81].copy()
        self.negate = np.array([False] * self.prob_orig.shape[0])
        if self.groupcode:
            self.causetext.drop(columns=list(self.causetext)[1], inplace=True)
        else:
            self.causetext.drop(columns=list(self.causetext)[2], inplace=True)

    def _setup_subpop(self):
        """Set subpop attribute."""
        if isinstance(self.subpop, pd.Series):
            self.subpop.fillna("missing", inplace=True)
            self.subpop = self.subpop.astype("string")
        elif isinstance(self.subpop, (str, list)):
            try:
                self.subpop = self.data[self.original_subpop].copy()
            except KeyError as exc:
                raise ArgumentException(
                    f"Error: invalid subpop name specification:\n{exc}")
            self.subpop.fillna("missing", inplace=True)
            self.subpop = self.subpop.astype("string")
            if self.subpop.ndim > 1:
                self.subpop = self.subpop.apply(
                    lambda x: "_".join(x.to_list()),
                    axis=1
                )

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
                raise ArgumentException(
                    "Minimum level (min_level) not specified")
            tab_min = np.array([1, 0.8, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005,
                                0.002, 0.001, 0.0005, 0.0001, 0.00001,
                                min_level])
            return tab_min
        else:
            if table_num_dev is None:
                raise ArgumentException(
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
            raise ArgumentException(
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
                raise ArgumentException(
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

    def _cond_initiate(self,
                       probbase_order: np.ndarray,
                       exp_ini: bool,
                       inter_ini: bool,
                       probbase_min: float,
                       probbase_max: float) -> np.ndarray:
        """Randomly initialize probbase from order matrix.

        :param probbase_order: order matrix
        :type probbase_order: numpy.ndarray
        :param exp_ini: initialize to exponential of uniform
        :type exp_ini: bool
        :param inter_ini: initialize to InterVA probbase values
        :type inter_ini: bool
        :param probbase_min: minimum of probbase values
        :type probbase_min: float
        :param probbase_max: maximum of probbase values
        :type probbase_max: float
        :return: new probbase matrix
        :rtype: numpy.ndarray
        """
        trunc_min = probbase_min
        trunc_max = probbase_max
        if probbase_min == 0:
            trunc_min = 0.01
        if probbase_max == 1:
            trunc_max = 0.99
        n_levels = int(probbase_order.max())
        levels = np.unique(probbase_order)
        levels = np.sort(levels)[::-1]
        if exp_ini:
            random_levels = np.random.uniform(low=np.log(trunc_min),
                                              high=np.log(trunc_max),
                                              size=n_levels)
            random_levels = np.exp(random_levels)
        else:
            random_levels = np.random.uniform(low=trunc_min,
                                              high=trunc_max,
                                              size=n_levels)
        random_levels = np.sort(random_levels)[::-1]
        if inter_ini:
            random_levels = self._interva_table(standard=True, min_level=0)
            random_levels = (random_levels * (trunc_max - trunc_min)) + trunc_min
        prob_random = probbase_order.copy()
        for i in range(n_levels):
            prob_random[probbase_order == levels[i]] = random_levels[i]

        return prob_random

    def _remove_bad(self,
                    is_numeric: bool,
                    version5: bool = True) -> None:
        """Function to remove data with no sex/age indicators

        :param is_numeric: Indicator if the input is already in numeric format.
        :type is_numeric: bool
        :param version5: Indicator for InterVA5 input (vs. InterVA4)
        :type version5: bool
        """
        if not is_numeric:
            data_num = np.zeros(self.data.shape)
            data_num[self.data.isin(["Y", "y"])] = 1
        else:
            data_num = self.data.copy()
        drop_rows = set()

        if version5:
            age_cols = range(5, 12)
            sex_cols = range(3, 5)
            ind_cols = range(20, 328)
        else:
            age_cols = range(1, 8)
            sex_cols = range(8, 10)
            ind_cols = range(22, 223)

        for i in range(self.data.shape[0]):
            if sum(data_num[i, age_cols]) < 1:
                drop_rows.add(i)
                tmp_id = self.data.iat[i, 0]
                self.error_log[tmp_id].append(
                    "Error in age indicator: not specified")
            if sum(data_num[i, sex_cols]) < 1:
                drop_rows.add(i)
                tmp_id = self.data.iat[i, 0]
                self.error_log[tmp_id].append(
                    "Error in sex indicator: not specified")
            if sum(data_num[i, ind_cols]) < 1:
                drop_rows.add(i)
                tmp_id = self.data.iat[i, 0]
                self.error_log[tmp_id].append(
                    "Error in indicators: no symptoms specified")
        drop_rows = list(drop_rows)
        self.data.drop(drop_rows, axis=0, inplace=True)
        if isinstance(self.subpop, (pd.DataFrame, pd.Series)):
            self.subpop.drop(drop_rows, axis=0, inplace=True)

    def _datacheck(self):
        """Run InterVA5 data checks on input and store log in vacheck_log. """
        # TODO: might need to allow this function to accept PYQT5 object for
        #       updating a GUI progress bar
        checked_input = np.zeros(self.data.shape)
        checked_input[self.data.isin(["Y", "y"])] = 1
        checked_input[~self.data.isin(["Y", "y", "N", "n"])] = np.nan
        checked_input = pd.DataFrame(checked_input, columns=self.data.columns)
        checked_input["ID"] = self.data["ID"].copy()
        checked_list = []
        pb = self.probbase.copy()

        for i in range(checked_input.shape[0]):
            checked_results = datacheck5(va_input=checked_input.iloc[i],
                                         va_id=checked_input.at[i, "ID"],
                                         probbase=pb,
                                         insilico_check=True)
            self.vacheck_log["first_pass"].extend(
                checked_results["first_pass"])
            self.vacheck_log["second_pass"].extend(
                checked_results["second_pass"])
            checked_list.append(checked_results["output"])
        self.data = pd.DataFrame(checked_list)

    def _remove_ext(self,
                    csmf_orig,
                    is_numeric,
                    external_causes,
                    external_symps):
        """
        Method to remove external causes/symptoms and assign deterministic
        causes.

        :param csmf_orig: probbase prior
        :type csmf_orig: list or numpy.ndarray (vector)
        :param is_numeric: Indicator for if data are numeric
        :type is_numeric: bool
        :param external_causes: indicator of elements that are external causes
        :type external_causes: list/numpy.ndarray (vector)
        :param external_symps: indicator for data columns with external symptoms
        :type external_symps: list/numpy.ndarray (vector)
        """
        n_all = self.data.shape[0]
        ext_data = self.data.iloc[:, external_symps + 1].copy()
        if is_numeric:
            neg = 0
            pos = 1
        else:
            neg = "N"
            pos = "Y"
        ext_where = (ext_data == pos).any(axis=1)
        ext_data = ext_data.loc[ext_where, :]
        self.ext_id = self.data.loc[ext_where, "ID"].copy()
        self.ext_sub = self.subpop.loc[ext_where].copy()
        prob_sub = self._change_inter(
            self.prob_orig[np.ix_(external_symps, external_causes)])
        csmf_sub = self._change_inter(
            csmf_orig[external_causes])
        self.prob_orig = np.delete(self.prob_orig, external_symps, axis=0)
        self.prob_orig = np.delete(self.prob_orig, external_causes, axis=1)
        self.negate = np.delete(self.negate, external_symps)
        if external_symps.shape[0] > 0:
            self.data.drop(
                self.data.columns[external_symps + 1], axis=1, inplace=True
            )
        if ext_where.sum() > 0:
            n_row = ext_data.shape[0]
            n_col = external_causes.shape[0]
            probs = np.ones((n_row, n_col))
            for i in range(n_row):
                for j in range(n_col):
                    symp_pos = ext_data.iloc[i] == pos
                    probs[i, j] = csmf_sub[j] * (prob_sub[symp_pos, j].prod())
                probs[i, :] = probs[i, :]/probs[i, :].sum()
            self.ext_prob = probs
            self.data = self.data[~ext_where]
        else:
            if self.subpop is not None:
                self.ext_csmf = []
                for i in range(4):
                    pass


