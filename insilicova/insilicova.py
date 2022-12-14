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

VALID_DATA_TYPES = ("WHO2016", "WHO2012")
VALID_EXCLUDE_IMPOSSIBLE_CAUSE = (
    "all", "interva", "none", "subset", "subset2")


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
                 cond_prob: Union[None, pd.DataFrame] = None,
                 cond_prob_num: Union[None, pd.DataFrame] = None,
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
                 impossible_combination: Union[None, pd.DataFrame] = None,
                 no_is_missing: bool = False,
                 indiv_ci: Union[None, float] = None,
                 groupcode: bool = False,
                 run: bool = True):

        # instance attributes from arguments
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
        self.run = run

        # retain copies of original input -- IS THIS NEEDED?
        self.original_data = data.copy()
        if isinstance(subpop, str):
            self.original_subpop = subpop
        elif subpop is not None:
            self.original_subpop = subpop.copy()
        else:
            self.original_subpop = None

        # attributes set by self._initialize_data_dependencies()
        self.probbase = None
        self.probbase_colnames = None
        self.causetext = None
        # attributes set by self._prep_data() or self._prep_data_custom()
        self.sys_prior = None
        self.prob_orig = None
        self.prob_orig_colnames = None
        self.negate = None
        self.va_labels = None
        self.va_causes = None
        self.external_causes = None
        self.external_symps = None
        self.subpop_order_list = None
        self.probbase_dev = None  # not yet implemented
        self.gstable_dev = None  # not yet implemented
        # attributes set by self.datacheck()
        self.error_log = defaultdict(list)
        self.data_checked = None
        self.vacheck_log = {"first_pass": [], "second_pass": []}
        # attributes set by self._remove_ext()
        self.ext_sub = None
        self.ext_id = None
        self.ext_cod = None
        self.ext_csmf = None
        self.ext_prob = None
        # attributes set by self._initialize_numerical_matrix()
        self.cond_prob_true = None
        self.prob_order = None
        self.table_dev = None  # not yet implemented
        self.table_num_dev = None  # not yet implemented
        # attributes set by self._check_data_dimensions()
        self.va_causes_current = None
        # attributes set by self._check_impossible_pairs()
        self.impossible = None
        # attributes set by self._physician_codes
        self.c_phy = None
        self.assignment = None
        self.va_causes_broader = None

        # results (should a run() method return an InSilicoVA data object?)
        self.results = None

        # attributes for future use? (unused in R code)
        # self.probbase_by_symp_dev
        self.customization_dev = False
        self.method = "normal"
        self.alpha_scale = None
        # self.n_level_dev = None
        # if self.n_level_dev is not None:
        #     self.n_level = self.n_level_dev
        self.n_level = 15

        if self.run:
            self._run()

    def _run(self):
        self._change_data_coding()
        self._check_args()
        self._initialize_data_dependencies()
        if self.subpop is not None:
            self._extract_subpop()
        if self.customization_dev:
            self._prep_data_custom()
        else:
            self._prep_data()
        self._standardize_upper()
        if self.datacheck:
            self._datacheck()
        if self.external_sep:
            self._remove_external_causes()
        if self.data.shape[0] > 0:
            self._check_missing_all()
            if not self.datacheck_missing and self.datacheck:
                warnings.warn(
                    "check missing after removing symptoms is disabled...\n",
                    UserWarning)
            self._initialize_numerical_matrix()
            self._check_data_dimensions()
            self._check_impossible_pairs()
            self._physician_codes()

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
        """Check for valid arguments."""
        if self.data_type not in VALID_DATA_TYPES:
            valid_str = ", ".join(VALID_DATA_TYPES)
            raise ArgumentException(
                f"Error: data_type must be one of the following: {valid_str}\n")
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
                if (self.sci.shape != (354, 87) or
                        not isinstance(self.sci, pd.DataFrame)):
                    raise ArgumentException(
                        "Error: invalid sci (must be DataFrame with "
                        "354 rows and 87 columns.\n")
                self.probbase = self.sci.copy()
            self.causetext = get_vadata("causetextV5", verbose=False)
        self.probbase.fillna(".", inplace=True)
        self.probbase_colnames = np.array(list(self.probbase))
        self.probbase = self.probbase.to_numpy(dtype=str)
        if self.groupcode:
            self.causetext.drop(columns=list(self.causetext)[1], inplace=True)
        else:
            self.causetext.drop(columns=list(self.causetext)[2], inplace=True)

    def _extract_subpop(self):
        """Set subpop attribute."""
        if isinstance(self.subpop, (str, list)):
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
        if len(self.subpop.fillna(".").unique()) == 1:
            warnings.warn("Only one level in subpopulation, running the  "
                          "dataset as on population.",
                          UserWarning)
            self.subpop = None

    def _prep_data(self):  # (down to line 976 in R code)
        """Set up probbase, check column names, and perform misc data prep."""
        if self.data_type == "WHO2012":
            pbcols = slice(16, 76)
        elif self.data_type == "WHO2016":
            pbcols = slice(20, 81)
        self.sys_prior = self._change_inter(
            self.probbase[0, pbcols], order=False, standard=True)
        self.prob_orig = self.probbase[1:, pbcols].copy()
        self.prob_orig_colnames = self.probbase_colnames[pbcols,].copy()
        self.negate = np.array([False] * self.prob_orig.shape[0])
        if self.data_type == "WHO2016":
            subst_vector = self.probbase[1:, 5]
            self.negate[subst_vector == "N"] = True
        if self.data.shape[1] != self.probbase.shape[0]:
            self.data.rename(columns=dict(
                zip(self.data.columns[1:],
                    self.data.columns[1:].str.lower())),
                inplace=True)
            if self.data_type == "WHO2012":
                correct_names = self.probbase[1:, 1]
            elif self.data_type == "WHO2016":
                correct_names = self.probbase[1:, 0]
            data_colnames = np.array(list(self.data))
            exist = np.in1d(correct_names, data_colnames)
            if (exist == False).sum() > 0:
                missing_cols = correct_names[~exist].tolist()
                raise DataException(
                    "Error: invalid data input format.  Symptom(s) not "
                    "found: "
                    ", ".join(missing_cols)
                )
            else:
                get_cols = [self.data.columns[0]]
                get_cols.extend(correct_names.tolist())
                self.data = self.data[get_cols]
        if self.data_type == "WHO2012":
            randomva1 = get_vadata("randomva1", verbose=False)
            self.va_labels = list(randomva1)
            self.va_causes = self.causetext.iloc[3:63, 1].copy()
            self.external_causes = np.arange(40, 51)
            self.external_symps = np.arange(210, 222)
        elif self.data_type == "WHO2016":
            randomva5 = get_vadata("randomva5", verbose=False)
            self.va_labels = list(randomva5)
            self.va_causes = self.causetext.iloc[3:64, 1].copy()
            self.external_causes = np.arange(49, 60)
            self.external_symps = np.arange(19, 38)
        self.va_causes.reset_index(drop=True, inplace=True)
        count_change_label = 0
        for i in range(len(self.va_labels)):
            data_lab = list(self.data)[i]
            va_lab = self.va_labels[i]
            if data_lab.lower() != va_lab.lower():
                warnings.warn(f"Input column '{data_lab}' does not match "
                              f"InterVA standard: '{va_lab}'.",
                              UserWarning)
                count_change_label += 1
        if count_change_label > 0:
            warnings.warn(f"{count_change_label} column names changed in "
                          "input. \n If the change is undesirable, please "
                          "change the input to match InterVA input format.\n",
                          UserWarning)
            self.data.columns = self.va_labels
        if self.cond_prob is not None:
            self.exclude_impossible_causes = "none"
            # self.va_causes = list(self.cond_prob)
            self.va_causes = pd.Series(list(self.cond_prob))
            cond_prob = self.cond_prob.fillna(".")
            self.prob_orig_colnames = np.array(list(cond_prob))
            self.prob_orig = cond_prob.to_numpy(dtype=str)
        if self.cond_prob_num is not None:
            self.update_cond_prob = False
            # self.va_causes = list(self.cond_prob_num)
            self.va_causes = pd.Series(list(self.cond_prob_num))
            cond_prob_num = self.cond_prob_num.fillna(".")
            self.prob_orig_colnames = np.array(list(cond_prob_num))
            self.prob_orig = cond_prob_num.to_numpy(dtype=str)

        if self.datacheck:
            v5 = self.data_type == "WHO2016"
            self._remove_bad(is_numeric=self.is_numeric, version5=v5)
            # if self.subpop is not None:
            #     self.subpop_order_list = list(self.subpop.unique())
            #     self.subpop_order_list.sort()
        else:
            self.error_log = None

        if self.subpop is not None:
            self.subpop_order_list = list(self.subpop.unique())
            self.subpop_order_list.sort()

    def _prep_data_custom(self):
        """Data prep with developer customization."""
        self.sys_prior = np.ones(self.probbase_dev.shape[1])
        correct_names = self.probbase_dev.index.tolist()
        data_colnames = np.array(list(self.data))
        exist = np.in1d(correct_names, data_colnames)
        if False in exist:
            raise ArgumentException(
                "Error: invalid data input, no matching symptoms found. \n")
        else:
            get_cols = [self.data.columns[0]]
            get_cols.extend(correct_names.tolist())
            self.data = self.data[get_cols]
        self.prob_orig_colnames = np.array(list(self.probbase_dev))
        self.prob_orig = self.probbase_dev.to_numpy(dtype=str, copy=True)
        self.va_labels = list(self.data)
        self.va_causes = self.gstable_dev
        self.external_causes = None
        self.error_log = None

        if self.subpop is not None:
            self.subpop_order_list = list(self.subpop.unique())
            self.subpop_order_list.sort()

    def _standardize_upper(self):
        """Change inputs (y, n) to uppercase."""
        self.data.replace({"y": "Y", "n": "N"}, inplace=True)

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
                new_tab[-1] = new_tab[-2] / 2
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
            random_levels = (random_levels * (
                    trunc_max - trunc_min)) + trunc_min
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

        if self.data_type == "WHO2012":
            # raise ArgumentException(
            #     "Data checks are not yet available for WHO 2012 data."
            # )
            return None
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
        checked = pd.DataFrame(checked_list)
        if sum(self.negate) > 0:
            shifted_index = np.where(self.negate)[0] + 1
            checked.iloc[:, shifted_index] = 1 - checked.iloc[:, shifted_index]

        checked.fillna(-9, inplace=True)

        for i in range(1, checked.shape[1]):
            self.data.iloc[checked.iloc[:, i] == 1, i] = "Y"
            self.data.iloc[checked.iloc[:, i] == 0, i] = ""
            self.data.iloc[checked.iloc[:, i] == -1, i] = "."
            self.data.iloc[checked.iloc[:, i] == -9, i] = "."
            self.data_checked = self.data.copy()
        if self.data_type == "WHO2016":
            self.data_checked.replace({"Y": "y", "": "n", ".": "-"},
                                      inplace=True)
        # R has a check at this point (around lined 1040) for self.data being
        # an empty data frame but I don't see how this could happen (maybe
        # after remove bad, or with the WHO2012 datacheck in Java?)

    def _remove_ext(self):
        """
        Method to remove external causes/symptoms and assign deterministic
        causes for data_type == 'WHO2012'.
        """
        pass

    def _remove_ext_v5(self):
        """
        Method to remove external causes/symptoms and assign deterministic
        causes for data_type == 'WHO2016'.
        """
        csmf_orig = self.probbase[0, 20:]
        n_all = self.data.shape[0]
        n_ext_causes = len(self.external_causes)
        ext_data = self.data.iloc[:, self.external_symps + 1].copy()
        if self.is_numeric:
            pos = 1
        else:
            pos = "Y"
        ext_where = (ext_data == pos).any(axis=1)
        ext_data = ext_data.loc[ext_where, :]
        self.ext_id = self.data.loc[ext_where, "ID"].copy()
        if self.subpop is not None:
            n_subpops = len(self.subpop_order_list)
            self.ext_sub = self.subpop.loc[ext_where].copy()
        prob_sub = self._change_inter(
            self.prob_orig[np.ix_(self.external_symps, self.external_causes)])
        csmf_sub = self._change_inter(
            csmf_orig[self.external_causes])
        self.prob_orig = np.delete(self.prob_orig, self.external_symps, axis=0)
        self.prob_orig = np.delete(self.prob_orig, self.external_causes,
                                   axis=1)
        self.prob_orig_colnames = np.delete(self.prob_orig_colnames,
                                            self.external_causes,
                                            axis=0)
        self.negate = np.delete(self.negate, self.external_symps)
        if self.external_symps.shape[0] > 0:
            self.data.drop(
                self.data.columns[self.external_symps + 1],
                axis=1, inplace=True)
        if ext_where.sum() > 0:
            n_row = ext_data.shape[0]
            n_col = self.external_causes.shape[0]
            probs = np.ones((n_row, n_col))
            for i in range(n_row):
                for j in range(n_col):
                    symp_pos = ext_data.iloc[i] == pos
                    symp_pos = symp_pos.to_numpy(dtype=bool)
                    probs[i, j] = csmf_sub[j] * (prob_sub[symp_pos, j].prod())
                probs[i, :] = probs[i, :] / sum(probs[i, :])
            self.ext_prob = probs
            self.data = self.data[~ext_where]
            if self.subpop is not None:
                self.ext_csmf = []
                for i in range(n_subpops):
                    self.ext_csmf.append(np.zeros(n_ext_causes))
                    # TODO: need a test where a subpop has 0 external symptoms
                    row_subpop_i = self.ext_sub == self.subpop_order_list[i]
                    row_subpop_i = row_subpop_i.to_numpy(dtype=bool)
                    ext_prob_temp = self.ext_prob[row_subpop_i, :]
                    if ext_prob_temp.shape[0] > 0:
                        self.ext_csmf[i] = (ext_prob_temp.mean(axis=0) *
                                            ext_prob_temp.shape[0]) / sum(
                            self.subpop == self.subpop_order_list[i])
            else:
                self.ext_csmf = self.ext_prob.mean(axis=0) * len(
                    self.ext_id) / n_all
            if ext_where.sum() > 0 and self.subpop is not None:
                self.subpop = self.subpop[~ext_where]
        else:
            if self.subpop is not None:
                self.ext_csmf = []
                for i in range(n_subpops):
                    self.ext_csmf.append(np.zeros(n_ext_causes))
            else:
                self.ext_csmf = np.zeros(n_ext_causes)
            self.ext_prob = np.zeros(
                (ext_data.shape[0], n_ext_causes))

    def _remove_external_causes(self):
        if self.data_type == "WHO2012":
            self._remove_ext()
        else:
            self._remove_ext_v5()
        if self.data.shape[0] == 0:
            warnings.warn("All deaths are assigned to external causes.  "
                          "A DataFrame of external causes is returned instead "
                          "of an InSilico object.\n",
                          UserWarning)
            if self.data_type == "WHO2012":
                pass
            else:
                self.results = pd.DataFrame({"ID": self.ext_id})
                names_external_causes = self.va_causes[self.external_causes]
                for i in range(len(names_external_causes)):
                    cause_i = names_external_causes.iloc[i]
                    self.results[cause_i] = self.ext_prob[:, i]

    def _check_missing_all(self):
        """
        Remove missing columns from the data (i.e., all values in column are
        missing) and probbase.
        """
        missing_all = []
        n_iterations = self.data.shape[1] - 1  # ignore ID column
        for i in range(n_iterations):
            # increment index by 1 for data (but not for probbase!)
            if all(self.data.iloc[:, (i + 1)] == "."):
                missing_all.append(i)
        if len(missing_all) > 0:
            missing_all_plus1 = np.array(missing_all) + 1
            missing_all_names = ", ".join(
                self.probbase[missing_all_plus1,
                              int(self.data_type == "WHO2012")])
            warnings.warn(f"{len(missing_all)} symptoms missing completely "
                          "and added to missing list. \n List of missing "
                          f"symptoms: {missing_all_names}\n",
                          UserWarning)
            drop_col_names = list(self.data.iloc[:, missing_all_plus1])
            self.data.drop(columns=drop_col_names, inplace=True)
            self.prob_orig = np.delete(self.prob_orig, missing_all, axis=0)
            if self.negate is not None:
                self.negate = np.delete(self.negate, missing_all, axis=0)

    def _initialize_numerical_matrix(self):
        if not self.customization_dev:
            if self.cond_prob_num is None:
                self.prob_order = self._change_inter(self.prob_orig,
                                                     order=True,
                                                     standard=True)
                self.cond_prob_true = self._change_inter(self.prob_orig,
                                                         order=False,
                                                         standard=True)
            else:
                self.cond_prob_true = self.prob_orig.copy()
                self.prob_order = np.ones(self.prob_orig.shape)
        else:
            if self.update_cond_prob:
                self.prob_order = self._change_inter(
                    self.prob_orig,
                    order=True,
                    standard=False,
                    table_dev=self.table_dev,
                    table_num_dev=self.table_num_dev)
                self.cond_prob_true = self._change_inter(
                    self.prob_orig,
                    order=False,
                    standard=False,
                    table_dev=self.table_dev,
                    table_num_dev=self.table_num_dev)
            else:
                self.cond_prob_true = self.prob_orig.copy()
                self.prob_order = np.ones(self.prob_orig.shape)

    def _check_data_dimensions(self):
        if self.data.shape[0] <= 1:
            raise DataException(
                "Not enough observations in the sample to run InSilicoVA.\n"
            )
        n_symptoms_ok = (self.data.shape[1] - 1) == self.cond_prob_true.shape[
            0]
        if n_symptoms_ok is False:
            raise DataException(
                "Number of symptoms is not right. \n"
            )
        if self.external_sep:
            self.va_causes_current = self.va_causes[
                self.external_causes].copy()
        else:
            self.va_causes_current = self.va_causes.copy()

    def _check_impossible_pairs(self):
        """Check impossible pairs of symptoms and causes (for the first nine
        demographic symptoms (7 age + 2 gender)).  Also, the value saved is
        the index (starting from 0?)

        format: (ss, cc, 0) if P(cc | ss = y) = 0
                (ss, cc, 1) if P(cc | ss = n) = 0
        """
        impossible = []
        data_colnames = list(self.data)[1:]
        impossible_colnames = ["symptom", "probbase_col", "value"]
        if self.impossible_combination is not None:
            n_combos = self.impossible_combination.shape[0]
            impossible_colnames = list(self.impossible_combination)

        if self.exclude_impossible_causes != "none" and self.customization_dev is False:
            if self.exclude_impossible_causes in ["subset", "subset2"]:
                if self.data_type == "WHO2012":
                    demog_set = ["elder", "midage", "adult", "child", "under5",
                                 "infant", "neonate", "male", "female",
                                 "magegp1", "magegp2", "magegp3", "died_d1",
                                 "died_d23", "died_d36", "died_w1", "no_life"]
                else:
                    demog_set = ["i019a", "i019b", "i022a", "i022b", "i022c",
                                 "i022d", "i022e", "i022f", "i022g", "i022h",
                                 "i022i", "i022j", "i022k", "i022l", "i022m",
                                 "i022n", "i114o"]
            else:
                demog_set = list(self.data)[1:]

            demog_index = [data_colnames.index(i) for i in demog_set if
                           i in data_colnames]
            n_causes = self.cond_prob_true.shape[1]
            for ss in demog_index:
                for cc in range(n_causes):
                    if self.cond_prob_true[ss, cc] == 0:
                        impossible.append([int(cc), int(ss), 0])
                    if (self.cond_prob_true[ss, cc] == 1 and
                            self.exclude_impossible_causes.lower() != "interva"):
                        impossible.append([int(cc), int(ss), 1])

            if self.exclude_impossible_causes == "subset2":
                # add pregnancy death fix
                add_impossible = pd.DataFrame(
                    {impossible_colnames[0]: ["i310o"] * 9})
                col1_values = ["b_0901", "b_0902", "b_0903", "b_0904",
                               "b_0905", "b_0906", "b_0907", "b_0908", "b_0999"]
                add_impossible[impossible_colnames[1]] = col1_values
                add_impossible[impossible_colnames[2]] = 1

                if self.impossible_combination is None:
                    self.impossible_combination = add_impossible
                else:
                    self.impossible_combination = pd.concat(
                        [self.impossible_combination, add_impossible],
                        ignore_index=True)
                # add prematurity fix
                ss = None
                if self.data_type == "WHO2012":
                    if "born_38" in data_colnames:
                        ss = data_colnames.index("born_38")
                        val_only_prem = 0
                        val_not_prem = 1
                else:
                    if "i367a" in data_colnames:
                        ss = data_colnames.index("i367a")
                        # if negated, then reverse
                        val_only_prem = int(self.negate[ss] is True)
                        val_not_prem = int(self.negate[ss] is False)
                cc_only_prem = self.va_causes_current == "Prematurity"
                if ss is not None and sum(cc_only_prem) == 1:
                    cc_only_index = self.va_causes_current[cc_only_prem].index[
                        0]
                    impossible.append([cc_only_index, ss, val_only_prem])
                cc_not_prem = self.va_causes_current == "Birth asphyxia"
                if ss is not None and sum(cc_not_prem) == 1:
                    cc_not_index = self.va_causes_current[cc_not_prem].index[0]
                    impossible.append([cc_not_index, ss, val_not_prem])

            if self.impossible_combination is not None:
                for ss, cc, vv in self.impossible_combination.itertuples(
                        index=False):
                    if ss in data_colnames and cc in self.prob_orig_colnames:
                        val = 1 - int(vv)
                        ss_index = data_colnames.index(ss)
                        cc_index = \
                            np.argwhere(self.prob_orig_colnames == cc)[0][0]
                        impossible.append([cc_index, ss_index, val])

        elif self.impossible_combination is not None:
            for ii in self.impossible_combination.itertuples(index=False):
                if ii[0] in data_colnames and ii[1] in self.prob_orig_colnames:
                    # val = 1 - int(vv)
                    ss_index = data_colnames.index(ii[0])
                    cc_index = \
                        np.argwhere(self.prob_orig_colnames == ii[1])[0][0]
                    # impossible.append([cc_index, ss_index, val])
                    impossible.append([cc_index, ss_index])
        else:
            # java checks if impossible has 3 columns and sets check impossible
            # cause flag to False if it has 4 columns
            impossible.append([0, 0, 0, 0])

        self.impossible = np.array(impossible, dtype=int)

    def _physician_codes(self):
        if self.phy_code is not None:
            pass
        else:
            # if no physician coding, everything is unknown
            self.c_phy = 1
            self.assignment = np.zeros((self.data.shape[0], 1))
            self.assignment[:, 0] = 1
            self.va_causes_broader = range(len(self.va_causes_current))
