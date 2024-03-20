# -*- coding: utf-8 -*-

"""
insilicova.insilicova
-------------------

This module contains the class for the InSilicoVA algorithm.
"""

from __future__ import annotations
from .exceptions import (ArgumentException, DataException, HaltGUIException,
                         InSilicoVAException)
from .utils import get_vadata
from ._sampler._sampler import Sampler
from .diag import csmf_diag
from .structures import InSilico, InSilicoAllExt
from typing import Union, Dict, TYPE_CHECKING
if TYPE_CHECKING:
    import PyQt5
from vacheck.datacheck5 import datacheck5
import warnings
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from platform import system

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
                 run: bool = True,
                 openva_app: Union[None,
                                   PyQt5.QtCore.pyqtSignal] = None,
                 state: Union[None,
                              PyQt5.QtCore.pyqtSignal] = None,
                 gui_ctrl: dict = {"break": False}) -> None:

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
        self.openva_app = openva_app
        self.state = state
        self.gui_ctrl = gui_ctrl

        # retain copies of original input
        self.original_data: pd.DataFrame = data.copy()
        if isinstance(subpop, str):
            self.original_subpop = subpop
        elif subpop is not None:
            self.original_subpop = subpop.copy()
        else:
            self.original_subpop = None

        # attributes set by self._initialize_data_dependencies()
        self._probbase: np.ndarray
        self._probbase_colnames: np.ndarray
        self._causetext: pd.DataFrame
        # attributes set by self._prep_data() or self._prep_data_custom()
        self._sys_prior: np.ndarray
        self._prob_orig: np.ndarray
        self._prob_orig_colnames: np.ndarray
        self._negate: Union[None, np.ndarray] = None
        self._va_labels: list
        self._va_causes: pd.Series
        self._external_causes: np.ndarray
        self._external_symps: np.ndarray
        self._subpop_order_list: list
        self._probbase_dev: pd.DataFrame  # not yet implemented
        self._gstable_dev: pd.Series  # not yet implemented
        # attributes set by self.remove_bad()
        self._error_log: Dict[str, list] = defaultdict(list)
        # attributes set by self.datacheck()
        self._data_checked: pd.DataFrame
        self._warning: Dict[str, list] = {"first_pass": [], "second_pass": []}
        # attributes set by self._remove_ext()
        self._ext_sub: pd.Series
        self._ext_id: pd.Series
        self._ext_cod: np.ndarray
        self._ext_csmf: dict
        self._ext_prob: np.ndarray
        self._ext_cod: Union[None, np.ndarray]
        # attributes set by self._check_missing_all
        self._missing_all: list
        # attributes set by self._initialize_numerical_matrix()
        self._cond_prob_true: np.ndarray
        self._prob_order: np.ndarray
        # _table_dev and _table_num_dev are not yet implemented
        self._table_dev: np.ndarray = np.array([chr(x) for x in range(65, 80)])
        self._table_num_dev: np.ndarray = np.arange(1, 16) / 15
        # attributes set by self._check_data_dimensions()
        self._va_causes_current: pd.Series
        # attributes set by self._check_impossible_pairs()
        self._impossible: np.ndarray
        # attributes set by self._physician_codes
        self._c_phy: int = 1
        self._assignment: np.ndarray
        self._va_causes_broader: range
        # attributes set by self._prior_truncated_beta
        self._prior_b_cond: float
        # attributes set by self._get_subpop_info()
        self._subpop_numeric: np.ndarray
        self._sublist: dict = {}
        # _subbelong is the same as _subpop_numeric
        # self._subbelong = None
        # attributes set by self._initialize_indicator_matrix()
        self._indic: np.ndarray
        self._id: pd.Series
        self._contains_missing: int
        # attributes set by self._initialize_parameters()
        self._mu: np.ndarray
        self._sigma2: float
        self._cond_prob: np.ndarray
        self._dist: np.ndarray
        self._n_level: int
        self._N_level: int
        self._level_exist: np.ndarray
        # self._level_values: np.ndarray
        # self._prior_a: np.ndarray
        # self._prior_b: np.ndarray
        # self._jumprange = self.jump_scale
        self._pool: int
        self._n_gibbs = self.n_sim
        # self._burn = self.burnin
        self._n_sub: int
        self._mu_last: np.ndarray
        self._sigma2_last: np.ndarray
        self._theta_last: np.ndarray
        self._keep_prob: bool = not self.update_cond_prob
        # attributes set by self._sample_posterior()
        self._posterior_results: Dict
        # attributes set by self._prepare_results()
        self.results: InSilico

        # attributes for future use? (unused in R code)
        self._probbase_by_symp_dev: bool = False
        self._customization_dev: bool = False
        self._method: str = "normal"
        self._alpha_scale: float = 1.0  # not yet implemented
        self.n_level_dev: int = 15  # not yet implemented
        # if self.n_level_dev is not None:
        #     self._n_level = self.n_level_dev
        self._n_level: int = 15

        if self.run:
            self._run()

    def __str__(self):
        if hasattr(self, "results"):
            if isinstance(self.results, InSilico):
                n_iter = (self.n_sim - self.burnin) / self.thin
                msg = ("InSilicoVA:\n"
                       f"{len(self.results.id)} deaths processed\n"
                       f"{self.n_sim} iterations performed, with the first "
                       f"{self.burnin} iterations discarded\n"
                       f"{int(n_iter)} iterations saved after thinning\n")
            else:
                msg = ("All deaths are assigned external causes.\n"
                       "Only external causes returned and IDs are returned "
                       "(not an InSilico object)\n")
        else:
            n_iter = (self.n_sim - self.burnin) / self.thin
            msg = f"""
            InSilicoVA:
            {self.data.shape[0]} deaths loaded
            {self.n_sim} iterations requested (the first {self.burnin}
            iterations will be discarded {int(n_iter)} iterations will saved
            after thinning (no results yet, need to use run() method)
            """
        return msg

    def _run(self):
        if self.state:
            self.state.emit("preparing data...")
        self._change_data_coding()
        self._check_args()
        self._initialize_data_dependencies()
        if self.subpop is not None:
            self._extract_subpop()
        if self._customization_dev:
            self._prep_data_custom()
        else:
            self._prep_data()
        if self.data.shape[0] == 0:
            warnings.warn(
                "NO VALID VA RECORDS (datacheck procedure invalidated all "
                "deaths)!  \nUnable to analyze data set.  Use get_errors() "
                "method for more details.",
                UserWarning)
            return None
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
            if self.phy_debias is not None:
                self.phy_code = self.phy_debias["code_debias"]
            self._physician_codes()
            self._prior_truncated_beta()
            if self.subpop is not None:
                self._get_subpop_info()
            self._initialize_indicator_matrix()
            self._initialize_parameters()
            if self.state:
                msg = "Running InSilicoVA..."
                if self.auto_length:
                    msg += ("\n(progress bar may reset once or twice since "
                            "auto increase is True)")
                self.state.emit(msg)
            self._sample_posterior()
            self._prepare_results()

    def _change_data_coding(self):
        if self.data_type == "WHO2016":
            col_names = list(self.data)[1:]  # ignore ID
            for i in col_names:
                self.data[i] = self.data[i].astype("string")
                if self.subpop is not None and i in self.subpop:
                    continue
                # self.data[i].fillna(".", inplace=True)
                self.data[i] = self.data[i].fillna(".")
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
            val_str = ", ".join(VALID_DATA_TYPES)
            raise ArgumentException(
                f"Error: data_type must be one of the following: {val_str}\n")
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
                          "). Failing to properly _negate all such symptoms "
                          "will lead to erroneous inference.",
                          UserWarning)
            self._data_checked = None
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
        """Set up _probbase SCI and cause list. """
        if self.data_type == "WHO2012":
            df_probbase = get_vadata("probbaseV3", verbose=False)
            df_probbase.replace({"sk_les": "skin_les"}, inplace=True)
            self._causetext = get_vadata("causetext", verbose=False)
        elif self.data_type == "WHO2016":
            if self.sci is None:
                df_probbase = get_vadata("probbaseV5", verbose=False)
                probbase_version = df_probbase.iat[0, 2]
                print(f"Using Probbase version: {probbase_version}\n")
            else:
                if (self.sci.shape != (354, 87) or
                        not isinstance(self.sci, pd.DataFrame)):
                    raise ArgumentException(
                        "Error: invalid sci (must be DataFrame with "
                        "354 rows and 87 columns).\n")
                df_probbase = self.sci.copy()
            self._causetext = get_vadata("causetextV5", verbose=False)
        df_probbase.fillna(".", inplace=True)
        self._probbase_colnames = np.array(list(df_probbase))
        self._probbase = df_probbase.to_numpy(dtype=str)
        if self.groupcode:
            self._causetext.drop(columns=list(self._causetext)[1],
                                 inplace=True)
        else:
            self._causetext.drop(columns=list(self._causetext)[2],
                                 inplace=True)

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
        """Set up _probbase, check column names, and perform misc data prep."""
        if self.data_type == "WHO2012":
            pbcols = slice(16, 76)
        elif self.data_type == "WHO2016":
            pbcols = slice(20, 81)
        self._sys_prior = self._change_inter(
            self._probbase[0, pbcols], order=False, standard=True)
        self._prob_orig = self._probbase[1:, pbcols].copy()
        self._prob_orig_colnames = self._probbase_colnames[pbcols, ].copy()
        self._negate = np.array([False] * self._prob_orig.shape[0])
        if self.data_type == "WHO2016":
            subst_vector = self._probbase[1:, 5]
            self._negate[subst_vector == "N"] = True
        if self.data.shape[1] != self._probbase.shape[0]:
            self.data.rename(columns=dict(
                zip(self.data.columns[1:],
                    self.data.columns[1:].str.lower())),
                inplace=True)
            if self.data_type == "WHO2012":
                correct_names = self._probbase[1:, 1]
            elif self.data_type == "WHO2016":
                correct_names = self._probbase[1:, 0]
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
            self._va_labels = list(randomva1)
            self._va_causes = self._causetext.iloc[3:63, 1].copy()
            self._external_causes = np.arange(40, 51)
            self._external_symps = np.arange(210, 222)
        elif self.data_type == "WHO2016":
            randomva5 = get_vadata("randomva5", verbose=False)
            self._va_labels = list(randomva5)
            self._va_causes = self._causetext.iloc[3:64, 1].copy()
            self._external_causes = np.arange(49, 60)
            self._external_symps = np.arange(19, 38)
        self._va_causes.reset_index(drop=True, inplace=True)
        count_change_label = 0
        for ind, val in enumerate(self._va_labels):
            data_lab = list(self.data)[ind]
            if data_lab.lower() != val.lower():
                warnings.warn(f"Input column '{data_lab}' does not match "
                              f"InterVA standard: '{val}'.",
                              UserWarning)
                count_change_label += 1
        if count_change_label > 0:
            warnings.warn(f"{count_change_label} column names changed in "
                          "input. \n If the change is undesirable, please "
                          "change the input to match InterVA input format.\n",
                          UserWarning)
            self.data.columns = self._va_labels
        if self.cond_prob is not None:
            self.exclude_impossible_causes = "none"
            # self._va_causes = list(self.cond_prob)
            self._va_causes = pd.Series(list(self.cond_prob))
            cond_prob = self.cond_prob.fillna(".")
            self._prob_orig_colnames = np.array(list(cond_prob))
            self._prob_orig = cond_prob.to_numpy(dtype=str)
        if self.cond_prob_num is not None:
            self.update_cond_prob = False
            # self._va_causes = list(self.cond_prob_num)
            self._va_causes = pd.Series(list(self.cond_prob_num))
            cond_prob_num = self.cond_prob_num.fillna(".")
            self._prob_orig_colnames = np.array(list(cond_prob_num))
            self._prob_orig = cond_prob_num.to_numpy(dtype=str)

        if self.datacheck:
            v5 = self.data_type == "WHO2016"
            self._remove_bad(is_numeric=self.is_numeric, version5=v5)
            # if self.subpop is not None:
            #     self._subpop_order_list = list(self.subpop.unique())
            #     self._subpop_order_list.sort()
        else:
            self._error_log["datacheck"].append("False")

        if self.subpop is not None:
            self._subpop_order_list = list(self.subpop.unique())
            self._subpop_order_list.sort()

    def _prep_data_custom(self):
        """Data prep with developer customization."""
        self._sys_prior = np.ones(self._probbase_dev.shape[1])
        correct_names = self._probbase_dev.index.tolist()
        data_colnames = np.array(list(self.data))
        exist = np.in1d(correct_names, data_colnames)
        if False in exist:
            raise ArgumentException(
                "Error: invalid data input, no matching symptoms found. \n")
        else:
            get_cols = [self.data.columns[0]]
            get_cols.extend(correct_names.tolist())
            self.data = self.data[get_cols]
        self._prob_orig_colnames = np.array(list(self._probbase_dev))
        self._prob_orig = self._probbase_dev.to_numpy(dtype=str, copy=True)
        self._va_labels = list(self.data)
        self._va_causes = self._gstable_dev
        self._external_causes = None
        self._error_log["datacheck"].append("customized")

        if self.subpop is not None:
            self._subpop_order_list = list(self.subpop.unique())
            self._subpop_order_list.sort()

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
        :returns: numeric values for mapping of InterVA _probbase levels
        (grades) to probabilities
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
                    "Numerical level table (_table_num_dev) not specified")
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
                "Dimension of _probbase prior is incorrect.")
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
        _table_dev
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
                    "_table_dev and _table_num_dev have different lengths"
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
        """Randomly initialize _probbase from order matrix.

        :param probbase_order: order matrix
        :type probbase_order: numpy.ndarray
        :param exp_ini: initialize to exponential of uniform
        :type exp_ini: bool
        :param inter_ini: initialize to InterVA _probbase values
        :type inter_ini: bool
        :param probbase_min: minimum of _probbase values
        :type probbase_min: float
        :param probbase_max: maximum of _probbase values
        :type probbase_max: float
        :return: new _probbase matrix
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
                self._error_log[tmp_id].append(
                    "Error in age indicator: not specified")
            if sum(data_num[i, sex_cols]) < 1:
                drop_rows.add(i)
                tmp_id = self.data.iat[i, 0]
                self._error_log[tmp_id].append(
                    "Error in sex indicator: not specified")
            if sum(data_num[i, ind_cols]) < 1:
                drop_rows.add(i)
                tmp_id = self.data.iat[i, 0]
                self._error_log[tmp_id].append(
                    "Error in indicators: no symptoms specified")
        drop_rows = list(drop_rows)
        self.data.drop(drop_rows, axis=0, inplace=True)
        if isinstance(self.subpop, (pd.DataFrame, pd.Series)):
            self.subpop.drop(drop_rows, axis=0, inplace=True)

    def _datacheck(self):
        """Run InterVA5 data checks on input and store log in _warning. """
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
        pb = self._probbase.copy()

        for i in range(checked_input.shape[0]):
            checked_results = datacheck5(va_input=checked_input.iloc[i],
                                         va_id=checked_input.at[i, "ID"],
                                         probbase=pb,
                                         insilico_check=True)
            self._warning["first_pass"].extend(
                checked_results["first_pass"])
            self._warning["second_pass"].extend(
                checked_results["second_pass"])
            checked_list.append(checked_results["output"])
            if self.gui_ctrl["break"]:
                raise HaltGUIException
        checked = pd.DataFrame(checked_list)
        if sum(self._negate) > 0:
            shifted_index = np.where(self._negate)[0] + 1
            checked.iloc[:, shifted_index] = 1 - checked.iloc[:, shifted_index]

        checked.fillna(-9, inplace=True)

        for i in range(1, checked.shape[1]):
            self.data.iloc[checked.iloc[:, i] == 1, i] = "Y"
            self.data.iloc[checked.iloc[:, i] == 0, i] = ""
            self.data.iloc[checked.iloc[:, i] == -1, i] = "."
            self.data.iloc[checked.iloc[:, i] == -9, i] = "."
            self._data_checked = self.data.copy()
        if self.data_type == "WHO2016":
            self._data_checked.replace({"Y": "y", "": "n", ".": "-"},
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
        csmf_orig = self._probbase[0, 20:]
        n_all = self.data.shape[0]
        n_ext_causes = len(self._external_causes)
        ext_data = self.data.iloc[:, self._external_symps + 1].copy()
        if self.is_numeric:
            pos = 1
        else:
            pos = "Y"
        ext_where = (ext_data == pos).any(axis=1)
        ext_data = ext_data.loc[ext_where, :]
        self._ext_id = self.data.loc[ext_where, "ID"].copy()
        if self.subpop is not None:
            n_subpops = len(self._subpop_order_list)
            self._ext_sub = self.subpop.loc[ext_where].copy()
        prob_sub = self._change_inter(
            self._prob_orig[np.ix_(self._external_symps,
                                   self._external_causes)])
        csmf_sub = self._change_inter(
            csmf_orig[self._external_causes])
        self._prob_orig = np.delete(self._prob_orig,
                                    self._external_symps,
                                    axis=0)
        self._prob_orig = np.delete(self._prob_orig,
                                    self._external_causes,
                                    axis=1)
        self._prob_orig_colnames = np.delete(self._prob_orig_colnames,
                                             self._external_causes,
                                             axis=0)
        self._negate = np.delete(self._negate, self._external_symps)
        if self._external_symps.shape[0] > 0:
            self.data.drop(
                self.data.columns[self._external_symps + 1],
                axis=1, inplace=True)
        if ext_where.sum() > 0:
            n_row = ext_data.shape[0]
            n_col = self._external_causes.shape[0]
            probs = np.ones((n_row, n_col))
            for i in range(n_row):
                for j in range(n_col):
                    symp_pos = ext_data.iloc[i] == pos
                    symp_pos = symp_pos.to_numpy(dtype=bool)
                    probs[i, j] = csmf_sub[j] * (prob_sub[symp_pos, j].prod())
                probs[i, :] = probs[i, :] / sum(probs[i, :])
            self._ext_prob = probs
            self.data = self.data[~ext_where]
            if self.subpop is not None:
                self._ext_csmf = []
                for i in range(n_subpops):
                    self._ext_csmf.append(np.zeros(n_ext_causes))
                    # TODO: need a test where a subpop has 0 external symptoms
                    row_subpop_i = self._ext_sub == self._subpop_order_list[i]
                    row_subpop_i = row_subpop_i.to_numpy(dtype=bool)
                    ext_prob_temp = self._ext_prob[row_subpop_i, :]
                    if ext_prob_temp.shape[0] > 0:
                        self._ext_csmf[i] = (ext_prob_temp.mean(axis=0) *
                                             ext_prob_temp.shape[0]) / sum(
                            self.subpop == self._subpop_order_list[i])
            else:
                self._ext_csmf = self._ext_prob.mean(axis=0) * len(
                    self._ext_id) / n_all
            if ext_where.sum() > 0 and self.subpop is not None:
                self.subpop = self.subpop[~ext_where]
        else:
            if self.subpop is not None:
                self._ext_csmf = []
                for i in range(n_subpops):
                    self._ext_csmf.append(np.zeros(n_ext_causes))
            else:
                self._ext_csmf = np.zeros(n_ext_causes)
            self._ext_prob = np.zeros(
                (ext_data.shape[0], n_ext_causes))
        self._ext_cod = None

    def _remove_external_causes(self):
        if self.data_type == "WHO2012":
            self._remove_ext()
        else:
            self._remove_ext_v5()
        if self.data.shape[0] == 0:
            warnings.warn("All deaths are assigned to external causes.  "
                          "A DataFrame of external causes is returned instead "
                          "of an InSilicoVA object.\n",
                          UserWarning)
            if self.data_type == "WHO2012":
                pass
            else:
                causes = pd.DataFrame({"ID": self._ext_id})
                names_external_causes = self._va_causes[self._external_causes]
                for i, val in enumerate(names_external_causes):
                    causes[val] = self._ext_prob[:, i]
                self.results = InSilicoAllExt(id=self._ext_id,
                                              causes=causes)

    def _check_missing_all(self):
        """
        Remove missing columns from the data (i.e., all values in column are
        missing) and _probbase.
        """
        self._missing_all = []
        n_iterations = self.data.shape[1] - 1  # ignore ID column
        for i in range(n_iterations):
            # increment index by 1 for data (but not for _probbase!)
            if all(self.data.iloc[:, (i + 1)] == "."):
                self._missing_all.append(i)
        if len(self._missing_all) > 0:
            missing_all_plus1 = np.array(self._missing_all) + 1
            missing_all_names = ", ".join(
                self._probbase[missing_all_plus1,
                               int(self.data_type == "WHO2012")])
            warnings.warn(f"{len(self._missing_all)} symptoms missing "
                          "completely and added to missing list. \n List of "
                          f"missing symptoms: {missing_all_names}\n",
                          UserWarning)
            drop_col_names = list(self.data.iloc[:, missing_all_plus1])
            self.data.drop(columns=drop_col_names, inplace=True)
            self._prob_orig = np.delete(self._prob_orig,
                                        self._missing_all,
                                        axis=0)
            if self._negate is not None:
                self._negate = np.delete(self._negate, self._missing_all,
                                         axis=0)

    def _initialize_numerical_matrix(self):
        if not self._customization_dev:
            if self.cond_prob_num is None:
                self._prob_order = self._change_inter(self._prob_orig,
                                                      order=True,
                                                      standard=True)
                self._cond_prob_true = self._change_inter(self._prob_orig,
                                                          order=False,
                                                          standard=True)
            else:
                self._cond_prob_true = self._prob_orig.copy()
                self._prob_order = np.ones(self._prob_orig.shape)
        else:
            if self.update_cond_prob:
                self._prob_order = self._change_inter(
                    self._prob_orig,
                    order=True,
                    standard=False,
                    table_dev=self._table_dev,
                    table_num_dev=self._table_num_dev)
                self._cond_prob_true = self._change_inter(
                    self._prob_orig,
                    order=False,
                    standard=False,
                    table_dev=self._table_dev,
                    table_num_dev=self._table_num_dev)
            else:
                self._cond_prob_true = self._prob_orig.copy()
                self._prob_order = np.ones(self._prob_orig.shape)

    def _check_data_dimensions(self):
        if self.data.shape[0] <= 1:
            raise DataException(
                "Not enough observations in the sample to run InSilicoVA.\n"
            )
        n_symp_ok = (self.data.shape[1] - 1) == self._cond_prob_true.shape[0]
        if n_symp_ok is False:
            raise DataException("Number of symptoms is not right. \n")
        if self.external_sep:
            # self._va_causes_current = self._va_causes[
            #     self._external_causes].copy()
            self._va_causes_current = self._va_causes.drop(
                self._external_causes)
        else:
            self._va_causes_current = self._va_causes.copy()

    def _check_impossible_pairs(self):
        """Check _impossible pairs of symptoms and causes (for the first nine
        demographic symptoms (7 age + 2 gender)).  Also, the value saved is
        the index (starting from 0?)

        format: (ss, cc, 0) if P(cc | ss = y) = 0
                (ss, cc, 1) if P(cc | ss = n) = 0
        """
        impossible = []
        data_colnames = list(self.data)[1:]
        impossible_colnames = ["symptom", "probbase_col", "value"]
        if self.impossible_combination is not None:
            impossible_colnames = list(self.impossible_combination)

        if self.exclude_impossible_causes != "none" and \
                self._customization_dev is False:
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
            n_causes = self._cond_prob_true.shape[1]
            for ss in demog_index:
                for cc in range(n_causes):
                    if self._cond_prob_true[ss, cc] == 0:
                        impossible.append([int(cc), int(ss), 0])
                    if (self._cond_prob_true[ss, cc] == 1 and
                            self.exclude_impossible_causes.lower() != "interva"):
                        impossible.append([int(cc), int(ss), 1])

            if self.exclude_impossible_causes == "subset2":
                # add pregnancy death fix
                add_impossible = pd.DataFrame(
                    {impossible_colnames[0]: ["i310o"] * 9})
                col1_values = ["b_0901", "b_0902", "b_0903", "b_0904",
                               "b_0905", "b_0906", "b_0907", "b_0908",
                               "b_0999"]
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
                        val_only_prem = int(self._negate[ss] is np.True_)
                        val_not_prem = int(self._negate[ss] is np.False_)
                cc_only_prem = self._va_causes_current == "Prematurity"
                if ss is not None and sum(cc_only_prem) == 1:
                    cc_only_index = self._va_causes_current[
                        cc_only_prem].index[0]
                    impossible.append([cc_only_index, ss, val_only_prem])
                cc_not_prem = self._va_causes_current == "Birth asphyxia"
                if ss is not None and sum(cc_not_prem) == 1:
                    cc_not_index = self._va_causes_current[
                        cc_not_prem].index[0]
                    impossible.append([cc_not_index, ss, val_not_prem])

            if self.impossible_combination is not None:
                for ss, cc, vv in self.impossible_combination.itertuples(
                        index=False):
                    if ss in data_colnames and cc in self._prob_orig_colnames:
                        val = 1 - int(vv)
                        ss_index = data_colnames.index(ss)
                        cc_index = np.argwhere(
                            self._prob_orig_colnames == cc)[0][0]
                        impossible.append([cc_index, ss_index, val])

        elif self.impossible_combination is not None:
            for ii in self.impossible_combination.itertuples(index=False):
                if ii[0] in data_colnames and \
                        ii[1] in self._prob_orig_colnames:
                    val = 1 - int(ii[2])
                    ss_index = data_colnames.index(ii[0])
                    cc_index = np.argwhere(
                        self._prob_orig_colnames == ii[1])[0][0]
                    impossible.append([cc_index, ss_index, val])
        else:
            # java checks if _impossible has 3 columns and sets check
            # _impossible cause flag to False if it has 4 columns
            impossible.append([0, 0, 0, 0])

        self._impossible = np.array(impossible, dtype=int)

    def _physician_codes(self):
        if self.phy_code is not None:
            pass
        else:
            # if no physician coding, everything is unknown
            self._assignment = np.zeros((self.data.shape[0], self._c_phy))
            self._assignment[:, 0] = 1
            self._va_causes_broader = range(len(self._va_causes_current))

    def _prior_truncated_beta(self):
        self._prior_b_cond = float(np.floor(1.5 * self.data.shape[0]))
        if self.keep_probbase_level:
            level_count = np.unique(self._cond_prob_true, return_counts=True)[
                1]
            level_count_med = np.median(level_count)
            self._prior_b_cond = float(np.floor(
                self._prior_b_cond * level_count_med * self.levels_strength))
        else:
            self._prior_b_cond = float(np.floor(
                self._prior_b_cond * self.levels_strength))
        if self.levels_prior is None:
            self.levels_prior = self._scale_vec_inter(
                np.array(range(1, self._n_level + 1)),
                scale_max=self._prior_b_cond * 0.99)

    def _get_subpop_info(self):
        if self.subpop.shape[0] != self.data.shape[0]:
            raise ArgumentException("Sub-population size not match data.\n")
        self._subpop_numeric = np.zeros(self.subpop.shape[0], dtype=int)
        self._subpop_order_list = list(self.subpop.unique())
        self._subpop_order_list.sort()
        for ind, val in enumerate(self._subpop_order_list):
            in_subpop_i = np.array(self.subpop == val, dtype=bool)
            self._sublist[val] = in_subpop_i
            self._subpop_numeric[in_subpop_i] = ind

    def _initialize_indicator_matrix(self):

        n, s = self.data.shape
        self._indic = np.zeros((n, s - 1))
        self._id = self.data.iloc[:, 0].copy()
        if self.is_numeric:
            self._indic = self.data.iloc[:, 1].to_numpy(copy=True)
            self._contains_missing = self.data.isin(["-1"]).any()
        else:
            for i in range(s - 1):
                temp_ind = np.array(self.data.iloc[:, i + 1] == "Y",
                                    dtype=bool)
                self._indic[temp_ind, i] = 1
            self._contains_missing = self.data.isin(["."]).any(axis=None)
            if self._contains_missing:
                for i in range(s - 1):
                    temp_ind = np.array(self.data.iloc[:, i + 1] == ".",
                                        dtype=bool)
                    self._indic[temp_ind, i] = -1
        if self.subpop is not None:
            for j in range(len(self._subpop_order_list)):
                for i in range(s - 1):
                    temp = self._indic[self._subpop_numeric == j, i].copy()
                    if np.all(temp == -1):
                        self._indic[self._subpop_numeric == j, i] = -2

    def _initialize_parameters(self):
        c = self._cond_prob_true.shape[1]
        csmf_prior = self._sys_prior / self._sys_prior.sum()
        if self._method == "dirichlet":
            alpha = csmf_prior * c * self._alpha_scale  # not used in R code!
        elif self._method == "normal":
            try:  # I don't see how this could raise an error (TypeError?)
                z = csmf_prior[0] / np.exp(1)
                self._mu = np.log(csmf_prior / z)
            except TypeError:
                self._mu = np.ones(c)
        self._sigma2 = 1.0

        if self.external_sep:
            # csmf_prior = np.delete(csmf_prior, self._external_causes)
            if self._method == "normal" and len(self._mu) != c:
                self._mu = np.delete(self._mu, self._external_causes)
            elif self._method == "dirichlet" and len(alpha):
                alpha = np.delete(alpha, self._external_causes)

        if self.update_cond_prob:
            self._cond_prob = self._cond_initiate(
                probbase_order=self._prob_order,
                exp_ini=True,
                inter_ini=True,
                probbase_min=self.trunc_min,
                probbase_max=self.trunc_max)
        elif self.cond_prob_num is None:
            self._cond_prob = self._change_inter(x=self._prob_orig,
                                                 standard=True)
        else:
            self._cond_prob = self._prob_orig.copy()
        # at this stage, there are few possibilities:
        # _customization_dev & update_cond_prob: need to check existing levels
        # not update_cond_prob: do nothing
        # cond_prob is not None: need to check existing levels
        if (self._customization_dev and self.update_cond_prob) or \
                self.cond_prob is not None:
            # get numerical levels
            if self._customization_dev:
                self._dist = self._interva_table(
                    standard=False,
                    table_num_dev=self._table_num_dev)
            else:
                self._dist = self._interva_table(standard=True,
                                                 min_level=0)
            # check existence of level's index in _prob_order
            self._level_exist = np.isin(
                np.arange(1, len(self._dist) + 1),
                np.unique(self._prob_order))
            # update order matrix
            prob_order_new = self._prob_order.copy()
            for i in range(len(self._dist)):
                if not self._level_exist[i]:
                    tmp_index = self._prob_order > (i + 1)
                    prob_order_new[tmp_index] = prob_order_new[tmp_index] - 1
            self._prob_order = prob_order_new
            # self._prob_order is the same as self._probbase_order
            # update level vector
            self._dist = self._dist[self._level_exist]
            self.levels_prior = self.levels_prior[self._level_exist]
            self._N_level = int(sum(self._level_exist))
        else:
            self._level_exist = np.repeat(True, self._n_level)
            # self._prob_order is the same as self._probbase_order
            self._N_level = self._n_level
            self._dist = self._interva_table(standard=True, min_level=0)
        self._pool = int(not self.keep_probbase_level) + int(
            self._probbase_by_symp_dev)
        if self.subpop is None:
            self._n_sub = 1
            self._subpop_numeric = np.zeros(self.data.shape[0], dtype=int)
        else:
            self._n_sub = len(np.unique(self._subpop_numeric))
        self._mu_last = np.zeros(
            shape=(self._n_sub, self._cond_prob_true.shape[1]))
        self._sigma2_last = np.zeros(self._n_sub)
        self._theta_last = np.zeros(
            shape=(self._n_sub, self._cond_prob_true.shape[1]))

    # function to summarize the current levels of probbase
    def _levelize(self, pool: int) -> dict:
        S = self.data.shape[1] - 1
        C = self._cond_prob_true.shape[1]
        probbase_order = self._prob_order.copy()
        probbase_level = {}
        # if pool = 0, doesn't matter which
        # if pool = 1, count level by cause
        if pool == 0 or pool == 1:
            # loop over all s and c combinations
            for s in range(S):
                for c in range(C):
                    # get level of this s-c combination
                    level = int(probbase_order[s, c])
                    # initialize if this cause has not been intialized in probbase_level yet
                    if c not in probbase_level:
                        # self.probbase_level[c] = dict of K-int and V-list of int
                        probbase_level[c] = {}
                        # initialize if this level under this cause has not been initialized
                    if level not in probbase_level[c]:
                        # self.probbase_level[c][level] = list of int
                        probbase_level[c][level] = []
                        # save the cause-level-symptom combination
                    probbase_level[c][level].append(s)
        # if pool = 2, count level by symptom
        elif pool == 2:
            # loop over all s and c combinations
            for s in range(S):
                for c in range(C):
                    # get level of this s-c combination
                    level = int(probbase_order[s, c])
                    # initialize if this cause has not been initialized in probbase_level yet
                    if s not in probbase_level:
                        # self.probbase_level[s] = dict with K-int and V-list of int
                        probbase_level[s] = {}
                    # initialize if this level under this cause has not been initialized
                    if level not in probbase_level[s]:
                        # self.probbase_level[s][level] = list of int
                        probbase_level[s][level] = []
                    # save the cause-level-symptom combination
                    probbase_level[s][level].append(c)
        return probbase_level

    def _sample_posterior(self):
        N = self.data.shape[0]
        S = self.data.shape[1] - 1
        C = self._cond_prob_true.shape[1]
        # N_sub = len(self._sublist)
        N_sub = self._n_sub
        N_level = sum(self._level_exist)
        subpop = self._subpop_numeric.copy()
        subpop = subpop.astype(np.int32)
        probbase = self._cond_prob.copy()
        probbase_order = self._prob_order.copy()
        probbase_order = probbase_order.astype(np.int32)
        level_values = self._dist.copy()
        pool = int(not self.keep_probbase_level + self._probbase_by_symp_dev)
        dim_args = [N, S, C, N_sub, N_level, pool, self.seed]
        count_m = np.zeros((S, C), dtype=np.int32)
        count_m_all = np.zeros((S, C), dtype=np.int32)
        count_c = np.zeros(C, dtype=np.int32)
        probbase_level = self._levelize(pool)
        sampler = Sampler(dim_args,
                          subpop,
                          probbase,
                          probbase_order,
                          probbase_level,
                          level_values,
                          count_m,
                          count_m_all,
                          count_c)
        dimensions = [N, S, C, N_sub, N_level]
        prior_a = self.levels_prior.copy()
        prior_b = self._prior_b_cond
        jumprange = self.jump_scale
        trunc_min = self.trunc_min
        trunc_max = self.trunc_max
        indic = self._indic.copy()
        contains_missing = bool(self._contains_missing)
        N_gibbs = self._n_gibbs
        burn = self.burnin
        thin = self.thin
        mu = self._mu
        sigma2 = self._sigma2
        # this_is_Unix = not system() == "Windows"
        use_probbase = self._keep_prob
        is_added = False
        mu_continue = self._mu_last
        sigma2_continue = self._sigma2_last
        theta_continue = self._theta_last
        C_phy = self._c_phy
        broader = self._va_causes_broader
        assignment = self._assignment
        impossible = self._impossible
        N_thin = int((N_gibbs - burn) / (thin))
        probbase_gibbs = np.zeros((N_thin, S, C))
        levels_gibbs = np.zeros((N_thin, N_level))
        pnb_mean = np.zeros((N, C))
        p_gibbs = np.zeros((N_thin, N_sub, C))
        pnb_mean = np.zeros((N, C))
        naccept = [0] * N_sub
        mu_now = np.zeros((N_sub, C))
        sigma2_now = [None] * N_sub
        theta_now = np.zeros((N_sub, C))
        p_now = np.zeros((N_sub, C))  # csmf_sub
        zero_matrix = np.zeros((N, C), dtype=np.int32)
        zero_group_matrix = np.zeros((N_sub, C), dtype=np.int32)
        remove_causes = np.zeros(N_sub)
        pnb = np.empty((N, C))
        y_new = np.empty(N)
        y = np.zeros((N_sub, C))
        N_out = 1 + N_sub * C * N_thin + N * C + N_sub * (C * 2 + 1)
        if pool == 0:
            N_out += N_level * N_thin
        else:
            N_out += S * C * N_thin
        parameters = np.empty(N_out)
        is_openva_app = self.openva_app is not None
        sampler.fit(prior_a, prior_b, jumprange, trunc_min, trunc_max,
                    indic, contains_missing,
                    N_gibbs, burn, thin, mu, sigma2, use_probbase, is_added,
                    mu_continue, sigma2_continue, theta_continue, impossible,
                    probbase_gibbs,
                    levels_gibbs,
                    p_gibbs,
                    pnb_mean,
                    naccept,
                    mu_now,
                    sigma2_now,
                    theta_now,
                    p_now,
                    zero_matrix,
                    zero_group_matrix,
                    remove_causes,
                    pnb,
                    y_new,
                    y,
                    parameters,
                    self.gui_ctrl,
                    is_openva_app,
                    self.openva_app)
        if self.gui_ctrl["break"]:
            raise HaltGUIException
        fit_results = {"N_sub": N_sub, "C": C, "S": S, "N_level": N_level,
                       "pool": pool,
                       "fit": parameters}
        results = self._parse_result(fit_results)
        # check convergence
        if results["csmf_sub"] is not None:
            conv = csmf_diag(results["csmf_sub"], self.conv_csmf,
                             test="heidel", verbose=False)
        else:
            conv = csmf_diag(results["p_hat"], self.conv_csmf,
                             test="heidel", verbose=False)
        if self.auto_length:
            add = 1
            while not conv and add < 3:
                mu_continue = results["mu_last"]
                sigma2_continue = results["sigma2_last"]
                theta_continue = results["theta_last"]
                # same length as previous chain if added the first time
                # double the length if the second time
                # self.n_sim = self.n_sim * 2
                # self.burnin = self.n_sim / 2
                N_gibbs = int(np.trunc(N_gibbs * (2 ** (add - 1))))
                burn = 0
                N_thin = int((N_gibbs - burn) / (thin))
                probbase_gibbs = np.zeros((N_thin, S, C))
                levels_gibbs = np.zeros((N_thin, N_level))
                p_gibbs = np.zeros((N_thin, N_sub, C))
                warnings.warn(
                    f"Not all causes with CSMF > {self.conv_csmf} are "
                    f"convergent.\n Increase chain length with another "
                    f"{N_gibbs} iterations.\n")
                N_out = (1 + N_sub * C * N_thin +
                         N * C + N_sub * (C * 2 + 1))
                if pool == 0:
                    N_out += N_level * N_thin
                else:
                    N_out += S * C * N_thin
                parameters = np.empty(N_out)
                sampler.fit(prior_a,
                            prior_b,
                            jumprange,
                            trunc_min,
                            trunc_max,
                            indic,
                            contains_missing,
                            N_gibbs,
                            burn,
                            thin,
                            mu,
                            sigma2,
                            use_probbase,
                            True,
                            mu_continue,
                            sigma2_continue,
                            theta_continue,
                            impossible,
                            probbase_gibbs,
                            levels_gibbs,
                            p_gibbs,
                            pnb_mean,
                            naccept,
                            mu_now,
                            sigma2_now,
                            theta_now,
                            p_now,
                            zero_matrix,
                            zero_group_matrix,
                            remove_causes,
                            pnb,
                            y_new,
                            y,
                            parameters,
                            self.gui_ctrl,
                            is_openva_app,
                            self.openva_app)
                if self.gui_ctrl["break"]:
                    raise HaltGUIException
                fit_results = {"N_sub": N_sub, "C": C, "S": S,
                               "N_level": N_level,
                               "pool": pool, "fit": parameters}
                results = self._parse_result(fit_results)
                add += 1
                # check convergence
                if results["csmf_sub"] is not None:
                    conv = csmf_diag(results["csmf_sub"], self.conv_csmf,
                                     test="heidel", verbose=False)
                else:
                    conv = csmf_diag(results["p_hat"], self.conv_csmf,
                                     test="heidel", verbose=False)
        if not conv:
            warnings.warn(
                f"Not all causes with CSMF > {self.conv_csmf} are "
                "convergent.\nPlease check using csmf_diag() for more "
                "information.\n")
        self._posterior_results = results

    def _parse_result(self, fit_results: Dict) -> Dict:
        n = self.data.shape[0]
        n_sub = int(fit_results["N_sub"])
        c = int(fit_results["C"])
        s = int(fit_results["S"])
        n_level = int(fit_results["N_level"])
        pool = int(fit_results["pool"])
        fit = fit_results["fit"]
        counter = 0
        csmf_sub = None
        p_hat = None
        probbase_gibbs = None
        level_gibbs = None

        # extract N_thin
        n_thin = int(fit[0])
        counter += 1
        # extract CSMF
        if n_sub > 1:
            csmf_sub = []
            for sub in range(n_sub):
                csmf_list = fit[counter:(counter + c * n_thin)]
                csmf_array = np.array(csmf_list).reshape((n_thin, c))
                csmf_sub.append(csmf_array)
                counter = counter + (c * n_thin)
        else:
            p_hat = fit[counter:(counter + c * n_thin)]
            p_hat = np.array(p_hat).reshape((n_thin, c))
            counter = counter + (c * n_thin)
        # extract individual probabilities
        p_indiv = fit[counter:(counter + n * c)]
        p_indiv = np.array(p_indiv).reshape((n, c))
        counter = counter + (n * c)
        # extract probbase
        if pool != 0:
            probbase_gibbs = fit[counter:(counter + s * c * n_thin)]
            # array(..., dim = c(a,b,c))
            # what it does is for each c, fill column
            # i.e., c(x[1,1,1], x[2,1,1], x[1,2,1], ...)
            # i.e. in Java, loop in the  order of C.j -> S.j -> N_thin
            probbase_gibbs = np.array(probbase_gibbs).reshape((n_thin, s, c),
                                                              order="F")
            counter = counter + (s * n * n_thin)
        else:
            level_gibbs = fit[counter:(counter + n_level * n_thin)]
            level_gibbs = np.array(level_gibbs).reshape((n_thin, n_level))
            counter = counter + (n_level * n_thin)
        # find last time configurations
        mu_last = fit[counter:(counter + n_sub * c)]
        mu_last = np.array(mu_last).reshape((n_sub, c))
        counter = counter + (n_sub * c)
        sigma2_last = fit[counter:(counter + n_sub)]
        counter = counter + n_sub
        theta_last = fit[counter:(counter + n_sub * c)]
        theta_last = np.array(theta_last).reshape((n_sub, c))
        counter = counter + (n_sub * c)

        return {"csmf_sub": csmf_sub, "p_hat": p_hat, "p_indiv": p_indiv,
                "probbase_gibbs": probbase_gibbs,
                "levels_gibbs": level_gibbs,
                "mu_last": mu_last, "sigma2_last": sigma2_last,
                "theta_last": theta_last,
                "pool": pool}

    def _prepare_results(self):
        csmf_sub = self._posterior_results["csmf_sub"]
        p_hat = self._posterior_results["p_hat"]
        p_indiv = self._posterior_results["p_indiv"]
        probbase_gibbs = self._posterior_results["probbase_gibbs"]
        levels_gibbs = pd.DataFrame(self._posterior_results["levels_gibbs"])
        pool = self._posterior_results["pool"]
        N = self.data.shape[0]
        C = self._cond_prob_true.shape[1]
        n_ext_id = len(self._ext_id)
        n_ext_causes = len(self._external_causes)
        # To make output consistent for further analysis,
        # add back external results
        # if separated external causes
        if self.external_sep:
            # get starting and ending index
            ext1 = self._external_causes[0]
            # ext2 = self._external_causes[1]
            # if with subgroup
            if self.subpop is not None:
                p_hat = None
                csmf_sub_all = []
                # iterate over all subpopulations
                for j, val in enumerate(self._sublist):
                    dims = (csmf_sub[j].shape[0],
                            C + n_ext_causes)
                    # csmf_sub_all.append(np.zeros(dims))
                    # rescale the non-external CSMF once the external
                    # cause are added
                    rescale = (
                        sum(self._sublist[val]) /
                        (sum(self._sublist[val]) +
                         sum(self._ext_sub == val))
                    )
                    temp = csmf_sub[j] * rescale
                    # combine the rescaled non-external CSMF with the
                    # external CSMF
                    n_ext = len(self._ext_csmf[j]) * temp.shape[0]
                    ext_temp = np.resize(self._ext_csmf[j], n_ext)
                    ext_temp.resize((temp.shape[0], len(self._ext_csmf[j])))
                    # TODO: is np.insert a better tool for this?
                    csmf_temp = np.concatenate(
                        (temp[:, 0:ext1], ext_temp, temp[:, ext1:]),
                        axis=1)
                    csmf_temp = np.nan_to_num(csmf_temp)
                    csmf_sub_all.append(csmf_temp)
                csmf_sub = csmf_sub_all
            # if no subgroup
            else:
                p_hat = p_hat * N / (N + n_ext_id)
                temp = p_hat[:, ext1:(C + 1)].copy()
                n_ext = p_hat.shape[0] * n_ext_causes
                extra = np.resize(self._ext_csmf, n_ext)
                extra.resize((p_hat.shape[0], n_ext_causes))
                p_hat = np.concatenate(
                    (p_hat[:, 0:ext1], extra, temp),
                    axis=1)

            p_indiv = np.concatenate(
                (p_indiv[:, 0:ext1],
                 np.zeros((p_indiv.shape[0], n_ext_causes)),
                 p_indiv[:, ext1:]),
                axis=1)
            p_indiv_ext = np.zeros((n_ext_id,
                                    (C + n_ext_causes)))
            if n_ext_id > 0:
                # WHO 2012
                if self._ext_prob is None:
                    for i in range(n_ext_id):
                        p_indiv_ext[i, self._ext_cod[i]] = 1
                # WHO 2016
                else:
                    p_indiv_ext[:,
                    self._external_causes] = self._ext_prob.copy()
            p_indiv = np.concatenate((p_indiv, p_indiv_ext), axis=0)
            np.nan_to_num(p_indiv)
            self._id = pd.concat((self._id, self._ext_id))
            if self.subpop is not None:
                self.subpop = pd.concat((self.subpop, self._ext_sub))
        # add column names to outcomes
        if pool != 0:
            if self.external_sep:
                # remove column "ID"
                self._va_labels.pop(0)
                # remove external
                self._va_labels = [i for j, i in enumerate(self._va_labels) if
                                   j not in self._external_symps]
                va_causes_ext = self._va_causes.drop(
                    index=self._external_causes)
                # remove missing, sequential deleting since that's how we
                # obtained indices before
                self._va_labels = [i for j, i in enumerate(self._va_labels) if
                                   j not in self._missing_all]
                list_pb_gibbs = []
                for i in range(probbase_gibbs.shape[0]):
                    tmp_df = pd.DataFrame(probbase_gibbs[i])
                    tmp_df = tmp_df.set_axis(self._va_labels, axis=0)
                    tmp_df = tmp_df.set_axis(va_causes_ext.to_list(),
                                             axis=1)
                    list_pb_gibbs.append(tmp_df.copy())
            else:
                list_pb_gibbs = []
                for i in range(probbase_gibbs.shape[0]):
                    tmp_df = pd.DataFrame(probbase_gibbs[i])
                    tmp_df = tmp_df.set_axis(self._va_labels, axis=0)
                    tmp_df = tmp_df.set_axis(self._va_causes.to_list(),
                                             axis=1)
                    list_pb_gibbs.append(tmp_df.copy())
        else:
            if self._customization_dev:
                levels_gibbs = levels_gibbs.set_axis(self._table_dev.tolist(),
                                                     axis=1)
            else:
                col_names = ["I", "A+", "A", "A-", "B+", "B", "B-", "C+", "C",
                             "C-", "D+", "D", "D-", "E", "N"]
                levels_gibbs = levels_gibbs.set_axis(col_names, axis=1)
            probbase_gibbs = levels_gibbs
        if self.subpop is not None:
            new_csmf_sub = []
            for i in range(len(csmf_sub)):
                tmp_df = pd.DataFrame(csmf_sub[i])
                tmp_df = tmp_df.set_axis(self._va_causes.to_list(), axis=1)
                new_csmf_sub.append(tmp_df.copy())
            p_hat = new_csmf_sub
        else:
            p_hat = pd.DataFrame(p_hat)
            p_hat = p_hat.set_axis(self._va_causes.to_list(), axis=1)
        p_indiv = pd.DataFrame(p_indiv)
        p_indiv = p_indiv.set_axis(self._va_causes.to_list(), axis=1)
        p_indiv = p_indiv.set_axis(self._id.to_list(), axis=0)
        if not self.update_cond_prob:
            probbase_gibbs = None
        clean_data = pd.DataFrame(self._indic)
        # because the external deaths might be appended to the end
        n_clean_data = clean_data.shape[0]
        clean_data = clean_data.set_axis(
            self._id.iloc[0:n_clean_data].to_list(), axis=0)
        self.results = InSilico(id=self._id,
                                data_final=clean_data,
                                data_checked=self._data_checked,
                                indiv_prob=p_indiv,
                                csmf=p_hat,
                                conditional_probs=probbase_gibbs,
                                probbase=self._prob_orig,
                                probbase_colnames=self._prob_orig_colnames,
                                missing_symptoms=self._missing_all,
                                external=self.external_sep,
                                external_causes=self._external_causes,
                                impossible_causes=self._impossible,
                                update_cond_prob=self.update_cond_prob,
                                keep_probbase_level=self.keep_probbase_level,
                                datacheck=self.datacheck,
                                n_sim=self.n_sim,
                                thin=self.thin,
                                burnin=self.burnin,
                                jump_scale=self.jump_scale,
                                levels_prior=self.levels_prior,
                                levels_strength=self.levels_strength,
                                trunc_min=self.trunc_min,
                                trunc_max=self.trunc_max,
                                subpop=self.subpop,
                                indiv_ci=self.indiv_ci,
                                is_customized=self._customization_dev,
                                errors=self._error_log,
                                warnings=self._warning,
                                data_type=self.data_type)

    def get_results(self) -> InSilico:
        if hasattr(self, "results"):
            return self.results
        else:
            raise InSilicoVAException("No results available.")

    def get_errors(self, print_errors: bool = True) -> Union[None, list]:
        error_list = [str(k) + " - " + i for k, v in
                      self._error_log.items() for i in v]
        if print_errors:
            print(error_list)
        else:
            return error_list
