# -*- coding: utf-8 -*-

"""
insilicova.structures
-------------------

This module contains data structures for results.
"""

from .exceptions import ArgumentException
from dataclasses import dataclass
from typing import Dict, Union
import warnings
import pandas as pd
import numpy as np


@dataclass
class InSilicoSummary:
    """Class for holding summary of InSilico."""
    id_all: pd.Series
    indiv: pd.DataFrame
    csmf: pd.DataFrame
    csmf_ordered: Union[Dict, pd.DataFrame]
    condprob: pd.DataFrame
    update_cond_prob: bool
    keep_probbase_level: bool
    datacheck: bool
    n_sim: int
    thin: int
    burnin: int
    # hiv: str
    # malaria: str
    jump_scale: float
    levels_prior: np.ndarray
    levels_strength: np.ndarray
    trunc_min: float
    trunc_max: float
    subpop_counts: pd.Series
    show_top: int
    id: pd.Series
    indiv_prob: pd.DataFrame
    indiv_ci: bool
    data_type: str

    # TODO: not sure how to make this useful
    def __repr__(self):
        cls = self.__class__
        msg = f"""
        <{str(cls)}>
        (called for InSilico.summary())"""
        return msg

    def __str__(self):
        if self.indiv_prob is not None:
            msg = (f"InSilicoVA fitted top {self.show_top} "
                   f"causes for death ID: {self.id}\n"
                   f"Credible intervals shown: {round(self.indiv_ci * 100)}% \n"
                   f"{round(self.indiv_prob, 4)}")
            return msg
        else:
            n_iter = int((self.n_sim - self.burnin)/self.thin)
            msg = (f"InSilico Call:\n"
                   f"{len(self.id_all)} deaths processed.\n"
                   f"{self.n_sim} iterations performed, "
                   f"with first {self.burnin} iterations discarded.\n"
                   f"{n_iter} iterations saved after thinning.")
            msg_p_mat = "Fitted with re-estimating conditional probability matrix."
            if not self.update_cond_prob:
                msg_p_mat = "Fitted with fixed conditional probability matrix"
            elif self.keep_probbase_level:
                msg_p_mat = "Fitted with re-estimated conditional probability level table"
            msg = msg + f"\n{msg_p_mat}"
            if self.datacheck:
                check_alg = "InterVA5" if self.data_type == "WHO2016" else "InterVA4"
                msg = msg + f"\nData consistency check performed as in {check_alg}\n"
            if self.subpop_counts is not None:
                msg = msg + f"\nSub population frequencies:\n{self.subpop_counts}\n"
            if not isinstance(self.csmf_ordered, dict):
                top = min(self.show_top, self.csmf_ordered.shape[0])
                msg = msg + f"\nTop {top} CSMF's\n{self.csmf_ordered[0:top]}"
            else:
                for k, v in self.csmf_ordered.items():
                    # top = min(self.show_top, self.csmf_ordered[key].shape[0])
                    top = min(self.show_top, v.shape[0])
                    msg = msg + (f"\n{k} - Top {top} CSMFs:\n"
                                 # f"{self.csmf_ordered[key][0:top]}")
                                 f"{v[0:top]}\n")

            return msg


@dataclass
class InSilico:
    """Class for holding results from InSilicoVA._run()."""
    id: pd.Series
    data_final: pd.DataFrame
    data_checked: pd.DataFrame
    indiv_prob: np.ndarray
    csmf: pd.DataFrame
    conditional_probs: np.ndarray
    probbase: np.ndarray
    probbase_colnames: np.ndarray
    missing_symptoms: np.ndarray
    external: bool
    external_causes: np.ndarray
    impossible_causes: np.ndarray
    update_cond_prob: bool
    keep_probbase_level: bool
    datacheck: bool
    n_sim: int
    thin: int
    burnin: int
    jump_scale: float
    levels_prior: np.ndarray
    levels_strength: np.ndarray
    trunc_min: float
    trunc_max: float
    subpop: pd.Series
    indiv_ci: bool
    is_customized: bool
    errors: Dict
    warnings: Dict
    data_type: str
    indiv_prob_median: Union[pd.DataFrame, None] = None
    indiv_prob_upper: Union[pd.DataFrame, None] = None
    indiv_prob_lower: Union[pd.DataFrame, None] = None
    summary: Union[pd.DataFrame, None] = None

    # TODO: not sure how to make this useful
    def __repr__(self):
        cls = self.__class__
        msg = f"""
        <{str(cls)}(id=self._id,
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
        data_type=self.data_type)>
        (call for InSilicoVA._prepare_results())"""
        return msg

    def __str__(self):
        n_iter = (self.n_sim - self.burnin)/self.thin
        msg = f"""
        InSilicoVA fitted object:
        {self.data_final.shape[0]} deaths processed
        {self.n_sim} iterations performed, with the first {self.burnin} iterations discarded
        {int(n_iter)} iterations saved after thinning"""
        return msg

    def get_summary(self,
                    ci_csmf: float = 0.95,
                    ci_cond: float = 0.95,
                    file_name: Union[None, str] = None,
                    top: int = 10,
                    va_id: Union[None, str] = None,
                    verbose: bool = True) -> None:
        self.summary = self._create_summary(
            ci_csmf=ci_csmf,
            ci_cond=ci_cond,
            file_name=file_name,
            top=top,
            va_id=va_id)
        if verbose:
            print(self.summary)

    def get_csmf(self,
                 ci_csmf: float = 0.95,
                 file_name: Union[None, str] = None,
                 top: int = 10) -> pd.DataFrame:
        if self.summary is None:
            self.get_summary(ci_csmf=ci_csmf, top=top, verbose=False)
        if isinstance(self.summary.csmf_ordered, dict):
            csmf_out = {}
            for k, v in self.summary.csmf_ordered.items():
                csmf_out[k] = v.iloc[0:top, :].copy()
        else:
            csmf_out = self.summary.csmf_ordered.iloc[0:top, :]
        if isinstance(file_name, str):
            if isinstance(self.summary.csmf_ordered, dict):
                csmf_csv = pd.concat(csmf_out)
                csmf_csv.index.names = ["Sub-population", "Cause"]
                csmf_csv.reset_index(level="Sub-population")
            else:
                csmf_csv = csmf_out.copy()
            try:
                csmf_csv.to_csv(file_name)
            except (OSError, PermissionError) as exc:
                raise ArgumentException(
                    "Unable to write CSV file.") from exc
        return csmf_out

    def _create_summary(self,
                        ci_csmf: float = 0.95,
                        ci_cond: float = 0.95,
                        file_name: Union[None, str] = None,
                        top: int = 10,
                        va_id: Union[None, str] = None) -> InSilicoSummary:
        id_all = self.id
        csmf = self.csmf
        if self.conditional_probs is None:
            cond_prob = False
        else:
            cond_prob = True
            cond = self.conditional_probs
            if self.conditional_probs.ndim == 2:
                is_level = True
            else:
                is_level = False
        indiv = self.indiv_prob
        if file_name is not None:
            self.indiv_prob.to_csv(file_name)
        # organize CSMF
        ci_low = (1 - ci_csmf) / 2
        ci_up = 1 - ci_low
        if isinstance(self.csmf, list):
            csmf_out = {}
            csmf_out_ordered = {}
            csmf_labels = list(self.subpop.unique())
            csmf_labels.sort()
            for i in range(len(self.csmf)):
                mean = self.csmf[i].mean(axis=0)
                median = self.csmf[i].median(axis=0)
                sd = self.csmf[i].std(axis=0)
                low = self.csmf[i].quantile(ci_low, axis=0)
                up = self.csmf[i].quantile(ci_up, axis=0)
                cond_out = pd.DataFrame({"Mean": mean, "Std.Error": sd,
                                         "Lower": low, "Median": median,
                                         "Upper": up})
                csmf_out[csmf_labels[i]] = cond_out
                csmf_ordered = cond_out.sort_values(by="Mean",
                                                    ascending=False).copy()
                csmf_out_ordered[csmf_labels[i]] = csmf_ordered
        else:
            mean = self.csmf.mean(axis=0)
            median = self.csmf.median(axis=0)
            sd = self.csmf.std(axis=0)
            low = self.csmf.quantile(ci_low, axis=0)
            up = self.csmf.quantile(ci_up, axis=0)
            csmf_out = pd.DataFrame({"Mean": mean, "Std.Error": sd,
                                     "Lower": low, "Median": median,
                                     "Upper": up})
            csmf_out_ordered = csmf_out.sort_values(by="Mean",
                                                    ascending=False).copy()
        # organize conditional probabilities matrix object
        ci_low = (1 - ci_cond) / 2
        ci_up = 1 - ci_low
        if cond_prob:
            if is_level:
                mean = cond.mean(axis=0)
                median = cond.median(axis=0)
                sd = cond.std(axis=0)
                low = cond.quantile(ci_low, axis=0)
                up = cond.quantile(ci_up, axis=0)
                cond_out = pd.DataFrame({"Mean": mean, "Std.Error": sd,
                                         "Lower": low, "Median": median,
                                         "Upper": up})
            else:
                mean = cond.mean(axis=0).T
                median = cond.median(axis=0).T
                sd = cond.sd(axis=0).T
                low = cond.quantile(ci_low, axis=0).T
                up = cond.quantile(ci_up, axis=0).T
                cond_out = {"Mean": mean, "Std.Error": sd, "Lower": low,
                            "Median": median, "Upper": up}
        else:
            cond_out = None

        # get subpopulation counts table
        if self.subpop is not None:
            subpop_counts = self.subpop.value_counts()
        else:
            subpop_counts = None

        if va_id is not None:
            if va_id not in indiv.index:
                raise ArgumentException("Invalid va_id (not found in data")
            which_to_print = self.indiv_prob.loc[va_id].sort_values(
                ascending=False)[0:top].index
            if self.indiv_prob_lower is None:
                warnings.warn(
                    "C.I. for individual probabilities have not been "
                    "calculated.  Please use get_indiv() function to update "
                    "C.I. first.\n",
                    UserWarning)
                indiv_prob = pd.DataFrame({
                    "Mean": self.indiv_prob.loc[va_id, which_to_print],
                    "Lower": np.nan,
                    "Median": np.nan,
                    "Upper": np.nan})
            else:
                indiv_prob = pd.DataFrame({
                    "Mean": self.indiv_prob.loc[va_id, which_to_print],
                    "Lower": self.indiv_prob_lower.loc[va_id, which_to_print],
                    "Median": self.indiv_prob_median.loc[va_id, which_to_print],
                    "Upper": self.indiv_prob_upper.loc[va_id, which_to_print]})
        else:
            indiv_prob = None

        return InSilicoSummary(
            id_all=id_all,
            indiv=indiv,
            csmf=csmf_out,
            csmf_ordered=csmf_out_ordered,
            condprob=cond_out,
            update_cond_prob=self.update_cond_prob,
            keep_probbase_level=self.keep_probbase_level,
            datacheck=self.datacheck,
            n_sim=self.n_sim,
            thin=self.thin,
            burnin=self.burnin,
            # hiv=self.hiv
            # malaria=self.malaria,
            jump_scale=self.jump_scale,
            levels_prior=self.levels_prior,
            levels_strength=self.levels_strength,
            trunc_min=self.trunc_min,
            trunc_max=self.trunc_max,
            subpop_counts=subpop_counts,
            show_top=top,
            id=id,
            indiv_prob=indiv_prob,
            indiv_ci=self.indiv_ci,
            data_type=self.data_type)


@dataclass
class InSilicoAllExt:
    """Class for holding results where all causes are external."""
    id: pd.Series
    causes: pd.DataFrame

    def __str__(self):
        msg = ("All deaths are assigned external causes.\n"
               "Only external causes returned and IDs are returned "
               "(not an InSilico object)\n")
        return msg
