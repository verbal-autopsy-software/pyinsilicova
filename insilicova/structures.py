# -*- coding: utf-8 -*-

"""
insilicova.structures
-------------------

This module contains data structures for results.
"""

from dataclasses import dataclass
from typing import Dict, Union
import pandas as pd
import numpy as np


@dataclass
class InSilico:
    """Class for holding results from InSilicoVA._run()."""
    id: pd.Series
    data_final: pd.DataFrame
    data_checked: pd.DataFrame
    indiv_prob: np.ndarray
    csmf: Dict
    conditional_probs: np.ndarray
    probbase: np.ndarray
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
    indiv_prob_median: Union[np.ndarray, None] = None
    indiv_prob_upper: Union[np.ndarray, None] = None
    indiv_prob_lower: Union[np.ndarray, None] = None

    # TODO: not sure how to make this useful
    def __repr__(self):
        cls = self.__class__.__name__
        msg = f"""
        <{cls}(id=self._id,
               data_final=clean_data,
               data_checked=self._data_checked,
               indiv_prob=p_indiv,
               csmf=p_hat,
               conditional_probs=probbase_gibbs,
               probbase=self._prob_orig,
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

    def __str__(self):
        n_iter = (self.n_sim - self.burnin)/self.thin
        msg = f"""
        InSilicoVA fitted object:
        {self.data_final.shape[0]} deaths processed
        {self.n_sim} iterations performed, with the first {self.burnin} iterations discarded
        {int(n_iter)} iterations saved after thinning
        """
        return msg

    def summary(self,
                ci_csmf: float = 0.95,
                ci_cond: float = 0.95,
                file_name: Union[None, str] = None,
                top: int = 10,
                va_id: Union[None, str] = None):
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
            csmf_out = []
            csmf_out_ordered = []
            for i in range(len(self.csmf)):
                mean = self.csmf[i].mean(axis=0)
                median = self.csmf[i].median(axis=0)
                sd = self.csmf[i].std(axis=0)
                low = self.csmf[i].quantile(ci_low, axis=0)
                up = self.csmf[i].quantile(ci_up, axis=0)
                # HERE should this be a pd.DataFrame?
                csmf_out.append([mean, sd, low, median, up])
                # names are "Mean", "Std.Error", "Lower", "Median", "Upper"
                csmf_ordered = mean.sort_values(ascending=False)
                csmf_out_ordered.append(csmf_ordered)
                # here you need the names of the subpopulations
        else:
            mean = self.csmf.mean(axis=0)
            median = self.csmf.median(axis=0)
            sd = self.csmf.std(axis=0)
            low = self.csmf.quantile(ci_low, axis=0)
            up = self.csmf.quantile(ci_up, axis=0)
            # HERE should this be a pd.DataFrame?
            csmf_out = [mean, sd, low, median, up]
            # names are "Mean", "Std.Error", "Lower", "Median", "Upper"
            csmf_out_ordered = mean.sort_values(ascending=False)

        # organize conditional probabilities matrix object
        ci_low = (1 - ci_cond) / 2
        ci_up = 1 - ci_low
        if cond_prob:

