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
