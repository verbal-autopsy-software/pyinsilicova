# -*- coding: utf-8 -*-

"""
insilicova.indiv
-------------------

This module contains functions for getting and updating individual COD
probabilities from InSilicoVA model fits.
"""

from insilicova.structures import InSilico
from pandas import DataFrame
from numpy import array, ndarray
from typing import Union, Dict

def get_indiv(insilico_obj: InSilico,
              data: DataFrame,
              ci: float = 0.95,
              is_aggregate: bool = False,
              by: Union[None, list] = None) -> Dict:
    """
    This function calculates individual probabilities for each death and
    provide posterior credible intervals for each estimate. The default set up
    is to calculate the 95% C.I. when running the inSilicoVA model. If a
    different C.I. is desired after running the model, this function provides
    faster re-calculation without refitting the model.

    :param insilico_obj: Fitted InSilicoVA object (InSilicoVA.get_results())
    :type insilico_obj: InSilico
    :param data: Data for the fitted InSilico object. The first column of the
    data should be the ID that matches the InSilico fitted model.
    :type data: DataFrame
    :param ci:  Width of credible interval for posterior estimates.
    :type ci: float
    :param is_aggregate: Indicator for constructing aggregated distribution
    rather than individual distributions.
    :type is_aggregate: bool
    :param by: List of column names to group by.
    :type by: list (or None)
    :returns: Individual mean COD distribution array (mean); individual median
    COD distribution array (median); individual lower & upper bounds for each
    COD probability (lower & upper).
    :rtype: dictionary of numpy arrays
    """
    pass


def update_indiv(insilico_obj: InSilico,
                 ci: float = 0.95):
    """
    Update individual COD probabilities from InSilicoVA Model Fits

    This function updates individual probabilities for each death and provide
    posterior credible intervals for each estimate.

    :param insilico_obj: Fitted InSilicoVA object (InSilicoVA.get_results())
    :type insilico_obj: InSilico
    :param ci:  Width of credible interval for posterior estimates.
    :type ci: float
    :returns: Updated InSilico object
    :rtype: InSilico
    """
    pass
