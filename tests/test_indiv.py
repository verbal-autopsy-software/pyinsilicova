# -*- coding: utf-8 -*-

import pandas as pd
import pytest
from insilicova.api import InSilicoVA
from insilicova.indiv import get_indiv, update_indiv
from insilicova.utils import get_vadata

va_data = get_vadata("randomva5", verbose=False)

with pytest.warns(UserWarning):
    default = InSilicoVA(va_data, subpop=["i019a"], n_sim=1000, burnin=100, thin=10,
                         auto_length=False)
insilico_obj = default.get_results()


def test_get_indiv_defaults():
    out = get_indiv(insilico_obj=insilico_obj,
                    data=va_data,
                    by=["i019a"])
    assert isinstance(out, pd.DataFrame)


def test_get_indiv_is_sample():
    out = get_indiv(insilico_obj=insilico_obj,
                    data=va_data,
                    ci=0.95,
                    is_aggregate=True,
                    by=["i019a"],
                    is_sample=True)
    assert isinstance(out, pd.DataFrame)


def test_get_indiv_is_aggregate():
    out = get_indiv(insilico_obj=insilico_obj,
                    data=va_data,
                    ci=0.95,
                    is_aggregate=True,
                    by=["i019a"],
                    is_sample=False)
    assert isinstance(out, pd.DataFrame)


def test_get_indiv_is_aggregate_no_by():
    with pytest.warns(UserWarning):
        out = get_indiv(insilico_obj=insilico_obj,
                        data=va_data,
                        ci=0.95,
                        is_aggregate=True,
                        by=None,
                        is_sample=False)
    assert isinstance(out, pd.DataFrame)


def test_update_indiv():
    assert insilico_obj.indiv_prob_median is None
    assert insilico_obj.indiv_prob_lower is None
    assert insilico_obj.indiv_prob_upper is None
    assert insilico_obj.indiv_ci is None
    update_indiv(
        insilico_obj=insilico_obj,
        data=va_data,
        ci=0.95,
        is_aggregate=False,
        by=None,
        is_sample=False)
    assert isinstance(insilico_obj.indiv_prob_median, pd.DataFrame)
    assert isinstance(insilico_obj.indiv_prob_lower, pd.DataFrame)
    assert isinstance(insilico_obj.indiv_prob_upper, pd.DataFrame)
    assert isinstance(insilico_obj.indiv_ci, float)
