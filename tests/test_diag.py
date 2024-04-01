# -*- coding: utf-8 -*-

import pandas as pd
import pytest
import numpy as np
from insilicova.api import InSilicoVA
from insilicova.utils import get_vadata
from insilicova.diag import csmf_diag

va_data = get_vadata("randomva5", verbose=False)

with pytest.warns(UserWarning):
    out1 = InSilicoVA(va_data, n_sim=1000, burnin=500,
                      thin=10, auto_length=False, seed=33)
    out2 = InSilicoVA(va_data, n_sim=1000, burnin=500,
                      thin=10, auto_length=False, seed=243)
    out3 = InSilicoVA(va_data, n_sim=1000, burnin=500,
                      thin=10, auto_length=False, seed=567)

with pytest.warns(UserWarning):
    out4 = InSilicoVA(va_data, subpop=["i019a"], n_sim=1000, burnin=500,
                      thin=10, auto_length=False, seed=645)
    out5 = InSilicoVA(va_data, subpop=["i019a"], n_sim=1000, burnin=500,
                      thin=10, auto_length=False, seed=7934)
    out6 = InSilicoVA(va_data, subpop=["i019a"], n_sim=1000, burnin=500,
                      thin=10, auto_length=False, seed=6)

results1 = out1.get_results()
results2 = out2.get_results()
results3 = out3.get_results()
results4 = out4.get_results()
results5 = out5.get_results()
results6 = out6.get_results()

csmf1 = np.delete(results1.csmf.to_numpy(),
                  results1.external_causes, axis=1)
csmf4_0 = np.delete(results4.csmf[0].to_numpy(),
                    results4.external_causes, axis=1)
csmf4_1 = np.delete(results4.csmf[1].to_numpy(),
                    results4.external_causes, axis=1)


def test_single_csmf_no_subpop():
    out_diag = csmf_diag(csmf1)
    assert isinstance(out_diag, pd.DataFrame)
    assert out_diag.shape[1] == 6


def test_csmf_list():
    csmf = [csmf4_0, csmf4_1]
    out_diag = csmf_diag(csmf)
    assert len(out_diag) == 2
    assert isinstance(out_diag[0], pd.DataFrame)
    assert isinstance(out_diag[1], pd.DataFrame)
    assert out_diag[0].shape[1] == 6
    assert out_diag[1].shape[1] == 6


def test_insilico_no_subpop():
    out_diag = csmf_diag(results1)
    assert isinstance(out_diag, pd.DataFrame)
    assert out_diag.shape[1] == 6


def test_insilico_with_subpop():
    out_diag = csmf_diag(results4, which_sub=0)
    assert len(out_diag) == 2
    assert isinstance(out_diag[0], pd.DataFrame)
    assert isinstance(out_diag[1], pd.DataFrame)
    assert out_diag[0].shape[1] == 6
    assert out_diag[1].shape[1] == 6


def test_insilico_list():
    csmf = [results1, results2, results3]
    out_diag = csmf_diag(csmf)
    assert len(out_diag) == 3
    assert isinstance(out_diag[0], pd.DataFrame)
    assert isinstance(out_diag[1], pd.DataFrame)
    assert isinstance(out_diag[2], pd.DataFrame)
    assert out_diag[0].shape[1] == 6
    assert out_diag[1].shape[1] == 6
    assert out_diag[2].shape[1] == 6


def test_insilico_list_with_subpops():
    csmf = [results4, results5, results6]
    out_diag = csmf_diag(csmf, which_sub=1)
    assert len(out_diag) == 3
    assert isinstance(out_diag[0], pd.DataFrame)
    assert isinstance(out_diag[1], pd.DataFrame)
    assert isinstance(out_diag[2], pd.DataFrame)
    assert out_diag[0].shape[1] == 6
    assert out_diag[1].shape[1] == 6
    assert out_diag[2].shape[1] == 6
