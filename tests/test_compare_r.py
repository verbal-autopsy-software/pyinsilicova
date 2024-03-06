# # -*- coding: utf-8 -*-

import pytest
from insilicova.api import InSilicoVA
from insilicova.indiv import get_indiv
from insilicova.utils import get_vadata
from rpy2.robjects.packages import data, importr
import rpy2.robjects as robjects
from rpy2.robjects.conversion import get_conversion, localconverter
from rpy2.robjects import pandas2ri


r_pack = importr("InSilicoVA")
robjects.r('''
  data(RandomVA5)
  r_out <- insilico(RandomVA5, data.type="WHO2016",
                    Nsim=4000, burnin=2000, thin=10)
  r_out_csmf <- as.data.frame(summary(r_out)$csmf.ordered)
  indiv_results <- get.indiv(r_out, RandomVA5)
  r_indiv_mean <- as.data.frame(indiv_results$mean)
  r_indiv_med <- as.data.frame(indiv_results$median)
  r_indiv_lower <- as.data.frame(indiv_results$lower)
  r_indiv_upper <- as.data.frame(indiv_results$upper)
''')
robj_csmf = robjects.globalenv["r_out_csmf"]
robj_imean = robjects.globalenv["r_indiv_mean"]
robj_imed = robjects.globalenv["r_indiv_med"]
robj_ilow = robjects.globalenv["r_indiv_lower"]
robj_iupp = robjects.globalenv["r_indiv_upper"]

with localconverter(robjects.default_converter + pandas2ri.converter):
    r_csmf = get_conversion().rpy2py(robj_csmf)
    r_indiv_mean = get_conversion().rpy2py(robj_imean)
    r_indiv_med = get_conversion().rpy2py(robj_imed)
    r_indiv_low = get_conversion().rpy2py(robj_ilow)
    r_indiv_upp = get_conversion().rpy2py(robj_iupp)

va_data = get_vadata("randomva5", verbose=False)
with pytest.warns(UserWarning):
    py_out = InSilicoVA(va_data)
py_results = py_out.get_results()
py_csmf = py_results.get_csmf(top=61)
py_indiv = get_indiv(py_results, va_data)
py_indiv_mean = py_indiv[py_indiv["Statistic"] == "mean"].drop(
    ["Statistic"], axis=1)
py_indiv_mean.set_index(["ID"], inplace=True)
py_indiv_med = py_indiv[py_indiv["Statistic"] == "median"].drop(
    ["Statistic"], axis=1)
py_indiv_med.set_index(["ID"], inplace=True)
py_indiv_low = py_indiv[py_indiv["Statistic"] == "lower"].drop(
    ["Statistic"], axis=1)
py_indiv_low.set_index(["ID"], inplace=True)
py_indiv_upp = py_indiv[py_indiv["Statistic"] == "upper"].drop(
    ["Statistic"], axis=1)
py_indiv_upp.set_index(["ID"], inplace=True)


def test_csmf_mean_top_15():
    diff = abs(r_csmf["Mean"][0:14] - py_csmf["Mean"][0:14])
    assert all(diff < 0.005)


def test_csmf_all():
    for i in range(r_csmf.shape[0]):
        for j in range(r_csmf.shape[1]):
            a = r_csmf.iloc[i, j]
            b = py_csmf.iloc[i, j]
            print(f"{abs(a - b) < 0.015}\n")
            assert abs(a - b) < 0.015


# Arbitrary (not sure of a good way to do this)
def test_indiv_mean():
    diff_mean = abs(r_indiv_mean - py_indiv_mean)
    big_diff_cols = diff_mean.quantile(.95, axis=0) > 0.05
    big_diff_rows = diff_mean.quantile(.95, axis=1) > 0.05
    assert big_diff_rows.sum() == 0
    assert big_diff_cols.sum() == 0

# for i in range(r_indiv_mean.shape[0]):
#     for j in range(r_indiv_mean.shape[1]):
#         d1 = abs(r_indiv_mean.iat[i, j] - py_indiv_mean.iat[i, j])
#         d = abs(r_indiv_mean.iat[i, j] - py_indiv_mean.iat[i, j]) < 0.01
#         if not d:
#             print(f"row {i}, col {j}: {d1}\n")


def test_indiv_median():
    diff_med = abs(r_indiv_med - py_indiv_med)
    big_diff_cols = diff_med.quantile(.95, axis=0) > 0.05
    big_diff_rows = diff_med.quantile(.95, axis=1) > 0.05
    assert big_diff_rows.sum() == 0
    assert big_diff_cols.sum() == 0


def test_indiv_lower():
    diff_low = abs(r_indiv_low - py_indiv_low)
    big_diff_cols = diff_low.quantile(.95, axis=0) > 0.05
    big_diff_rows = diff_low.quantile(.95, axis=1) > 0.05
    assert big_diff_rows.sum() == 0
    assert big_diff_cols.sum() == 0


def test_indiv_upper():
    diff_upp = abs(r_indiv_upp - py_indiv_upp)
    big_diff_cols = diff_upp.quantile(.95, axis=0) > 0.05
    big_diff_rows = diff_upp.quantile(.95, axis=1) > 0.05
    assert big_diff_rows.sum() == 0
    assert big_diff_cols.sum() == 0
