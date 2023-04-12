# # -*- coding: utf-8 -*-

import pytest
from insilicova.api import InSilicoVA
from insilicova.utils import get_vadata
from rpy2.robjects.packages import data, importr
import rpy2.robjects as robjects
from rpy2.robjects.conversion import get_conversion, localconverter
from rpy2.robjects import pandas2ri


r_pack = importr("InSilicoVA")
robjects.r('''
  data(RandomVA5)
  r_out <- insilico(RandomVA5, data.type="WHO2016")
  r_out_csmf <- as.data.frame(summary(r_out)$csmf.ordered)
''')
r_csmf = robjects.globalenv["r_out_csmf"]

with localconverter(robjects.default_converter + pandas2ri.converter):
    r_csmf_check = get_conversion().rpy2py(r_csmf)

va_data = get_vadata("randomva5", verbose=False)
py_out = InSilicoVA(va_data)
py_csmf = py_out.get_results().get_csmf(top=61)


def test_csmf_mean_top_15():
    diff = abs(r_csmf_check["Mean"][0:14] - py_csmf["Mean"][0:14])
    assert all(diff < 0.005)


def test_csmf_all():
    for i in range(r_csmf_check.shape[0]):
        for j in range(r_csmf_check.shape[1]):
            a = r_csmf_check.iloc[i, j]
            b = py_csmf.iloc[i][j]
            print(f"{abs(a - b) < 0.015}\n")
            assert abs(a - b) < 0.015
