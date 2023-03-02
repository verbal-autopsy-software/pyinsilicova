# -*- coding: utf-8 -*-

import pandas as pd
import pytest
import numpy as np
import os
import sys
from typing import Dict
from insilicova.api import InSilicoVA
from insilicova.indiv import get_indiv
from insilicova.exceptions import ArgumentException, DataException
from insilicova.utils import get_vadata

va_data = get_vadata("randomva5", verbose=False)

default = InSilicoVA(va_data, subpop=["i019a"], n_sim=200, thin=1, burnin=100,
                     auto_length=False)
insilico_obj = default.get_results()

def test_indiv():
    out = get_indiv(insilico_obj=insilico_obj,
                    data=va_data,
                    ci=0.95,
                    is_aggregate=True,
                    by=["i019a"],
                    is_sample=True)
    assert False