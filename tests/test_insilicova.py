# -*- coding: utf-8 -*-

import pytest
import numpy as np
import os
import sys
from interva.interva5 import get_example_input
from insilicova.insilicova import InSilicoVA, InSilicoVAException

va_data = get_example_input()


def test_data_type_warning():
    with pytest.warns(UserWarning):
        InSilicoVA(data=va_data, data_type="WHO2016", datacheck=False)


@pytest.mark.skip("A reminder to update this test when fnc is created")
def test_warning_write(tmp_path):
    results = InSilicoVA(data=va_data,
                         directory=tmp_path,
                         warning_write=True)
    log_file = tmp_path / "errorlog_insilico.txt"
    assert os.path.isfile(log_file)


@pytest.mark.xfail(sys.platform == "win32",
                   reason="doesn't trigger windows permission error")
def test_warning_write_exception():
    with pytest.raises(InSilicoVAException):
        InSilicoVA(data=va_data, warning_write=True, directory="/bad/path")


class TestInterVATable:
    results = InSilicoVA(data=va_data)
    answer = np.array([1, 0.8, 0.5, 0.2, 0.1,
                       0.05, 0.02, 0.01, 0.005, 0.002,
                       0.001, 0.0005, 0.0001, 0.00001, 0])
    interva_table = results._interva_table(min_level=0)

    def test_interva_table_return(self):
        np.testing.assert_array_equal(self.interva_table, self.answer)

    def test_interva_table_rtype(self):
        assert type(self.interva_table) is np.ndarray

    def test_interva_table_table_num_dev(self):
        new_input = self.answer[::-1].copy()
        new_answer = self.answer.copy()
        new_answer[14] = new_answer[13] / 2
        new_interva_table = self.results._interva_table(
            standard=False,
            table_num_dev=new_input)
        np.testing.assert_array_equal(new_answer, new_interva_table)

    def test_interva_table_exception_min_level(self):
        with pytest.raises(InSilicoVAException):
            self.results._interva_table(standard=True)

    def test_interva_table_exception_table_num_dev(self):
        with pytest.raises(InSilicoVAException):
            self.results._interva_table(standard=False)
