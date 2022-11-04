# -*- coding: utf-8 -*-
import pandas as pd
import pytest
import numpy as np
import os
import sys
from insilicova.insilicova import InSilicoVA
from insilicova.exceptions import InSilicoVAException
from insilicova.utils import get_vadata

va_data = get_vadata("randomva5", verbose=False)
default = InSilicoVA(va_data)
probbase5 = get_vadata("probbase5", verbose=False)
va_data1 = get_vadata("randomva1", verbose=False)
default1 = InSilicoVA(va_data1)


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


@pytest.mark.parametrize("test_args, expected",
                         [((2, None), 15), ((None, 2), 15)])
def test_scale_vec_inter(test_args, expected):
    input_vec = np.arange(1, 16)
    output_vec = default._scale_vec_inter(input_vec,
                                          scale=test_args[0],
                                          scale_max=test_args[1])
    assert len(output_vec) == expected
    assert type(output_vec) is np.ndarray
    assert output_vec.argmax() == 0


def test_change_inter():
    input_array = np.array(["I", "A+", "A", "A-", "B+", "B", "B-",
                            "C+", "C", "C-", "D+", "D", "D-", "E", "N"])
    expected_array = np.array([1, 0.8, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005,
                               0.002, 0.001, 0.0005, 0.0001, 0.00001, 0])
    output_array = default._change_inter(input_array)
    np.testing.assert_array_equal(output_array, expected_array)
    # TODO: add another test for _change_iter that reproduces R code
    # insilico_core.r line 571


def test_cond_initiate():
    trunc_min = 0
    trunc_max = 1
    pbase = probbase5.iloc[1:, 20:81].copy()
    prob_order = default._change_inter(pbase.to_numpy(),
                                       order=True,
                                       standard=True)
    cond_prob = default._cond_initiate(prob_order,
                                       exp_ini=True,
                                       inter_ini=True,
                                       probbase_min=trunc_min,
                                       probbase_max=trunc_max)
    assert cond_prob.shape == prob_order.shape
    assert cond_prob.min() >= .01
    assert cond_prob.max() <= .99


class TestRemoveBad:

    n_bad_age = 5
    df_age = va_data1.copy()
    age_ind = df_age.index[0:n_bad_age]
    age_col = ["elder", "midage", "adult", "child",
               "under5", "infant", "neonate"]
    df_age.loc[age_ind, age_col] = "N"
    ins_bad_age = InSilicoVA(df_age)
    bad_age_id = set(df_age["ID"])
    ins_bad_age._remove_bad(is_numeric=False, version5=False)

    n_bad_sex = 10
    df_sex = va_data1.copy()
    sex_ind = df_sex.index[0:n_bad_sex]
    df_sex.loc[sex_ind, ["male", "female"]] = "N"
    ins_bad_sex = InSilicoVA(df_sex)
    bad_sex_id = set(df_sex["ID"][0:n_bad_sex])
    ins_bad_sex._remove_bad(is_numeric=False, version5=False)

    n_bad_all = 15
    df_all = va_data1.copy()
    all_ind = df_all.index[0:n_bad_all]
    data_col = list(va_data1)
    remove_col = age_col.copy()
    remove_col.extend(["ID", "male", "female"])
    keep_col = list(set(data_col) - set(remove_col))
    df_all.loc[all_ind, keep_col] = "N"
    ins_bad_all = InSilicoVA(df_all)
    bad_all_id = set(df_all["ID"])
    ins_bad_all._remove_bad(is_numeric=False, version5=False)

    def test_remove_bad_age(self):
        old_n = self.ins_bad_age.original_data.shape[0]
        new_n = self.ins_bad_age.data.shape[0]
        # assumes randomva1 has no missing age values
        assert new_n == (old_n - self.n_bad_age)

    def test_err_bad_age(self):
        err_ids = set(self.ins_bad_age.error_log.keys())
        err_id = err_ids.pop()
        err_msg = "Error in age indicator: not specified"
        assert err_ids.issubset(self.bad_age_id)
        assert self.ins_bad_age.error_log[err_id] == [err_msg]

    def test_remove_bad_sex(self):
        old_n = self.ins_bad_sex.original_data.shape[0]
        new_n = self.ins_bad_sex.data.shape[0]
        # assumes randomva1 has no missing sex values
        assert new_n == (old_n - self.n_bad_sex)

    def test_err_bad_sex(self):
        err_ids = set(self.ins_bad_sex.error_log.keys())
        err_id = err_ids.pop()
        err_msg = "Error in sex indicator: not specified"
        assert err_ids.issubset(self.bad_sex_id)
        assert self.ins_bad_sex.error_log[err_id] == [err_msg]

    def test_remove_bad_all(self):
        old_n = self.ins_bad_all.original_data.shape[0]
        new_n = self.ins_bad_all.data.shape[0]
        # assumes randomva1 has no deaths with all missing (except age & sex)
        assert new_n == (old_n - self.n_bad_all)

    def test_err_bad_all(self):
        err_ids = set(self.ins_bad_all.error_log.keys())
        err_id = err_ids.pop()
        err_msg = "Error in indicators: no symptoms specified"
        assert err_ids.issubset(self.bad_all_id)
        assert self.ins_bad_all.error_log[err_id] == [err_msg]


class TestRemoveBadV5:

    n_bad_age = 5
    df_age = va_data.copy()
    age_ind = df_age.index[0:n_bad_age]

    age_col = ["i022" + x for x in ["a", "b", "c", "d", "e", "f", "g"]]
    df_age.loc[age_ind, age_col] = "N"
    ins_bad_age = InSilicoVA(df_age)
    bad_age_id = set(df_age["ID"])
    ins_bad_age._remove_bad(is_numeric=False)

    n_bad_sex = 10
    df_sex = va_data.copy()
    sex_ind = df_sex.index[0:n_bad_sex]
    df_sex.loc[sex_ind, ["i019a", "i019b"]] = "N"
    ins_bad_sex = InSilicoVA(df_sex)
    bad_sex_id = set(df_sex["ID"][0:n_bad_sex])
    ins_bad_sex._remove_bad(is_numeric=False)

    n_bad_all = 15
    df_all = va_data.copy()
    all_ind = df_all.index[0:n_bad_all]
    data_col = list(va_data)
    remove_col = age_col.copy()
    remove_col.extend(["ID", "i019a", "i019b"])
    keep_col = list(set(data_col) - set(remove_col))
    df_all.loc[all_ind, keep_col] = "N"
    ins_bad_all = InSilicoVA(df_all)
    bad_all_id = set(df_all["ID"])
    ins_bad_all._remove_bad(is_numeric=False)

    def test_remove_bad_age(self):
        old_n = self.ins_bad_age.original_data.shape[0]
        new_n = self.ins_bad_age.data.shape[0]
        # assumes randomva1 has no missing age values
        assert new_n == (old_n - self.n_bad_age)

    def test_err_bad_age(self):
        err_ids = set(self.ins_bad_age.error_log.keys())
        err_id = err_ids.pop()
        err_msg = ["Error in age indicator: not specified"]
        assert err_ids.issubset(self.bad_age_id)
        assert self.ins_bad_age.error_log[err_id] == err_msg

    def test_remove_bad_sex(self):
        old_n = self.ins_bad_sex.original_data.shape[0]
        new_n = self.ins_bad_sex.data.shape[0]
        # assumes randomva1 has no missing sex values
        assert new_n == (old_n - self.n_bad_sex)

    def test_err_bad_sex(self):
        err_ids = set(self.ins_bad_sex.error_log.keys())
        err_id = err_ids.pop()
        err_msg = ["Error in sex indicator: not specified"]
        assert err_ids.issubset(self.bad_sex_id)
        assert self.ins_bad_sex.error_log[err_id] == err_msg

    def test_remove_bad_all(self):
        old_n = self.ins_bad_all.original_data.shape[0]
        new_n = self.ins_bad_all.data.shape[0]
        # assumes randomva1 has no deaths with all missing (except age & sex)
        assert new_n == (old_n - self.n_bad_all)

    def test_err_bad_all(self):
        err_ids = set(self.ins_bad_all.error_log.keys())
        err_id = err_ids.pop()
        err_msg = ["Error in indicators: no symptoms specified"]
        assert err_ids.issubset(self.bad_all_id)
        assert self.ins_bad_all.error_log[err_id] == err_msg
