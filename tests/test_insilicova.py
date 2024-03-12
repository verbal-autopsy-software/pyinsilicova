# -*- coding: utf-8 -*-

import pandas as pd
import pytest
import numpy as np
import os
import sys
from typing import Dict
# import threading
from insilicova.api import InSilicoVA
from insilicova.exceptions import (ArgumentException, DataException,
                                   HaltGUIException, SamplerException)
from insilicova.utils import get_vadata
from insilicova.structures import InSilico, InSilicoAllExt

va_data = get_vadata("randomva5", verbose=False)


with pytest.warns(UserWarning):
        default = InSilicoVA(va_data, subpop=["i019a"], n_sim=1000, burnin=300,
                             thin=10, auto_length=False)

va_data1 = get_vadata("randomva1", verbose=False)
probbase3 = get_vadata("probbaseV3", verbose=False)
probbase5 = get_vadata("probbaseV5", verbose=False)
causetext = get_vadata("causetext", verbose=False)
causetext5 = get_vadata("causetextV5", verbose=False)


class TestChangeDataCoding:
    tmp_data1 = va_data.copy()
    tmp_data1.iloc[0:10, 1] = "maybe"
    spop = "i019a"
    tmp_data1[spop] = tmp_data1[spop].replace(".", np.NaN)
    tmp_out1 = InSilicoVA(tmp_data1, subpop=spop, run=False)
    tmp_out1._change_data_coding()
    tmp_data_exc = va_data.copy()
    tmp_data_exc.iloc[1, 0:10] = ""

    def test_change_values(self):
        assert all(self.tmp_out1.data.iloc[0:10, 1] == ".")

    def test_blank_exception(self):
        with pytest.raises(DataException):
            InSilicoVA(data=self.tmp_data_exc)


def test_data_type_warning():
    with pytest.warns(UserWarning):
        InSilicoVA(data=va_data, data_type="WHO2016", datacheck=False,
                   n_sim=10, burnin=1, thin=1, auto_length=False)


def test_wrong_data_type():
    with pytest.raises(ArgumentException):
        InSilicoVA(data=va_data, data_type="2016WHO")


def test_null_jump_scale():
    with pytest.raises(ArgumentException):
        InSilicoVA(data=va_data, jump_scale=None)


def test_null_chain_etc():
    with pytest.raises(ArgumentException):
        InSilicoVA(data=va_data, n_sim=None, thin=None, burnin=None)


# def test_probbase_by_symp_exc():
#     with pytest.raises(ArgumentException):
#         InSilicoVA(data=va_data,
#                    keep_probbase_level=True,
#                    probbase_by_symp_dev=True)


@pytest.mark.skip("A reminder to update this test when fnc is created")
def test_warning_write(tmp_path):
    with pytest.warns(UserWarning):
        results = InSilicoVA(data=va_data,
                             directory=str(tmp_path),
                             warning_write=True,
                             n_sim=10, burnin=1, thin=1, auto_length=False)
    log_file = tmp_path / "errorlog_insilico.txt"
    assert os.path.isfile(log_file)


@pytest.mark.xfail(sys.platform == "win32",
                   reason="doesn't trigger windows permission error")
def test_warning_write_exception():
    with pytest.raises(ArgumentException):
        InSilicoVA(data=va_data, warning_write=True, directory="/bad/path")


def test_va_data_has_col_i183o():
    tmp_data = va_data.rename(columns={"i183a": "i183o"}).copy()
    tmp_data.loc[0:18, "i183o"] = "y"
    out = InSilicoVA(data=tmp_data, run=False)
    out._change_data_coding()
    with pytest.warns(UserWarning):
        out._check_args()
    assert "i183o" in list(tmp_data)
    assert "i183a" in list(out.data)


def test_arg_probbase_who2012():
    out = InSilicoVA(data=va_data1, data_type="WHO2012",
                     groupcode=False, run=False)
    out._initialize_data_dependencies()
    assert isinstance(out._probbase, np.ndarray)
    assert out._probbase.shape == probbase3.shape
    assert all(
        out._causetext.fillna(".").iloc[:, 0] == causetext.fillna(".").iloc[:, 0])
    assert all(
        out._causetext.fillna(".").iloc[:, 1] == causetext.fillna(".").iloc[:, 1])


def test_arg_probbase_who2016():
    out = InSilicoVA(data=va_data, data_type="WHO2016",
                     groupcode=False, run=False)
    out._initialize_data_dependencies()
    assert isinstance(out._probbase, np.ndarray)
    assert out._probbase.shape == probbase5.shape
    assert out._probbase.dtype.type is np.str_
    assert all(out._causetext.iloc[:, 0] == causetext5.iloc[:, 0])
    assert all(out._causetext.iloc[:, 1] == causetext5.iloc[:, 1])


def test_arg_probbase_custom():
    tmp_pb = probbase5.copy()
    version = "new _probbase"
    tmp_pb.iloc[0, 2] = version
    out = InSilicoVA(data=va_data, sci=tmp_pb, run=False)
    out._initialize_data_dependencies()
    assert out._probbase[0, 2] == version


def test_arg_probbase_custom_exc():
    sci = pd.DataFrame(np.random.rand(10, 10))
    with pytest.raises(ArgumentException):
        InSilicoVA(data=va_data, sci=sci)


def test_arg_groupcode_true():
    out = InSilicoVA(data=va_data, groupcode=True, run=False)
    out._initialize_data_dependencies()
    assert all(out._causetext.iloc[:, 1] == causetext5.iloc[:, 2])


def test_arg_groupcode_2012_true():
    out = InSilicoVA(data=va_data1, data_type="WHO2012",
                     groupcode=True, run=False)
    out._initialize_data_dependencies()
    assert all(
        out._causetext.fillna(".").iloc[:, 1] == causetext.fillna(".").iloc[:, 2])


def test_subpop_exception():
    with pytest.raises(ArgumentException):
        InSilicoVA(data=va_data, subpop=["not_a_column"])


def test_subpop_setup():
    subpop_out = InSilicoVA(va_data,
                            subpop=["i004a", "i019a", "i022a"],
                            run=False)
    subpop_out._extract_subpop()
    assert subpop_out.subpop.ndim == 1
    assert subpop_out.subpop.shape[0] == subpop_out.data.shape[0]
    assert isinstance(subpop_out.subpop, pd.Series)


def test_subpop_setup_1_level():
    tmp_data = va_data.copy()
    tmp_data.iloc[:, 3] = "one"
    subpop_out = InSilicoVA(va_data,
                            subpop=tmp_data.iloc[:, 3], run=False)
    with pytest.warns(UserWarning):
        subpop_out._extract_subpop()
    assert subpop_out.subpop is None


def test_prep_data_error_2012():
    tmp_data = va_data1.copy()
    tmp_data.drop(columns=va_data1.columns[33], inplace=True)
    with pytest.raises(DataException):
        out = InSilicoVA(tmp_data, data_type="WHO2012")


def test_prep_data_2012():
    tmp_data = va_data1.copy()
    tmp_data.columns = tmp_data.columns.str.upper()
    tmp_data["new_col"] = "not a VA column"
    out = InSilicoVA(tmp_data, data_type="WHO2012", run=False)
    out._change_data_coding()
    out._check_args()
    out._initialize_data_dependencies()
    out._change_data_coding()
    out._prep_data()
    assert out.data.shape == va_data1.shape
    assert all(out.data.columns == va_data1.columns)


def test_prep_data_error_2016():
    tmp_data = va_data.copy()
    tmp_data.drop(columns=va_data.columns[33], inplace=True)
    with pytest.raises(DataException):
        out = InSilicoVA(tmp_data, data_type="WHO2016")


def test_prep_data_no_valid_records():
    tmp_data = va_data.copy()
    tmp_data.iloc[:, 1:] = "."
    out = InSilicoVA(tmp_data, data_type="WHO2016", run=False)
    assert len(out._error_log) == 0
    with pytest.warns(UserWarning):
        out._run()
    assert hasattr(out, "_data_checked") is False
    assert len(out._error_log) == tmp_data.shape[0]


def test_prep_data_2016():
    # not removing external causes & symptoms
    tmp_data = va_data.copy()
    tmp_data.columns = tmp_data.columns.str.upper()
    tmp_data["new_col"] = "not a VA column"
    out = InSilicoVA(tmp_data, data_type="WHO2016", run=False)
    out._change_data_coding()
    out._check_args()
    out._initialize_data_dependencies()
    out._change_data_coding()
    out._prep_data()
    assert out.data.shape == va_data.shape
    assert all(out.data.columns == va_data.columns)


def test_prep_data_2016_with_remove_external():
    tmp_data = va_data.copy()
    tmp_data.columns = tmp_data.columns.str.upper()
    tmp_data["new_col"] = "not a VA column"
    out = InSilicoVA(tmp_data, data_type="WHO2016", run=False)
    out._change_data_coding()
    out._check_args()
    out._initialize_data_dependencies()
    out._change_data_coding()
    out._prep_data()
    out._standardize_upper()
    out._datacheck()
    out._remove_external_causes()
    ext_shape = (len(out._ext_id), len(out._external_symps))
    out_shape = [a + b for a, b in zip(out.data.shape, ext_shape)]
    assert tuple(out_shape) == va_data.shape
    assert set(list(out.data)).issubset(set(list(va_data)))


def test_prep_data_change_label():
    tmp_data = va_data.copy()
    # new_names = list(tmp_data)[100]
    # new_names.extend([x + "_new" for x in tmp_data.columns[100:]])
    # tmp_data.columns = new_names
    tmp_data.rename(columns={"i019a": "imale", "i019b": "ifema"},
                    inplace=True)
    pb = probbase5.copy()
    pb.replace(to_replace=r"^i019a", value="imale",
               regex=True, inplace=True)
    pb.replace(to_replace=r"^i019b", value="ifema",
               regex=True, inplace=True)
    out = InSilicoVA(data=tmp_data, sci=pb, run=False)
    out._change_data_coding()
    out._check_args()
    out._initialize_data_dependencies()
    out._change_data_coding()
    with pytest.warns(UserWarning):
        out._prep_data()
    assert all(out.data.columns == va_data.columns)


def test_prep_data_cond_prob():
    pb = probbase5.iloc[1:, slice(20, 81)].copy()
    new_names = list(pb)[0:20]
    new_names.extend([x + "_new" for x in pb.columns[20:]])
    pb.columns = new_names
    out = InSilicoVA(data=va_data, cond_prob=pb, run=False)
    out._change_data_coding()
    out._check_args()
    out._initialize_data_dependencies()
    out._change_data_coding()
    out._prep_data()
    assert out.exclude_impossible_causes == "none"
    assert all(out._va_causes.tolist() == pb.columns)


class TestInterVATable:
    results = InSilicoVA(data=va_data, run=False)
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
        with pytest.raises(ArgumentException):
            self.results._interva_table(standard=True)

    def test_interva_table_exception_table_num_dev(self):
        with pytest.raises(ArgumentException):
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
    df_age = df_age.astype({i: "O" for i in age_col})
    df_age.loc[age_ind, age_col] = "N"
    ins_bad_age = InSilicoVA(df_age, data_type="WHO2012", run=False)
    ins_bad_age._change_data_coding()
    ins_bad_age._check_args()
    ins_bad_age._initialize_data_dependencies()
    ins_bad_age._change_data_coding()
    ins_bad_age._prep_data()
    # ins_bad_age._remove_bad(is_numeric=False, version5=False)
    bad_age_id = set(df_age["ID"])

    n_bad_sex = 10
    df_sex = va_data1.copy()
    sex_ind = df_sex.index[0:n_bad_sex]
    df_sex.loc[sex_ind, ["male", "female"]] = "N"
    ins_bad_sex = InSilicoVA(df_sex, data_type="WHO2012", run=False)
    ins_bad_sex._change_data_coding()
    ins_bad_sex._check_args()
    ins_bad_sex._initialize_data_dependencies()
    ins_bad_sex._change_data_coding()
    ins_bad_sex._prep_data()
    # ins_bad_sex._remove_bad(is_numeric=False, version5=False)
    bad_sex_id = set(df_sex["ID"][0:n_bad_sex])

    n_bad_all = 15
    df_all = va_data1.copy()
    all_ind = df_all.index[0:n_bad_all]
    data_col = list(va_data1)
    remove_col = age_col.copy()
    remove_col.extend(["ID", "male", "female"])
    keep_col = list(set(data_col) - set(remove_col))
    df_all = df_all.astype({i: "O" for i in keep_col})
    df_all.loc[all_ind, keep_col] = "N"
    ins_bad_all = InSilicoVA(df_all, data_type="WHO2012", run=False)
    ins_bad_all._change_data_coding()
    ins_bad_all._check_args()
    ins_bad_all._initialize_data_dependencies()
    ins_bad_all._change_data_coding()
    ins_bad_all._prep_data()
    # ins_bad_all._remove_bad(is_numeric=False, version5=False)
    bad_all_id = set(df_all["ID"])

    def test_remove_bad_age(self):
        old_n = self.ins_bad_age.original_data.shape[0]
        new_n = self.ins_bad_age.data.shape[0]
        # assumes randomva1 has no missing age values
        assert new_n == (old_n - self.n_bad_age)

    def test_err_bad_age(self):
        err_ids = set(self.ins_bad_age._error_log.keys())
        err_id = err_ids.pop()
        err_msg = "Error in age indicator: not specified"
        assert err_ids.issubset(self.bad_age_id)
        assert self.ins_bad_age._error_log[err_id] == [err_msg]

    def test_remove_bad_sex(self):
        old_n = self.ins_bad_sex.original_data.shape[0]
        new_n = self.ins_bad_sex.data.shape[0]
        # assumes randomva1 has no missing sex values
        assert new_n == (old_n - self.n_bad_sex)

    def test_err_bad_sex(self):
        err_ids = set(self.ins_bad_sex._error_log.keys())
        err_id = err_ids.pop()
        err_msg = "Error in sex indicator: not specified"
        assert err_ids.issubset(self.bad_sex_id)
        assert self.ins_bad_sex._error_log[err_id] == [err_msg]

    def test_remove_bad_all(self):
        old_n = self.ins_bad_all.original_data.shape[0]
        new_n = self.ins_bad_all.data.shape[0]
        # assumes randomva1 has no deaths with all missing (except age & sex)
        assert new_n == (old_n - self.n_bad_all)

    def test_err_bad_all(self):
        err_ids = set(self.ins_bad_all._error_log.keys())
        err_id = err_ids.pop()
        err_msg = "Error in indicators: no symptoms specified"
        assert err_ids.issubset(self.bad_all_id)
        assert self.ins_bad_all._error_log[err_id] == [err_msg]


class TestRemoveBadV5:
    n_bad_age = 5
    df_age = va_data.copy()
    age_ind = df_age.index[0:n_bad_age]

    age_col = ["i022" + x for x in ["a", "b", "c", "d", "e", "f", "g"]]
    df_age.loc[age_ind, age_col] = "N"
    ins_bad_age = InSilicoVA(df_age, run=False)
    ins_bad_age._change_data_coding()
    ins_bad_age._check_args()
    ins_bad_age._initialize_data_dependencies()
    ins_bad_age._change_data_coding()
    ins_bad_age._prep_data()
    bad_age_id = set(df_age["ID"])
    # ins_bad_age._remove_bad(is_numeric=False)

    n_bad_sex = 10
    df_sex = va_data.copy()
    sex_ind = df_sex.index[0:n_bad_sex]
    df_sex.loc[sex_ind, ["i019a", "i019b"]] = "N"
    ins_bad_sex = InSilicoVA(df_sex, run=False)
    ins_bad_sex._change_data_coding()
    ins_bad_sex._check_args()
    ins_bad_sex._initialize_data_dependencies()
    ins_bad_sex._change_data_coding()
    ins_bad_sex._prep_data()
    bad_sex_id = set(df_sex["ID"][0:n_bad_sex])
    # ins_bad_sex._remove_bad(is_numeric=False)

    n_bad_all = 15
    df_all = va_data.copy()
    all_ind = df_all.index[0:n_bad_all]
    data_col = list(va_data)
    remove_col = age_col.copy()
    remove_col.extend(["ID", "i019a", "i019b"])
    keep_col = list(set(data_col) - set(remove_col))
    df_all.loc[all_ind, keep_col] = "N"
    ins_bad_all = InSilicoVA(df_all, run=False)
    ins_bad_all._change_data_coding()
    ins_bad_all._check_args()
    ins_bad_all._initialize_data_dependencies()
    ins_bad_all._change_data_coding()
    ins_bad_all._prep_data()
    bad_all_id = set(df_all["ID"])
    # ins_bad_all._remove_bad(is_numeric=False)

    def test_remove_bad_age(self):
        old_n = self.ins_bad_age.original_data.shape[0]
        new_n = self.ins_bad_age.data.shape[0]
        # assumes randomva1 has no missing age values
        assert new_n == (old_n - self.n_bad_age)

    def test_err_bad_age(self):
        err_ids = set(self.ins_bad_age._error_log.keys())
        err_id = err_ids.pop()
        err_msg = ["Error in age indicator: not specified"]
        assert err_ids.issubset(self.bad_age_id)
        assert self.ins_bad_age._error_log[err_id] == err_msg

    def test_remove_bad_sex(self):
        old_n = self.ins_bad_sex.original_data.shape[0]
        new_n = self.ins_bad_sex.data.shape[0]
        # assumes randomva1 has no missing sex values
        assert new_n == (old_n - self.n_bad_sex)

    def test_err_bad_sex(self):
        err_ids = set(self.ins_bad_sex._error_log.keys())
        err_id = err_ids.pop()
        err_msg = ["Error in sex indicator: not specified"]
        assert err_ids.issubset(self.bad_sex_id)
        assert self.ins_bad_sex._error_log[err_id] == err_msg

    def test_remove_bad_all(self):
        old_n = self.ins_bad_all.original_data.shape[0]
        new_n = self.ins_bad_all.data.shape[0]
        # assumes randomva1 has no deaths with all missing (except age & sex)
        assert new_n == (old_n - self.n_bad_all)

    def test_err_bad_all(self):
        err_ids = set(self.ins_bad_all._error_log.keys())
        err_id = err_ids.pop()
        err_msg = ["Error in indicators: no symptoms specified"]
        assert err_ids.issubset(self.bad_all_id)
        assert self.ins_bad_all._error_log[err_id] == err_msg


def test_datacheck5_values():
    assert default.data.iloc[:, 1:].isin(["Y", "", "."]).all(axis=None)


def test_datacheck5_checked_data():
    assert default._data_checked.iloc[:, 1:].isin(["y", "n", "-"]).all(
        axis=None)


def test_datacheck5_log():
    assert len(default._warning["first_pass"]) > 0
    assert len(default._warning["second_pass"]) > 0


class TestRemoveExt:
    n_external = 19
    tmp_data = va_data.copy()
    all_external = va_data.copy()
    orig_shape = va_data.shape
    ext_causes = np.arange(49, 60)
    ext_symptoms = np.arange(19, 38)

    ext_col = tmp_data.iloc[:, 1:].columns[ext_symptoms]
    tmp_data.loc[:, ext_col] = "n"
    tmp_data.loc[tmp_data.index[0:n_external], ext_col[0]] = "y"
    all_external.loc[:, ext_col] = "n"
    all_external.loc[:, ext_col[0]] = "y"
    tmp_out = InSilicoVA(tmp_data, data_type="WHO2016",
                         subpop=["i019a"], run=False)
    tmp_out._change_data_coding()
    tmp_out._check_args()
    tmp_out._initialize_data_dependencies()
    tmp_out._extract_subpop()
    tmp_out._prep_data()
    tmp_out._datacheck()
    tmp_out._remove_external_causes()

    def test_data_shape(self):
        assert self.tmp_out.data.shape[0] == (
                    self.orig_shape[0] - self.n_external)

    def test_ext_id_shape(self):
        assert self.tmp_out._ext_id.shape[0] == self.n_external

    def test_ext_sub_shape(self):
        assert self.tmp_out._ext_sub.shape[0] == self.n_external

    def test_ext_csmf(self):
        assert self.tmp_out._ext_csmf is not None

    def test_negate_shape(self):
        # subtract 1 extra for the ID column
        assert self.tmp_out._negate.shape[0] == (self.orig_shape[1] - 1 -
                                                 self.ext_symptoms.shape[0])

    def test_all_external_results(self):
        with pytest.warns(UserWarning):
            all_ext_out = InSilicoVA(self.all_external,
                                     data_type="WHO2016",
                                     subpop=["i019a"],
                                     n_sim=10, burnin=1,
                                     thin=1, auto_length=False)
        # add 1 for ID column
        assert all_ext_out.results.causes.shape == (self.all_external.shape[0],
                                                    len(self.ext_causes) + 1)


class TestCheckMissingAll:
    tmp_data = va_data.copy()
    tmp_data.iloc[:, 1:] = "y"
    missing_col_names = ["i109o", "i127o", "i214o"]
    n_miss = len(missing_col_names)
    tmp_data[missing_col_names] = "."
    tmp_out = InSilicoVA(tmp_data,
                         datacheck=False,
                         data_type="WHO2016",
                         run=False)
    tmp_out._change_data_coding()
    with pytest.warns(UserWarning):
        tmp_out._check_args()
    tmp_out._initialize_data_dependencies()
    tmp_out._prep_data()
    pre_nrow_prob_orig = tmp_out._prob_orig.shape[0]
    pre_n_negate = tmp_out._negate.shape[0]
    with pytest.warns(UserWarning):
        tmp_out._check_missing_all()

    def test_columns_are_removed_from_data(self):
        data_names = set(list(self.tmp_out.data))
        missing_names = set(self.missing_col_names)
        assert data_names.isdisjoint(missing_names)

    def test_rows_are_removed_from_prob_orig(self):
        assert self.tmp_out._prob_orig.shape[0] == (
                self.pre_nrow_prob_orig - self.n_miss)

    def test_columns_are_removed_from_negate(self):
        assert self.tmp_out._negate.shape[0] == (
                self.pre_n_negate - self.n_miss)

    @pytest.mark.parametrize("test_input", missing_col_names)
    def test_columns_rows_are_removed_from_probbase(self, test_input):
        assert test_input not in self.tmp_out._prob_orig[:, 0]


def test_warning_datacheck_missing():
    with pytest.warns(UserWarning):
        tmp_out = InSilicoVA(va_data,
                             datacheck_missing=False,
                             datacheck=False,
                             n_sim=10, burnin=1, thin=1, auto_length=False)


def test_initialize_numerical_matrix_1():
    """_customization_dev = False & cond_prob_num = None"""
    assert isinstance(default._prob_order, np.ndarray)
    assert isinstance(default._cond_prob_true, np.ndarray)


def test_initialize_numerical_matrix_2():
    """_customization_dev = False & cond_prob_num not None"""
    new_cond_prob_num = probbase5.iloc[1:, 20:81].copy()
    new_cond_prob_num.iloc[20, 20] = "A+"
    tmp_out = InSilicoVA(va_data,
                         cond_prob_num=new_cond_prob_num,
                         run=False)
    tmp_out._change_data_coding()
    tmp_out._check_args()
    tmp_out._initialize_data_dependencies()
    tmp_out._prep_data()
    tmp_out._standardize_upper()
    tmp_out._datacheck()
    tmp_out._remove_external_causes()
    with pytest.warns(UserWarning):
        tmp_out._check_missing_all()
    tmp_out._initialize_numerical_matrix()

    assert isinstance(tmp_out._prob_order, np.ndarray)
    assert isinstance(tmp_out._cond_prob_true, np.ndarray)


# _customization_dev not yet implemented
# def test_initialize_numerical_matrix_3():
#     """_customization_dev = True & update_cond_prob = True"""
#     tmp_out = InSilicoVA(va_data,
#                          _customization_dev=True,
#                          update_cond_prob=True)
#     assert isinstance(tmp_out._prob_order, np.ndarray)
#     assert isinstance(tmp_out._cond_prob_true, np.ndarray)
#
#
# _customization_dev not yet implemented
# def test_initialize_numerical_matrix_4():
#     """_customization_dev = True & update_cond_prob = False"""
#     tmp_out = InSilicoVA(va_data,
#                          _customization_dev=True,
#                          update_cond_prob=False)
#     assert isinstance(tmp_out._prob_order, np.ndarray)
#     assert isinstance(tmp_out._cond_prob_true, np.ndarray)


def test_check_data_dimensions_warn_n_obs():
    tmp_out = InSilicoVA(va_data, run=False)
    tmp_out._change_data_coding()
    tmp_out._check_args()
    tmp_out._initialize_data_dependencies()
    tmp_out._prep_data()
    tmp_out._remove_external_causes()
    tmp_out._initialize_numerical_matrix()
    tmp_out.data = pd.DataFrame(tmp_out.data.iloc[0, :]).transpose()
    with pytest.raises(DataException):
        tmp_out._check_data_dimensions()


def test_check_data_dimensions_warn_n_symptoms():
    tmp_out = InSilicoVA(va_data, run=False)
    tmp_out._change_data_coding()
    tmp_out._check_args()
    tmp_out._initialize_data_dependencies()
    tmp_out._prep_data()
    tmp_out._remove_external_causes()
    tmp_out._initialize_numerical_matrix()
    keep_columns = list(tmp_out.data)[0:250]
    tmp_out.data.drop(columns=keep_columns, inplace=True)
    with pytest.raises(DataException):
        tmp_out._check_data_dimensions()


def test_check_data_dimensions_va_causes_current():
    assert isinstance(default._va_causes_current, pd.Series)


def test_check_impossible_paris():
    symp = ["i125o", "i127o", "i128o"]
    # these symptoms have some non-missing values, so they will not get removed
    # e.g. va_data["i25o"].value_counts()
    cause = ["b_0299", "b_0499", "b_0502"]
    impossible = pd.DataFrame({"symp": symp, "cause": cause, "val": [0, 0, 1]})
    with pytest.warns(UserWarning):
        tmp_out = InSilicoVA(va_data,
                             exclude_impossible_causes="none",
                             impossible_combination=impossible,
                             n_sim=10, burnin=1, thin=1, auto_length=False)
    assert tmp_out._impossible.shape == impossible.shape


def test_prior_truncated_beta_1():
    tmp_out = InSilicoVA(va_data,
                         subpop="i019a",
                         run=False)
    tmp_out._change_data_coding()
    tmp_out._check_args()
    tmp_out._initialize_data_dependencies()
    tmp_out._extract_subpop()
    tmp_out._prep_data()
    tmp_out._standardize_upper()
    tmp_out._datacheck()
    tmp_out._remove_external_causes()
    with pytest.warns(UserWarning):
        tmp_out._check_missing_all()
    tmp_out._initialize_numerical_matrix()
    tmp_out._check_data_dimensions()
    tmp_out._check_impossible_pairs()
    n = tmp_out.data.shape[0]
    prior_b_cond_tmp = np.floor(1.5 * n)
    levels_count_med = pd.DataFrame(
        tmp_out._cond_prob_true
    ).stack().value_counts().median()
    prior_b_cond = prior_b_cond_tmp * levels_count_med * tmp_out.levels_strength
    expected = int(str(prior_b_cond).split(".")[0])
    tmp_out._prior_truncated_beta()
    assert expected == tmp_out._prior_b_cond
    assert tmp_out.levels_prior is not None


def test_prior_truncated_beta_2():
    tmp_out = InSilicoVA(va_data,
                         subpop="i019a",
                         keep_probbase_level=False,
                         run=False)
    tmp_out._change_data_coding()
    tmp_out._check_args()
    tmp_out._initialize_data_dependencies()
    tmp_out._extract_subpop()
    tmp_out._prep_data()
    tmp_out._standardize_upper()
    tmp_out._datacheck()
    tmp_out._remove_external_causes()
    with pytest.warns(UserWarning):
        tmp_out._check_missing_all()
    tmp_out._initialize_numerical_matrix()
    tmp_out._check_data_dimensions()
    tmp_out._check_impossible_pairs()
    n = tmp_out.data.shape[0]
    prior_b_cond_tmp = np.floor(1.5 * n)
    prior_b_cond = prior_b_cond_tmp * tmp_out.levels_strength
    expected = int(str(prior_b_cond).split(".")[0])
    tmp_out._prior_truncated_beta()
    assert expected == tmp_out._prior_b_cond
    assert tmp_out.levels_prior is not None


def test_get_subpop_info_warning():
    tmp_out = InSilicoVA(va_data, subpop="i019a", run=False)
    tmp_out._change_data_coding()
    tmp_out._check_args()
    tmp_out._initialize_data_dependencies()
    tmp_out._extract_subpop()
    tmp_out._prep_data()
    tmp_out._standardize_upper()
    tmp_out._datacheck()
    tmp_out._remove_external_causes()
    with pytest.warns(UserWarning):
        tmp_out._check_missing_all()
    tmp_out._initialize_numerical_matrix()
    tmp_out._check_data_dimensions()
    tmp_out._check_impossible_pairs()
    tmp_out._prior_truncated_beta()
    tmp_out.subpop = pd.DataFrame({"A": [1]})
    with pytest.raises(ArgumentException):
        tmp_out._get_subpop_info()


def test_get_subpop_info():
    tmp_out = InSilicoVA(va_data, subpop="i019a", run=False)
    tmp_out._change_data_coding()
    tmp_out._check_args()
    tmp_out._initialize_data_dependencies()
    tmp_out._extract_subpop()
    tmp_out._prep_data()
    tmp_out._standardize_upper()
    tmp_out._datacheck()
    tmp_out._remove_external_causes()
    with pytest.warns(UserWarning):
        tmp_out._check_missing_all()
    tmp_out._initialize_numerical_matrix()
    tmp_out._check_data_dimensions()
    tmp_out._check_impossible_pairs()
    tmp_out._prior_truncated_beta()
    tmp_out._get_subpop_info()
    assert set(tmp_out._sublist.keys()) == {"y", "."}


class TestInitializeIndicatorMatrix:

    tmp_out = InSilicoVA(va_data, subpop="i019a", run=False)
    tmp_out._change_data_coding()
    tmp_out._check_args()
    tmp_out._initialize_data_dependencies()
    tmp_out._extract_subpop()
    tmp_out._prep_data()
    tmp_out._standardize_upper()
    tmp_out._datacheck()
    tmp_out._remove_external_causes()
    with pytest.warns(UserWarning):
        tmp_out._check_missing_all()
    tmp_out._initialize_numerical_matrix()
    tmp_out._check_data_dimensions()
    tmp_out._check_impossible_pairs()
    tmp_out._prior_truncated_beta()
    tmp_out._get_subpop_info()
    tmp_out._initialize_indicator_matrix()

    def test_indic(self):
        assert np.in1d(self.tmp_out._indic, [0, 1, -1, -2]).all()

    def test_id(self):
        assert isinstance(self.tmp_out._id, pd.Series)

    def test_contains_missing(self):
        assert isinstance(self.tmp_out._contains_missing, np.bool_)


def test_initialize_parameters():
    tmp_out = InSilicoVA(va_data,
                         subpop="i019a",
                         run=False)
    tmp_out._change_data_coding()
    tmp_out._check_args()
    tmp_out._initialize_data_dependencies()
    tmp_out._extract_subpop()
    tmp_out._prep_data()
    tmp_out._standardize_upper()
    tmp_out._datacheck()
    tmp_out._remove_external_causes()
    with pytest.warns(UserWarning):
        tmp_out._check_missing_all()
    tmp_out._initialize_numerical_matrix()
    tmp_out._check_data_dimensions()
    tmp_out._check_impossible_pairs()
    tmp_out._prior_truncated_beta()
    tmp_out._get_subpop_info()
    tmp_out._initialize_indicator_matrix()
    tmp_out._initialize_parameters()

    assert isinstance(tmp_out._mu, np.ndarray)
    assert tmp_out._sigma2 == 1.0
    assert isinstance(tmp_out._cond_prob, np.ndarray)
    assert isinstance(tmp_out._dist, np.ndarray)
    assert isinstance(tmp_out._level_exist, np.ndarray)
    assert isinstance(tmp_out._N_level, int)


def test_sampler():
    tmp_out = InSilicoVA(va_data,
                         subpop="i019a",
                         run=False,
                         n_sim=1000,
                         burnin=500,
                         thin=10,
                         auto_length=False)
    tmp_out._change_data_coding()
    tmp_out._check_args()
    tmp_out._initialize_data_dependencies()
    tmp_out._extract_subpop()
    tmp_out._prep_data()
    tmp_out._standardize_upper()
    tmp_out._datacheck()
    tmp_out._remove_external_causes()
    with pytest.warns(UserWarning):
        tmp_out._check_missing_all()
    tmp_out._initialize_numerical_matrix()
    tmp_out._check_data_dimensions()
    tmp_out._check_impossible_pairs()
    tmp_out._physician_codes()
    tmp_out._prior_truncated_beta()
    tmp_out._get_subpop_info()
    tmp_out._initialize_indicator_matrix()
    tmp_out._initialize_parameters()
    with pytest.warns(UserWarning):
        tmp_out._sample_posterior()

    assert isinstance(tmp_out._posterior_results, Dict)


def test_results():
    assert isinstance(default.results, InSilico)


def test_all_external():
    # external causes in external dat set
    # 34, 52, 68, 83, 96, 124, 132, 151, 159, 161, 180, 192 (minus 1)
    ext_deaths = [33, 51, 67, 82, 95, 123, 131, 150, 158, 160, 179, 191]
    tmp_data = va_data.iloc[ext_deaths, :].copy()
    with pytest.warns(UserWarning):
        out = InSilicoVA(tmp_data)
    assert isinstance(out.results, InSilicoAllExt)


# def run_insilicova(ctrl):
#     out = InSilicoVA(va_data, gui_ctrl=ctrl)
#     return out

# ctrl = {"break": False}
# x = threading.Thread(target=run_insilicova, args=(ctrl, ))
# x.start()
# ctrl["break"] = True
