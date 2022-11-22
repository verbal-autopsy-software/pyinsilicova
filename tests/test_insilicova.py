# -*- coding: utf-8 -*-

import pandas as pd
import pytest
import numpy as np
import os
import sys
from insilicova.insilicova import InSilicoVA
from insilicova.exceptions import ArgumentException, DataException
from insilicova.utils import get_vadata

va_data = get_vadata("randomva5", verbose=False)
default = InSilicoVA(va_data, subpop=["i019a"])
va_data1 = get_vadata("randomva1", verbose=False)
default1 = InSilicoVA(va_data1, data_type="WHO2012")
probbase3 = get_vadata("probbaseV3", verbose=False)
probbase5 = get_vadata("probbaseV5", verbose=False)
causetext = get_vadata("causetext", verbose=False)
causetext5 = get_vadata("causetextV5", verbose=False)


class TestChangeDataCoding:
    tmp_data1 = va_data.copy()
    tmp_data1.iloc[0:10, 1] = "maybe"
    spop = "i019a"
    tmp_data1[spop].replace(".", np.NaN, inplace=True)
    tmp_out1 = InSilicoVA(tmp_data1, subpop=spop)
    tmp_data_exc = va_data.copy()
    tmp_data_exc.iloc[1, 0:10] = ""

    def test_change_values(self):
        assert all(self.tmp_out1.data.iloc[0:10, 1] == ".")

    def test_blank_exception(self):
        with pytest.raises(DataException):
            InSilicoVA(data=self.tmp_data_exc)


def test_data_type_warning():
    with pytest.warns(UserWarning):
        InSilicoVA(data=va_data, data_type="WHO2016", datacheck=False)


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
    results = InSilicoVA(data=va_data,
                         directory=tmp_path,
                         warning_write=True)
    log_file = tmp_path / "errorlog_insilico.txt"
    assert os.path.isfile(log_file)


@pytest.mark.xfail(sys.platform == "win32",
                   reason="doesn't trigger windows permission error")
def test_warning_write_exception():
    with pytest.raises(ArgumentException):
        InSilicoVA(data=va_data, warning_write=True, directory="/bad/path")


def test_va_data_has_col_i183o():
    tmp_data = va_data.rename(columns={"i183a": "i183o"}).copy()
    with pytest.warns(UserWarning):
        out = InSilicoVA(data=tmp_data)
    assert "i183o" in list(tmp_data)
    assert "i183a" in list(out.data)


def test_arg_probbase_who2012():
    out = InSilicoVA(data=va_data1, data_type="WHO2012", groupcode=False)
    assert isinstance(out.probbase, np.ndarray)
    assert out.probbase.shape == probbase3.shape
    assert all(out.causetext.fillna(".").iloc[:, 0] == causetext.fillna(".").iloc[:, 0])
    assert all(out.causetext.fillna(".").iloc[:, 1] == causetext.fillna(".").iloc[:, 1])


def test_arg_probbase_who2016():
    out = InSilicoVA(data=va_data, data_type="WHO2016", groupcode=False)
    assert isinstance(out.probbase, np.ndarray)
    assert out.probbase.shape == probbase5.shape
    assert out.probbase.dtype.type is np.str_
    assert all(out.causetext.iloc[:, 0] == causetext5.iloc[:, 0])
    assert all(out.causetext.iloc[:, 1] == causetext5.iloc[:, 1])


def test_arg_probbase_custom():
    tmp_pb = probbase5.copy()
    version = "new probbase"
    tmp_pb.iloc[0, 2] = version
    out = InSilicoVA(data=va_data, sci=tmp_pb)
    assert out.probbase[0, 2] == version


def test_arg_probbase_custom_exc():
    sci = pd.DataFrame(np.random.rand(10, 10))
    with pytest.raises(ArgumentException):
        InSilicoVA(data=va_data, sci=sci)


def test_arg_groupcode_true():
    out = InSilicoVA(data=va_data, groupcode=True)
    assert all(out.causetext.iloc[:, 1] == causetext5.iloc[:, 2])


def test_arg_groupcode_2012_true():
    out = InSilicoVA(data=va_data1, data_type="WHO2012", groupcode=True)
    assert all(out.causetext.fillna(".").iloc[:, 1] == causetext.fillna(".").iloc[:, 2])


def test_subpop_exception():
    with pytest.raises(ArgumentException):
        InSilicoVA(data=va_data, subpop=["not_a_column"])


def test_subpop_setup():
    subpop_out = InSilicoVA(va_data,
                            subpop=["i004a", "i019a", "i022a"])
    assert subpop_out.subpop.ndim == 1
    assert subpop_out.subpop.shape[0] == subpop_out.data.shape[0]
    assert isinstance(subpop_out.subpop, pd.Series)


def test_subpop_setup_1_level():
    tmp_data = va_data.copy()
    tmp_data.iloc[:, 3] = "one"
    with pytest.warns(UserWarning):
        subpop_out = InSilicoVA(va_data,
                                subpop=tmp_data.iloc[:, 3])
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
    out = InSilicoVA(tmp_data, data_type="WHO2012")
    assert out.data.shape == va_data1.shape
    assert all(out.data.columns == va_data1.columns)


def test_prep_data_error_2016():
    tmp_data = va_data.copy()
    tmp_data.drop(columns=va_data.columns[33], inplace=True)
    with pytest.raises(DataException):
        out = InSilicoVA(tmp_data, data_type="WHO2016")


def test_prep_data_2016():
    tmp_data = va_data.copy()
    tmp_data.columns = tmp_data.columns.str.upper()
    tmp_data["new_col"] = "not a VA column"
    out = InSilicoVA(tmp_data, data_type="WHO2016")
    assert out.data.shape == va_data.shape
    assert all(out.data.columns == va_data.columns)


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
    with pytest.warns(UserWarning):
        out = InSilicoVA(data=tmp_data, sci=pb)
    assert all(out.data.columns == out.va_labels)


def test_prep_data_cond_prob():
    pb = probbase5.copy()
    new_names = list(pb)[0:100]
    new_names.extend([x + "_new" for x in pb.columns[100:]])
    pb.columns = new_names
    out = InSilicoVA(data=va_data, cond_prob=pb)
    assert out.prob_orig.equals(pb)
    assert out.exclude_impossible_causes == "none"
    assert all(out.va_causes == pb.columns)


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
    df_age.loc[age_ind, age_col] = "N"
    ins_bad_age = InSilicoVA(df_age, data_type="WHO2012")
    bad_age_id = set(df_age["ID"])
    ins_bad_age._remove_bad(is_numeric=False, version5=False)

    n_bad_sex = 10
    df_sex = va_data1.copy()
    sex_ind = df_sex.index[0:n_bad_sex]
    df_sex.loc[sex_ind, ["male", "female"]] = "N"
    ins_bad_sex = InSilicoVA(df_sex, data_type="WHO2012")
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
    ins_bad_all = InSilicoVA(df_all, data_type="WHO2012")
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


def test_datacheck5_values():
    assert default.data.iloc[:, 1:].isin(["Y", "", "."]).all(axis=None)


def test_datacheck5_checked_data():
    assert default.data_checked.iloc[:, 1:].isin(["y", "n", "-"]).all(axis=None)


def test_datacheck5_log():
    assert len(default.vacheck_log["first_pass"]) > 0
    assert len(default.vacheck_log["second_pass"]) > 0

# class TestRemoveExt:
#
#     orig_shape = default.data.shape
#     n_external = 19
#
#     prob_orig = probbase5.iloc[1:, 20:81].to_numpy(copy=True)
#     subst = probbase5.iloc[1:, 5].to_numpy(copy=True)
#     csmf_orig = probbase5.iloc[0, 20:]
#     ext_causes = np.arange(49, 60)
#     ext_symptoms = np.arange(19, 38)
#     is_numeric = True
#     # subpop = va_data["i019a"].astype("string", copy=True)
#     # subpop_order_list = np.sort(subpop.unique())
#     # negate = (subst == "N")
#
#     ext_col = default.data.iloc[:, 1:].columns[ext_symptoms]
#     default.data.loc[:, ext_col] = 0
#     default.data.loc[default.data.index[0:n_external], ext_col[0]] = 1
#     default._remove_ext(csmf_orig, is_numeric, ext_causes, ext_symptoms)
#
#     def test_data_shape(self):
#         assert default.data.shape[0] == (self.orig_shape[0] - self.n_external)
#
#     def test_ext_id_shape(self):
#         assert default.ext_id.shape[0] == self.n_external
#
#     def test_ext_sub_shape(self):
#         assert default.ext_sub.shape[0] == self.n_external
#
#     def test_ext_csmf(self):
#         assert default.ext_csmf is not None
#
#     def test_negate_shape(self):
#         assert default.negate.shape[0] == (self.orig_shape[0] - self.ext_symptoms.shape[0])
