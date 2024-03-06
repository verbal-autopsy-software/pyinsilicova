# -*- coding: utf-8 -*-

"""
insilicova.indiv
-------------------

This module contains functions for getting and updating individual COD
probabilities from InSilicoVA model fits.
"""
from insilicova.structures import InSilico
from insilicova.exceptions import ArgumentException, DataException
# from insilicova.sampler import Sampler
from insilicova.utils import get_vadata
import warnings
import pandas as pd
import numpy as np
from typing import Dict, Union


def get_indiv(insilico_obj: InSilico,
              data: Union[None, pd.DataFrame],
              ci: float = 0.95,
              is_aggregate: bool = False,
              by: Union[None, list] = None,
              is_sample: bool = False) -> pd.DataFrame:
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
    :param is_sample: Indicator for returning the posterior samples of
    individual probabilities instead of posterior summaries.
    :type is_sample: bool
    :returns: Individual mean COD distribution array (mean); individual median
    COD distribution array (median); individual lower & upper bounds for each
    COD probability (lower & upper).
    :rtype: pandas DataFrame
    """
    id = insilico_obj.id
    # if grouping is provided
    if is_aggregate:
        if data is None:
            raise ArgumentException("Data not provided for grouping")
        va_data = pd.DataFrame(data).copy()
        # check by variable
        if by is None:
            warnings.warn("No groups specified, aggregate for all deaths")
            va_data["sub"] = "All"
            col_index = ["sub"]
        else:
            if not isinstance(by, list):
                raise ArgumentException("Please specify by variable as a list.")
            else:
                col_index = by.copy()
                if not set(by).issubset(set(list(data))):
                    raise ArgumentException(
                        "Contains unknown grouping variables!")
        # check id
        if data.columns[0] != "ID":
            warnings.warn("The first column of data is assumed to be ID")
            va_data.rename(columns={data.columns[0]: "ID"}, inplace=True)
        n_match = len(set(insilico_obj.id).intersection(va_data["ID"]))
        if n_match == 0:
            raise ArgumentException(
                "No data has the matching ID in the fitted insilico object.")
        else:
            print(f"{n_match} deaths matched from data to the fitted object.")
        # get grouping vector and names for all data
        col_index.insert(0, "ID")
        datagroup_all = va_data.loc[:, col_index].copy()
        datagroup_all.fillna("missing", inplace=True)
        datagroup_all = datagroup_all.astype("string")
        if len(col_index) > 2:
            datagroup_all["final_group"] = datagroup_all.loc[:, by].apply(
                lambda x: "_".join(x.to_list()),
                axis=1)
        else:
            datagroup_all["final_group"] = datagroup_all[col_index[1]].copy()
        # get the group name of subset in insilico fitted data
        index_tmp = datagroup_all["ID"].isin(insilico_obj.data_final.index)
        if sum(index_tmp) == 0:
            raise DataException("No matching ID found in the data.")
        datagroup = datagroup_all["final_group"][index_tmp].copy()
        datagroup = datagroup.dropna()
        allgroups = list(datagroup.sort_values().unique())
        datagroup = [allgroups.index(i) if i in allgroups else np.nan
                     for i in datagroup]
        datagroup_all["final_group"] = [allgroups.index(i) if i in allgroups else np.nan
                     for i in datagroup_all["final_group"]]
        n_groups = len(allgroups)

    # 3d array of csmf to be Nsub * Nitr * C
    if insilico_obj.subpop is None:
        csmf = np.zeros((1, ) + insilico_obj.csmf.shape)
        csmf[0] = insilico_obj.csmf.copy()
        subpop = np.zeros(len(insilico_obj.id), dtype=int)
    else:
        n_sub = len(insilico_obj.csmf)
        dims = (n_sub, ) + insilico_obj.csmf[0].shape
        csmf = np.zeros(dims)
        label_sub = list(insilico_obj.subpop.unique())
        label_sub.sort()
        for i in range(n_sub):
            csmf[i, :, :] = insilico_obj.csmf[i]
        subpop = [label_sub.index(i) for i in insilico_obj.subpop]
        subpop = np.array(subpop)
    if insilico_obj.external:
        csmf = np.delete(csmf, insilico_obj.external_causes, axis=2)
        csmf = np.apply_along_axis(lambda x: x/sum(x) if sum(x) != 0 else x,
                                   2,
                                   csmf)
    # get condprob to be Nitr * S * C array
    if insilico_obj.update_cond_prob is False:
        condprob = np.zeros((csmf.shape[1], ) + insilico_obj.probbase.shape)
        for i in range(csmf.shape[1]):
            condprob[i, :, :] = insilico_obj.probbase.copy()
    else:
        if insilico_obj.keep_probbase_level:
            condprob = np.zeros(
                (insilico_obj.conditional_probs.shape[0], ) +
                insilico_obj.probbase.shape)
            for i in range(condprob.shape[0]):
                # fix for interVA probbase
                obj_probbase = insilico_obj.probbase.copy()
                obj_probbase[insilico_obj.probbase == "B -"] = "B-"
                obj_probbase[insilico_obj.probbase == ""] = "N"
                tmp_idx = [list(insilico_obj.conditional_probs).index(j) for
                           j in insilico_obj.probbase.flatten("F")]
                temp = insilico_obj.conditional_probs.iloc[i, tmp_idx]
                temp = np.array(temp).reshape(insilico_obj.probbase.shape,
                                              order="F")
                condprob[i, :, :] = temp
        else:
            condprob = insilico_obj.conditional_probs.copy()
    n_sub, n_itr, c = csmf.shape
    s = condprob.shape[1]
    # condprob = condprob.flatten("F")
    # csmf = csmf.flatten("F")
    if is_sample:
        print("Calculating individual COD posterior draws\n")
        indivprob = IndivProbSample(
            data=insilico_obj.data_final.to_numpy(),
            impossible=insilico_obj.impossible_causes,
            csmf=csmf,
            subpop=subpop.tolist(),
            condprob=condprob,
            Nsub=n_sub,
            Nitr=n_itr,
            C=c,
            S=s)
        # if not customized probbase, use longer cause names for output
        if not insilico_obj.is_customized:
            if insilico_obj.data_type == "WHO2012":
                causetext = get_vadata("causetext", verbose=False)
            else:
                causetext = get_vadata("causetextV5", verbose=False)
            pb_colnames = list(insilico_obj.probbase_colnames)
            match_cause = [pb_colnames.index(i) if i in pb_colnames else np.nan
                           for i in causetext.iloc[:, 0]]
            match_cause = np.array(match_cause)
            index_cause = np.argsort(match_cause[~np.isnan(match_cause)])
            indivprob_cause_names = causetext.iloc[index_cause, 1]
        else:
            indivprob_cause_names = insilico_obj.probbase_colnames
        # add back all external cause death 41:51 in stanard VA
        if insilico_obj.external:
            external_causes = insilico_obj.external_causes
            c0 = indivprob.shape[1]
            ext_flag = insilico_obj.indiv_prob.iloc[:, external_causes].sum(axis=1)
            ext_probs = insilico_obj.indiv_prob.loc[ext_flag > 0, :]

            dims = (indivprob.shape[0] + int(sum(ext_flag)),
                    indivprob.shape[1] + len(external_causes),
                    indivprob.shape[2])
            indivprob_ext = np.zeros(dims)
            zero_filler = np.zeros((indivprob.shape[0], len(external_causes)))
            for t in range(n_itr): # t=0
                tmp = np.concatenate(
                    (indivprob[:, :external_causes[0], t],
                     zero_filler,
                     indivprob[:, external_causes[0]:c0, t]),
                    axis=1)
                indivprob_ext[:indivprob.shape[0], :, t] = tmp
                indivprob_ext[indivprob.shape[0]:indivprob_ext.shape[0], :, t] = ext_probs
            indivprob_ext = indivprob_ext.transpose(0, 2, 1)
            id_list = id.to_list()
            match_id = [id_list.index(i) for i in insilico_obj.data_final.index]
            id_indprob_ext = id.iloc[match_id].to_list()
            id_indprob_ext.extend(id[ext_flag.to_numpy() > 0].to_list())
        else:
            indivprob_ext = indivprob
        d0, d1, d2 = indivprob_ext.shape
        indivprob_ext = indivprob_ext.reshape((d0 * d1, d2))
        indivprob_ext = np.column_stack((np.repeat(id_indprob_ext, d1),
                                         np.tile(np.arange(1, (d1 + 1)), d0),
                                         indivprob_ext))
        # TODO: test if list(insilico_obj.indiv_prob) should be
        #       indivprob_cause_names (not using it)
        colnames = ["ID", "Iteration"] + list(insilico_obj.indiv_prob)
        indivprob_ext = pd.DataFrame(indivprob_ext, columns=colnames)
        return indivprob_ext
    elif (not is_aggregate) and (not is_sample):
        print("Calculating individual COD distributions...\n")
        indiv = IndivProb(
            data=insilico_obj.data_final.to_numpy(),
            impossible=insilico_obj.impossible_causes,
            csmf=csmf,
            subpop=subpop.tolist(),
            condprob=condprob,
            p0=(1 - ci)/2,
            p1=1 - (1 - ci)/2,
            Nsub=n_sub,
            Nitr=n_itr,
            C=c,
            S=s)
        # if not customized probbase, use longer cause names for output
        if not insilico_obj.is_customized:
            if insilico_obj.data_type == "WHO2012":
                causetext = get_vadata("causetext", verbose=False)
            else:
                causetext = get_vadata("causetextV5", verbose=False)
            pb_colnames = list(insilico_obj.probbase_colnames)
            match_cause = [pb_colnames.index(i) if i in pb_colnames else np.nan
                           for i in causetext.iloc[:, 0]]
            match_cause = np.array(match_cause)
            index_cause = np.argsort(match_cause[~np.isnan(match_cause)])
            indivprob_cause_names = causetext.iloc[index_cause, 1]
        else:
            indivprob_cause_names = insilico_obj.probbase_colnames
        k = int(indiv.shape[0] / 4)
        # add back all external cause of death 41-51 in standard VA
        if insilico_obj.external:
            external_causes = insilico_obj.external_causes
            c0 = indiv.shape[1]
            ext_flag = insilico_obj.indiv_prob.iloc[:, external_causes].sum(axis=1)
            ext_probs = insilico_obj.indiv_prob.loc[ext_flag > 0, :]
            zero_filler = np.zeros((indiv.shape[0], len(external_causes)))
            indiv = np.concatenate(
                    (indiv[:, :external_causes[0]],
                     zero_filler,
                     indiv[:, external_causes[0]:c0]),
                    axis=1)
            id_list = id.to_list()
            match_id = [id_list.index(i) for i in insilico_obj.data_final.index]
            id_out = id.iloc[match_id].to_list()
            id_out.extend(id[ext_flag.to_numpy() > 0].to_list())
        else:
            id_out = id
            ext_probs = None
        mean = indiv[:k, :]
        median = indiv[k:(2*k), :]
        lower = indiv[(2*k):(3*k), :]
        upper = indiv[(3*k):(4*k), :]
        if ext_probs is not None:
            mean = np.concatenate((mean, ext_probs), axis=0)
            median = np.concatenate((median, ext_probs), axis=0)
            lower = np.concatenate((lower, ext_probs), axis=0)
            upper = np.concatenate((upper, ext_probs), axis=0)
        mean = pd.DataFrame(mean, columns=list(insilico_obj.indiv_prob))
        mean.insert(loc=0, column="Statistic", value="mean")
        mean.insert(loc=0, column="ID", value=id_out)
        median = pd.DataFrame(median, columns=list(insilico_obj.indiv_prob))
        median.insert(loc=0, column="Statistic", value="median")
        median.insert(loc=0, column="ID", value=id_out)
        lower = pd.DataFrame(lower, columns=list(insilico_obj.indiv_prob))
        lower.insert(loc=0, column="Statistic", value="lower")
        lower.insert(loc=0, column="ID", value=id_out)
        upper = pd.DataFrame(upper, columns=list(insilico_obj.indiv_prob))
        upper.insert(loc=0, column="Statistic", value="upper")
        upper.insert(loc=0, column="ID", value=id_out)
        indiv_out = pd.concat((mean, median, lower, upper), axis=0)
        return indiv_out
    else:
        print("Aggregating individual COD distributions...\n")
        indiv = AggIndivProb(
            data=insilico_obj.data_final.to_numpy(),
            impossible=insilico_obj.impossible_causes,
            csmf=csmf,
            subpop=subpop.tolist(),
            condprob=condprob,
            group=datagroup,
            Ngroup=n_groups,
            p0=(1 - ci)/2,
            p1=1 - (1 - ci)/2,
            Nsub=n_sub,
            Nitr=n_itr,
            C=c,
            S=s)
        if insilico_obj.data_type == "WHO2012":
            causetext = get_vadata("causetext", verbose=False)
        else:
            causetext = get_vadata("causetextV5", verbose=False)
        weight = indiv[:, (indiv.shape[1] - 1)]
        indiv = indiv[:, :(indiv.shape[1] - 1)]
        pb_colnames = list(insilico_obj.probbase_colnames)
        match_cause = [pb_colnames.index(i) if i in pb_colnames else np.nan
                       for i in causetext.iloc[:, 0]]
        match_cause = np.array(match_cause)
        index_cause = np.argsort(match_cause[~np.isnan(match_cause)])
        indivprob_cause_names = causetext.iloc[index_cause, 1]
        k = int(indiv.shape[0] / 4)

        # add back all external cause of death 41:51 in standard VA
        if insilico_obj.external:
            external_causes = insilico_obj.external_causes
            c0 = indiv.shape[1]
            ext_flag = insilico_obj.indiv_prob.iloc[:, external_causes].sum(axis=1)
            ext_probs = insilico_obj.indiv_prob.loc[ext_flag > 0, :]
            ext_probs = ext_probs.iloc[:, external_causes]
            ext_probs["ID"] = ext_probs.index.copy()
            ext_probs = ext_probs.merge(datagroup_all[["ID", "final_group"]])
            prob_columns = set(list(ext_probs)).difference(["ID", "final_group"])
            prob_columns = list(prob_columns)
            ext_probs_agg = np.zeros((len(allgroups), len(prob_columns)))
            ext_weight = np.zeros(len(allgroups))
            for i in range(len(allgroups)):
                this_group = ext_probs["final_group"] == i
                if sum(this_group) > 0:
                    ext_probs_agg[i, :] = ext_probs.loc[this_group, prob_columns].mean(axis=0)
                    ext_weight[i] = sum(this_group)
                    ext_probs_agg[i, :] *= ext_weight[i]
            ext_probs_agg_4fold = np.concatenate(
                (ext_probs_agg, ext_probs_agg, ext_probs_agg, ext_probs_agg))
            # multiply group size
            indiv = indiv * weight[:, None]
            indiv = np.concatenate(
                (indiv[:, :external_causes[0]],
                 ext_probs_agg_4fold,
                 indiv[:, external_causes[0]:c0]), axis=1)
            # divide by full size
            indiv = indiv / (weight + np.tile(ext_weight, 4))[:, None]
        else:
            id_out = id
            ext_probs = None
        indiv = pd.DataFrame(indiv, columns=list(insilico_obj.indiv_prob))
        if by is None:
            indiv = indiv.transpose()
            indiv.columns = ["Mean", "Median", "Lower", "Upper"]
        else:
            indiv.insert(loc=0,
                         column="Statistic",
                         value=np.repeat(["mean", "median", "lower", "upper"], 2))
            indiv.insert(loc=0,
                         column="Group",
                         value=np.tile(allgroups, 4))
        return indiv

def update_indiv(insilico_obj: InSilico,
                 data: Union[None, pd.DataFrame],
                 ci: float = 0.95,
                 is_aggregate: bool = False,
                 by: Union[None, list] = None,
                 is_sample: bool = False) -> None:
    """
    Update individual COD probabilities from InSilicoVA Model Fits

    This function updates individual probabilities for each death and provide
    posterior credible intervals for each estimate.

    :param insilico_obj: Fitted InSilicoVA object (InSilicoVA.get_results())
    :type insilico_obj: InSilico
    :param data: Data for the fitted InSilico object. The first column of the
    data should be the ID that matches the InSilico fitted model.
    :type data: pandas DataFrame
    :param ci:  Width of credible interval for posterior estimates.
    :type ci: float
    :param is_aggregate: Indicator for constructing aggregated distribution
    rather than individual distributions.
    :type is_aggregate: bool
    :param by: List of column names to group by.
    :type by: list (or None)
    :param is_sample: Indicator for returning the posterior samples of
    individual probabilities instead of posterior summaries.
    :type is_sample: bool
    """
    indiv = get_indiv(insilico_obj=insilico_obj,
                      data=data,
                      ci=ci,
                      is_aggregate=is_aggregate,
                      by=by,
                      is_sample=is_sample)
    median = indiv.loc[indiv["Statistic"] == "median"]
    median = median.drop("Statistic", axis=1)
    lower = indiv.loc[indiv["Statistic"] == "lower"]
    lower = lower.drop("Statistic", axis=1)
    upper = indiv.loc[indiv["Statistic"] == "upper"]
    upper = upper.drop("Statistic", axis=1)
    insilico_obj.indiv_prob_median = median
    insilico_obj.indiv_prob_lower = lower
    insilico_obj.indiv_prob_upper = upper
    insilico_obj.indiv_ci = ci
    # TODO: need to account for different options (e.g., is_sample=True,
    #       is_aggregate=True and by=None)
    return insilico_obj

def IndivProb(data: np.ndarray,
              impossible: np.ndarray,
              csmf: np.ndarray,
              subpop: list,
              condprob: np.ndarray,
              p0: float,
              p1: float,
              Nsub: int,
              Nitr: int,
              C: int,
              S: int) -> np.ndarray:

    N = data.shape[0]

    zero_matrix = np.ones((N, C))
    if impossible.shape[1] == 2:
        for i in range(N):
            for k in range(impossible.shape[0]):
                if data[i, impossible[k, 1]] == 1:
                    zero_matrix[i, impossible[k, 0]] = 0
    allp = np.zeros((N, C, Nitr))
    for i in range(N):
        sub = subpop[i]
        for t in range(Nitr):
            allp[i, :, t] = csmf[sub, t, :] * zero_matrix[i, :]
            factor = np.ones((S, C))
            index_eq1 = data[i, :] == 1
            index_eq0 = data[i, :] == 0
            factor[index_eq1, :] = condprob[t, index_eq1, :].copy()
            factor[index_eq0, :] = 1 - condprob[t, index_eq0, :].copy()
            index_eq1_or_eq0 = np.isin(data[i, :], [0, 1])
            allp[i, :, t] *= np.prod(factor[index_eq1_or_eq0], axis=0)
            sum_allp = sum(allp[i, :, t])
            if sum_allp > 0:
                allp[i, :, t] = allp[i, :, t] / sum_allp
    out = np.zeros((N * 4, C))
    out[:N, :] = allp.mean(axis=2)
    out[N:(2*N), :] = np.percentile(a=allp, q=0.5, axis=2)
    out[(2 * N):(3 * N), :] = np.percentile(a=allp, q=p0, axis=2)
    out[(3 * N):(5 * N), :] = np.percentile(a=allp, q=p1, axis=2)
    return out

def AggIndivProb(data: np.ndarray,
                 impossible: np.ndarray,
                 csmf: np.ndarray,
                 subpop: list,
                 condprob: np.ndarray,
                 group: list,
                 Ngroup: int,
                 p0: float,
                 p1: float,
                 Nsub: int,
                 Nitr: int,
                 C: int,
                 S: int) -> np.ndarray:

    N = data.shape[0]
    zero_matrix = np.ones((N, C))
    if impossible.shape[1] == 2:
        for i in range(N):
            for k in range(impossible.shape[0]):
                if data[i, impossible[k, 1]] == 1:
                    zero_matrix[i, impossible[k, 0]] = 0
    allp = np.zeros((Ngroup, C, Nitr))
    size = np.zeros(Ngroup, dtype=int)
    for i in range(N):
        # get subpop index for death i
        sub = subpop[i]
        g = group[i]
        size[g] += 1
        for t in range(Nitr):
            tmp = csmf[sub, t, :] * zero_matrix[i, :]
            factor = np.ones((S, C))
            index_eq1 = data[i, :] == 1
            index_eq0 = data[i, :] == 0
            factor[index_eq1, :] = condprob[t, index_eq1, :].copy()
            factor[index_eq0, :] = 1 - condprob[t, index_eq0, :].copy()
            index_eq1_or_eq0 = np.isin(data[i, :], [0, 1])
            tmp *= np.prod(factor[index_eq1_or_eq0], axis=0)
            sum_tmp = sum(tmp)
            if sum_tmp == 0:
                allp[g, :, t] = tmp
            else:
                allp[g, :, t] = tmp / sum_tmp

    out = np.zeros((Ngroup * 4, C + 1))
    for i in range(Ngroup):
        for c in range(C):
            out[i, c] = allp[i, c, :].mean() / size[i]
            out[i + Ngroup, c] = np.percentile(allp[i, c, :], 0.5) / size[
                i]
            out[i + Ngroup * 2, c] = np.percentile(allp[i, c, :], p0) / \
                                     size[i]
            out[i + Ngroup * 3, c] = np.percentile(allp[i, c, :], p1) / \
                                     size[i]
        out[i][C] = size[i]
        out[i + Ngroup, C] = size[i]
        out[i + Ngroup * 2, C] = size[i]
        out[i + Ngroup * 3, C] = size[i]

    return out

def IndivProbSample(data: np.ndarray,
                    impossible: np.ndarray,
                    csmf: np.ndarray,
                    subpop: list,
                    condprob: np.ndarray,
                    Nsub: int,
                    Nitr: int,
                    C: int,
                    S: int) -> np.ndarray:

    N = data.shape[0]
    zero_matrix = np.ones((N, C))
    if impossible.shape[1] == 2:
        for i in range(N):
            for k in range(impossible.shape[0]):
                # if data[i, impossible[k, 1] - 1] == 1:
                #     zero_matrix[i, impossible[k, 0] - 1] = 0
                if data[i, impossible[k, 1]] == 1:
                    zero_matrix[i, impossible[k, 0]] = 0
    allp = np.zeros((N, C, Nitr))
    for i in range(N):
        sub = subpop[i]
        for t in range(Nitr):
            allp[i, :, t] = csmf[sub, t, :] * zero_matrix[i, :]
            factor = np.ones((S, C))
            index_eq1 = data[i, :] == 1
            index_eq0 = data[i, :] == 0
            factor[index_eq1, :] = condprob[t, index_eq1, :].copy()
            factor[index_eq0, :] = 1 - condprob[t, index_eq0, :].copy()
            index_eq1_or_eq0 = np.isin(data[i, :], [0, 1])
            allp[i, :, t] *= np.prod(factor[index_eq1_or_eq0], axis=0)
            sum_allp = sum(allp[i, :, t])
            if sum_allp > 0:
                allp[i, :, t] = allp[i, :, t]/sum_allp
    return allp
