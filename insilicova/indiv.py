# -*- coding: utf-8 -*-

"""
insilicova.indiv
-------------------

This module contains functions for getting and updating individual COD
probabilities from InSilicoVA model fits.
"""
import numpy as np

from insilicova.structures import InSilico
from insilicova.exceptions import ArgumentException, DataException
from insilicova.sampler import Sampler
from insilicova.utils import get_vadata
import warnings
from pandas import DataFrame
from numpy import apply_along_axis, array, delete, nan, ones, zeros
from typing import Dict, Union

def get_indiv(insilico_obj: InSilico,
              data: Union[None, DataFrame],
              ci: float = 0.95,
              is_aggregate: bool = False,
              by: Union[None, list] = None,
              is_sample: bool = False) -> Dict:
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
    :rtype: dictionary of numpy arrays
    """
    id = insilico_obj.id
    # if grouping is provided
    if is_aggregate:
        if data is None:
            raise ArgumentException("Data not provided for grouping")
        va_data = DataFrame(data).copy()
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
        index_tmp = va_data.iloc[:, 0].isin(insilico_obj.data_final.index)
        if sum(index_tmp) == 0:
            raise DataException("No matching ID found in the data.")
        datagroup = datagroup_all["final_group"][index_tmp].copy()
        allgroups = list(datagroup.sort_values().unique())
        datagroup = [allgroups.index(i) if i in allgroups else nan
                     for i in datagroup_all["final_group"]]
        n_groups = len(allgroups)

    # 3d array of csmf to be Nsub * Nitr * C
    if insilico_obj.subpop is None:
        csmf = zeros((1, ) + insilico_obj.csmf.shape)
        csmf[0] = insilico_obj.csmf.copy()
        subpop = ones(len(insilico_obj.id))
    else:
        n_sub = len(insilico_obj.csmf)
        dims = (n_sub, ) + insilico_obj.csmf[0].shape
        csmf = zeros(dims)
        label_sub = list(insilico_obj.subpop.unique())
        label_sub.sort()
        for i in range(n_sub):
            csmf[i, :, :] = insilico_obj.csmf[i]
        subpop = [label_sub.index(i) for i in insilico_obj.subpop]
        subpop = array(subpop)
    if insilico_obj.external:
        csmf = delete(csmf, insilico_obj.external_causes, axis=2)
        csmf = apply_along_axis(lambda x: x/sum(x) if sum(x) != 0 else x,
                                2,
                                csmf)
    # get condprob to be Nitr * S * C array
    if insilico_obj.update_cond_prob is False:
        condprob = zeros((csmf.shape[1], ) + insilico_obj.probbase.shape)
        for i in range(csmf.shape[1]):
            condprob[i, :, :] = insilico_obj.probbase.copy()
    else:
        if insilico_obj.keep_probbase_level:
            condprob = zeros(
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
                temp = array(temp).reshape(insilico_obj.probbase.shape,
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
        indivprob = Sampler.IndivProbSample(
            data=insilico_obj.data_final.to_numpy(),
            impossible=insilico_obj.impossible_causes,
            csmf=csmf,
            subpop=subpop,
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
                                         np.tile(np.arange(1, (d1 + 1), d0)),
                                         indivprob_ext))
        colnames = ["ID", "Iteration"] + list(insilico_obj.indiv_prob)
        indivprob_ext = DataFrame(indivprob_ext, columns=colnames)
        return indivprob_ext
    elif (not is_aggregate) and  (not is_sample):
        print("Calculating individual COD distributions...\n")
    else:
        print("Aggregating individual COD distributions...\n")

def update_indiv(insilico_obj: InSilico,
                 ci: float = 0.95):
    """
    Update individual COD probabilities from InSilicoVA Model Fits

    This function updates individual probabilities for each death and provide
    posterior credible intervals for each estimate.

    :param insilico_obj: Fitted InSilicoVA object (InSilicoVA.get_results())
    :type insilico_obj: InSilico
    :param ci:  Width of credible interval for posterior estimates.
    :type ci: float
    :returns: Updated InSilico object
    :rtype: InSilico
    """
    pass
