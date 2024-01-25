# -*- coding: utf-8 -*-

"""
insilicova.diag
-------------------

This module contains functions that check if the CSMF chains have converged.
"""
from .exceptions import SamplerException, ArgumentException
from .structures import InSilico
from copy import deepcopy
from math import gamma, pi
import numpy as np
from typing import Union, Dict, List
from statsmodels.tsa.stattools import acovf
from statsmodels.api import OLS
from scipy.special import kv
from pandas import DataFrame


def csmf_diag(csmf: Union[InSilico, List, np.ndarray, DataFrame],
              conv_csmf: float = 0.02,
              test: str = "heidel",
              verbose: bool = True,
              autoburnin: bool = False,
              which_sub: Union[None, int] = None) -> Union[int, List]:
    """
    Convergence test for fitted InSilicoVA model.

    Produce convergence test for CSMFs from fitted
    :class:`InSilicoVA <insilicova.insilicova.InSilicoVA>` objects.

    The tests are performed using :func:`diag._heidel_diag` and
    :func:`diag._gelman_diag` functions (ported from code in the
    excellent R package code https://CRAN.R-project.org/package=coda). The
    function takes either one or a list of output from
    :class:`InSilicoVA <insilicova.insilicova.InSilicoVA}`, or only the
    iteration by CSMF matrix.  Usually in practice, many causes with very tiny
    CSMF are hard to converge based on standard tests, thus it is suggested to
    check convergence for only causes with mean CSMF over certain threshold by
    setting proper ``conv_csmf``.

    Note for Gelman and Rubin's test, all chains should have the same length.
    If the chains are sampled with automatically length determination, they
    might not be comparable by this test.

    :param csmf: It could be either fitted "insilico" object, a list of
    fitted "insilico" object from different chains of the same length
    but different starting values, i.e., different seed. Or it could be the
    matrix of CSMF obtained by insilico, or the list of matrices of
    CSMF. All CSMF could contain more than one subpopulation, but should be in
    the same format and order. And notice if the raw CSMF is used instead of
    the "insilico" object, external causes might need to be removed
    manually by user is external_sep is True when fitting the model.
    :param conv_csmf: The minimum mean CSMF to be checked. Default to be 0.02,
    which means any causes with mean CSMF lower than 0.02 will not be tested.
    :type conv_csmf: float
    :param test: Type of test. Currently supporting Gelman and Rubin's test
    (test = "gelman") for multi-chain test, and Heidelberger and Welch's
    test (test = "heidel") for single-chain test.
    :type test: str
    :param verbose: Logical indicator to return the test detail instead of one
    logical outcome for Heidelberger and Welch's test. Default to be TRUE.
    :type test: bool
    :param autoburnin: Logical indicator of whether to omit the first half of the
    chain as burn in. Default to be FALSE since insilico return only the
    iterations after burnin by default.
    :type autoburnin: bool
    :param which_sub: the name of the sub-population to test when there are
    multiple in the fitted object.
    :type which_sub: numpy.ndarray
    """
    if test not in ("gelman", "heidel"):
        raise ArgumentException(
            "Test needs to be either 'gelman' or 'heidel'.")
    if test == "gelman":
        raise ArgumentException(
            "'gelman' test has not been implemented yet (use 'heidel').")

    check_csmf = []
    csmf_column_names = []
    # input is a single csmf matrix
    if isinstance(csmf, np.ndarray):
        check_csmf.append(csmf.copy())
    if isinstance(csmf, DataFrame):
        tmp_csmf = csmf.to_numpy().copy()
        check_csmf.append(tmp_csmf)
    # input is list of csmf matrices
    if isinstance(csmf, list):
        if isinstance(csmf[0], np.ndarray):
            check_csmf = deepcopy(csmf)
    # input is a single InSilico instance (w/ & w/o subpops)
    if isinstance(csmf, InSilico):
        # with subpops
        if isinstance(csmf.csmf, list):
            for i in range(len(csmf.csmf)):
                tmp_names = csmf.csmf[i].columns.copy()
                tmp_names = tmp_names.to_numpy()
                tmp_csmf = csmf.csmf[i].copy()
                tmp_csmf = tmp_csmf.to_numpy()
                if csmf.external:
                    tmp_csmf = np.delete(tmp_csmf,
                                         csmf.external_causes, axis=1)
                    tmp_names = np.delete(tmp_names, csmf.external_causes)
                check_csmf.append(tmp_csmf)
                csmf_column_names.append(tmp_names)
        # without subpops
        else:
            tmp_names = csmf.csmf.columns.copy()
            tmp_names = tmp_names.to_numpy()
            tmp_csmf = csmf.csmf.copy()
            tmp_csmf = tmp_csmf.to_numpy()
            if csmf.external:
                tmp_csmf = np.delete(tmp_csmf,
                                     csmf.external_causes, axis=1)
                tmp_names = np.delete(tmp_names, csmf.external_causes)
            check_csmf.append(tmp_csmf)
            csmf_column_names.append(tmp_names)
    # input is list of InSilico instances (w/ & w/o subpops)
    if isinstance(csmf, list):
        if isinstance(csmf[0], InSilico):
            # w/ subpops
            if isinstance(csmf[0].csmf, list):
                for i in range(len(csmf)):
                    tmp_names = csmf[i].csmf[which_sub].columns.copy()
                    tmp_names = tmp_names.to_numpy()
                    tmp_csmf = csmf[i].csmf[which_sub].copy()
                    tmp_csmf = tmp_csmf.to_numpy()
                    if csmf[i].external:
                        tmp_csmf = np.delete(tmp_csmf,
                                             csmf[i].external_causes, axis=1)
                        tmp_names = np.delete(tmp_names,
                                              csmf[i].external_causes)
                    check_csmf.append(tmp_csmf)
                    csmf_column_names.append(tmp_names)
            # w/o subpops
            else:
                for i in range(len(csmf)):
                    tmp_names = csmf[i].csmf.columns.copy()
                    tmp_names = tmp_names.to_numpy()
                    tmp_csmf = csmf[i].csmf.copy()
                    tmp_csmf = tmp_csmf.to_numpy()
                    if csmf[i].external:
                        tmp_csmf = np.delete(tmp_csmf,
                                             csmf[i].external_causes, axis=1)
                        tmp_names = np.delete(tmp_names,
                                              csmf[i].external_causes)
                    check_csmf.append(tmp_csmf)
                    csmf_column_names.append(tmp_names)
    # check conv_csmf value
    if conv_csmf >= 1 or conv_csmf <= 0:
        raise ArgumentException(
            "Invalid threshold (i.e., argument for conv_csmf parameter.")
    if test == "gelman" and not isinstance(csmf, list):
        raise ArgumentException(
            "Need more than one chain to perform Gelman and Rubin's test.")
    # Heidel test
    if test == "heidel":
        testout = []
        conv = 1
        for i in range(len(check_csmf)):
            cod_names = None
            if len(csmf_column_names) > 0:
                cod_names = csmf_column_names[i]
            out, out_names = _heidel_single(check_csmf[i],
                                            conv_csmf,
                                            cod_names)
            # conv = conv * testout["stest"].prod() * testout["htest"].prod()
            conv = conv * out[:, 0].prod() * out[:, 3].prod()
            df_out = DataFrame(out, index=out_names,
                               columns=["Stationarity test", "start iteration",
                                        "p-value", "Halfwidth test", "Mean",
                                        "Halfwidth"])
            # df_out["Stationarity test"].replace(
            #     {1.0: "passed", 0.0: "failed"}, inplace=True)
            # df_out["Halfwidth test"].replace(
            #     {1.0: "passed", 0.0: "failed"}, inplace=True)
            df_out["Stationarity test"] = df_out["Stationarity test"].replace(
                {1.0: "passed", 0.0: "failed"})
            df_out["Halfwidth test"] = df_out["Halfwidth test"].replace(
                {1.0: "passed", 0.0: "failed"})
            testout.append(df_out)
    # Gelman test
    if test == "gelman":
        raise ArgumentException(
            "'gelman' test has not been implemented yet (use 'heidel').")
    if np.isnan(conv):
        conv = 0
    if not verbose and test == "heidel":
        return conv
    else:
        if len(testout) == 1:
            return testout[0]
        else:
            return testout


def _heidel_single(one: np.ndarray,
                   conv_csmf: float,
                   cod_names: Union[None, np.ndarray]) -> tuple:
    # sort chain by CSMF mean values
    mean = one.mean(axis=0)
    ordered_one = one[:, mean.argsort()[::-1]].copy()
    ordered_names = None
    if cod_names is not None:
        ordered_names = cod_names[mean.argsort()[::-1]]
    mean = ordered_one.mean(axis=0)
    ordered_one = ordered_one[:, mean > conv_csmf]
    if cod_names is not None:
        ordered_names = ordered_names[mean > conv_csmf]
    test = _heidel_diag(ordered_one)
    # TODO: this needs to be updated for dealing with DataFrame
    #       see below (_heidel_diag)
    return test, ordered_names


def _gelman_single(input_list, conv_csmf):
    pass


def _gelman_diag(x: np.ndarray,
                 eps: float = 0.1,
                 pvalue: float = 0.05):
    pass


def _heidel_diag(x: np.ndarray,
                 eps: float = 0.1,
                 pvalue: float = 0.05) -> np.ndarray:
    """
    Ported from R package coda::heidel.diag
    https://CRAN.R-project.org/package=coda

    and implemented to fit the specific needs of csmf_diag (i.e., making
    assumptions about default arguments and data structures).

    The description from coda:

    'heidel.diag' is a run length control diagnostic based on a
    criterion of relative accuracy for the estimate of the mean.  The
    default setting corresponds to a relative accuracy of two
    significant digits.

    'heidel.diag' also implements a convergence diagnostic, and
    removes up to half the chain in order to ensure that the means are
    estimated from a chain that has converged.

    :param x: MCMC chain
    :type x: numpy.ndarray
    :param eps: Target value for ratio of halfwidth to sample mean
    :type eps: float
    :param pvalue: significance level to use
    :type pvalue: float
    """
    n_iter, n_var = x.shape
    dt = {"names": ["stest", "start", "pvalue", "htest", "mean", "halfwidth"],
          "formats": [np.float64]*6}
    HW_mat = np.zeros((n_var, 6))
    for j in range(n_var):
        start_vec = np.arange(start=1,
                              stop=int((n_iter + 1)/2),
                              step=max(int(n_iter/10), 1))
        Y = x[:, j].copy()
        Y_start = (int(np.ceil(x.shape[0] / 2)) - 1)
        Y_ar = Y[Y_start:]
        S0 = _spectrum0_ar(Y_ar)["spec"]
        converged = False
        for i in range(len(start_vec)):  # i=4
            Y_start = start_vec[i] - 1
            Y_window = Y[Y_start:, ]
            n = Y_window.shape[0]
            ybar = Y_window.mean()
            B = Y_window.cumsum() - ybar * np.arange(1, (n+1))
            Bsq = B*B / (n*S0)
            I = sum(Bsq)/n
            pcramer_test = _pcramer(I) < (1 - pvalue)
            converged = not np.isnan(I) and pcramer_test
            if converged:
                break
        S0ci = _spectrum0_ar(Y_window)["spec"]
        halfwidth = 1.96 * np.sqrt(S0ci/n)
        passed_hw = not np.isnan(halfwidth) and (abs(halfwidth/ybar) <= eps)
        if not converged or np.isnan(I) or np.isnan(halfwidth):
            nstart = np.nan
            passed_hw = np.nan
            halfwidth = np.nan
            ybar = np.nan
        else:
            nstart = Y_start
        HW_mat[j, :] = [converged, nstart, 1 - _pcramer(I),
                        passed_hw, ybar, halfwidth]

    # TODO: this needs to return a DataFrame with cause names for each row
    return HW_mat


def _spectrum0_ar(x: np.ndarray) -> Dict:  # assume x.shape = (n,)

    n = x.shape[0]
    z = np.concatenate(
        (np.ones(n), np.arange(1, n + 1)))
    z = z.reshape((n, 2), order="F")
    lm_out = OLS(x, z).fit()
    sd_resid = np.std(lm_out.resid)
    if sd_resid == 0:
        v0 = 0
        order = 0
    else:
        ar_out = _ar_yw(x, aic=True)
        v0 = ar_out["var_pred"] / ((1 - sum(ar_out["ar"]))**2)
        order = ar_out["order"]

    return {"spec": v0, "order": order}


def _pcramer(q: float,
             eps: float = 1.0e-5) -> float:
    """
    Distribution function of the Cramer-von Mises statistic
    Reproducing R function coda::pcramer
    https://CRAN.R-project.org/package=coda
    """
    log_eps = np.log(eps)
    y = np.zeros(4)
    for k in range(4):
        z = gamma(k + 0.5) * np.sqrt(4 * k + 1) / (gamma(k + 1) * pi**(3/2) * np.sqrt(q))
        u = ((4 * k + 1)**2) / (16 * q)
        if u <= -log_eps:
            y[k] = z * np.exp(-u) * kv(1 / 4, u)
    return sum(y)


def _ar_yw(x: np.ndarray,
           aic: bool = True,
           order_max: Union[None, int] = None,
           demean: bool = True) -> Dict:
    """Autoregressive model using Yule-Walker method."""
    chain = x.copy()
    if demean:
        chain = chain - chain.mean()
    n_obs = chain.shape[0]  # assuming no missing values
    if order_max is None:
        order_max = min(n_obs - 1,
                        np.floor(10 * np.log10(n_obs)))
        order_max = int(order_max)
    chain_acf = acovf(chain, nlag=order_max)
    if chain_acf[0] == 0:
        raise SamplerException("Zero-variance chain.")
    f = np.zeros((order_max, order_max))
    var = np.zeros(order_max)
    a = np.ones(order_max)
    z = _eureka(order_max, chain_acf, chain_acf, f, var, a)
    var_pred = np.insert(z["vars"][0:], obj=0, values=chain_acf[0])
    order_vec = np.arange((order_max + 1))
    chain_aic = (n_obs * np.log(var_pred)) + (2 * order_vec) + (2 * demean)
    if aic:
        order = np.nanargmin(chain_aic)
    else:
        order = order_max
    ar = z["coefs"][(order - 1), np.arange(order)]
    var_pred = var_pred[order]
    var_pred = var_pred * n_obs / (n_obs - (order + 1))

    return {"order": order, "ar": ar, "var_pred": var_pred}


def _eureka(lr: int,
            r: np.ndarray,
            g: np.ndarray,
            f: np.ndarray,
            var: np.ndarray,
            a: np.ndarray) -> Dict:
    """
    Reproducing the Fortran code (in R's stats package) called eureka:
    https://svn.r-project.org/R/trunk/src/library/stats/src/eureka.f
    """

    v = r[0]
    d = r[1]
    f[0, 0] = g[1] / v
    q = f[0, 0] * r[1]
    var[0] = (1 - f[0, 0] * f[0, 0]) * r[0]
    if lr == 1:
        return {"coefs": f, "vars": var}
    for l in range(2, (lr + 1)):
        a[(l - 1)] = -d / v
        if l > 2:
            l1 = int((l - 2) / 2)
            l2 = int(l1 + 1)
            for j in range(2, (l2 + 1)):
                hold = a[(j - 1)]
                k = l - j + 1
                a[(j - 1)] = a[(j - 1)] + a[(l - 1)] * a[(k - 1)]
                a[(k - 1)] = a[(k - 1)] + a[(l - 1)] * hold
            if (2 * l1) != (l - 2):
                a[l2] = a[l2] * (1.0 + a[(l - 1)])
        v = v + a[(l - 1)] * d
        f[(l - 1), (l - 1)] = (g[l] - q) / v
        for j in range(1, l):
            lm1 = l - 1
            lm2 = l - 2
            jm1 = j - 1
            f[lm1, jm1] = f[lm2, jm1] + f[lm1, lm1] * a[(l - j)]
        # estimate the innovations variance
        var[(l - 1)] = var[(l - 2)] * (
                    1 - f[(l - 1), (l - 1)] * f[(l - 1), (l - 1)])
        if l < lr:
            d = 0.0
            q = 0.0
            for i in range(1, (l + 1)):
                k = l - i + 2
                d = d + a[(i - 1)] * r[(k - 1)]
                q = q + f[(l - 1), (i - 1)] * r[(k - 1)]

    return {"coefs": f, "vars": var}
