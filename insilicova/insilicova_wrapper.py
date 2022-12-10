# -*- coding: utf-8 -*-

"""
insilicova.insilicova_wrapper
------------------------

This module contains the wrapper class for the InsilicoVA algorithm.
"""

# from pandas import (DataFrame, Series)
from numpy import ndarray, zeros
from numpy.linalg import norm
from random import Random


class InsilicoVA:
    """InsilicoVA algorithm for assigning cause of death.
    
    :param varname: what is it
    :type varname: what is type
    """

    # Initialization of InSilico Sampler
    def __init__(self, N: int, S: int, C: int, N_sub: int, N_level: int,
                 subpop: list, probbase: ndarray, probbase_order: ndarray, 
                 level_values: list, pool: int):
        self.N = N
        self.S = S
        self.C = C
        self.N_sub = N_sub
        self.N_level = N_level
        self.probbase = probbase
        self.probbase_order = probbase_order
        # self.count_m = DataFrame(index=range(S),columns=range(C))
        self.count_m = zeros((S, C), dtype=int)
        self.count_m_all = zeros((S, C), dtype=int)
        self.count_c = [0] * C
        self.level_values = level_values
        # dict of K-int + V-dict, of K-int + V-list of int
        self.probbase_level = {}
        self.levelize(pool)

    # Initialization with physician coding, have to initialize first
    def initate_phys_coding(self, C_phy: int, broader: list):
        self.C_phy = C_phy
        self.broader = broader
        self.initiate_cat_map()

    # Initialize Category Mapping
    def initiate_cat_map(self):
        # dict with K-int + V-set of int
        # unknown causes
        self.catmap = {0: set()}
        for j in range(1, len(self.broader)):
            self.catmap[0].add(j)
        # actual causes
        for i in range(1, self.C_phy):
            self.catmap[i] = set()
            for j in range(1, len(self.broader)):
                if int(self.broader[j]) == i:
                    self.catmap[i].add(j)

    # function to summarize the current levels of probbase
    def levelize(self, pool: int):
        # if pool = 0, doesn't matter which
        # if pool = 1, count level by cause
        if pool == 0 or pool == 1:
            # loop over all s and c combinations
            for s in range(self.S):
                for c in range(self.C):
                    # get level of this s-c combination
                    level = int(self.probbase_order[s, c])
                    # initialize if this cause has not been intialized in probbase_level yet
                    if self.probbase_level[c] is None:
                        self.probbase_level[c] = {}
                    # initialize if this level under this cause has not been initialized
                    if self.probbase_level[c][level] is None:
                        self.probbase_level[c][level] = []
                    # save the cause-level-symptom combination
                    self.probbase_level[c][level].append(s)
        # if pool = 2, count level by symptom
        elif pool == 2:
            # loop over all s and c combinations
            for s in range(self.S):
                for c in range(self.C):
                    # get level of this s-c combination
                    level = self.probbase_order[s, c]
                    # initialize if this cause has not been intialized in probbase_level yet
                    if self.probbase_level[s] is None:
                        self.probbase_level[s] = {}
                    # initialize if this level under this cause has not been initialized
                    if self.probbase_level[s][level] is None:
                        self.probbase_level[s][level] = []
                    # save the cause-level-symptom combination
                    self.probbase_level[s][level].append(c)

    # function to count the current configurations for TruncBeta
    def countCurrent(self, indic: ndarray, ynew: list):
        # counts of yes
        self.count_m = zeros((self.S, self.C), dtype=int)
        # counts of yes and no
        self.c = zeros((self.S, self.C), dtype=int)
        self.count_c = zeros((self.C,), dtype=int)
        
        # loop over all individuals
        for n in range(self.N):
            # cause of this death
            c_current = ynew[n]
            # add to total counts
            self.count_c[c_current] += 1
            # loop over all symptoms
            for s in range(self.S):
                if indic[n, s] == 1:
                    self.count_m[s, c_current] += 1
                    self.count_m_all[s, c_current] += 1
                elif indic[n, c_current] == 0:
                    self.count_m_all[s, c_current] += 1

    # function to calculate p.nb with sub-population
    # some key dimensions: indic (N x S),csmf_sub (Nsub x C), subpop (N)
    # output: nb (N by C)
    def pnb(self, contains_missing: bool, indic: ndarray, csmf_sub: ndarray, subpop: list, zero_matrix: ndarray):
        # initailize p.nb matrix to p.hat
        nb = zeros((self.N, self.C))
        # pnb_{nc} <- csmf_{subpop_n, c}
        for n in range(self.N):
            for c in range(self.C):
                nb[n, c] = csmf_sub[subpop[n]][c] * zero_matrix[n][c]
        # calculate posterior
        for n in range(self.N):
            # find which symptoms are not missing for this death
            nomissing = []
            for s in range(len(indic[n])):
                if indic[n][c] >= 0: nomissing.append(s)
            # loop over cause-symptoms combination to calculate naive bayes prob
            for c in range(self.C):
                for s in nomissing:
                    if indic[n][s] > 0:
                        nb[n][c] *= self.probbase[s][c]
                    else:
                        nb[n][c] *= (1 - self.probbase[s][c])
            # normalization
            nb[n] = norm(nb[n])
        return nb

    # function to calculate p.nb with sub-population and physician coding
    # some key dimensions: indic (N x S),csmf_sub (Nsub x C), subpop (N), assignment(N)
    # output: nb (N by C)
    def pnb(self, contains_missing: bool, indic: ndarray, csmf_sub: ndarray, subpop: list, g_new: list, zero_matrix: ndarray):
        # initialize p.nb matrix to p.hat
        nb = zeros((self.N, self.C))
        # pnb_{nc} <- csmf_{subpop_n, c}

        # sample a cause group first, using the same as sample new Y
        # default to zero, update those with physician codes, normalize at the end
        for n in range(self.N):
            for c in self.catmap[g_new[n]]:
                nb[n][c] = csmf_sub[subpop[n]][c] * zero_matrix[n][c]
        
        # calculate posterior
        for n in range(self.N):
            nomissing = []
            for s in range(len(indic[n])):
                if indic[n][s] >= 0: nomissing.append(s)
            # loop over cause-symptoms combination to calculate naive bayes prob
            for c in self.catmap[g_new[n]]:
                for s in nomissing:
                    if indic[n][s] > 0:
                        nb[n][c] *= self.probbase[s][c]
                    else:
                        nb[n][c] *= (1 - self.probbase[s][c])
            # normalization
            nb[n] = norm(nb[n])
        return nb
        
    # function to sample a multinomial, note the sampled y starts from 0!
    def sampleY(self, pnb: ndarray, rand: Random):
        y = zeros((len(pnb),), dtype=int)
        # loop over every death
        for n in range(len(pnb)):
            u = rand.random()
            cum = 0
            for c in range(len(pnb[n])):
                cum += pnb[n][c]
                if u < cum:
                    y[n] = c
                    break
        return y

    # function to update theta
    # input (dimensions): jumprange, mu (C) sigma, theta (C), Y(C), N, jump.prop(NOT USING), rngN, rngU
    # new input: zero_vector: define which causes are set to zero 
    def thetaBlockUpdate(self, jumprange: float, mu: list, sigma2: float, theta: list, Y: list, jump_prop: bool, rngN, rand: Random, zero_vector: list):
        pass

    # Update Sep 30, 2015: truncated beta for new SCI
    # function to perform Truncated beta distribution for the whole conditional probability matrix.
    # update cond.prob within the class.
    # Note: between symptom comparison not performed here. Truncation only performed within same symptom
    #
    # input:
    # rand: randome number generator
    # prior_a: prior of alpha vector
    # prior_b: prior of beta
    # trunc_max, trunc_Min: max/min of truncation
    # used fields:
    # probbase_order, level_values, probbase_level, count_m, count_m_all, count_c
    #   key: 1 to C, second key: 1:N_level, value: which symptoms
    #   HashMap<Integer, HashMap<Integer, ArrayList<Integer>>> probbase_level;
    def truncBeta2(self, rand: Random, prior_a: list, prior_b: float, trunc_min: float, trunc_max: float):
        pass

    # function to perform Truncated beta distribution for the whole conditional probability matrix.
    # update cond.prob within the class.
    # Note: between causes comparison not performed here. Truncation only performed within same cause
    # 
    # input:
    # rand: randome number generator
    # prior_a: prior of alpha vector
    # prior_b: prior of beta
    # trunc_max, trunc_Min: max/min of truncation
    # used fields:
    # probbase_order, level_values, probbase_level, count_m, count_m_all, count_c
    #   key: 1 to C, second key: 1:N_level, value: which symptoms
    #   HashMap<Integer, HashMap<Integer, ArrayList<Integer>>> probbase_level;
    def truncBeta(self, rand: Random, prior_a: list, prior_b: float, trunc_min: float, trunc_max: float):
        pass

    # function to perform Truncated beta distribution for the probbase table.
    # update cond.prob within the class. Return table vector.
    #
    # input:
    # rand: randome number generator
    # prior_a: prior of alpha vector
    # prior_b: prior of beta
    # last: current probbase
    # trunc_max, trunc_Min: max/min of truncation
    # used fields:
    # probbase_order, level_values, probbase_level, count_m, count_m_all, count_c
    #   key: 1 to C, second key: 1:N_level, value: which symptoms
    #   HashMap<Integer, HashMap<Integer, ArrayList<Integer>>> probbase_level;
    def truncBeta_pool(self, rand: Random, prior_a: list, prior_b: float, trunc_min: float, trunc_max: float):
        pass

    # Main function to perform insilico sampling
    #
    # key parameters:
    # indic: N by S matrix
    # subpop: length N vector of sub-population assignment
    #         Note: should start from 0. All 0 if no sub-population exist.
    # contains_missing: if there are missing
    # pool: if estimating the probbase table only
    #       update Sep 30, 2015: pool = 0, only table; pool = 1, by cause; pool = 2, compare only within each symptom
    # seed, N_gibbs, thin: integers
    # mu: vector initialized in R
    # sigma2: value initialized in R
    def fit(self, dimensions: list, probbase: ndarray, probbase_order: ndarray, level_values: list, prior_a: list, prior_b: float, jumprange: float, trunc_min: float, trunc_max: float, indic: ndarray, subpop: list, contains_missing: int, pool: int, seed: int, N_gibbs: int, burn: int, thin: int, mu: list, sigma2: float, this_is_Unix: bool, useProbbase: bool, isAdded: bool, mu_continue: ndarray, sigma2_continue: list, theta_continue: ndarray, C_phy: int, broader: list, assignment: ndarray, impossible: ndarray):
        pass

    # @param DontAsk: M by 8 matrix (java index + 1, 0 if not exist)
    # @param AskIf: M by 2 matrix (java index + 1, 0 if not exist)
    # @param data: N by M matrix
    # @param warningfile: warning file
    # @return
    def Datacheck(self, DontAsk: ndarray, AskIf: ndarray, zero_to_missing_list: list, data: ndarray, ids: list, symps: list, warningfile: str):
        pass

    # @param DontAsk: M by 8 matrix (java index + 1, 0 if not exist)
    # @param AskIf: M by 2 matrix (java index + 1, 0 if not exist)
    # @param data: N by M matrix
    # @return
    def Datacheck(self, DontAsk: ndarray, AskIf: ndarray, zero_to_missing_list: list, data: ndarray):
        ids = []
        symps = []
        warningfile = "warnings.txt"
        return self.Datacheck(DontAsk, AskIf, zero_to_missing_list, data, ids, symps, warningfile)

    def IndivProb(self, data: ndarray, impossible: ndarray, csmf0: list, subpop: list, condprob0: list, p0: float, p1: float, Nsub: int, Nitr: int, C: int, S: int):
        pass

    def AggIndivProb(self, data: ndarray, impossible: ndarray, csmf0: list, subpop: list, condprob0: list, group: list, Ngroup: list, p0: float, p1: float, Nsub: int, Nitr: int, C: int, S: int):
        pass

    def IndivProbSample(self, data: ndarray, impossible: ndarray, csmf0: list, subpop: list, condprob0: list, p0: float, p1: float, Nsub: int, Nitr: int, C: int, S: int):
        pass
