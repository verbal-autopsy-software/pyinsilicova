# -*- coding: utf-8 -*-

"""
insilicova.sampler
------------------------

This module contains the wrapper class for the InsilicoVA algorithm.
"""

# from pandas import (DataFrame, Series)
import numpy as np
from scipy.stats import beta
from random import Random
from math import exp, log


class Sampler:
    """InsilicoVA algorithm for assigning cause of death.
    
    :param N: number of deaths
    :type N: int
    :param S: number of symptoms
    :type S: int
    :param C: number of causes
    :type C: int
    :param N_sub: number of sub populations
    :type N_sub: int
    :param N_level: number of levels in the probbase table
    :type N_level: int
    :param probbase: S by C matrix of probbase
    :type probbase: numpy ndarray
    :param probbase_order: S by C matrix of probbase in order index, value from 1 (I) to 15 (N)
    :type probbase_order: numpy ndarray
    :param level_values: vector of length N_level, prior of the level values
    :type level_values: list
    :param probbase_level: dictionary that tells which symptoms are at which level in each cause
        - top layer:    key = 0 to C-1
        - second layer: key = 1 to N_level
        - value set:    which symptoms (0 to S-1)
    :type probbase_level: dictionary
    :param count_m: S by C matrix, counts of "Y"
    :type count_m: numpy ndarray
    :param count_m_all: S by C matrix, counts of "Y" and "" (yes and no, not missing)
    :type count_m_all: numpy ndarray
    :param count_c: length C vector, counts of each cause
    :type count_c: list
    
    # stuff related to physician coding
    :param C_phy: number of physician coded categories
    :type C_phy: int
    :param broader: length C conversion vector
    :type broader: list
    :param catmap: dictionary that tells which causes go into each category
        - key: category, value: set of causes
    :type catmap: dictionary
    """

    # Initialization of InSilico Sampler
    def __init__(self, N: int, S: int, C: int, N_sub: int, N_level: int,
                 subpop: list, probbase: np.ndarray, probbase_order: np.ndarray, 
                 level_values: list, pool: int):
        self.N = N
        self.S = S
        self.C = C
        self.N_sub = N_sub
        self.N_level = N_level
        # self.probbase = probbase.to_numpy() if probbase is pd.DataFrame
        self.probbase = probbase
        # self.probbase = probbase_order.to_numpy() if probbase_order is pd.DataFrame
        self.probbase_order = probbase_order
        # self.count_m = DataFrame(index=range(S),columns=range(C))
        self.count_m = np.zeros((S, C), dtype=int)
        self.count_m_all = np.zeros((S, C), dtype=int)
        self.count_c = [None] * C
        self.level_values = level_values
        # self.probbase_level = dict of K-int + V-dict, of K-int + V-list of int
        self.probbase_level = {}
        self.levelize(pool)

    # Initialization with physician coding, have to initialize first
    def initate_phys_coding(self, C_phy: int, broader: list):
        # added variables
        self.C_phy = C_phy
        self.broader = broader
        self.initiate_cat_map()

    # Initialize Category Mapping
    def initiate_cat_map(self):
        # self.catmap = dict with K-int + V-set of int
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
                    if c not in self.probbase_level:
                        # self.probbase_level[c] = dict of K-int and V-list of int
                        self.probbase_level[c] = {}
                    # initialize if this level under this cause has not been initialized
                    if level not in self.probbase_level[c]:
                        # self.probbase_level[c][level] = list of int
                        self.probbase_level[c][level] = []
                    # save the cause-level-symptom combination
                    self.probbase_level[c][level].append(s)
        # if pool = 2, count level by symptom
        elif pool == 2:
            # loop over all s and c combinations
            for s in range(self.S):
                for c in range(self.C):
                    # get level of this s-c combination
                    level = int(self.probbase_order[s, c])
                    # initialize if this cause has not been intialized in probbase_level yet
                    if s not in self.probbase_level:
                        # self.probbase_level[s] = dict with K-int and V-list of int
                        self.probbase_level[s] = {}
                    # initialize if this level under this cause has not been initialized
                    if level not in self.probbase_level[s]:
                        # self.probbase_level[s][level] = list of int
                        self.probbase_level[s][level] = []
                    # save the cause-level-symptom combination
                    self.probbase_level[s][level].append(c)

    # function to count the current configurations for TruncBeta
    def countCurrent(self, indic: np.ndarray, ynew: list):
        # counts of yes
        self.count_m = np.zeros((self.S, self.C), dtype=int)
        # counts of yes and no
        self.c = np.zeros((self.S, self.C), dtype=int)
        self.count_c = [0] * self.C
        
        # loop over all individuals
        for n in range(self.N):
            # cause of this death
            c_current = int(ynew[n])
            # add to total counts
            self.count_c[c_current] += 1
            # loop over all symptoms
            for s in range(self.S):
                # add to counts of this symptom's appearance
                if indic[n, s] == 1:
                    self.count_m[s, c_current] += 1
                    self.count_m_all[s, c_current] += 1
                elif indic[n, s] == 0:
                    self.count_m_all[s, c_current] += 1

    # function to calculate p.nb with sub-population
    # some key dimensions: indic (N x S), csmf_sub (Nsub x C), subpop (N)
    # output: nb (N by C)
    def pnb(self, contains_missing: bool, indic: np.ndarray, csmf_sub: np.ndarray, subpop: list, zero_matrix: np.ndarray) -> np.ndarray: 
        # initialize p.nb matrix to p.hat
        nb = np.zeros((self.N, self.C))
        # pnb_{nc} <- csmf_{subpop_n, c}
        for n in range(self.N):
            for c in range(self.C):
                nb[n, c] = csmf_sub[subpop[n],c] * zero_matrix[n,c]
        # calculate posterior
        for n in range(self.N):
            # find which symptoms are not missing for this death
            # nomissing = list of int
            nomissing = []
            for s in range(len(indic[n])):
                if indic[n,s] >= 0: nomissing.append(s)
            # loop over cause-symptoms combination to calculate naive bayes prob
            for c in range(self.C):
                for s in nomissing:
                    if indic[n,s] > 0:
                        nb[n,c] *= self.probbase[s,c]
                    else:
                        nb[n,c] *= (1 - self.probbase[s,c])
            # normalization
            nb[n] = np.linalg.norm(nb[n])
        return nb

    # function to calculate p.nb with sub-population and physician coding
    # some key dimensions: indic (N x S),csmf_sub (Nsub x C), subpop (N), assignment(N)
    # output: nb (N by C)
    def pnb2(self, contains_missing: bool, indic: np.ndarray, csmf_sub: np.ndarray, subpop: list, g_new: list, zero_matrix: np.ndarray) -> np.ndarray:
        # initialize p.nb matrix to p.hat
        nb = np.zeros((self.N, self.C))
        # pnb_{nc} <- csmf_{subpop_n, c}

        # sample a cause group first, using the same as sample new Y
        # default to zero, update those with physician codes, normalize at the end
        for n in range(self.N):
            for c in self.catmap[g_new[n]]:
                nb[n,c] = csmf_sub[subpop[n],c] * zero_matrix[n,c]
        
        # calculate posterior
        for n in range(self.N):
            # find which symptoms are not missing for this death
            # nomissing = list of int
            nomissing = []
            for s in range(len(indic[n])):
                if indic[n,s] >= 0: nomissing.append(s)
            # loop over cause-symptoms combination to calculate naive bayes prob
            for c in self.catmap[g_new[n]]:
                for s in nomissing:
                    if indic[n,s] > 0:
                        nb[n,c] *= self.probbase[s,c]
                    else:
                        nb[n,c] *= (1 - self.probbase[s,c])
            # normalization
            nb[n] = np.linalg.norm(nb[n])
        return nb
        
    # function to sample a multinomial, note the sampled y starts from 0!
    def sampleY(self, pnb: np.ndarray, rand: Random) -> list:
        y = [None] * len(pnb)
        # loop over every death
        for n in range(len(pnb)):
            # naive implementation of categorical distribution
            u = rand.random()
            cum = 0.0
            for c in range(len(pnb[n])):
                cum += pnb[n,c]
                if u < cum:
                    y[n] = c
                    break
        return y

    # function to update theta
    # input (dimensions): jumprange, mu (C) sigma, theta (C), Y(C), N, jump.prop(NOT USING), rngN, rngU
    # new input: zero_vector: define which causes are set to zero 
    def thetaBlockUpdate(self, jumprange: float, mu: list, sigma2: float, theta: list, Y: list, jump_prop: bool, rngN: np.random.Generator, rand: Random, zero_vector: list):
        # initalize jump range, use the same value for all causes
        jump = [None] * self.C
        for c in range(self.C): jump[c] = jumprange
        
        # if jump_prop, not using
        
        # initialize new theta
        theta_new = [None] * self.C
        # fix first theta (not removed)
        fix = 0
        for i in range(len(zero_vector)):
            if zero_vector[i] != 0:
                fix = i
                break
        theta_new[fix] = 1.0
        # calculate sum(exp(theta)), the normalizing constant
        expsum = exp(1.0)
        expsum_new = exp(1.0)
        
        # sample new theta proposal
        for c in range(fix+1, self.C):
            theta_new[c] = rngN.normal(theta[c], jump[c]) * zero_vector[c]
            expsum += exp(theta[c])
            expsum_new += exp(theta_new[c])

        # calculate likelihood
        logTrans = 0.0
        # l = Y * (theta_new - theta - log(expsum_new/expsum)) - ((theta_new - mu)^2 - (theta - mu)^2 ) / (2*sigma^2)
        for c in range(self.C):
            if zero_vector[c] > 0:
                diffquad = (theta_new[c] - mu[c]) * (theta_new[c] - mu[c]) - (theta[c] - mu[c]) * (theta[c] - mu[c])
                logTrans += Y[c] * (theta_new[c] - theta[c] - log(expsum_new / expsum)) - 1 / (2*sigma2) * diffquad
        
        # accept or reject
        u = log(rand.random())
        if logTrans >= 0:
            # print("+")
            pass
        else:
            # print("-")
            return(theta)
        return(theta_new)

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
        
        a = 0.0
        b = 0.0
        # create a new transpose probbase matrix
        new_probbase = np.zeros((self.C, self.S))
        for s in range(self.S):
            for c in range(self.C):
                new_probbase[c,s] = self.probbase[s,c]
        # loop over causes c
        for s in range(self.S):
            # find which level-symptom combinations under this cause
            levels_under_s = self.probbase_level[s]
            # get the conditional probabilities of each symptoms given cause c
            prob_under_s = self.probbase[s]
            # create new instance of this vector
            new_prob_under_s = [None] * self.C
            # find the list of all levels present under cause c
            exist_levels_under_s = []
            for l in range(1, self.N_level+1):
                if l in levels_under_s: exist_levels_under_s.append(l)
            # loop over level l, in ascending order
            for index in range(len(exist_levels_under_s)):
                l_current = exist_levels_under_s[index]
                # loop over symptoms s in the level l
                for c in levels_under_s[l_current]:
                    # find the counts of this symptom
                    count = self.count_m[s,c]
                    count_all = self.count_m_all[s,c]
                    lower = 0.0
                    upper = 1.0
                    # if the highest level
                    if index == 0:
                        # find which level is next
                        l_next = exist_levels_under_s[index+1]
                        # find the max of those symptoms
                        lower = max(int(np.max(prob_under_s)), levels_under_s[l_next])
                        # make sure not lower than lower bound
                        lower = max(lower, trunc_min)
                        upper = trunc_max
                        # if the lowest level
                    elif index == (len(exist_levels_under_s) - 1):
                        lower = trunc_min
                        l_prev = exist_levels_under_s[index-1]
                        upper = min(int(min(new_prob_under_s)), levels_under_s[l_prev])
                        upper = min(upper, trunc_max)
                        # if in the middle
                    else:
                        l_next = exist_levels_under_s[index + 1]
                        lower = max(int(np.max(prob_under_s)), levels_under_s[l_next])
                        lower = max(lower, trunc_min)
                        l_prev = exist_levels_under_s[index-1]
                        upper = min(int(min(new_prob_under_s)), levels_under_s[l_prev])
                        upper = min(upper, trunc_max)
                    # if range is invalid, use higher case
                    if lower >= upper: 
                        new_prob_under_s[c] = upper
                    else:
                        # find the beta distribution parameters, note level starting from 1
                        a = prior_a[l_current-1] + count
                        b = prior_b + count_all - a
                        # sample inverse beta distribution
                        value = lower
                        # ymin and ymax currently are ndarrays but should be floats/doubles
                        ymin = np.sum(beta.cdf(lower, a, b))
                        ymax = np.sum(beta.cdf(upper, a, b))
                        # handling boundary case
                        if abs(ymax - ymin) < 1e-8:
                            new_prob_under_s[c] = (upper + lower) / 2.0
                        else:
                            value = beta.ppf(rand.random() * (ymax - ymin) + ymin)
                            if value == 0:
                                print("lower %.6f, upper 5.6f, sampled 0\n", ymin, ymax)
                            new_prob_under_s[c] = value
            # update this column of probbase
            for c in range(self.C):
                self.probbase[s,c] = new_prob_under_s[c]
                    
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
        
        a = 0.0
        b = 0.0
        # create a new probbase matrix
        new_probbase = np.zeros((self.S, self.C))
        for s in range(self.S):
            for c in range(self.C):
                new_probbase[s,c] = self.probbase[s,c]
        # loop over causes c
        for c in range(self.C):
            # find which level-symptom combinations under this cause
            # levels_under_c = dict with K-int and V-list of int
            levels_under_c = dict(self.probbase_level[c])
            # get the conditional probabilities of each symptoms given cause c
            # prob_under_c = list of int
            prob_under_c = list(self.probbase[:, c])
            # create new instance of this vector
            # new_prob_under_c = list of floats
            new_prob_under_c = [None] * self.S
            # find the list of all levels present under cause c
            # exist_levels_under_c = list of int
            exist_levels_under_c = []
            for l in range(1, self.N_level+1):
                if l in levels_under_c:
                    exist_levels_under_c.append(l)
            # loop over level l, in ascending order
            for index in range(len(exist_levels_under_c)):
                l_current = int(exist_levels_under_c[index])
                # loop over symptoms s in the level l
                for s in levels_under_c[l_current]:
                    # find the counts of this symptom
                    count = int(self.count_m[s,c])
                    count_all = int(self.count_m_all[s,c])
                    lower = 0.0
                    upper = 1.0
                    # if the highest level
                    if index == 0:
                        # find which level is next
                        l_next = int(exist_levels_under_c[index+1])
                        # find the max of those symptoms
                        lower = max(max(prob_under_c), max(levels_under_c[l_next]))
                        # make sure not lower than lower bound
                        lower = max(lower, trunc_min)
                        upper = trunc_max
                        # if the lowest level
                    elif index == (len(exist_levels_under_c) - 1):
                        lower = trunc_min
                        l_prev = int(exist_levels_under_c[index-1])
                        upper = min(min(new_prob_under_c), min(levels_under_c[l_prev]))
                        upper = min(upper, trunc_max)
                        # if in the middle
                    else:
                        l_next = int(exist_levels_under_c[index+1])
                        lower = max(max(prob_under_c), max(levels_under_c[l_next]))
                        lower = max(lower, trunc_min)
                        l_prev = exist_levels_under_c[index-1]
                        upper = min(min(new_prob_under_c), min(levels_under_c[l_prev]))
                        upper = min(upper, trunc_max)
                    # if range is invalid, use higher case
                    if lower >= upper:
                        new_prob_under_c[s] = upper
                    else:
                        # find the beta distribution parameters, note level starting from 1
                        a = prior_a[l_current-1] + count
                        b = prior_b + count_all - a
                        # sample inverse beta distribution
                        value = lower
                        # ymin and ymax currently are ndarrays but should be floats/doubles
                        ymin = np.sum(beta.cdf(lower, a, b))
                        ymax = np.sum(beta.cdf(upper, a, b))
                        # handling boundary case
                        if abs(ymax - ymin) < 1e-8:
                            new_prob_under_c[s] = (upper + lower) / 2.0
                        else:
                            value = beta.ppf(rand.random() * (ymax - ymin) + ymin)
                            if value == 0:
                                print("lower %.6f, upper 5.6f, sampled 0\n", ymin, ymax)
                            new_prob_under_c[s] = value
            # update this column of probbase
            for s in range(self.S): self.probbase[s,c] = new_prob_under_c[s]

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
        
        a = 0.0
        b = 0.0
        # initialize a new probbase copy
        new_probbase = np.zeros((self.S, self.C))
        for s in range(self.S):
            for c in range(self.C):
                new_probbase[s,c] = self.probbase[s,c]
        # assume all levels exist in data, if not, need to modify the R codes before calling
        new_level_values = [None] * self.N_level
        for l in range(1, self.N_level+1):
            count = 0
            count_all = 0
            # count appearances
            for c in range(self.C):
                if l in self.probbase_level[c]:
                    for s in self.probbase_level[c,l]:
                        count += self.count_m[s,c]
                        count_all += self.count_m_all[s,c]
            lower = 0.0
            upper = 1.0
            # note l starts from 1, level_values have index starting from 0
            if l == 1:
                # lower bound is the max of this next level
                lower = max(self.level_values[l], trunc_min)
                upper = trunc_max
            elif l == self.N_level:
                lower = trunc_min
                # upper bound is the min of previous level
                upper = min(new_level_values[l-2], trunc_max)
                # upper = min(self.level_values[l-2], upper)
            else:
                lower = max(self.level_values[l], trunc_min)
                upper = min(new_level_values[l-2], trunc_max)
                # upper = min(self.level_values[l-2], upper)
            if lower >= upper: new_level_values[l-1] = upper
            else:
                a = prior_a[l-1] + count
                b = prior_b + count_all - a
                # sample inverse beta distribution
                value = lower
                # ymin and ymax currently are ndarrays but should be floats/doubles
                ymin = np.sum(beta.cdf(lower, a, b))
                ymax = np.sum(beta.cdf(upper, a, b))
                # handling boundary case
                if abs(ymax - ymin) < 1e-8:
                    new_level_values[l-1] = (upper + lower) / 2.0
                else:
                    value = beta.ppf(rand.random() * (ymax - ymin) + ymin)
                    if value == 0:
                        print("lower %.6f, upper 5.6f, sampled 0\n", ymin, ymax)
                    new_level_values[l-1] = value
        self.level_values = new_level_values
        for s in range(self.S):
            for c in range(self.C):
                self.probbase[s,c] = self.level_values[int(self.probbase_order[s,c]-1)]
        return
                

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
    def fit(self, dimensions: list, probbase: np.ndarray, probbase_order: np.ndarray, level_values: list, prior_a: list, prior_b: float, jumprange: float, trunc_min: float, trunc_max: float, indic: np.ndarray, subpop: list, contains_missing: int, pool: int, seed: int, N_gibbs: int, burn: int, thin: int, mu: list, sigma2: float, this_is_Unix: bool, useProbbase: bool, isAdded: bool, mu_continue: np.ndarray, sigma2_continue: list, theta_continue: np.ndarray, C_phy: int, broader: list, assignment: np.ndarray, impossible: np.ndarray):
        pass

    # @param DontAsk: M by 8 matrix (java index + 1, 0 if not exist)
    # @param AskIf: M by 2 matrix (java index + 1, 0 if not exist)
    # @param data: N by M matrix
    # @param warningfile: warning file
    # @return
    def Datacheck(self, DontAsk: np.ndarray, AskIf: np.ndarray, zero_to_missing_list: list, data: np.ndarray, ids: list, symps: list, warningfile: str):
        pass

    # @param DontAsk: M by 8 matrix (java index + 1, 0 if not exist)
    # @param AskIf: M by 2 matrix (java index + 1, 0 if not exist)
    # @param data: N by M matrix
    # @return
    def Datacheck(self, DontAsk: np.ndarray, AskIf: np.ndarray, zero_to_missing_list: list, data: np.ndarray):
        ids = []
        symps = []
        warningfile = "warnings.txt"
        return self.Datacheck(DontAsk, AskIf, zero_to_missing_list, data, ids, symps, warningfile)

    def IndivProb(self, data: np.ndarray,
                  impossible: np.ndarray,
                  csmf0: list,
                  subpop: list,
                  condprob0: list,
                  p0: float,
                  p1: float,
                  Nsub: int,
                  Nitr: int,
                  C: int,
                  S: int) -> np.ndarray:

        N = data.shape[0]
        csmf = np.zeros((Nsub, Nitr, C))
        condprob = np.zeros((Nitr, S, C))
        for i in range(Nsub):
            for j in range(Nitr):
                for k in range(C):
                    csmf[i, j, k] = csmf0[k * Nsub * Nitr + j * Nsub + i]

        for i in range(Nitr):
            for j in range(S):
                for k in range(C):
                    condprob[i, j, k] = condprob0[k * Nitr * S + j * Nitr + i]

        zero_matrix = np.ones((N, C))
        if impossible.shape[1] == 2:
            for i in range(N):
                for k in range(impossible.shape[0]):
                    if data[i, impossible[k, 1] - 1] == 1:
                        zero_matrix[i, impossible[k, 0] - 1] = 0
        allp = np.zeros((N, C, Nitr))
        for i in range(N):
            sub = subpop[i] - 1
            for t in range(Nitr):
                sum_allp = 0
                for c in range(C):
                    allp[i, c, t] = csmf[sub, t, c] * zero_matrix[i, c]
                    for s in range(S):
                        # if condprob[t, s, c] == 0:
                        #     print(".")
                        if data[i, s] == 1:
                            allp[i, c, t] *= condprob[t, s, c]
                        elif data[i, s] == 0:
                            allp[i, c, t] *= (1 - condprob[t, s, c])
                    sum_allp += allp[i, c, t]
                for c in range(C):
                    if sum_allp == 0:
                        allp[i, c, t] = allp[i, c, t]
                    else:
                        allp[i, c, t] = allp[i, c, t] / sum_allp
        out = np.zeros((N*4, C))
        for i in range(N):
            for c in range(C):
                out[i, c] = allp[i, c, :].mean()
                out[i+N, c] = np.percentile(allp[i, c, :], 0.5)
                out[i + N*2, c] = np.percentile(allp[i, c, :], p0)
                out[i + N*3, c] = np.percentile(allp[i, c, :], p1)

        return out

    def AggIndivProb(self,
                     data: np.ndarray,
                     impossible: np.ndarray,
                     csmf0: list,
                     subpop: list,
                     condprob0: list,
                     group: list,
                     Ngroup: list,
                     p0: float,
                     p1: float,
                     Nsub: int,
                     Nitr: int,
                     C: int,
                     S: int) -> np.ndarray:

        N = data.shape[0]
        csmf = np.zeros((Nsub, Nitr, C))
        condprob = np.zeros((Nitr, S, C))
        for i in range(Nsub):
            for j in range(Nitr):
                for k in range(C):
                    csmf[i, j, k] = csmf0[k * Nsub * Nitr + j * Nsub + i]

        for i in range(Nitr):
            for j in range(S):
                for k in range(C):
                    condprob[i, j, k] = condprob0[k * Nitr * S + j * Nitr + i]

        zero_matrix = np.ones((N, C))
        if impossible.shape[1] == 2:
            for i in range(N):
                for k in range(impossible.shape[0]):
                    if data[i, impossible[k, 1] - 1] == 1:
                        zero_matrix[i, impossible[k, 0] - 1] = 0
        allp = np.zeros((Ngroup, C, Nitr))
        size = np.zeros(Ngroup, dtype=int)
        for i in range(N):
            # get subpop index for death i
            sub = subpop[i] - 1
            g = group[i] - 1
            size[g] += 1

            for t in range(Nitr):
                tmp = float(C)
                sum_allp = 0
                for c in range(C):
                    tmp[c] = csmf[sub, t, c] * zero_matrix[i, c]
                    for s in range(S):
                        if data[i, s] == 1:
                            tmp[c] *= condprob[t, s, c]
                        elif data[i, s] == 0:
                            tmp[c] *= (1 - condprob[t, s, c])
                    sum_allp += tmp[c]
                for c in range(C):
                    if sum_allp == 0:
                        allp[g, c, t] = tmp[c]
                    else:
                        allp[g, c, t] = tmp[c] / sum_allp

        out = np.zeros((Ngroup*4, C+1))
        for i in range(Ngroup):
            for c in range(C):
                out[i, c] = allp[i, c, :].mean() / size[i]
                out[i + Ngroup, c] = np.percentile(allp[i, c, :], 0.5) / size[i]
                out[i + Ngroup*2, c] = np.percentile(allp[i, c, :], p0) / size[i]
                out[i + Ngroup*3, c] = np.percentile(allp[i, c, :], p1) / size[i]
            out[i][C] = size[i]
            out[i + Ngroup, C] = size[i]
            out[i + Ngroup*2, C] = size[i]
            out[i + Ngroup*3, C] = size[i]

        return out

    def IndivProbSample(self,
                        data: np.ndarray,
                        impossible: np.ndarray,
                        csmf0: list,
                        subpop: list,
                        condprob0: list,
                        Nsub: int,
                        Nitr: int,
                        C: int,
                        S: int) -> np.ndarray:

        N = data.shape[0]
        csmf = np.zeros((Nsub, Nitr, C))
        condprob = np.zeros((Nitr, S, C))
        for i in range(Nsub):
            for j in range(Nitr):
                for k in range(C):
                    csmf[i, j, k] = csmf0[k * Nsub * Nitr + j * Nsub + i]

        for i in range(Nitr):
            for j in range(S):
                for k in range(C):
                    condprob[i, j, k] = condprob0[k * Nitr * S + j * Nitr + i]

        zero_matrix = np.ones((N, C))
        if impossible.shape[1] == 2:
            for i in range(N):
                for k in range(impossible.shape[0]):
                    if data[i, impossible[k, 1] - 1] == 1:
                        zero_matrix[i, impossible[k, 0] - 1] = 0
        allp = np.zeros((N, C, Nitr))
        for i in range(N):
            sub = subpop[i] - 1
            for t in range(Nitr):
                sum_allp = 0
                for c in range(C):
                    allp[i, c, t] = csmf[sub, t, c] * zero_matrix[i, c]
                    for s in range(S):
                        # if condprob[t, s, c] == 0:
                        #     print(".")
                        if data[i, s] == 1:
                            allp[i, c, t] *= condprob[t, s, c]
                        elif data[i, s] == 0:
                            allp[i, c, t] *= (1 - condprob[t, s, c])
                    sum_allp += allp[i, c, t]
                for c in range(C):
                    if sum_allp == 0:
                        allp[i, c, t] = allp[i, c, t]
                    else:
                        allp[i, c, t] = allp[i, c, t] / sum_allp

        return allp
