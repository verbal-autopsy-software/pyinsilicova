# -*- coding: utf-8 -*-

"""
insilicova.utils
-------------------

This module provides functions for loading example data.
"""

from typing import Union
from pprint import pprint
from pkgutil import get_data
from io import BytesIO
from pandas import read_csv, DataFrame
from .exceptions import InSilicoVAException


DATA_DICT = {"causetext": ("Translation of cause-of-death abbreviation codes"
                           " into their full names"),
             "causetextV5": ("Translation of cause-of-death abbreviation codes"
                             " into their full names (used by InterVA5)"),
             "condprob": ("Conditional probability matrix used by InterVA4-2."
                          "  There are 60 causes and 245 symptoms"),
             "condprobnum": ("Condition probability matrix used by InterVA4-2"
                             " (60 causes and 245 symptoms)."),
             "_probbase": ("Conditional probabilities of symptoms given the "
                          "cause of death with the data check rules. "
                          "The values are from InterVA4-2."),
             "probbaseV3": ("Conditional probabilities of symptoms given the "
                           "cause of death.  The values are from "
                           "InterVA-4.03."),
             "probbaseV5": ("Conditional probabilities of symptoms given the "
                           "cause of death.  The values are from "
                           "InterVA-5."),
             "random_physician": ("randomva2 data with additional columns "
                                  "specifying physician ID and codes"),
             "randomva1": ("Dataset consisting of 1000 arbitrary VA records "
                           "in the format expected by InterVA-4."),
             "randomva2": ("Dataset consisting of 1000 arbitrary VA records "
                           "in the format expected by InterVA-4 with "
                           "additional columns specifying age and sex, which "
                           "can serve as characteristics in sub-population "
                           "estimation."),
             "randomva5": ("Dataset consisting of 200 arbitrary VA records "
                           "in the format expected by InterVA-5."),
             "sample_category": ("Matrix explaining correspondence between "
                                 "InterVA causes and the physician coded "
                                 "cause categories"),
             "sample_physician": ("This is in the same format of the output "
                                  "running physician_debias. It is a data "
                                  "frame of 100 rows, and column represents "
                                  "ID and probability of the cause in each "
                                  "category.")}


def get_vadata(data_name: Union[str, None] = None,
               width: int = 80,
               verbose: bool = True) -> Union[None, DataFrame]:
    """
    Get example data or print the names and descriptions of the example data
    sets that are available.

    :param data_name: name of example data that will be returned (or no
    argument to print the example data sets with descriptions).
    :type data_name: str or None
    :param width: line width used when printing data dictionary
    :type width: int
    :param verbose: Should the data description be printed
    :type verbose: bool
    :return: example data if valid name is provided, otherwise print data
    descriptions
    :rtype: pandas.DataFrame
    """

    if data_name is None:
        if verbose:
            return pprint(DATA_DICT, width=width)
        else:
            return pprint(list(DATA_DICT.keys()))
    if data_name not in DATA_DICT.keys():
        raise InSilicoVAException(
            "Error from get_vadata -- "
            f"\n There is no example dataset named {data_name}.\n ")
    if verbose:
        print(f"\n {data_name}: {DATA_DICT[data_name]} \n")
    example_data_bytes = get_data(__name__,
                                  f"data/{data_name}.csv")
    example_data = read_csv(BytesIO(example_data_bytes))
    return example_data
