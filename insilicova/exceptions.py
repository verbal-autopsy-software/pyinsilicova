# -*- coding: utf-8 -*-

"""
insilicova.exceptions
-------------------

This module contains insilicova exceptions.
"""


class InSilicoVAException(Exception):
    """Base exception for package"""
    pass


class ArgumentException(InSilicoVAException):
    """Exception involving options passed to InSilicoVA arguments."""
    pass


class DataException(InSilicoVAException):
    """Exception involving VA data passed to InSilicoVA."""
    pass


class SamplerException(InSilicoVAException):
    """Exception involving InSilicoVA sampler."""
    pass


class HaltGUIException(InSilicoVAException):
    """GUI signaled InSilicoVA to stop."""
    pass
