#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:33:44 2022

@author: H.C. de Ferrante
"""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Dict, Tuple, List, Set
from math import isnan

# Define a type hint for a single key-value pair.
KeyValuePair = Tuple[str, Any]
InnerTuple = Tuple[str, Tuple[KeyValuePair]]
RuleTuple = Tuple[InnerTuple, ...]

def round_to_int(x: float):
    if isnan(x):
        return x
    return int(x + 0.5)

def round_to_decimals(x: float, p: int):
    if isnan(x):
        return x
    p = float(10**p)
    return int(x * p + 0.5)/p

def len_setdiff(s1: Set[Any], s2: Set[Any]):
    return sum(1 for item in s1 if item not in s2)

class DotDict(dict):
    """Helper class which allows access with dot operator
    """
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for key, val in arg.items():
                    self[key] = val

        if kwargs:
            for key, val in kwargs.items():
                self[key] = val

    def __getattr__(self, attr) -> Any:
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]

    def __deepcopy__(self, memo=None):
        return DotDict(deepcopy(dict(self), memo=memo))


def zip_recursively_to_tuple(obl):
    """Convert dictionary of rules to RuleTuple.

    Leave dictionaries with strings as keys intact.
    These are nested dictionaries.

    At the "deepest" dictionary level, it should be a RuleTuple
    """
    if isinstance(obl, Mapping) and all(isinstance(k, str) for k in obl.keys()):
        return {
            key: zip_recursively_to_tuple(val) for key, val in obl.items()
            }
    else:
        return tuple(
            (
                key[0],
                tuple((c, a) for c, a in zip(conditions[0], conditions[1]))
            )
            for key, conditions in obl.items()
        )
