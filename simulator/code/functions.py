#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:33:44 2022

@author: H.C. de Ferrante
"""

from typing import List, Callable
from simulator.magic_values.rules import BLOOD_GROUP_COMPATIBILITY_DICT
import simulator.magic_values.etkass_settings as es
import re


def determine_eligible_bloodgroups(
        donor_bloodgroup: str, type_rules: str
        ) -> List[str]:
    """Function to retrieve eligible bloodgroups."""

    assert type_rules in BLOOD_GROUP_COMPATIBILITY_DICT.keys(), \
        f'type rules should be one of:\n\t' \
        f'{", ".join(BLOOD_GROUP_COMPATIBILITY_DICT)}'

    return BLOOD_GROUP_COMPATIBILITY_DICT[type_rules][donor_bloodgroup]


def find_deepest(data):
    """Find deepest values for dictionary"""
    if not any([isinstance(data.get(k), dict) for k in data]):
        return data
    else:
        for dkey in data:
            if isinstance(data.get(dkey), dict):
                return find_deepest(data.get(dkey))
            else:
                continue


def construct_piecewise_term(
    trafo: str, trafo_x: Callable = es.identity
) -> Callable:
    """Construct a piecewise term"""

    relation, constant = trafo.split('_')

    assert relation in ['under', 'over'], \
        f'Relation should be "under" or "over", not {relation}'

    constant = re.sub('(?<=[0-9])p(?=[0-9])', '.', constant)

    if 'log' in constant:
        constant = constant.replace('log', '')
    if 'pc' in constant:
        constant = float(constant.replace('pc', ''))/100
    else:
        if 'm' in constant:
            constant = -float(''.join(c for c in constant if c.isdigit()))
        else:
            constant = float(constant)

    if relation == 'under':
        def fun(xvals):
            return max(0, trafo_x(constant)-trafo_x(xvals))
        return fun
    else:
        def fun(xvals):
            return max(0, trafo_x(xvals)-trafo_x(constant))
        return fun



def construct_alloc_fun(
    trafo: str
) -> Callable:

    if trafo.startswith('under') or trafo.startswith('over') or trafo.startswith('between'):
        return construct_inequality_indicator(trafo)
    elif trafo.startswith('lt') or trafo.startswith('gt'):
        return construct_slope_term(trafo)
    else:
        raise Exception(f'{trafo} is not a valid transformation')


def construct_minus_fun(
    var: str
) -> Callable:

    assert '_minus_' in var, \
        f'Relation should be a minus relation'
    var1, var2 = var.split('_minus_')

    def fun(xdict):
            return xdict[var1]-xdict[var2]
    return fun


def construct_inequality_indicator(
    trafo: str
) -> Callable:
    """Construct a piecewise term"""

    if trafo.count('_') == 2:
        relation, constant1, constant2 = trafo.split('_', 2)
    else:
        relation, constant1, constant2 = trafo.split('_', 2) + [None]

    assert relation in ['under', 'over', 'between'], \
        f'Relation should be "under" or "over", not {relation}'

    constant1 = convert_threshold_to_string(constant1)
    if constant2:
        constant2 = convert_threshold_to_string(constant2)

    if relation == 'under':
        def fun(xvals):
            return 1 if xvals <= constant1 else 0
        return fun
    elif relation == 'over':
        def fun(xvals):
            return 1 if xvals > constant1 else 0
        return fun
    elif relation == 'between':
        def fun(xvals):
            return 1 if (xvals >= constant1) and (xvals <= constant2) else 0
        return fun


def split_string(s):
    """Splits string into text and numeric part"""
    head = s.rstrip('0123456789')
    tail = s[len(head):]
    return head, tail

def convert_threshold_to_string(s):
    s = re.sub('(?<=[0-9])p(?=[0-9])', '.', s)
    if 'm' in s:
        return -float(''.join(c for c in s if c.isdigit()))
    else:
        return float(s)

def construct_slope_term(
    trafo: str
) -> Callable:
    """Construct a piecewise term"""

    relation, min_or_max = trafo.split('_')
    rel_operator, ref_lvl = split_string(relation)
    minmax, minmax_lvl = split_string(min_or_max)

    assert rel_operator in ['lt', 'gt'], \
        f'Relation should be "lt" or "gt", not {relation}'

    ref_lvl = convert_threshold_to_string(ref_lvl)
    minmax_lvl = convert_threshold_to_string(minmax_lvl)

    if rel_operator == 'lt':
        assert minmax == 'min', \
            f'Expected a minimum for {rel_operator} operator, not {minmax}'
        def fun(xvals):
            return max(ref_lvl - max(xvals, minmax_lvl), 0)
        return fun
    elif rel_operator == 'gt':
        assert minmax == 'max', \
            f'Expected a max for {rel_operator} operator, not {minmax}'
        def fun(xvals):
            return max(min(xvals, minmax_lvl) - ref_lvl, 0)
        return fun
    else:
        raise Exception("Unknown operator")
