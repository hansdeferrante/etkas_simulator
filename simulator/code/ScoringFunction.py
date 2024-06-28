#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19th

@author: H.C. de Ferrante
"""

from typing import Tuple, Union, Optional, TYPE_CHECKING, Any, Dict
import simulator.magic_values.etkass_settings as es
import simulator.magic_values.column_names as cn
import simulator.magic_values.magic_values_rules as mr
from simulator.code.utils import round_to_decimals, round_to_int
from simulator.code.functions import construct_alloc_fun, construct_minus_fun
from math import isnan
from collections import defaultdict
import numpy as np

if TYPE_CHECKING:
    from simulator.code import entities
    from simulator.code import AllocationSystem

def clamp(x: float, lims: Tuple[float, float], default_lim: int = 0) -> float:
    """Force number between limits. Do not return a number."""
    if isnan(x):
        return(lims[default_lim])
    return max(min(lims[1], x), lims[0])

def get_fun_name(x: Any):
    if isinstance(x, str):
        return x
    else:
        return x.__name__


class MatchPointFunction:
    """Class which implements a scoring function
    ...

    Attributes   #noqa
    ----------
    coef: dict[str, float]
        coefficients to use to calculate score
    intercept: float
        intercept for calculating score
    trafos: dict[str, str]
        transformations to apply to score component
    caps: dict[str, Tuple[float, float]]
        caps to apply to score component
    limits: Tuple[float, float]
        caps to apply to final score
    round: bool
        whether to round scores to nearest integer

    Methods
    -------
    calc_score() -> float
    """

    def __init__(
            self,
            coef: dict[str, float],
            intercept: float,
            points_comp_to_group: Dict[str, str],
            trafos: Optional[Dict[str, str]] = {},
            point_multiplier: Optional[Dict[str, Any]] = None,
            caps: Optional[Dict[str, Tuple[float, float]]] = {},
            limits: Optional[Tuple[float, float]] = None,
            clamp_defaults: Optional[Dict[str, int]] = None,
            round: bool = True
    ) -> None:
        self.coef = {k.lower(): fl for k, fl in coef.items()}
        self.variables = set(self.coef.keys())
        self.interactions = {
            key: key.split(':')
            for key in self.coef.keys()
            if ':' in key
        }
        self.all_raw_variables = {
            var for var in self.coef.keys() if var not in self.interactions.keys()
        }.union(
            {var.split('-')[0] for var in sum(self.interactions.values(), [])}
        )

        # Format interactions as a tuple
        self.interactions = {
            key: [val.split('-') if '-' in val else val for val in values]
            for key, values in self.interactions.items()
        }


        # Implementation of a gravity model
        if point_multiplier is not None:
            self.point_multiplier = {
                mn.lower(): es.TRAFOS[mn.lower()] for mn in point_multiplier
            }
        else:
            self.point_multiplier = None

        # Transformations to apply
        if trafos is None:
            self.trafos = {}
        else:
            self.trafos = {
                k.lower(): es.TRAFOS[trafo_name] for k, trafo_name in trafos.items()
            }
        self.caps = caps

        if clamp_defaults:
            self.clamp_defaults = clamp_defaults

        self._vars_to_construct = None
        self._vars_to_construct_initialized = False

        self.limits = limits
        self.round=round
        self.points_comp_to_group = points_comp_to_group

    def add_interactions(self, d: Dict[str, Any]) -> None:
        d[cn.INTERCEPT] = 1
        for inter, (var1, var2) in self.interactions.items():
            if isinstance(var1, str) & isinstance(var2, str):
                d[inter] = d[var1] * d[var2]
            elif isinstance(var1, str):
                d[inter] = d[var1] * (d[var2[0]] == var2[1])
            elif isinstance(var2, str):
                d[inter] = (d[var1[0]] == var1[1]) * d[var2]
            else:
                d[inter] = (d[var1[0]] == var1[1]) * (d[var2[0]] == var2[1])

    def add_multipliers(self, d: Dict[str, Any]) -> None:
        if self.point_multiplier is not None:
            for var, trafo in self.point_multiplier.items():
                d[var] = trafo(d)

    def get_total_multiplier(self, d: Dict[str, Any]) -> float:
        if self.point_multiplier is None:
            return 1
        else:
            return np.prod(
                list(
                    d[variable]
                    for variable in self.point_multiplier.keys()
                )
            )

    def calc_score(
            self,
            match_record: dict[str, Any]
            ) -> float:
        """Calculate the score"""

        self.add_interactions(match_record)
        self.add_multipliers(match_record)

        # Calculate score. This is faster with a for-loop than
        # list comprehension.
        score=0
        for key, coef in self.coef.items():
            if key in self.trafos:
                score += self.trafos[key](match_record[key])*coef
            elif (value := match_record[key]) > 0:
                score += value*coef

        if self.point_multiplier is not None:
            total_multiplier = self.get_total_multiplier(match_record)
            score = score * total_multiplier

        #input('Next')
        if self.round:
            return round_to_int(score)
        return score

    def calc_patient_score(
        self,
        pd: dict[str, Any]
    ) -> float:

        # Calculate score. This is faster with a for-loop than
        # list comprehension.
        score=0
        for key, coef in self.coef.items():
            if key in self.trafos and key in pd:
                score += self.trafos[key](pd[key])*coef
            elif key in pd and pd[key] and pd[key] > 0:
                score += pd[key]*coef
        if self.round:
            return round_to_int(score)

        return score

    def calc_score_components(
            self,
            match_record: dict[str, Any]
            ) -> Dict[str, float]:
        """Calculate the score"""

        sc = defaultdict(float)

        self.add_interactions(match_record)
        self.add_multipliers(match_record)

        # Calculate score
        for k, coef in self.coef.items():
            if k not in self.points_comp_to_group:
                raise Exception(f'{k} not in points components')
            if k in self.trafos:
                sc[self.points_comp_to_group.get(k)] += self.trafos[k](match_record[k])*coef
            else:
                sc[self.points_comp_to_group.get(k)] += match_record[k] * coef

        if self.point_multiplier is not None:
            total_multiplier = self.get_total_multiplier(match_record)
            return {
                k: round_to_int(v*total_multiplier) for k, v in sc.items()
            }
        return {
            k: round_to_int(v) for k, v in sc.items()
        }

    def __str__(self):
        fcoefs = [
            f'{round_to_decimals(v, p=3)}*{get_fun_name(self.trafos.get(k, "I"))}({k})'
            for k, v in self.coef.items()
            ]
        if self.point_multiplier is None:
            return ' + '.join(fcoefs)
        else:
            return f'{"*".join(self.point_multiplier.keys()).lower()}*({" + ".join(fcoefs)})'

    def set_vars_to_construct(self, in_dict: dict[str, Any]):
        self._vars_to_construct = tuple(
            var for var in self.all_raw_variables
            if var not in in_dict.keys() and var != cn.INTERCEPT
        )
        self._vars_to_construct_initialized = True

    @property
    def vars_to_construct(self):
        return self._vars_to_construct
