#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19th

@author: H.C. de Ferrante
"""

from typing import Tuple, Union, Optional, TYPE_CHECKING, Any, Dict
import simulator.magic_values.etkass_settings as es
import simulator.magic_values.column_names as cn
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
            caps: Optional[Dict[str, Tuple[float, float]]] = {},
            limits: Optional[Tuple[float, float]] = None,
            clamp_defaults: Optional[Dict[str, int]] = None,
            round: bool = True
    ) -> None:
        self.intercept = intercept
        self.coef = {k.lower(): fl for k, fl in coef.items()}
        self.interactions = {
            key: key.split(':')
            for key in self.coef.keys()
            if ':' in key
        }
        self.trafos = trafos
        self.caps = caps

        if clamp_defaults:
            self.clamp_defaults = clamp_defaults

        self.limits = limits
        self.round=round
        self.points_comp_to_group = points_comp_to_group

    def add_interactions(self, d: Dict[str, Any]) -> None:
        d[cn.INTERCEPT] = 1
        for inter, (var1, var2) in self.interactions.items():
            d[inter] = d[var1] * d[var2]


    def calc_score(
            self,
            match_record: dict[str, Any]
            ) -> float:
        """Calculate the score"""

        self.add_interactions(match_record)

        # Calculate score. This is faster with a for-loop than
        # list comprehension.
        score=0
        for key, coef in self.coef.items():
            if (value := match_record[key]) > 0:
                score += value*coef
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
            if key in pd and pd[key] and pd[key] > 0:
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

        # Calculate score
        for k, coef in self.coef.items():
            sc[self.points_comp_to_group.get(k)] += match_record[k] * coef

        sc[
            self.points_comp_to_group.get(cn.INTERCEPT)
        ] += self.intercept

        return {
            k: round_to_int(v) for k, v in sc.items()
        }

    def __str__(self):
        fcoefs = [
            f'{round_to_decimals(v, p=3)}*{self.trafos.get(k, "I")}({k})'
            for k, v in self.coef.items()
            ]
        if self.intercept != 0:
            return ' + '.join([str(self.intercept)] + fcoefs)
        else:
            return ' + '.join(fcoefs)
