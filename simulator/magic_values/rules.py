#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:33:44 2022

Rules for allocation

@author: H.C. de Ferrante
"""

from math import isnan
from simulator.code.utils import zip_recursively_to_tuple

import simulator.magic_values.column_names as cn
import simulator.magic_values.column_groups as cg
import simulator.magic_values.etkass_settings as es
import simulator.magic_values.magic_values_rules as mgr

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from simulator.code import entities

def le_91(_t):
    return _t <= 91
def le_183(_t):
    return _t <= 183
def le_275(_t):
    return _t <= 275
def le_365(_t):
    return _t <= 365

FRACTIONS_RETURN_WAITINGTIME = {
    le_91: 1,
    le_183: 0.75,
    le_275: 0.50,
    le_365: 0.25
}


# Center information
FL_CENTER_CODES = {
    'A': mgr.AUSTRIA,
    'B': mgr.BELGIUM,
    'C': mgr.CROATIA,
    'G': mgr.GERMANY,
    'H': mgr.HUNGARY,
    'L': mgr.BELGIUM,
    'N': mgr.NETHERLANDS,
    'S': mgr.SLOVENIA,
    'O': mgr.OTHER
}

DICT_CENTERS_TO_REGIONS = {
    'GHMTP': 'GND',
    'GHOTP': 'GND',
    'GBMTP': 'GND',
    'GBCTP': 'GNO',
    'GBETP': 'GNO',
    'GBATP': 'GNW',
    'GESTP': 'GNW',
    'GHGTP': 'GND',
    'GHBTP': 'GBW',
    'GTUTP': 'GBW',
    'GSTTP': 'GBW',
    'GRBTP': 'GBY',
    'GFMTP': 'GMI',
    'GMZTP': 'GMI',
    'GJETP': 'GOS',
    'GLPTP': 'GOS',
    'GBOTP': 'GNW',
    'GMLTP': 'GBY',
    'GMNTP': 'GNW',
    'GKKTP': 'GNW',
    'GKITP': 'GND',
    'GGOTP': 'GND',
    'GMHTP': 'GBY',
    'GKLTP': 'GNW',
    'GMBTP': 'GOS',
    'GNBTP': 'GBY',
    'GAKTP': 'GNW',
    'GROTP': 'GNO',
    'GWZTP': 'GBY',
    'GHSTP': 'GMI',
    'GKMTP': 'GNW',
    'GBDTP': 'GNO',
    'GHATP': 'GOS',
    'GNOOR': 'GNO',
    'GBWOR': 'GBW',
    'GOSOR': 'GOS',
    'GMIOR': 'GMI',
    'GNWOR': 'GNW',
    'GBYOR': 'GBY',
    'GAUTP': 'GBY',
    'GERTP': 'GBY',
    'GDMTP': 'GBY',
    'GNBTP': 'GBY',
    'GNDOR': 'GND',
    'GFRTP': 'GBW',
    'GLUTP': 'GND',
    'GMBTP': 'GNO',
    'GMATP': 'GBW',
    'GULTP': 'GBW',
    'GGITP': 'GMI',
    'GFDTP': 'GMI',
    'GKSTP': 'GMI',
    'GMRTP': 'GMI',
    'GBBTP': 'GNW',
    'GDUTP': 'GNW',
    'GDRTP': 'GOS',
    'GFDTP': 'GMI'
}


DICT_ESP_SUBREGIONS = {
    'GHBTP': 'STUTTGART',   #GBW
    'GMATP': 'STUTTGART',   #GBW
    'GSTTP': 'STUTTGART',   #GBW
    'GTUTP': 'STUTTGART',   #GBW
    'GFRTP': 'FREIBURG',    #GBW
    'GAUTP': 'MUNCHEN',     #GBY
    'GMHTP': 'MUNCHEN',     #GBY
    'GMLTP': 'MUNCHEN',     #GBY
    'GRBTP': 'MUNCHEN',
    'GNBTP': 'ERLANGEN',
    'GWZTP': 'ERLANGEN',    #GBY
    'GFMTP': 'MAINZ',       #GMI
    'GMZTP': 'MAINZ',
    'GHSTP': 'HOMBURG',
    'GKSTP': 'HOMBURG',
    'GFDTP': 'MARBURG',
    'GGITP': 'MARBURG',
    'GMRTP': 'MARBURG',     #GMI
    'GBMTP': 'HANNOVER',    #GND
    'GHOTP': 'HANNOVER',
    'GHMTP': 'HANNOVER',
    'GHGTP': 'HAMBURG',
    'GKITP': 'HAMBURG',
    'GLUTP': 'HAMBURG',     #GND
    'GBETP': 'BERLIN',      #GNO
    'GBCTP': 'BERLIN',
    'GROTP': 'ROSTOCK',     #GNO
    'GBBTP': 'DUSSELDORF',  #GNW
    'GDUTP': 'DUSSELDORF',
    'GESTP': 'DUSSELDORF',
    'GAKTP': 'KOLNBONN',
    'GBOTP': 'KOLNBONN',
    'GKLTP': 'KOLNBONN',
    'GKMTP': 'KOLNBONN',
    'GMNTP': 'MUENSTER',    #GNW
    'GDRTP': 'LEIPZIG',     #GOS
    'GHATP': 'LEIPZIG',
    'GJETP': 'LEIPZIG',
    'GLPTP': 'LEIPZIG'
}

# Dictionary of blood group compatibility rules. For each rule,
# the key refers to the donor blood group, and the values
# are eligible recipients.
BLOOD_GROUP_COMPATIBILITY_DICT = {
    cn.BGC_FULL: {  # Full compatibility
        cn.BG_A: set((cn.BG_A, cn.BG_AB)),
        cn.BG_B: set((cn.BG_B, cn.BG_AB)),
        cn.BG_AB: set((cn.BG_AB,)),
        cn.BG_O: set((cn.BG_A, cn.BG_B, cn.BG_AB, cn.BG_O))
    },
    cn.BGC_TYPE1: {  # ET compatible
        cn.BG_A: set((cn.BG_A, cn.BG_AB)),
        cn.BG_B: set((cn.BG_B, cn.BG_AB)),
        cn.BG_AB: set((cn.BG_AB, )),
        cn.BG_O: set((cn.BG_B, cn.BG_O))
    },
    cn.BGC_TYPE2: {  # TPG compatible
            cn.BG_A: set((cn.BG_A, cn.BG_AB)),
            cn.BG_B: set((cn.BG_B, cn.BG_AB)),
            cn.BG_AB: set((cn.BG_AB,)),
            cn.BG_O: set((cn.BG_O,))
        },
}

BLOOD_GROUP_INCOMPATIBILITY_DICT = {
    k: set(cg.BG_LEVELS).difference(v)
    for k, v in BLOOD_GROUP_COMPATIBILITY_DICT[cn.BGC_FULL].items()
}


# Functions to check whether donor/patient are pediatric
def check_etkas_ped_don(donor: 'entities.Donor') -> bool:
    """Checks whether a donor is pediatric (i.e. <46kg)"""
    if not isnan(donor.__dict__[cn.D_AGE]):
        return donor.__dict__[cn.D_AGE] < 16
    return False


def check_etkas_ped_rec(
        age_at_listing: float,
        match_age: float,
        age_first_dial: Optional[float],
        prev_txp_ped: Optional[bool],
        time_since_prev_txp: Optional[float]
    ) -> bool:
    """Checks whether a patient is pediatric (i.e. <16 years)"""

    if age_at_listing < 16:
        if not isnan(match_age) and match_age < 17:
            return True
        else:
            if age_first_dial and age_first_dial < 17:
                return True

    if prev_txp_ped and time_since_prev_txp:
        if prev_txp_ped and time_since_prev_txp < 91:
            return True

    return False

