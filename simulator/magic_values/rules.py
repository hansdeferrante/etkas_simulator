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


DEFAULT_MELD_ETCOMP_THRESH = 30

# NSE rules for ET countries. Tuples of
# index in NSE_ID, initial 30-d equivalent,
# and 90-day upgrade, and max equivalent.
NSE_UPGRADES = [
    (1, 10, 10, 40),
    (2, 15, 10, 40)
]


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
    'GHOTP': 'GND',
    'GBCTP': 'GNO',
    'GESTP': 'GNW',
    'GHGTP': 'GND',
    'GHBTP': 'GBW',
    'GTUTP': 'GBW',
    'GRBTP': 'GBY',
    'GFMTP': 'GMI',
    'GMZTP': 'GMI',
    'GJETP': 'GOS',
    'GLPTP': 'GOS',
    'GBOTP': 'GNW',
    'GMLTP': 'GBY',
    'GMNTP': 'GNW',
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
    'GNDOR': 'GND',
    'GFRTP': 'GBW',
    'GLUTP': 'GND',
    'GMATP': 'GBW',
    'GULTP': 'GBW',
    'GGITP': 'GMI',
    'GFDTP': 'GMI',
    'GKSTP': 'GMI',
    'GMRTP': 'GMI',
    'GBBTP': 'GNW',
    'GDUTP': 'GNW',
    'GDRTP': 'GOS',
    'GFDTP': 'GMI',
    'GMRTP': 'GMI'
}

# Blood group rules which are applied to donor/recipient matches
# (tables 4.1.1-4.1.3 in the FS.)
RECIPIENT_ELIGIBILITY_TABLES = {
    (cn.TAB2, 1): ([cn.TYPE_OFFER], ['Split']),
    (cn.TAB2, 2): ([cn.D_PED, cn.R_PED], [True, True]),
    (cn.TAB1, 1): ([cn.D_ALLOC_COUNTRY], [mgr.GERMANY])
}
DEFAULT_BG_TAB_COLLE = cn.TAB3

# Blood group compatibility tables. Tab1 is used for German donors,
# tab2 for split donors and pediatric donor/recipients, tab3 is
# used for non-German donors (see above). Each blood group table
# is a dictionary. Keys are tuples (to disambiguate), where the
# first value refers to the type of blood group rule that is to
# be applied. The values are tuples of column variables/values,
# which must be equal to belong to the group.
BG_COMPATIBILITY_TAB = {}

BG_COMPATIBILITY_DEFAULTS = {
    cn.TAB1: cn.BGC_TYPE2,
    cn.TAB2: cn.BGC_FULL,
    cn.TAB3: cn.BGC_TYPE2
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


# Zip rules
RECIPIENT_ELIGIBILITY_TABLES = zip_recursively_to_tuple(RECIPIENT_ELIGIBILITY_TABLES)
BG_COMPATIBILITY_TAB = zip_recursively_to_tuple(BG_COMPATIBILITY_TAB)



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

