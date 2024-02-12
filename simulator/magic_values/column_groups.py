#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:33:44 2022

blood group compatibility rules

@author: H.C. de Ferrante
"""

import sys

sys.path.append('./')
if True:  # noqa: E402
    import simulator.magic_values.column_names as cn

# Other groups
BG_LEVELS = (cn.BG_A, cn.BG_AB, cn.BG_O, cn.BG_B)


MATCHLIST_COLS = (
    cn.ID_MTR, cn.MATCH_DATE, cn.ID_TXP,
    cn.TYPE_OFFER, cn.ID_DONOR, cn.D_DCD, cn.D_WEIGHT,
    cn.D_BLOODGROUP, cn.D_COUNTRY, cn.D_ALLOC_CENTER,
    cn.REC_OFFERED, cn.R_MATCH_AGE, cn.MATCH_CRITERIUM,
    cn.URGENCY_CODE, cn.R_BLOODGROUP,
    cn.REC_REQ, cn.REQ_INT, cn.RECIPIENT_COUNTRY,
    cn.RECIPIENT_CENTER, cn.R_WEIGHT, cn.RNK_OTHER,
    cn.RNK_ACCEPTED, cn.ACCEPTED, cn.ACTION_CODE,
    cn.ACTION_CODE_DESC, cn.FINAL_REC_URG_AT_TRANSPLANT
)

MTCH_COLS = (
    cn.TOTAL_MATCH_POINTS,
)

BG_TABS = (cn.TAB1, cn.TAB2, cn.TAB3)
BG_RULES = (cn.BGC_FULL, cn.BGC_TYPE1, cn.BGC_TYPE2)
BG_RULES_EXT = BG_RULES + BG_TABS + \
    (
        cn.BG_COMP, cn.BG_IDENTICAL, cn.BG_TAB_COL,
        cn.BG_RULE_COL, cn.BG_PRIORITY
    )

ACCEPTANCE_CODES = (
    cn.T1, cn.T3, cn.CR, cn.RR, cn.FP
)

REGIONAL_INFO = (
    cn.ALLOCATION_LOC, cn.ALLOCATION_INT,
    cn.ALLOCATION_NAT, cn.ALLOCATION_REG
)

TEST_MRL_COLS = (
    cn.ID_RECIPIENT, cn.ID_MTR, cn.D_COUNTRY, cn.RECIPIENT_COUNTRY,
    cn.TYPE_OFFER, cn.D_ALLOC_REGION, cn.D_ALLOC_COUNTRY,
    cn.D_ALLOC_CENTER, cn.RECIPIENT_CENTER, cn.PATIENT_IS_HU,
    cn.R_PED, cn.D_BLOODGROUP, cn.R_BLOODGROUP,
    cn.ALLOCATION_LOC, cn.R_MATCH_AGE
)

PROFILE_VARS = (
    cn.PROFILE_HBSAG, cn.PROFILE_HCVAB, cn.PROFILE_HBCAB, cn.PROFILE_SEPSIS,
    cn.PROFILE_MENINGITIS, cn.PROFILE_MALIGNANCY, cn.PROFILE_DRUG_ABUSE,
    cn.PROFILE_RESCUE, cn.PROFILE_EUTHANASIA
)

