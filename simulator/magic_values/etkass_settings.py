#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:33:44 2022

Magic values for the simulator. Only magic values
which are to some extent immodifiable are included.

@author: H.C. de Ferrante
"""

from datetime import timedelta, datetime
from typing import Any
import numpy as np
import simulator.magic_values.magic_values_rules as mgr
import simulator.magic_values.column_names as cn
import simulator.magic_values.column_groups as cg
import simulator.magic_values.magic_values_rules as mr
from functools import reduce
from operator import or_
from itertools import product

DAYS_PER_YEAR = timedelta(days=365.25)
DEFAULT_DATE_TIME = '%Y-%m-%d %H:%M:%S'

READ_CSV_ENGINE = 'pyarrow'

DEFAULT_ATTR_ORDER = (
    cn.MTCH_TIER,
    cn.TOTAL_MATCH_POINTS,
    cn.MM_TOTAL,
    cn.YEARS_ON_DIAL,
    cn.ID_RECIPIENT
)

AGE_ESP_ELIGIBLE = 65

HLA_MQS = list(
    product(
        range(3),
        range(3),
        range(3)
    )
)
HLA_MQS_STR = list(
    ''.join((str(i) for i in item)) for item in HLA_MQS
)
PROFILE_HLA_MQS = {
    hla: f'profile_hla{mq_str}' for hla, mq_str in zip(HLA_MQS, HLA_MQS_STR)
}


# Transformations
def identity(__x: Any):
    """Identitity transformation function"""
    return __x


def square(__x: Any):
    """Identitity transformation function"""
    return __x**2


TRAFOS = {
    cn.IDENTITY_TRAFO: identity,
    cn.LOG: np.log
}

# Directory with simulation settings
DIR_SIM_SETTINGS = 'simulator/sim_yamls/'
DIR_ACCEPTANCE_COEFS = 'simulator/magic_values/acceptance/'
DIR_POSTTXP_COEFS = 'simulator/magic_values/post_txp/'
DIR_TEST_LR = 'data/test/'

# Path to rescue probs
PATH_RESCUE_PROBABILITIES = (
    'simulator/magic_values/probabilities_rescue_triggered.csv'
)

# Paths to files with travel information
PATH_DRIVING_DISTANCE = (
    'simulator/magic_values/travel_distances/driving_distances.csv'
)
PATH_DRIVING_TIMES = (
    'simulator/magic_values/travel_distances/driving_times.csv'
)
PATH_FLIGHT_TIMES = (
    'simulator/magic_values/travel_distances/total_flight_times.csv'
)

# Time / km above which plane is transport mode of choice.
# Note, the acceptance model was calibrated on these.
MAX_DRIVE_TIME: float = 5
MAX_DRIVE_KM: float = 300

# Paths relevant for the acceptance module
ACCEPTANCE_PATHS = {
    k: DIR_ACCEPTANCE_COEFS + v for k, v in {
        'rd': 'coefs_recipient_driven.csv'
    }.items()
}
LR_TEST_FILES = {
    k: DIR_TEST_LR + v for k, v in {
        'rd': 'acceptance_pd.csv',
    }.items()
}

# Paths relevant for post-transplant survival predictions
POSTTXP_RELISTPROB_PATHS = {
    k: DIR_POSTTXP_COEFS + v for k, v in {
        cn.T: 'prob_relist_T.csv',
        cn.HU: 'prob_relist_HU.csv'
    }.items()
}
POSTTXP_SURV_PATHS = {
    k: DIR_POSTTXP_COEFS + v for k, v in {
        cn.T: 'posttx_coefs_T.csv',
        cn.HU: 'posttx_coefs_HU.csv'
    }.items()
}
POSTTXP_DEATHPROBS_PATHS = {
    k: DIR_POSTTXP_COEFS + v for k, v in {
        cn.T: 'posttx_fracdeath_T.csv',
        cn.HU: 'posttx_fracdeath_HU.csv'
    }.items()
}
POSTTXP_SURV_TESTPATHS = {
    k: DIR_TEST_LR + v for k, v in {
        cn.T: 'posttx_testcases_T.csv',
        cn.HU: 'posttx_testcases_HU.csv'
    }.items()
}

OFFER_INHERIT_COLS = {
    'donor': [
        cn.D_AGE, cn.DONOR_DEATH_CAUSE_GROUP,
        cn.D_TUMOR_HISTORY, cn.D_MARGINAL_FREE_TEXT,
        cn.D_DCD, cn.D_DIABETES, cn.RESCUE
    ],
    'patient': [
        cn.PATIENT_SEX, cn.URGENCY_CODE, cn.IS_RETRANSPLANT, cn.R_PED
    ]
}

# Settings for post-transplant module
POSTTXP_DISCRETE_MATCH_VARS = [cn.REREG_RETURN_DIAL_TIME, cn.RECIPIENT_COUNTRY]
POSTTXP_CONTINUOUS_MATCH_VARS = {
    cn.RETRANSPLANT: [
        cn.AGE_PREV_TXP, cn.TIME_SINCE_PREV_TXP, cn.TIME_LIST_TO_REALEVENT
    ],
    cn.OFFER: [cn.R_MATCH_AGE, cn.TIME_TO_REREG, cn.TIME_LIST_TO_REALEVENT]
}
POSTTXP_MIN_MATCHES = 5
POSTTXP_MATCH_CALIPERS = [20.0, 1.0, 1.0]


def log_ceil(x):
    return np.log(np.ceil(x+2))


POSTTXP_TRANSFORMATIONS = [identity, log_ceil, log_ceil]

POSTTXP_COPY_VARS = [
    cn.ID_RECIPIENT, cn.R_BLOODGROUP, cn.R_WEIGHT,
    cn.R_HEIGHT, cn.RECIPIENT_CENTER,
    cn.RECIPIENT_COUNTRY, cn.RECIPIENT_REGION,
    cn.R_DOB, cn.PATIENT_SEX,
    'profile'
]

hlas_to_load = (
    mr.HLA_A, mr.HLA_B, mr.HLA_DR
)

# Allowed values for various factors.
ALLOWED_BLOODGROUPS = set([mr.A, mr.B, mr.AB, mr.O])
ALLOWED_STATUSES = set([mr.NT, mr.T, mr.HU, mr.I, mr.HI])
EXIT_STATUSES = set([mr.FU, mr.R, mr.D])
ACTIVE_STATUSES = set([mr.T, mr.HU, mr.I, mr.HI])
ALL_STATUS_CODES = ALLOWED_STATUSES.union(EXIT_STATUSES)

# Countries & specification of country rules.
ET_COUNTRIES = set([mgr.NETHERLANDS, mgr.BELGIUM, mgr.GERMANY, mgr.AUSTRIA, mgr.HUNGARY,
                mgr.SLOVENIA, mgr.CROATIA, mgr.LUXEMBOURG])
ET_COUNTRIES_OR_OTHER = ET_COUNTRIES.union(set([mgr.OTHER]))
RECIPIENT_DRIVEN_COUNTRIES = set([mgr.GERMANY, mgr.BELGIUM, mgr.NETHERLANDS])
CENTER_DRIVEN_COUNTRIES = set([mgr.SLOVENIA, mgr.CROATIA, mgr.HUNGARY, mgr.AUSTRIA])
COUNTRIES_REGIONAL_BALANCES = set([mgr.AUSTRIA])
REGIONAL_BALANCE_CENTERS = set(
    [
        mgr.CENTER_INNSBRUCK, mgr.CENTER_GRAZ, mgr.CENTER_VIENNA,
        mgr.CENTER_UPPERAUSTRIA
    ]
)

# Check ETKAS / ESP eligibility every 30 days
CHECK_ETKAS_ESP_ELIGIBILITY = 30
COUNTRIES_ESP_ETKAS_MUTUALLY_EXCLUSIVE = set([mgr.GERMANY])
ESP_CHOICE_ENFORCED = {
    mgr.GERMANY: datetime(year=2011, month=1, day=1)
}

# Regional balance center groups
REG_BAL_CENTER_GROUPS = {
    'AWDTP': mgr.CENTER_VIENNA,
    'AOLTP': mgr.CENTER_UPPERAUSTRIA
}

NATION_GROUPS = {
    mgr.LUXEMBOURG: mgr.BELGIUM
}

# DCD accepting countries
DCD_COUNTRIES = set([mgr.NETHERLANDS, mgr.AUSTRIA, mgr.BELGIUM])
DCD_ACCEPTING_COUNTRIES = DCD_COUNTRIES.union(mgr.SLOVENIA)

# Countries which use centers for obligations
CNTR_OBLIGATION_CNTRIES = set([mgr.AUSTRIA])

# Save transplantation probabilities
WINDOWS_TRANSPLANT_PROBS = (
    28, 90, 365
)
PREFIX_SURVIVAL = 'psurv_posttxp'
COLS_TRANSPLANT_PROBS = list(
    f'{PREFIX_SURVIVAL}_{w}' for w in WINDOWS_TRANSPLANT_PROBS
)

# Pre-specify columns for simulation results
OUTPUT_COLS_DISCARDS = (
    'reporting_date',
    cn.ID_DONOR, cn.D_WEIGHT, cn.D_DCD,
    'donor_age', cn.D_BLOODGROUP,
    cn.N_OFFERS, cn.N_PROFILE_TURNDOWNS,
    cn.TYPE_OFFER_DETAILED
)
OUTPUT_COLS_PATIENTS = (
    cn.ID_RECIPIENT, cn.ID_REGISTRATION,
    cn.RECIPIENT_CENTER, cn.LISTING_DATE,
    cn.EXIT_STATUS, cn.EXIT_DATE, cn.URGENCY_REASON,
    cn.FINAL_REC_URG_AT_TRANSPLANT,
    cn.LAST_NONNT_HU,
    cn.TIME_SINCE_PREV_TXP,
    cn.TYPE_RETX,
    cn.RECIPIENT_COUNTRY,
    cn.ANY_ACTIVE,
    cn.R_DOB, cn.ANY_HU,
    cn.PATIENT_SEX,
    cn.INIT_URG_CODE,
    cn.R_BLOODGROUP,
    cn.DISEASE_SINCE,
    cn.DISEASE_GROUP,
    cn.URGENCY_CODE,
    cn.R_HEIGHT
)
OUTPUT_COLS_EXITS = (
    cn.ID_RECIPIENT, cn.ID_REGISTRATION,
    cn.TYPE_RETX, cn.ID_DONOR, cn.D_DCD, cn.EXIT_STATUS,
    cn.URGENCY_REASON, cn.LISTING_DATE,
    cn.EXIT_DATE, cn.MATCH_CRITERIUM, cn.GEOGRAPHY_MATCH,
    cn.MATCH_ABROAD, cn.RECIPIENT_CENTER,
    cn.RECIPIENT_COUNTRY, cn.R_BLOODGROUP, cn.D_BLOODGROUP,
    cn.MATCH_DATE, cn.PATIENT_RANK,
    cn.RANK, cn.D_ALLOC_CENTER, cn.D_ALLOC_COUNTRY,
    cn.R_MATCH_AGE, cn.R_PED,
    cn.PATIENT_IS_HU,
    cn.TYPE_TRANSPLANTED, cn.PATIENT_SEX,
    cn.ANY_HU, cn.ACCEPTANCE_REASON,
    cn.OFFERED_TO, cn.DISEASE_GROUP, cn.DISEASE_SINCE,
    cn.PROFILE_COMPATIBLE, cn.TYPE_OFFER_DETAILED,
    cn.PROB_ACCEPT_C, cn.PROB_ACCEPT_P, cn.DRAWN_PROB, cn.DRAWN_PROB_C
 ) + tuple(cg.MTCH_COLS) + tuple(COLS_TRANSPLANT_PROBS)
OUTPUT_COLS_EXIT_CONSTRUCTED = (
    cn.ID_REREGISTRATION, cn.TIME_WAITED
)

MATCH_INFO_COLS = (
    cn.ID_MTR, cn.MATCH_DATE, cn.OFFERED_TO, cn.N_OFFERS, cn.ID_DONOR,
    cn.D_DCD, cn.D_COUNTRY, cn.D_ALLOC_COUNTRY,
    cn.D_ALLOC_REGION, cn.D_ALLOC_CENTER,
    cn.D_WEIGHT, cn.D_AGE,
    cn.TYPE_OFFER, cn.TYPE_OFFER_DETAILED,
    cn.RECIPIENT_CENTER, cn.ID_RECIPIENT, cn.ID_REGISTRATION,
    cn.LISTING_DATE, cn.RECIPIENT_COUNTRY, cn.R_BLOODGROUP,
    cn.D_BLOODGROUP, cn.BG_PRIORITY, cn.R_MATCH_AGE, cn.R_WEIGHT,
    cn.R_PED, cn.MATCH_CRITERIUM, cn.GEOGRAPHY_MATCH,
    cn.DONOR_DEATH_CAUSE_GROUP, cn.D_TUMOR_HISTORY,
    cn.PATIENT_SEX, cn.PATIENT_RANK, cn.RANK, cn.ACCEPTED,
    cn.ACCEPTANCE_REASON, cn.PROB_ACCEPT_C, cn.PROB_ACCEPT_P, cn.DRAWN_PROB,
    cn.DRAWN_PROB_C, cn.PROFILE_COMPATIBLE, cn.D_HEIGHT, cn.R_HEIGHT
) + tuple(cg.MTCH_COLS)

MATCH_INFO_COLS_SET = set(MATCH_INFO_COLS)


# Cut-off for transplantation, where candidate's reregistration time is returned
CUTOFF_REREG_RETURN_DIAL_TIME = 90

# Add type statuses
STATUS_TYPES = set(
    (mgr.URG, mgr.FU, mgr.PRF, mgr.DIAG, mgr.HLA, mgr.UNACC, mgr.AM, mgr.DIAL)
)

# Event types
EVENT_TYPES = [cn.PAT, cn.DON]

# Terminal status updates
TERMINAL_STATUSES = [cn.R, cn.D]
EXITING_STATUSES = [cn.R, cn.D, cn.FU]

# Tiers in which profiles are ignored
IGNORE_PROFILE_TIERS = [cn.TIER_E]

# HU/ACO tiers (after reversing the alphabetical order of tiers)
TIERS_HUACO = [cn.TIER_D, cn.TIER_E]
TIERS_HUACO_SPLIT = [cn.TIER_B, cn.TIER_C, cn.TIER_E, cn.TIER_F]

# Implemented acceptance module policies
PATIENT_ACCEPTANCE_POLICIES = ['Always', 'LR']
CENTER_ACCEPTANCE_POLICIES = ['Always', 'LR']

# By which group to calculate MMAT
MMAT_GROUPING_VAR = 'recipient_country'
MMAT_TIME_HORIZON = 365
MMAT_MIN_MELD = 18


# Required donor and patient information for match records
MTCH_RCRD_DONOR_COLS = (
    cn.ID_DONOR, cn.DONOR_COUNTRY, cn.D_BLOODGROUP, cn.D_DCD
    )
MTCH_RCRD_PAT_COLS =  (
    cn.ID_RECIPIENT, cn.R_BLOODGROUP, cn.RECIPIENT_COUNTRY,
    cn.RECIPIENT_CENTER, cn.PATIENT_IS_HU
    )

HLA_FORBIDDEN_CHARACTERS = ('\*', ':')


UNSPLITTABLES = {
    mr.DR3: {mr.DR17, mr.DR18}
}
UNSPLITTABLE_BROADS = set(UNSPLITTABLES.keys())
UNSPLITTABLE_SPLITS = reduce(or_, UNSPLITTABLES.values())

MATCH_TO_BROADS = {
    v: f'mmb_{v}' for v in mr.HLA_LOCI
}
MATCH_TO_SPLITS = {
    v: f'mms_{v}' for v in mr.HLA_LOCI
}


MISMATCH_STR_DEFINITION = (
    mr.MMB_HLA_A, mr.MMB_HLA_B, mr.MMS_HLA_DR
)

POINT_GROUPS = {
    cn.POINTS_HLA: (cn.INTERCEPT, mr.MMB_HLA_A, mr.MMB_HLA_B, mr.MMS_HLA_DR),
    cn.POINTS_HLA_PED: (f'{var}:{cn.R_PED}' for var in (cn.INTERCEPT, mr.MMB_HLA_A, mr.MMB_HLA_B, mr.MMS_HLA_DR)),
    'bal_nat': (cn.BALANCE_NAT, ),
    'bal_reg': (cn.BALANCE_REG,),
    cn.POINTS_WAIT: (cn.YEARS_ON_DIAL, cn.PREVIOUS_WT),
    cn.POINTS_DIST: (cn.ALLOCATION_NAT, cn.ALLOCATION_REG, cn.ALLOCATION_LOC),
    cn.POINTS_PED: (cn.R_PED,),
    cn.POINTS_MMP: (cn.ET_MMP,),
    cn.POINTS_HU: (cn.PATIENT_IS_HU,)
}
POINT_COMPONENTS_TO_GROUP = {
    iv: group.removeprefix('wfmr_xc_').upper()
    for group, it in POINT_GROUPS.items() for iv in it
}



