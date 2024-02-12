#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:33:44 2022

@author: H.C. de Ferrante
"""

import warnings
from typing import List, Dict, Tuple, Optional, Callable, DefaultDict, Union, Any, Set
from datetime import timedelta, datetime
from math import floor, isnan, ceil, prod
from itertools import count
from copy import deepcopy, copy
from warnings import warn
from collections import defaultdict, deque

import yaml
import pandas as pd
import numpy as np
from simulator.code.utils import round_to_int, round_to_decimals, len_setdiff, DotDict
import simulator.magic_values.etkass_settings as es
import simulator.magic_values.magic_values_rules as mgr
from simulator.code.StatusUpdate import StatusUpdate, ProfileUpdate
from simulator.code.PatientStatusQueue import PatientStatusQueue
from simulator.code.SimResults import SimResults
import simulator.magic_values.column_names as cn
import simulator.magic_values.magic_values_rules as mr
import simulator.code.read_input_files as rdr
from simulator.code.ScoringFunction import clamp

from simulator.magic_values.inputfile_settings import DEFAULT_DMY_HMS_FORMAT
from simulator.code.read_input_files import read_hla_match_table

def isDigit(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def nanOrNone(x):
    if x is None:
        return True
    if isnan(x):
        return True
    return False

class Donor:
    """Class which implements a donor

    ...

    Attributes   #noqa
    ----------
    id_donor: int
        integer for the donor ID
    donor_country: str
        name of country of donation
    don_region: str
        name of donor region
    donor_center: str
        name of donating center (5-letter code).
    donor_age: int
        Donor age
    bloodgroup: str
        recipient blood group
    reporting_date: datetime
        Time of entry on the liver waitlist


    Methods
    -------
    arrival_at():
        time at which donor arrives, in time from simulation date.
    """

    def __init__(
            self, id_donor: int, n_kidneys_available: int,
            bloodgroup: str, donor_country: str,
            donor_center: str, donor_region: str,
            reporting_date: datetime, donor_dcd: int,
            hla: str,
            hypertension: bool,
            diabetes: int,
            cardiac_arrest: bool,
            last_creat: float,
            smoker: bool,
            age: int, hbsag: bool,
            hcvab: bool, hbcab: bool,
            sepsis: bool, meningitis: bool,
            malignancy: bool, drug_abuse: bool,
            euthanasia: bool,
            rescue: bool, death_cause_group: str,
            hla_system: 'entities.HLASystem' = None,
            donor_marginal_free_text: Optional[int] = 0,
            tumor_history: Optional[int] = 0,
            height: Optional[float] = None,
            weight: Optional[float] = None
        ) -> None:

        self.id_donor = id_donor
        self.reporting_date = reporting_date
        self.n_kidneys_available = n_kidneys_available

        # Geographic information
        assert donor_country in es.ET_COUNTRIES, \
            f'donor country should be one of:\n\t' \
            f'{", ".join(es.ET_COUNTRIES)}, not {donor_country}'
        self.donor_country = donor_country
        self.donor_region = donor_region
        self.donor_center = donor_center
        self.__dict__[cn.D_PED] = None

        # Donor blood group
        assert bloodgroup in es.ALLOWED_BLOODGROUPS, \
            f'blood group should be one of:\n\t' \
            f'{", ".join(es.ALLOWED_BLOODGROUPS)}'
        self.d_bloodgroup = str(bloodgroup)

        self.graft_dcd = bool(donor_dcd)

        # Profile info
        if age is not None:
            self.__dict__[cn.D_AGE] = age
        self.__dict__[cn.D_HBSAG] = hbsag if hbsag is not None and  \
            hbsag is not np.nan else False
        self.__dict__[cn.D_HCVAB] = hcvab if hcvab is not None and \
            hcvab is not np.nan else False
        self.__dict__[cn.D_HBCAB] = hbcab if hbcab is not None and \
            hbcab is not np.nan else False
        self.sepsis = sepsis if sepsis is not None and \
            sepsis is not np.nan else False
        self.meningitis = meningitis if meningitis is not None and \
            meningitis is not np.nan else False
        self.malignancy = malignancy if malignancy is not None and \
            malignancy is not np.nan else False
        self.drug_abuse = drug_abuse if drug_abuse is not None and \
            drug_abuse is not np.nan else False
        self.euthanasia = euthanasia if euthanasia is not None and \
            euthanasia is not np.nan else False
        self.rescue = rescue if rescue is not None and \
            rescue is not np.nan else False
        self.death_cause_group = death_cause_group \
            if death_cause_group is not None and \
            death_cause_group is not np.nan else False

        self.donor_weight = weight
        self.donor_height = height

        # Extra information needed for predicting acceptance
        self.__dict__[cn.D_MARGINAL_FREE_TEXT] = donor_marginal_free_text \
            if not nanOrNone(donor_marginal_free_text) else 0
        self.__dict__[cn.D_TUMOR_HISTORY] = tumor_history \
            if not nanOrNone(tumor_history) else 0
        self.__dict__[cn.D_DIABETES] = diabetes \
            if not nanOrNone(diabetes) else 0
        self.__dict__[cn.D_SMOKING] = smoker \
            if not nanOrNone(smoker) else 0
        self.__dict__[cn.D_HYPERTENSION] = hypertension \
            if not nanOrNone(hypertension) else 0
        self.__dict__[cn.D_LAST_CREAT] = last_creat \
            if not nanOrNone(last_creat) else 0
        self.__dict__[cn.D_CARREST] = cardiac_arrest \
            if not nanOrNone(cardiac_arrest) else 0

        self._needed_match_info = None
        self._offer_inherit_cols = None

        self.hla = hla
        for fb in es.HLA_FORBIDDEN_CHARACTERS:
            if isinstance(hla, str) and fb in hla:
                warnings.warn(f'Forbidden character {fb} in donor HLA-input string! ({hla})')
                break
        if isinstance(hla_system, HLASystem):
            self.hla_broads, self.hla_splits, self.hla_alleles = hla_system.return_structured_hla(
                hla
            )
            self.hla_broads_homozygous = {
                k: len(v) == 1 for k, v in self.hla_broads.items()
            }
            self.hla_splits_homozygous = {
                k: len(v) == 1 for k, v in self.hla_splits.items()
            }
            self._fully_homozygous = None
            self.all_antigens = hla_system.return_all_antigens(hla)

    def arrival_at(self, sim_date: datetime):
        """Retrieve calendar time arrival time."""
        return (
                self.reporting_date - sim_date
        ) / timedelta(days=1)

    def determine_fully_homozygous(self, hla_system: 'entities.HLASystem') -> bool:
        if self._fully_homozygous is not None:
            return self._fully_homozygous
        else:
            for nb in hla_system.needed_broad_mismatches:
                if len(self.hla_broads[nb]) != 1:
                    self._fully_homozygous = False
                    break
            else:
                for ns in hla_system.needed_split_mismatches:
                    if len(self.hla_splits[ns]) != 1:
                        self._fully_homozygous = False
                    else:
                        self._fully_homozygous = True
            return self._fully_homozygous

    def get_pediatric(self, ped_fun: Callable) -> bool:
        """Check whether donor is pediatric"""
        if self.__dict__[cn.D_PED] is None:
            self.__dict__[cn.D_PED] = ped_fun(self)
        return self.__dict__[cn.D_PED]

    def __str__(self):
        return(
            f'Donor {self.id_donor}, reported on '
            f'{self.reporting_date} in {self.donor_center}'
            f', dcd: {self.graft_dcd}, bg: {self.d_bloodgroup} '
            f'and age {self.__dict__[cn.D_AGE] } with '
            f'{self.n_kidneys_available} ESP/ETKAS kidneys'
            )

    def __repr__(self):
        return(
            f'Donor {self.id_donor}, reported on '
            f'{self.reporting_date} in {self.donor_center}'
            f', dcd: {self.graft_dcd}, bg: {self.d_bloodgroup}'
            f' and age {self.__dict__[cn.D_AGE] }'
            )

    @property
    def needed_match_info(self):
        if not self._needed_match_info:
            self._needed_match_info = {
                k: self.__dict__[k]
                for k
                in es.MTCH_RCRD_DONOR_COLS
            }
        return self._needed_match_info

    @property
    def offer_inherit_cols(self):
        if not self._offer_inherit_cols:
            self._offer_inherit_cols = {
                k: self.__dict__[k] for k in es.OFFER_INHERIT_COLS['donor']
            }
        return self._offer_inherit_cols


class Patient:
    """Class which implements a patient
    ...

    Attributes   #noqa
    ----------
    id_recipient: int
        integer for the recipient ID
    r_dob: datetime
        patient DOB
    age_at_listing: int
        age in days at listing
    recipient_country: str
        name of country of listing (not nationality!)
    recipient_region: str
        name of region where candidate's from
    recipient_center: str
        name of listing center (5-letter code).
    bloodgroup: str
        recipient blood group
    hla_broads: Dict[str, Set[str]]
        broad HLAs per locus
    hla_splits: Dict[str, Set[str]]
        split HLAs per locus
    ped_status: Optional[ExceptionScore]
        pediatric status
    listing_date: datetime
        Time of entry on the liver waitlist
    urgency_code: str
        Status of listing (HU/NT/T/HI/I)
    urgency_reason: Optional[str]
        urgency reason
    last_update_time: float
        time at which the last update was issued
    hu_since: float
        time since registration since when patient is HU
    future_statuses: Optional[PatientStatusQueue]
        status queue (ordered list of status updates)
    exit_status: Optional[str]
        exit status
    exit_date: Optional[datetime]
        exit time
    sim_set: DotDict,
        simulation settings under which pat. is created
    profile: Optional[Profile]
        patient's donor profile
    patient_sex: str
        sex of the patient,
    time_since_prev_txp: float
        Time elapsed since previous TXP
    id_received_donor: int
        donor ID
    id_registration: int
        registration ID [optional]
    seed: int
        Optional seed for random number generator
    time_to_dereg: float
        time to deregistration (in days from registration)
    rng_acceptance: Generator
        Random number generator for acceptance probabilities

    Methods
    -------
    get_age_at_t(time: float)
        Return age at time t
    get_next_update_time()
        Return time at which next update is issued
    get_acceptance_prob():
        Return an acceptance probability
    """

    def __init__(
            self, id_recipient: int, recipient_country: str,
            recipient_center: str, recipient_region: str, bloodgroup: str,
            listing_date: datetime, urgency_code: str,
            hla: str,
            r_dob: datetime,
            sim_set: DotDict,
            sex: Optional[str] = None,
            type_retx: str = cn.NO_RETRANSPLANT,
            time_since_prev_txp: Optional[float] = None,
            prev_txp_ped: Optional[bool] = None,
            prev_txp_living: Optional[bool] = None,
            id_reg: Optional[int] = None,
            date_first_dial: Optional[datetime] = None,
            seed: Optional[int] = None,
            hla_system: Optional['entities.HLASystem'] = None,
            mmp: Optional[float] = None,
            previous_wt: Optional[float] = None,
            kidney_program: Optional[str] = None
        ) -> None:

        self.id_recipient = int(id_recipient)
        self.id_registration = id_reg if id_reg else None
        self.id_received_donor = None
        self.sim_set = sim_set
        self._seed_acceptance_behavior(seed=seed)

        # Load patient bloodgroup
        assert bloodgroup in es.ALLOWED_BLOODGROUPS, \
            f'blood group should be one of:\n\t' \
            f'{", ".join(es.ALLOWED_BLOODGROUPS)}'
        self.r_bloodgroup = str(bloodgroup)

        # Check listing status
        assert urgency_code in es.ALLOWED_STATUSES, \
            f'listing status should be one of:\n\t' \
            f'{", ".join(es.ALLOWED_STATUSES)}, not {urgency_code}'
        self.urgency_code = str(urgency_code)
        self.__dict__[cn.PATIENT_IS_HU] = str(urgency_code) == cn.HU
        self.__dict__[cn.ANY_HU] = self.__dict__[cn.PATIENT_IS_HU]
        self.__dict__[cn.AM_STATUS] = False
        self.__dict__[cn.ANY_ACTIVE] = False
        self.urgency_reason = None

        # Set other information relating to listing statuses
        # Status updates
        self.future_statuses = None
        self.__dict__[cn.EXIT_STATUS] = None
        self.__dict__[cn.EXIT_DATE] = None
        self.last_update_time = None

        # Set patient center, region, and country
        self.__dict__[cn.RECIPIENT_CENTER] = str(recipient_center)
        self.__dict__[cn.RECIPIENT_REGION] = str(recipient_region)
        self.__dict__[cn.RECIPIENT_COUNTRY] = str(recipient_country)
        self.__dict__[cn.PATIENT_COUNTRY] = str(recipient_country)

        # Set date of birth, listing date, and dial date
        if not isinstance(r_dob, datetime):
            raise TypeError(
                f'r_dob must be datetime, not a {type(r_dob)}'
                )
        self.r_dob = r_dob
        self.initialize_etkas_esp_eligibility()
        self.set_listing_date(listing_date=listing_date)
        self.set_dial_date(date_first_dial)

        # Initialize patient information
        self.ped_status = None
        self.initialized = False
        self.profile = None
        self.__dict__[cn.PATIENT_SEX] = sex
        self.__dict__[cn.PATIENT_FEMALE] = int(sex == 'Female')
        self.disease_group = None

        # Set transplant history & previous waiting time
        self.__dict__[cn.TIME_SINCE_PREV_TXP] = (
            None if pd.isnull(time_since_prev_txp) else time_since_prev_txp
        )
        if time_since_prev_txp is None or pd.isnull(time_since_prev_txp):
            self.prev_txp_ped = 0
            self.prev_txp_living = 0
            self.__dict__[cn.AGE_PREV_TXP] = None
            self.__dict__[cn.TIME_SINCE_PREV_TXP]
        else:
            self.prev_txp_ped = prev_txp_ped
            self.prev_txp_living = prev_txp_living
            self.__dict__[cn.TIME_SINCE_PREV_TXP] = time_since_prev_txp
            self.__dict__[cn.AGE_PREV_TXP] = (
                (
                    self.age_days_at_listing
                ) - time_since_prev_txp
            ) / 365.25
        assert type_retx in (
            cn.NO_RETRANSPLANT, cn.RETRANSPLANT, cn.RETRANSPLANT_DURING_SIM
        )
        self.__dict__[cn.TYPE_RETX] = type_retx
        self.__dict__[cn.IS_RETRANSPLANT] = (
            0 if type_retx and type_retx == cn.NO_RETRANSPLANT
            else 1 if type_retx
            else None
        )
        # Add dummy indicating whether candidate is eligible for returned dial time
        self.__dict__[cn.REREG_RETURN_DIAL_TIME] = (
            self.__dict__[cn.TIME_SINCE_PREV_TXP] <= es.CUTOFF_REREG_RETURN_DIAL_TIME
            if self.__dict__[cn.TIME_SINCE_PREV_TXP]
            else False
        )
        self.pediatric_retransplant = (
            self.__dict__[cn.IS_RETRANSPLANT] and
            self.__dict__[cn.AGE_PREV_TXP] < 17
        ) or self.prev_txp_ped
        if previous_wt:
            self.__dict__[cn.PREVIOUS_WT] = previous_wt / 365.25
        else:
            self.__dict__[cn.PREVIOUS_WT] = 0

        # Geographic information
        assert recipient_country in es.ET_COUNTRIES, \
            f'listing country should be one of:\n\t' \
            f'{", ".join(es.ET_COUNTRIES)}, not {recipient_country}'

        # Initialize HLA information
        self.hla_system = hla_system
        for fb in es.HLA_FORBIDDEN_CHARACTERS:
            if isinstance(hla, str) and fb in hla:
                warnings.warn(f'Forbidden character {fb} in patient HLA-input string! ({hla})')
                break
        self._homozygosity_per_locus = None
        if isinstance(hla, str):
            self.set_match_hlas(hla_string = hla)
        else:
            self.set_match_hlas(hla_string='')
        self.vpra = 0

        # Participation in ETKAS or ESP (needed for Germany, if ESP)
        self.kidney_program: str = (
            kidney_program if isinstance(kidney_program, str) else mr.ETKAS
        )

        # Items we want to store, for rapid access through functions and properties
        self.__dict__[cn.R_PED] = None
        self._needed_match_info = None
        self._offer_inherit_cols = None
        self._active = None
        self._pediatric = None
        self._time_dial_start = None
        self._homozygosity_level = None
        self._et_mmp = None
        self._am = None
        self._bg = None
        self.__dict__[cn.UNACC_ANT] = None
        self._unacceptables = None
        self._esp_eligible = None
        self._etkas_eligible = None
        self._dcd_country = None
        self._eligibility_last_assessed = -1e10
        self._time_dial_at_listing = None

    def set_diag(self, upd: StatusUpdate):
        if self.__dict__[cn.DISEASE_GROUP] != upd.status_detail:
            self.__dict__[cn.DISEASE_GROUP] = upd.status_detail
            self.__dict__[cn.DISEASE_SINCE] = upd.arrival_time

    def set_dial_date(self, date_first_dial: Optional[datetime]) -> None:
        self.date_first_dial = date_first_dial
        self._time_dial_start = None
        if date_first_dial:
            self.age_first_dial = (date_first_dial - self.r_dob) / timedelta(days=365.25)
        else:
            self.age_first_dial = None

    def set_listing_date(self, listing_date: datetime) -> None:
        assert isinstance(listing_date, datetime), \
            f'listing date should be datetime, not {type(listing_date)}'

        self.__dict__[cn.LISTING_DATE] = listing_date
        self.age_days_at_listing = round_to_int(
            (listing_date - self.r_dob) / timedelta(days=1)
            )

        self.listing_offset = (
            listing_date - self.sim_set.SIM_START_DATE
        ) / timedelta(days=1)

    def initialize_etkas_esp_eligibility(self) -> None:
        # Calculate date at which patient becomes ESP eligible,
        # as well as simulation time when patient becomes ESP
        # eligible
        self.date_esp_eligible = (
            self.r_dob + timedelta(days = 365.25) * es.AGE_ESP_ELIGIBLE
        )
        self.sim_time_esp_eligible: float = (
            self.date_esp_eligible - self.sim_set.SIM_START_DATE
        ) / timedelta(days=1)
        self.always_etkas_aged: bool = (
            self.date_esp_eligible > self.sim_set.SIM_END_DATE
        )
        self.country_always_etkas_eligible: bool = (
            not self.__dict__[cn.RECIPIENT_COUNTRY] in es.COUNTRIES_ESP_ETKAS_MUTUALLY_EXCLUSIVE
        )
        self.program_choice_required_after: float = self.sim_set['times_esp_eligible'].get(
            self.__dict__[cn.RECIPIENT_COUNTRY],
            1e10
        )

    def set_match_hlas(self, hla_string: str) -> None:
        self.hla = hla_string
        if isinstance(self.hla_system, HLASystem):
            self.hla_broads, self.hla_splits, self.hla_alleles = self.hla_system.return_structured_hla(
                hla_string
            )
            self.__dict__.update(
                {
                    f'hz_{k}': v for k, v in
                    self.get_homozygosity_per_locus().items()
                }
            )
        else:
            self.hla_broads = None
            self.hla_splits = None

    def set_urgency_code(self, code: str, reason: Optional[str] = None):
        """Update urgency code with string"""
        if not code in es.ALL_STATUS_CODES:
            raise Exception(
                f'listing status should be one of:\n\t' \
                f'{", ".join(es.ALL_STATUS_CODES)},\n not {code}'
            )
        self.__dict__[cn.URGENCY_CODE] = code
        self.__dict__[cn.PATIENT_IS_HU] = code == cn.HU
        self._active = None
        if reason:
            self.__dict__[cn.URGENCY_REASON] = reason

    def set_transplanted(
            self, tx_date: datetime,
            donor: Optional[Donor] = None,
            sim_results: Optional[SimResults] = None,
            match_record: Optional[
                'simulator.code.AllocationSystem.MatchRecord'
            ] = None
    ):
        """Set patient to transplanted"""
        if self.future_statuses:
            self.future_statuses.clear()
        self.__dict__[cn.EXIT_DATE] = tx_date
        self.__dict__[cn.EXIT_STATUS] = cn.FU
        self.__dict__[cn.FINAL_REC_URG_AT_TRANSPLANT] = (
            self.__dict__[cn.URGENCY_CODE]
        )
        self.set_urgency_code(cn.FU)
        if donor:
            self.id_received_donor = donor.id_donor
        else:
            self.id_received_donor = 'Donor'
        if sim_results:
            sim_results.save_transplantation(
                pat=self, matchr=match_record
            )

    def get_acceptance_prob(self) -> float:
        """Simulate an acceptance probability"""
        prob = self.rng_acceptance.random()
        self.__dict__[cn.DRAWN_PROB] = round_to_decimals(prob, 3)
        return prob

    def get_dial_time_sim_start(self) -> Optional[float]:
        """Get dialysis start time, relative to simulation time"""
        if self._time_dial_start:
            return self._time_dial_start
        if self.date_first_dial:
            self._time_dial_start = (
                self.date_first_dial - self.sim_set.SIM_START_DATE
                ) / timedelta(days=1)
            return self._time_dial_start
        else:
            return None

    def get_dial_time_at_listing(self) -> Optional[float]:
        """Get dialysis start time, relative to simulation time"""
        if self._time_dial_at_listing:
            return self._time_dial_at_listing
        if self.date_first_dial is not None:
            self._time_dial_at_listing = (
                self.date_first_dial - self.__dict__[cn.TIME_REGISTRATION]
                ) / timedelta(days=1)
            return self._time_dial_at_listing
        else:
            return 0

    def get_pediatric(self, ped_fun: Callable, match_age: float) -> bool:
        """Check whether recipient is pediatric. Only check for
            candidates who were pediatric at past matches.
        """
        if self._pediatric or self._pediatric is None:
            self._pediatric = ped_fun(
                age_at_listing = self.age_days_at_listing / 365.25,
                match_age = match_age,
                age_first_dial = self.age_first_dial,
                prev_txp_ped = self.pediatric_retransplant,
                time_since_prev_txp = self.__dict__[cn.TIME_SINCE_PREV_TXP]
            )
            self.__dict__[cn.R_PED] = self._pediatric
        return self._pediatric

    def get_homozygosity_level(self) -> int:
        if self._homozygosity_level is not None:
            return self._homozygosity_level
        elif isinstance(self.hla_system, HLASystem):
            self._homozygosity_level = 0
            for nb in self.hla_system.needed_broad_mismatches:
                if len(self.hla_broads[nb]) == 1:
                    self._homozygosity_level += 1
            for ns in self.hla_system.needed_split_mismatches:
                if len(self.hla_splits[ns]) == 1:
                    self._homozygosity_level += 1
            return self._homozygosity_level
        else:
            raise Exception('Cannot retrieve homozygosity level, without hla system.')

    def get_homozygosity_per_locus(self) -> Dict[str, int]:
        if self._homozygosity_per_locus is not None:
            return self._homozygosity_per_locus
        elif isinstance(self.hla_system, HLASystem):
            self._homozygosity_per_locus = dict()
            for nb in self.hla_system.needed_broad_mismatches:
                self._homozygosity_per_locus[nb] = int(len(self.hla_broads[nb]) == 1)
            for ns in self.hla_system.needed_split_mismatches:
                self._homozygosity_per_locus[ns] = int(len(self.hla_splits[ns]) == 1)
            return self._homozygosity_per_locus
        else:
            raise Exception('Cannot retrieve homozygosity level, without hla system.')

    def get_et_mmp(self) -> float:
        """Calculates the ET mismatch probability, under the assumption that
            HLAs, blood types, and unacceptables are independent. This is how
            ET calculates the mismatch probability
        """
        if self._et_mmp:
            return self._et_mmp
        elif isinstance(self.hla_system, HLASystem):
            et_don_mmprobs, et_hla_mmfreqs = self.hla_system.calculate_et_donor_mmps(
                p = self
            )
            self._et_mmp = et_don_mmprobs.get(cn.ET_MMP_SPLIT, et_don_mmprobs.get(cn.ET_MMP_BROAD, 0))
            self.__dict__[cn.ET_HLA_MISMATCHFREQ] = et_hla_mmfreqs.get(
                cn.ET_MMP_SPLIT, et_hla_mmfreqs.get(cn.ET_MMP_BROAD, 0)
            )
            return self._et_mmp
        else:
            raise Exception('Cannot calculate ET-MMP, without hla system.')

    def get_mmp(self) -> float:
        """Calculates the ET mismatch probability, under the assumption that
            HLAs, blood types, and unacceptables are independent. This is how
            ET calculates the mismatch probability
        """
        if self._mmp:
            return self._mmp
        elif isinstance(self.hla_system, HLASystem):
            mmps, prob_max_1mms = self.hla_system.calculate_mismatch_probabilities(
                p = self
            )
            self._mmp = mmps.get(cn.ET_MMP_SPLIT, mmps.get(cn.ET_MMP_BROAD, 0))
            self.__dict__[cn.ET_HLA_MISMATCHFREQ] = prob_max_1mms.get(
                cn.ET_MMP_SPLIT, mmps.get(cn.ET_MMP_BROAD, 0)
            )
            return self._mmp
        else:
            raise Exception('Cannot calculate ET-MMP, without hla system.')

    def get_age_at_update_time(self, t: float) -> int:
        """Return age at time t (in days)"""
        return int((self.age_days_at_listing + t) / 365.25)

    def get_age_at_sim_time(self, s: float) -> int:
        """Return age at time s (in years)"""
        return int((self.age_days_at_listing + s - self.listing_offset) / 365.25)

    def update_etkas_esp_eligibility(self, s: float) -> None:
        """Update a candidate's ETKAS/ESP eligibility.

        1. Candidates aged lt65 for the entire simulation are ETKAS eligible
        2. Candidates in countries who do not enforce ETKAS/ESP choice are ETKAS eligible,
        and ESP eligible if aged 65+
        3. In countries which enforce ETKAS/ESP choice, ESP and ETKAS
           are mutually exclusive
        """
        if self.always_etkas_aged:
            self._etkas_eligible = True
            self._esp_eligible = False
        elif (
            self.country_always_etkas_eligible or
            s <= self.program_choice_required_after
        ):
            self._etkas_eligible = True
            self._esp_eligible = s >= self.sim_time_esp_eligible
        else:
            self._etkas_eligible = (
                s < self.sim_time_esp_eligible or
                (
                    s >= self.sim_time_esp_eligible and
                    self.kidney_program == mr.ETKAS
                )
            )
            self._esp_eligible = not self._etkas_eligible
        self._eligibility_last_assessed = s


    def get_etkas_eligible(self, s: float) -> bool:
        """Return whether the patient is ETKAS eligible."""
        if self.always_etkas_aged or self.country_always_etkas_eligible:
            return True
        elif (
                s and
                (s - self._eligibility_last_assessed) >= es.CHECK_ETKAS_ESP_ELIGIBILITY
            ):
            self.update_etkas_esp_eligibility(s=s)
        return self._etkas_eligible


    def get_esp_eligible(self, s: float) -> bool:
        """Return whether the patient is ESP eligible"""
        if self.always_etkas_aged:
            return False
        elif (
                s and
                (s - self._eligibility_last_assessed) >= es.CHECK_ETKAS_ESP_ELIGIBILITY
            ):
            self.update_etkas_esp_eligibility(s=s)
        return self._esp_eligible


    def get_next_update_time(self) -> Optional[float]:
        """
            Return time at which next update is issued,
            offset from calendar time
        """
        if (
            self.future_statuses is not None and
            not self.future_statuses.is_empty()
        ):
            return(
                self.listing_offset +
                self.future_statuses.first().arrival_time
            )

    def _seed_acceptance_behavior(self, seed: Optional[int]) -> None:
        """Use common random numbers to pre-set patient acceptance behavior"""
        if seed is None:
            self.seed = 1
        elif self.id_registration is None:
            warn(
                f'No registration ID is set. '
                f'Setting seed to the registration ID.'
            )
            self.seed = self.seed
        else:
            self.seed = seed * self.id_registration
        self.rng_acceptance = np.random.default_rng(self.seed)

    def reset_matchrecord_info(self):
        self._needed_match_info = None
        self._offer_inherit_cols = None

    def preload_status_updates(
            self,
            fut_stat: PatientStatusQueue
            ):
        """Initialize status updates for patient"""
        self.initialized = True
        self.__dict__[cn.TIME_LIST_TO_REALEVENT] = (
            fut_stat.return_time_to_exit(
                exit_statuses=es.EXITING_STATUSES
            )
        )
        self.future_statuses = fut_stat

    def do_patient_update(
            self,
            sim_results: Optional[SimResults] = None
            ):
        """Update patient with first next status"""
        if (
            self.future_statuses is not None and
            not self.future_statuses.is_empty()
        ):
            stat_update = self.future_statuses.next()
            current_wt = (
                (stat_update.arrival_time - float(self.last_update_time)) if
                self.last_update_time and self.urgency_code != cn.NT else
                0
            )
            upd_time = stat_update.arrival_time

            if not self.__dict__[cn.ANY_ACTIVE]:
                if not stat_update.before_sim_start:
                    if self.__dict__[cn.URGENCY_CODE] != cn.NT:
                        self.__dict__[cn.ANY_ACTIVE] = True

            # Now reset cached match record information; we no longer need them.
            # Also reset other information, inherited to match records
            self.reset_matchrecord_info()

            # Update the patient with the update status
            if self.sim_set.USE_REAL_FU and stat_update.synthetic:
                pass
            elif stat_update.type_status == mgr.URG:
                if stat_update.status_value == cn.FU:
                    if (
                        self.sim_set.USE_REAL_FU or
                        stat_update.before_sim_start
                    ):
                        self.set_transplanted(
                            self.__dict__[cn.LISTING_DATE] + timedelta(
                                days=stat_update.arrival_time
                                )
                        )
                    elif self.am:
                        self.set_transplanted(
                            self.__dict__[cn.LISTING_DATE] + timedelta(
                                days=stat_update.arrival_time
                            )
                        )
                elif stat_update.status_value in es.TERMINAL_STATUSES:
                    self.future_statuses.clear()
                    self.__dict__[cn.FINAL_REC_URG_AT_TRANSPLANT] = (
                        self.urgency_code
                    )
                    self.set_urgency_code(
                        stat_update.status_value,
                        stat_update.status_detail
                        )
                    self.__dict__[cn.EXIT_STATUS] = stat_update.status_value
                    self.__dict__[cn.EXIT_DATE] = (
                        self.__dict__[cn.LISTING_DATE] +
                        timedelta(days=stat_update.arrival_time)
                    )
                    if sim_results:
                        sim_results.save_exit(self)
                else:
                    self._update_urgency(stat_update)
            elif stat_update.type_status == mgr.DIAG:
                self._update_diag(stat_update)
            elif stat_update.type_status == mgr.HLA:
                self._update_hla(stat_update)
            elif stat_update.type_status == mgr.AM:
                self._update_am(stat_update)
            elif stat_update.type_status == mgr.DIAL:
                self._update_dialysis_status(stat_update)
            elif stat_update.type_status == mgr.UNACC:
                self._update_unacceptables(stat_update)
            elif isinstance(stat_update, ProfileUpdate):
                self.profile = stat_update.profile
                self._other_profile_compatible = None
            else:
                print(
                    f'Stopped early for recipient {self.id_recipient} '
                    f'due to unknown status ({stat_update.type_status}):'
                    f'\n{stat_update.type_status}'
                    )
                exit()

            self.last_update_time = upd_time

    def trigger_historic_updates(
        self
    ):
        """Trigger all updates which occured before sim start"""
        if self.future_statuses is not None:
            while (
                not self.future_statuses.is_empty() and
                (
                    self.future_statuses.first().before_sim_start or
                    (self.future_statuses.first().arrival_time < 0)
                )
            ):
                self.do_patient_update()
            if self.__dict__[cn.URGENCY_CODE] != 'NT':
                self.__dict__[cn.ANY_ACTIVE] = True
            self.__dict__[cn.INIT_URG_CODE] = (
                self.__dict__[cn.URGENCY_CODE]
            )

    def schedule_death(self, fail_date: datetime) -> None:
        """
            Schedule a death update at the fail date. Needed for constructing
            synthetic re-registrations.
        """
        death_event = StatusUpdate(
            type_status=mgr.URG,
            arrival_time=(
                fail_date - self.__dict__[cn.LISTING_DATE]
            ) / timedelta(days=1),
            status_detail='',
            status_value=cn.D,
            sim_start_time=(
                self.__dict__[cn.LISTING_DATE] - self.sim_set.SIM_START_DATE
            ) / timedelta(days=1)
        )
        self.future_statuses.add(
            death_event
        )
        self.future_statuses.truncate_after(
            truncate_time=(
                fail_date - self.__dict__[cn.LISTING_DATE]
            ) / timedelta(days=1)
        )

    def trigger_all_status_update(
            self,
            verbose: bool = False,
            trigger_until = Optional[float]
            ) -> None:
        """Function to trigger all status updates for a patient"""
        if self.future_statuses is not None:
            while not self.future_statuses.is_empty():
                if verbose:
                    print(self.future_statuses.first())
                self.do_patient_update()
                if verbose:
                    last_sim_update_time = self.last_update_time + self.listing_offset
                    print(
                        f'Current age: {self.get_age_at_update_time(self.last_update_time)}, '
                        f'Kidney program: {self.kidney_program}, '
                        f'ESP-eligible: {self.get_esp_eligible(last_sim_update_time)}, '
                        f'ETKAS_eligible: {self.get_etkas_eligible(last_sim_update_time)},\n\t'
                        f'diag: "{self.disease_group}", vpra: {self.vpra}, '
                        f'unacc: {", ".join(self.unacceptables)}, '
                        f'et-mmp: {self.get_et_mmp()}, mmp: {self.__dict__.get(cn.HLA_MISMATCHFREQ, None)}, AM: {self.am}'
                    )
                if self.get_next_update_time() and isinstance(trigger_until, float):
                    if self.get_next_update_time() > trigger_until:
                        break

    def _update_am(self, upd: StatusUpdate):
        self._etkas_eligible = None
        self._esp_eligible = None
        if upd.status_value == 'A':
            self.__dict__[cn.AM_STATUS] = True
            self._am = True
        else:
            self.__dict__[cn.AM_STATUS] = False
            self._am = False

    def _update_diag(self, upd: StatusUpdate):
        """Update downmarked status"""
        self.set_diag(upd)

    def _update_dialysis_status(self, upd: StatusUpdate):
        dt = datetime.strptime(upd.status_value, DEFAULT_DMY_HMS_FORMAT)
        self.set_dial_date(dt)

    def _update_hla(self, upd: StatusUpdate):
        self.__dict__[cn.HLA_MISMATCHFREQ] = float(upd.status_detail)
        self.set_match_hlas(upd.status_value)
        self._et_mmp = None
        self._homozygosity_level = None
        self._homozygosity_per_locus = None

    def _update_unacceptables(self, upd: StatusUpdate):
        if isinstance(upd.status_value, str):
            self.__dict__[cn.VPRA] = float(upd.status_detail) / 100
            self.__dict__[cn.UNACC_ANT] = upd.status_value
        else:
            self.__dict__[cn.UNACC_ANT] = ''
            self.__dict__[cn.VPRA] = 0
        self._unacceptables = None
        self._et_mmp = None

    def _update_urgency(self, upd: StatusUpdate):
        """Update urgency code"""
        assert upd.status_value in es.ALLOWED_STATUSES, \
            f'listing status should be one of:\n\t' \
            f'{", ".join(es.ALLOWED_STATUSES)}, not {upd.status_value}'
        self.set_urgency_code(
            upd.status_value,
            upd.status_detail
        )

        # Track other information
        if self.urgency_code == mr.T:
            self.__dict__[cn.LAST_NONNT_HU] = False
            if not self.__dict__[cn.ANY_ACTIVE]:
                self.__dict__[cn.ANY_ACTIVE] = True
        elif self.urgency_code == cn.HU:
            self.__dict__[cn.LAST_NONNT_HU] = True
            if not self.__dict__[cn.ANY_ACTIVE]:
                self.__dict__[cn.ANY_ACTIVE] = True
            self.__dict__[cn.ANY_HU] = True
            self.hu_since = upd.arrival_time
        else:
            self.hu_since = np.nan

    #def track_match_points(self, points: float):
    #    """Function to track match points with exponential moving average.
    #        Does not really help.
    #    """
    #    if self._previous_ema_matchpoints is None:
    #        self._previous_ema_matchpoints = points
    #    else:
    #        self._previous_ema_matchpoints = (
    #            (1-self._ema_alpha) * self._previous_ema_matchpoints + self._ema_alpha * points
    #        )

    #def retrieve_ema_match_points(self) -> float:
    #    # If an exponential average of match points is available, return it
    #    if self._previous_ema_matchpoints:
    #        return self._previous_ema_matchpoints

    #    # Otherwise, return waiting points at listing
    #    if self.get_dial_time_sim_start():
    #        if (dial_time_listing := self.get_dial_time_sim_start() - self.listing_offset) > 0:
    #            waitpoints =  (dial_time_listing + self.__dict__[cn.PREVIOUS_WT])
    #        else:
    #            waitpoints = self.__dict__[cn.PREVIOUS_WT]
    #    else:
    #        waitpoints = self.__dict__[cn.PREVIOUS_WT]
    #    return waitpoints * self.sim_set.POINTS_ETKAS.get('YEARS_ON_DIAL')

    @property
    def am(self):
        if self._am is None:
            self._am = self.__dict__[cn.AM_STATUS]
        return self._am

    @property
    def dcd_country(self):
        if self._dcd_country is None:
            self._dcd_country = (
                self.__dict__[cn.RECIPIENT_COUNTRY]
                in es.DCD_ACCEPTING_COUNTRIES
            )
        return self._dcd_country

    @property
    def bloodgroup(self):
        if self._bg is None:
            self._bg = self.__dict__[cn.R_BLOODGROUP]
        return self._bg

    @property
    def unacceptables(self):
        if self._unacceptables is None and isinstance(self.hla_system, HLASystem):
                if self.__dict__[cn.UNACC_ANT]:
                    self._unacceptables = self.hla_system.return_all_antigens(
                        self.__dict__[cn.UNACC_ANT]
                    )
                else:
                    self._unacceptables = set()
        return self._unacceptables

    @property
    def needed_match_info(self):
        if not self._needed_match_info:
            self._needed_match_info = {
                k: self.__dict__[k]
                for k
                in es.MTCH_RCRD_PAT_COLS
            }
        return self._needed_match_info

    @property
    def offer_inherit_cols(self):
        if not self._offer_inherit_cols:
            self._offer_inherit_cols = {
                k: self.__dict__[k] for k in es.OFFER_INHERIT_COLS['patient']
            }
        return self._offer_inherit_cols

    @property
    def active(self):
        if self._active is None:
            self._active = (
                self.urgency_code in es.ACTIVE_STATUSES
            )
        return self._active

    def is_initialized(self) -> bool:
        """Whether future statuses have been loaded for the patient"""
        return self.initialized

    def _print_dict(self, dic) -> None:
        """Method to print a dictionary."""
        print(' '.join([f'{k}: {str(v).ljust(4)}\t' for k, v in dic.items()]))

    def __str__(self):
        if self.__dict__[cn.EXIT_STATUS] is not None:
            return(
                f'Patient {self.id_recipient}, exited on '
                f'{self.__dict__[cn.EXIT_DATE]} with status '
                f' {self.__dict__[cn.EXIT_STATUS]} '
                f'(delisting reason: {self.urgency_reason})'
                f' with bloodgroup {self.r_bloodgroup}'
                )

        return(
            f'Patient {self.id_recipient}, listed on '
            f'{self.__dict__[cn.LISTING_DATE].strftime("%d-%m-%Y")} '
            f'at center {self.__dict__[cn.RECIPIENT_CENTER]} with '
            f'current status {self.urgency_code}'
            f' with bloodgroup {self.r_bloodgroup}'
            )


    def __repr__(self):
        return(
            f'Patient {self.id_recipient}, listed on ' \
            f'{self.__dict__[cn.LISTING_DATE].strftime("%d-%m-%Y")}: ' \
            f'at center {self.__dict__[cn.RECIPIENT_CENTER]}' \
            f' with bloodgroup {self.r_bloodgroup}\n'
        )


class Profile:
    """Class which implements an obligation
    ...

    Attributes   #noqa
    ----------


    Methods
    -------
    _check_acceptable(self, don: Donor, mq: Optional: Dict[Tuple[int, ...], str]) -> bool
        Check whether donor is acceptable according to profile.
    """
    __slots__ = [
            'min_age', 'max_age', 'min_weight', 'max_weight', 'hbsag',
            'hcvab', 'hbcab', 'sepsis', 'meningitis', 'malignancy',
            'drug_abuse',  'rescue', 'acceptable_types',
            'euthanasia', 'dcd', 'match_qualities'
            ]

    def __init__(
        self, min_age: int, max_age: int,
        hbsag: bool, hcvab: bool, hbcab: bool,
        sepsis: bool, meningitis: bool,
        malignancy: bool, drug_abuse: bool,
        rescue: bool, euthanasia: bool, dcd: bool,
        match_qualities: Dict[Tuple[int, int, int], bool]
    ) -> None:

        self.min_age = min_age
        self.max_age = max_age
        self.hbsag = hbsag
        self.hcvab = hcvab
        self.hbcab = hbcab
        self.sepsis = sepsis
        self.meningitis = meningitis
        self.malignancy = malignancy
        self.drug_abuse = drug_abuse
        self.rescue = rescue
        self.euthanasia = euthanasia
        self.dcd = dcd
        self.match_qualities = match_qualities


    def _check_acceptable(self, don: Donor, verbose=False) -> bool:
        """Check whether donor is acceptable to patient."""

        don_dict = don.__dict__
        if don_dict[cn.D_DCD] > self.dcd:
            if verbose:
                print('DCD-incompatible')
            return False
        if don_dict[cn.D_AGE] < self.min_age:
            if verbose:
                print(f'{don_dict[cn.D_AGE]} >= {self.min_age}')
            return False
        if don_dict[cn.D_AGE] > self.max_age:
            if verbose:
                print(f'Donor age {don_dict[cn.D_AGE]} > {self.max_age}')
            return False
        if don.__dict__[cn.D_HBSAG] > self.hbsag:
            if verbose:
                print('HBsag incompatible')
            return False
        if don.__dict__[cn.D_HCVAB] > self.hcvab:
            if verbose:
                print('HCVab incompatible')
            return False
        if don.__dict__[cn.D_HBCAB] > self.hbcab:
            if verbose:
                print('HBCab incompatible')
            return False
        if don.sepsis > self.sepsis:
            if verbose:
                print('sepsis incompatible')
            return False
        if don.meningitis > self.meningitis:
            if verbose:
                print('meningitis incompatible')
            return False
        if don.malignancy > self.malignancy:
            if verbose:
                print('malignancy incompatible')
            return False
        if don.rescue > self.rescue:
            if verbose:
                print('rescue incompatible')
            return False
        if don.euthanasia > self.euthanasia:
            if verbose:
                print('Euthanasia incompatible')
            return False

        return True

    def _check_hla_acceptable(self, mq: Optional[Tuple[int, ...]] = None, verbose=False) -> bool:
        if mq is not None and self.match_qualities[mq] == 0:
            if verbose:
                print('HLA-incompatible')
            return False
        elif mq is not None:
            return True

    def __str__(self):
        return str(
            {slot: getattr(self, slot) for slot in self.__slots__}
        )


class HLASystem:
    """Class which implements the HLA system
    ...

    Attributes   #noqa
    ----------
    alleles: Set[str]
        allele codes
    splits: Set[str]
        split HLA codes
    broads: Set[str]
        broad HLA codes
    self.alleles_by_type: Dict[str, Set[str]]
        alleles organized by locus
    self.splits_by_type: Dict[str, Set[str]]
        splits organized by locus
    self.broads_by_type: Dict[str, Set[str]]
        broads organized by locus
    self.broads_to_splits: Dict[str, Dict[str, Set[str]]
        per locus mapping of broads to set of splits.
    self.codes_to_broad: Dict[str, Dict[str, str]]
        per locus mapping of HLA-code to corresponding broad
    self.codes_to_split: Dict[str, Dict[str, str]]
        per locus mapping of HLA-code to corresponding split
    self.unsplittable_broads: Dict[str, Set[str]]
        unsplittable broads organized per locus
    self.loci_zero_mismatch: Set[str]
        loci for full house match
    self.allele_frequencies_split: Dict[str, float]
        allele frequences at split level
    self.allele_frequencies_broad: Dict[str, float]
        allele frequencies at broad level

    Methods
    -------
    get_age_at_t(time: float)
        Return age at time t

    """
    def __init__(
            self,
            sim_set: DotDict
        ):

        self.sim_set = sim_set
        hla_match_table = read_hla_match_table(
            sim_set.PATH_MATCH_TABLE
        )

        hla_match_table.orig_allele = copy(hla_match_table.allele)

        if self.sim_set.RETURN_ALLELES:
            self.return_alleles = True
        else:
            self.return_alleles = False

        for forbidden_character in es.HLA_FORBIDDEN_CHARACTERS:
            hla_match_table.allele.replace(forbidden_character, '', inplace=True, regex=True)
            hla_match_table.split.replace(forbidden_character, '', inplace=True, regex=True)
            hla_match_table.broad.replace(forbidden_character, '', inplace=True, regex=True)

        self.cleaned_alle_to_orig = {allele: orig for allele, orig in zip(
            hla_match_table.allele.str.upper(),
            hla_match_table.orig_allele
            )
        }

        hla_match_table.allele = hla_match_table.allele.str.upper()
        hla_match_table.split = hla_match_table.split.str.upper()
        hla_match_table.broad = hla_match_table.broad.str.upper()

        self.alleles = set(hla_match_table.allele.unique())
        self.splits = set(hla_match_table.split.unique())
        self.broads = set(hla_match_table.broad.unique())

        self.all_match_codes = (
            self.alleles.union(self.splits).union(self.broads).union(mr.PUBLICS)
        )
        self.alleles_by_type = defaultdict(dict)
        self.splits_by_type = defaultdict(dict)
        self.broads_by_type = defaultdict(dict)
        self.broads_to_splits = defaultdict(lambda: defaultdict(dict))
        self.broads_to_alleles = defaultdict(lambda: defaultdict(dict))

        self.loci_zero_mismatch = sim_set.LOCI_ZERO_MISMATCH

        code_to_matchinfo = defaultdict(
            lambda: defaultdict(dict)
        )

        if sim_set.HLAS_TO_LOAD:
            hlas_to_load = sim_set.HLAS_TO_LOAD
        else:
            hlas_to_load = es.hlas_to_load

        # Load from match table the alleles
        for hla_locus, df_sel in hla_match_table.groupby('type'):
            if not hla_locus in hlas_to_load:
                continue
            self.alleles_by_type[hla_locus] = set(df_sel.allele.unique())
            self.splits_by_type[hla_locus] = set(df_sel.split.unique())
            self.broads_by_type[hla_locus] = set(df_sel.broad.unique())
            for broad, d_alleles in df_sel.groupby('broad'):
                self.broads_to_splits[hla_locus][broad] = set(d_alleles.split.unique())
                self.broads_to_alleles[hla_locus][broad] = set(d_alleles.allele.unique())
                for allele, split, broad in d_alleles.loc[:, [cn.ALLELE, cn.SPLIT, cn.BROAD]].to_records(index=False):
                    code_to_matchinfo[cn.ALLELE].update(
                        {
                            allele: (hla_locus, split)
                        }
                    )
                    code_to_matchinfo[cn.SPLIT].update(
                        {
                            allele: (hla_locus, split),
                            split: (hla_locus, split)
                        }
                    )
                    code_to_matchinfo[cn.BROAD].update(
                        {
                            allele: (hla_locus, broad),
                            split: (hla_locus, broad),
                            broad: (hla_locus, broad)
                        }
                    )

        self.codes_to_allele = code_to_matchinfo[cn.ALLELE]
        self.codes_to_broad = code_to_matchinfo[cn.BROAD]
        self.codes_to_split = code_to_matchinfo[cn.SPLIT]
        self.unsplittable_broads = {
            locus: (
                {hla for hla in d.keys() if len(d[hla]) == 1}
                .union(es.UNSPLITTABLE_BROADS)
            ) for locus, d in self.broads_to_splits.items()
        }
        self.splittable_broads = {
            locus: (
                {hla for hla in d.keys() if len(d[hla]) > 1}
                .difference(es.UNSPLITTABLE_BROADS)
            ) for locus, d in self.broads_to_splits.items()
        }

        self.missing_splits = defaultdict(int)

        self.needed_broad_mismatches = (
            set(sim_set.needed_broad_mismatches) if sim_set.needed_broad_mismatches is not None
            else {mr.HLA_A, mr.HLA_B}
        )
        self.needed_split_mismatches = (
            set(sim_set.needed_split_mismatches) if sim_set.needed_split_mismatches is not None
            else {mr.HLA_DR}
        )

        self.mmp_loci_splits = tuple(self.sim_set.get('LOCI_MMP_SPLIT', self.needed_split_mismatches))
        self.mmp_loci_broads = tuple(self.sim_set.get('LOCI_MMP_BROAD', self.needed_broad_mismatches))

        self._needed_mms = None
        self.k_needed_mms = (
            len(self.needed_split_mismatches) +
            len(self.needed_broad_mismatches)
        )
        with open(sim_set.PATH_ALLELE_FREQUENCIES_SPLIT, "r", encoding='utf-8') as file:
            self.allele_frequencies_split: Dict[str, Dict[str, float]] = yaml.load(file, Loader=yaml.FullLoader)
        with open(sim_set.PATH_ALLELE_FREQUENCIES_BROAD, "r", encoding='utf-8') as file:
            self.allele_frequencies_broad: Dict[str, Dict[str, float]] = yaml.load(file, Loader=yaml.FullLoader)
        with open(sim_set.PATH_BG_FREQUENCIES, "r", encoding='utf-8') as file:
            self.bg_frequencies: Dict[str, float] = yaml.load(file, Loader=yaml.FullLoader)

        self._donor_pool_hlas = None


    def return_all_antigens(self, input_string: str) -> Set[str]:
        input_string_upp = input_string.upper()
        for code in input_string_upp.split():
           if code not in self.all_match_codes:
               print(f'{code} not a valid code')
        return set(
            code for code in input_string_upp.split(' ')
            if code in self.all_match_codes
        )

    def return_structured_hla(
            self,
            input_string: str
        ) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Optional[Dict[str, Set[str]]]]:
        if isinstance(input_string, str):
            alleles=defaultdict(set)
            broads=defaultdict(set)
            splits=defaultdict(set)
            for code in input_string.upper().split():
                if code not in self.all_match_codes:
                    self.missing_splits[code] += 1
                else:
                    if code in self.codes_to_broad:
                        locus, broad = self.codes_to_broad[code]
                        broads[locus].add(broad)
                        if broad in es.UNSPLITTABLE_BROADS:
                            splits[locus].add(broad)
                    if code in self.codes_to_split:
                        locus, split = self.codes_to_split[code]
                        if '?' in split:
                            pass
                        elif not split in es.UNSPLITTABLE_SPLITS:
                            splits[locus].add(split)
                    if self.return_alleles and code in self.codes_to_allele:
                        locus, split = self.codes_to_allele[code]
                        alleles[locus].add(code)

            if self.return_alleles:
                return broads, splits, alleles
            else:
                return broads, splits, None

    def order_structured_hla(
            self, input_string: str
    ):

        for fb in es.HLA_FORBIDDEN_CHARACTERS:
            if fb in input_string:
                raise Exception(f'Forbidden character {fb} in donor HLA-input string! ({input_string})')

        broads, splits, alleles = self.return_structured_hla(input_string)
        antigen_sets = defaultdict(list)
        for locus, broads_in_loc in broads.items():
            for broad in broads_in_loc:
                # In case a split is known for the broad, add split and check whether allele is present
                for split in splits[locus]:
                    if self.codes_to_broad[split][1] == broad:
                        if len(matching_alleles := self.broads_to_alleles[locus][broad].intersection(
                            alleles[locus]
                        )) > 0:
                            for allele in matching_alleles:
                                if self.codes_to_split[allele][1] == split:
                                    antigen_sets[locus].append((broad, split, allele))
                        else:
                            antigen_sets[locus].append((broad, split, None))
                # In case no split is known for the broad, still check whether allele is present
                if len(
                    splits[locus].intersection(
                        self.broads_to_splits[locus][broad]
                    )
                ) == 0:
                    if len(
                        (matching_alleles := alleles[locus].intersection(
                            self.broads_to_alleles[locus][broad]
                        ))
                    ) == 0 and (
                        broad not in self.unsplittable_broads[locus]
                    ):
                        antigen_sets[locus].append((broad, None, None))
                    else:
                        for allele in matching_alleles:
                            antigen_sets[locus].append((broad, None, allele))

        # Check whether all antigens were recognized.
        recognized_antigens = {
                locus: set(antigen for antigens in list_of_tuples for antigen in antigens if antigen is not None)
                for locus, list_of_tuples in antigen_sets.items()
        }

        for locus, antigens in recognized_antigens.items():
            missing_antigens = broads[locus].difference(antigens).union(
                splits[locus].difference(antigens)
            )
            if alleles:
                missing_antigens = missing_antigens.union(
                    alleles[locus].difference(antigens)
                )
            if len(missing_antigens) > 0:
                raise Exception(f'The following antigens were not recognized: {missing_antigens}. Structured antigens: {antigen_sets}')

        # Check whether HLA locii are at most 2
        for locus, ant_set in antigen_sets.items():
            if len(ant_set) > 2:
                raise Exception(f'Found for locus {locus} the following antigen sets:\n {antigen_sets}')

        # Sort & convert into a dictionary structure
        sorted_antigens = {
            locus: sorted(antigens) for locus, antigens in antigen_sets.items()
        }
        print(sorted_antigens)

        structured_antigens = {
            f"{locus}_{i+1}_{'broad' if k == 0 else 'split' if k == 1 else 'allele'}": self.cleaned_alle_to_orig[ant] if k == 2 and ant is not None else ant
            for locus, antigen_sets in sorted_antigens.items()
            for i, antigen_set in enumerate(antigen_sets)
            for k, ant in enumerate(antigen_set)
        }

        return structured_antigens


    def _determine_broad_mismatches(
            self,
            d: Donor,
            p: Patient,
            loci: Optional[Set[str]] = None
        ) -> Dict[str, int]:
        return {
            locus: len(hla_values.difference(p.hla_broads[locus]))
            for locus, hla_values in d.hla_broads.items()
            if locus in loci or loci is None
        }


    def _determine_split_mismatches(self, d: Donor, p: Patient, loci: Optional[Set[str]] = None):
        """Determine mismatches at the split level.

        This function falls back to matching at the broad level, if splits are unknown.
        """
        if loci is None:
            loci = d.hla_splits.keys()
        mm = {
            locus: len(hla_values.difference(p.hla_splits[locus]))
            for locus, hla_values in d.hla_splits.items()
            if locus in loci
        }
        for locus in loci:
            if len(d.hla_splits[locus]) < len(d.hla_broads[locus]):
                common_broads = d.hla_broads[locus].intersection(p.hla_broads[locus])
                r_match_hlas = copy(p.hla_broads[locus])
                d_match_hlas = copy(d.hla_broads[locus])
                for cb in common_broads:
                    if cb in self.splittable_broads[locus]:
                        rs = self.broads_to_splits[locus][cb].intersection(p.hla_splits[locus])
                        if rs:
                            ds = self.broads_to_splits[locus][cb].intersection(d.hla_splits[locus])
                            if ds:
                                r_match_hlas.union(rs).remove(cb)
                                d_match_hlas.union(rs).remove(cb)
                mm[locus] = len(d_match_hlas.difference(r_match_hlas))
        return mm

    def determine_mismatches(self, d: Donor, p: Patient):
        mm = {
            **{f'mmb_{k}': v for k, v in self._determine_broad_mismatches(d=d, p=p, loci=self.needed_broad_mismatches).items()},
            **{f'mms_{k}': v for k, v in self._determine_split_mismatches(d=d, p=p, loci=self.needed_split_mismatches).items()}
        }
        if self._needed_mms is None:
            self._needed_mms = set(mm.keys())

        if len(mm) != self.k_needed_mms:
            return None
        else:
            mm[cn.MM_TOTAL] = -sum(mm.values())
            return mm

    def _calc_et_prob1mm(
            self,
            hlas: Dict[str, Set[str]],
            freqs: Dict[str, Dict[str, float]]
    ) -> Tuple[float, float]:
        """ This function calculates the ET HLA-match frequency.
            This is the probability that an arriving donor has at most
            1 HLA mismatch in total on the HLA-A, HLA-B, and HLA-DR loci.

            Note that this HLA mismatch probability is calculated under the
            assumption that HLAs are independently and identically distributed
        """

        # Calculate per locus the probability that a random HLA matches to it.
        prob_match_per_locus = {
            locus: sum(freqs[locus].get(antigen, 1e-4) for antigen in list(antigens))
            for locus, antigens in hlas.items()
        }

        # Probability of 0 mismatches in total
        prob_both_match = {
            locus: prob_match**2 for locus, prob_match in prob_match_per_locus.items()
        }
        MMP0 = 1
        for prob in prob_both_match.values():
            MMP0 = MMP0 * prob

        # Probability of exactly 1 mismatch per locus
        prob_1heterozygous_mismatch = {
            locus: x*(1-x)+(1-x)*x for locus, x in prob_match_per_locus.items()
        }

        # Probability of a homozygous mismatch
        prob_1homozygous_mismatch = {
            locus: sum(
                freq ** 2 * (antigen not in hlas[locus]) for antigen, freq in freqs_per_locus.items()
            ) for locus, freqs_per_locus in freqs.items()
        }

        # Probability that there is no mismatch on other loci
        try:
            prob_no_mismatch_on_other_loci = {
                locus: MMP0 / prob_match**2 for
                locus, prob_match in prob_match_per_locus.items()
            }
        except: # in case of division by zero
            prob_no_mismatch_on_other_loci = {
                locus: prod(v for k, v in prob_match_per_locus.items() if k != locus)**2
                for locus in prob_match_per_locus.keys()
            }

        prob_exactly_one_mismatch = {}
        for locus, prob_nomm_other_loci in prob_no_mismatch_on_other_loci.items():
            prob_exactly_one_mismatch[locus] = (
                prob_nomm_other_loci *
                (prob_1homozygous_mismatch[locus] + prob_1heterozygous_mismatch[locus])
            )

        MMP1 = sum(prob_exactly_one_mismatch.values())

        return MMP0, MMP1

    def _calculate_donor_mismatch_probability(
        self,
        prob_hla_match: float,
        bg: str,
        vpra: float,
        n_donors: int = 1000
    ) -> float:
        """Function which calculates the donor mismatch probability, based on
           a HLA match probability, candidate bloodgroup, and the candidate vPRA.

           This function assumes that unacceptable antigens, a HLA match, and BG
           are independently distributed.
        """
        prob_bg_match = self.bg_frequencies[bg]
        return round_to_decimals(
                100*((1-(prob_bg_match * (1-vpra) * (prob_hla_match)))**n_donors),
                4
            )

    def _calculate_hla_mismatch_frequency(
        self,
        prob_hla_match: float,
        n_donors: int = 1000
    ) -> float:
        return round_to_decimals(
            100*((1-prob_hla_match)**n_donors),
            4
        )


    def _calc_et_donor_mmp_and_matchfreq(
            self,
            bg: str,
            vpra: float,
            n_donors: int = 1000,
            **kwargs
        ) -> Tuple[float, float]:
        """This function calculates the ET donor mismatch probability.
        It returns the ET donor MMP, and ET HLA-mismatch frequency."""

        MMP0, MMP1 = self._calc_et_prob1mm(
            **kwargs
        )
        return (
            self._calculate_donor_mismatch_probability(
                prob_hla_match=MMP0+MMP1,
                bg=bg,
                vpra=vpra,
                n_donors=n_donors
            ),
            self._calculate_hla_mismatch_frequency(
                prob_hla_match = MMP0+MMP1,
                n_donors=n_donors
            )
        )

    def _calc_donor_mmp_and_matchfreq(
            self,
            bg: str,
            vpra: float,
            n_donors: int = 1000,
            hla_mismatchfreq: Optional[float] = None,
            prob_hla_match: Optional[float] = None,
            **kwargs
        ) -> Tuple[float, float]:
        """This function calculates the donor mismatch probability and match frequency,
           without assuming HLAs are iid."""

        if prob_hla_match is None:
            if hla_mismatchfreq is None:
                raise Exception('Cannot calculate mismatch probability when HLA mismatch freq / prob HLA match are unknown')
            else:
                prob_hla_match = 1-((hla_mismatchfreq / 100) ** 1e-3)

        if prob_hla_match is not None:
            return (
                self._calculate_donor_mismatch_probability(
                    prob_hla_match=prob_hla_match,
                    bg=bg,
                    vpra=vpra,
                    n_donors=n_donors
                ),
                self._calculate_hla_mismatch_frequency(
                    prob_hla_match = prob_hla_match,
                    n_donors=n_donors
                )
            )

    def calculate_et_donor_mmps(self, p: Patient) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate ET donor mismatch probabilities and ET donor mismatch frequencies.

        Note that these calculations assume:
            (i) independence between HLA on the loci,
            (ii) independence between vPRA, BG, and HLA
        """
        mmps = dict()

        et_mmp_broad, et_matchfreq_broad = self._calc_et_donor_mmp_and_matchfreq(
            hlas = p.hla_broads,
            bg = p.__dict__[cn.R_BLOODGROUP],
            vpra = p.__dict__.get(cn.VPRA, 0),
            freqs = self.allele_frequencies_broad
        )

        match_hlas = {
            locus: p.hla_splits[locus] for locus in self.mmp_loci_splits
        } | {
            locus: p.hla_broads[locus] for locus in self.mmp_loci_broads
        }

        et_mmp_split, et_matchfreq_split = self._calc_et_donor_mmp_and_matchfreq(
            hlas = match_hlas,
            bg = p.__dict__[cn.R_BLOODGROUP],
            vpra = p.__dict__.get(cn.VPRA, 0),
            freqs = self.allele_frequencies_split
        )

        mmps = {
            cn.ET_MMP_BROAD: et_mmp_broad,
            cn.ET_MMP_SPLIT: et_mmp_split
        }
        match_freqs = {
            cn.ET_MMP_BROAD: et_matchfreq_broad,
            cn.ET_MMP_SPLIT: et_matchfreq_split
        }

        return mmps, match_freqs


    def calculate_vpra_from_string(self, unacc_str: str):
        if isinstance(unacc_str, str):
            unacceptables = self.return_all_antigens(unacc_str)
            return (
                sum(len(unacceptables.intersection(a)) > 0
                for a in self.donor_pool_hlas) / len(self.donor_pool_hlas)
            )
        else:
            return 0

    @property
    def donor_pool_hlas(self):
        if self._donor_pool_hlas is not None:
            return self._donor_pool_hlas
        elif self.sim_set.PATH_DONOR_POOL is not None:
            df_don_pool = rdr.read_donor_pool(self.sim_set.PATH_DONOR_POOL)
            df_don_pool_hlas = df_don_pool.loc[:, [cn.ID_DONOR, cn.DONOR_HLA]].drop_duplicates()
            self._donor_pool_hlas = list(
                self.return_all_antigens(s) for s in df_don_pool_hlas.donor_hla
            )
            return self._donor_pool_hlas
        else:
            raise Exception(
                "Specify a path to a donor pool to calculate vPRAs in the yml file."
            )

class InternationalTransplantation:
    """Class which implements a transplantation to be balanced
    ...

    Attributes   #noqa
    ----------
    transplant_type: str
        import or export
    transplant_value: str
        import (+1) or export (-1)
    txp_time: float
        time of transplantation
    txp_expiry_time: float
        time at which transplantation does not count any more

    Methods
    -------

    """

    def __init__(
        self,
        group: Tuple[str],
        level: str,
        party: str,
        party_center: str,
        transplant_type: str,
        txp_time: Optional[float] = None,
        txp_expiry_time: Optional[float] = None
    ):
        self.group = group
        self.level = level
        self.type = type
        self.party = party if party != mr.LUXEMBOURG else mr.BELGIUM
        if party_center in es.REG_BAL_CENTER_GROUPS:
            self.party_center = es.REG_BAL_CENTER_GROUPS[party_center]
        else:
            self.party_center = party_center
        self.transplant_type = transplant_type
        if transplant_type == mr.EXPORT:
            self.transplant_value = -1
        elif transplant_type == mr.IMPORT:
            self.transplant_value = +1
        else:
            raise Exception(
                f'Transplant type should be {mr.EXPORT} or {mr.IMPORT}'
                f'not {transplant_type}'
            )
        self.txp_time = txp_time
        self.txp_expiry_time = txp_expiry_time

    def __radd__(self, other):
        if isinstance(other, int):
            return self.transplant_value + other
        else:
            return self.transplant_value + other.transplant_value

    def __add__(self, other):
        return self.transplant_value + other.transplant_value

    def __str__(self):
        return f'{self.group}, {self.transplant_type} for {self.party} ' \
               f'at {round(self.txp_time, 1)} which expires at {round(self.txp_expiry_time, 1)} '

    def __repr__(self):
        return f'{self.group}, {self.transplant_type} for {self.party} ' \
               f'at {round(self.txp_time, 1)} which expires at {round(self.txp_expiry_time, 1)} '

    def __lt__(self, other):
        """Sort by expiry time."""
        return self.txp_expiry_time < other.txp_expiry_time


class BalanceSystem:
    """ Balancing system of ETKAS, potentially with a grouping var
    ...

    Attributes   #noqa
    ----------
    verbose: int
        whether to print messages of updates to obligations. 0 prints nothing,
        1 prints some messages, 2 prints all.

    Methods
    -------


    """

    def __init__(
            self,
            nations: Set[str],
            initial_national_balance: Dict[Tuple[str, ...], Dict[str, List[InternationalTransplantation]]],
            group_vars: Optional[Set[str]] = None,
            verbose: int = 1,
            update_balances: bool = True
        ) -> None:

        self._parties = nations
        self.n_parties = len(self._parties)
        self.group_vars = group_vars
        self.verbose = int(verbose)

        self.initial_national_balance = defaultdict(lambda: defaultdict(list))
        self.initial_national_balance.update(initial_national_balance)
        self.national_balances = deepcopy(initial_national_balance)

        self.initial_regional_balances = {
            group + (country,): balances
            for group in self.national_balances.keys()
            for country, balances in self.return_regional_balances(group_values=group).items()
        }

        self.balance_update_interval = 1
        self.time_last_update = -1000
        self._last_nat_bal = None
        self._last_nat_bal_norm = None
        self._last_reg_bal = None
        self._last_reg_bal_norm = None
        self._first_expiry = None
        self.update_balances = update_balances

        self.generated_balances = {
            group_var: {
                tp: [] for tp in obl.keys()
                }
            for group_var, obl in initial_national_balance.items()
        }

    def remove_expired_balance_items(self, current_time: float):
        if current_time > self.first_expiry:
            for balances_by_country in self.national_balances.values():
                for balances in balances_by_country.values():
                    while balances[0].txp_expiry_time < current_time:
                        balances.popleft()
            self._first_expiry = None

    def normalize(self, d: Dict[str, int]):
        max_dict = max(d.values())
        return {
            k: max_dict-v for k, v in d.items()
        }

    def return_national_balances(
            self,
            group_values: Optional[Tuple[str]] = None,
            normalize: bool = False,
            current_time: Optional[float] = 0
            ) -> Dict[str, int]:

        if current_time:
            # Update balance points daily
            if (current_time - self.time_last_update) <= 1:
                if normalize and self._last_nat_bal_norm:
                    return self._last_nat_bal_norm
                elif not normalize and self._last_nat_bal:
                    return self._last_nat_bal
            else:
                self.time_last_update = current_time
                if self.update_balances:
                    self.remove_expired_balance_items(current_time=current_time)

        if group_values is None:
            group = ('1',)
        else:
            group = tuple(group_values)

        national_balances: Dict[str, int] = {
            k: sum(v) for k, v in self.national_balances[group].items()
        }
        for orig_country, dest_country in es.NATION_GROUPS.items():
            if dest_country in national_balances:
                national_balances[orig_country] = national_balances[dest_country]

        self._last_nat_bal_norm = self.normalize(national_balances)
        self._last_nat_bal = national_balances
        if normalize:
            return self._last_nat_bal_norm
        return self._last_nat_bal

    def return_regional_balances(
            self,
            group_values: Optional[Tuple[str]] = None,
            normalize: bool = False,
            current_time: Optional[Union[float, int]] = None
        ) -> Dict[str, Dict[str, int]]:

        if current_time:
            if (current_time - self.time_last_update) <= 1:
                if normalize and self._last_reg_bal_norm:
                    return self._last_reg_bal_norm
                elif not normalize and self._last_reg_bal_norm:
                    return self._last_reg_bal_norm
            else:
                self.time_last_update = current_time

        if group_values is None:
            group = ('1',)
        else:
            group = tuple(group_values)

        regional_balances = defaultdict(lambda: defaultdict(int))
        for cntry in es.COUNTRIES_REGIONAL_BALANCES:

            for b in self.national_balances[group][cntry]:
                regional_balances[b.party][b.party_center] += b.transplant_value

        if normalize:
            expected_balances = {
                cntry: sum(self.national_balances[group][cntry]) / len(regional_balances[cntry])
                for cntry in es.COUNTRIES_REGIONAL_BALANCES
            }
            regional_balances = {
                cntry: {
                    ctr: expected_balances[cntry]-balance
                    for ctr, balance in center_balances.items()
                } for cntry, center_balances in regional_balances.items()
            }

        for cntry, center_dict in regional_balances.items():
            for orig_center, group in es.REG_BAL_CENTER_GROUPS.items():
                if group in center_dict:
                    center_dict.update(
                        {orig_center: center_dict[group]}
                    )

        if normalize:
            self._last_reg_bal_norm = regional_balances
            return self._last_reg_bal_norm
        else:
            self._last_reg_bal = regional_balances
            return self._last_reg_bal


    def __str__(self) -> str:
        """Print the outstanding obligations per bloodgroup"""
        df_obl = self._current_bal_as_df(which_bal='national')
        return df_obl.to_string()

    def _current_bal_as_df(self, which_bal: str) -> pd.DataFrame:
        """Return current obligations as pd.DataFrame"""
        match which_bal:
            case mr.NATIONAL:
                which_bal = self.national_balances
                group_colnames = self.group_vars
                balkey = 'balance'
                df_obl = pd.DataFrame.from_records(
                    [(group, country, sum(v)) for group in which_bal.keys()
                    for country, v in which_bal[group].items()],
                    columns=['group', 'party', balkey]
                )
                if group_colnames:
                    df_obl[list(group_colnames)] = pd.DataFrame(
                        df_obl['group'].tolist(), index=df_obl.index
                    )
                df_obl = df_obl.drop(columns=['group'])
            case mr.REGIONAL:
                balkey='balance'
                if self.group_vars:
                    group_colnames = self.group_vars + (cn.D_COUNTRY,)
                else:
                    group_colnames = ('group', cn.D_COUNTRY,)
                which_bal = {
                    group + (country,): balances
                    for group in self.national_balances.keys()
                    for country, balances in self.return_regional_balances(group_values=group).items()
                }
                df_obl = pd.DataFrame.from_records(
                    [
                        (group, center, balance)
                        for group, centerdict in which_bal.items()
                        for center, balance in centerdict.items()
                    ],
                    columns=['group', cn.D_CENTER, balkey]
                )
                if group_colnames:
                    df_obl[list(group_colnames)] = pd.DataFrame(
                        df_obl['group'].tolist(), index=df_obl.index
                    )
                df_obl = df_obl.drop(columns=['group'])
            case _:
                raise Exception(
                    f'which_bal should be {mr.REGIONAL} or {mr.NATIONAL}'
                    f' not {which_bal}'
                )
        return df_obl

    def add_balance_from_txp(
            self, txp_time: float, expiry_time: float,
            rcrd: Dict[str, Any]
    ):
        if expiry_time < self.first_expiry:
            self._first_expiry = None
        self._add_balance_from_record(
            rcrd=rcrd,
            txp_time=txp_time,
            expiry_time=expiry_time,
            target_dict=self.national_balances,
            group_vars=self.group_vars
        )

    @classmethod
    def _add_balance_from_record(
        cls,
        rcrd: Dict[str, Any],
        txp_time: float,
        expiry_time: float,
        target_dict: DefaultDict[Tuple[str, ...], DefaultDict[str, List[InternationalTransplantation]]],
        group_vars: Optional[Set[str]] = None
    ) -> None:

        if group_vars:
            group = tuple(rcrd[gv] for gv in group_vars)
        else:
            group = ('1',)

        donor_party = (
                rcrd[cn.D_ALLOC_COUNTRY] if rcrd[cn.D_ALLOC_COUNTRY] != mr.LUXEMBOURG
                else mr.BELGIUM
            )
        rec_party = (
            rcrd[cn.RECIPIENT_COUNTRY]
            if rcrd[cn.RECIPIENT_COUNTRY] != mr.LUXEMBOURG
            else mr.BELGIUM
        )

        target_dict[group][donor_party].append(
                InternationalTransplantation(
                    group=group,
                    level=mr.NATIONAL,
                    party=donor_party,
                    party_center=rcrd[cn.D_ALLOC_CENTER],
                    transplant_type=mr.EXPORT,
                    txp_time=txp_time,
                    txp_expiry_time=expiry_time
                )
            )
        target_dict[group][rec_party].append(
            InternationalTransplantation(
                group=group,
                level=mr.NATIONAL,
                party=rec_party,
                party_center=rcrd[cn.RECIPIENT_CENTER],
                transplant_type=mr.IMPORT,
                txp_time=txp_time,
                txp_expiry_time=expiry_time
            )
        )

    @classmethod
    def from_balance_df(
        cls, ss: DotDict, df_init_balances: pd.DataFrame, group_vars: Optional[Set[str]] = None,
        update_balances: bool = True
    ):
        # Add tstart and tstop columns for existing balances
        df_init_balances.loc[:, cn.TSTART] = (
            df_init_balances.loc[:, cn.D_DATE].values - ss.SIM_START_DATE
            ) / timedelta(days=1)
        df_init_balances.loc[:, cn.TSTOP] = (
            df_init_balances.loc[:, cn.TSTART] + ss.WINDOW_FOR_BALANCE
        )

        init_balances = defaultdict(lambda: defaultdict(deque))

        for rcrd in df_init_balances.to_dict(orient='records'):
            cls._add_balance_from_record(
                rcrd,
                target_dict=init_balances,
                group_vars=group_vars,
                txp_time=rcrd[cn.TSTART],
                expiry_time=rcrd[cn.TSTOP]
            )

        return cls(
            nations=es.ET_COUNTRIES,
            initial_national_balance=init_balances,
            verbose=False,
            group_vars=group_vars,
            update_balances=update_balances
        )

    @property
    def parties(self) -> Set[str]:
        """Parties involved in the Exception System"""
        return self._parties

    @property
    def first_expiry(self) -> float:
        """Return first expiry time"""
        if self._first_expiry is None:
            self._first_expiry = min(
                bal_txps[0]
                for balances in self.national_balances.values()
                for bal_txps in balances.values()
                if len(bal_txps) > 0
            ).txp_expiry_time
        if self._first_expiry:
            return self._first_expiry

    # TODO: add code to trim balances, i.e. remove expired balances
