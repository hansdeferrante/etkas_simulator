#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:33:44 2022

@author: H.C. de Ferrante
"""

from datetime import timedelta, datetime
from typing import List, Any, Dict, Optional, Type, Tuple, Union, Generator
from itertools import count
import pandas as pd
import numpy as np

from simulator.code.utils import round_to_decimals, round_to_int
from simulator.code.entities import Patient, Donor, HLASystem, BalanceSystem, Profile
import simulator.magic_values.column_names as cn
import simulator.magic_values.column_groups as cg
import simulator.magic_values.etkass_settings as es
import simulator.magic_values.magic_values_rules as mgr

from simulator.code.ScoringFunction import MatchPointFunction
from simulator.magic_values.rules import (
    FL_CENTER_CODES, DICT_CENTERS_TO_REGIONS
    )


class MatchRecord:
    """Class which implements an MatchList
    ...

    Attributes   #noqa
    ----------
    patient: Patient
        Patient
    donor: Donor
        Donor info
    match_date: datetime
        Date that match list is generated.

    Methods
    -------

    """

    _ids = count(0)

    def __init__(
            self, patient: Patient, donor: Donor,
            match_date: datetime,
            type_offer_detailed: int,
            alloc_center: str,
            calc_points: MatchPointFunction,
            alloc_region: Optional[str],
            alloc_country: str,
            hla_system: HLASystem,
            bal_system: BalanceSystem,
            match_time: float,
            store_score_components: bool = False,
            attr_order_match: Optional[Tuple[str]] = None,
            id_mtr: Optional[int] = None,
            center_travel_times: Optional[
                Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, str]]]
            ] = None
            ) -> None:

        if attr_order_match is None:
            self.attr_order_match = (cn.TOTAL_MATCH_POINTS,)
        else:
            self.attr_order_match = attr_order_match

        self.id_mr = next(self._ids)
        self.id_mtr = id_mtr
        self.__dict__[cn.MATCH_DATE] = match_date
        self.match_time = match_time
        self.patient = patient
        self.donor = donor
        self.store_score_components = store_score_components
        self._other_profile_compatible = None
        self._mq_compatible = None
        self._no_unacceptables = None
        self.center_travel_times = center_travel_times

        # Copy over selected attributes from patient and donor
        self.__dict__.update(
            {
                **self.patient.needed_match_info,
                **self.donor.needed_match_info
            }
        )

        self.__dict__[cn.TYPE_OFFER_DETAILED] = type_offer_detailed

        # Add match list geography.
        self.__dict__[cn.D_ALLOC_CENTER] = alloc_center
        self.__dict__[cn.D_ALLOC_REGION] = alloc_region
        self.__dict__[cn.D_ALLOC_COUNTRY] = alloc_country

        # Add match age
        self.__dict__[cn.R_MATCH_AGE] = (
            (self.patient.age_days_at_listing + self.match_time - self.patient.listing_offset) /
            365.25
        )

        # Match distances
        self._alloc_nat = None
        self._alloc_reg = None

        # Match points and match tuple
        self.total_match_points = 0
        self._match_tuple = None

        # HLA matches
        hlas = hla_system.determine_mismatches(
                d=self.donor,
                p=self.patient
            )
        try:
            self.__dict__.update(
                hlas
            )
        except:
            raise Exception(
                f'Cannot calculate HLAs for donor {self.donor.id_donor} and {self.patient.id_recipient}\n'
                f'with donor hla: {self.donor.hla_broads}\n'
                f'with pat.  hla: {self.patient.hla_broads}'
            )
            exit()


        for k in hla_system.loci_zero_mismatch:
            if self.__dict__.get(k) != 0:
                self.__dict__[cn.ZERO_MISMATCH] = False
                break
        else:
            self.__dict__[cn.ZERO_MISMATCH] = True


    @property
    def match_tuple(self):
        if not self._match_tuple:
            self._match_tuple = self.return_match_tuple()
        return self._match_tuple

    def return_match_info(
        self, cols: Optional[Tuple[str, ...]] = None
    ) -> Dict[str, Any]:
        """Return relevant match information"""
        if cols is None:
            cols = es.MATCH_INFO_COLS
        result_dict = {}

        for key in cols:
            if key in self.__dict__:
                result_dict[key] = self.__dict__[key]
            elif key in self.donor.__dict__:
                result_dict[key] = self.donor.__dict__[key]
            elif key in self.patient.__dict__:
                result_dict[key] = self.patient.__dict__[key]

        return result_dict

    def return_match_tuple(self):
        """Return a match tuple"""
        return tuple(
                self.__dict__[attr] for attr in self.attr_order_match
        )

    def add_patient_rank(self, rnk: int) -> None:
        """
        Add patient rank in current sorted list,
        and add it as tie-breaker.
        """
        self.__dict__[cn.PATIENT_RANK] = int(rnk+1)
        self.attr_order_match += (cn.PATIENT_RANK, )

    def set_acceptance(self, reason: str):
        if reason in cg.ACCEPTANCE_CODES:
            self.__dict__[cn.ACCEPTANCE_REASON] = reason
            if reason == cn.T1 or reason == cn.T3:
                self.__dict__[cn.ACCEPTED] = 1
            else:
                self.__dict__[cn.ACCEPTED] = 0
        else:
            raise ValueError(
                f'{reason} is not a valid acceptance reason.'
            )

    def determine_mismatch_string(self):
        mms = (
            str(
                self.__dict__.get(
                    es.MATCH_TO_SPLITS[loc],
                    self.__dict__.get(es.MATCH_TO_BROADS[loc], ' ')
                )
            ) for loc in mgr.HLA_LOCI)
        return(''.join(mms).rstrip(' '))

    def _determine_match_distance(self):
        """Determine match criterium for allocation"""
        self_dict = self.__dict__
        pat_dict = self.patient.__dict__
        if (
            pat_dict[cn.RECIPIENT_COUNTRY] !=
            self_dict[cn.D_ALLOC_COUNTRY]
        ):
            self_dict[cn.GEOGRAPHY_MATCH] = cn.A
            self_dict[cn.MATCH_DISTANCE] = cn.I
        else:
            # Determine match criteria & geography in case of
            # national allocation
            if (
                pat_dict[cn.RECIPIENT_CENTER] ==
                self_dict[cn.D_ALLOC_CENTER]
            ):
                self_dict[cn.GEOGRAPHY_MATCH] = cn.L
            elif self.allocation_regional:
                self_dict[cn.GEOGRAPHY_MATCH] = cn.R
            elif self.allocation_national:
                self_dict[cn.GEOGRAPHY_MATCH] = cn.H

            # Determine the match distances
            if (
                self_dict[cn.D_ALLOC_COUNTRY] == mgr.GERMANY or
                self_dict[cn.D_ALLOC_COUNTRY] == mgr.AUSTRIA or
                self_dict[cn.D_ALLOC_COUNTRY] == mgr.BELGIUM
            ):
                if (
                    pat_dict[cn.RECIPIENT_REGION] ==
                    self_dict[cn.D_ALLOC_REGION]
                ):
                    self_dict[cn.MATCH_DISTANCE] = cn.L
                else:
                    self_dict[cn.MATCH_DISTANCE] = cn.N
            else:
                self_dict[cn.MATCH_DISTANCE] = cn.L

        # Convert to information necessary for allocation
        self_dict[cn.MATCH_LOCAL] = 1 if self_dict[cn.MATCH_DISTANCE] == cn.L else 0
        self_dict[cn.MATCH_NATIONAL] = 1 if self_dict[cn.MATCH_DISTANCE] == cn.N else 0
        self_dict[cn.MATCH_INTERNATIONAL] = 1 if self_dict[cn.MATCH_DISTANCE] == cn.I else 0

    def __repr__(self):
        return(
            f'{self.determine_mismatch_string()} offer to '
            f'{self.__dict__[cn.ID_RECIPIENT] } '
            f'({self.__dict__[cn.RECIPIENT_CENTER]}) '
            f'from {self.__dict__[cn.D_ALLOC_CENTER] } '
            f'with {self.total_match_points} points with '
            f'date first dial: {self.patient.date_first_dial}'
        )

    def __str__(self):
        """Match record"""
        if not self.store_score_components:
            points_str = f'{str(self.total_match_points).rjust(4, " ")} total match points'
        else:
            points_str = (
                f'{self.total_match_points}pts:\n\t' + ' '.join(
                    f"{v} {str(k).removeprefix('wfmr_xc_').upper()}".ljust(12)
                    for k, v in self.sc.items()
                )
            )
        try:
            fdial = self.patient.date_first_dial.strftime("%Y-%m-%d")
        except:
            fdial = 'none'
        if cn.MTCH_TIER in self.__dict__:
            tier_str = f' in tier {self.__dict__[cn.MTCH_TIER].ljust(2, " ")}'
        else:
            tier_str = ''
        accr = self.__dict__.get(cn.ACCEPTANCE_REASON) if self.__dict__.get(cn.ACCEPTANCE_REASON) else ""
        return(
            f'{self.determine_mismatch_string()} offer{tier_str} to '
            f'{str(self.__dict__[cn.ID_RECIPIENT]).rjust(6, "0")} ({self.__dict__[cn.RECIPIENT_CENTER]}) '
            f'on {self.date_match.strftime("%Y-%m-%d")} '
            f'from {self.__dict__[cn.D_ALLOC_CENTER] } '
            f'with date first dial: {fdial} '
            f'with {points_str} ({type(self).__name__}, {accr})'
        )

    def __lt__(self, other):
        """Order by match tuple."""
        return self.match_tuple > other.match_tuple

    def _determine_allocation_national(self):
        """Determine whether allocation is regional"""
        self_dict = self.__dict__
        pat_dict = self.patient.__dict__

        if (
            self_dict[cn.D_ALLOC_COUNTRY] == pat_dict[cn.RECIPIENT_COUNTRY]
        ):
            return True
        else:
            return False

    def _determine_allocation_regional(self):
        """Determine whether allocation is regional"""
        self_dict = self.__dict__
        pat_dict = self.patient.__dict__

        if (
            self.allocation_national and
            (
                self_dict[cn.D_ALLOC_COUNTRY] == mgr.GERMANY or
                self_dict[cn.D_ALLOC_COUNTRY] == mgr.AUSTRIA or
                self_dict[cn.D_ALLOC_COUNTRY] == mgr.BELGIUM
            )
        ):
            if (
                pat_dict[cn.RECIPIENT_REGION] ==
                   self_dict[cn.D_ALLOC_REGION]
                ):
                    return True
            else:
                return False
        else:
            return False

    @property
    def allocation_national(self):
        if self._alloc_nat is None:
            self._alloc_nat = self._determine_allocation_national()
        return self._alloc_nat

    @property
    def allocation_regional(self):
        if self._alloc_reg is None:
            self._alloc_reg = self._determine_allocation_regional()
        return self._alloc_reg

    @property
    def other_profile_compatible(self):
        if self._other_profile_compatible is not None:
            return self._other_profile_compatible
        if isinstance(self.patient.profile, Profile):
            self._other_profile_compatible = self.patient.profile._check_acceptable(
                    self.donor
                )
        else:
            self._other_profile_compatible = True
        self.__dict__[cn.PROFILE_COMPATIBLE] = self._other_profile_compatible
        return self._other_profile_compatible

    @property
    def no_unacceptable_antigens(self):
        if self._no_unacceptables is not None:
            return self._no_unacceptables
        else:
            if self.patient.unacceptables is not None:
                self._no_unacceptables = self.patient.unacceptables.isdisjoint(
                    self.donor.all_antigens
                )
            else:
                self._no_unacceptables = True
        return self._no_unacceptables


class MatchList:
    """Class which implements an MatchList
    ...

    Attributes   #noqa
    ----------
    donor: Donor
        Donor that is on offer
    match_date: datetime
        Date that match list is generated.
    match_list: List[MatchRecordCurrentETKAS]
        Match list, consisting of list of match records

    Methods
    -------

    """

    _ids = count(0)

    def __init__(
            self, patients: Generator[Patient, None, None], donor: Donor,
            match_date: datetime,
            hla_system: HLASystem,
            bal_system: BalanceSystem,
            calc_points: MatchPointFunction,
            sim_start_date: datetime,
            type_offer: int = 1,
            alloc_center: Optional[str] = None,
            record_class: Type[MatchRecord] = MatchRecord,
            sort: bool = True,
            store_score_components: bool = False,
            attr_order_match: Optional[List[str]] = None,
            travel_time_dict: Optional[Dict[str, Any]] = None
            ) -> None:

        self.__dict__[cn.MATCH_DATE] = match_date
        if sim_start_date is not None:
            self.match_time = (
                (match_date-sim_start_date) /
                timedelta(days=1)
            )
        else:
            self.match_time = 0

        self.id_mtr = next(self._ids)
        self.donor = donor
        if not isinstance(match_date, datetime):
            raise TypeError(
                f'match_date must be datetime,'
                f'not a {type(match_date)}'
                )

        alloc_center = (
            alloc_center if alloc_center is not None
            else donor.donor_center
        )
        alloc_region = DICT_CENTERS_TO_REGIONS.get(
            alloc_center,
            None
            )
        alloc_country = FL_CENTER_CODES[alloc_center[0]]
        self.__dict__[cn.D_ALLOC_COUNTRY] = alloc_country

        if (alloc_country == mgr.GERMANY and alloc_region is None):
            raise Exception(
                f'No region defined for German center: {alloc_center}'
            )

        if travel_time_dict:
            self.center_travel_times = travel_time_dict[
                self.donor.__dict__[cn.DONOR_CENTER]
            ]
        else:
            self.center_travel_times = None


        self.match_list = [
                record_class(
                    patient=pat, donor=donor,
                    match_date=match_date,
                    type_offer_detailed=type_offer,
                    alloc_center=alloc_center,
                    alloc_region=alloc_region,
                    alloc_country=alloc_country,
                    match_time=self.match_time,
                    id_mtr=self.id_mtr,
                    hla_system=hla_system,
                    bal_system=bal_system,
                    attr_order_match=attr_order_match,
                    calc_points=calc_points,
                    store_score_components=store_score_components,
                    center_travel_times=self.center_travel_times
                ) for pat in patients
            ]

        if sort:
            self.sorted = True
            self.match_list.sort()
        else:
            self.sorted = False

        # Type offer
        self.__dict__[cn.TYPE_OFFER_DETAILED] = type_offer

    def is_empty(self) -> bool:
        """Check if event is empty"""
        return len(self.match_list) == 0

    def return_match_list(
            self
    ) -> List[MatchRecord]:
        if self.sorted:
            return [m for m in self.match_list]
        else:
            raise Exception("Cannot return a match list that is not sorted!")

    def return_match_info(
        self
    ) -> List[Dict[str, Any]]:
        """Return match lists"""
        return [
            matchr.return_match_info() for matchr in self.match_list
            ]

    def return_match_df(self) -> Optional[pd.DataFrame]:
        """Print match list as DataFrame"""
        if self.match_list is not None:
            return pd.DataFrame.from_records(
                [mr.return_match_info() for mr in self.match_list],
                columns=es.MATCH_INFO_COLS
            )

    def print_match_list(self) -> None:
        """Print match list as DataFrame"""
        print(self.return_match_df())

    def __str__(self) -> str:
        string = ''
        for evnt in sorted(self.match_list):
            string += str(evnt) + '\n'
        return string

    def __repr__(self):
        """Print the match list"""
        string = ''
        for evnt in sorted(self.match_list):
            string += str(evnt) + '\n'
        return string

    def __len__(self):
        return len(self.match_list)


