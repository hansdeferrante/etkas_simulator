from typing import List, Callable, Any, Tuple, Dict, Optional, Union, Generator
from math import isnan
import pandas as pd
import numpy as np

import simulator.magic_values.column_names as cn
import simulator.magic_values.column_groups as cg
import simulator.magic_values.etkass_settings as es
import simulator.magic_values.magic_values_rules as mgr
from simulator.code.PostTransplantPredictor import PostTransplantPredictor
from simulator.code.AllocationSystem import MatchList, MatchRecord
from simulator.code.entities import Patient, Donor, HLASystem, BalanceSystem, Profile
from simulator.code.utils import RuleTuple, round_to_decimals
from simulator.code.ScoringFunction import MatchPointFunction
from simulator.magic_values.rules import (
    BG_COMPATIBILITY_DEFAULTS,
    BLOOD_GROUP_COMPATIBILITY_DICT,
    RECIPIENT_ELIGIBILITY_TABLES, DEFAULT_BG_TAB_COLLE,
    BG_COMPATIBILITY_TAB,
    check_etkas_ped_don,
    check_etkas_ped_rec
)


class MatchRecordCurrentETKAS(MatchRecord):
    """Class which implements an match record for current ETLAS
    ...

    Attributes   #noqa
    ----------
    patient: Patient
        Patient
    donor: Donor
        Donor info
    match_date: datetime
        Date that match list is generated.
    """
    check_d_pediatric: Callable = check_etkas_ped_don
    check_r_pediatric: Callable = check_etkas_ped_rec

    et_comp_thresh = None

    @profile
    def __init__(
            self,
            patient: Patient,
            *args,
            **kwargs
        ) -> None:

        # Construct general match records for the patient.
        super(MatchRecordCurrentETKAS, self).__init__(
            patient=patient,
            *args, **kwargs
            )

        bal_system = kwargs['bal_system']
        hla_system = kwargs['hla_system']
        calc_points = kwargs['calc_points']
        initialize_unacceptable_mrs = kwargs.get('initialize_unacceptable_mrs', True)

        if initialize_unacceptable_mrs or self._initialized:

            # Determine whether patient is pediatric.
            self._determine_pediatric()
            self._determine_match_criterium()

            # Determine years the patient has been on dialysis
            if (dial_time := self.patient.get_dial_start_time()) is not None:
                if self.match_time > dial_time:
                    self.__dict__[cn.YEARS_ON_DIAL] = (
                        self.match_time -
                        dial_time
                    ) / 365.25
                else:
                    self.__dict__[cn.YEARS_ON_DIAL] = 0
            else:
                self.__dict__[cn.YEARS_ON_DIAL] = 0

            # Add previously accrued wait-time for re-transplant candidates
            self.__dict__[cn.PREVIOUS_WT] = patient.__dict__[cn.PREVIOUS_WT]

            # If the donor / recipient is 000M, we need to determine
            # whether the donor is fully homozygous, and if so, the
            # homozygosity level of the recipient
            if self.__dict__[cn.ZERO_MISMATCH]:
                self.__dict__[cn.D_FULLY_HOMOZYGOUS] = (
                    self.donor.determine_fully_homozygous(
                        hla_system=hla_system
                        )
                )
                if self.__dict__[cn.D_FULLY_HOMOZYGOUS]:
                    self.__dict__[cn.R_HOMOZYGOSITY_LEVEL] = (
                        self.patient.get_homozygosity_level()
                    )
                else:
                    self.__dict__[cn.R_HOMOZYGOSITY_LEVEL] = 0

            self._determine_match_tier()

            # Calculate mismatch points (approximately, based on loaded frequencies)
            self.__dict__[cn.ET_MMP] = self.patient.get_et_mmp()
            self.add_balance_points(bal_system=bal_system)

            # Calculate match points (as well as components if necessary)
            self.total_match_points = calc_points.calc_score(
                self.__dict__
            )
            if self.store_score_components:
                self.sc = calc_points.calc_score_components(
                    self.__dict__
                )
            else:
                self.sc = None

        else:
            self.__dict__[cn.MTCH_TIER]='B'
            for col in es.DEFAULT_ATTR_ORDER:
                if col not in self.__dict__:
                    self.__dict__[col] = 0

        # Set match tuple
        self._match_tuple = None
        self._mq = None
        self._compatible = None

    def _determine_match_tier(self) -> None:
        if self.__dict__[cn.ZERO_MISMATCH]:
            if self.__dict__[cn.D_FULLY_HOMOZYGOUS]:
                mtch_tier = (
                    cn.D + str(self.__dict__[cn.R_HOMOZYGOSITY_LEVEL])
                )
            else:
                mtch_tier = (
                    cn.D + '0'
                )
        else:
            if self.__dict__[cn.R_PED] and self.__dict__[cn.D_PED]:
                mtch_tier = 'C'
            else:
                mtch_tier = 'B'
        self.__dict__[cn.MTCH_TIER] = mtch_tier

    def add_balance_points(self, bal_system: BalanceSystem) -> None:

        # National balance should always be inserted
        self.__dict__[cn.BALANCE_NAT] = bal_system.return_national_balances(
            normalize=True, current_time = self.match_time
        )[self.__dict__[cn.RECIPIENT_COUNTRY]]

        # Regional balance may be inserted if recipient equals donor
        # country, and if they are in the regional balance countries
        if (
            self.__dict__[cn.RECIPIENT_COUNTRY] in es.COUNTRIES_REGIONAL_BALANCES and
            self.__dict__[cn.D_ALLOC_COUNTRY] in es.COUNTRIES_REGIONAL_BALANCES
        ):
            if (
                reg_bals := bal_system.return_regional_balances(
                    normalize=True, current_time = self.match_time
                )
            ):
                self.__dict__[cn.BALANCE_REG] = (
                    reg_bals[self.__dict__[cn.RECIPIENT_COUNTRY]]
                    [self.__dict__[cn.RECIPIENT_CENTER]]
                )
            else:
                self.__dict__[cn.BALANCE_REG] = 0
        else:
            self.__dict__[cn.BALANCE_REG] = 0


    def _initialize_acceptance_information(self) -> None:

        # Copy over patient and donor information, needed for acceptance.
        self.__dict__.update(self.donor.offer_inherit_cols)
        self.__dict__.update(self.patient.offer_inherit_cols)

        # Information copied over manually
        self.__dict__[cn.D_MALIGNANCY] = self.donor.malignancy
        self.__dict__[cn.D_DRUG_ABUSE] = self.donor.drug_abuse
        self.__dict__[cn.D_MARGINAL] = self.donor.marginal
        self.__dict__[cn.LOCAL_MATCH_CROATIA] = (
            self.__dict__[cn.RECIPIENT_COUNTRY] == mgr.CROATIA and
            self.__dict__[cn.GEOGRAPHY_MATCH] == 'L'
        )

        # Determine match abroad (part of acceptance)
        self._determine_match_abroad()


    def _initialize_posttxp_information(
        self, ptp: PostTransplantPredictor
    ) -> None:

        # Date relative to 2014
        self.__dict__[cn.YEAR_TXP_RT2014] = (
            self.__dict__[cn.MATCH_DATE].year - 2014
        )

        # Time since previous transplant
        if self.patient.__dict__[cn.TIME_SINCE_PREV_TXP] is None:
            self.__dict__[cn.TIME_SINCE_PREV_TXP_CAT] = 'none'
        elif self.patient.__dict__[cn.TIME_SINCE_PREV_TXP] < 14:
            self.__dict__[cn.TIME_SINCE_PREV_TXP_CAT] = '2weeks'
        elif self.patient.__dict__[cn.TIME_SINCE_PREV_TXP] < 90:
            self.__dict__[cn.TIME_SINCE_PREV_TXP_CAT] = '2weeks90days'
        elif self.patient.__dict__[cn.TIME_SINCE_PREV_TXP] < 365:
            self.__dict__[cn.TIME_SINCE_PREV_TXP_CAT] = '90days1year'
        elif self.patient.__dict__[cn.TIME_SINCE_PREV_TXP] < 3*365:
            self.__dict__[cn.TIME_SINCE_PREV_TXP_CAT] = '1to3years'
        else:
            self.__dict__[cn.TIME_SINCE_PREV_TXP_CAT] = 'over3years'

        # self._calculate_posttxp_survival(ptp=ptp)

    def _calculate_posttxp_survival(
        self, ptp: PostTransplantPredictor
    ) -> None:
        for window in es.WINDOWS_TRANSPLANT_PROBS:
            self.__dict__[f'{es.PREFIX_SURVIVAL}_{window}'] = round_to_decimals(
                ptp.calculate_survival(
                    offer=self,
                    time=window
                ),
                3
            )

    def _determine_pediatric(
            self
            ):
        """Determine whether donor/patient are pediatric.
        """

        pat_dict = self.patient.__dict__

        if pat_dict['_pediatric'] or pat_dict['_pediatric'] is None:
            self.__dict__[cn.R_PED] = self.patient.get_pediatric(
                ped_fun = MatchRecordCurrentETKAS.check_r_pediatric,
                match_age = self.__dict__[cn.R_MATCH_AGE]
            )
        else:
            self.__dict__[cn.R_PED] = pat_dict['_pediatric']

        if self.donor.__dict__[cn.D_PED] is None:
            self.__dict__[cn.D_PED] = self.donor.get_pediatric(
                ped_fun = MatchRecordCurrentETKAS.check_d_pediatric
            )
        else:
            self.__dict__[cn.D_PED] = self.donor.__dict__[cn.D_PED]


    def _determine_match_criterium(self):
        """Determine match criterium for allocation"""
        self_dict = self.__dict__
        pat_dict = self.patient.__dict__

        if (
            pat_dict[cn.RECIPIENT_COUNTRY] !=
            self_dict[cn.D_ALLOC_COUNTRY]
        ):
            self_dict[cn.MATCH_CRITERIUM] = cn.INT
            self_dict[cn.GEOGRAPHY_MATCH] = cn.A
            self_dict[cn.ALLOCATION_LOC] = False
            self_dict[cn.ALLOCATION_REG] = False
            self_dict[cn.ALLOCATION_NAT] = False
            self_dict[cn.ALLOCATION_INT] = True
        else:
            if (
                pat_dict[cn.RECIPIENT_CENTER] ==
                self_dict[cn.D_ALLOC_CENTER]
            ):
                self_dict[cn.MATCH_CRITERIUM] = cn.LOC
                self_dict[cn.ALLOCATION_LOC] = True
                if self_dict[cn.D_ALLOC_COUNTRY] == mgr.GERMANY:
                    self_dict[cn.ALLOCATION_REG] = True
                else:
                    self_dict[cn.ALLOCATION_REG] = False
                self_dict[cn.ALLOCATION_NAT] = True
                self_dict[cn.ALLOCATION_INT] = False
            elif (
                pat_dict[cn.RECIPIENT_REGION] ==
                self_dict[cn.D_ALLOC_REGION]
            ):
                self_dict[cn.MATCH_CRITERIUM] = cn.REG
                self_dict[cn.ALLOCATION_LOC] = False
                self_dict[cn.ALLOCATION_REG] = True
                self_dict[cn.ALLOCATION_NAT] = True
                self_dict[cn.ALLOCATION_INT] = False
            elif (
                pat_dict[cn.RECIPIENT_COUNTRY] ==
                self_dict[cn.D_ALLOC_COUNTRY]
            ):
                self_dict[cn.MATCH_CRITERIUM] = cn.NAT
                self_dict[cn.ALLOCATION_LOC] = False
                self_dict[cn.ALLOCATION_REG] = False
                self_dict[cn.ALLOCATION_NAT] = True
                self_dict[cn.ALLOCATION_INT] = False
            if (
                pat_dict[cn.RECIPIENT_CENTER] ==
                self_dict[cn.D_ALLOC_CENTER]
            ):
                self_dict[cn.GEOGRAPHY_MATCH] = cn.L
            elif self_dict[cn.ALLOCATION_REG]:
                self_dict[cn.GEOGRAPHY_MATCH] = cn.R
            elif self_dict[cn.ALLOCATION_NAT]:
                self_dict[cn.GEOGRAPHY_MATCH] = cn.H

    def _determine_rescue_priority(self):
        """Determine which centers have local priority in rescue"""
        if (
            self.__dict__[cn.ALLOCATION_REG] and
            self.__dict__[cn.RECIPIENT_COUNTRY] == mgr.GERMANY
        ):
            self.__dict__[cn.RESCUE_PRIORITY] = 1
        elif (
            self.__dict__[cn.RECIPIENT_COUNTRY] == mgr.BELGIUM
            and self.__dict__[cn.ALLOCATION_LOC]
        ):
            self.__dict__[cn.RESCUE_PRIORITY] = 1
        else:
            self.__dict__[cn.RESCUE_PRIORITY] = 0

    def _determine_match_abroad(self):
        """Determine match abroad"""

        if self.__dict__[cn.GEOGRAPHY_MATCH] == 'L':
            if self.__dict__[cn.RECIPIENT_COUNTRY] == mgr.GERMANY:
                self.__dict__[cn.MATCH_ABROAD] = 'R'
            else:
                self.__dict__[cn.MATCH_ABROAD] = 'H'
        elif self.__dict__[cn.GEOGRAPHY_MATCH] == 'R':
            if self.__dict__[cn.RECIPIENT_COUNTRY] != mgr.GERMANY:
                self.__dict__[cn.MATCH_ABROAD] = 'H'
            else:
                self.__dict__[cn.MATCH_ABROAD] = 'R'
        else:
            self.__dict__[cn.MATCH_ABROAD] = self.__dict__[cn.GEOGRAPHY_MATCH]

    @property
    def match_tuple(self):
        if not self._match_tuple:
            self._match_tuple = self.return_match_tuple()
        return self._match_tuple

    @property
    def mq(self) -> Tuple[int, ...]:
        if self._mq is not None:
            return self._mq
        else:
            self._mq = tuple(
                self.__dict__[k] for k in es.MISMATCH_STR_DEFINITION
            )
        return self._mq

    @property
    def match_quality_compatible(self):
        if self._mq_compatible:
            return self._mq_compatible
        if isinstance(self.patient.profile, Profile):
            self._mq_compatible = self.patient.profile._check_hla_acceptable(
                    mq=self.mq
                )
        else:
            self._mq_compatible = True
        return self._mq_compatible

    @property
    def profile_compatible(self):
        if self.no_unacceptable_antigens is False:
            return False
        elif self.other_profile_compatible is False:
            return False
        elif self.match_quality_compatible is False:
            return False
        else:
            if self.match_quality_compatible and self.other_profile_compatible and self.no_unacceptable_antigens:
                return True
            print(
                f'mq: {self.mq}, nounacc: {self.no_unacceptable_antigens}, opc: {self.other_profile_compatible}, mqc: {self.match_quality_compatible}\n'
                f'unacc: {self.patient.unacceptables}'
            )

    def __lt__(self, other):
        """Youngest obligation first (index then corresponds to age)."""
        return self.match_tuple > other.match_tuple


class MatchListCurrentETKAS(MatchList):
    """Class implementing a match list for the Current ETLAS system.
    ...

    Attributes   #noqa
    ----------
    match_list: List[MatchRecordCurrentETKAS]
        Match list

    Methods
    -------

    """
    def __init__(
            self,
            sort: bool = False,
            record_class = MatchRecordCurrentETKAS,
            store_score_components: bool = False,
            *args,
            **kwargs
    ) -> None:
        super(
            MatchListCurrentETKAS, self
        ).__init__(
            sort=sort,
            record_class=record_class,
            attr_order_match=es.DEFAULT_ATTR_ORDER,
            store_score_components=store_score_components,
            *args,
            **kwargs
            )

        if kwargs.get('initialize_unacceptable_mrs'):
            self.match_list.sort()
        else:
            self.match_list = sorted(
                mr for mr in self.match_list
                if mr.profile_compatible
            )
        self.sorted = True

        for rank, match_record in enumerate(self.match_list):
            if isinstance(match_record, MatchRecordCurrentETKAS):
                match_record.add_patient_rank(rank)


    def _initialize_rescue_priorities(self) -> None:
        for mr in self.match_list:
            mr._determine_rescue_priority()

    def return_match_list(
            self
    ) -> List[MatchRecordCurrentETKAS]:
        return [m for m in self.match_list]

    def return_patient_ids(
        self
    ) -> List[int]:
        """Return matching patients"""
        patients = []
        for matchr in self.return_match_list():
            if isinstance(matchr, MatchRecordCurrentETKAS):
                patients.append(matchr.patient.id_recipient)
        return patients

    def return_match_info(
        self
    ) -> List[Dict[str, Any]]:
        """Return match lists"""
        return [
            matchr.return_match_info() for matchr in self.match_list
            ]

    def return_match_df(
            self
            ) -> Optional[pd.DataFrame]:
        """Print match list as DataFrame"""
        if self.match_list is not None:
            matchl = pd.DataFrame.from_records(
                [matchr.return_match_info() for matchr in self.match_list],
                columns=es.MATCH_INFO_COLS
            )
            matchl[cn.ID_DONOR] = matchl[cn.ID_DONOR].astype('Int64')
            matchl[cn.ID_RECIPIENT] = matchl[cn.ID_RECIPIENT].astype('Int64')

            return matchl
