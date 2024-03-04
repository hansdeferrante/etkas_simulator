from typing import List, Callable, Any, Tuple, Dict, Optional, Union, Generator
from math import isnan
from datetime import datetime
import pandas as pd
import numpy as np

import simulator.magic_values.column_names as cn
import simulator.magic_values.column_groups as cg
import simulator.magic_values.etkass_settings as es
import simulator.magic_values.magic_values_rules as mgr
from simulator.code.PostTransplantPredictor import PostTransplantPredictor
from simulator.code.AllocationSystem import MatchList, MatchRecord
from simulator.code.entities import Patient, Donor, HLASystem, BalanceSystem, Profile
from simulator.code.utils import round_to_decimals
from simulator.code.ScoringFunction import MatchPointFunction
from simulator.magic_values.rules import (
    DICT_ESP_SUBREGIONS,
    check_etkas_ped_don,
    check_etkas_ped_rec
)


class PatientMatchRecord:
    """Class to calculate points for patient items only
    """
    def __init__(
            self, patient: Patient,
            match_time: float,
            calc_points: MatchPointFunction,
            bal_system: BalanceSystem
            ) -> None:

        self.id_reg = patient.id_registration

        # National balance points
        self.__dict__[cn.BALANCE_NAT] = bal_system.return_national_balances(
            normalize=True, current_time = match_time,
        )[patient.__dict__[cn.RECIPIENT_COUNTRY]]

        # Determine years the patient has been on dialysis
        if (dial_time := patient.get_dial_time_sim_start()) is not None:
            if match_time > dial_time:
                self.__dict__[cn.YEARS_ON_DIAL] = (
                    match_time -
                    dial_time
                ) / 365.25
            else:
                self.__dict__[cn.YEARS_ON_DIAL] = 0
        else:
            self.__dict__[cn.YEARS_ON_DIAL] = 0

        # ET MMP
        self.__dict__[cn.ET_MMP] = patient.get_et_mmp()

        # Add previously accrued wait-time for re-transplant candidates
        self.__dict__[cn.PREVIOUS_WT] = patient.__dict__[cn.PREVIOUS_WT]
        self.__dict__[cn.R_PED] = patient.__dict__.get('_pediatric', False)

        self.total_match_points = calc_points.calc_patient_score(
            self.__dict__
        )

    def __str__(self):
        return f'Patient {self.id_reg} with {self.total_match_points} pts'


class MatchRecordCurrentETKAS(MatchRecord):
    """Class which implements an match record for current ETKAS
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

        self.__dict__[cn.TYPE_RECORD] = 'ETKAS'

        bal_system = kwargs['bal_system']
        hla_system = kwargs['hla_system']
        calc_points = kwargs['calc_points']


        # Determine whether patient is pediatric.
        self._determine_pediatric()
        self._determine_match_distance()

        # Determine years the patient has been on dialysis
        if (dial_time := self.patient.get_dial_time_sim_start()) is not None:
            if self.match_time > dial_time:
                self.__dict__[cn.YEARS_ON_DIAL] = (
                    self.match_time -
                    dial_time
                ) / 365.25
                self.__dict__[cn.ON_DIAL] = 1
            else:
                self.__dict__[cn.YEARS_ON_DIAL] = 0
                self.__dict__[cn.ON_DIAL] = 0
        else:
            self.__dict__[cn.YEARS_ON_DIAL] = 0
            self.__dict__[cn.ON_DIAL] = 0

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
        if self.store_score_components:
            self.sc = calc_points.calc_score_components(
                self.__dict__
            )
            self.total_match_points = sum(
                self.sc.values()
            )
            self.__dict__.update(self.sc)
        else:
            self.total_match_points = calc_points.calc_score(
                self.__dict__
            )
            self.sc = None

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

        # Initialize travel times
        if self.center_travel_times:
            self.__dict__.update(
                self.center_travel_times[
                    self.__dict__[cn.RECIPIENT_CENTER]
                ]
            )

        # Copy over patient and donor information, needed for acceptance.
        self.__dict__.update(self.donor.offer_inherit_cols)
        self.__dict__.update(self.patient.offer_inherit_cols)

        # Information copied over manually
        self.__dict__[cn.D_MALIGNANCY] = self.donor.malignancy
        self.__dict__[cn.D_DRUG_ABUSE] = self.donor.drug_abuse
        self.__dict__[cn.VPRA_PERCENT] = self.__dict__[cn.VPRA]*100

        # Information
        self.__dict__[cn.DONOR_AGE_75P] = int(
            self.__dict__[cn.D_AGE] >= 75
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

        self._calculate_posttxp_survival(ptp=ptp)

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


    def _determine_extalloc_priority(self):
        """Prioritize local allocation in case of extended allocation."""
        if (
            self.__dict__[cn.RECIPIENT_COUNTRY] == mgr.GERMANY
        ):
            if self.allocation_regional:
                self.__dict__[cn.EXT_ALLOC_PRIORITY] = 1
            else:
                self.__dict__[cn.EXT_ALLOC_PRIORITY] = 0
        elif self.allocation_national:
            self.__dict__[cn.EXT_ALLOC_PRIORITY] = 1
        else:
            self.__dict__[cn.EXT_ALLOC_PRIORITY] = 0


    def _determine_match_abroad(self):
        """Determine match abroad"""
        self.__dict__[cn.MATCH_ABROAD]: int = (
            0 if (
                self.__dict__[cn.PATIENT_COUNTRY] ==
                self.__dict__[cn.DONOR_COUNTRY]
            )
            else 1
        )

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
    """Class implementing a match list for the Current ETKAS system.
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
            travel_time_dict: Optional[Dict[str, Any]] = None,
            *args,
            **kwargs
    ) -> None:
        super(
            MatchListCurrentETKAS, self
        ).__init__(
            sort=sort,
            record_class=record_class,
            attr_order_match=es.DEFAULT_ETKAS_ATTR_ORDER,
            store_score_components=store_score_components,
            travel_time_dict=travel_time_dict,
            *args,
            **kwargs
            )

        self.match_list.sort()
        self.sorted = True

        self.ext_alloc_priority = False

        for rank, match_record in enumerate(self.match_list):
            if isinstance(match_record, MatchRecordCurrentETKAS):
                match_record.add_patient_rank(rank)


    def _initialize_extalloc_priorities(self) -> None:
        self.ext_alloc_priority = True
        for mr in self.match_list:
            if isinstance(mr, MatchRecordCurrentETKAS):
                mr._determine_extalloc_priority()

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


class MatchRecordCurrentESP(MatchRecord):
    """Class which implements an match record for current ESP
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
    def __init__(
            self,
            patient: Patient,
            *args,
            **kwargs
        ) -> None:

        # Construct general match records for the patient.
        super(MatchRecordCurrentESP, self).__init__(
            patient=patient,
            *args, **kwargs
            )

        self.__dict__[cn.TYPE_RECORD] = 'ESP'

        self._determine_match_distance()

        # Determine years the patient has been on dialysis
        if (dial_time := self.patient.get_dial_time_sim_start()) is not None:
            if self.match_time > dial_time:
                self.__dict__[cn.YEARS_ON_DIAL] = (
                    self.match_time -
                    dial_time
                ) / 365.25
                self.__dict__[cn.ON_DIAL] = 1
            else:
                self.__dict__[cn.YEARS_ON_DIAL] = 0
                self.__dict__[cn.ON_DIAL] = 0
        else:
            self.__dict__[cn.YEARS_ON_DIAL] = 0
            self.__dict__[cn.ON_DIAL] = 0

        # Add previously accrued wait-time for re-transplant candidates
        self.__dict__[cn.PREVIOUS_WT] = patient.__dict__[cn.PREVIOUS_WT]

        # Calculate match points (as well as components if necessary)
        calc_points = kwargs['calc_points']
        if self.store_score_components:
            self.sc = calc_points.calc_score_components(
                self.__dict__
            )
            self.total_match_points = sum(
                self.sc.values()
            )
            self.__dict__.update(self.sc)
        else:
            self.total_match_points = calc_points.calc_score(
                self.__dict__
            )
            self.sc = None

        self.determine_esp_priority()

        # Set match tuple
        self._match_tuple = None
        self._mq = None
        self._compatible = None

    def _initialize_acceptance_information(self) -> None:

        # Initialize travel times to recipient centers
        if self.center_travel_times:
            self.__dict__.update(
                self.center_travel_times[
                    self.__dict__[cn.RECIPIENT_CENTER]
                ]
            )

        # Copy over patient and donor information, needed for acceptance.
        self.__dict__.update(self.donor.offer_inherit_cols)
        self.__dict__.update(self.patient.offer_inherit_cols)

        # Information copied over manually
        self.__dict__[cn.D_MALIGNANCY] = self.donor.malignancy
        self.__dict__[cn.D_DRUG_ABUSE] = self.donor.drug_abuse
        self.__dict__[cn.VPRA_PERCENT] = self.__dict__[cn.VPRA]*100

        # Information
        self.__dict__[cn.DONOR_AGE_75P] = int(
            self.__dict__[cn.D_AGE] >= 75
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

    def determine_esp_priority(self):
        donor_country = self.__dict__[cn.D_ALLOC_COUNTRY]
        if donor_country == mgr.AUSTRIA:
            if self.__dict__[cn.MATCH_LOCAL]:
                self.__dict__[cn.ESP_PRIORITY] = 1
            else:
                self.__dict__[cn.ESP_PRIORITY] = -1
        elif donor_country == mgr.GERMANY:
            if self._same_esp_subregion():
                self.__dict__[cn.ESP_PRIORITY] = 1
            elif self.allocation_regional:
                self.__dict__[cn.ESP_PRIORITY] = 0
            else:
                self.__dict__[cn.ESP_PRIORITY] = -1
        elif donor_country in {mgr.BELGIUM, mgr.HUNGARY}:
            if self.__dict__[cn.MATCH_LOCAL]:
                self.__dict__[cn.ESP_PRIORITY] = 1
            elif self.__dict__[cn.MATCH_NATIONAL]:
                self.__dict__[cn.ESP_PRIORITY] = 0
            else:
                self.__dict__[cn.ESP_PRIORITY] = -1
        else:
            if self.__dict__[cn.MATCH_NATIONAL]:
                self.__dict__[cn.ESP_PRIORITY] = 1
            else:
                self.__dict__[cn.ESP_PRIORITY] = -1

    def _same_esp_subregion(self):
        if (
            self.__dict__[cn.D_ALLOC_COUNTRY] == mgr.GERMANY and
            self.allocation_national
        ):
            d_ctr = self.__dict__[cn.D_ALLOC_CENTER]
            r_ctr = self.__dict__[cn.RECIPIENT_CENTER]

            if (
                r_ctr in DICT_ESP_SUBREGIONS and
                d_ctr in DICT_ESP_SUBREGIONS
            ):
                if DICT_ESP_SUBREGIONS[r_ctr] == DICT_ESP_SUBREGIONS[r_ctr]:
                    return True
        return False


    def _determine_match_abroad(self):
        """Determine match abroad"""

        if self.__dict__[cn.GEOGRAPHY_MATCH] == 'A':
            self.__dict__[cn.MATCH_ABROAD] = 1
        else:
            self.__dict__[cn.MATCH_ABROAD] = 0

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


class MatchListESP(MatchList):
    """Class implementing a match list for the Current ETKAS system.
    ...

    Attributes   #noqa
    ----------
    match_list: List[MatchRecordCurrentESP]
        Match list

    Methods
    -------

    """
    def __init__(
            self,
            sort: bool = False,
            record_class = MatchRecordCurrentESP,
            store_score_components: bool = False,
            travel_time_dict: Optional[Dict[str, Any]] = None,
            *args,
            **kwargs
    ) -> None:
        super(
            MatchListESP, self
        ).__init__(
            sort=sort,
            record_class=record_class,
            attr_order_match=es.DEFAULT_ESP_ATTR_ORDER,
            store_score_components=store_score_components,
            travel_time_dict=travel_time_dict,
            *args,
            **kwargs
            )

        # Filter to national candidates
        self.match_list = list(
            mr for mr in self.match_list
            if mr.__dict__[cn.ESP_PRIORITY] >= 0
        )

        self.match_list.sort()
        self.sorted = True
        if travel_time_dict:
            self.center_travel_times = travel_time_dict[
                self.donor.__dict__[cn.DONOR_CENTER]
            ]
        else:
            self.center_travel_times = None

        for rank, match_record in enumerate(self.match_list):
            if isinstance(match_record, MatchRecordCurrentESP):
                match_record.add_patient_rank(rank)

    def return_match_list(
            self
    ) -> List[MatchRecordCurrentESP]:
        return [m for m in self.match_list]

    def return_patient_ids(
        self
    ) -> List[int]:
        """Return matching patients"""
        patients = []
        for matchr in self.return_match_list():
            if isinstance(matchr, MatchRecordCurrentESP):
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
