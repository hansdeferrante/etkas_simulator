#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13

@author: H.C. de Ferrante
"""

from typing import Optional, Tuple, Dict, Union, List, Callable
from math import isnan
from operator import attrgetter
from collections import defaultdict
import numpy as np
import pandas as pd

import simulator.magic_values.column_names as cn
import simulator.magic_values.etkass_settings as es
import simulator.magic_values.magic_values_rules as mgr
from simulator.code.utils import round_to_decimals, round_to_int
from simulator.code.functions import construct_piecewise_term
from simulator.code.current_etkas.CurrentETKAS import \
    MatchListCurrentETKAS, MatchRecordCurrentETKAS
from simulator.code.AllocationSystem import MatchRecord
from simulator.code.entities import Donor, Patient, Profile
from simulator.code.read_input_files import read_rescue_probs

import numpy as np


def logit(prob: float) -> float:
    """Calculates logit for probability"""
    return np.log(prob) - np.log(1 - prob)


def inv_logit(logit_: float) -> float:
    """Calculates probability for logit"""
    return np.exp(logit_) / (1 + np.exp(logit_))


class AcceptanceModule:
    """ Class to implement an acceptance module,
        based on logistic regression.

    Attributes   #noqa
    ----------

    Methods
    -------

    """
    def __init__(
        self,
        seed,
        patient_acc_policy: str,
        center_acc_policy: str,
        generate_splits: bool = True,
        dict_paths_coefs: Dict[str, str] = es.ACCEPTANCE_PATHS,
        separate_huaco_model: bool = True,
        separate_ped_model: bool = True,
        simulate_rescue: bool = False,
        path_rescue_probs: Optional[str] = None,
        verbose: Optional[int] = None,
        simulate_random_effects: Optional[bool] = True
    ):
        # Initialize random number generators
        rng = np.random.default_rng(seed=seed)
        seeds = rng.choice(999999, size=10, replace=False)
        self.rng_center = np.random.default_rng(seed=seeds[1])
        self.rng_patient_per_center = np.random.default_rng(seed=seeds[2])
        self.rng_rescue = np.random.default_rng(seed=seeds[3])
        self.rng_random_eff = np.random.default_rng(seed=seeds[4])

        self.verbose = verbose if verbose else 0

        # Set patient acceptance policy.
        if patient_acc_policy.lower() == 'LR'.lower():
            self.determine_patient_acceptance = self._patient_accept_lr
        elif patient_acc_policy.lower() == 'Always'.lower():
            self.determine_patient_acceptance = self._patient_accept_always
        else:
            raise ValueError(
                f'Patient acceptance policy should be one of '
                f'{", ".join(es.PATIENT_ACCEPTANCE_POLICIES)}, '
                f'not {patient_acc_policy}'
            )

        # Set center acceptance policy.
        if center_acc_policy.lower() == 'LR'.lower():
            self.determine_center_acceptance = self._center_accept_lr
        elif center_acc_policy.lower() == 'Always'.lower():
            self.determine_center_acceptance = self._center_accept_always
        else:
            raise ValueError(
                f'Patient acceptance policy should be one of '
                f'{", ".join(es.CENTER_ACCEPTANCE_POLICIES)}, '
                f'not {center_acc_policy}'
            )

        # Save whether to use separate models for HU/ACO and regular
        if separate_huaco_model:
            if separate_ped_model:
                self.calculate_prob_patient_accept = (
                    self._calc_prob_accept_pedad_huaco
                )
            else:
                self.calculate_prob_patient_accept = (
                    self._calc_prob_accept_huaco
                )
        else:
            if separate_ped_model:
                print(
                    "Cannot simulate separate pediatric "
                    "model without separate HU/ACO model."
                )
                exit()
            self.calculate_prob_patient_accept = self._calc_prob_accept

        self.simulate_rescue = simulate_rescue
        if simulate_rescue:
            if path_rescue_probs is None:
                path_rescue_probs = es.PATH_RESCUE_PROBABILITIES

            self.rescue_init_probs = read_rescue_probs(path_rescue_probs)

        self.simulate_random_effects = simulate_random_effects
        if self.simulate_random_effects:
            self.random_effects = {
               'rd_adult_huaco': {cn.ID_REGISTRATION: 1.14}
            }
            self.realizations_random_effects = {
                k: {} for k in self.random_effects.keys()
            }

    def generate_offers_to_rescue(self, d_country: str) -> int:
        """ Sample the number of rejections made at triggering rescue/
            extended allocation from the empirical distribution per country
        """
        r_prob = self.rng_rescue.random()
        prob_dict = self.rescue_init_probs[d_country]
        if any(prob_dict[cn.PROB_TILL_RESCUE] > r_prob):
            which_n_offers = np.argmax(
                prob_dict[cn.PROB_TILL_RESCUE] > r_prob
            )
            n_offers = prob_dict[cn.N_OFFERS_TILL_RESCUE][
                which_n_offers
            ]
        else:
            n_offers = max(prob_dict[cn.N_OFFERS_TILL_RESCUE])
        return int(n_offers)


    def return_transplanted_organ(
            self, match_record: Union[MatchRecord, MatchRecordCurrentETKAS]
    ) -> int:
        return match_record.__dict__[cn.TYPE_OFFER_DETAILED]

    def _patient_accept_always(
        self, match_record: Union[MatchRecord, MatchRecordCurrentETKAS]
    ) -> float:
        """Policy to always accept the offer, if profile permits."""
        if match_record.patient.profile is not None:
            if match_record.profile_compatible:
                match_record.set_acceptance(
                    reason=cn.T3 if match_record.donor.rescue else cn.T1
                )
                return True
        match_record.set_acceptance(reason=cn.FP)
        return False

    def _patient_accept_lr(
        self, match_record: Union[MatchRecord, MatchRecordCurrentETKAS]
    ) -> float:
        """Check acceptance with LR if profile acceptable or not checked."""
        if match_record.profile_compatible:
            if (
                self.calculate_prob_patient_accept(
                    offer=match_record,
                    verbose=0
                    ) >= match_record.patient.get_acceptance_prob()
            ):
                match_record.set_acceptance(
                    reason=cn.T3 if match_record.donor.rescue else cn.T1
                )
                return True
            else:
                match_record.set_acceptance(
                    reason=cn.RR
                )
        else:
            match_record.set_acceptance(
                reason=cn.FP
            )
        return False

    def _center_accept_always(
        self, center_offer: MatchRecord,
        verbose: Optional[int] = None
    ) -> bool:
        center_offer.set_acceptance(
            reason=cn.T3 if center_offer.donor.rescue else cn.T1
        )
        return True

    def _center_accept_lr(
        self, center_offer: MatchRecord,
        verbose: Optional[int] = None
    ) -> bool:
        """Check whether the center accepts."""
        center_offer.__dict__[cn.DRAWN_PROB_C] = self.rng_center.random()
        if (
            self.calculate_center_offer_accept(
                    offer=center_offer,
                    verbose=verbose
                ) >= center_offer.__dict__[cn.DRAWN_PROB_C]
        ):
            center_offer.set_acceptance(
                reason=cn.T3 if center_offer.donor.rescue else cn.T1
            )
            return True
        return False

    def _calc_prob_accept_pedad_huaco(
        self, offer: MatchRecord, verbose: Optional[int] = None
    ):
        """Calculate probability acceptance with separate
        HU/ACO and adult/ped models"""
        if verbose is None:
            verbose = self.verbose
        if verbose > 1:
            print('*******')

        if offer.patient.r_aco or offer.__dict__[cn.PATIENT_IS_HU]:
            if offer.__dict__[cn.R_MATCH_AGE] < 18:
                selected_model = 'rd_ped_huaco'
            else:
                selected_model = 'rd_adult_huaco'
        else:
            if offer.__dict__[cn.R_MATCH_AGE] < 18:
                selected_model = 'rd_ped_reg'
            else:
                selected_model = 'rd_adult_reg'

        if (
            self.simulate_random_effects and
            selected_model in self.random_effects.keys()
        ):
            if (
                offer.patient.__dict__[cn.ID_REGISTRATION] not in
                self.realizations_random_effects[selected_model]
            ):
                re = self.rng_random_eff.normal(
                    loc=0,
                    scale=self.random_effects[selected_model][
                        cn.ID_REGISTRATION
                        ]
                )
                self.realizations_random_effects[selected_model][
                    offer.patient.__dict__[cn.ID_REGISTRATION]
                ] = re
            else:
                re = self.realizations_random_effects[selected_model][
                    offer.patient.__dict__[cn.ID_REGISTRATION]
                ]
        else:
            re = None

        return self._calculate_logit(
                offer=offer,
                which=selected_model,
                verbose=verbose,
                realization_intercept=re
            )

    def _calc_prob_accept_huaco(
        self, offer: MatchRecord, verbose: Optional[int] = None
    ):
        """Calculate probability acceptance with separate HU/ACO models"""
        if verbose is None:
            verbose = self.verbose
        if verbose > 1:
            print('*******')
        if offer.patient.r_aco or offer.__dict__[cn.PATIENT_IS_HU]:
            selected_model = 'rd_huaco'
        else:
            selected_model = 'rd_ped'

        return self._calculate_logit(
                offer=offer,
                which=selected_model,
                verbose=verbose
        )

    def _calc_pcd_select(
        self, offer: MatchRecord, verbose: Optional[int] = None
    ):
        """Calculate probability acceptance with separate HU/ACO models"""
        if verbose is None:
            verbose = self.verbose
        if verbose > 1:
            print('*******')
        return self._calculate_logit(
                offer=offer,
                which='pcd_ped' if offer.__dict__[cn.R_PED] else 'pcd_adult',
                verbose=verbose
        )

    def _calc_prob_accept(
        self, offer: MatchRecord, verbose: Optional[int] = None
    ):
        """Calculate probability acceptance with separate HU/ACO models"""
        if verbose is None:
            verbose = self.verbose
        if verbose > 1:
            print('*******')
        return self._calculate_logit(
                offer=offer,
                which='rd',
                verbose=verbose
        )

    def calculate_center_offer_accept(
            self, offer: MatchRecord,
            verbose: Optional[int] = 0
    ) -> float:
        """Calculate probability center accepts"""
        if verbose and verbose > 1:
            print('*******')
        return self._calculate_logit(
                offer=offer,
                which='cd',
                verbose=verbose
        )

    def calculate_prob_split(
            self, offer: MatchRecord, verbose: Optional[int] = None
    ) -> float:
        """Calculate probability center accepts"""

        if offer.__dict__[cn.RECIPIENT_CENTER] in es.CENTERS_WHICH_SPLIT:
            return self._calculate_logit(
                    offer=offer,
                    which='sp',
                    verbose=verbose
                )
        else:
            return 0

    def _calculate_logit(
            self, offer: MatchRecord,
            which: str, verbose: Optional[int] = None,
            realization_intercept: Optional[float] = None
    ) -> float:
        """Calculate probability patient accepts"""
        if verbose is None:
            verbose = self.verbose

        # Realization of random intercept
        if realization_intercept:
            slogit = realization_intercept
        else:
            slogit = 0

        for key, fe_dict in self.fixed_effects[which].items():
            slogit_b4 = slogit
            var2 = None
            if isinstance(fe_dict, dict):
                sel_coefs = fe_dict.get(
                    str(int(offer.__dict__[key]))
                    if isinstance(offer.__dict__[key], bool)
                    else str(offer.__dict__[key]),
                    0
                )
                if isinstance(sel_coefs, dict):
                    for var2, dict2 in sel_coefs.items():
                        slogit += dict2.get(
                            str(offer.__dict__[var2]),
                            0
                        )
                else:
                    slogit += fe_dict.get(
                        str(int(offer.__dict__[key]))
                        if isinstance(offer.__dict__[key], bool)
                        else str(offer.__dict__[key]),
                        0
                    )
            elif key == 'intercept':
                slogit += fe_dict
            else:
                slogit += offer.__dict__[key] * fe_dict

            if (slogit_b4 != slogit) & (verbose > 1):
                if key in offer.__dict__:
                    if var2:
                        print(
                            f'{key}-{offer.__dict__[key]}:'
                            f'{var2}-{offer.__dict__[var2]}: '
                            f'{slogit-slogit_b4}'
                        )
                    else:
                        print(
                            f'{key}-{offer.__dict__[key]}: '
                            f'{slogit-slogit_b4}'
                        )
                else:
                    print(f'{key}: {slogit-slogit_b4}')

        for orig_var, fe_dict in (
            self.continuous_transformations[which].items()
        ):
            for coef_to_get, trafo in fe_dict.items():
                if (value := offer.__dict__[orig_var]) is not None:
                    contr = (
                        trafo(value) *
                        self.continuous_effects[which][orig_var][coef_to_get]
                    )
                    slogit += contr

                    if (contr != 0) & (verbose > 1):
                        print(f'{coef_to_get}-{value}: {contr}')
                else:
                    print(f'{orig_var} yields None for {offer}')

        if which == 'cd':
            offer.__dict__[cn.PROB_ACCEPT_C] = round_to_decimals(inv_logit(slogit), 3)
        elif which != 'sp':
            offer.__dict__[cn.PROB_ACCEPT_P] = round_to_decimals(inv_logit(slogit), 3)
            if verbose:
                print(f'{which}: {round_to_decimals(inv_logit(slogit), 3)}')
                print(inv_logit(slogit))
        return inv_logit(slogit)

    def simulate_liver_allocation(
        self, match_list: MatchListCurrentETKAS
    ) -> Optional[MatchRecord]:
        """ Iterate over a list of match records. If we model rescue alloc,
            we simulate when rescue will be triggered and terminate
            recipient-driven allocation. In that case, we simulate further
            allocation with the donor identified as a rescue donor, and
            prioritize locally in Belgium / regionally in Germany.
        """
        # When not simulating rescue, offer to all patients on the match list
        if not self.simulate_rescue:
            acc_matchrecord, _ = self._find_accepting_matchrecord(
                match_list.return_match_list()
            )
            return acc_matchrecord
        else:
            if (
                match_list.donor.__dict__[cn.D_DCD] and
                match_list.__dict__[cn.D_ALLOC_COUNTRY] != mgr.NETHERLANDS
            ):
                offers_till_rescue = 9999
            else:
                # Draw number of offers until rescue is initiated
                offers_till_rescue = self.generate_offers_to_rescue(
                    match_list.__dict__[cn.D_ALLOC_COUNTRY]
                )

            # Allocate until `offers_till_rescue` rescue offers have been made.
            # This returns the match record for the accepting patient
            # (`acc_matchrecord`) if the graft was accepted, and a list of
            # centers willing to accept the organ (determined at the
            # center level)
            acc_matchrecord, center_willingness = (
                self._find_accepting_matchrecord(
                    match_list.return_match_list(),
                    max_offers=offers_till_rescue,
                    max_offers_per_center=5
                )
            )

            # If rescue was triggered before an acceptance, continue
            # allocation with prioritization for candidates with
            # rescue priority (local in BE, regional in DE).
            if not acc_matchrecord:
                match_list._initialize_rescue_priorities()
                match_list.donor.rescue = True

                # If rescue is triggered, prioritize remaining waitlist
                # on rescue priority (i.e. local in Belgium, regional in DE)
                acc_matchrecord, _ = self._find_accepting_matchrecord(
                    list(
                        sorted(
                            match_list.return_match_list(),
                            key=attrgetter(cn.RESCUE_PRIORITY),
                            reverse=True
                        )
                    ),
                    center_willing_to_accept=center_willingness
                )

            return acc_matchrecord

    def _find_accepting_matchrecord(
            self,
            match_records_list: List[
                MatchRecord
            ],
            max_offers: Optional[int] = 9999,
            max_offers_per_center: int = 9999,
            center_willing_to_accept: Optional[Dict[str, bool]] = None
    ) -> Tuple[Optional[MatchRecord], Dict[str, bool]]:
        """ Iterate over all match records in the match list, and simulate
            if the match object accepts the graft offer.

        Parameters
        ------------
        match_records_list: List[
            Union[MatchRecord, MatchRecordCurrentETKAS]
            ]
            match list to iterate over, and make offers to
        max_offers: int
            maximum number of offers that will be made
            (rescue allocation is triggered after it)
        max_offers_per_center: int
            maximum number of offers per center that counts towards
            triggering rescue allocation. Recipients will still be
            offered the graft.
        center_willing_to_accept: Optional[Dict[str, bool]]
            Dictionary of center names with booleans whether they are
            willing to consider the graft or not.
        """
        count_rejections_total = 0
        n_rejections_per_center = defaultdict(int)
        if center_willing_to_accept is None:
            center_willing_to_accept = {}

        # Make offers to match objects. We count an offer as made if
        # (i) it is center-driven offer,
        # (ii) it is the first offer to a center in non-HU/ACO RD allocation,
        # (iii) it is an offer to a filtered match list candidate
        for match_object in match_records_list:
            # If more than max number of offers are made, break
            # to initiate rescue.
            if max_offers and count_rejections_total >= max_offers:
                break

            # Skip patients who already rejected the offer
            if match_object.__dict__.get(cn.ACCEPTANCE_REASON, None):
                continue
            elif hasattr(match_object, '_initialize_acceptance_information'):
                match_object._initialize_acceptance_information()


            # Recipient-driven allocation
            if not (
                match_object.__dict__[cn.RECIPIENT_CENTER] in
                center_willing_to_accept
            ):
                if match_object.profile_compatible:
                    if self.determine_center_acceptance(
                        match_object, verbose=0
                    ):
                        center_willing_to_accept[
                            match_object.__dict__[cn.RECIPIENT_CENTER]
                        ] = True
                    else:
                        center_willing_to_accept[
                            match_object.__dict__[cn.RECIPIENT_CENTER]
                        ] = False
                        count_rejections_total += 1
                else:
                    match_object.set_acceptance(
                        reason=cn.FP
                    )

            # 2) If center finds patient acceptable, make a patient-driven offer
            if center_willing_to_accept.get(
                match_object.__dict__[cn.RECIPIENT_CENTER],
                False
            ):
                # Make offer if the center has received fewer than
                # `max_offers_per_center` offers, or if rescue was
                # triggered
                if (
                    n_rejections_per_center[
                        match_object.__dict__[cn.RECIPIENT_CENTER]
                    ] < max_offers_per_center
                ) or match_object.donor.rescue:
                    if self.determine_patient_acceptance(match_object):
                        return match_object, center_willing_to_accept
                    elif (
                        match_object.__dict__[
                            cn.ACCEPTANCE_REASON
                            ] != cn.FP
                    ):
                        n_rejections_per_center[
                            match_object.__dict__[cn.RECIPIENT_CENTER]
                            ] += 1
                        count_rejections_total += 1
            else:
                # Else record center rejection (CR)
                match_object.set_acceptance(
                    reason=cn.CR
                )

        return None, center_willing_to_accept


    def fe_coefs_to_dict(self, level: pd.Series, coef: pd.Series):
        """Process coefficients to dictionary"""
        fe_dict = {}
        for lev, val in zip(level, coef):
            if ':' in str(lev):
                l1, lv2 = lev.split(':')
                var2, l2 = lv2.split('-')

                if l1 in fe_dict:

                    if var2 in fe_dict[l1]:
                        fe_dict[l1][var2].update(
                            {l2: val}
                        )
                    else:
                        fe_dict[l1].update(
                            {var2: {l2: val}}
                        )
                else:
                    fe_dict.update(
                        {l1: {var2: {l2: val}}}
                    )
            else:
                fe_dict.update(
                    {lev: val}
                )
        return fe_dict

    def _initialize_lr_coefs(
        self,
        dict_paths_coefs: Dict[str, str],
        path_coef_split: str
    ):
        """Initialize the logistic regression coefficients for
            recipient (rd) and center-driven (cd) allocation
        """

        self.__dict__['sp'] = pd.read_csv(path_coef_split, dtype='str')
        self.__dict__['sp']['coef'] = self.__dict__['sp']['coef'].astype(float)

        for k, v in dict_paths_coefs.items():
            self.__dict__[k] = pd.read_csv(v, dtype='str')
            self.__dict__[k]['coef'] = self.__dict__[k]['coef'].astype(float)

        coef_keys = list(dict_paths_coefs.keys()) + ['sp']

        # Create dictionary for fixed effects
        self.fixed_effects = {}
        for dict_name in coef_keys:
            self.fixed_effects[dict_name] = (
                self.__dict__[dict_name].loc[
                    ~ self.__dict__[dict_name].variable_transformed.notna()
                ].groupby('variable').
                apply(
                    lambda x: self.fe_coefs_to_dict(x['level'], x['coef'])
                ).to_dict()
            )

        for _, td in self.fixed_effects.items():
            for k, v in td.items():
                if (
                    isinstance(list(v.keys())[0], float) and
                    isnan(list(v.keys())[0])
                ):
                    td[k] = list(v.values())[0]

        self.continuous_transformations = {}
        self.continuous_effects = {}
        for dict_name in coef_keys:
            self.continuous_transformations[dict_name] = (
                self.__dict__[dict_name].loc[
                    self.__dict__[dict_name]
                    .variable_transformed.notna()
                ].groupby('variable').
                apply(
                    lambda x: {
                        k: construct_piecewise_term(v)
                        for k, v in zip(
                            x['variable_transformed'],
                            x['variable_transformation']
                        )
                    }
                ).to_dict()
            )

            self.continuous_effects[dict_name] = (
                self.__dict__[dict_name].loc[
                    self.__dict__[dict_name]
                    .variable_transformed.notna()
                ].groupby('variable').
                apply(
                    lambda x: dict(zip(x['variable_transformed'], x['coef']))
                ).to_dict()
            )
