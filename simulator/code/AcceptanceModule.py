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
    MatchListCurrentETKAS, MatchRecordCurrentETKAS, \
    MatchListESP, MatchRecordCurrentESP
from simulator.code.AllocationSystem import MatchRecord, MatchList
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
        dict_paths_coefs: Dict[str, str] = es.ACCEPTANCE_PATHS,
        simulate_rescue: bool = False,
        paths_rescue_probs: Optional[Dict[str, str]] = None,
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

        # Initialize coefficients for the logistic regression
        if (
            (patient_acc_policy == 'LR') |
            (center_acc_policy == 'LR')
        ):
            self._initialize_lr_coefs(
                dict_paths_coefs=dict_paths_coefs
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

        self.calculate_prob_patient_accept = self._calc_prob_accept

        self.simulate_rescue = simulate_rescue
        if simulate_rescue:
            if paths_rescue_probs is None:
                paths_rescue_probs = es.PATHS_RESCUE_PROBABILITIES

            self.rescue_init_probs = {
                key: read_rescue_probs(path)
                for key, path in es.PATHS_RESCUE_PROBABILITIES.items()
            }

        self.simulate_random_effects = simulate_random_effects
        if self.simulate_random_effects:
            self.random_effects = {
            }
            self.realizations_random_effects = {
                k: {} for k in self.random_effects.keys()
            }

    def generate_offers_to_rescue(self, d_country: str, program: str) -> int:
        """ Sample the number of rejections made at triggering rescue/
            extended allocation from the empirical distribution per country
        """
        if program == mgr.ESP:
            prob_dict = self.rescue_init_probs[mgr.ESP]
        elif program == mgr.ETKAS:
            prob_dict = self.rescue_init_probs[mgr.ETKAS][d_country]
        else:
            raise Exception(
                "Don't know how to generate rescue offers for the "
                "{program}-allocation program. Try {mgr.ETKAS} or {mgr.ESP}")
        r_prob = self.rng_rescue.random()
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
                    verbose=self.verbose
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

    def calc_prob_enbloc(
        self, offer: MatchRecord, verbose: Optional[int] = None
    ):
        """Calculate probability acceptance with separate HU/ACO models"""
        if verbose is None:
            verbose = self.verbose
        if verbose > 1:
            print('*******')
        return self._calculate_logit(
                offer=offer,
                which='enbloc',
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
                    None
                )
                if isinstance(sel_coefs, dict):
                    for var2, dict2 in sel_coefs.items():
                        slogit += dict2.get(
                            str(offer.__dict__[var2]),
                            0
                        )
                elif sel_coefs is None:
                    newkey = (
                        str(int(offer.__dict__[key]))
                        if isinstance(offer.__dict__[key], bool)
                        else str(offer.__dict__[key])
                    )
                    if self.reference_levels[which][key] is None:
                        fe_dict[newkey] = 0
                        self.reference_levels[which][key] = newkey
                    else:
                        raise Exception(
                            f'Multiple reference levels for {key}:\n'
                            f'\t{self.reference_levels[which][key]} and '
                            f'{newkey}\n are both assumed reference levels.\n'
                            f'Existing keys are:\n{self.fixed_effects[which][key]}'

                        )
                else:
                    slogit += sel_coefs

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
        elif which == 'rd':
            offer.__dict__[cn.PROB_ACCEPT_P] = round_to_decimals(inv_logit(slogit), 3)
            if verbose:
                print(f'{which}: {round_to_decimals(inv_logit(slogit), 3)}')
        elif which == 'enbloc':
            offer.__dict__[cn.PROB_ENBLOC] = round_to_decimals(inv_logit(slogit), 3)
        else:
            print(f'{which} is not a valid option for prediction of acceptance with the ETKAS simulator')
        return inv_logit(slogit)

    def simulate_esp_allocation(self, match_list: MatchListESP, n_kidneys_available=int):
        """ Iterate over all match records. Do not model rescue allocation."""

        # TODO: Add rescue allocation for ESP? Not sure when that is triggered / what happens
        if self.simulate_rescue:
            offers_till_rescue = self.generate_offers_to_rescue(
                    match_list.__dict__[cn.D_ALLOC_COUNTRY],
                    program = mgr.ESP
            )
        else:
            offers_till_rescue = 9999

        # TODO: in ESP, rescue is not triggerable currently; allocation then stops
        # and we move on to ETKAS for allocation. In reality, also very rarely
        # ESP-aged kidneys are allocated via rescue, so perhaps not an issue?
        acc_matchrecords, center_willingness = self._find_accepting_matchrecords(
            match_list,
            max_offers=offers_till_rescue,
            max_offers_per_center=5,
            n_kidneys_available=n_kidneys_available
        )
        return acc_matchrecords

    def simulate_etkas_allocation(
        self, match_list: MatchListCurrentETKAS,
        n_kidneys_available: int
    ) -> Optional[List[MatchRecord]]:
        """ Iterate over a list of match records. If we model rescue alloc,
            we simulate when rescue will be triggered and terminate
            recipient-driven allocation. In that case, we simulate further
            allocation with the donor identified as a rescue donor, and
            prioritize locally in Belgium / regionally in Germany.
        """
        # When not simulating rescue, offer to all patients on the match list
        if self.simulate_rescue:
            offers_till_rescue = self.generate_offers_to_rescue(
                    match_list.__dict__[cn.D_ALLOC_COUNTRY],
                    program = mgr.ETKAS
            )
        else:
            offers_till_rescue = 9999

        # Allocate until `offers_till_rescue` rescue offers have been made.
        # This returns the match record for the accepting patient
        # (`acc_matchrecord`) if the graft was accepted, and a list of
        # centers willing to accept the organ (determined at the
        # center level)
        acc_matchrecords, center_willingness = (
            self._find_accepting_matchrecords(
                match_list,
                max_offers=offers_till_rescue,
                max_offers_per_center=5,
                n_kidneys_available=n_kidneys_available
            )
        )
        if acc_matchrecords:
            n_kidneys_accepted = sum(
                2 if mr.__dict__[cn.ENBLOC]
                else 1
                for mr in acc_matchrecords
            )
        else:
            n_kidneys_accepted = 0

        # If rescue was triggered before an acceptance, continue
        # allocation with prioritization for candidates with
        # rescue priority (local in BE, regional in DE).
        if acc_matchrecords is None or (
            n_kidneys_accepted < n_kidneys_available
        ):
            match_list._initialize_extalloc_priorities()
            match_list.donor.rescue = True

            if acc_matchrecords is not None:
                n_kidneys_available -= n_kidneys_accepted

            # If rescue is triggered, prioritize remaining waitlist
            # on rescue priority (i.e. local in Belgium, regional in DE)
            acc_matchrecords_rescue, _ = self._find_accepting_matchrecords(
                match_list,
                center_willing_to_accept=center_willingness,
                n_kidneys_available=n_kidneys_available
            )

            if acc_matchrecords_rescue is not None:
                if acc_matchrecords is None:
                    return acc_matchrecords_rescue
                else:
                    return acc_matchrecords + acc_matchrecords_rescue

        return acc_matchrecords

    def _find_accepting_matchrecords(
            self,
            match_list: Union[MatchListCurrentETKAS, MatchListESP, MatchList],
            n_kidneys_available: int,
            max_offers: Optional[int] = 9999,
            max_offers_per_center: int = 9999,
            center_willing_to_accept: Optional[Dict[str, bool]] = None
    ) -> Tuple[Optional[List[MatchRecord]], Dict[str, bool]]:
        """ Iterate over all match records in the match list, and simulate
            if the match object accepts the graft offer.

        Parameters
        ------------
        match_list: Union[MatchListCurrentETKAS, MatchListESP, MatchList]
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

        if len(match_list) == 0:
            return None, {}

        # If extended allocation has been triggered, order in offer
        # of extended allocation.
        if (
            isinstance(match_list, MatchListCurrentETKAS) and
            match_list.ext_alloc_priority
        ):
            match_records_list = list(
                sorted(
                    match_list.return_match_list(),
                    key=attrgetter(cn.EXT_ALLOC_PRIORITY),
                    reverse=True
                )
            )
        else:
            match_records_list = match_list.return_match_list()

        match_objects: List[MatchRecord] = list()
        n_kidneys_accepted = 0
        # Make offers to match objects. We count an offer as made if
        # (i) it is center-driven offer,
        # (ii) it is the first offer to a center in non-HU/ACO RD allocation,
        # (iii) it is an offer to a filtered match list candidate
        for match_object in match_records_list:
            # If more than max number of offers are made, break
            # to initiate rescue.
            if max_offers and count_rejections_total >= max_offers:
                break

            # Skip patients who already rejected the offer.
            # This can happen in case a candidate rejected an
            # offer prior to rescue / extended allocation.
            if match_object.__dict__.get(cn.ACCEPTANCE_REASON, None):
                continue
            elif hasattr(match_object, '_initialize_acceptance_information'):
                match_object._initialize_acceptance_information()

            # 1) determine whether center is willing to accept
            if not (
                match_object.__dict__[cn.RECIPIENT_CENTER] in
                center_willing_to_accept
            ):
                if match_object.profile_compatible:
                    if self.determine_center_acceptance(
                        match_object
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
                    match_object.set_acceptance(reason=cn.FP)
                    continue

            # 2) if center finds patient acceptable, make a patient-driven offer
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
                        match_objects.append(match_object)
                        # if two kidneys are available, and none were
                        # accepted yet, simulate whether candidate accepts
                        # kidneys en bloc
                        if (
                            n_kidneys_available == 2 and
                            n_kidneys_accepted == 0
                        ):
                            if (
                                self.calc_prob_enbloc(
                                    offer=match_object,
                                    verbose=self.verbose
                                ) > match_object.patient.get_acceptance_prob()
                            ):
                                match_object.__dict__[cn.ENBLOC] = 1
                                n_kidneys_accepted += 2
                            else:
                                match_object.__dict__[cn.ENBLOC] = 0
                                n_kidneys_accepted += 1
                        else:
                            match_object.__dict__[cn.ENBLOC] = 0
                            n_kidneys_accepted += 1
                        if n_kidneys_accepted == n_kidneys_available:
                            return match_objects, center_willing_to_accept
                        elif n_kidneys_accepted > n_kidneys_available:
                            raise Exception(
                                f'{n_kidneys_accepted} candidate(s) have accepted the kidney,'
                                f'while only {n_kidneys_available} were available.'
                            )
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
            else:
                # Else record center rejection (CR)
                match_object.set_acceptance(
                    reason=cn.CR
                )

        # If too little acceptances in regular allocation, return
        # the accepting match record.
        if len(match_objects) >= 0:
            return match_objects, center_willing_to_accept

        # In case of no acceptance in regular allocation, return
        # center willingness to accept (to trigger rescue)
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
        dict_paths_coefs: Dict[str, str]
    ):
        """Initialize the logistic regression coefficients for
            recipient (rd) and center-driven (cd) allocation
        """

        for k, v in dict_paths_coefs.items():
            self.__dict__[k] = pd.read_csv(v, dtype='str')
            self.__dict__[k]['coef'] = self.__dict__[k]['coef'].astype(float)

        coef_keys = list(dict_paths_coefs.keys())

        # Create dictionary for fixed effects
        self.fixed_effects = {}
        self.reference_levels = {}
        for dict_name in coef_keys:
            self.fixed_effects[dict_name] = (
                self.__dict__[dict_name].loc[
                    ~ self.__dict__[dict_name].variable_transformed.notna()
                ].groupby('variable').
                apply(
                    lambda x: self.fe_coefs_to_dict(x['level'], x['coef'])
                ).to_dict()
            )
            self.reference_levels[dict_name] = defaultdict(lambda: None)

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
