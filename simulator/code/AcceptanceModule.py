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
from simulator.code.read_input_files import read_rescue_baseline_hazards

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
        path_coefs_rescueprobs: Optional[str] = None,
        path_basehaz_rescueprobs: Optional[str] = None,
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

        # If simulating rescue allocation, also read in fixed effects for Cox model
        self.simulate_rescue = simulate_rescue
        if simulate_rescue:
            if path_basehaz_rescueprobs is None:
                path_basehaz_rescueprobs = es.PATH_RESCUE_COX_BH
            if path_coefs_rescueprobs is None:
                path_coefs_rescueprobs = es.PATH_RESCUE_COX_COEFS

            self.rescue_bh = read_rescue_baseline_hazards(path_basehaz_rescueprobs)
            dict_paths_coefs.update(
                {'coxph_rescue': path_coefs_rescueprobs}
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

        self.simulate_random_effects = simulate_random_effects
        if self.simulate_random_effects:
            self.random_effects = {
                'rd': {
                    # cn.ID_REGISTRATION: 0.57532,
                    cn.ID_DONOR: 0.53926
                },
                'cd': {
                    #cn.ID_DONOR: 0.75508
                }
            }
            self.realizations_random_effects = {
                k: defaultdict(dict) for k in self.random_effects.keys()
            }


    def _generate_rescue_eventcurve(
            self, donor: Donor,
            strata_column: Optional[str] = cn.DONOR_COUNTRY,
            verbose: Optional[int] = 0) -> Tuple[
        np.ndarray,
        np.ndarray
    ]:

        lp = self._calculate_lp(
            item=donor.__dict__,
            which='coxph_rescue',
            realization_intercept=0,
            verbose=verbose
        )

        if strata_column is not None:
            stratum = donor.__dict__[strata_column]
        else:
            stratum = np.nan

        ind_cbh = self.rescue_bh[stratum][cn.CBH_RESCUE] * np.exp(lp)

        return self.rescue_bh[stratum][cn.N_OFFERS_TILL_RESCUE], 1-np.exp(-ind_cbh)


    def generate_offers_to_rescue(
            self, donor: Donor,
            program: Optional[str] = None,
            verbose: Optional[int]=0
        ) -> int:
        """ Sample the number of rejections made at triggering rescue/
            extended allocation from the empirical distribution per country
        """

        n_offers, event_probs = self._generate_rescue_eventcurve(
            donor=donor,
            verbose=verbose
        )
        r_prob = self.rng_rescue.random()
        if all(r_prob < event_probs):
            return int(0)
        elif any(r_prob < event_probs):
            which_n_offers = np.argmax(
                event_probs > r_prob
            )
            kth_offer = n_offers[which_n_offers]
        else:
            kth_offer = len(n_offers)

        if verbose:
            print(f'{kth_offer-1} due to prob {r_prob}')

        return int(kth_offer-1)

    def predict_rescue_prob(
            self, donor: Donor, kth_offer: int,
            verbose: Optional[int]=0) -> float:
        n_offers, event_probs = self._generate_rescue_eventcurve(
            donor=donor,
            verbose=2
        )
        which_prob = np.argmax(
            n_offers >= kth_offer
        )
        return(event_probs[which_prob])


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
        k_previous_center_rejections: int,
        verbose: Optional[int] = None
    ) -> bool:
        center_offer.set_acceptance(
            reason=cn.T3 if center_offer.donor.rescue else cn.T1
        )
        return True

    def _center_accept_lr(
        self, center_offer: MatchRecord,
        k_previous_center_rejections: int,
        verbose: Optional[int] = None
    ) -> bool:
        """Check whether the center accepts."""

        # Slovenia has very few patients; leads to collinearity issues.
        if center_offer.__dict__[cn.PATIENT_COUNTRY] == mgr.SLOVENIA:
            return True

        center_offer.__dict__[cn.K_PREVIOUS_CENTER_REJECTIONS] = (
            str(k_previous_center_rejections) if k_previous_center_rejections < 5
            else '5+'
        )
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
        self, offer: MatchRecord, verbose: Optional[int] = None,
        selected_model: str = 'rd'
    ):
        """Calculate probability acceptance with separate HU/ACO models"""
        if verbose is None:
            verbose = self.verbose
        if verbose > 1:
            print('*******')

        realization_intercept = 0
        if self.simulate_random_effects:
            for var, sd in self.random_effects[selected_model].items():
                if (re := self.realizations_random_effects[selected_model][var].get(offer.__dict__[var])) is not None:
                    realization_intercept += re
                else:
                    re = self.rng_random_eff.normal(
                        loc=0,
                        scale=sd
                    )
                    self.realizations_random_effects[selected_model][
                        var
                    ][offer.__dict__[var]] = re
                    realization_intercept += re

        return self._calculate_logit(
                offer=offer,
                which='rd',
                verbose=verbose,
                realization_intercept=realization_intercept
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
            verbose: Optional[int] = 0,
            selected_model: str = 'cd'
    ) -> float:
        """Calculate probability center accepts"""
        if verbose and verbose > 1:
            print('*******')

        realization_intercept = 0
        if self.simulate_random_effects:
            for var, sd in self.random_effects[selected_model].items():
                if (re := self.realizations_random_effects[selected_model][var].get(offer.__dict__[var])) is not None:
                    realization_intercept += re
                else:
                    re = self.rng_random_eff.normal(
                        loc=0,
                        scale=sd
                    )
                    self.realizations_random_effects[selected_model][
                        var
                    ][offer.__dict__[var]] = re
                    realization_intercept += re
        if verbose:
            print(self._calculate_logit(
                        offer=offer,
                        which=selected_model,
                        realization_intercept=realization_intercept
            )
            )

        return self._calculate_logit(
                offer=offer,
                which=selected_model,
                verbose=verbose,
                realization_intercept=realization_intercept
        )


    def _calculate_lp(
            self, item: Dict,
            which: str, verbose: Optional[int] = None,
            realization_intercept: Optional[float] = None
    ):
        # Realization of random intercept
        if realization_intercept:
            lp = realization_intercept
        else:
            lp = 0

        for key, fe_dict in self.fixed_effects[which].items():
            slogit_b4 = lp
            var2 = None
            if isinstance(fe_dict, dict):
                if es.REFERENCE in fe_dict:
                    var_slope = 0
                    for variable, subgroup in fe_dict.items():
                        if variable == es.REFERENCE:
                            var_slope += subgroup
                        else:
                            var_slope += fe_dict[variable].get(
                                str(item[variable]),
                                0
                            )
                    lp += var_slope * item[key]
                else:
                    # If it is a regular item, it is not a slope. Simply add the matching coefficient.
                    sel_coefs = fe_dict.get(
                        str(int(item[key]))
                        if isinstance(item[key], bool)
                        else str(item[key]),
                        None
                    )
                    if isinstance(sel_coefs, dict):
                        for var2, dict2 in sel_coefs.items():
                            if var2 == es.REFERENCE:
                                lp += dict2
                            else:
                                lp += dict2.get(
                                    str(item[var2]),
                                    0
                                )
                    elif sel_coefs is None:
                        newkey = (
                            str(int(item[key]))
                            if isinstance(item[key], bool)
                            else str(item[key])
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
                        lp += sel_coefs

            elif key == 'intercept':
                lp += fe_dict
            else:
                lp += item[key] * fe_dict

            if (slogit_b4 != lp) & (verbose > 1):
                if key in item:
                    if var2:
                        print(
                            f'{key}-{item[key]}:'
                            f'{var2}-{item[var2]}: '
                            f'{lp-slogit_b4}'
                        )
                    else:
                        print(
                            f'{key}-{item[key]}: '
                            f'{lp-slogit_b4}'
                        )
                else:
                    print(f'{key}: {lp-slogit_b4}')

        for orig_var, fe_dict in (
            self.continuous_transformations[which].items()
        ):
            for coef_to_get, trafo in fe_dict.items():
                if (value := item[orig_var]) is not None:
                    contr = (
                        trafo(value) *
                        self.continuous_effects[which][orig_var][coef_to_get]
                    )
                    lp += contr

                    if (contr != 0) & (verbose > 1):
                        print(f'{coef_to_get}-{value}: {contr}')
                else:
                    print(f'{orig_var} yields None for {item}')

        return(lp)

    def _calculate_logit(
            self, offer: MatchRecord,
            which: str, verbose: Optional[int] = None,
            realization_intercept: Optional[float] = None
    ) -> float:
        """Calculate probability patient accepts"""
        if verbose is None:
            verbose = self.verbose

        slogit = self._calculate_lp(
            item=offer.__dict__,
            which=which,
            verbose=verbose,
            realization_intercept=realization_intercept
        )

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

    def simulate_esp_allocation(
            self, match_list: MatchListESP,
            n_kidneys_available=int
        ) -> Tuple[Optional[List[MatchRecord]], Dict[str, bool], bool]:
        """ Iterate over all match records. Do not model rescue allocation."""

        if self.simulate_rescue:
            offers_till_rescue = self.generate_offers_to_rescue(
                    donor=match_list.donor,
                    program = mgr.ESP
            )
        else:
            offers_till_rescue = 9999

        # TODO: in ESP, rescue is not triggerable currently; allocation then stops
        # and we move on to ETKAS for allocation. In reality, also very rarely
        # ESP-aged kidneys are allocated via rescue, so perhaps not an issue?
        acc_matchrecords, center_willingness, rescue_triggered = self._find_accepting_matchrecords(
            match_list,
            max_offers=offers_till_rescue,
            max_offers_per_center=5,
            n_kidneys_available=n_kidneys_available
        )

        if rescue_triggered:
            match_list.donor.rescue = True

        return acc_matchrecords, center_willingness, rescue_triggered

    def simulate_etkas_allocation(
        self, match_list: MatchListCurrentETKAS,
        n_kidneys_available: int,
        esp_rescue: bool = False,
        center_willingness: Optional[Dict[str, bool]] = None
    ) -> Optional[List[MatchRecord]]:
        """ Iterate over a list of match records. If we model rescue alloc,
            we simulate when rescue will be triggered and terminate
            recipient-driven allocation. In that case, we simulate further
            allocation with the donor identified as a rescue donor, and
            prioritize locally in Belgium / regionally in Germany.
        """
        # When not simulating rescue, offer to all patients on the match list
        if esp_rescue:
            offers_till_rescue=0
            print('ESP with rescue triggered; 0 offers made until rescue in ETKAS')
            print(match_list.donor.rescue)
            print(center_willingness)
        else:
            if self.simulate_rescue:
                offers_till_rescue = self.generate_offers_to_rescue(
                        match_list.donor,
                        program = mgr.ETKAS
                )
            else:
                offers_till_rescue = 9999

        # Allocate until `offers_till_rescue` rescue offers have been made.
        # This returns the match record for the accepting patient
        # (`acc_matchrecord`) if the graft was accepted, and a list of
        # centers willing to accept the organ (determined at the
        # center level)
        acc_matchrecords, center_willingness, _ = (
            self._find_accepting_matchrecords(
                match_list,
                max_offers=offers_till_rescue,
                max_offers_per_center=5,
                n_kidneys_available=n_kidneys_available,
                center_willing_to_accept=center_willingness
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
        if (
            n_kidneys_accepted < n_kidneys_available
        ):
            match_list._initialize_extalloc_priorities()
            match_list.donor.rescue = True

            if acc_matchrecords is not None:
                n_kidneys_available -= n_kidneys_accepted

            # If rescue is triggered, prioritize remaining waitlist
            # on rescue priority (i.e. local in Belgium, regional in DE)
            acc_matchrecords_rescue, _, _ = self._find_accepting_matchrecords(
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
    ) -> Tuple[Optional[List[MatchRecord]], Dict[str, bool], bool]:
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
        rescue_triggered: bool = False
        n_rejections_per_center = defaultdict(int)
        if center_willing_to_accept is None:
            center_willing_to_accept = {}

        if len(match_list) == 0:
            return None, {}, rescue_triggered

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

            # Check whether offer is profile compatible.
            if not match_object.profile_compatible:
                match_object.set_acceptance(reason=cn.FP)
                continue

            # For profile compatible offers, initialize acceptance information
            if hasattr(match_object, '_initialize_acceptance_information'):
                match_object._initialize_acceptance_information()

            # 1) determine whether center is willing to accept
            if not (
                match_object.__dict__[cn.RECIPIENT_CENTER] in
                center_willing_to_accept
            ):
                if self.determine_center_acceptance(
                    match_object,
                    k_previous_center_rejections=sum(
                        not value for _, value in center_willing_to_accept.items()
                    )
                ):
                    center_willing_to_accept[
                        match_object.__dict__[cn.RECIPIENT_CENTER]
                    ] = True
                else:
                    center_willing_to_accept[
                        match_object.__dict__[cn.RECIPIENT_CENTER]
                    ] = False
                    count_rejections_total += 1

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
                            return match_objects, center_willing_to_accept, rescue_triggered
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
            return match_objects, center_willing_to_accept, rescue_triggered

        # In case of no acceptance in regular allocation, return
        # center willingness to accept (to trigger rescue)
        return None, center_willing_to_accept, rescue_triggered


    def fe_coefs_to_dict(self, level: pd.Series, coef: pd.Series):
        """Process coefficients to dictionary"""
        fe_dict = {}
        for lev, val in zip(level, coef):
            if ':' in str(lev):
                l1, lv2 = lev.split(':')
                var2, l2 = lv2.split('-')
                if len(l1) == 0:
                    if var2 in fe_dict:
                        fe_dict[var2].update(
                            {l2: val}
                        )
                    else:
                        fe_dict[var2] = {l2: val}
                elif l1 in fe_dict:
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
                if not pd.isnull(level).all() and any(level.str.contains(':', regex=False)):
                    if isinstance(lev, float) and isnan(lev):
                        fe_dict = {es.REFERENCE: val}
                    else:
                        fe_dict[lev] = {es.REFERENCE: val}
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
            self.__dict__[k]['coef'].fillna(value=0, inplace=True)

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
