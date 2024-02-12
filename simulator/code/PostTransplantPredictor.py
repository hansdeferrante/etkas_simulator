#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13

@author: H.C. de Ferrante
"""

from typing import Callable, Dict, Tuple, List, Optional
from math import isnan, log, exp
from datetime import datetime, timedelta
from functools import reduce
from copy import deepcopy
from collections import defaultdict
import numpy as np
import pandas as pd

from simulator.code.utils import round_to_int
import simulator.magic_values.column_names as cn
import simulator.magic_values.magic_values_rules as mgr
import simulator.magic_values.etkass_settings as es
from simulator.magic_values.rules import FRACTIONS_RETURN_WAITINGTIME
from simulator.code.functions import construct_piecewise_term
from simulator.code.entities import Patient
from simulator.code.AllocationSystem import MatchRecord


def inv_logit(xvals: float) -> float:
    """Calculates probability for logit"""
    return np.exp(xvals) / (1 + np.exp(xvals))


class PostTransplantPredictor:
    """
        Class for predicting post-transplant survival based on Weibull
        models and for generating a synthetic patient based on historically
        observed retransplantations.

    Attributes   #noqa
    ----------

    Methods
    -------
    calculate_survival(offer, time, verbosity):
        calculate survival probability at time
    calculate_survival_time(offer, surv, verbosity):
        calculate survival time
    simulate_failure_date(offer, surv, verbosity):
        calculate random failure date

    """
    def __init__(
        self,
        offset_ids_transplants: int,
        seed: int,
        retransplants: Optional[Dict[int, Patient]] = None,
        dict_paths_coefs: Dict[str, str] = es.POSTTXP_SURV_PATHS,
        dict_paths_relist: Dict[str, str] = es.POSTTXP_RELISTPROB_PATHS,
        discrete_match_vars: Optional[List[str]] = None,
        cvars_trafos: Optional[List[Callable]] = None,
        cvars_caliper: Optional[List[float]] = None,
        continuous_match_vars_rec: Optional[List[str]] = None,
        continuous_match_vars_off: Optional[List[str]] = None,
        min_matches: int = 10,
        only_exiting_after: bool = False,
        exit_after_thresh: int = 2*365
    ):

        self._initialize_weibull_coefs(
            dict_paths_coefs=dict_paths_coefs
        )
        self._initialize_time_to_relist(
            dict_paths_relist=dict_paths_relist
        )
        self.offset_ids_transplants = offset_ids_transplants

        self.only_exiting_after = only_exiting_after
        self.exit_after_thresh = exit_after_thresh

        self.rng_surv = np.random.default_rng(seed)
        self.rng_random_subject = np.random.default_rng(seed+1)

        if retransplants:

            self.synth_regs = 1

            # Discrete match variables, on which we try to match
            self.discrete_match_vars = discrete_match_vars

            # Continuous match variables, matching based on Mahalanobis dist.
            assert continuous_match_vars_off and continuous_match_vars_rec, \
                'Continuous match variables have to be supplied'
            self.cvars_rec = continuous_match_vars_rec
            self.cvars_off = continuous_match_vars_off

            # Transformations to apply to continuous variables
            if cvars_trafos is None:
                self.cvars_trafos = (
                    es.identity for _ in self.cvars_rec
                )
            else:
                self.cvars_trafos = cvars_trafos

            # Calipers to apply to continuous match variables
            # On top of Mahalanobis distance matching
            self.cvars_caliper = cvars_caliper

            assert len(self.cvars_rec) == len(self.cvars_off), \
                'The same number of continuous variables has to be ' \
                'supplied for the patient & offer information'

            # Minimum number of matches required
            self.min_matches = min_matches

            # List of retransplants.
            self.retransplants = retransplants
            k_retransplants_removed = 0
            for k in list(self.retransplants.keys()):
                if not self.retransplants[k].__dict__[cn.TIME_SINCE_PREV_TXP]:
                    self.retransplants.pop(k)
                    k_retransplants_removed += 1
            print(
                f'Removed {k_retransplants_removed} retransplants '
                f'for which time since previous transplant is unknown'
            )

            # Save discrete variables as dictionary
            if self.discrete_match_vars:
                self.dvals = {
                    dvar: np.array(
                        [
                            p.__dict__[dvar] for p
                            in retransplants.values()
                        ]
                    )
                    for dvar in self.discrete_match_vars
                }
                self.allvars = self.cvars_rec + self.discrete_match_vars
            else:
                self.dvals = None
                self.allvars = self.cvars_rec

            # Save continuous variables as matrix, and
            # calculate statistics needed for Mahalanobis
            self.cvals_rec = np.vstack(
                [
                    np.array(
                        [p.__dict__[cvar] for cvar in self.cvars_rec],
                        dtype=float
                    ) for p in retransplants.values()
                ]
            )
            for idx, trafo in enumerate(self.cvars_trafos):
                self.cvals_rec[:, idx] = trafo(self.cvals_rec[:, idx])

            self.cvar_invcov = np.linalg.inv(
                np.cov(self.cvals_rec.T)
            )

            # Real exit times
            self.retxs_real_exit_times = (
                np.array(
                    [
                        p.future_statuses.return_time_to_exit(
                            exit_statuses=es.EXITING_STATUSES
                        ) for p
                        in retransplants.values()
                    ],
                    dtype=float
                )
            )

            # Determine number of matches
            self.match_var_counts = defaultdict(int)

    def calculate_survival(
            self,
            offer: MatchRecord,
            time: float,
            verbosity: int = 0):
        """ Calculate survival probability at time for
            offer based on Weibull model.
        """

        verbose = True if verbosity == 2 else False

        scale = self._calculate_weibull_scale(
                        offer,
                        # stratum=offer.__dict__[cn.URGENCY_CODE],
                        verbose=verbose,
                        transform=True
                    )
        if verbosity > 0:
            print(f'Scale parameter: {scale}')

        shape = self._calculate_weibull_shape(
            offer,
            # stratum=offer.__dict__[cn.URGENCY_CODE],
            verbose=verbose,
            transform=True
        )
        if verbosity > 0:
            print(f'Shape parameter: {shape}')

        surv = exp(-(time / scale) ** shape)

        if verbosity > 0:
            print(f'Calculated survival: {surv}')

        return surv

    def calculate_survival_time(
            self,
            offer: MatchRecord,
            surv: float,
            verbosity: int = 0):
        """ Calculate offer-specific survival time for a
            given survival probability surv
        """

        verbose = True if verbosity == 2 else False

        scale = self._calculate_weibull_scale(
                        offer,
                        # stratum=offer.__dict__[cn.URGENCY_CODE],
                        verbose=verbose,
                        transform=True
                    )
        if verbosity > 0:
            print(f'Scale parameter: {scale}')

        shape = self._calculate_weibull_shape(
            offer,
            # stratum=offer.__dict__[cn.URGENCY_CODE],
            verbose=verbose,
            transform=True
        )
        if verbosity > 0:
            print(f'Shape parameter: {shape}')

        time = scale*(-log(surv))**(1/shape)

        if verbosity > 0:
            print(f'Calculated survival time: {time}')

        return time

    def simulate_failure_cause_and_relisttime(
        self,
        offer: MatchRecord,
         time: float,
        stratum: Optional[str] = None
    ) -> Tuple[str, Optional[float]]:
        """ Simulate a failure cause, i.e. calculate the probability
            that an offer results in a patient failure, and return
            patient failure with this probability
        """

        # Select right relisting probabilities
        relist_info = self.times_to_relist[
            offer.__dict__.get(stratum) if stratum else None
        ]

        # Determine the right horizon to sample from
        horizon = 0
        for key in relist_info['rel_time_event'].keys():
            if time < key:
                horizon = key
                break

        prob = self.rng_surv.random()

        # Draw events that occured, based on sampled probability
        event_prob_gt = np.greater(
            prob,
            relist_info['event_prob'][horizon]
        )

        # Final event time is the time difference between reregistration
        # and retransplantation/death
        if event_prob_gt.any():
            diff_time = relist_info['rel_time_event'][horizon][
                event_prob_gt
            ][-1] * time
            if diff_time > 0:
                return cn.PATIENT_RELISTING, time - diff_time
        return cn.PATIENT_FAILURE, None

    def simulate_failure_date(
            self,
            offer: MatchRecord,
            current_date: datetime,
            max_surv_hor: float = 20*365,
            verbosity: int = 0
    ) -> Tuple[Optional[datetime], Optional[datetime], Optional[str]]:
        """ Simulate a failure date, a relisting date, and
            failure cause (graft/patient failure)
        """

        surv = self.rng_surv.random()

        surv_offset = self.calculate_survival_time(
            offer=offer,
            surv=surv,
            verbosity=0
        )

        if surv_offset <= max_surv_hor:
            surv_cause, relist_time = (
                self.simulate_failure_cause_and_relisttime(
                    offer=offer,
                    time=surv_offset
                )
            )
            return (
                current_date + timedelta(days=surv_offset),
                current_date + timedelta(
                    days=relist_time if relist_time else 0
                )
                if surv_cause == cn.PATIENT_RELISTING else None,
                surv_cause
            )
        else:
            return None, None, None

    def _calculate_weibull_param(
            self, offer: MatchRecord,
            param: str, stratum: Optional[str] = None,
            verbose=False
    ) -> float:
        """
            Calculate LP for Weibull model (both scale & shape)
        """

        linpred = 0

        for k, d in self.fixed_effects[stratum][param].items():
            linpred_b4 = linpred
            var2 = None
            if isinstance(d, dict):
                sel_coefs = d.get(
                    str(int(offer.__dict__[k]))
                    if isinstance(offer.__dict__[k], bool)
                    else str(offer.__dict__[k]),
                    0
                )
                if isinstance(sel_coefs, dict):
                    (var2, dict2), = sel_coefs.items()
                    linpred += dict2.get(
                        offer.__dict__[var2],
                        0
                    )
                else:
                    linpred += d.get(
                        str(int(offer.__dict__[k]))
                        if isinstance(offer.__dict__[k], bool)
                        else str(offer.__dict__[k]),
                        0
                    )
            elif k == 'intercept':
                linpred += d
            else:
                linpred += offer.__dict__[k] * d

            if (linpred_b4 != linpred) & verbose:
                if isnan(linpred):
                    print(
                        f'Linear predictor becomes nan when'
                        f' processing {k} ({offer.__dict__[k]})'
                    )
                    exit()

                if k in offer.__dict__:
                    if var2:
                        print(
                            f'{k}-{offer.__dict__[k]}:{var2}-'
                            f'{offer.__dict__[var2]}: {linpred-linpred_b4}')
                    else:
                        print(f'{k}-{offer.__dict__[k]}: {linpred-linpred_b4}')
                else:
                    print(f'{k}: {linpred-linpred_b4}')

        for orig_var, d in self.continuous_transformations[
            stratum
        ][param].items():
            for coef_to_get, trafo in d.items():

                contr = (
                    trafo(offer.__dict__[orig_var]) *
                    self.continuous_effects[
                        stratum
                    ][param][orig_var][coef_to_get]
                )
                linpred += contr

                if (contr != 0) & verbose:
                    print(f'{coef_to_get}-{offer.__dict__[orig_var]}: {contr}')

        return linpred

    def _calculate_weibull_scale(
            self, offer: MatchRecord,
            stratum: Optional[str] = None,
            param: str = 'scale',
            verbose=False,
            transform: bool = True
    ) -> float:
        """Calculate weibull scale parameter for an offer"""
        linpred = self._calculate_weibull_param(
            offer=offer, stratum=stratum, param=param,
            verbose=verbose
        )
        if transform:
            return exp(linpred)
        else:
            return linpred

    def _calculate_weibull_shape(
            self, offer: MatchRecord,
            stratum: Optional[str]=None,
            param: str = 'shape',
            verbose=False,
            transform: bool = True
    ) -> float:
        """Calculate weibull shape parameter for an offer"""
        linpred = self._calculate_weibull_param(
            offer=offer, stratum=stratum, param=param,
            verbose=verbose
        )
        if transform:
            return 1/linpred
        else:
            return linpred

    def fe_coefs_to_dict(self, level: np.ndarray, coef: np.ndarray):
        """
            Convert numpy arrays with fixed effects coefficients and levels to
            a dictionary
        """
        fe_dict = {}
        for level, val in zip(level, coef):
            if isinstance(level, str) and ':' in level:
                lev1, lv2 = level.split(':')
                var2, lev2 = lv2.split('-')

                if lev1 in fe_dict:
                    if var2 in fe_dict[lev1]:
                        fe_dict[lev1][var2].update(
                            {lev2: val}
                        )
                    else:
                        fe_dict[lev1].update(
                            {var2: {lev2: val}}
                        )
                else:
                    fe_dict.update(
                        {lev1: {var2: {lev2: val}}}
                    )
            else:
                fe_dict.update(
                    {level: val}
                )
        return fe_dict

    def _initialize_weibull_coefs(
        self,
        dict_paths_coefs: Dict[str, str]
    ):
        """ Initialize the Weibull coefficients for
            recipient (rd) and center-driven (cd) allocation,
            separately for HU and T. This creates separate
            dictionaries for shape and scale parameters.
        """

        for k, v in dict_paths_coefs.items():
            d_weibull: pd.DataFrame = pd.read_csv(v, dtype='str')
            d_weibull['coef'] = d_weibull['coef'].astype(float)
            self.__dict__[k] = {
                'shape': d_weibull.loc[
                    d_weibull.loc[:, 'param_type'] == 'shape', :
                    ].drop(columns='param_type'),
                'scale': d_weibull.loc[
                    d_weibull.loc[:, 'param_type'] == 'coef', :
                    ].drop(columns='param_type')
            }

        # Create dictionary for fixed effects
        self.fixed_effects = {}
        gamma_dist_names = list(dict_paths_coefs.keys())
        for dist in gamma_dist_names:
            self.fixed_effects[dist] = {}
            for param_type in ('scale', 'shape'):
                self.fixed_effects[dist].update(
                    {
                        param_type: (
                            self.__dict__[dist][param_type].loc[
                                ~ self.__dict__[dist][param_type]
                                .variable_transformed.notna()
                            ].groupby('variable').
                            apply(
                                lambda x: self.fe_coefs_to_dict(
                                    x['level'],
                                    x['coef']
                                )
                            ).to_dict()
                        )
                    }
                )

        # Convert dictionaries with a single key to a value.
        for gamma_dict in self.fixed_effects.values():
            for td in gamma_dict.values():
                for k, v in td.items():
                    if (
                        isinstance(list(v.keys())[0], float) and
                        isnan(list(v.keys())[0])
                    ):
                        td[k] = list(v.values())[0]

        self.continuous_transformations = {}
        self.continuous_effects = {}

        for dist in gamma_dist_names:
            self.continuous_transformations[dist] = {}
            self.continuous_effects[dist] = {}
            for param_type in self.fixed_effects[dist].keys():
                self.continuous_transformations[dist][param_type] = (
                    self.__dict__[dist][param_type].loc[
                        self.__dict__[dist][
                            param_type
                        ].variable_transformed.notna()].
                    groupby('variable').
                    apply(
                        lambda x: {
                            k: PostTransplantPredictor.
                            construct_transformation(v)
                            for k, v in zip(
                                x['variable_transformed'],
                                x['variable_transformation']
                            )
                        }
                    ).to_dict()
                )

                self.continuous_effects[dist][param_type] = (
                    self.__dict__[dist][param_type].loc[
                        self.__dict__[dist][param_type].
                        variable_transformed.notna()
                    ].groupby('variable').
                    apply(
                        lambda x: dict(
                                zip(
                                    x['variable_transformed'],
                                    x['coef']
                                )
                            )
                    ).to_dict()
                )

    def _initialize_time_to_relist(
        self,
        dict_paths_relist: Dict[str, str] = es.POSTTXP_RELISTPROB_PATHS
    ):
        """
            Initialize coefficients for logistic regression
            for fraction death (per country only)
        """
        self.times_to_relist = {}
        for k, v in dict_paths_relist.items():
            data_ = pd.read_csv(v, dtype='str')
            data_.loc[:, ['rel_time_event', 'event_prob', 'strata']] = (
                data_.loc[
                    :,
                    ['rel_time_event', 'event_prob', 'strata']
                ].astype(float)
            )
            self.times_to_relist[k] = (
                data_
                .groupby('strata', sort=True)[['rel_time_event', 'event_prob']]
                .agg(
                    lambda x: np.array(x, dtype=float)
                ).to_dict()
            )

    def _calculate_maha_distance(
        self, indict: Dict
    ):
        """Calculate the Mahalanobis distance for an offer"""

        # Create x vector for observation
        x = np.array(
            [indict[col] for col in self.cvars_off],
            dtype=float
        )
        for idx, trafo in enumerate(self.cvars_trafos):
            x[idx] = trafo(x[idx])

        # Normalize x by values of other points.
        x_mu = (x - self.cvals_rec)

        # Calculate Mahalanobis distance
        return np.sqrt(
            np.diag(
                np.dot(
                    np.dot(x_mu, self.cvar_invcov),
                    x_mu.T
                )
            )
        )

    def check_caliper(
        self, indict: Dict
    ):
        """
            Check whether difference for continuous variables falls
            within caliper
        """
        if self.cvars_caliper:
            xvals = np.array(
                [indict[col] for col in self.cvars_off],
                dtype=float
            )
            for idx, trafo in enumerate(self.cvars_trafos):
                xvals[idx] = trafo(xvals[idx])
            caliper_compatible = np.array(
                abs(self.cvals_rec-xvals) <= self.cvars_caliper,
                dtype=bool
            )
            return caliper_compatible.all(axis=1)
        else:
            return np.ones((self.cvals_rec.shape[0], 1), dtype='bool')

    def generate_synthetic_reregistration(
        self,
        relist_date: datetime,
        fail_date: datetime,
        curr_date: datetime,
        offer: MatchRecord,
        verbosity: int = 0
    ) -> Patient:
        """
            Generate a synthetic reregistration based on Mahalanobis distance
            matching
        """

        offer = deepcopy(offer)

        # Add time to reregistration, time from reregistration to
        # real event time, and time to real event time from current date.
        offer.__dict__[cn.TIME_TO_REREG] = (
            relist_date - curr_date
        ) / timedelta(days=1)
        offer.__dict__[cn.TIME_LIST_TO_REALEVENT] = (
            fail_date - relist_date
        ) / timedelta(days=1)
        offer.__dict__[cn.REREG_RETURN_DIAL_TIME] = (
            offer.__dict__[cn.TIME_TO_REREG] <
            es.CUTOFF_REREG_RETURN_DIAL_TIME
        )

        # Reregistration ID.
        id_rereg = self._find_similar_id_reregistration(
            offer=offer, verbosity=0
        )

        # Make deep copy of the to be matched patient
        synth_patient = deepcopy(self.retransplants[id_rereg])
        synth_patient.reset_matchrecord_info()

        # Ignore patient profile for retransplant candidate.
        # We copy it over from the original patient.
        if synth_patient.future_statuses:
            synth_patient.future_statuses.remove_status_types(
                remove_event_types=es.STATUS_TYPES_TO_REMOVE
            )

        # Copy over patient specific attributes. First
        # get the patient's MMP (needed to obtain match frequency).
        offer.patient.get_et_mmp()
        for attr in es.POSTTXP_COPY_VARS:
            synth_patient.__dict__[attr] = offer.patient.__dict__[attr]

        # Fix other patient attributes
        synth_patient.prev_txp_living = False
        synth_patient.prev_txp_ped = offer.patient.__dict__[cn.R_PED]
        synth_patient.__dict__[cn.LISTING_DATE] = relist_date
        synth_patient.__dict__[cn.PREV_TX_DATE] = curr_date
        synth_patient.__dict__[cn.TIME_SINCE_PREV_TXP] = (
            relist_date - curr_date
        ) / timedelta(days=1)
        synth_patient.__dict__[cn.TYPE_RETX] = cn.RETRANSPLANT_DURING_SIM
        synth_patient.__dict__[cn.RETRANSPLANT] = 1
        synth_patient.age_days_at_listing = round_to_int(
            (relist_date - synth_patient.r_dob) / timedelta(days=1)
            )
        synth_patient.listing_offset = (
            synth_patient.__dict__[cn.LISTING_DATE] -
            synth_patient.sim_set.SIM_START_DATE
        ) / timedelta(days=1)

        # Fix new dialysis date. Do this based on matched synthetic re-registration.
        delta_dialdate_relistingdate: float = (
            synth_patient.get_dial_time_at_listing()
            if (
                synth_patient.get_dial_time_at_listing() is not None and
                not isnan(synth_patient.get_dial_time_at_listing())
            )
            else 0
        )
        if (pot_dial_date := relist_date + timedelta(days=delta_dialdate_relistingdate)) <= curr_date:
            new_dial_date = curr_date
        else:
            new_dial_date = pot_dial_date
        synth_patient.set_dial_date(new_dial_date)

        # Fix previously accrued waiting time, based on new dialysis date.
        time_to_redial = (new_dial_date - curr_date) / timedelta(days=1)
        for frac_fun, frac in FRACTIONS_RETURN_WAITINGTIME.items():
            if frac_fun(time_to_redial):
                break
        else:
            frac = 0

        total_waittime = (
            offer.patient.__dict__[cn.PREVIOUS_WT] +
            offer.__dict__[cn.YEARS_ON_DIAL]
        )
        synth_patient.__dict__[cn.PREVIOUS_T] = (
            total_waittime * frac
        )

        if verbosity > 0:
            print('Synthetic patient info:')
            print(synth_patient.__dict__)
            print('Transplanted patient info:')
            print(offer.patient.__dict__)
            print(f'curr_date: {curr_date}, relist_date: {relist_date}, delta: {delta_dialdate_relistingdate}, new_dial_date: {new_dial_date}, {time_to_redial}, frac: {frac}, total: {total_waittime}, return: {frac*total_waittime}')

        # If we force matches to exit after, and real time-to-event is
        # within the threshold schedule a terminal death event
        if self.only_exiting_after & (
            offer.__dict__[cn.TIME_LIST_TO_REALEVENT] <=
            self.exit_after_thresh
        ):
            synth_patient.schedule_death(
                fail_date=fail_date
            )

        if synth_patient.id_registration:
            synth_patient.id_registration = (
                self.offset_ids_transplants +
                self.synth_regs
            )
            self.synth_regs += 1

        return synth_patient

    def _find_similar_id_reregistration(
        self, offer: MatchRecord, verbosity: int = 0
    ) -> int:
        """
            Return the registration ID for a similar
            real retransplantation, based on Mahalanobis distance
        """

        # Calculate Mahalanobis distance of offer to candidates
        maha_distances = self._calculate_maha_distance(indict=offer.__dict__)

        # Check whether patient is caliper compatible
        caliper_compatible = self.check_caliper(indict=offer.__dict__)

        # If we match only among patients with time-to-exit greater than
        # the to-be-matched offer
        if (
            self.only_exiting_after and
            (
                offer.__dict__[cn.TIME_LIST_TO_REALEVENT] <
                self.exit_after_thresh
            )
        ):
            allowed_matches = (
                offer.__dict__[cn.TIME_LIST_TO_REALEVENT] <=
                self.retxs_real_exit_times
            ) & caliper_compatible
        else:
            allowed_matches = caliper_compatible

        # If we also have to match on discrete variables
        if self.dvals:
            # Check whether discrete variables match between offer
            # and existing retransplantations
            matching_dvars = [
                v == offer.__dict__[k] for k, v in self.dvals.items()
            ]
            dvars_to_match = len(self.dvals)

            # Match on as many discrete vars as needed
            # until enough matches are found.
            n_matches = 0
            while (n_matches < self.min_matches) and (dvars_to_match > 0):
                d_matching = reduce(
                    lambda a, b: a & b,
                    matching_dvars[0:dvars_to_match]
                    ) & allowed_matches
                n_matches = sum(d_matching)
                if n_matches >= self.min_matches:
                    break
                dvars_to_match -= 1
            else:
                d_matching = allowed_matches

            self.match_var_counts[dvars_to_match] += 1

            # Print out on how many characteristics we matched, if
            # verbose and not all.
            if (dvars_to_match < len(self.dvals)) and (verbosity > 0):
                print(f'Matched on {dvars_to_match} characteristics')
        else:
            d_matching: np.ndarray = allowed_matches

        # Apply very large penalty if not matching.
        penalty = 1e9 * (~ d_matching).astype(float)

        # Retrieve the IDs of the closest compatible matches
        idx = np.argpartition(
            abs(maha_distances + penalty),
            self.min_matches
        )
        keys = np.fromiter(self.retransplants.keys(), dtype=int)
        keys_closest = keys[idx[:self.min_matches]]

        if verbosity > 1:
            print('To match:')
            print(
                ' '.join(
                    [
                        f"{col}: {offer.__dict__[col]}"
                        for col in self.cvars_off
                    ]
                )
            )
            print('Matches:')
            for key in keys_closest:
                p = self.retransplants[key]
                print(
                    list(
                        (f'{col}: {p.__dict__[col]}' for col in self.allvars))
                )
                time_to_exit = (
                    p.future_statuses.return_time_to_exit(es.TERMINAL_STATUSES)
                )
                print(
                    f'Time-to-terminal status: {time_to_exit}'
                )
                print(
                    p.future_statuses.return_status_types(
                        event_types=['URG']
                        )[-1].status_value
                )
                input("Press enter to see next patient")

        # Return a random comparable patient.
        return self.rng_random_subject.choice(keys_closest)

    @staticmethod
    def construct_transformation(trafo: str) -> Callable:
        """Construct needed transformation"""

        if trafo == 'log':
            return log
        if '_log' in trafo:
            return construct_piecewise_term(
                trafo=trafo,
                trafo_x=log
            )

        return construct_piecewise_term(
            trafo=trafo,
            trafo_x=es.identity
        )
