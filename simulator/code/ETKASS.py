#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 16-08-2022

@author: H.C. de Ferrante
"""

import os
from itertools import product
from datetime import timedelta, datetime
from typing import Optional, Dict, List, ValuesView
from collections import defaultdict
from statistics import median
import time
import typing

from simulator.code.utils import DotDict, round_to_decimals
from simulator.code.current_etkas.CurrentETKAS import \
    MatchListCurrentETKAS, MatchRecordCurrentETKAS
from simulator.code.AcceptanceModule import AcceptanceModule
from simulator.code.PostTransplantPredictor import PostTransplantPredictor
from simulator.code.EventQueue import EventQueue
from simulator.code.Event import Event
from simulator.code.load_entities import \
        preload_profiles, preload_status_updates, load_patients, \
        load_donors, load_retransplantations, HLASystem, \
        load_balances
from simulator.code.entities import (
    Patient
)
from simulator.code.SimResults import SimResults
import simulator.magic_values.column_names as cn
import simulator.magic_values.etkass_settings as es
from simulator.magic_values.rules import (
    BLOOD_GROUP_INCOMPATIBILITY_DICT
)

import simulator.magic_values.rules as r

if typing.TYPE_CHECKING:
    from simulator.code import AllocationSystem


class ActiveList:
    """
        Class which tracks actively listed patients, per blood type.
        The order of the list is on dialysis time (if it exists).
        Retrieving patients from the active lists prevents having to
        iterate over all patients, and speeds up sorting.
    """
    def __init__(self, init_patients: Dict[int, Patient]):
        self._active_lists: Dict[str, Dict[int, Patient]] = {
            bg: dict() for bg in es.ALLOWED_BLOODGROUPS
        }
        for id_reg, pat in init_patients.items():
            if pat.active:
                self._active_lists[pat.bloodgroup][id_reg] = pat
        self.resort_lists()

    def get_active_list(self, bg: str) -> ValuesView[Patient]:
        """Returning patients, ordered with priority for bg"""
        return self._active_lists[bg].values()

    def add_to_active_list(self, identifier: int, pat: Patient) -> None:
        """Adding an identifier to the active list"""
        self._active_lists[pat.bloodgroup][identifier] = pat

    def pop_from_active_lists(self, identifier: int) -> None:
        """Remove a patient identifier from active list"""
        for active_list in self._active_lists.values():
            k = active_list.pop(identifier, None)
            if k is None:
                return None

    def is_active(self, identifier: int) -> bool:
        for active_list in self._active_lists.values():
            if identifier in active_list:
                return True
            else:
                return False
        return False

    def resort_lists(self):
        for bg, active_list in self._active_lists.items():

            # Sort dictionary by patient (i.e. match MELD)
            self._active_lists[bg] = dict(
                sorted(
                    active_list.items(),
                    key = lambda x: (
                        x[1].__dict__[cn.DATE_FIRST_DIAL] or datetime.now()
                    )
                )
            )

class ETKASS:
    """Class which implements an etkass simulator

    Attributes   #noqa
    ----------
    patients: List[Patient]
        list of patients
    sim_set: DotDict
        simulation settings
    max_n_patients: Optional[int]
        maximum number of patients to read in
    max_n_statuses: Optional[int]
        maximum number of statuses to read in
    sim_time: float
        simulation time
    ...
    """
    def __init__(
        self,
        sim_set: DotDict,
        max_n_patients: Optional[int] = None,
        verbose: Optional[bool] = True
    ):

        # Read in simulation settings
        self.sim_set = sim_set
        self.init_unacc_mrs = sim_set.get('INITIALIZE_UNACCEPTABLE_MRS', False)
        self.sim_start_date = sim_set['SIM_START_DATE']
        self.verbose = verbose
        self.sim_rescue = sim_set.get('SIMULATE_RESCUE', False)
        self.hla_system = HLASystem(sim_set)
        self.bal_system = load_balances(sim_set)

        # Set up times, and resort list every 30 days
        self.sim_time = 0
        self._time_active_list_resorted = 0
        self._resort_every_k_days = 30

        # Keep track of number of obligations generated
        self.n_obligations = 0

        # Set up (empty) event queue
        self.event_queue = EventQueue()

        self.max_sim_time = (
            (
                self.sim_set.SIM_END_DATE - self.sim_start_date
            ) / timedelta(days=1)
        )

        # Initialize the acceptance module
        self.acceptance_module = AcceptanceModule(
            seed=self.sim_set.SEED,
            center_acc_policy=str(sim_set.CENTER_ACC_POLICY),
            patient_acc_policy=str(sim_set.PATIENT_ACC_POLICY),
            separate_huaco_model=bool(sim_set.SEPARATE_HUACO_ACCEPTANCE),
            separate_ped_model=bool(sim_set.SEPARATE_PED_ACCEPTANCE),
            simulate_rescue=self.sim_rescue
        )
        if self.verbose:
            print('Loaded acceptance module')

        # Load donors
        self.donors = load_donors(
            sim_set=self.sim_set,
            hla_system=self.hla_system
        )

        # Load patients, and initialize a DataFrame with
        # blood group compatibilities.
        if self.verbose:
            print('Loading patient registrations, statuses, and profiles')
        self.patients = load_patients(
            sim_set=self.sim_set,
            nrows=max_n_patients,
            hla_system=self.hla_system
        )

        # Preload status updates
        preload_status_updates(
            patients=self.patients,
            sim_set=sim_set
        )

        # Preload profile information
        preload_profiles(
            patients=self.patients,
            sim_set=sim_set
        )

        # If simulating re-transplantations, load in retransplantations
        # Note that re-transplantations future to simulation start time
        # are not loaded in by load_patients if SIM_RETX.
        if sim_set.SIM_RETX:
            self.n_events = 0

            # Load retransplantations
            if self.verbose:
                print('Loading retransplantations')
            self.retransplantations = load_retransplantations(
                sim_set=self.sim_set,
                hla_system=self.hla_system,
                nrows=max_n_patients
            )
            preload_status_updates(
                patients=self.retransplantations,
                sim_set=sim_set,
                end_date_col='LOAD_RETXS_TO'
            )
            self.remove_patients_without_statusupdates(
                pat_dict_name='retransplantations'
            )

        # Trigger all historic updates for the real patient list.
        for pat in self.patients.values():
            pat.trigger_historic_updates()

        # Remove patients from real registrations that are
        # not initialized / never have any status update.
        self.remove_patients_without_statusupdates(
            pat_dict_name='patients',
            verbosity=1
        )

        # Initialize EventQueue with patient updates
        # for all patients, active or not
        for pat_id, pat in self.patients.items():
            get_next_update_time = pat.get_next_update_time()
            if get_next_update_time:
                self.event_queue.add(
                    Event(
                        type_event=cn.PAT,
                        event_time=get_next_update_time,
                        identifier=pat_id
                    )
                )

        # Initialize queue of active patients
        self.active_lists = ActiveList(
            init_patients=self.patients
        )
        if self.verbose:
            print('Initialized patients')

        # Initialize the post-transplant module.
        if sim_set.SIM_RETX:
            self.ptp = PostTransplantPredictor(
                seed=self.sim_set.SEED,
                offset_ids_transplants=max(
                    max(self.patients.keys()),
                    max(self.retransplantations.keys())
                ),
                retransplants=self.retransplantations,
                discrete_match_vars=es.POSTTXP_DISCRETE_MATCH_VARS,
                cvars_trafos=es.POSTTXP_TRANSFORMATIONS,
                cvars_caliper=es.POSTTXP_MATCH_CALIPERS,
                continuous_match_vars_rec=(
                    es.POSTTXP_CONTINUOUS_MATCH_VARS[cn.RETRANSPLANT]
                ),
                continuous_match_vars_off=(
                    es.POSTTXP_CONTINUOUS_MATCH_VARS[cn.OFFER]
                ),
                min_matches=es.POSTTXP_MIN_MATCHES
            )

        # Schedule events for donors.
        for don_id, don in self.donors.items():
            self.event_queue.add(
                Event(
                    type_event=cn.DON,
                    event_time=don.arrival_at(
                        self.sim_start_date
                        ),
                    identifier=don_id
                )
            )

        # Initialize simulation results & path to match list file.
        self.sim_results = SimResults(
            cols_to_save_exit=es.OUTPUT_COLS_EXITS + (self.sim_set.LAB_MELD,),
            cols_to_save_discard=es.OUTPUT_COLS_DISCARDS,
            cols_to_save_patients=es.OUTPUT_COLS_PATIENTS + (self.sim_set.LAB_MELD,),
            sim_set=self.sim_set
        )
        self.match_list_file = (
            str(self.sim_set.RESULTS_FOLDER) +
            str(self.sim_set.PATH_MATCH_LISTS)
        )

    def remove_patients_without_statusupdates(
        self, pat_dict_name: str, verbosity=0
    ) -> None:
        """Removes patients from dict without any updates."""

        n_removed = 0
        for id_reg, pat in list(self.__dict__[pat_dict_name].items()):
            if (
                not pat.is_initialized() or
                pat.__dict__[cn.EXIT_STATUS] is not None
            ):
                n_removed += 1

                del self.__dict__[pat_dict_name][id_reg]
        if self.verbose:
            print(
                f'Removed {n_removed} patients from self.{pat_dict_name} '
                f"that already exited or who don't have any status updates "
                f"(e.g., never any known biomarker)"
                )

    def get_active_list(self, bg: str) -> ValuesView[Patient]:
        return self.active_lists.get_active_list(bg)

    def simulate_allocation(
            self,
            verbose: bool = False,
            print_progress_every_k_days=90
    ) -> Optional[MatchListCurrentETKAS]:
        """Simulate the allocation algorithm"""

        # Set seed and maintain count for how many days were simulated.
        next_k_days = 1

        # Save start time, and print simulation period.
        start_time = time.time()
        print(
            f'Simulating ETKAS from {self.sim_start_date.strftime("%Y-%m-%d")}'
            f' to {self.sim_set.SIM_END_DATE.strftime("%Y-%m-%d")} with allocation '
            f'based on: \n  {self.sim_set["calc_score"]}'
        )

        # Start with an empty match list file.
        if self.sim_set.SAVE_MATCH_LISTS:
            if os.path.exists(self.match_list_file):
                os.remove(
                    self.match_list_file
                )
            if os.path.exists(self.match_list_file + '.gzip'):
                os.remove(
                    self.match_list_file + '.gzip'
                )

        # Simulate until simulation is finished.
        while (
            not self.event_queue.is_empty() and
            (self.sim_time < self.max_sim_time)
        ):
            event = self.event_queue.next()

            # Print simulation progress
            if self.sim_time / print_progress_every_k_days >= next_k_days:
                print(
                    'Simulated up to {0}'.format(
                        (
                            self.sim_start_date +
                            timedelta(days=self.sim_time)
                        ).strftime('%Y-%m-%d')
                    )
                )
                next_k_days += 1

            # Resort list every k days
            if (
                self.sim_time -
                self._time_active_list_resorted
            ) > self._resort_every_k_days:
                self.active_lists.resort_lists()
                self._time_active_list_resorted = self.sim_time

            self.sim_time = event.event_time
            if verbose:
                print(f'{self.sim_time:.2f}: {event}')

            if event.type_event == cn.PAT:
                # Update patient information, and schedule future event
                # if patient remains active.
                self.patients[event.identifier].do_patient_update(
                    sim_results=self.sim_results
                )
                # If patient does not exit, schedule a new future event
                # time at his next update.
                if self.patients[event.identifier].exit_status is None:
                    event.update_event_time(
                        new_time=self.patients[
                            event.identifier
                            ].get_next_update_time()
                    )
                    self.event_queue.add(
                        event
                    )

                # If patient becomes active, add to active list,
                if (
                    not self.active_lists.is_active(event.identifier) and
                    self.patients[event.identifier].active
                ):
                    self.active_lists.add_to_active_list(
                        event.identifier,
                        self.patients[event.identifier]
                        )
                if not self.patients[event.identifier].active:
                    self.active_lists.pop_from_active_lists(
                        event.identifier
                    )


            elif event.type_event == cn.DON:
                donor = self.donors[event.identifier]
                donor_dict = donor.__dict__

                # Construct an MatchList with all patients
                # that are BG-compatible, have an active status,
                # and a first MELD score
                match_list = MatchListCurrentETKAS(
                    patients=(
                        p for p in self.get_active_list(donor_dict[cn.D_BLOODGROUP]) if (
                            (
                                donor_dict[cn.D_DCD] == 0 or p.dcd_country
                            )
                        )
                    ),
                    donor=donor,
                    match_date=(
                        self.sim_start_date +
                        timedelta(days=self.sim_time)
                        ),
                    sim_start_date=self.sim_start_date,
                    hla_system=self.hla_system,
                    bal_system=self.bal_system,
                    calc_points=self.sim_set.calc_score,
                    store_score_components=False,
                    initialize_unacceptable_mrs=self.init_unacc_mrs
                )

                # Now determine which patient accepts the organ
                accepted_mr = (
                    self.acceptance_module.simulate_liver_allocation(
                        match_list
                    )
                )

                # Save match list.
                if self.sim_set.SAVE_MATCH_LISTS:
                    self.sim_results.save_match_list(
                        match_list
                    )

                if accepted_mr:

                    # Set patient who accepted the organ to transplanted.
                    accepted_mr.patient.set_transplanted(
                        tx_date=(
                            self.sim_start_date +
                            timedelta(days=self.sim_time)
                        ),
                        donor=donor,
                        match_record=accepted_mr,
                        sim_results=self.sim_results
                    )
                    self.active_lists.pop_from_active_lists(
                        accepted_mr.patient.id_registration
                    )

                    # Simulate post-transplant survival, i.e. simulate
                    # patient / graft failure, and add a synthetic
                    # reregistration if graft fails before end
                    # of simulation period.
                    if self.sim_set.SIM_RETX:
                        self.simulate_posttxp(
                            txp=accepted_mr
                        )

                else:
                    self.sim_results.save_discard(
                        matchl_=match_list
                        )
            else:
                raise ValueError(
                    f"{event.type_event} is not a valid event type."
                    )
        print(
            "--- Finished simulation in {0} seconds ---" .
            format(round_to_decimals(time.time() - start_time, 1))
        )


    def simulate_posttxp(self, txp: 'AllocationSystem.MatchRecord'):
        """Code to simulate post-transplant outcomes for transplanted
            patients. This simulates failure and relisting dates, but
            also adds a synthetic re-registration to the simulation
            in case the re-listing occurs during the sim period.
        """

        # Simulate post-transplant survival
        date_fail, date_relist, cause_fail = self.ptp.simulate_failure_date(
            offer=txp,
            current_date=(
                self.sim_start_date +
                timedelta(days=self.sim_time)
            )
        )

        self.sim_results.save_posttransplant(
            date_failure=date_fail,
            date_relist=date_relist,
            cens_date=self.sim_set.SIM_END_DATE,
            matchr_=txp,
            rereg_id=(
                (
                    self.ptp.offset_ids_transplants +
                    self.ptp.synth_regs
                )
                if date_relist and date_relist < self.sim_set.SIM_END_DATE
                else None
            )
        )

        if (
            cause_fail == cn.PATIENT_RELISTING and
            date_relist and
            date_relist < self.sim_set.SIM_END_DATE
        ):
            # If time-to-event is longer in matched patient than non-matched
            # patient. But then only match to patients with
            # longer time-to-events.
            synth_reg = self.ptp.generate_synthetic_reregistration(
                offer=txp,
                relist_date=date_relist,
                fail_date=date_fail,
                curr_date=(
                    self.sim_start_date +
                    timedelta(days=self.sim_time)
                ),
                verbosity=0
            )

            self.patients[synth_reg.__dict__[cn.ID_REGISTRATION]] = synth_reg
            next_update_time = synth_reg.get_next_update_time()

            if next_update_time:
                self.event_queue.add(
                    Event(
                        type_event=cn.PAT,
                        event_time=next_update_time,
                        identifier=synth_reg.__dict__[cn.ID_REGISTRATION]
                    )
                )
