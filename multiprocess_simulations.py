#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu 20 10 2022

@author: H.C. de Ferrante
"""

from __future__ import division
from multiprocessing import Pool, cpu_count, Lock
from time import time
from typing import Tuple, List
from datetime import datetime
import os
import simulator.magic_values.etkass_settings as es
from pathlib import Path
import re
from random import shuffle


def list_full_paths(directories: Tuple[str, List[str]]):
    if isinstance(directories, str):
        return [os.path.join(directories, file) for file in os.listdir(directories)]
    elif isinstance(directories, list):
        return [
                os.path.join(directory, file)
                for directory in directories
                for file in os.listdir(directory)
        ]


def simulate_allocation(sim_path, skip_if_exists: bool = True):

    from simulator.code.ETKASS import ETKASS
    from simulator.code.read_input_files import read_sim_settings

    sim_set = read_sim_settings(
                sim_path
            )

    patient_file = Path(
        f'{sim_set.RESULTS_FOLDER}/{sim_set.PATH_FINAL_PATIENT_STATUS}'
    )
    if patient_file.is_file() and skip_if_exists:
        print(f'{sim_path} already processed... Continuing with next one')
        return 0
    elif not Path(sim_set.PATH_STATUS_UPDATES).is_file():
        print(f"Status update file '{sim_set.PATH_STATUS_UPDATES} does not exist. Skipping...")
        return 0
    else:
        print(f'Working on {sim_path}')

    # Read in simulation settings
    with lock_init:
        simulator = ETKASS(
            sim_set=sim_set,
            verbose=False
        )

    try:
        simulator.simulate_allocation(
            verbose=False,
            print_progress_every_k_days=365
            )
    except Exception as e:
        print('\n\n***********')
        msg = f'An error occurred for {sim_path}'
        print(msg)
        print(e)
        print('\n\n***********')
        return 0

    with lock_gzip:
        simulator.sim_results.save_outcomes_to_file(
            patients=simulator.patients,
            cens_date=sim_set.SIM_END_DATE
        )

    return 1


def init(il, gl):
    global lock_gzip
    lock_gzip = gl
    global lock_init
    lock_init = il


if __name__ == '__main__':

    n_workers = cpu_count()
    n_workers = 6

    paths = list_full_paths(
        [
            os.path.join(es.DIR_SIM_SETTINGS, path)
            for path in ['2024-06-27']
            ]
    )
    shuffle(paths)


    pattern = re.compile('.*\\.yml')
    paths = [path for path in paths if pattern.match(path)]

    init_lock = Lock()
    gzip_lock = Lock()

    start = time()
    print(f'Processing {len(paths)} files')

    with Pool(
        processes=n_workers,
        initializer=init,
        initargs=(init_lock, gzip_lock)
    ) as p:

        for i, _ in enumerate(p.imap(simulate_allocation, paths)):
            print('\rdone {0:%}'.format(i/len(paths)))

    end = time()
    print(f'{len(paths)} jobs executed in {(end - start)/60:.1f} minutes')
