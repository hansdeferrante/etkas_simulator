#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:33:44 2022

@author: H.C. de Ferrante
"""

import sys
import os
import pandas as pd

from simulator.code.ETKASS import ETKASS
from simulator.code.read_input_files import read_sim_settings
import simulator.magic_values.etkass_settings as es
import simulator.code.read_input_files as rdr
import simulator.magic_values.column_names as cn
from datetime import datetime

if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)

    sim_set = read_sim_settings(
                os.path.join(
                    es.DIR_SIM_SETTINGS,
                    '2024-06-27',
                    'CurrentETKAS_vPRA_sliding_scale_b2_150p_2_2.yml'
                )
            )
    #sim_set.SIM_END_DATE = datetime(year=2016, month=3, day=31)
    sim_set.SAVE_MATCH_LISTS=True

    # Read in simulation settings
    simulator = ETKASS(
        sim_set=sim_set,
        verbose=0
    )

    ml = simulator.simulate_allocation(
        verbose=False
    )

    simulator.sim_results.save_outcomes_to_file(
        patients=simulator.patients,
        cens_date=sim_set.SIM_END_DATE
    )
