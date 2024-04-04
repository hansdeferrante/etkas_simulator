#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri 11-02-2022

Script to develop the standard exception system.

@author: H.C. de Ferrante
"""

import sys
sys.path.append('./')
from operator import attrgetter
if True:  # noqa E402
    from datetime import timedelta
    from simulator.code.entities import Patient, Donor
    import simulator.magic_values.column_names as cn
    import simulator.code.read_input_files as rdr
    import simulator.magic_values.etkass_settings as es
    from simulator.code.current_etkas.CurrentETKAS import (
        MatchListCurrentETKAS
    )
    from simulator.code.entities import HLASystem
    import pandas as pd
    import os
    from typing import Dict
    from simulator.magic_values.inputfile_settings import DTYPE_DONORLIST
    import simulator.magic_values.magic_values_rules as mr


if __name__ == '__main__':

    pd.set_option('display.max_rows', 50)
    ss = rdr.read_sim_settings(
        os.path.join(
                es.DIR_SIM_SETTINGS,
            'sim_settings_sel_hlas.yaml'
        )
    )
    all_loci = ('hla_a', 'hla_b', 'hla_c', 'hla_dpa', 'hla_dpb', 'hla_dqa', 'hla_dqb', 'hla_dr', 'hla_drb345')

    ss.needed_broad_mismatches = all_loci
    ss.needed_split_mismatches = all_loci
    ss.HLAS_TO_LOAD = all_loci

    hla_system = HLASystem(ss)
    DUMMY_DATE = pd.Timestamp('2000-01-01')
    df_karma_cohort = pd.read_csv(
        'data/KARMA/karma_final_cohort/karma_study_lean_hla_merge_lbl.csv',
        delimiter=';'
    )
    hla_cols = (
        'rec_hla_full', 'rec_hla_full_curr', 'first_don_hla_full', 'first_don_hla_full_curr', 'second_don_hla_full'
    )

    pd_dfs = []
    for hla_string_col in hla_cols:
        df_karma_cohort.loc[:, 'fixed_hla_string'] = (
            rdr.fix_hla_string(df_karma_cohort.loc[:, hla_string_col])
        )

        patients = (
            Patient(
                id_recipient=1,
                r_dob=(
                    DUMMY_DATE -
                    timedelta(days=365.25 * 45)
                ),
                recipient_country='Netherlands',
                recipient_region='',
                recipient_center='NRDTP',
                bloodgroup='A',
                hla=rcrd['fixed_hla_string'],
                listing_date=DUMMY_DATE,
                urgency_code=mr.NT,
                sim_set = ss,
                hla_system=hla_system
            )
            for rcrd in df_karma_cohort.to_dict(orient='records')
        )

        code_order = list(
            f'{locus}_{num+1}_{level}' for locus in ss.HLAS_TO_LOAD
            for num in range(2)
            for level in ('broad', 'split', 'allele')
        )

        df_structured_hlas = pd.DataFrame.from_records(
            (
                hla_system.order_structured_hla(
                    patient.hla
                )
                for patient in patients
            ),
            columns=code_order
        )
        allele_cols = list(
            code for code in code_order if 'allele' in code
        )

        df_structured_hlas.fillna('', inplace=True)
        df_structured_hlas = pd.concat(
            [
                df_karma_cohort.loc[:, ['record_id', hla_string_col]],
                df_structured_hlas
            ],
            axis=1
        )
        df_structured_hlas = df_structured_hlas.rename(
            columns={hla_string_col: 'hla_string'}
        )
        df_structured_hlas.loc[:, 'type_hla'] = hla_string_col
        pd_dfs.append(df_structured_hlas)

    df_formatted_hlas = pd.concat(
        pd_dfs,
        axis=0
    )

    df_formatted_hlas.to_csv(
        'data/KARMA/karma_final_cohort/all_hlas.csv',
        index=False,
        sep=','
    )
