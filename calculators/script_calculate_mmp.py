#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri 11-02-2022


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
    import simulator.code.load_entities as le
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
            'sim_settings.yaml'
        )
    )

    hla_system = HLASystem(ss)
    DUMMY_DATE = pd.Timestamp('2000-01-01')

    df_stat = pd.read_csv(
            'raw_data/status_updates.csv'
        )
    df_stat.loc[df_stat.TYPE_UPDATE == 'HLA', 'VARIABLE_VALUE'] = (
        rdr.fix_hla_string(df_stat.loc[df_stat.TYPE_UPDATE == 'HLA', 'VARIABLE_VALUE'])
    )

    # Read patients
    df_pats = le._read_patients_rich(ss)
    df_pats = df_pats.dropna(subset=[cn.PATIENT_HLA])
    df_pats = df_pats.dropna(subset = cn.PATIENT_HLA)
    df_pats = df_pats.loc[df_pats.loc[:, cn.PATIENT_HLA] != '', :]
    df_pats = df_pats.loc[df_pats.recipient_country.isin(es.ET_COUNTRIES), :]

    all_hlas = (
        df_pats.loc[:, cn.PATIENT_HLA].drop_duplicates().to_list() +
        df_stat.loc[df_stat.TYPE_UPDATE == 'HLA', ['VARIABLE_VALUE']].drop_duplicates().VARIABLE_VALUE.to_list()
    )

    patients = [
        Patient(
            id_recipient=1,
            r_dob=(
                DUMMY_DATE -
                timedelta(days=365.25 * 50)
            ),
            recipient_country='Germany',
            recipient_region='GND',
            recipient_center='GHMTP',
            bloodgroup='O',
            hla=hla,
            listing_date=DUMMY_DATE,
            urgency_code=mr.NT,
            sim_set = ss,
            hla_system=hla_system
        )
        for hla in all_hlas
    ]

    results_hla_mp = (
        {
            **{'r_hla': pat.hla},
            **hla_system.calculate_hla_match_potential(
                pat,
                hla_match_pot_definitions=es.HLA_MATCH_POTENTIALS
            )
        } for pat in patients
    )

    # Construct measures for MMP.
    d_mmp = pd.DataFrame.from_records(results_hla_mp)
    d_mmp = d_mmp.drop_duplicates()
    for col in d_mmp.columns:
        if 'hmp_' in col:
            newcolname = col.replace('hmp_', 'hlamismatchfreq_')
            d_mmp.loc[:, newcolname] = d_mmp.loc[:, col].transform(
                lambda x: round(100*(1-x)**1000, 5)
            )


    d_mmp.columns = d_mmp.columns.str.lower()

    d_mmp.to_csv('data/favorably_matched_haplotype_frequencies2016.csv', index=False)


