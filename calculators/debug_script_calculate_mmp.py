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

    print('Loading data')
    df_don = rdr.read_donor_pool('data/donor_pool.csv')

    df_don_hlas = df_don.loc[:, [cn.ID_DONOR, cn.DONOR_HLA]].drop_duplicates()

    antigen_sets = [set(s.split(' ')) for s in df_don_hlas.donor_hla]

    donors = [
        Donor(
            id_donor=1255,
            donor_country='Belgium',
            donor_region='BLGTP',
            donor_center='BLGTP',
            bloodgroup='O',
            reporting_date=DUMMY_DATE,
            weight=50,
            donor_dcd=False,
            hla=s,
            hla_system=hla_system,
            hypertension=False,
            diabetes=False,
            cardiac_arrest=False,
            last_creat=1.5,
            smoker=False,
            age=45,
            hbsag=False,
            hcvab=False,
            hbcab=False,
            sepsis=False,
            meningitis=False,
            malignancy=False,
            drug_abuse=False,
            euthanasia=False,
            rescue=False,
            death_cause_group='Anoxia',
            n_kidneys_available=2
        )
        for s in df_don_hlas.donor_hla
    ]

    df_stat = pd.read_csv(
            'raw_data/status_updates.csv'
        )
    df_stat = df_stat.loc[
        df_stat.VARIABLE_VALUE.str.contains('A2 A*02:01 A10 A25 A*25:01 B15 B62 B*15:01 B22 B55 B*55:01', regex=False).fillna(False), :
    ]
    df_stat.loc[df_stat.TYPE_UPDATE == 'HLA', 'VARIABLE_VALUE'] = (
        rdr.fix_hla_string(df_stat.loc[df_stat.TYPE_UPDATE == 'HLA', 'VARIABLE_VALUE'])
    )

    all_hlas = (
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
            recipient_region='Germany',
            recipient_center='Germany',
            bloodgroup='O',
            hla=hla,
            listing_date=DUMMY_DATE,
            urgency_code=mr.NT,
            sim_set = ss,
            hla_system=hla_system
        )
        for hla in all_hlas
    ]

    print('Determining mismatches')
    hla_mismatch_probabilities = list()
    for i, pat in enumerate(patients):
        if i>0 and i % 1000 == 0:
            print(f'Processed {round(i/len(patients)*100)}% of all HLAs')
        mm_dicts = (hla_system.determine_mismatches(d=d, p=pat) for d in donors)
        prob_atmost1mm = sum(
            (mm_dict['mmb_hla_a'] + mm_dict['mmb_hla_b'] + mm_dict['mms_hla_dr']) <= 1 for mm_dict in mm_dicts if mm_dict
            ) / len(donors)
        hla_mismatchfreq = (100*((1-prob_atmost1mm)**1000))
        hla_mismatch_probabilities.append(
            {'r_hla': pat.hla, 'prob_atmost1mm': prob_atmost1mm, 'hla_mismatchfreq': round(hla_mismatchfreq, 5)}
        )

    d_mmp = pd.DataFrame.from_records(hla_mismatch_probabilities)
    d_mmp = d_mmp.drop_duplicates()

    # Write to file
    d_mmp.to_csv('raw_data/calculated_mismatchfreqs.csv', index=False)

