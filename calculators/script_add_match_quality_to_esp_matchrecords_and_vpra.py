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
    from numpy import isnan
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
    df_pats = pd.read_csv(
        'raw_data/patients.csv'
    ).groupby('ID_RECIPIENT').tail(1).loc[:, ['ID_RECIPIENT', 'R_HLA']].rename(columns={'ID_RECIPIENT': 'PATIENT_OFFERED', 'R_HLA': 'PATIENT_HLA'})

    # Load HLA system and match lists
    hla_system = HLASystem(ss)
    DUMMY_DATE = pd.Timestamp('2000-01-01')
    df_ml = pd.read_csv(
        'raw_data/all_esp_match_lists.csv',
        encoding='utf-8'
    )
    # df_ml = df_ml.loc[df_ml.PATIENT_ACTION_CODE.notna(), :]
    df_ml = df_ml.loc[df_ml.PATIENT_ACTION_CODE != 'FP', :]
    df_ml = df_ml.loc[~ df_ml.PATIENT_OFFERED.isna()]
    df_ml.PATIENT_OFFERED = df_ml.PATIENT_OFFERED.astype(int)
    hla_dict = {
        rcrd['PATIENT_OFFERED']: rcrd['PATIENT_HLA']
        for rcrd in df_pats.to_dict(orient='records')}

    df_ml['PATIENT_HLA'] = df_ml['PATIENT_HLA'].fillna(df_ml['PATIENT_OFFERED'].map(hla_dict))

    df_ml.loc[:, 'PATIENT_HLA'] = (
        rdr.fix_hla_string(df_ml.loc[:, 'PATIENT_HLA'])
    )
    df_ml.loc[:, 'D_HLA_FULL'] = (
        rdr.fix_hla_string(df_ml.loc[:, 'D_HLA_FULL'])
    )

    # Drop records with missing HLAs
    nrow_before = df_ml.shape[0]
    df_ml = df_ml[(df_ml['D_HLA_FULL'] != '') & ~df_ml['D_HLA_FULL'].isna()]
    df_ml = df_ml[(df_ml['PATIENT_HLA'] != '') & ~df_ml['PATIENT_HLA'].isna()]
    df_ml.loc[:, 'UNACCEPTABLE_ANTIGENS'] = rdr.fix_hla_string(
        df_ml.loc[:, 'UNACCEPTABLE_ANTIGENS']
    )
    for hla_str in ('A', 'B', 'DR'):
        df_ml = df_ml[df_ml.D_HLA_FULL.str.contains(hla_str, regex=False)]
        df_ml = df_ml[df_ml.PATIENT_HLA.str.contains(hla_str, regex=False)]
    print(f'Discarded {nrow_before - df_ml.shape[0]} rows with missing/incomplete HLAs')

    # Construct vPRAs
    d_vpras = pd.DataFrame.from_records(
        {'r_vpra': round(hla_system.calculate_vpra_from_string(unaccs), ndigits=4)}
        for unaccs in df_ml.loc[:, 'UNACCEPTABLE_ANTIGENS']
    )

    # Make generators for donors and patients to determine match qualities.
    donors = (
        Donor(
            id_donor=1255,
            donor_country='Belgium',
            donor_region='BLGTP',
            donor_center='BLGTP',
            bloodgroup='O',
            reporting_date=DUMMY_DATE,
            donor_dcd=False,
            hla=s,
            hla_system=hla_system,
            n_kidneys_available=2,
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
            urine_protein=0,
            cmv=False
        )
        for s in df_ml.D_HLA_FULL
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
            hla=rcrd['PATIENT_HLA'],
            listing_date=DUMMY_DATE,
            urgency_code=mr.NT,
            sim_set = ss,
            hla_system=hla_system
        )
        for rcrd in df_ml.to_dict(orient='records')
    )

    # Construct a match quality data.frame from all donors and patients
    df_mq = pd.DataFrame.from_records(
        hla_system.determine_mismatches(d=d, p=p)
        for i, (p, d) in enumerate(zip(patients, donors))
    )

    # Re-construct the patient generator, so that we can also determine homozygosity of
    # waitlisted candidates.
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
            hla=rcrd['PATIENT_HLA'],
            listing_date=DUMMY_DATE,
            urgency_code=mr.NT,
            sim_set = ss,
            hla_system=hla_system
        )
        for rcrd in df_ml.to_dict(orient='records')
    )
    df_homozyg = pd.DataFrame.from_records(
        p.get_homozygosity_per_locus() for p in patients
    ).add_prefix('hz_')
    df_with_mq = pd.concat([df_ml.reset_index(), df_mq.reset_index(), df_homozyg.reset_index(), d_vpras.reset_index()], axis=1)
    df_with_mq = df_with_mq.drop(columns='index')

    df_with_mq.to_csv('raw_data/all_esp_match_lists_with_mq_vpra.csv', index=False)
