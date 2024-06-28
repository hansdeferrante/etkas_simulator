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
    import simulator.magic_values.etkass_settings as es
    from simulator.code.current_etkas.CurrentETKAS import (
        MatchListCurrentETKAS
    )
    from simulator.code.entities import HLASystem
    import pandas as pd
    from numpy import isnan
    import os
    from copy import deepcopy
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

    # Load HLA system and match lists
    ss.needed_broad_mismatches = ('hla_b', 'hla_a', 'hla_dr')
    ss.needed_split_mismatches = ('hla_b', 'hla_a', 'hla_dr')
    hla_system = HLASystem(ss)

    DUMMY_DATE = pd.Timestamp('2000-01-01')
    df_txp = pd.read_csv(
        'raw_data/all_accepted_match_lists.csv',
        encoding='utf-8'
    )
    df_txp.loc[:, 'patient_hla_orig'] = deepcopy(df_txp.loc[:, 'PATIENT_HLA'])


    df_pats = pd.read_csv(
        'raw_data/patients.csv'
    ).groupby('ID_RECIPIENT').tail(1).loc[:, ['ID_RECIPIENT', 'R_HLA']].rename(columns={'ID_RECIPIENT': 'PATIENT_OFFERED', 'R_HLA': 'PATIENT_HLA'})
    hla_dict = {
        rcrd['PATIENT_OFFERED']: rcrd['PATIENT_HLA']
        for rcrd in df_pats.to_dict(orient='records')}

    df_txp['PATIENT_HLA'] = df_txp['PATIENT_HLA'].fillna(df_txp['PATIENT_OFFERED'].map(hla_dict))


    df_txp.loc[:, 'PATIENT_HLA'] = (
        rdr.fix_hla_string(df_txp.loc[:, 'PATIENT_HLA'])
    )
    df_txp.loc[:, 'D_HLA_FULL'] = (
        rdr.fix_hla_string(df_txp.loc[:, 'D_HLA_FULL'])
    )
    df_txp.loc[:, 'rn'] = range(df_txp.shape[0])

    df_txp_orig = deepcopy(df_txp)

    # Drop records with missing candidate HLAs
    nrow_before = df_txp.shape[0]
    df_txp = df_txp[(df_txp['PATIENT_HLA'] != '') & ~df_txp['PATIENT_HLA'].isna()]
    for hla_str in ('A', 'B', 'DR'):
        df_txp = df_txp[df_txp.PATIENT_HLA.str.contains(hla_str, regex=False)]
    print(f'Discarded {nrow_before - df_txp.shape[0]} rows with missing/incomplete HLAs for patient')

    df_complete_patient_hla = deepcopy(df_txp)

    # Drop records with missing donor HLAs
    nrow_before = df_txp.shape[0]
    df_txp = df_txp[(df_txp['D_HLA_FULL'] != '') & ~df_txp['D_HLA_FULL'].isna()]
    for hla_str in ('A', 'B', 'DR'):
        df_txp = df_txp[df_txp.D_HLA_FULL.str.contains(hla_str, regex=False)]
    print(f'Discarded {nrow_before - df_txp.shape[0]} rows with missing/incomplete HLAs for donor')


    # Construct vPRA data frame for all candidates with known patient hla.
    d_vpras = pd.DataFrame.from_records(
        {
            'rn': rn,
            'vpra_man': round(hla_system.calculate_vpra_from_string(unaccs), ndigits=4)
        } for rn, unaccs in zip(
            df_complete_patient_hla.loc[:, 'rn'],
            df_complete_patient_hla.loc[:, 'UNACCEPTABLE_ANTIGENS']
        )
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
            cmv=False,
            urine_protein=0
        )
        for s in df_txp.D_HLA_FULL
    )
    patients = (
        Patient(
            id_recipient=rcrd['rn'],
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
        for rcrd in df_txp.to_dict(orient='records')
    )

    # Construct a match quality data.frame from all donors and patients
    df_mq = pd.DataFrame.from_records(
        hla_system.determine_mismatches(d=d, p=p) | {'rn': p.id_recipient}
        for i, (p, d) in enumerate(zip(patients, donors))
    )
    df_mq.loc[:, cn.MM_TOTAL] = df_mq.mmb_hla_a + df_mq.mmb_hla_b + df_mq.mms_hla_dr

    # Re-construct the patient generator, so that we can also determine homozygosity of
    # waitlisted candidates.
    patients = (
        Patient(
            id_recipient=rcrd['rn'],
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
        for rcrd in df_complete_patient_hla.to_dict(orient='records')
    )
    df_homozyg = pd.DataFrame.from_records(
        p.get_homozygosity_per_locus() | {'rn': p.id_recipient} for p in patients
    ).add_prefix('hz_').rename(columns={'hz_rn': 'rn'})
    df_complete_patient_hla = pd.merge(
        df_complete_patient_hla,
        df_homozyg,
        on=['rn']
    )
    df_complete_patient_hla = pd.merge(
        df_complete_patient_hla,
        d_vpras,
        on=['rn']
    )
    new_cols_mq = df_mq.columns.difference(df_txp_orig.columns).difference(['index']).to_list()
    new_cols_patient = df_complete_patient_hla.columns.difference(df_txp_orig.columns).difference(['index']).to_list()
    df_txp_final = pd.merge(
        left = df_txp_orig,
        right = df_complete_patient_hla.loc[:, ['rn'] + new_cols_patient],
        on = 'rn',
        how='left'
    )
    df_txp_final = pd.merge(
        left = df_txp_final,
        right = df_mq.loc[:, ['rn'] + new_cols_mq],
        on = 'rn',
        how='left'
    )

    df_txp_final.drop(columns=['rn'], inplace=True)

    df_txp_final.to_csv('raw_data/all_accepted_match_lists_with_mq_and_vpra.csv', index=False)
