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

    # Load HLA system and match lists
    ss.needed_broad_mismatches = ('hla_b', 'hla_a', 'hla_dr')
    ss.needed_split_mismatches = ('hla_b', 'hla_a', 'hla_dr')
    hla_system = HLASystem(ss)

    DUMMY_DATE = pd.Timestamp('2000-01-01')
    df_txp = pd.read_csv(
        'raw_data/transplantations.csv',
        encoding='utf-8'
    )
    df_txp.loc[:, 'CURRENT_PATIENT_HLA'] = (
        rdr.fix_hla_string(df_txp.loc[:, 'CURRENT_PATIENT_HLA'])
    )
    df_txp.loc[:, 'DONOR_HLA'] = (
        rdr.fix_hla_string(df_txp.loc[:, 'DONOR_HLA'])
    )

    # Drop records with missing HLAs
    nrow_before = df_txp.shape[0]
    df_txp = df_txp[(df_txp['DONOR_HLA'] != '') & ~df_txp['DONOR_HLA'].isna()]
    df_txp = df_txp[(df_txp['CURRENT_PATIENT_HLA'] != '') & ~df_txp['CURRENT_PATIENT_HLA'].isna()]
    for hla_str in ('A', 'B', 'DR'):
        df_txp = df_txp[df_txp.DONOR_HLA.str.contains(hla_str, regex=False)]
        df_txp = df_txp[df_txp.CURRENT_PATIENT_HLA.str.contains(hla_str, regex=False)]
    print(f'Discarded {nrow_before - df_txp.shape[0]} rows with missing/incomplete HLAs')

    # Construct vPRAs
    d_vpras = pd.DataFrame.from_records(
        {
            'vpra_man': round(hla_system.calculate_vpra_from_string(unaccs), ndigits=4)
        } for unaccs in df_txp.loc[:, 'UNACCEPTABLE_ANTIGENS']
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
            cmv=0
        )
        for s in df_txp.DONOR_HLA
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
            hla=rcrd['CURRENT_PATIENT_HLA'],
            listing_date=DUMMY_DATE,
            urgency_code=mr.NT,
            sim_set = ss,
            hla_system=hla_system
        )
        for rcrd in df_txp.to_dict(orient='records')
    )

    # Construct a match quality data.frame from all donors and patients
    df_mq = pd.DataFrame.from_records(

        hla_system.determine_mismatches(d=d, p=p)
        for i, (p, d) in enumerate(zip(patients, donors))
    )

    df_mq[cn.MM_TOTAL] = -1 * df_mq[cn.MM_TOTAL]

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
            hla=rcrd['CURRENT_PATIENT_HLA'],
            listing_date=DUMMY_DATE,
            urgency_code=mr.NT,
            sim_set = ss,
            hla_system=hla_system
        )
        for rcrd in df_txp.to_dict(orient='records')
    )
    df_homozyg = pd.DataFrame.from_records(
        p.get_homozygosity_per_locus() for p in patients
    ).add_prefix('hz_')
    df_with_mq = pd.concat([df_txp.reset_index(), df_mq.reset_index(), df_homozyg.reset_index(), d_vpras.reset_index()], axis=1)
    df_with_mq = df_with_mq.drop(columns='index')

    df_with_mq.to_csv('raw_data/transplantations_with_mq_vpra.csv', index=False)
