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

    ss.needed_broad_mismatches = ('hla_b', 'hla_a', 'hla_dr', 'hla_dqa', 'hla_dqb', 'hla_c')
    ss.needed_split_mismatches = ('hla_b', 'hla_a', 'hla_dr', 'hla_dqa', 'hla_dqb', 'hla_c')

    hla_system = HLASystem(ss)
    # print(hla_system.splits_to_alleles['hla_dr']['DR3'])

    hla_system.order_structured_hla(
        'A3 A03XX A19 A30 A30XX B18 B18XX B35 B35XX CW3 C03XX CW5 C05XX DR3 DR17 DRB10301 DR8 DRB10801 DQ2 DQB10201 DQ4 DQB10402'
    )

    DUMMY_DATE = pd.Timestamp('2000-01-01')
    df_txp = pd.read_csv(
        'raw_data/transplantations_klundert.csv',
        dtype={
            cn.ID_DONOR: 'str', cn.ID_MTR: 'str', cn.ID_TXP: 'str',
            cn.ID_RECIPIENT: 'str', cn.ID_REGISTRATION: 'str'
        }
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

    # This is an invalid HLA-string in the data ware house.
    df_txp = df_txp[~ df_txp.CURRENT_PATIENT_HLA.str.contains("B5 B51 B52 B35", regex=False)]

    patients = (
        Patient(
            id_recipient=rcrd['ID_TXP'],
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

    code_order = list(
        f'{locus}_{num+1}_{level}' for locus in ss.HLAS_TO_LOAD
        for num in range(2)
        for level in ('broad', 'split', 'allele')
    )
    code_order = [cn.ID_TXP] + code_order

    allele_cols = list(
        code for code in code_order if 'allele' in code
    )
    df_structured_patient_hlas = pd.DataFrame.from_records(
        (
            hla_system.order_structured_hla(
                patient.hla
            ) | {cn.ID_TXP: patient.id_recipient}
            for patient in patients
        ),
        columns=code_order
    )

    # Make generators for donors and patients to determine match qualities.
    donors = (
        Donor(
            id_donor=rcrd['ID_TXP'],
            donor_country='Belgium',
            donor_region='BLGTP',
            donor_center='BLGTP',
            bloodgroup='O',
            reporting_date=DUMMY_DATE,
            donor_dcd=False,
            hla=rcrd['DONOR_HLA'],
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
        for rcrd in df_txp.to_dict(orient='records')
    )
    df_structured_donor_hlas = pd.DataFrame.from_records(
        (
            hla_system.order_structured_hla(
                donor.hla
            ) | {cn.ID_TXP: donor.id_donor}
            for donor in donors
        ),
        columns=code_order
    ).fillna('')

    df_structured_patient_hlas = pd.merge(
        df_txp.loc[:, ['ID_TXP', 'YEAR_TXP', 'CURRENT_PATIENT_HLA']].rename(columns=str.lower),
        df_structured_patient_hlas,
        on = cn.ID_TXP,
        how='left'
    )
    df_structured_donor_hlas = pd.merge(
        df_txp.loc[:, ['ID_TXP', 'YEAR_TXP', 'DONOR_HLA']].rename(columns=str.lower),
        df_structured_donor_hlas,
        on = cn.ID_TXP,
        how='left'
    )

    df_structured_patient_hlas.to_csv(
        'X:/PRJ/Machine learning/afd/Hans_de_Ferrante/ETKAS/Klundert_ML_HLA/raw_data/patient_hlas.csv',
        index=False,
        sep=','
    )

    df_structured_donor_hlas.to_csv(
        'X:/PRJ/Machine learning/afd/Hans_de_Ferrante/ETKAS/Klundert_ML_HLA/raw_data/donor_hlas.csv',
        index=False,
        sep=','
    )
