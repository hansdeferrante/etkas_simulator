#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri 11-02-2022


@author: H.C. de Ferrante
"""

import sys
sys.path.append('./')
if True:  # noqa E402
    from simulator.code.entities import Donor
    import simulator.code.read_input_files as rdr
    import simulator.magic_values.etkass_settings as es
    from simulator.code.entities import HLASystem
    import pandas as pd
    import os
    import simulator.magic_values.magic_values_rules as mr

    from collections import defaultdict
    import yaml


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
    donors = [
        Donor(
            id_donor=1255,
            donor_country='Belgium',
            donor_region='BLGTP',
            donor_center='BLGTP',
            bloodgroup=bg,
            reporting_date=DUMMY_DATE,
            weight=50,
            donor_dcd=False,
            hla=hla,
            hla_system=hla_system
        )
        for hla, bg in zip(df_don.donor_hla, df_don.d_bloodgroup)
    ]

    broad_counts = defaultdict(lambda: defaultdict(float))
    split_counts = defaultdict(lambda: defaultdict(float))
    bg_counts = defaultdict(int)

    for don in donors:
        bg_counts[don.d_bloodgroup] += 1
        for locus, hlas in don.hla_broads.items():
            for hla in hlas:
                if '?' in hla:
                    continue
                if len(hlas) == 1:
                    broad_counts[locus][hla] += 2
                else:
                    broad_counts[locus][hla] += 1
        for locus, hlas in don.hla_splits.items():
            for hla in hlas:
                if '?' in hla:
                    continue
                if len(hlas) == 1:
                    split_counts[locus][hla] += 2
                else:
                    split_counts[locus][hla] += 1


    allele_frequencies_splits = allele_frequencies = {
        **{k: {ik: iv / sum(v.values()) for ik, iv in v.items()}  for k, v in broad_counts.items() if k in hla_system.needed_broad_mismatches},
        **{k: {ik: iv / sum(v.values()) for ik, iv in v.items()}  for k, v in split_counts.items() if k in hla_system.needed_split_mismatches}
    }
    allele_frequencies_broads = {
        k: {ik: iv / sum(v.values()) for ik, iv in v.items()} for k, v in broad_counts.items()
    }
    bg_frequencies = {ik: iv / len(donors) for ik, iv in bg_counts.items()}

    print({k: sum(v.values()) for k, v in broad_counts.items()})
    print({k: sum(v.values()) for k, v in split_counts.items()})

    with open('simulator/magic_values/allele_frequencies_split.yaml', 'w') as outfile:
        yaml.dump(allele_frequencies_splits, outfile, default_flow_style=False)
    with open('simulator/magic_values/allele_frequencies_broad.yaml', 'w') as outfile:
        yaml.dump(allele_frequencies_broads, outfile, default_flow_style=False)
    with open('simulator/magic_values/bg_frequencies.yaml', 'w') as outfile:
        yaml.dump(bg_frequencies, outfile, default_flow_style=False)

