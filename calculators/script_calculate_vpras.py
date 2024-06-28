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
    df_stat.loc[df_stat.TYPE_UPDATE == 'UNACC', 'VARIABLE_VALUE'] = (
        rdr.fix_hla_string(df_stat.loc[df_stat.TYPE_UPDATE == 'UNACC', 'VARIABLE_VALUE'])
    )
    vpra_dicts = df_stat.loc[df_stat.TYPE_UPDATE == 'UNACC', ['VARIABLE_VALUE', 'VARIABLE_DETAIL']].drop_duplicates().to_dict('records')
    for d in vpra_dicts:
        d.update({'calc_vpra': hla_system.calculate_vpra_from_string(d['VARIABLE_VALUE'])})

    d_vpra = pd.DataFrame.from_records(vpra_dicts)
    d_vpra.to_csv('raw_data/calculated_vpras.csv', index=False)

