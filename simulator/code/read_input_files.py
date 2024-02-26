#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 17:33:44 2022

Scripts to read in input files.

@author: H.C. de Ferrante
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from functools import reduce
from numpy import arange, genfromtxt, ndarray, array
import pandas as pd
import warnings
import yaml

from simulator.code.utils import DotDict
import simulator.magic_values.inputfile_settings as dtypes
import simulator.magic_values.column_names as cn
import simulator.magic_values.etkass_settings as es
import simulator.magic_values.magic_values_rules as mgr
from simulator.magic_values.column_groups import MATCHLIST_COLS
from simulator.magic_values.etkass_settings import \
    ET_COUNTRIES
from simulator.code.ScoringFunction import MatchPointFunction

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def read_datetimes_dayfirst(x):
    return pd.to_datetime(x, dayfirst=True)


def _read_with_datetime_cols(
        input_path: str,
        dtps: Dict[str, Any],
        casecols: bool = False,
        usecols: Optional[List[str]] = None,
        datecols: Optional[List[str]] = None,
        **kwargs
) -> pd.DataFrame:
    """Read in pd.DataFrame with datecols as datetime"""

    if usecols is None:
        usecols = list(k.lower() for k in dtps.keys())
    if casecols:
        usecols = [c.upper() for c in usecols]
        dtps = {k.upper(): v for k, v in dtps.items()}
        if datecols:
            datecols = [c.upper() for c in datecols]
    else:
        usecols = [c.lower() for c in usecols]
        dtps = {k.lower(): v for k, v in dtps.items()}


    data_ = pd.read_csv(
        input_path,
        dtype=dtps,
        parse_dates=datecols if datecols else False,
        date_parser=read_datetimes_dayfirst,
        usecols=lambda x: x in usecols,
        **kwargs
    )
    assert isinstance(data_, pd.DataFrame), \
        f'Expected DataFrame, not {type(data_)}'

    if (missing_cols := set(data_.columns).difference(set(dtps.keys()))):
        print(
            f'Warning, following data types not specified for {input_path}:\n'
            f'{missing_cols}'
        )

    # Read in as standard datetime object, not pd.Timestamp
    if datecols:
        for date_col in datecols:
            data_[date_col] = pd.Series(
                data_[date_col].dt.to_pydatetime(),
                dtype='object'
            )

    if casecols:
        data_.columns = data_.columns.str.lower()

    return data_


def read_hla_match_table(input_path: str):
    d_hla_ = pd.read_csv(input_path)
    return d_hla_


def fix_hla_string(str_col: pd.Series) -> pd.Series:
    """Harmonize a HLA string column being read in"""

    for forbidden_character in es.HLA_FORBIDDEN_CHARACTERS:
        str_col = str_col.replace(forbidden_character, '', regex=True)
    str_col = str_col.str.upper()
    str_col = str_col.replace(
        to_replace = {
            '(?<=\\s)RB': 'DRB',
            '(?<=\\s)QB': 'DQB',
            'CW(?=([A-Z]|[0-9]){4})': 'C',
            '/[0-9]+': ''
        },
        regex=True
        )
    return str_col


def read_transplantations(input_path: str, usecols=None, **kwargs) -> pd.DataFrame:
    """Read in the MatchList."""

    if usecols is None:
        datecols = [x for x in (cn.DATE_TXP, cn.DATE_FIRST_DIAL)
        ]
        usecols = list(dtypes.DTYPE_TRANSPLANTLIST.keys())
    else:
        datecols = [
            x for x in
            set(usecols).intersection(
                set([x for x in (cn.DATE_TXP, cn.DATE_FIRST_DIAL)])
                )
            ]


    d_o = _read_with_datetime_cols(
        input_path=input_path,
        dtps=dtypes.DTYPE_TRANSPLANTLIST,
        casecols=False,
        usecols=usecols,
        datecols=datecols,
        **kwargs
    )

    d_o = d_o.drop_duplicates()
    d_o.columns = d_o.columns.str.lower()

    d_o[cn.D_DCD] = d_o[cn.D_DCD].fillna(0)

    for col in (cn.CURRENT_PATIENT_HLA, cn.DONOR_HLA):
        d_o[col] = fix_hla_string(d_o[col])

    # Assert column names are correct
    if usecols is None:
        for col in MATCHLIST_COLS:
            assert col in d_o.columns, \
                f'{col} should be column for {input_path}'

    return d_o


def read_offerlist(input_path: str, usecols=None, **kwargs) -> pd.DataFrame:
    """Read in the MatchList."""

    if usecols is None:
        datecols = [x.upper() for x in [cn.WFTS_MATCH_DATE, cn.PATIENT_DOB]]
        usecols = list(dtypes.DTYPE_OFFERLIST.keys())
    else:
        datecols = [
            x.upper() for x in
            set(usecols).intersection(
                set([x.upper() for x in [cn.WFTS_MATCH_DATE, cn.PATIENT_DOB]])
                )
            ]

    d_o = _read_with_datetime_cols(
        input_path=input_path,
        dtps=dtypes.DTYPE_OFFERLIST,
        casecols=True,
        usecols=usecols,
        datecols=datecols,
        **kwargs
    )

    d_o[cn.PATIENT_URGENCY] = d_o[cn.PATIENT_URGENCY].str.replace(r'_.+', '', regex=True)
    d_o.loc[:, cn.PATIENT_HLA_LITERAL] = (
        fix_hla_string(
            d_o.loc[:, cn.PATIENT_HLA_LITERAL]
        )
    )

    d_o = d_o.drop_duplicates()
    d_o.columns = d_o.columns.str.lower()

    d_o[cn.D_DCD] = d_o[cn.D_DCD].fillna(0)

    # Assert column names are correct
    if usecols is None:
        for col in MATCHLIST_COLS:
            assert col in d_o.columns, \
                f'{col} should be column for {input_path}'

    return d_o


def read_patients(
    input_path: str,
    datecols: Optional[List[str]] = None,
    usecols: Optional[List[str]] = None,
    **kwargs
) -> pd.DataFrame:
    """"Read in patient file."""

    if datecols is None:
        datecols = [cn.TIME_REGISTRATION, cn.R_DOB, cn.DATE_FIRST_DIAL]
    if usecols is None:
        usecols = list(dtypes.DTYPE_PATIENTLIST.keys())

    data_ = _read_with_datetime_cols(
        input_path=input_path,
        dtps=dtypes.DTYPE_PATIENTLIST,
        casecols=False,
        usecols=usecols,
        datecols=datecols,
        **kwargs
    )
    assert isinstance(data_, pd.DataFrame), \
        f'Expected DataFrame, not {type(data_)}'

    # Sort recipients by id & time of registration.
    data_: pd.DataFrame = data_.sort_values(
        by=[cn.ID_RECIPIENT, cn.TIME_REGISTRATION]
    )
    data_.reset_index(inplace=True)

    # Add re-transplantation information
    idx = data_[cn.TIME_SINCE_PREV_TXP].notna()

    data_.loc[idx, cn.PREV_TX_DATE] = (
        data_.loc[idx, cn.TIME_REGISTRATION] -
        data_.loc[idx, cn.TIME_SINCE_PREV_TXP].apply(
            lambda x: timedelta(days=x)
        )
    )

    return data_


def read_rescue_probs(
    input_path: str
) -> Dict[str, ndarray]:
    """Read in rescue probabilities"""
    rescue_probs = pd.read_csv(
        input_path,
        delimiter=','
    )
    if 'strata' in rescue_probs.columns:
        rescue_probs = {
            n: x.iloc[:, 0:2] for n, x in rescue_probs.groupby('strata')
            }
        rescue_probs = {
            k: {
                cn.N_OFFERS_TILL_RESCUE: v['offers_before_rescue'].to_numpy(),
                cn.PROB_TILL_RESCUE: v['prob'].to_numpy()
            }
            for k, v in rescue_probs.items()
        }
    else:
        rescue_probs = {
            cn.N_OFFERS_TILL_RESCUE: rescue_probs.loc[:, 'offers_before_rescue'].to_numpy(),
            cn.PROB_TILL_RESCUE: rescue_probs.loc[:, 'prob'].to_numpy()
        }
    return rescue_probs


def read_donors(
        input_path: str,
        datecols: Optional[List[str]] = None,
        usecols: Optional[List[str]] = None,
        **kwargs
) -> pd.DataFrame:
    """"Read in patient file."""

    if usecols is None:
        usecols = list(dtypes.DTYPE_DONORLIST.keys())
    if datecols is None:
        datecols = [cn.D_DATE]

    data_: pd.DataFrame = _read_with_datetime_cols(
        input_path=input_path,
        dtps=dtypes.DTYPE_DONORLIST,
        casecols=True,
        usecols=usecols,
        datecols=datecols,
        **kwargs
    )

    for col in (cn.DONOR_HLA, ):
        data_[col] = fix_hla_string(data_[col])

    # Remove donors not from ET
    data_ = data_[data_[cn.D_COUNTRY].isin(es.ET_COUNTRIES)]

    data_ = data_.fillna(
        value=dtypes.DTYPE_DONOR_FILL_NAS
    )

    return data_


def read_donor_balances(
        input_path: str,
        datecols: Optional[List[str]] = None,
        usecols: Optional[List[str]] = None,
        **kwargs
) -> pd.DataFrame:
    """"Read in patient file."""

    if usecols is None:
        usecols = list(dtypes.DTYPE_DONORBALLIST.keys())
    if datecols is None:
        datecols = [cn.D_DATE]

    data_: pd.DataFrame = _read_with_datetime_cols(
        input_path=input_path,
        dtps=dtypes.DTYPE_DONORBALLIST,
        casecols=True,
        usecols=usecols,
        datecols=datecols,
        **kwargs
    )

    # Remove donors not from ET
    data_ = data_[data_[cn.D_ALLOC_COUNTRY].isin(es.ET_COUNTRIES)]
    data_ = data_.fillna(
        value=dtypes.DTYPE_DONOR_FILL_NAS
    )
    return data_

def read_historic_donor_balances(
        input_path: str,
        sim_start_date: datetime,
        max_window_length: 99999,
        datecols: Optional[List[str]] = None,
        usecols: Optional[List[str]] = None,
        **kwargs
):
    data_ = read_donor_balances(
        input_path=input_path,
        usecols=usecols,
        datecols=datecols,
        **kwargs
    )

    # Filter to donors which occured prior to simulation start date
    data_ = data_[data_[cn.D_DATE] <= sim_start_date]
    data_ = data_[data_[cn.D_DATE] >= sim_start_date - timedelta(days=max_window_length)]

    return data_


def read_nonetkasesp_balances(
        input_path: str,
        sim_start_date: datetime,
        sim_end_date: datetime,
        datecols: Optional[List[str]] = None,
        usecols: Optional[List[str]] = None,
        **kwargs
) -> pd.DataFrame:
    """"Read in patient file."""

    data_ = read_donor_balances(
        input_path=input_path,
        usecols=usecols,
        datecols=datecols,
        **kwargs
    )

    # Filter to donors which occured during the sim period
    data_ = data_[data_[cn.D_DATE] >= sim_start_date]
    data_ = data_[data_[cn.D_DATE] <= sim_end_date]

    # Filter to non-ETKAS/ESP donors.
    data_ = data_[data_[cn.NON_ETKAS_ESP] == 1]

    data_ = data_.fillna(
        value=dtypes.DTYPE_DONOR_FILL_NAS
    )

    return data_


def read_donor_pool(
        input_path: str,
        datecols: Optional[List[str]] = None,
        usecols: Optional[List[str]] = None,
        **kwargs
) -> pd.DataFrame:
    """"Read in patient file."""

    if usecols is None:
        usecols = list(dtypes.DTYPE_DONORLIST.keys())

    data_: pd.DataFrame = _read_with_datetime_cols(
        input_path=input_path,
        dtps=dtypes.DTYPE_DONORLIST,
        casecols=True,
        usecols=usecols,
        datecols=datecols,
        **kwargs
    )

    data_[cn.DONOR_HLA] = fix_hla_string(
        data_[cn.DONOR_HLA]
    )

    data_ = data_.fillna(
        value=dtypes.DTYPE_DONOR_FILL_NAS
    )

    return data_


def read_acos(input_path: str, **kwargs) -> pd.DataFrame:
    """"Read in patient file."""

    data_ = pd.read_csv(
        filepath_or_buffer=input_path,
        dtype=dtypes.DTYPE_ACOS,
        date_parser=read_datetimes_dayfirst,
        usecols=list(dtypes.DTYPE_ACOS.keys()),
        **kwargs
    )
    assert isinstance(data_, pd.DataFrame), \
        f'Expected DataFrame, not {type(data_)}'
    data_.columns = data_.columns.str.lower()

    return data_


def read_diags(input_path: str, **kwargs) -> pd.DataFrame:
    """"Read in patient file."""

    data_ = pd.read_csv(
        filepath_or_buffer=input_path,
        dtype=dtypes.DTYPE_DIAGS,
        date_parser=read_datetimes_dayfirst,
        usecols=list(dtypes.DTYPE_DIAGS.keys()),
        **kwargs
    )
    assert isinstance(data_, pd.DataFrame), \
        f'Expected DataFrame, not {type(data_)}'
    data_.columns = data_.columns.str.lower()

    return data_


def read_travel_times(
    path_drive_time: str = es.PATH_DRIVING_TIMES
) -> Dict[str, Dict[str, float]]:
    """Read in expected travelling times by car between centers"""

    travel_info = pd.read_csv(
        path_drive_time
    )

    # Select the right travel time.
    travel_dict = (
        travel_info.groupby(cn.FROM_CENTER)[
            [cn.FROM_CENTER, cn.TO_CENTER, cn.DRIVING_TIME]
        ].apply(lambda x: x.set_index(cn.TO_CENTER).to_dict(orient='index'))
        .to_dict()
    )

    return(travel_dict)


def read_profiles(input_path: str, usecols: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
    """"Read in patient file."""

    if usecols is None:
        usecols = list(dtypes.DTYPE_PROFILES.keys())

    data_ = _read_with_datetime_cols(
        input_path,
        dtps=dtypes.DTYPE_PROFILES,
        casecols=False,
        **kwargs
    )

    assert isinstance(data_, pd.DataFrame), \
        f'Expected DataFrame, not {type(data_)}'
    data_.columns = data_.columns.str.lower()

    return data_


def read_status_updates(
        input_path: str, sim_set: DotDict,
        start_date_col: Optional[str] = None,
        end_date_col: Optional[str] = None,
        **kwargs) -> pd.DataFrame:
    """Read in status updates."""

    d_s = pd.read_csv(
        input_path,
        dtype=dtypes.DTYPE_STATUSUPDATES,
        usecols=list(dtypes.DTYPE_STATUSUPDATES.keys()),
        parse_dates=[cn.LISTING_DATE],
        **kwargs
    )
    assert isinstance(d_s, pd.DataFrame), \
        f'Expected DataFrame, not {type(d_s)}'
    d_s.columns = d_s.columns.str.lower()

    end_date: datetime = (
        sim_set.SIM_END_DATE if not end_date_col
        else sim_set.__dict__[end_date_col]
    )
    d_s = d_s.loc[
        d_s[cn.LISTING_DATE] <= end_date,
        :
    ]

    # Set variable detail to removal reason for patients removed.
    d_s.loc[
        d_s[cn.STATUS_VALUE] == 'R',
        cn.STATUS_DETAIL
    ] = d_s.loc[
        d_s[cn.STATUS_VALUE] == 'R',
        cn.REMOVAL_REASON
    ]

    # Fix hla strings
    for type_upd in (mgr.HLA, mgr.UNACC):
        d_s.loc[d_s[cn.TYPE_UPDATE] == type_upd, cn.STATUS_VALUE] = (
            fix_hla_string(d_s.loc[d_s[cn.TYPE_UPDATE] == type_upd, cn.STATUS_VALUE])
        )


    return d_s


def read_sim_settings(
        ss_path: str,
        date_settings: Optional[List[str]] = None
) -> DotDict:
    """Read in simulation settings"""
    with open(ss_path, "r", encoding='utf-8') as file:
        sim_set: Dict[str, Any] = yaml.load(file, Loader=yaml.FullLoader)

    if date_settings is None:
        date_settings = [
            'SIM_END_DATE', 'SIM_START_DATE',
            'LOAD_RETXS_TO', 'LOAD_RETXS_FROM'
        ]

    # Fix dates
    min_time = datetime.min.time()
    for k in date_settings:
        sim_set[k] = datetime.combine(
            sim_set[k], min_time
        )

    sim_set['calc_etkas_score'] = MatchPointFunction(
        intercept = sim_set[cn.POINTS_ETKAS.upper()]['INTERCEPT'],
        coef = sim_set[cn.POINTS_ETKAS.upper()],
        points_comp_to_group=es.POINT_COMPONENTS_TO_GROUP
    )
    sim_set['calc_esp_score'] = MatchPointFunction(
        intercept = sim_set[cn.POINTS_ESP.upper()]['INTERCEPT'],
        coef = sim_set[cn.POINTS_ESP.upper()],
        points_comp_to_group=es.POINT_COMPONENTS_TO_GROUP
    )

    sim_set['times_esp_eligible'] = {
        k: (v - sim_set['SIM_START_DATE']) / timedelta(days=1)
        for k, v in es.ESP_CHOICE_ENFORCED.items()
    }

    return DotDict(sim_set)

