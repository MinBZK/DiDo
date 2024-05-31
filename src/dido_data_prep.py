"""
Prepares data for import into an SQL database.
"""

import os
import gc
import sys
import csv
import time
import copy
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from requests.auth import HTTPBasicAuth

import psutil

import s3_helper
import api_postcode
import dido_common as dc
import simple_table as st

from dido_common import DiDoError

# pylint: disable=bare-except, line-too-long, consider-using-enumerate
# pylint: disable=logging-fstring-interpolation, too-many-locals
# pylint: disable=pointless-string-statement, consider-using-dict-items

# print all columns of dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# raise exceptions on all numpy warnings
np.seterr(all='raise')


def find_data_files(supplier_config, supplier_id: str, root_directory: str):
    """ Looks for datafiles in the root/data directory

    Args:
        root_directory (str): name of the root directory
        suppliers (_type_): list of suppliers

    Returns:
        dict: dictionary for each supplier with data files
    """

    # build a dictionary containing the suppliers containing data
    data_dirs = {}

    if 'data_file' in supplier_config:
        data_file = supplier_config['data_file']

        # when data_file is specified it must contain something, if not: crash
        if len(data_file) == 0:
            raise DiDoError('"data_file" specification in SUPPLIERS section is empty')

        server = ''
        server_path = ''

        # check if first character in data_file is a /
        if data_file[0] == '/':
            # absolute path
            pad, fn, ext = dc.split_filename(data_file)
            filename = fn + ext

        # check id data reside on s3 bucket
        elif data_file[:2] == 's3':
            server = 's3'
            pad, fn, ext = dc.split_filename(data_file)
            server_path = data_file
            filename = fn + ext
            pad = os.path.join(root_directory, 'data', supplier_id)

        else:
            # path relative to root_directory/data
            pad = os.path.join(root_directory, 'data', supplier_id)
            filename = data_file

        # if

        header_file = ''
        if 'header_file' in supplier_config:
            header_file = supplier_config['header_file']

        supp_dict = {'data_file': filename,
                     'header_file': header_file,
                     'server': server,
                     'server_path': server_path,
                     'path': pad
                    }

        data_dirs[supplier_id] = supp_dict

    # if

    return data_dirs

### find_data_files ###


def load_schema(table_name: str, server_config: dict) -> pd.DataFrame:
    """ Load a table from a schema and database specified in server_config

    Args:
        table_name (str): name of the table to load
        server_config (dict): dictionary containing postgres parameters

    Returns:
        pd.DataFrame: SQL table loaded from postgres
    """
    return st.sql_select(
        table_name = table_name,
        columns = '*',
        sql_server_config = server_config,
        verbose = False,
    ).fillna('')

### load_schema ###


def check_data_dirs(data_dirs: dict) -> bool:
    """ Checks whether no more than one data file is supplied in todo

    Args:
        data_dirs (dict): data directories containing data
    """
    ok = True
    # check the number of data files (= .csv files) per supplier
    # must be 0 or 1
    for supplier in data_dirs.keys():
        file_list = data_dirs[supplier]['data_file']
        if len(file_list) > 1:
            ok = False
            logger.error(f'*** only up to 1 data file allowed in {supplier}')

    # for

    return ok

### check_data_dirs ###


def check_kolom(data: pd.DataFrame, schema: pd.DataFrame, kolom: str):
    """ look if columns names can be found in first line
        if so, then headers present, else not

    Args:
        data (pd.DataFrame): first line of data
        schema (pd.DataFrame): schema
        kolom (str): column name to match

    Returns:
        bool: True of all data row elements match with schema.columns
    """

    for i in range(data.shape[0]):
        # find index of 'kolomnaam' in schema.columns
        columns = schema.columns.tolist()
        index = columns.index(kolom)

        # first_line_of_data contains first line of the dat file
        # if it is a header, the names must be equal in case and name with
        # 'kolomnaam' in schema
        if data.iloc[0, i] != schema.iloc[i, index]:

            return False

        # if
    # for

    return True

### check_kolom ###


def get_bootstrap_data_headers(server_config: dict):
    # Read bootstrap data
    bootstrap_data = dc.load_odl_table(dc.EXTRA_TEMPLATE, server_config)
    columns = bootstrap_data.loc[:, 'kolomnaam'].tolist()

    return columns

### get_bootstrap_data_headers ###


def load_pdirekt_header_file(supplier_config: dict,
                              supplier: str,
                              schema: pd.DataFrame,
                              encoding: str,
                             ):
    """ Loads a P-Direkt header file

    It returns column FIELDNAME as a results

    Args:
        supplier_config (dict): configuration of the supply
        supplier (str): name of the supplier
        schema (pd.DataFrame): schema definition
        encoding (str): encoding of the file, same as data

    Returns:
        list: the content of the FILEDNAME column as a list
    """
    data_dicts = dc.get_par(supplier_config, 'data_description')

    _, fn, ext = dc.split_filename(data_dicts[supplier]['header_file'])
    filename = os.path.join(data_dicts[supplier]['path'], fn + ext)

    logger.info('')
    logger.info(f'Reading header file {fn} , encoding: {encoding}')

    # fetch the header file
    server_type = data_dicts[supplier]['server']
    if server_type == 's3':
        server_path = data_dicts[supplier]['header_file']
        s3_helper.s3_command_get_file(
            download_to = filename,
            filepath_s3 = server_path,
            force_overwrite = True,
        )

    # load the header into a dataframe
    headers = pd.read_csv(
        filename,
        sep = ';',
        dtype = str,
        keep_default_na = False,
        skiprows = 5,
        engine = 'c',
        encoding = encoding,
    )

    column_names = headers['FIELDNAME'].tolist()

    return column_names

### load_pdirekt_header_file ###


def load_data(supplier_config: dict,
              supplier: str,
              schema: pd.DataFrame,
              headers: dict,
              sample_size: int,
              server_config: dict,
              encoding: str,
             ) -> pd.DataFrame:
    """ Loads data from file for a supplier

    Args:
        supplier_config (dict): dictionary with info for each supplier
        supplier (str): name of the supplier to load data for

    Returns:
        pd.DataFrame: csv file loaded into dataframe
    """
    data_dicts = dc.get_par(supplier_config, 'data_description')

    # get the delivery_type, if omitted use mode: insert as default
    # delivery_type = dc.get_par(supplier_config, 'delivery_type', {'mode': 'insert'})

    cpu = time.time()

    _, fn, _ = dc.split_filename(data_dicts[supplier]['data_file'])
    filename = os.path.join(data_dicts[supplier]['path'],
                            data_dicts[supplier]['data_file'])

    logger.info('')
    logger.info(f'Reading data file {fn}, encoding: {encoding}')

    # test if data is to be fetched from s3: if so, copy to root/data/<supplier>
    server_type = data_dicts[supplier]['server']
    if server_type == 's3':
        server_path = data_dicts[supplier]['server_path']
        s3_helper.s3_command_get_file(
            download_to = filename,
            filepath_s3 = server_path,
            force_overwrite = True
        )

    # set delimiter when present
    delimiter = dc.get_par(supplier_config, 'delimiter', ';')

    # check if HEADERS is specified for this supplier
    headers_present = False
    if supplier in headers.keys():
        headers_present = headers[supplier]

    logger.info ('')
    logger.info(f'Reading: {filename}')
    if not headers_present:
        logger.warning('!!! Data contains no headers')
        logger.warning('!!! Headers are used from the schema in schema order')
        logger.warning('!!! DiDo cannot verify the correctness of the headers, please do so yourself')
        logger.info('')

        # read the data using the C engine, this one is better
        # to handle large files
        data = pd.read_csv(filename,
            sep = delimiter,
            dtype = str,
            keep_default_na = False,
            header = None,
            nrows = sample_size,
            engine = 'c',
            encoding = encoding,
        )

    else:
        logger.info('Reading with headers')
        data = pd.read_csv(
            filename,
            sep = delimiter,
            dtype = str,
            keep_default_na = False,
            nrows = sample_size,
            engine = 'c',
            encoding = encoding,
        )

    # if

    # find out if the file contains headers or if a header sheet is provided

    # Fetch schema columns
    columns = schema['kolomnaam'].tolist()

    # The data is provided raw, without columns and without bronbestand data
    # To add columns remove the bronbestand columns from the schema kolomnaam
    # This yields a list of columns which can be added to the data
    extra_columns = get_bootstrap_data_headers(server_config)
    for col in extra_columns:
        columns.remove(col)

    ram = data.memory_usage(index=True).sum()
    cpu = time.time() - cpu

    logger.info(f'[{len(data)} records loaded in {cpu:.0f} seconds]')
    logger.info(f'[{len(data)} records and {len(data.columns)} columns require {ram:,} bytes of RAM]')

    return data

### load_data ###


def select_rows(data: pd.DataFrame,
                # supplier_config: dict,
                supplier: str,
                schema: pd.DataFrame,
                meta: pd.DataFrame,
                select_data: dict,
                # headers: dict,
                # sample_size: int,
                # server_config: dict,
                # encoding: str,
               ) -> pd.DataFrame:

    select_data = select_data[supplier]
    cols = list(select_data.keys())
    if len(cols) > 0:
        logger.info('[selecting rows from data]')

        # Initialize variables used in iteration
        for col in cols:
            old_size = len(data)

            selection_list = select_data[col]
            data = data[data[col].isin(selection_list)]
            new_size = len(data)
            logger.info(f'Selecting for column {col}: size from {old_size} to {new_size}')

        # for
        logger.info('')

    # if

    return data

### select_rows ###


def evaluate_headers(data: pd.DataFrame,
                     supplier_config: dict,
                     supplier: str,
                     schema: pd.DataFrame,
                     headers: dict,
                     encoding: str,
                    ):
    """ Compares a list of headers with schema columns and flags inconsistencies

    Args:
        data (pd.DataFrame): data frame containing data and headers
        supplier_config (dict): dictionary with info for each supplier
        supplier (str): current supplier
        schema (pd.DataFrame): schema definition of the data
    """
    data_dicts = dc.get_par(supplier_config, 'data_description')
    header_columns = None

    logger.info('')

    # 1. if not, test if headers are present in the data
    headers_present = False
    if supplier in headers.keys():
        headers_present = headers[supplier]

        if headers_present:
            header_columns = data.columns

            logger.info('Headers present in data')
            logger.debug(f'Data columns: {header_columns}')

    # 2. test if a header file is present
    if 'header_file' in supplier_config.keys():
        # a header sheet is provided

        _, fn, ext = dc.split_filename(data_dicts[supplier]['header_file'])
        header_columns = load_pdirekt_header_file(
            supplier_config = supplier_config,
            supplier = supplier,
            schema = schema,
            encoding = encoding,
        )

        logger.info(f'Headers present in header file {fn}{ext}')
        logger.debug(f'Columns: {header_columns}')

        if headers_present:
            logger.warning('!!! You specify that headers are present and a header file')
            logger.warning('!!! This means that the headers from the header file are used')
        # if
    # if

    # convert header_columns to list
    header_columns = list(header_columns)

    # TODO: to strip the dido meta columns from the schema, simply the first
    # 6 columns are stripped. This should be derived from the extra_columns themselves

    # 3. if not, get them from the schema
    if header_columns is None or len(header_columns) == 0:
        header_columns = schema['kolomnaam'].tolist()[6:]

        logger.info('No headers supplied, taken from schema')
        logger.debug(f'Columns: {header_columns}')

    header_columns = [dc.change_column_name(x) for x in header_columns]

    # remove meta columns from schema, currently the first six columns
    schema_columns = schema['kolomnaam'].tolist()[6:]

    # now they should be of equal length
    errors = False
    logger.info(f'Number of schema columns: {len(schema_columns)} vs. supplied headers: {len(header_columns)}')

    # if not, examine what's wrong
    if len(schema_columns) != len(header_columns):
        errors = True
        logger.error('Number of columns do not match')

        logger.info('')
        logger.info('*** Header columns not in schema:')
        for col in header_columns:
            if col not in schema_columns:
                logger.info(col)

        logger.info('')
        logger.info('*** Data dictionary columns not in headers:')
        for col in schema_columns:
            if col not in header_columns:
                logger.info(col)

    else:
        # if lengths are equal, column names should be as well
        for i in range(len(header_columns)):
            if schema_columns[i] != header_columns[i]:
                errors = True
                logger.error(f'*** schema column name, {schema_columns[i]}, '
                             f'not equal to header column name: {header_columns[i]}')
            # if
        # for
    # if

    if errors:
        raise DiDoError('*** Serious problems in description of the data, DiDo cannot continue')

    else:
        logger.info('[Provided header names are consistrent with schema]')

    return header_columns

### evaluate_headers ###


def process_strip_space(data: pd.DataFrame,
                        supplier: str,
                        schema: pd.DataFrame,
                        meta: pd.DataFrame,
                        strip_space: dict,
                       ):
    mem = psutil.virtual_memory()
    seq = 0

    # see if white space strip is defined for supplier
    strip_cols = strip_space[supplier]
    if strip_cols == ['*']:
        strip_cols = data.columns.tolist()

    logger.info('[Stripping white space]')

    # strip white space from designated columns
    for col in strip_cols:
        if col in data.columns:
            pg_col = dc.change_column_name(col)
            if schema.loc[pg_col, 'leverancier_kolomnaam'] != dc.VAL_DIDO_GEN:
                elapsed = time.time()
                #data[col] = data[col].replace({"^\s+|\s+$": ""}, regex = True)
                data[col] = data[col].str.strip()
                elapsed = time.time() - elapsed
                seq += 1
                logger.info(f'{seq:4d}.  {col} ({elapsed:.0f}s, {mem.used:,} Bytes)')

        else:
            logger.warning(f'!!! STRIP_SPACE: column {col} not present in the data, ignored.')

        # if
    # for

    return data

### process_strip_space ###


def process_comma_conversion(data: pd.DataFrame,
                             supplier: str,
                             schema: pd.DataFrame,
                             meta: pd.DataFrame,
                             real_types: list,
                           ):

    mem = psutil.virtual_memory()
    cols = schema.index.tolist()
    seq = 0

    # create a translation table for decimal comma
    translation_table = str.maketrans({',': '.'})
    logger.info('')
    logger.info('[Decimal conversion for column with real and like datatypes]')

    # data_cols = [dc.change_column_name(col) for col in data.columns]
    for col in data.columns:
        # if col in data.columns:
        pg_col = dc.change_column_name(col)
        if schema.loc[pg_col, 'leverancier_kolomnaam'] != dc.VAL_DIDO_GEN:
            datatype = schema.loc[pg_col, 'datatype']

            # only convert for Postgres real types
            if datatype in real_types:
                elapsed = time.time()
                data[col] = data[col].str.translate(translation_table)
                elapsed = time.time() - elapsed
                seq += 1
                logger.info(f'{seq:4d}. {col}: {datatype} ({elapsed:.0f}s, {mem.used:,} Bytes)')
            # if
        # if
    # for

    return data

### process_comma_conversion ###


def process_renames(data: pd.DataFrame,
                    supplier: str,
                    schema: pd.DataFrame,
                    meta: pd.DataFrame,
                    renames: dict,
                   ):

    mem = psutil.virtual_memory()
    cols = renames[supplier]
    seq = 0

    if cols is not None: # no renames
        logger.info('')
        logger.info('[Renaming data in columns]')

        # Initialize variables used in iteration
        rename_dict = {}
        todo_cols = list(cols.keys())
        rename_cols = []

        logger.info('[Solving datatypes]')
        # Translate generic data types into corresponding columns
        for col in todo_cols:
            new_col = col.strip()
            if len(new_col) > 1:
                # test if column name is <datatype>
                if new_col[0] == '<' and new_col[-1] == '>':
                    new_col = new_col[1:-1]

                    # collect all columns with that datatype
                    logger.info('')
                    logger.info(f'[Columns ascribed to datatype {col}]')
                    datatypes_found = schema[schema['datatype'] == new_col]
                    if len(datatypes_found) == 0:
                        logger.warning(f'!!! None found')

                    # The index of iterrows is kolomnaam
                    for col_name, row in datatypes_found.iterrows():
                        # add all columns with this datatype and with the rename
                        # value to the rename_dict dictionary instead of the
                        # <datatype> specification
                        if col_name not in rename_cols and col_name in data.columns:
                            seq += 1
                            logger.info(f'{seq:4d}. {col_name}')
                            rename_dict[col_name] = cols[col]

                        # if
                    # for

                else:
                    # "normal" column, add to dictionary
                    rename_dict[col_name] = cols[col]

                # if

            else:
                # "normal" column, add to dictionary
                rename_dict[col_name] = cols[col]

            # if
        # for

        logger.info('')
        logger.info('[Renaming]')
        seq = 0
        for col, rename_values in rename_dict.items():
            # if exists: rename data
            if col in data.columns:
                if schema.loc[col, 'leverancier_kolomnaam'] != dc.VAL_DIDO_GEN:
                    # rename_values = rename_dict[col]
                    regex = False
                    if 're' in rename_values.keys():
                        regex = rename_values['re']
                        rename_values.pop('re')

                    elapsed = time.time()
                    data[col] = data[col].replace(rename_values, regex = regex)
                    elapsed = time.time() - elapsed
                    seq += 1
                    logger.info(f'{seq:4d}. {col} ({elapsed:.0f}s, {mem.used:,} Bytes)')

            else:
                logger.warning(f'!!! Column to rename "{col}" does not exist, ignored.')

            # if
        # for
    # if

    return data

### process_renames ###

def process_data(data: pd.DataFrame,
                 supplier: str,
                 schema: pd.DataFrame,
                 meta: pd.DataFrame,
                 select_data: dict,
                 renames: dict,
                 real_types: list,
                 strip_space: dict,
                ):
    # set kolomnaam as index
    schema = schema.set_index('kolomnaam')

    mem = psutil.virtual_memory()
    dc.report_ram('Memory use at process_data')
    logger.info('')

    # select rows
    if select_data is not None and supplier in select_data.keys():
        cpu = time.time()

        data = select_rows(
            data = data,
            supplier = supplier,
            schema = schema,
            meta = meta,
            select_data = select_data,
        )

        cpu = time.time() - cpu
        logger.info(f'{len(data)} records stripped from whitespace in {cpu:.0f} seconds')

    # if

    # strip space left and right
    if strip_space is not None and supplier in strip_space.keys():
        cpu = time.time()

        data = process_strip_space(
            data = data,
            supplier = supplier,
            schema = schema,
            meta = meta,
            strip_space = strip_space,
        )

        cpu = time.time() - cpu
        logger.info(f'{len(data)} records stripped from whitespace in {cpu:.0f} seconds')

    else:
        logger.info('[White space will not be stripped from any column]')

    # if

    # convert decimal commas for real values
    decimaal_teken = meta.iloc[0].loc['bronbestand_decimaal']

    # only convert decimal commas when specified in meta data
    if decimaal_teken != '.':
        cpu = time.time()

        data = process_comma_conversion(
            data = data,
            supplier = supplier,
            schema = schema,
            meta = meta,
            real_types = real_types,
        )

        cpu = time.time() - cpu
        logger.info(f'{len(data)} records processed for decimal points in {cpu:.0f} seconds')

    else:
        logger.info('[Decimal point has been specified for this data, no decimal point conversion]')

    # if

    # rename data
    if renames is not None and supplier in renames.keys():
        cpu = time.time()

        data = process_renames(
            data = data,
            supplier = supplier,
            schema = schema,
            meta = meta,
            renames = renames,
        )

        cpu = time.time() - cpu
        logger.info(f'{len(data)} records renamed in {cpu:.0f} seconds')

    else:
        logger.info('[No renames have been specified]')

    # if

    # reset schema to original state
    schema = schema.reset_index()
    logger.info('')

    return data

### process_data ###


def save_data(data: pd.DataFrame,
              data_dirs: dict,
              supplier: str,
              working_directory: str
             ):
    cpu = time.time()

    _, fn, _ = dc.split_filename(data_dirs[supplier]['data_file'])
    savename = os.path.join(working_directory,
                           'todo',
                           supplier,
                           data_dirs[supplier]['data_file'])
    logger.info(f'Writing: {savename}')

    data.to_csv(savename, sep = ';', index = False, quoting = csv.QUOTE_ALL, encoding = 'utf8')

    cpu = time.time() - cpu

    logger.info(f'{len(data)} records written in {cpu:.0f} seconds')

    return


def get_api_potential_deliveries(origin: dict, config_dict: dict) -> pd.DataFrame:
    # Read the api keys from environment file
    key = config_dict['KEY']
    secret = config_dict['SECRET']

    # read url's to access the data
    url_account = origin['url_account']
    url_delivery =  origin['url_delivery']

    # authenticate
    authentication = HTTPBasicAuth(key, secret)

    # get other variables: datetime.strptime(dateString, "%d-%B-%Y")
    start_date = origin['start_date']

    # get account id, quit when unsuccesful
    status, number = api_postcode.get_account_id(url_account, authentication)
    if status != api_postcode.HTTP_OK:
        logger.error(f'*** Error ({status}) when trying to login into API account, exit 1')

        return 1, None

    else:
        logger.info('')
        logger.info(f'==> logged in into account {number}')

    # if

    status, subs = api_postcode.get_account_subscriptions(url_account, authentication, number)
    if status != api_postcode.HTTP_OK:
        logger.error(f'*** Error ({status}) when trying to get subscription from API account, exit 2')

        return 2, None

    else:
        logger.info('')
        logger.info(f'==> Succesfully obtained subscriptions')
        api_postcode.show_account_info([subs])

    # if

    from_date = (start_date + timedelta(days = 1)).strftime('%Y%m%d')
    to_date = datetime.today().strftime('%Y%m%d')

    logger.info(f'==> Fetching deliveries valid between {from_date} and {to_date}')
    status, delivs = api_postcode.get_subscriptions(url_delivery,
                                       authentication,
                                       from_date,
                                       to_date)
    if status != api_postcode.HTTP_OK:
        logger.error(f'*** Error ({status}) when trying to get pending deliveries, exit 3')

        return 3, None

    else:
        logger.info('')
        logger.info(f'==> Got info on deliveries to download')
        api_postcode.show_account_info(delivs)

    # if

    deliveries = pd.DataFrame(delivs)

    return 0, deliveries

### get_api_possible_downloads ###


def get_db_already_delivered(table_name: str, server_config: dict) -> pd.DataFrame:
    delivered = st.table_to_dataframe(table_name, sql_server_config = server_config)

    return delivered

### get_db_already_delivered ###


def get_cached_deliveries(cache: str) -> pd.DataFrame:

    cached_files = [f for f in os.listdir(cache)
                    if os.path.isfile(os.path.join(cache, f))]

    file_list = []
    for filename in cached_files:
        splits = filename.split('_')
        if splits[0] == 'full':
            file_list.append({'file': filename,
                              'from': '20130101',
                              'to': splits[1]})
        else:
            splits = splits[1].split('-')
            file_list.append({'file': filename,
                              'from': splits[0],
                              'to': splits[1][:-4]})

    cached = pd.DataFrame(file_list)

    return cached

### get_cached_deliveries ###


def get_list_of_deliveries(origin: dict,
                           config_dict: dict,
                           table_name: str,
                           server_config: dict,
                           cache_dir: str):

    _, potential_deliveries = get_api_potential_deliveries(origin, config_dict)
    already_delivered = get_db_already_delivered(table_name, server_config)
    cached_deliveries = get_cached_deliveries(cache_dir)

    return already_delivered

### get_list_of_deliveries ###


def get_api_downloads(deliveries):
    logger.info(f'===> Downloading deliveries to {work_dir}')
    status, downloads = api_postcode.get_deliveries(url_delivery, authentication, delivs, work_dir)
    if status != api_postcode.HTTP_OK:
        logger.error(f'*** Error ({status}) downloading deliveries failed, exit 4')

        return 4

    else:
        logger.info('')
        logger.info(f'==> Downloading files succesful: {downloads}')

    # if

    return 0

### get_api_downloads ###


def prepare_one_delivery(cargo: dict,
                        leverancier_id: str,
                        project_name: str,
                        real_types: list,
                       ):

    config_dict = cargo['config']
    delivery_config = cargo['delivery']

    # get some limiting variables
    sample_size, sample_fraction, max_errors = dc.get_limits(delivery_config)

    root_dir = config_dict['ROOT_DIR']
    work_dir = config_dict['WORK_DIR']
    db_servers = config_dict['SERVER_CONFIGS']
    select_data = dc.get_par(delivery_config, 'SELECT', None)
    renames = dc.get_par(delivery_config, 'RENAME_DATA', None)
    strip_space = dc.get_par(delivery_config, 'STRIP_SPACE', None)
    headers = dc.get_par(delivery_config, 'HEADERS', {})

    origin = dc.get_par(cargo, 'origin', {'input': '<file>'})
    if origin['input'] == '<table>':
        logger.info(f'Input for supplier {leverancier_id} is is a table: {origin["table_name"]}')
        logger.warning('!!! There are no checks on imported tables')

        return

    elif origin['input'] == '<api>':
        # TODO: this code probably does not work
        logger.info(f'Inlezen via API van {origin["source"]}')
        table_names = dc.get_table_names(project_name, leverancier_id)
        cache_dir = os.path.join(root_dir, 'data', leverancier_id)
        levering_table = get_list_of_deliveries(
            origin = origin,
            config_dict = config_dict,
            table_name = table_names[dc.TAG_TABLE_DELIVERY],
            server_config = db_servers['DATA_SERVER_CONFIG'],
            cache_dir = cache_dir,
        )

    elif origin['input'] != '<file>':
        raise DiDoError('*** origin "input" can only be <file>, <api> or <table>. ' \
                        '"' + origin["input"] + '" was found.')

    # if

    # get encoding of the data
    encoding = dc.get_par(cargo, 'data_encoding', 'utf8')

    if leverancier_id not in  cargo['data_description']:
        logger.info(f'No datafile specified for {leverancier_id}')

    else:
        # fetch schema from database
        schema_name = dc.get_table_name(project_name, leverancier_id, dc.TAG_TABLE_SCHEMA, 'description')
        leverancier_schema = load_schema(schema_name, db_servers['DATA_SERVER_CONFIG'])

        # fetch meta from database
        meta_name = dc.get_table_name(project_name, leverancier_id, dc.TAG_TABLE_META, 'data') # f'{project_name}_{leverancier_id}_{TAG_TABLE_META}_data'
        leverancier_metadata = load_schema(meta_name, db_servers['DATA_SERVER_CONFIG'])
        if len(leverancier_metadata) == 0:
            raise DiDoError('Meta data are empty, check whether the database has been correctly constructed (if at all)')

        dc.report_ram('[Memory use before loading data]')

        # Read data
        data = load_data(
            supplier_config = cargo,
            supplier = leverancier_id,
            schema = leverancier_schema,
            headers = headers,
            sample_size = sample_size,
            server_config = db_servers['ODL_SERVER_CONFIG'],
            encoding = encoding,
        )

        # Acquire column names
        headers = evaluate_headers(
            data = data,
            supplier_config = cargo,
            supplier = leverancier_id,
            schema = leverancier_schema,
            headers = headers,
            encoding = encoding,
        )
        if headers is not None:
            data.columns = headers

        # Create a deepcopy of renames as it will be modified inside process_data
        rename_copy = copy.deepcopy(renames)

        # Rename applicable data
        data = process_data(
            data = data,
            supplier = leverancier_id,
            schema = leverancier_schema,
            meta = leverancier_metadata,
            select_data = select_data,
            renames = rename_copy,
            real_types = real_types,
            strip_space = strip_space,
        )

        # Save results
        save_data(
            data = data,
            data_dirs = cargo['data_description'],
            supplier = leverancier_id,
            working_directory = work_dir,
        )

        data = None
        gc.collect()
        dc.report_ram('End of prepare_one_delivery')

    # if

    return

### prepare_one_delivery ###


def dido_data_prep(header: str):
    cpu = time.time()

    # read commandline parameters
    appname, args = dc.read_cli()

    # read the configuration file
    config_dict = dc.read_config(args.project)
    dc.display_dido_header(header, config_dict)

    delivery_config = dc.read_delivery_config(
        project_path = config_dict['ROOT_DIR'],
        delivery_filename = args.delivery,
    )

    # get project environment
    # project_name = config_dict['PROJECT_NAME']
    root_dir = config_dict['ROOT_DIR']
    work_dir = config_dict['WORK_DIR']
    leveranciers = config_dict['SUPPLIERS']

    overwrite = dc.get_par(delivery_config, 'ENFORCE_PREP_IF_TABLE_EXISTS', False)

    # get the database server definitions
    db_servers = config_dict['SERVER_CONFIGS']

    # select which suppliers to process
    suppliers_to_process = dc.get_par(config_dict, 'SUPPLIERS_TO_PROCESS', '*')

    # get real_types
    allowed_datatypes, sub_types = dc.create_data_types()
    real_types = sub_types['real']

    # just * means process all
    if suppliers_to_process == '*':
        suppliers_to_process = leveranciers.keys()

    ok = True

    # process each supplier
    for leverancier_id in suppliers_to_process:
        dc.subheader(f'{leverancier_id}', '=')

        leverancier, projects = dc.get_supplier_projects(
            config = delivery_config,
            supplier = leverancier_id,
            # project_name = project_name,
            delivery = leverancier_id,
            keyword = 'DELIVERIES',
        )

        for project_key in projects.keys():
            dc.subheader(f'Project: {project_key}', '-')
            project = projects[project_key]

            # get all cargo from the delivery_dict
            cargo_dict = dc.get_cargo(delivery_config, leverancier_id, project_key)
            # cargo_dict = dc.get_cargo(projects, project_key)

            # process all deliveries
            for cargo_name in cargo_dict.keys():
                dc.subheader(f'Delivery: {cargo_name}', '.')

                # get cargo associated with the cargo_name
                cargo = cargo_dict[cargo_name]
                cargo = dc.enhance_cargo_dict(cargo, cargo_name, leverancier_id)

                # add config and delivery dicts as they are needed while processing the cargo
                cargo['config'] = config_dict
                cargo['delivery'] = delivery_config

                # present all deliveries and the selected one
                logger.info('Delivery configs supplied (> is selected)')
                for key in cargo_dict.keys():
                    if key == cargo_name:
                        logger.info(f" > {key}")
                    else:
                        logger.info(f" - {key}")
                    # if
                # for

                # delivery exists in the database. If so, skip this delivery
                if dc.delivery_exists(cargo, leverancier_id, project_key, db_servers):
                    logger.info('')
                    logger.error(f'*** delivery already exists: '
                                f'{leverancier_id} - {cargo_name}, skipped')

                    if not overwrite:
                        logger.info('Specify ENFORCE_PREP_IF_TABLE_EXISTS: yes in your delivery.yaml')
                        logger.info('if you wish to check the data quality')

                        continue

                    else:
                        logger.warning('!!! ENFORCE_PREP_IF_TABLE_EXISTS: yes specified, '
                                    'data will be overwritten')
                    # if
                # if
                data_description = find_data_files(cargo, leverancier_id, root_dir)
                cargo['data_description'] = data_description

                prepare_one_delivery(cargo, leverancier_id, project_key, real_types)

            # for
    # for

    cpu = time.time() - cpu
    logger.info('')
    logger.info(f'[Ready {appname["name"]} in {cpu:.0f} seconds]')
    logger.info('')

    return ok

### dido_data_prep ###


if __name__ == '__main__':
    # read commandline parameters
    cli, args = dc.read_cli()

    # create logger in project directory
    try:
        log_file = os.path.join(args.project, 'logs/' + cli['name'] + '.log')

    except:
        appname = cli['name'] + cli['ext']
        print(f'*** No log file found in project directory for {appname}')
        print('')
        if args.project is None:
            print('You specified no project directory')
            print('Are you sure you specified the correct --project <path/to/project> parameter?')

        else:
            print('Probabbly there is no logs directory in the project map')
            print('You also might have specified the wrong project directory')

        # if

        sys.exit(1)

    # try..except

    logger = dc.create_log(log_file, level = 'DEBUG')

    # go
    dido_data_prep('Header Check and Data Mover')
