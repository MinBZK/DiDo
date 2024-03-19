import os
import gc
import sys
import csv
import time
import copy
import psutil
import logging
import requests
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from requests.auth import HTTPBasicAuth

import s3_helper
import api_postcode
import dido_common as dc
import simple_table as st

from dido_common import DiDoError

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


def test_for_headers(supplier_config: dict,
                     filename: str,
                     schema: pd.DataFrame,
                     extra_list: list,
                     encoding: str = 'utf-8'
                    ):
    """ Check is the data file contains headers and whether the amount of columns are correct

    The number of columns in the data should equal those of the schema minus the
    number of meta columns from the extra table
    When it is a mutation data set extra columns are provided for the mutation scheme of
    the supplier. Should be for corrected as well.

    Args:
        supplier_config (dict): config dictionary
        filename (str): File to read data from
        schema (pd.DataFrame): schema of the data
        extra_list (list): list with extra columns
        encoding (str, optional): text encoding. Defaults to 'utf-8'.

    Returns:
        _type_: _description_
    """

    # fetch some parameters from the configuration file
    delimiter = dc.get_par(supplier_config, 'delimiter', ';')
    delivery_type = dc.get_par(supplier_config, 'delivery_type', {'mode': 'insert'})
    delivery_mode = dc.get_par(delivery_type, 'mode', 'insert')

    # fetch the mutation instructions when in mutation mode
    mutation_instr = []
    if delivery_mode == 'mutate':
        mutation_instr = dc.get_par(delivery_type, 'mutation_instructions', [])

    first_line_of_data = pd.read_csv(
        filename,
        sep = delimiter,
        dtype = str,
        keep_default_na = False,
        nrows = 1,
        header = None,
        index_col = None,
        engine = 'python',
        encoding = encoding,
    )

    n_schema_cols = schema.shape[0]
    n_data_cols = first_line_of_data.shape[1]
    n_instr_cols = len(mutation_instr)
    headers_present = 'none'

    if n_schema_cols == n_data_cols - n_instr_cols + len(extra_list):
        logger.info(f'schema and data have an equal amount of columns: {n_data_cols}')


        if not check_kolom(first_line_of_data, schema, 'kolomnaam'):
            if check_kolom(first_line_of_data, schema, 'leverancier_kolomnaam'):
                headers_present = 'leverancier_kolomnaam'

    else:
        logger.error('*** data and schema columns differ:')
        logger.error(f'schema: {schema.shape}')
        logger.error(f'data:   {first_line_of_data.shape}')

    # if

    return headers_present

### test_for_headers ###


def get_bootstrap_data_headers(server_config: dict):
    # Read bootstrap data
    bootstrap_data = dc.load_odl_table(dc.EXTRA_TEMPLATE, server_config)
    columns = bootstrap_data.loc[:, 'kolomnaam'].tolist()

    return columns

### test_for_headers ###


def load_separate_header_file(supplier_config: dict,
                              supplier: str,
                              schema: pd.DataFrame,
                              encoding: str,
                             ):
    data_dicts = dc.get_par(supplier_config, 'data_description')

    _, fn, ext = dc.split_filename(data_dicts[supplier]['header_file'])
    filename = os.path.join(data_dicts[supplier]['path'], fn + ext)

    logger.info(f'Reading header file {fn} , encoding: {encoding}')

    # fetch the header file
    server_type = data_dicts[supplier]['server']
    if server_type == 's3':
        server_path = data_dicts[supplier]['header_file']
        s3_helper.s3_command_get_file(
            download_to = filename,
            filepath_s3 = server_path,
            force_overwrite = True
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

    # Convert colmn names to Postgres standard names
    schema_columns = schema['kolomnaam'].tolist()[6:]
    header_columns = [dc.change_column_name(x)
                      for x in headers['FIELDNAME']]

    print(schema_columns)
    print(header_columns)

    print(f'lengths: schema {len(schema_columns)}, headers {len(header_columns)}')
    print('Header columns not in datadictionary:')
    for col in header_columns:
        if col not in schema_columns:
            print(col)

    print('Data dictionary columns not in headers:')
    for col in schema_columns:
        if col not in header_columns:
            print(col)

    return

### load_separate_header_file ###


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
    delivery_type = dc.get_par(supplier_config, 'delivery_type', {'mode': 'insert'})

    cpu = time.time()
    _, fn, _ = dc.split_filename(data_dicts[supplier]['data_file'])
    filename = os.path.join(data_dicts[supplier]['path'],
                            data_dicts[supplier]['data_file'])

    logger.info(f'Reading data file {fn}, encoding: {encoding}')

    server_type = data_dicts[supplier]['server']
    if server_type == 's3':
        server_path = data_dicts[supplier]['server_path']
        s3_helper.s3_command_get_file(
            download_to = filename,
            filepath_s3 = server_path,
            force_overwrite = True
        )

    # schema = data_dicts[supplier]['schema']
    columns = schema['kolomnaam'].tolist()

    # The data is provided raw, without columns and without bronbestand data
    # To add columns remove the brondbestand columns from the schema kolomnaam
    # This yields a list of columns which can be added to the data
    extra_columns = get_bootstrap_data_headers(server_config)
    for col in extra_columns:
        columns.remove(col)

    # set delimiter when present
    delimiter = dc.get_par(supplier_config, 'delimiter', ';')

    # find out if the file contains headers
    if supplier in headers.keys():
        headers_present = headers[supplier]

    else:
        headers_present = False

    # if

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
        data = pd.read_csv(filename,
            sep = delimiter,
            dtype = str,
            keep_default_na = False,
            nrows = sample_size,
            engine = 'c',
            encoding = encoding,
        )

    # if

    ram = data.memory_usage(index=True).sum()
    cpu = time.time() - cpu

    logger.info(f'[{len(data)} records loaded in {cpu:.0f} seconds]')
    logger.info(f'[{len(data)} records and {len(data.columns)} columns require {ram:,} bytes of RAM]')

    return data

### load_data ###


def process_data(data: pd.DataFrame,
                 supplier: str,
                 schema: pd.DataFrame,
                 meta: pd.DataFrame,
                 renames: dict,
                 real_types: list,
                 strip_space: dict,
                ):
    # set kolomnaam as index
    schema = schema.set_index('kolomnaam')

    mem = psutil.virtual_memory()
    dc.report_ram('Memory use at process_data')
    logger.info('')

    print(strip_space.keys())
    # strip space left and right
    if strip_space is not None and supplier in strip_space.keys():
        cpu = time.time()
        seq = 0

        # see if white space strip is defined for supplier
        strip_cols = strip_space[supplier]
        if strip_cols == ['*']:
            strip_cols = data.columns.tolist()

        logger.info('[Stripping white space]')

        # strip white space from designated columns
        #data.columns = [col.lower().strip() for col in data.columns]
        #print(data.columns)
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

        cpu = time.time() - cpu
        logger.info(f'{len(data)} records stripped from whitespace in {cpu:.0f} seconds')

    else:
        logger.info('[White space will not be stripped from any column]')

    # if

    # convert decimal commas for real values
    decimaal_teken = meta.iloc[0].loc['bronbestand_decimaal']

    # only convert decimal commas when specified in meta data
    if decimaal_teken != '.':
        cols = schema.index.tolist()
        cpu = time.time()
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

        cpu = time.time() - cpu
        logger.info(f'{len(data)} records processed for decimal points in {cpu:.0f} seconds')

    else:
        logger.info('[Decimal point has been specified for this data, no decimal point conversion]')

    # if

    # rename data
    if renames is not None and supplier in renames.keys():
        cpu = time.time()
        cols = renames[supplier]
        seq = 0

        if cols is not None: # no renames
            logger.info('')
            logger.info('[Renaming data in columns]')

            # iterate over columns to rename
            rename_cols = list(cols.keys())
            for col in rename_cols:
                # it exists: rename data
                if col in data.columns:
                    if schema.loc[col, 'leverancier_kolomnaam'] != dc.VAL_DIDO_GEN:
                        rename_values = cols[col]
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

    return

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
    renames = dc.get_par(delivery_config, 'RENAME_DATA', None)
    strip_space = dc.get_par(delivery_config, 'STRIP_SPACE', None)
    headers = dc.get_par(delivery_config, 'HEADERS', {})

    origin = dc.get_par(cargo, 'origin', {'input': '<file>'})
    if origin['input'] == '<table>':
        logger.info(f'Input for supplier {leverancier_id} is is a table: {origin["table_name"]}')
        logger.warning('!!! There are no checks on imported tables')

        return

    elif origin['input'] == '<api>':
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

        # check for header file
        if 'header_file' in cargo.keys():
            result = load_separate_header_file(
                supplier_config = cargo,
                supplier = leverancier_id,
                schema = leverancier_schema,
                encoding = encoding,
            )
        # if

        data = load_data(
            supplier_config = cargo,
            supplier = leverancier_id,
            schema = leverancier_schema,
            headers = headers,
            sample_size = sample_size,
            server_config = db_servers['ODL_SERVER_CONFIG'],
            encoding = encoding,
        )
        # create a deepcopy of renames as it will be modified inside process_data
        rename_copy = copy.deepcopy(renames)
        if data is not None:
            data = process_data(
                data = data,
                supplier = leverancier_id,
                schema = leverancier_schema,
                meta = leverancier_metadata,
                renames = rename_copy,
                real_types = real_types,
                strip_space = strip_space,
            )

            save_data(
                data = data,
                data_dirs = cargo['data_description'],
                supplier = leverancier_id,
                working_directory = work_dir,
            )

            data = None
            gc.collect()
            dc.report_ram('End of prepare_one_delivery')

        else:
            ok = False
            logger.warning(f'!!! No datafile processed for {leverancier_id}')

        # if

    # if

    return

### prepare_one_delivery ###


def dido_data_prep(header: str):
    cpu = time.time()

    dc.display_dido_header(header)

    # read commandline parameters
    appname, args = dc.read_cli()

    # read the configuration file
    config_dict = dc.read_config(args.project)
    delivery_config = dc.read_delivery_config(config_dict['ROOT_DIR'], args.delivery)
    # x = delivery_config['RENAME_DATA']['pdirekt']['pds_datum_ambtsjubileum']
    # print(x)

    # get project environment
    project_name = config_dict['PROJECT_NAME']
    root_dir = config_dict['ROOT_DIR']
    work_dir = config_dict['WORK_DIR']
    leveranciers = config_dict['SUPPLIERS']
    columns_to_write = config_dict['COLUMNS']
    table_desc = config_dict['TABLES']
    report_periods = config_dict['REPORT_PERIODS']

    renames = dc.get_par(delivery_config, 'RENAME_DATA', None)
    strip_space = dc.get_par(delivery_config, 'STRIP_SPACE', None)
    headers = dc.get_par(delivery_config, 'HEADERS', {})
    overwrite = dc.get_par(delivery_config, 'ENFORCE_PREP_IF_TABLE_EXISTS', False)

    # get the database server definitions
    db_servers = config_dict['SERVER_CONFIGS']
    odl_server_config = db_servers['ODL_SERVER_CONFIG']
    data_server_config = db_servers['DATA_SERVER_CONFIG']
    foreign_server_config = db_servers['FOREIGN_SERVER_CONFIG']

    sql_filename = os.path.join(work_dir, 'sql', 'create-tables.sql')
    doc_filename = os.path.join(work_dir, 'docs', 'create-docs.md')

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
        logger.info('')
        logger.info(f'=== {leverancier_id} ===')
        logger.info('')

        # get all cargo from the delivery_dict
        cargo_dict = dc.get_cargo(delivery_config, leverancier_id)

        # process all deliveries
        for cargo_name in cargo_dict.keys():
            logger.info('')
            print_name = f'--- Delivery {cargo_name} ---'
            logger.info(len(print_name) * '-')
            logger.info(print_name)
            logger.info(len(print_name) * '-')
            logger.info('')

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
            if dc.delivery_exists(cargo, leverancier_id, project_name, db_servers):
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

            prepare_one_delivery(cargo, leverancier_id, project_name, real_types)

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
    log_file = os.path.join(args.project, 'logs/' + cli['name'] + '.log')
    logger = dc.create_log(log_file, level = 'DEBUG')

    # go
    dido_data_prep('Header Check and Data Mover')
