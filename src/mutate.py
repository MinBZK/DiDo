import os
import sys
import math
import time
import numpy as np
import pandas as pd
import psycopg2
import datetime

from os.path import join, splitext, dirname, basename
from logging import DEBUG, INFO
from datetime import timedelta
from sqlalchemy import create_engine

# from common import create_log, read_env, get_headers_and_types
# from common import read_schema_file
from simple_table import sql_select


def get_mutation_dir_names(mutation_dir: str, key_string: str) -> list:
    # check for mutations
    # get all files from mutation directory
    entries = os.listdir(mutation_dir)
    mutations = []
    for dirname in entries:
        # if an entry is a directpry name
        dirpath = os.path.join(mutation_dir, dirname)
        if os.path.isdir(dirpath):
            global_test = dirname[:len(key_string)]

            # if the filename contains the fingerlogger.info for a mutation file
            if global_test == key_string:
                #dirname = os.path.join()
                mutations.append(dirpath) # add to mutation file list

            # if
        # if
    # for

    if len(mutations) <= 0:
        raise ValueError('No mutations to process')

    # read first mutation
    # sort the list so that the file names with a lower date come first
    mutations.sort()

    return mutations

### get_mutation_dir_names ###


def get_mutation_file_names(mutation_dir: str, key_string: str) -> list:
    # check for mutations
    # get all files from mutation directory
    entries = os.listdir(mutation_dir)
    mutations = []
    for filename in entries:
        # if an entry is a filename (not a directory or link)
        if os.path.isfile(os.path.join(mutation_dir, filename)):
            global_test = filename[:len(key_string)]

            # if the filename contains the fingerlogger.info for a mutation file
            if global_test == key_string:
                mutations.append(filename) # add to mutation file list

            # if
        # if
    # for

    if len(mutations) <= 0:
        raise ValueError('No mutations to process')

    # read first mutation
    # sort the list so that the file names with a lower date come first
    mutations.sort()

    return mutations

### get_mutation_file_names ###


def read_postcodes(data_types: dict) -> pd.DataFrame:
    cpu = time.time()
    # fetch the table to operate upon
    if not global_test:
        postcodes = sql_select(env['TABLE_NAME'],
                            columns = 'id, perceelid',
                            host = env['POSTGRES_HOST'],
                            db = env['POSTGRES_DB'],
                            schema = env['SCHEMA_NAME'],
                            port = env['POSTGRES_PORT'],
                            username = env['POSTGRES_USER'],
                            password = env['POSTGRES_PASSWORD'])

        postcodes.to_csv('test-bag-light.csv',
                         sep = ';', index = False)

    else:
        # temporary measure to speed up reading the test table
        parse_dates = ['startdatum', 'einddatum']
        postcodes = pd.read_csv('/home/arnold/opslag/data/test-bag-light.csv',
                                sep = ';',
                                dtype = data_types,
                                parse_dates = parse_dates)
        try:
            postcodes = postcodes.drop(columns = 'Unnamed: 0')
            postcodes = postcodes.drop(columns = 'Unnamed: 0.1')
            postcodes = postcodes.drop(columns = 'Unnamed: 0.2')

        except:
            pass

        # try..except

    # if

    cpu = time.time() - cpu
    logger.info(postcodes.dtypes)
    logger.info(f'{len(postcodes)} postcodes. CPU = {cpu:.0f} seconds')

    return postcodes

### read_postcodes ###


def read_mutation_file(filename: str,
                       schema: pd.DataFrame,
                       header_names: list,
                       data_types: dict
                      ) -> pd.DataFrame:
    cpu = time.time()

    # Create header names for the mutation table
    headers = header_names
    headers.remove('startdatum')
    headers.remove('einddatum')
    headers.insert(0, 'action')
    logger.info(headers)

    # create data types for the mutation table
    types = data_types
    del types['startdatum']
    del types['einddatum']
    types['action'] = object

    # read mutations
    mutations = pd.read_csv(filename,
                            sep = ',',
                            names = headers,
                            dtype = types,
                            quotechar = '"',
                            encoding = 'UTF-8'
                           )
    cpu = time.time() - cpu

    logger.info(f'{len(mutations)} mutaties. CPU = {cpu:.0f} sekonden')
    logger.info(f'{len(schema)} schema kolommen')
    logger.info(f'{len(mutations.columns)} mutatie kolommen')
    logger.info(mutations.dtypes)

    return mutations

### read_mutation_file ###


def record_delete(perceel_ids: set,
                  schema: str,
                  table: str,
                  id: str,
                  end_date: datetime,
                  create = True) -> tuple[str, int]:

    """ Creates an SQL DELETE statement

    Args:
        perceel_ids (set): perceel_ids to check
        schema (str): name of the schema
        table (str): name of the table
        id (str): perceelid of record to be deleted
        end_date (datetime): end date to set record top
        create (bool, optional): when False id may not exist. Defaults to True.

    Returns:
        tuple[str, int]:
            action to be taken
            number of errors detected creating the statement
    """


    errors: int = 0
    message: str = ''
    action: str = ''
    if create:
        action = '\n-- Delete\n'

        # when perceel_ids is None, there is no check on perceelid presence
        if perceel_ids is not None:
            # check on occurrence of perceelid
            if id not in perceel_ids:
                message = f'\n-- Delete error: {id};'
                logger.debug(message)
                errors = 1
            # if
        # if
    # if

    action += f"UPDATE {schema}.{table} " \
              f"SET einddatum = '" + end_date.strftime("%Y-%m-%d") + "'" \
              f"WHERE perceelid = '{id}' " \
              f"AND einddatum = '9999-12-31'; "\
              f"{message}\n"

    return action, errors

### record_delete ###


def record_insert(perceel_ids: set,
                  schema: str,
                  table: str,
                  id: str,
                  start_date: datetime,
                  row: pd.Series,
                  data_types: dict,
                  create: bool = True) -> tuple[str, int]:

    """ Creates an SQL INSERT statement

    Args:
        perceel_ids (set): set of existing perceel_ids
        schema (str): name of the schema
        table (str): name of the table
        id (str): id of record to insert
        start_date (datetime): date sice the inserted record is valid
        row (pd.Series): row with the data to insert for id
        data_types (dict): data types of each element of the row
        create (bool, optional): True when new record is created, False when
            the insert is p[art of an update. Defaults to True.

    Returns:
        action: str: results in an SQL INSERT statement
        errors: int: number of errors detected creating this INSERT
    """

    errors: int = 0
    message: str = ''
    action: str = ''
    if create:
        action: str = '\n-- Insert\n'

        # check on perceelid presence when perceel_ids is not None
        if perceel_ids is not None:

            # ensure that perceelid does not exist when create is True
            if id in perceel_ids:
                message = f'\n-- Insert error: {id}'
                logger.debug(message)
                errors = 1
            # if
        # if
    # if

    cols = 'startdatum, einddatum, '
    vals = f"'{start_date.strftime('%Y-%m-%d')}', '9999-12-31', "
    for idx in row.index[1:]:
        if data_types[idx] == 'object':
            value = str(row[idx])

            # when 'nan' or empty string replace by NULL
            if value == "'nan'" or value == "nan" or len(value) == 0:
                value = "NULL"

            else:
                # when string replace all single quotes by double single quotes
                # for acceptance by SQL
                value = value.replace("'", "''")

                # surround by single quotes for SQL
                value = "'" + value + "'"

            # if

        else:
            # replace nan by NULL for acceptance by SQL
            if math.isnan(row[idx]):
                value = 'NULL'
            else:
                value = str(row[idx])
            # if
        # if

        cols += idx + ', '
        vals += value + ', '
    # for

    # remove comman and space at end
    cols = cols[:-2]
    vals = vals[:-2]

    action += f'INSERT INTO {schema}.{table}' \
              f'({cols})\nVALUES ({vals});' \
              f'{message}\n'

    return action, errors

### record_insert ###


def record_update(postcodes: pd.DataFrame,
                  schema: str,
                  table: str,
                  id: str,
                  end_date: datetime,
                  start_date: datetime,
                  row: pd.Series,
                  data_types: list,
                 ) -> tuple[str, int]:

    actions: str = '\n-- Update\n'
    errors: int = 0

    # when perceel_ids is None, there is no check on perceelid presence
    if perceel_ids is not None:

        # update id must exist in perceel_ids
        if id not in perceel_ids:
            message = f'\n-- Update error: {id};'
            logger.debug(message)
            errors = 1
        # if
    # if

    action1, _ = record_delete(postcodes, schema, table, id, end_date,
                            create = False)
    action2, _ = record_insert(postcodes, schema, table, id, start_date,
                            row, data_types, create = False)

    actions = actions + action1 + action2

    if errors > 0:
        actions += f'-- Update error: {id}\n'

    return actions, errors

### record_delete ###


def record_downdate(id: str) -> str:
    # inversion of record_update, for later implementation

    return '\n-- Downdate\n', 0

### record_downdate ###


def generate_start_end_dates(method: str, base: str, periode: str):
    """ generates start and end date of current mutations

    Args:
        base (str): filename without extension, format: mutation_ddmmyyyy-ddmmyyyy

    Returns:
        datetine, datetime: start resp. end date
    """

    if method == 'filename':
        parts: list = base.split('_')
        dates: list = parts[-1].split('-')

        start_date = datetime.datetime.strptime(dates[0], '%Y%m%d').date()
        end_date = datetime.datetime.strptime(dates[1], '%Y%m%d').date()
        new_date: datetime = datetime.date(end_date.year, end_date.month, 1)
        if new_date >= start_date and new_date <= end_date:
            print('ok')
        end_date: datetime = new_date + timedelta(days = -1) # datetime.datetime(year, end_mo + 1, 1)

    elif method =='period':
        items = periode.split('-')
        year = items[0]
        period_type = items[1][0]
        period_index = int(items[1:])

        if period_type == 'M':
            date_str = f'{year}-{period_index}-01 00:00:00'
            new_date = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
            end_date = new_date + timedelta(seconds = -1)

    return new_date, end_date

### generate_start_end_dates ###


def update_meta_sql(meta_schema: pd.DataFrame,
                    meta_name: str,
                    mutation_filename: str) -> str:

    # create data type for each column
    insert_cols: str = ''
    insert_vals: str = ''

    for idx, row in meta_schema.iterrows():
        if row['kolomnaam'] not in ['*', 'id']:
            kolom = row['kolomnaam']
            insert_cols += kolom + ', '

            if kolom == 'proces':
                value = 'mutation'

            elif kolom == 'filenaam':
                value = mutation_filename

            elif kolom == 'verwerkingsdatum':
                value = 'now()'

            else:
                value = row['initial'].replace("'", "")

            # if

            insert_vals += "'" + value + "', "

        # if

    # for

    # remove last comma and newline
    insert_cols = insert_cols[:-2]
    insert_vals = insert_vals[:-2]

    # create a table definition and instruction to read starttabel.csv
    tbd: str = f"INSERT INTO {schema_name}.{meta_name}({insert_cols})\n"
    tbd += f"VALUES ({insert_vals});\n\n"

    logger.debug(tbd)

    return tbd

### create_meta_sql ###


def mutate(filename: str,
           schema: str,
           table: str,
           perceel_ids: set,
           start_date: datetime,
           end_date: datetime,
           mutaties: pd.DataFrame,
           data_types: dict) -> str:

    '''
    # determine start and end date
    base: str = os.path.splitext(os.path.basename(filename))[0]
    parts: list = base.split('_')
    months: list = parts[-1].split('-')
    year: int = 2000 + int(parts[-2])
    start_mo = int(months[1])
    start_date: datetime = datetime.date(year, start_mo, 1)
    end_date: datetime = start_date + timedelta(days = -1) # datetime.datetime(year, end_mo + 1, 1)
    '''
    # initialize statistics
    n_inserts: int = 0
    n_deletes: int = 0
    n_updates: int = 0
    n_downdates: int = 0

    insert_errors: int = 0
    delete_errors: int = 0
    update_errors: int = 0
    cpu: float = time.time()

    # loop over mutations
    actions = 'BEGIN;\n'
    logger.info(f'Processing {len(mutaties)} mutations')
    for idx, row in mutaties.iterrows():
        insert_error: int = 0
        delete_error: int = 0
        update_error: int = 0

        # fetch id
        id = row['perceelid']

        # what to do?
        action = row.iloc[0]

        if action == 'insert':
            n_inserts += 1
            action, insert_error = record_insert(perceel_ids, schema, table, id,
                                     start_date, row, data_types)

        elif action == 'delete':
            n_deletes += 1
            action, delete_error = record_delete(perceel_ids, schema, table, id, end_date)

        elif action == 'update':
            n_updates += 1
            action, update_error = record_update(perceel_ids, schema, table, id,
                                     end_date, start_date, row, data_types)

        elif action == 'downdate':
            action, _ = record_downdate(id)
            n_downdates += 1

        # if

        actions += action
        insert_errors += insert_error
        delete_errors += delete_error
        update_errors += update_error

    # for

    actions += '\nCOMMIT;\n\n'
    actions += f'REINDEX TABLE {schema_name}.{table_name};\n\n'

    cpu = time.time() - cpu

    logger.info('')
    logger.info(f'Action         N    Errors')
    logger.info(f'insert{n_inserts:10}{insert_errors:10} ')
    logger.info(f'delete{n_deletes:10}{delete_errors:10} ')
    logger.info(f'update{n_updates:10}{update_errors:10} ')
    logger.info(f'downdate{n_downdates:8}')

    logger.info('')
    logger.info(f'{len(mutaties)} mutations processed in {cpu:.0f} sekonden')

    return actions

### mutate ###


def check_consistencies(mutations: pd.DataFrame, perceel_ids: set):
    id_inserts = set(mutations[mutations['action'] == 'insert']['perceelid'])
    id_deletes = set(mutations[mutations['action'] == 'delete']['perceelid'])
    id_updates = set(mutations[mutations['action'] == 'update']['perceelid'])
    id_downdates = set(mutations[mutations['action'] == 'downdate']['perceelid'])

    logger.info('')
    logger.info(f'{len(id_inserts)} inserts')
    logger.info(f'{len(id_deletes)} deletes')
    logger.info(f'{len(id_updates)} updates')
    logger.info(f'{len(id_downdates)} downdates')

    doorsnede_inserts = id_inserts - perceel_ids
    doorsnede_deletes = id_deletes - perceel_ids
    doorsnede_updates = id_updates - perceel_ids
    doorsnede_downdates = id_downdates - perceel_ids

    logger.info('')
    logger.info(f'{len(doorsnede_inserts)} doorsnede inserts')
    logger.info(f'{len(doorsnede_deletes)} doorsnede deletes')
    logger.info(f'{len(doorsnede_updates)} doorsnede updates')
    logger.info(f'{len(doorsnede_downdates)} doorsnede downdates')

    return

### check_consistencies ###

"""
print ("Argument List:", str(sys.argv))
# when run stand-alone, then there is just one argument: the path to the script


if len(sys.argv) == 1:
    src_name = sys.argv[0]
    root_dir = dirname(dirname(src_name))
    mutation_csv = None

# when called from init script there are 3 arguments
# 1. path to script, 2. current source directory, 3. path to postcodetabel
elif len(sys.argv) == 3:
    src_name = sys.argv[0]
    root_dir = dirname(sys.argv[1])
    mutation_csv = os.path.basename(sys.argv[2])

else:
    print(sys.argv)
    raise ValueError('*** wrong number of arguments: either none or two')

# if

mutation_base = splitext(mutation_csv)[0]

print('===>', src_name, '<===')
print('===>', root_dir, '<===')
print('===>', mutation_csv, '<===')
print('===>', mutation_base, '<===')

# Read environment file and initialize variables
env = read_env('config/.env')
print(env)

schema_name: str = env['POSTGRES_SCHEMA']
table_name: str = env['TABLE_NAME']
table_name_history: str = env['TABLE_NAME_HISTORY']
work_dir: str = env['WORK_DIR']
meta_name: str = join(work_dir, join('docs/schema', env['SCHEMA_POSTCODES']))
schema_history: str = join(work_dir, join('docs/schema', meta_name))

#mutation_csv: str = env['POSTCODE_FILE']
mutation_table_name: str = meta_name # env['META_NAME']

# create csv file name containing the mutations
mutation_file: str = os.path.join(work_dir, os.path.join('todo', mutation_csv))

# name of sql transact ion file
transaction_file: str = os.path.join(work_dir,
                            os.path.join('sql', mutation_base)) + '.sql'

# name of log file
log_file: str = os.path.join(work_dir,
                             os.path.join('logs', mutation_base)) + '.log'

# create a logger and log file
logger = create_log(log_file, DEBUG)

# get start and end date
start_date, end_date = generate_start_end_dates(mutation_base)

# read schema file with data definitions of mutation file
logger.info('')
logger.info('=== Schema ===')
filename = os.path.join(work_dir, os.path.join('docs', meta_name))
logger.info(f'Reading from: {filename}')
schema = read_schema_file(filename)
header_names, data_types = get_headers_and_types(schema)

logger.info('')
logger.info('=== Update meta information ===')
meta_file: str = meta_name.replace('pcdata', 'meta')
logger.info(f'Reading from: {filename}')
filename = os.path.join(work_dir, os.path.join('docs', meta_file))
meta_schema = read_schema_file(filename)
meta_sql = update_meta_sql(meta_schema, mutation_table_name, mutation_file)

# read mutations file
logger.info('')
logger.info('=== Mutaties ===')
mutations = read_mutation_file(mutation_file, schema, header_names, data_types)

# read perceelids for consistency check
logger.info('')
logger.info("=== Perceelid's ===")
postcodes = read_postcodes({'id': 'int', 'perceelid': 'str'})
perceel_ids = set(postcodes['perceelid'])

# create transactions SQL string
logger.info('')
logger.info(f'=== Transactions for {mutation_base} ===')
transactions = mutate(transaction_file,
                      schema_name,
                      table_name,
                      perceel_ids,
                      start_date,
                      end_date,
                      mutations,
                      data_types
                     )

check_consistencies(mutations, perceel_ids)

# write SQL to file. To create the table and load the initial
# postcodes execute:
# psql -h <host> -p <port> -U <user> -f create-postcode-tabel.sql <database>

logger.info('')

with open(transaction_file, 'w') as outfile:
    outfile.write(meta_sql)
    outfile.write(transactions)

logger.info(f'\nTransactions written to {transaction_file}')

logger.info('[Ready]')
"""