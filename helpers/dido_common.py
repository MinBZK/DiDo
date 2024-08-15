"""
dido_common.py is a library with common routines for DiDo.
"""
# from logging.config import dictConfig # required to import logging
# import random

import os
import re
import sys
import logging
import argparse
import sqlalchemy

import pandas as pd

from datetime import datetime
from logging.config import dictConfig
from os.path import join, splitext, dirname, basename, exists
import psutil
import s3_helper

import yaml

import simple_table as st

logger = logging.getLogger()

# define constants
# set defaults
SCHEMA_TEMPLATE = 'bronbestand_attribuutmeta_description'
META_TEMPLATE   = 'bronbestand_bestandmeta_description'
EXTRA_TEMPLATE  = 'bronbestand_attribuutextra_description'

# TAG_TABLE refer to TABLES indices in the confif.yaml file
TAG_TABLE_SCHEMA   = 'schema'
TAG_TABLE_META     = 'meta'
TAG_TABLE_EXTRA    = 'extra'
TAG_TABLE_DELIVERY = 'levering'
TAG_TABLE_QUALITY  = 'datakwaliteit'

# Tags refer to dictionary indices of each table
TAG_TABLES = 'tables'
TAG_PREFIX = 'prefix'
TAG_SUFFIX = 'suffix'
TAG_SCHEMA = 'schema'
TAG_DATA   = 'data'
TAG_DESC   = 'description'

# Required ODL columns to add to schema
ODL_RECORDNO         = 'bronbestand_recordnummer'
ODL_CODE_BRONBESTAND = 'code_bronbestand'
ODL_LEVERING_FREK    = 'levering_rapportageperiode'
ODL_DELIVERY_DATE    = 'levering_leveringsdatum'
ODL_DATUM_BEGIN      = 'record_datum_begin'
ODL_DATUM_EINDE      = 'record_datum_einde'
ODL_SYSDATUM         = 'sysdatum'

# special column names that require action
COL_CREATED_BY = 'created_by'

# 1,2,3,6,8 controls in code
# betekenis van datakwaliteitcodes
VALUE_OK = 0 # "Valide waarde"
VALUE_NOT_IN_LIST = 1  # "Domein - Waarde niet in lijst"
VALUE_MANDATORY_NOT_SPECIFIED = 2 #"Constraint - Geen waarde, wel verplicht"
VALUE_NOT_BETWEEN_MINMAX = 3 #"Domein - Waarde ligt niet tussen minimum en maximum"
VALUE_OUT_OF_REACH = 4  #"Waarde buiten bereik"
VALUE_IMPROBABLE = 5 #"Waarde niet waarschijnlijk"
VALUE_WRONG_DATATYPE = 6 #"Datatype - Waarde geen juist datatype"
VALUE_HAS_WRONG_FORMAT = 7 #"Waarde geen juist formaat"
VALUE_NOT_CONFORM_RE = 8 #"Domein - Waarde voldoet niet aan reguliere expressie"

# Allowed column names
ALLOWED_COLUMN_NAMES = ['kolomnaam', 'datatype', 'leverancier_kolomnaam',
                        'leverancier_kolomtype', 'gebruiker_info',
                        'beschrijving']
# Directory constants
DIR_SCHEMAS = 'schemas'
DIR_DOCS    = 'docs'
DIR_DONE    = 'done'
DIR_TODO    = 'todo'
DIR_SQL     = 'sql'

VAL_DIDO_GEN = '(dido generated)'

# Version labels
ODL_VERSION_MAJOR  = 'odl_version_major'
ODL_VERSION_MINOR  = 'odl_version_minor'
ODL_VERSION_PATCH  = 'odl_version_patch'
ODL_VERSION_MAJOR_DATE  = 'odl_version_major_date'
ODL_VERSION_MINOR_DATE  = 'odl_version_minor_date'
ODL_VERSION_PATCH_DATE  = 'odl_version_patch_date'
DIDO_VERSION_MAJOR = 'dido_version_major'
DIDO_VERSION_MINOR = 'dido_version_minor'
DIDO_VERSION_PATCH = 'dido_version_patch'
DIDO_VERSION_MAJOR_DATE = 'dido_version_major_date'
DIDO_VERSION_MINOR_DATE = 'dido_version_minor_date'
DIDO_VERSION_PATCH_DATE = 'dido_version_patch_date'

# Miscellaneous
DATE_FORMAT = '%Y-%m-%d'
TIME_FORMAT = '%H:%M:%S'
DATETIME_FORMAT = f'{DATE_FORMAT} {TIME_FORMAT}'

# Get deliveries and count of all deliveries in database
DB_SHOWD = """SELECT {schema}.{table}.levering_rapportageperiode, count(levering_rapportageperiode)
 FROM {schema}.{table}
 GROUP BY {schema}.{table}.levering_rapportageperiode;
 """
# Statitics from database
DB_STATS = """SELECT  count({column}) AS n,
        sum(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END) AS misdat,
        min({column}),
        max({column}),
        avg({column}) AS mean,
        PERCENTILE_CONT(0.5) WITHIN GROUP(ORDER BY {column}) AS median,
        stddev({column}) AS std
FROM {schema}.{table};
"""

DB_FREQS = """SELECT {table}.{column}, 100 * (count(*) / tablestat.total::float) AS percent_total
 FROM {schema}.{table}
 CROSS JOIN (SELECT count(*) AS total FROM {schema}.{table}) AS tablestat
 WHERE {schema}.{table}.{column} IS NOT NULL
 GROUP BY tablestat.total, {schema}.{table}.{column}
 ORDER BY percent_total DESC;
"""


class DiDoError(Exception):
    """ To be raised for DiDo exceptions

    Args:
        Exception (str): Exception to be raised
    """
    def __init__(self, message):
        print('')

        # get info where the exception occurred
        exc_type, exc_obj, exc_tb = sys.exc_info()
        if exc_tb is not None:
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(f'> Exception {exc_type} occurred in line {exc_tb.tb_lineno} of file {fname} <')

        else:
            print(f'> Exception {exc_type} occurred <')

        print('')

        self.message = 'DiDo Fatal Error: ' + message
        logger.critical(self.message)
        super().__init__(self.message)

        # sys.exit()
    ### __init__ ###

### Class: DiDoError ###


def create_log(filename: str, level: int = logging.INFO, reset: bool = True) -> object:
    """ Returns a log configuration

    The log configuration returned displays a normal message at the console
    and an extended version (including level, data, etc) to the log file.

    Get a logger as follows:

    logger = common.create_log(log_file, level = loggin.DEBUG)
    logger.info(f'log_file is {log_file}') # writes a message at level info
    logger.error('System crash, quitting') # writes message at level error

    Args:
        filename -- which need to be logged/monitored
        level -- treshold level only log messages above level

    Returns:
        logger object
    """
    if reset:
        file_mode = 'w'
    else:
        file_mode = 'a'

    logging_configuration = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(module)s: %(message)s'
            },
            'brief': {
                'format': '%(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': logging.INFO,
                'formatter': 'brief',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',  # default stderr
            },
            'file': {
                'level': logging.DEBUG,
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': filename,
                'mode': file_mode,
            },
        },
        'loggers': {
            '': {
                'level': level,
                'handlers': ['console', 'file']
            },
        },
    }

    log = logging.getLogger()
    log.setLevel(logging.DEBUG)

    try:
        dictConfig(logging_configuration)

    except:
        print('*** Error while initializing logger.')
        print(f'*** Either the path to the log file does not exist: {filename}')
        print('*** or the "logs" directory does not exist in that path')

        raise DiDoError('*** Aborting execution')

    # try..except

    return log

### create_log ###


def iso_cet_date(datum: datetime, tz: str = ''):
    result = datum.strftime(f"%Y-%m-%d %H:%M:%S {tz}")

    return result

### iso_cet_date ###


def split_filename(filename: str) -> tuple:
    """ Splits filename into dirpath, filename and .extension

    Args:
        filename (str): filename to split

    Returns:
        directory, filename without extension, .extension (including period)
    """
    filepath_without_extension, extension = os.path.splitext(filename)
    dirpath = os.path.dirname(filepath_without_extension)
    filebase = os.path.basename(filepath_without_extension)

    return dirpath, filebase, extension


def get_files_from_dir(folder: str):
    """ Return all files from directory, ignore directories

    Args:
        folder (str): directory to get files from

    Returns:
        list: list of files found
    """
    # remove space
    folder = folder.strip()

    # folder name should end with slash
    if not folder.endswith('/'):
        folder += '/'

    # when requested from s3 bucket, call s3 helper function
    if folder.startswith('s3://'):
        files = s3_helper.s3_command_ls_return_fullpath(folder=folder)
    else:
        files = [item for item in os.listdir(folder)
                    if os.path.isfile(os.path.join(folder, item))]

    return files

### get_files_from_dir ###


def change_column_name(col_name: str, empty: str = 'kolom_') -> str:
    """align to snake_case; only alfanum and underscores, multiple underscores reduced to one

    Args:
        col_name -- name to be changed
        empty -- prefix for random name

    Returns:
        adjusted columnname
    """
    # remove outer whitespace
    col_name = col_name.strip().lower()

    # non-alfanums to underscore, multiple underscore to one
    col_name = re.sub('[^0-9a-zA-Z_]+', '_', col_name)

    # remove _ at beginning and end
    col_name = col_name.strip('_')

    # return if column starts with letter
    for i, letter in enumerate(col_name):
        if letter.isalpha():
            return col_name[i:]

    # if no letter return random name
    return f'{empty}{random.randint(1000, 9999)}'


def get_headers_and_types(schema: pd.DataFrame) -> tuple:
    """ of provided df

    Args:
        schema(pd.DataFrame): to get headers and types from

    Returns:
        headers in list, datatypes in dict {header:datatype}
    """
    headers = []
    datatypes = {}

    for row in schema.itertuples():
        header = row.kolomnaam
        if header not in ['*', 'type']:
            dtype = row.pandastype
            headers.append(header)
            datatypes[header] = dtype

    return headers, datatypes

### get_headers_and_types ###


def read_schema_file(filename: str) -> pd.DataFrame:
    """ mutation schema file

    Args:
        filename -- csv schema file

    Returns:
        contents
    """
    schema = pd.read_csv(filename, sep = ';', quotechar = '"', keep_default_na = False, encoding = 'UTF-8')

    # Postgres surrounds text by "'"; use double single quotes to have them accepted
    schema['beschrijving'] = schema['beschrijving'].str.replace("'", "''")

    # Two spaces at line end ensures newline in markdown
    schema['beschrijving'] = schema['beschrijving'].str.replace("\n", "  \n")

    return schema

### read_schema_file ###


def report_ram(message: str):
    logger = logging.getLogger()
    mem = psutil.virtual_memory()

    logger.info('')
    logger.info(message)
    logger.info(f'   Total memory:    {mem.total:,}')
    logger.info(f'   Memory used:     {mem.used:,}')
    logger.info(f'   Memory available {mem.available:,}')

    return

### report_ram ###


def get_par(config: dict, key: str, default = None):
    """ Gets the value of a key from a dictionary or DataFrame index

    when config is a dictionary its key is fetched, when it is a dataframe
    the key is looked up from the index.

    In both cases applies that when a key is not found the default is returned.

    Args:
        config (dict): dictuionary or DataFrame to fetch the value from
        key (str): value to find in dictionary
        default (type, optional): default when value is not in dict.
            Defaults to None.

    Returns:
        config[key] when key is present, else default when not None

    Raises:
        DiDoError when key not found and default is not None
    """
    if isinstance(config, pd.DataFrame):
        # config is DataFrame, look for key in index
        if key in config.index:
            return str(config.loc[key])
        else:
            return default
        # if

    else:
        # dictionary, return config[key] when present
        if key in config:
            return config[key]
        else:
            return default
        # if

    # if
### get_par ###


def get_par_par(config: dict, key_1: str, key_2: str, default = None):
    """ Gets the value of a key from a dictionary from which the result is a dictionary

   Key_1 points to a dictionary inside the dictionary, key_2 points to a
   value inside that dictionary. When either is not found default is returned

    Args:
        config (dict): dictionary or DataFrame to fetch the value from
        key_1 (str): value to find in config
        key_2 (str): value to find in config[key_1]
        default (_type_, optional): default when value is not in dict. Defaults to None.
    """

    # dictionary, return config[key] when present
    if key_1 in config:
        sub_config = config[key_1]

        if key_2 in sub_config:
            return sub_config[key_2]

        else:
            return default

    else:
        return default

### get_par ###


def read_cli():
    """ Read command line arguments

    Returns:
        str: path to the application
        object: argparse arguments
    """
    pad, fn, ext = split_filename(sys.argv[0])
    app_path = {'path': pad, 'name': fn, 'ext': ext}

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c", "--compare",
                           help="Compare action: 'dump' or 'compare'",
                           choices=['compare', 'dump'],
                           default='compare', const='compare', nargs='?')
    argParser.add_argument("-d", "--delivery",
                           help="Name of delivery file in root/data directory")
    argParser.add_argument("--date", help="Date to take snapshot of database ")
    argParser.add_argument("-p", "--project", help="Path to project directory")
    argParser.add_argument("-r", "--reset",
                           help="Empties the logfile before writing it",
                           action='store_const', const='reset')
    argParser.add_argument("-s", "--supplier", help="Name of supplier")
    argParser.add_argument("-t", "--target",
                           help="File name to read target data from")
    argParser.add_argument("-v", "--view",
                           help="View database table in dido_compare",
                           action='store_const', const='view' )
    argParser.add_argument("--Yes",  help="Answer Yes to all questions",
                           action='store_const', const='Ja')

    args = argParser.parse_args()

    if args.project is None:
        print('\n*** No -p <project-dir> specified at program call\n')
        sys.exit(1)

    return  app_path, args

### read_cli ###


def load_parameters() -> dict:
    """ Laadt parameters uit config file

    Returns:
        dict: dictionary with parameters from file
    """
    with open('config/dido.yaml', encoding = 'utf8', mode = "r") as infile:
        parameters = yaml.safe_load(infile)

    return parameters

### load_parameters ###


def create_data_types():
    parameters = load_parameters()
    basis_types = parameters['BASIC_TYPES']
    sub_types = parameters['SUB_TYPES']
    data_types = {}

    # enumerate over all sub_types
    for key in sub_types.keys():
        type_list = sub_types[key]

        # enumerate over each sub_type in the argument list
        for sub_type in type_list:
            # Assign type definition from basic type to subtype
            data_types[sub_type] = basis_types[key]

        # for
    # for

    logger.debug(f'Data types:\n {data_types}')

    return data_types, sub_types

### create_data_types ###


def load_sql() -> str:
    """ read the SQL support functions if available

    Returns:
        str: SQL statements as a string
    """
    sql = ''
    with open('config/dido_functions.sql', encoding = 'utf8', mode = "r") as infile:
        sql = infile.read().strip()

    return sql

### load_sql ###


def load_pgpass(host: str, port: str, db: str, user: str = ''):
    """ Searches for a password in .pgpass

    Function implements algorithm as described in:
    https://www.postgresql.org/docs/current/libpq-pgpass.html
    An field in an entry may be * in which case no selection is made for that field.

    Args:
        host (str, optional): hostname or IP.
        port (str, optional): port.
        db (str, optional): database name.
        user (str, optional): user name . Defaults to ''.

    Returns:
        None: if .pgpass is not found in the user root
        (None, None): if no entries match the criteria
        (str, str): (user name, password) of the first matching entry

    Example:
        credentials = load_pgpass(host = '10.10.12.12', db = '*')
        if credentials is None:
            print('No .pgpass file found')

        else:
            (user, pw) = credentials
            if user is None or pw is None:
                print('No matching entry in .pgpass found')

            else: # use and pw contain selected user and password in .pgpass
                print(user) # never print passwords
    """

    # check if a .pgpass exists, if not: return None
    pgpass_filename = os.path.expanduser('~/.pgpass')
    if os.path.isfile(pgpass_filename):
        pgpass = pd.read_csv(
            pgpass_filename,
            sep = ':',
            dtype = str,
            keep_default_na = False,
            header = None,
            index_col = None,
            engine = 'python',
            encoding = 'utf8',
        )

    else:
        return None

    # if host is specified select for host
    pgpass_host = pgpass.iloc[:, 0]
    pgpass = pgpass.loc[(pgpass_host == host) | (pgpass_host == '*')]
    logger.debug(f'Host - host: {host}, port: {port}, database: {db}, user: {user}: {len(pgpass)}')

    # if port is specified select for port
    port = str(port) # force the port to be an integer
    pgpass_port = pgpass.iloc[:, 1]
    pgpass = pgpass.loc[(pgpass_port == port) | (pgpass_port == '*')]
    logger.debug(f'Port - host: {host}, port: {port} and database: {db} user: {user}: {len(pgpass)}')

    # if database is specified select for database
    pgpass_db = pgpass.iloc[:, 2]
    pgpass = pgpass.loc[(pgpass_db == db) | (pgpass_db == '*')]
    logger.debug(f'Db - host: {host}, port: {port} and database: {db} user: {user}: {len(pgpass)}')

    # if user is specified select for user
    if len(user) > 0:
        pgpass_user = pgpass.iloc[:, 3]
        pgpass = pgpass.loc[(pgpass_user == user) | (pgpass_user == '*')]
        logger.debug(f'User - host: {host}, port: {port} and database: {db} user: {user}: {len(pgpass)}')

    # when no match is found, return (None, None)
    if len(pgpass) < 1:
        logger.warning(f'No candidate left in .pgpass after applying host: {host}, '
                        f'port: {port}, db: {db} and user: {user}')
        return (None, None)

    logger.debug(f'{len(pgpass)} candidates left in .pgpass after applying host: {host}, '
                 f'port: {port}, db: {db} and user: {user}. First picked')

    return (pgpass.iloc[0, 3], pgpass.iloc[0, 4])

### load_pgpass ###


def read_config(project_dir: str) -> dict:
    """ Reads config and environment file and initializes variables.

    The config file is read from the <project_dir>/config directory, just like
    the .env file. The config file is read as a dictionary into the config variable.
    The servers declared in the SERVER_CONFIGS section are updated with the
    credentials of .env.

    Next, program wide parameters are read from the ODL database: odl_parameters
    and odl_rapportageperiodes. These are assigned to the config dictionary.

    Args:
        project_dir (str): directory path pointing to the root of the
            project directory. Subdirectories are at least: config, root and work.

    Returns:
        dict: the config dictionary enriched with additional information
    """
    # read the configfile
    configfile = os.path.join(project_dir, 'config', 'config.yaml')
    logger.info(f'[Bootstrap: {configfile}]')
    with open(configfile, encoding = 'utf8', mode = "r") as infile:
        config = yaml.safe_load(infile)

    config['PROJECT_DIR'] = project_dir

    sql = load_sql()

    item_names = ['ROOT_DIR', 'WORK_DIR', 'HOST', 'SERVER_CONFIGS']
    errors = False
    for item in item_names:
        if item not in config.keys():
            errors = True
            logger.critical(f'{item} not specified in config.yaml')

    if errors:
        raise DiDoError('Er ontbreken elementen in config.yaml, zie hierboven')

    config['ROOT_DIR'] = os.path.join(project_dir, config['ROOT_DIR'])
    config['WORK_DIR'] = os.path.join(project_dir, config['WORK_DIR'])
    config['SQL_SUPPORT'] = sql

    # load dido.yaml
    parameters = load_parameters()

    # get the server variable
    servers = config['SERVER_CONFIGS'].keys()

    # check if the server to use exists
    host = config['HOST'].lower().strip()
    if host not in parameters['SERVERS'].keys():
        for server in parameters.keys():
            logger.info(f' - {server}')

        raise DiDoError(f'*** Server to use in config file {host} is not among the allowed servers')

    # find IP belonging to host
    host_ip = parameters['SERVERS'][host]

    logger.debug(config['SERVER_CONFIGS'])

    # Credentials
    env = load_credentials(project_dir)
    if 'POSTGRES_USER' in env.keys():
        user = env['POSTGRES_USER']
    else:
        user = ''

    # assign env credentials and server to the server_configs in the config file
    for server in config['SERVER_CONFIGS']:
        config['SERVER_CONFIGS'][server]['POSTGRES_HOST'] = host_ip
        port = config['SERVER_CONFIGS'][server]['POSTGRES_PORT']
        db = config['SERVER_CONFIGS'][server]['POSTGRES_DB']

        # check for user name and pw in .pgpass resp. .env
        creds = load_pgpass(host = host_ip, port = port, db = db, user = user)

        # if .pgpass not found or entries in .pgpass not found
        if creds is None or creds == (None, None):
            try:
                config['SERVER_CONFIGS'][server]['POSTGRES_USER'] = user
                config['SERVER_CONFIGS'][server]['POSTGRES_PW'] = env['POSTGRES_PASSWORD']
            except:
                raise DiDoError('No valid credentials found in .pgpass or config/.env')

        # .pgpass found with correct entries
        else:
            config['SERVER_CONFIGS'][server]['POSTGRES_USER'] = creds[0]
            config['SERVER_CONFIGS'][server]['POSTGRES_PW'] = creds[1]

        # if
    # for

    # store other settings of .env in config
    for key in env.keys():
        if key not in ['POSTGRES_USER', 'POSTGRES_PASSWORD']:
            config[key] = env[key]

    # fetch rapportage leveringsperiodes from odl
    odl_server = config['SERVER_CONFIGS']['ODL_SERVER_CONFIG']
    #odl_server['POSTGRES_HOST'] = '10.10.12.6'
    rapportage_periodes = load_odl_table('odl_rapportageperiodes_description', odl_server)

    # assign these tot the config file
    config['REPORT_PERIODS'] = rapportage_periodes
    config['PARAMETERS'] = parameters

    return config

### read_config ###


def get_config_file(config_path: str, config_name: str):
    """ Read a .yaml file from the config directory

    Args:
        config_path (str): path to the config directory
        config_name (str): name of the config file
    """
    configfile = os.path.join(config_path, config_name)

    with open(configfile, encoding = 'utf8', mode = "r") as infile:
        config = yaml.safe_load(infile)

    return config

### get_config_file ###


def read_delivery_config(project_path: str,
                         delivery_filename: str,
                        ):
    """Read a delivery.yaml file

    Args:
        project_path (str): path to the project
        delivery_filename (str): filename of the delivery file
            (defaults to delivery.yaml)

    Returns:
        dict: the delivery.yaml file
    """
    delivery_filename = os.path.join(project_path, 'config', delivery_filename)
    try:
        with open(delivery_filename, encoding = 'utf8', mode = "r") as infile:
            delivery = yaml.safe_load(infile)

    except Exception as err:
        logger.critical('*** ' + str(err))
        raise DiDoError(f'*** Delivery file not found: {delivery_filename}')

    return delivery

### read_delivery_config ###


def load_credentials(project_dir: str) -> dict:
    """ Reads the .env file as dict, returns empty dict when no file is found

    Args:
        project_dir (str): project directory of the expected config/.env file

    Returns:
        dict: the .env file as dict or an empty dict when config/.env is not found
    """
    env_filename = '.env' # config['ENV']
    env_filename = os.path.join(project_dir, os.path.join('config', env_filename))

    if os.path.isfile(env_filename):
        with open(env_filename, encoding = 'utf8', mode = "r") as envfile:
            env = yaml.safe_load(envfile)

        logger.info('Credentials read from config/.env')

    else:
        # initialize empty env dictionary
        logger.debug('No credentials file found: config/.env')

        env = {}

    # if

    return env

### load_credentials ###


def compute_periods(period: str, value: int, servers: dict):
    # get levering_rapportage period table from odl
    periods = st.table_to_dataframe(
        table_name = 'odl_rapportageperiodes_description',
        sql_server_config = servers['ODL_SERVER_CONFIG'],
    )

    # split into year-<letter><int> anf check
    try:
        year = period.split('-', 1)[0]
        rest = period.split('-', 1)[1]
        qualifier = rest[0]
        if qualifier not in ['I', 'J']:
            counter = int(rest[1:])

        if len(year) != 4:
            raise DiDoError('*** Year should be between 1000-9999')

        year = int(year)

    except Exception as err:
        logger.error('compute_periods: ' + str(err))
        raise DiDoError('*** Malformed period, should be of form: '
                        'YYYY-<period><int>')

    # try..except

    # get allowed periods and set index to allowed periods
    allowed = periods['leverancier_kolomtype'].tolist()
    periods = periods.set_index('leverancier_kolomtype')
    if qualifier not in allowed:
        raise DiDoError(f'*** Qualifier {period} not allowed: {allowed}')

    # get max value for qualifier
    domain = periods.loc[qualifier, 'domein']
    if qualifier not in ['A', 'I', 'J']:
        max_value = int(domain.split(':')[1])
    else:
        max_value = 0

    # apply operation
    if qualifier == 'J':
        year += value
        counter = ''

    elif qualifier in ['A']:
        counter += value
        if counter < 1:
            raise DiDoError(f'*** Compute_period: qualifier "A": value '
                            f' < 1: {counter}')

    elif qualifier != 'I':
        counter += value
        while counter > max_value:
            year += 1
            counter -= max_value

        while counter < 1:
            year -= 1
            counter += max_value

        if year < 1000 or year > 9999:
            raise DiDoError(f'*** Compute_period: year {year} exceeds bounds '
                             '[1000-9999]')

        # if
    # if

    new_period = f'{year}-{qualifier}{counter}'

    return new_period

### compute_periods ###


def center_text(text: str, width: int):
    delta = width - len(text)
    delta_2 = int(delta / 2)
    front = delta_2
    back = delta - delta_2
    text = '*' + front * ' ' + text + back * ' ' + '*'

    return text

### center_text ###


def display_dido_header(text: str = None, config = None):
    if text is None:
        return

    dido = ' DiDo - Document Data Definition Generator '
    text = f' {text} '
    dido_len = len(dido)
    if len(text) > dido_len:
        dido_len = len(text)

    center_text(dido, dido_len)

    # delta = len(dido) - len(text)
    # delta_2 = int(delta / 2)
    # front = delta_2
    # back = delta - delta_2
    # text = '*' + front * ' ' + text + back * ' ' + '*'
    # dido = '*' + dido + '*'
    asterisks = (dido_len + 2) * '*'
    between = '*' + (dido_len) * ' ' + '*'

    logger.info('')
    logger.info(asterisks)
    logger.info(center_text(' ', dido_len))
    logger.info(center_text(dido, dido_len))
    logger.info(center_text(' ', dido_len))
    logger.info(center_text(text, dido_len))
    logger.info(center_text(' ', dido_len))

    if config is not None:
        major = get_par_par(config, 'PARAMETERS', 'DIDO_VERSION_MAJOR', '')
        minor = get_par_par(config, 'PARAMETERS', 'DIDO_VERSION_MINOR', '')
        patch = get_par_par(config, 'PARAMETERS', 'DIDO_VERSION_PATCH', '')
        version = f' DiDo version {major}.{minor}.{patch} '
        version = center_text(version, dido_len)
        logger.info(version)
        logger.info(center_text(' ', dido_len))

    logger.info(asterisks)
    logger.info('')

    return

### display_dido_header ###


def subheader(text: str, char: str):
    text = 3 * char + ' ' + text + ' ' + 3 * char
    logger.info('')
    logger.info(len(text) * char)
    logger.info(text)
    logger.info(len(text) * char)
    logger.info('')

    return
### subheader ###


def get_limits(config: dict):
    """ Read LIMITS

    Args:
        config (dict): configuration containing the limits

    Returns:
        (tuple): the limited variables
    """
    max_rows = None
    max_errors = None
    data_test_fraction = 0 # limitations['data_test_fraction']

    if 'LIMITS' in config:
        limitations = config['LIMITS']
        max_rows = limitations['max_rows']
        max_errors = limitations['max_errors']

        if max_rows < 1:
            max_rows = None

    return max_rows, data_test_fraction, max_errors

### get_limits ###


def load_odl_table(table_name: str, server_config: dict) -> pd.DataFrame:
    """  Load ODL from PostgreSQL

    Currently only used to load data_kwaliteit_codes. This is stored centrally
    in the techniek/odl database.

    Args:
        table_name (str): name of the postgres table, schema is predefined
        odl_server_config (dict): contains postgres access data of the odl server

    Returns:
        pd.DataFrame: Operationele Data Laag
    """
    result = st.sql_select(
        table_name = table_name,
        columns = '*',
        sql_server_config = server_config,
        verbose = False,
    ).fillna('')

    return result

### load_odl_table ###


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


def get_server(config: dict, index: str, username: str, password: str) -> dict:
    server_config = config[index]
    server_config['POSTGRES_USER'] = username
    server_config['POSTGRES_PW'] = password

    # del env

    return server_config

### get_server ###


def get_table_name(project_name: str, supplier: str, table_info: str, postfix: str):
    table_name = f'{supplier}_{project_name}_{table_info}_{postfix}'

    return table_name

### get_table_name ###


def get_table_names(project_name: str, supplier: str, postfix: str = 'data') -> dict:
    """ Get table names of project for specific supplier.

    Args:
        project_name (str): Name of project
        leverancier (str): Name of leverancier
        postfix (str): either 'data' or 'description'

    Returns:
        dict: dictionary with all data table names
    """
    tables_name: dict = {TAG_TABLE_SCHEMA: get_table_name(project_name, supplier, TAG_TABLE_SCHEMA, postfix),
                         TAG_TABLE_META: get_table_name(project_name, supplier, TAG_TABLE_META, postfix),
                         TAG_TABLE_DELIVERY: get_table_name(project_name, supplier, TAG_TABLE_DELIVERY, postfix),
                         TAG_TABLE_QUALITY: get_table_name(project_name, supplier, TAG_TABLE_QUALITY, postfix),
                        }

    return tables_name

### get_table_names ###


def get_suppliers(config_dict: dict) -> dict:
    """ Returns all suppliers and their products from config.yaml

    Args:
        config_dict (dict): the SUPPLIERS part from config.yaml

    Returns:
        dict: dictionary of suppliers and for each supplier a dictionary
            with the projects
    """
    suppliers_dict = dict()
    for supplier_name, projects in config_dict.items():
        current_supplier = dict()
        for project_name, value in projects.items():
            current_supplier[project_name] = dict()

        suppliers_dict[supplier_name] = current_supplier

    # for

    return suppliers_dict

### get_suppliers ###


def add_deliveries_to_suppliers(suppliers_dict: dict, delivery_dict: dict) -> dict:
    """ Adds a dictionary with deliveries to each project in the suppliers_dict

    Args:
        suppliers_dict (dict): dictionary with suppliers and projects
        delivery_dict (dict): the DELIVERIES part of delivery.yaml

    Raises:
        dc.DidoError: when delivery.yaml contains suppliers or projects
            not in config.yaml an exception is raised


    Returns:
        dict: a dictionary with delivery information added to each project
            in suppliers_dict
    """
    errors = False
    # consistency check 1: all delivery suppliers should be known in suppliers_dict
    for delivery_supplier, _ in delivery_dict.items():
        if delivery_supplier not in suppliers_dict.keys():
            errors = True
            logger.error(f'*** Supplier in delivery ({delivery_supplier}) '
                         'is not known among the suppliers in "config.yaml"')

    # consistency check 2: all delivery projects should be known in suppliers_dict
    for delivery_name, delivery_supplier in delivery_dict.items():
        for delivery_project, _ in delivery_supplier.items():
            if delivery_project not in suppliers_dict[delivery_name].keys():
                errors = True
                logger.error(f'*** Project in delivery ({delivery_project}) '
                            'is not known among the projects in "config.yaml"')

    if errors:
        raise dc.DidoError('*** DiDo cannot continue with these errors')


    # add delivery info to supplier info
    for supplier_name, _ in delivery_dict.items():
        for project_name, deliveries in delivery_supplier.items():
            # add existing tables names
            tables_dict = dict()

            # for all data tables
            table_data = get_table_names(project_name, supplier_name, 'data')
            for key, value in table_data.items():
                tables_dict[key] = value

            # and all description tables
            # table_data = get_data_table_names(project_name, supplier_name, 'description')
            # for key, value in table_data.items():
            #     tables_dict[key] = value

            suppliers_dict[supplier_name][project_name] \
                ['tables_def'] = tables_dict

            # add all deliveries
            all_deliveries = dict()
            for delivery_name, delivery_info in deliveries.items():
                # remove 'delivery_' from delivery_name
                if delivery_name.startswith('delivery_'):
                    delivery_id = delivery_name[len('delivery_'):]
                else:
                    delivery_id = delivery_name

                all_deliveries[delivery_id] = -1

            suppliers_dict[supplier_name][project_name] \
                ['deliveries_def'] = all_deliveries

            # for
        # for
    # for

    return suppliers_dict

### add_deliveries_to_suppliers ###


def add_table_info_to_deliveries(suppliers_dict: dict, server_config: dict):
    """ Fetches table info from the database and adds it to the suppliers dict

    Args:
        suppliers_dict (dict): contains suppliers, projects and dlivery defs
            from delivery.yaml
        server_config (dict): definition of server to fetch table info from

    Returns:
        _type_: suppliers_dict enhanced with delivery information from the database
    """
    for supplier_name, projects in suppliers_dict.items():
        for project_name in projects.keys():

            # get information for each project for each supplier
            table_names = get_table_names(project_name, supplier_name, 'data')
            tables = {}

            # collect information for each table
            for key in table_names.keys():
                info = get_table_info(table_names[key], table_names, server_config)
                tables[key] = info

            # for

            suppliers_dict[supplier_name][project_name]['tables_found'] = tables

            # iterate over all deliveries found in the database and build
            # dictionary out of it
            deliveries_found = {}
            delivs = tables[TAG_TABLE_SCHEMA]['deliveries']
            if delivs is not None:
                for row in delivs.iterrows():
                    info = row[1]
                    delivery_id = info['levering_rapportageperiode']
                    count = info['count']
                    deliveries_found[delivery_id] = count

                # for
            # if

            # assign delivery dictionary to suppliers_dict
            suppliers_dict[supplier_name][project_name]['deliveries_found'] = deliveries_found

        # for

    # for

    return suppliers_dict

### add_table_info_to_deliveries ###


def get_table_info(table_name: str, tables_name: dict, server_config: dict):
    """ Requests info of a specific dido table

    Args:
        table_name (str): Name of the table to request info on
        tables_name (dict): Dict of all possible table names
        server_config (dict): Server configuration for the database

    Returns:
        _type_: _description_
    """
    # Initialize info record
    info = {'table': table_name,
            'exists': True,
            'records': 0,
            'deliveries': None,
        }

    try:
        query_result = st.sql_select(
            table_name = table_name,
            columns = 'count(*)',
            verbose = False,
            sql_server_config = server_config
        )

        info['records'] = query_result.iloc[0].loc['count']
        if table_name == tables_name[TAG_TABLE_SCHEMA]:
            query_result = st.sql_select(
                table_name = table_name,
                columns = 'DISTINCT levering_rapportageperiode, count(*) ',
                groupby = 'levering_rapportageperiode ORDER BY levering_rapportageperiode',
                verbose = False,
                sql_server_config = server_config
            )
            if len(query_result) > 0:
                info['deliveries'] = query_result

        elif table_name == tables_name[TAG_TABLE_DELIVERY]:
            query_result = st.sql_select(
                table_name = table_name,
                columns = 'DISTINCT levering_rapportageperiode',
                verbose = False,
                sql_server_config = server_config
            )

        # if

    except sqlalchemy.exc.ProgrammingError:
        info['exists'] = False

    return info

### get_table_info ###


def get_supplier_projects(config: dict,
                          supplier: str,
                          delivery,
                          keyword: str,
                         ):
    """ Returns supplier s info from supplier.

        The relevant delivery info is copied and all info about deliveries
        is deleted.

    Args:
        suppliers (dict): dictionary of suppliers
        supplier (str): name of the supplier to select
        delivery (int ro str): sequence number of delivery to apply

    Returns:
        dict: dictionary of supplier addjusted with correct delivery
    """
    suppliers = config[keyword]
    leverancier = suppliers[supplier].copy()

    # test if supplier contains projects
    project_keys = leverancier.keys()

    # old style projects: no projects means al is one project (project_name)
    if len(project_keys) == 0:
        raise DiDoError(f'*** No projects define for supplier {supplier}')

    # project found, only statements just below supplier are projects
    projects = {}
    for proj in project_keys:
        projects[proj] = leverancier[proj].copy()

    leverancier['config'] = config
    leverancier['supplier_id'] = supplier

    return leverancier, projects

### get_supplier_dict ###


def get_supplier_name(supplier_info: dict) -> str:
    if supplier_info is None or len(supplier_info) == 0:
        logger.error('*** dido_kill_project: there are no suppliers found')

        return ''

    # if

    supplier_list = list(supplier_info.keys())
    logger.info('Suppliers found:')
    for supplier_name in supplier_list:
        logger.info(f' - {supplier_name}')

    supplier_name = ''
    while supplier_name not in supplier_list:
        supplier_name = \
            input('Which supplier to select? (ctrl-C to exit): ')

    # while

    return supplier_name

### get_supplier_name ###


def get_project_name(supplier_info: dict, supplier_name: str) -> str:
    projects = supplier_info[supplier_name]
    if projects is None or len(projects) == 0:
        logger.error(f'No projects found for {supplier_name}')

        return ''

    # if

    project_list = list(supplier_info[supplier_name].keys())
    logger.info(f'Projects found for {supplier_name}:')
    for project_name in project_list:
        logger.info(f' - {project_name}')

    project_name = ''
    while project_name not in project_list:
        project_name = \
            input('Which project to select from {suppolier_name}? (ctrl-C to exit): ')

    # while

    return project_name

### get_project_name ###


def enhance_cargo_dict(cargo_dict: dict, cargo_name, supplier_name: str):
    """ Returns supplier s info from supplier.

        The relevant delivery info is copied and all info about deliveries
        is deleted.

    Args:
        suppliers (dict): dictionary of suppliers
        supplier (str): name of the supplier to select
        delivery (int ro str): sequence number of delivery to apply

    Returns:
        dict: dictionary of supplier addjusted with correct delivery
    """
    splits = cargo_name.split('_')
    if splits[0] != 'delivery':
        raise DiDoError(f'*** Delivery should start with "delivery_", error for "{cargo_name}"')

    cargo_dict[ODL_LEVERING_FREK] = splits[1]
    cargo_dict['supplier_id'] = supplier_name

    return cargo_dict

### get_supplier_dict ###


def display_leveranties(leveranciers: dict):
    """ Displays all supplies for all suppliers and all projects in
        a user friendly way

    Args:
        leveranciers (dict): overview of suppliers, projects and deliveries
    """
    for leverancier_naam, leverancier_info in leveranciers.items():
        logger.info(f'Supplier: {leverancier_naam}')
        for project_name, project_info in leverancier_info.items():
            logger.info(f'  Project: {project_name}')
            if 'tables_found' in project_info.keys():
                logger.info('')
                logger.info('  Required Tables')
                tables = project_info['tables_found']
                table_count = 0
                all_exist = True
                for table, info in tables.items():
                    n_delivs = 0
                    exists = 'non-existent'
                    if info['exists']:
                        if info['deliveries'] is not None:
                            n_delivs = len(info['deliveries'])
                        else:
                            n_delives = 0
                    else:
                        all_exist = False

                    s = info["table"]
                    logger.info(f'  - {s:<40s}' \
                        f'{exists:>15s} ' \
                        f'{info["records"]:12,d}')
                # for

                if not all_exist:
                    logger.info(f'  * Not all required tables exist for ' \
                        f'{project_name}, DiDo will not work for this project *')

            # if

            if 'deliveries_found' in project_info.keys():
                logger.info('')
                logger.info('  Deliveries')
                deliveries = project_info['deliveries_found']
                n = 0
                for periode, telling in deliveries.items():
                    n += 1
                    logger.info(f'  - {periode:<10s} {telling:9d}')

                if n == 0:
                    logger.info(f'  * There are no deliveries for {project_name} *')

            # if

            logger.info('')

        # for
    # for

    return

### display_leveranties ###


def get_cargo(cargo_config: dict, supplier: str, project_key: str):
    cargo = cargo_config['DELIVERIES'][supplier][project_key]

    return cargo

### get_cargo ###


def get_current_delivery_seq(project_name: str, supplier: str, server_config: dict):
    tables = get_table_names(project_name, supplier)

    table_name = tables[TAG_TABLE_DELIVERY]
    try:
        count = st.table_size(table_name, True, server_config)
    except:
        raise DiDoError(f'Table {table_name} does not exist. Are you sure you used create_table with the current config.yaml file?')
    # try..except

    return count

### get_current_delivery_seq ###


def report_psql_use(table: str, servers: dict, tables_exist: bool, overwrite: bool):
    """ Reports to a user he shoukld use psql. The correct command is displayed

    Args:
        table (str): name of the fiole containing the SQL instructions
        servers (dict): dictionary of server configurations
        tables_exist (bool): True if tables already exist, else False
    """
    host = servers['DATA_SERVER_CONFIG']['POSTGRES_HOST']
    user = servers['DATA_SERVER_CONFIG']['POSTGRES_USER']
    db = servers['DATA_SERVER_CONFIG']['POSTGRES_DB']

    report_string = f'$ psql -h {host} -U {user} -f {table}.sql {db}'

    logger.info('')
    logger.info("Don't forget to run psql to create or fill the tables")
    logger.info("This can best be done from your work/sql directory")
    logger.info('Suggested command:')
    logger.info('')
    logger.info(report_string)

    if tables_exist:
        logger.info('')
        logger.error('*** You have been warned that the tables you want to create already exists.')
        if overwrite:
            logger.error('*** Current tables including contents will be deleted ***')
            logger.error('*** Be sure that is what you wish ***')

        else:
            logger.error('*** If you really want to recreate these tables, thereby erasing current contents')
            logger.error('*** run dido_kill_project.py ***')

    logger.info('')

    return

### report_psql_use ###


def show_database(server_config: dict,
                  table_name: str = '',
                  pfun = logger.debug,
                  title: str = '',
                 ):

    pfun('')
    if len(title) > 0:
        logger.info(title)
    pfun(f'Server:   {server_config["POSTGRES_HOST"]}')
    pfun(f'Port:     {server_config["POSTGRES_PORT"]}')
    pfun(f'Database: {server_config["POSTGRES_DB"]}')
    pfun(f'Schema:   {server_config["POSTGRES_SCHEMA"]}')
    pfun(f'User:     {server_config["POSTGRES_USER"]}')
    pfun('')
    if len(table_name) > 0:
        pfun(f'Table:    {table_name}')

    return

### show_database ###


def delivery_exists(delivery: dict,
                    supplier_id: str,
                    project_name: str,
                    cargo_name: str,
                    server_configs: dict
                   ) -> bool:
    """ Checks whether a delivery exists. Column 'levering_rapportageperiode is used
        to check this.

    Args:
        delivery (dict): dictionary containing the deliverydescription.
            Used to get the key 'levering_rapportageperiode' with thge current delivery
        supplier_id (str): supplier name, used for table name construction
        project_name (str): name of the project, used for table name construction
        server_configs (dict): the existence of deliveries is chcked in the data server database

    Returns:
        bool: True = delivery exists, else not
    """
    # fetch a dataframe with deliveries from the data table using distinct
    # on levering_rapportageperiode

    server_config = server_configs['DATA_SERVER_CONFIG']
    table_name = get_table_name(project_name, supplier_id, TAG_TABLE_DELIVERY, 'data')
    try:
        leveringen = st.sql_select(
            table_name = table_name,
            columns = f'DISTINCT {ODL_LEVERING_FREK}',
            sql_server_config = server_config,
            verbose = False,
        )
    except sqlalchemy.exc.ProgrammingError:
        logging.critical(f'*** Tables do not exist for {supplier_id}_{project_name}_...')
        show_database(
            server_config = server_config,
            table_name = table_name,
            pfun = logger.info,
            title = '',
        )
        raise DiDoError('Error while fetching tables. Did you ever apply create_tables '
                        f'for {supplier_id}_{project_name}?')

    # try..except

    leveringen_lijst = leveringen[ODL_LEVERING_FREK].tolist()
    if len(leveringen_lijst) > 0:
        show_database(
            server_config = server_config,
            table_name = table_name,
            pfun = logger.debug,
            title = '',
        )
        logger.debug(f'Deliveries present in the database: {leveringen_lijst}')

    # if

    current_delivery = cargo_name.split('_')[1]
    exists = current_delivery in leveringen_lijst
    if exists:
        logger.debug(f'Delivery {current_delivery} already in the database')

    return exists

### delivery_exists ###
