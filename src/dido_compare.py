import os
import re
import sys
import json
import time
import locale
from datetime import datetime
import numpy as np
import pandas as pd
import sqlalchemy

import common
import simple_table

from dido_common import DiDoError
from dido_common import read_cli, read_config, get_limits, display_dido_header
from dido_common import load_schema, get_table_names, load_odl_table
from dido_common import TAG_TABLE_SCHEMA, TAG_TABLE_META, TAG_TABLE_DELIVERY, TAG_TABLE_QUALITY
from dido_common import TAG_TABLES, TAG_PREFIX, TAG_SUFFIX, TAG_SCHEMA, TAG_DATA
from dido_common import DIR_SCHEMAS, DIR_DOCS, DIR_DONE, DIR_TODO, DIR_SQL
from dido_common import VALUE_OK, VALUE_NOT_IN_LIST, VALUE_MANDATORY_NOT_SPECIFIED
from dido_common import VALUE_NOT_BETWEEN_MINMAX, VALUE_OUT_OF_REACH, VALUE_IMPROBABLE
from dido_common import VALUE_WRONG_DATATYPE, VALUE_HAS_WRONG_FORMAT, VALUE_NOT_CONFORM_RE
from dido_common import ODL_RECORDNO, ODL_CODE_BRONBESTAND, ODL_LEVERING_FREK, ODL_SYSDATUM
from dido_common import ODL_DELIVERY_DATE, EXTRA_TEMPLATE

# pylint: disable=line-too-long

# print all columns of dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# raise exceptions on all numpy warnings
np.seterr(all='raise')


def get_info(table_name: str, tables_name: dict, server_config: dict):
    """ Requests info of a specific d3g table

    Args:
        table_name (str): Name of the table to request info on
        tables_name (dict): Dict of all possible table names
        server_config (dict): Server configuration for the database

    Returns:
        _type_: _description_
    """
    info = {'table': 'Tabel bestaat',
            'records': 0,
            'deliveries': None,
           }

    try:
        result = simple_table.sql_select(table_name = table_name,
                                        columns = 'count(*)',
                                        verbose = False,
                                        sql_server_config = server_config
                                        )

        info['records'] = result.iloc[0].loc['count']
        if table_name == tables_name[TAG_TABLE_SCHEMA]:
            result = simple_table.sql_select(table_name = table_name,
                                            columns = 'DISTINCT levering_rapportageperiode, count(*)',
                                            where = 'group by levering_rapportageperiode order by levering_rapportageperiode',
                                            verbose = False,
                                            sql_server_config = server_config
                                            )
            if len(result) > 0:
                info['deliveries'] = result

        elif table_name == tables_name[TAG_TABLE_DELIVERY]:
            result = simple_table.sql_select(table_name = table_name,
                                            columns = 'DISTINCT levering_rapportageperiode',
                                            verbose = False,
                                            sql_server_config = server_config
                                            )

    except sqlalchemy.exc.ProgrammingError:
        info['table'] = '*** Tabel bestaat niet ***'

    return info


def get_bootstrap_data_headers(server_config: dict):
    # Read bootstrap data
    bootstrap_data = load_odl_table(EXTRA_TEMPLATE, server_config)
    columns = bootstrap_data.loc[:, 'kolomnaam'].tolist()

    return columns


def get_leveranties(project_name: str, supplier: str, server_config: dict) -> dict:
    """ get all leveranties for this supplier

    Args:
        project_name (str): Name of project
        leverancier (str): Name of leverancier
        server_config (dict): Database access properties

    Returns:
        dict: for all suppliers and data table a dict of all supplies
    """
    tables_name = get_table_names(project_name, supplier)
    tables_info = {}

    for key in tables_name.keys():
        info = get_info(tables_name[key], tables_name, server_config)
        tables_info[key] = info

    return tables_info


def display_leveranciers(project_name: str, leveranciers: dict, data_server_config: dict):
    """ Displays the supplies of of specific supplier

    Args:
        leveranties (dict): Dictionary of all supplies (by get_leveranties)
        supplier (str): Name of supplier to request list of supplies from
    """
    logger.info('')
    logger.info(f'Overzicht van leveranciers in config.yaml')

    leverancier_met_data = []

    for key in leveranciers.keys():
        # leveranties = get_leveranties(project_name, key, data_server_config)
        leverancier_met_data.append(key)
        logger.info(f' - {key}')

    return leverancier_met_data


def show_database(title: str, config: dict):
    logger.info(title)
    logger.info(f'Server:   {config["POSTGRES_HOST"]}')
    logger.info(f'Port:     {config["POSTGRES_PORT"]}')
    logger.info(f'Database: {config["POSTGRES_DB"]}')
    logger.info(f'Schema:   {config["POSTGRES_SCHEMA"]}')
    logger.info(f'User:     {config["POSTGRES_USER"]}')
    logger.info('')

    return


def dump_table(table_name: str, target_name: str, snap_date: object, server_config: dict):
    schema = server_config['POSTGRES_SCHEMA']
    query = f"SELECT * FROM {schema}.show_mutations_at('{snap_date}')"
    table = simple_table.query_to_dataframe(query,
                                            sql_server_config = server_config)

    table.to_csv(target_name, sep=';', index = False)

    logger.info(f'Table:      {table_name}')
    logger.info(f'Written to: {target_name}')

    return

def compare_table(table_df: pd.DataFrame,
                  target_df: pd.DataFrame,
                  table_name: str,
                  snap_date: object) -> bool:


    table_df['bronbestand_recordnummer'] = table_df['bronbestand_recordnummer'].astype(int)
    table_df = table_df.set_index('bronbestand_recordnummer')
    table_df = table_df.sort_values(by = ['bronbestand_recordnummer'])
    table_df = table_df.sort_index()

    target_df['bronbestand_recordnummer'] = target_df['bronbestand_recordnummer'].astype(int)
    target_df = target_df.set_index('bronbestand_recordnummer')
    table_df = table_df.sort_index()

    logger.info('')
    logger.info('Comparing database table with pre-stored test set')
    logger.info('While reporting results:')
    logger.info(f'  self = database table to compare ({table_name})')
    logger.info('  other = prestored test set')
    logger.info('')

    # are shapes equal?
    shape_result = target_df.shape == table_df.shape
    if shape_result:
        logger.info(f'Shapes: equal')
    else:
        logger.error('Shapes are not equal')

    logger.info(f'  self:  {table_df.shape}')
    logger.info(f'  other: {target_df.shape}')

    if not shape_result:
        return False

    # are column names equal?
    logger.info('')
    logger.info('Comparing column names')
    column_names_eq = True
    for i in range(len(table_df.columns)):
        table_col = table_df.columns[i]
        target_col = target_df.columns[i]
        equals = table_col == target_col

        logger.info(f'{i:3} - self  {table_col}')
        if equals:
            logger.info(f'    - other {target_col}')
        else:
            logger.info(f'    x other {target_col}')

        if table_col != target_col:
            column_names_eq = False

    if not column_names_eq:
        logger.error('Column names are not equal')
        return False

    # match table values with stored test results
    result = table_df.compare(target_df, align_axis = 0)
    equal_values = result.shape == (0, 0)
    logger.info('')

    if equal_values:
        logger.info('Value comparison:')
        logger.info('  Values of table and test set are equal')

    else:
        logger.error(f'Value Compare: False')
        logger.info('Differences shown below')
        logger.info(f'\n{result}')

    return equal_values


def dido_compare(header: str):
    """_summary_

    Args:
        header (str): _description_

    Raises:
        DiDoError: _description_
        DiDoError: _description_
    """
    cpu = time.time()
    display_dido_header(header)

    # read commandline parameters
    appname, args = read_cli()

    # read the configuration file
    config_dict = read_config(args.project)

    # read command line parameters
    supplier_to_compare = args.supplier
    if supplier_to_compare is None:
        raise DiDoError('--supplier <suppliername> not specified')

        raise SystemExit(1)

    view_data = args.view == 'view'

    # Read the date as string and convert to date format
    if args.date is not None:
        snap_date = datetime.strptime(args.date, '%Y-%m-%d').date()

    target = args.target
    compare = args.compare

    # get some limiting variables
    SAMPLE_SIZE, SAMPLE_FRACTION, MAX_ERRORS = get_limits(config_dict)

    # get the database server definitions
    db_servers = config_dict['SERVER_CONFIGS']
    odl_server_config = db_servers['ODL_SERVER_CONFIG']
    data_server_config = db_servers['DATA_SERVER_CONFIG']
    foreign_server_config = db_servers['FOREIGN_SERVER_CONFIG']

    # get project environment
    project_name = config_dict['PROJECT_NAME']
    root_dir = config_dict['ROOT_DIR']
    work_dir = config_dict['WORK_DIR']
    leveranciers = config_dict['SUPPLIERS']
    columns_to_write = config_dict['COLUMNS']
    table_desc = config_dict['TABLES']
    report_periods = config_dict['REPORT_PERIODS']
    parameters = config_dict['PARAMETERS']

    table_data_names = get_table_names(project_name, supplier_to_compare)
    table_desc_names = get_table_names(project_name, supplier_to_compare, 'description')

    cmp_dir = os.path.join(work_dir, 'cmp', supplier_to_compare)
    tests_dir = os.path.join(work_dir, 'tests', supplier_to_compare)
    table_name = table_data_names[TAG_TABLE_SCHEMA]

    # get the ODL extra columns to remove these of the tables to compare
    xtra_columns = get_bootstrap_data_headers(odl_server_config)
    xtra_columns.remove('bronbestand_recordnummer')

    if target is None:
        target = table_name + '.csv'

    result = True
    if compare =='dump':
        target = os.path.join(cmp_dir, target)
        dump_table("show_mutations_at('2023-06-15')", target, snap_date, data_server_config)

    elif compare == 'compare':
        # get target and remove odl columns
        target_name = os.path.join(tests_dir, target)
        target_df = pd.read_csv(target_name, sep = ';').astype(str)
        target_df = target_df.drop(labels = xtra_columns, axis = 'columns')

        # get table to compare from database and remove odl columns
        schema = data_server_config['POSTGRES_SCHEMA']
        query = f"SELECT * FROM {schema}.show_mutations_at('{snap_date}')"
        table_df = simple_table.query_to_dataframe(
            query,
            sql_server_config = data_server_config
            ).astype(str)

        table_df = table_df.drop(labels = xtra_columns, axis = 'columns')
        if view_data:
            logger.info('')
            logger.info(f'View of table {table_name}\n{table_df}')

        result = compare_table(table_df, target_df, table_name, snap_date)#, data_server_config)

    else:
        logger.critical('--compare: accepts only post gres test for empty string"dump" or "compare"')

        raise SystemExit(1)

    if not result:
        logger.info('')
        logger.error('*** Table and target are not equal, exit with value 1')

        raise SystemExit(1)

    cpu = time.time() - cpu
    logger.info('')
    logger.info(f'[Ready {appname["name"]} in {cpu:.02f} seconds]')
    logger.info('')

    return


if __name__ == '__main__':
    # read commandline parameters to create log_file from
    cli, args = read_cli()
    reset = args.reset == 'reset'

    # create logger in project directory
    log_file = os.path.join(args.project, 'logs/' + cli['name'] + '.log')
    logger = common.create_log(log_file, level = 'DEBUG', reset = reset)

    # go
    dido_compare('Compares a table with a target .csv file')
