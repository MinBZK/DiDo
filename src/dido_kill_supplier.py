"""
Kills all tables and their contents from one, sp[ecific supplier of a project.

Lists all suppliers and promps the user for the name to erase.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import sqlalchemy

import dido_common as dc
import simple_table as st

# pylint: disable=bare-except, line-too-long, consider-using-enumerate
# pylint: disable=logging-fstring-interpolation, too-many-locals
# pylint: disable=pointless-string-statement

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
        result = st.sql_select(
            table_name = table_name,
            columns = 'count(*)',
            verbose = False,
            sql_server_config = server_config
        )

        info['records'] = result.iloc[0].loc['count']
        if table_name == tables_name[dc.TAG_TABLE_SCHEMA]:
            result = st.sql_select(
                table_name = table_name,
                columns = 'DISTINCT levering_rapportageperiode, count(*)',
                where = 'GROUP BY levering_rapportageperiode '
                        'ORDER BY levering_rapportageperiode',
                verbose = False,
                sql_server_config = server_config
            )
            if len(result) > 0:
                info['deliveries'] = result

        elif table_name == tables_name[dc.TAG_TABLE_DELIVERY]:
            result = st.sql_select(
                table_name = table_name,
                columns = 'DISTINCT levering_rapportageperiode',
                verbose = False,
                sql_server_config = server_config
            )

    except sqlalchemy.exc.ProgrammingError:
        info['table'] = '*** Tabel bestaat niet ***'

    return info

### get_info ###


def get_deliveries(project_name: str, supplier: str, server_config: dict) -> dict:
    """ get all leveranties for this supplier

    Args:
        project_name (str): Name of project
        leverancier (str): Name of leverancier
        server_config (dict): Database access properties

    Returns:
        dict: for all suppliers and data table a dict of all supplies
    """
    tables_name = dc.get_table_names(project_name, supplier)
    tables_info = {}

    for key in tables_name.keys():
        info = get_info(tables_name[key], tables_name, server_config)
        tables_info[key] = info

    return tables_info

### get_deliveries ###


def fetch_suppliers(leveranciers: dict, data_server_config: dict):
    """ Displays the supplies of of specific supplier

    Args:
        leveranties (dict): Dictionary of all supplies (by get_leveranties)
        supplier (str): Name of supplier to request list of supplies from
    """
    logger.info('')
    logger.info(f'Overzicht van leveranciers in config.yaml')

    leverancier_met_data = []

    for key in leveranciers.keys():
        leverancier_met_data.append(key)
        logger.info(f' - {key}')

    return leverancier_met_data

### fetch_suppliers ###


def show_database(title: str, config: dict):
    logger.info(title)
    logger.info(f'Server:   {config["POSTGRES_HOST"]}')
    logger.info(f'Port:     {config["POSTGRES_PORT"]}')
    logger.info(f'Database: {config["POSTGRES_DB"]}')
    logger.info(f'Schema:   {config["POSTGRES_SCHEMA"]}')
    logger.info(f'User:     {config["POSTGRES_USER"]}')
    logger.info('')

    return

### show_database ###


def get_table_names(supplier_id: str, servers: dict):
    """ Get a dataframe of all tables of a schema

    Args:
        supplier_id (str): name of the supplier
        schema (str): name of database schema to request all tables from
        servers (dict): list of all database servers

    Returns:
        A dataframe with information on all tables in schema
    """
    data_server = servers['DATA_SERVER_CONFIG']
    query = "SELECT * FROM information_schema.tables WHERE " \
           f"table_schema = '{data_server['POSTGRES_SCHEMA']}';"
    result = st.query_to_dataframe(query, sql_server_config = data_server)

    # result = result[result['table_name'].str.contains('schema_data')]
    tables = result['table_name'].tolist()

    return tables

### get_table_names ###


def select_tables_from_supplier(tables: list,
                                supplier_id: str,
                                project_idx: int,
                               ):

    new_tables = []
    projects = []
    for supplier in tables:
        if supplier.startswith(supplier_id):
            new_tables.append(supplier)
            splits = supplier.split('_')
            projects.append(splits[project_idx])

    projects = set(projects)

    return new_tables, projects

### select_tables_from_supplier ###


### @@@ Hier gebleven
def delete_tables(table_names: list, servers: dict):
    server = servers['DATA_SERVER_CONFIG']
    schema = server['POSTGRES_SCHEMA']
    query = 'DROP TABLE '
    for table_name in table_names:
        query += f'{schema}.{table_name}, '

    query = query [:-2] + ';'
    print(query)

    result = st.sql_statement(
        statement = query,
        sql_server_config = server,
    )

    print(result)

    return

### delete_tables ###


def dido_kill(header: str):
    cpu = time.time()

    dc.display_dido_header(header)

    # read commandline parameters
    appname, args = dc.read_cli()

    # read the configuration file
    config_dict = dc.read_config(args.project)
    supplier_to_be_killed = args.supplier
    yes_to_all_questions = args.Yes

    # get the database server definitions
    db_servers = config_dict['SERVER_CONFIGS']
    odl_server_config = db_servers['ODL_SERVER_CONFIG']
    data_server_config = db_servers['DATA_SERVER_CONFIG']
    foreign_server_config = db_servers['FOREIGN_SERVER_CONFIG']

    # get project environment
    root_dir = config_dict['ROOT_DIR']
    work_dir = config_dict['WORK_DIR']
    leveranciers = config_dict['SUPPLIERS']
    columns_to_write = config_dict['COLUMNS']
    report_periods = config_dict['REPORT_PERIODS']
    parameters = config_dict['PARAMETERS']

   # create the output file names
    report_csv_filename = os.path.join(work_dir, dc.DIR_DOCS, 'all-import-errors.csv')
    report_doc_filename = os.path.join(work_dir, dc.DIR_DOCS, 'all-import-errors.md')
    sql_filename        = os.path.join(work_dir, dc.DIR_SQL, 'remove-deliveries.sql')

    show_database('Tables are destroyed in the following database',
                  data_server_config)
    # if there is no supplier that received any supply, there is nothing to remove.
    # The program terminates
    n_suppliers = fetch_suppliers(leveranciers, data_server_config)
    if len(n_suppliers) < 1:
        logger.info('')
        logger.warning('!!! No suppliers in config.yaml, DiDo quits')

        sys.exit()

    #if

    # get name of supplier to delete
    logger.info('')
    if supplier_to_be_killed is not None:
        leverancier = supplier_to_be_killed
        logger.info(f'Leverancier gekregen vanuit commandline: {leverancier}')

    else:
        leverancier = ''

    leverancier = 'pdirekt'
    while leverancier not in leveranciers:
        prompt = 'Enter supplier name (ctrl-C to exit): '
        leverancier = input(prompt).lower()
        logger.debug(f'{prompt}{leverancier}')

        # check if it exists
        if leverancier not in leveranciers:
            logger.error(f'*** Supplier "{leverancier}" not specified in config.yaml')

    # while

    table_list = get_table_names(leverancier, db_servers)
    table_list, project_names = select_tables_from_supplier(
        tables = table_list,
        supplier_id = leverancier,
        project_idx = 1,
    )

    if len(table_list) < 1:
        logger.info('')
        logger.info(f'No projects for {leverancier}, DiDo quits.')

        sys.exit()

    logger.info('The following projects and all of their data will be destroyed')
    for name in project_names:
        logger.info(f' - {name}')

    logger.info('')
    prompt = 'Are you sure to destroy all data (Ja/Nee): '
    response = input(prompt)
    logger.debug(f'{prompt}{response}')

    if response != 'Ja':
        logger.info('Opting to not destroy all data')
        logger.info('')

        sys.exit()

    delete_tables(table_list, db_servers)

    cpu = time.time() - cpu
    logger.info('')
    logger.info(f'[Ready {appname["name"]} in {cpu:.0f} seconds]')
    logger.info('')

    return


if __name__ == '__main__':
    # read commandline parameters to create log_file from
    cli, args = dc.read_cli()

    # create logger in project directory
    log_file = os.path.join(args.project, 'logs/' + cli['name'] + '.log')
    logger = dc.create_log(log_file, level = 'DEBUG')

    # go
    dido_kill('KILL ALL tables of selected Supplier!')
