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


def select_tables_for(tables: list,
                      supplier_name: str,
                      project_name: int,
                     ):

    new_tables = []
    identifier = f'{supplier_name}_{project_name}_'
    for table_name in tables:
        if table_name.startswith(identifier):
            new_tables.append(table_name)

    return new_tables

### select_tables_for ###


def delete_tables(table_names: list, servers: dict):
    server = servers['DATA_SERVER_CONFIG']
    schema = server['POSTGRES_SCHEMA']
    query = 'DROP TABLE '
    for table_name in table_names:
        query += f'{schema}.{table_name}, '

    query = query [:-2] + ';'

    result = st.sql_statement(
        statement = query,
        sql_server_config = server,
    )

    logger.info(result)

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

    # get the database server definitions
    db_servers = config_dict['SERVER_CONFIGS']

    # get project environment
    leveranciers = config_dict['SUPPLIERS']

    dc.show_database(
        server_config = db_servers['DATA_SERVER_CONFIG'],
        pfun = logger.info,
        title = 'Tables are destroyed in the following database',
    )

    # if there is no supplier that received any supply, there is nothing to remove.
    # The program terminates
    supplier_info = dc.get_suppliers(leveranciers)
    supplier_info = dc.add_table_info_to_deliveries(
        suppliers_dict = supplier_info,
        server_config = db_servers['DATA_SERVER_CONFIG']
    )

    # get name of supplier to delete
    logger.info('Selecting supplier and project to kill')
    supplier_name = dc.get_supplier_name(supplier_info)
    logger.info(f'[Supplier selected is {supplier_name}]')

    # get project name to delete for this supplier
    project_name = dc.get_project_name(supplier_info, supplier_name)
    logger.info(f'[Project selected is {project_name}]')

    table_list = get_table_names(supplier_name, db_servers)

    table_list = select_tables_for(
        tables = table_list,
        supplier_name = supplier_name,
        project_name = project_name,
    )

    if len(table_list) == 0:
        logger.error(f'*** No tables for supplier {supplier_name}, project {project_name} ***')

    else:
        logger.info('The following projects and all of their data will be destroyed')
        for name in table_list:
            logger.info(f' - {name}')

        # for

        logger.info('')
        response = input('Is that ok with you (Ja, nee)? ')
        if response == 'Ja':
            delete_tables(table_list, db_servers)

        else:
            logger.info('Nothing deleted')

        # if
    # if

    cpu = time.time() - cpu
    logger.info('')
    logger.info(f'[Ready {appname["name"]} in {cpu:.0f} seconds]')
    logger.info('')

    return

### dido_kill ###


if __name__ == '__main__':
    # read commandline parameters to create log_file from
    cli, args = dc.read_cli()

    # create logger in project directory
    log_file = os.path.join(args.project, 'logs/' + cli['name'] + '.log')
    logger = dc.create_log(log_file, level = 'DEBUG')

    # go
    dido_kill('KILL ALL tables of selected Supplier!')
