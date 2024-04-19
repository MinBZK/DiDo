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
                where = 'group by levering_rapportageperiode order by levering_rapportageperiode',
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


def get_leveranties(project_name: str, supplier: str, server_config: dict) -> dict:
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
    project_name = config_dict['PROJECT_NAME']
    root_dir = config_dict['ROOT_DIR']
    work_dir = config_dict['WORK_DIR']
    leveranciers = config_dict['SUPPLIERS']
    columns_to_write = config_dict['COLUMNS']
    table_desc = config_dict['TABLES']
    report_periods = config_dict['REPORT_PERIODS']
    parameters = config_dict['PARAMETERS']

   # create the output file names
    report_csv_filename = os.path.join(work_dir, dc.DIR_DOCS, 'all-import-errors.csv')
    report_doc_filename = os.path.join(work_dir, dc.DIR_DOCS, 'all-import-errors.md')
    sql_filename        = os.path.join(work_dir, dc.DIR_SQL, 'remove-deliveries.sql')

    show_database('Tabellen worden vernietigd in de volgende database', data_server_config)
    # if there is no supplier that received any supply, there is nothing to remove.
    # The program terminates
    n_suppliers = display_leveranciers(project_name, leveranciers, data_server_config)
    if len(n_suppliers) < 1:
        logger.info('')
        logger.warning('!!! Er zijn geen leveranciers te bekennen in config.yaml.')
        logger.warning('!!! Er valt niets te verwijderen, het programma wordt beeindigd.')

        sys.exit()

    #if

    # get name of supplier to delete
    logger.info('')
    if supplier_to_be_killed is not None:
        leverancier = supplier_to_be_killed
        logger.info(f'Leverancier gekregen vanuit commandline: {leverancier}')
    else:
        leverancier = ''

    while leverancier not in leveranciers:
        prompt = 'Typ de naam van de leverancier in: '
        leverancier = input(prompt).lower()
        logger.debug(f'{prompt}{leverancier}')

        # check if it exists
        if leverancier not in leveranciers:
            logger.error(f'*** Geen bestaande leveranciersnaam in dit project: "{leverancier}"')
            logger.info('[Gebruik ctrl-C als u wilt stoppen zonder iets te vernietigen]')

    # while

    data_tables = dc.get_table_names(project_name, leverancier, 'data')
    desc_tables = dc.get_table_names(project_name, leverancier, 'description')
    tables_to_remove = []
    schema = data_server_config['POSTGRES_SCHEMA']
    for key in data_tables.keys():
        tables_to_remove.append(schema + '.' + data_tables[key])
        tables_to_remove.append(schema + '.' + desc_tables[key])

    logger.info('')
    n = 0
    logger.info(f'*** De volgende {len(tables_to_remove)} tabellen van *{leverancier}* worden vernietigd:')
    for table in tables_to_remove:
        # table contains schema before name, not accepted by table_exists, remove it
        table = table.split('.')[1]
        if st.table_exists(table, sql_server_config = data_server_config):
            n_records = st.table_size(table, sql_server_config = data_server_config)
            logger.info(f' - {table} bevat {n_records} records')
            n += 1
        else:
            logger.info(f' - {table} bestaat niet')

    logger.info('')

    if n > 0:
        logger.info('')
        if yes_to_all_questions is not None:
            logger.info(f'Reponse via commandline: {yes_to_all_questions}')

            response = yes_to_all_questions
        else:
            prompt = 'Weet je het heel zeker? (Ja/nee): '
            response = input(prompt)
            logger.debug(prompt + response)

        if response =='Ja':

            # prepare SQL statements and run these thru simple_table
            logger.info('')
            table_names = dc.get_table_names(project_name, leverancier)
            sql = ''
            for table in tables_to_remove:
                sql += f'DROP TABLE IF EXISTS {table} CASCADE;\n'

            logger.debug(sql)
            st.sql_statement(sql, verbose = True, sql_server_config = data_server_config)
            logger.info('Tabellen zijn verwijderd')

        else:
            logger.info('Ok, er is niets verwijderd')

        # if

    else:
        logger.info(f'{leverancier} heeft geen tabellen, er wordt niets vernietigd.')

    # if

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
