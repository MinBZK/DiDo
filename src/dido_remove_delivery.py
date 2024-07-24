import os
import sys
import time
import numpy as np
import pandas as pd

import dido_common as dc
import simple_table as st

# print all columns of dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# raise exceptions on all numpy warnings
np.seterr(all='raise')

# pylint: disable=bare-except, line-too-long, consider-using-enumerate
# pylint: disable=logging-fstring-interpolation, too-many-locals
# pylint: disable=pointless-string-statement, consider-using-dict-items


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

    # try:
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
            columns = 'DISTINCT levering_rapportageperiode, count(*) ',
            groupby = 'levering_rapportageperiode order by levering_rapportageperiode',
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

    # except sqlalchemy.exc.ProgrammingError:
    #     info['table'] = '*** Tabel bestaat niet ***'

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
    tables_name = dc.get_table_names(project_name, supplier, 'data')
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
    logger.info(f'[Overzicht van leveranciers]')

    leverancier_met_data = []

    for key in leveranciers.keys():
        leveranties = get_leveranties(project_name, key, data_server_config)
        if leveranties['schema']['records'] > 0:
            leverancier_met_data.append(key)

            logger.info(f' - {key}')

    return leverancier_met_data


def display_projects(leverancier: str, server_config):
    schema = server_config['POSTGRES_SCHEMA']
    sql = f"SELECT * FROM information_schema.tables "
          f"WHERE table_schema = '{schema};'"

    schema_table = st.sql_statement(sql, serverconfig)
    projects = []






def display_leveranties(leveranties: dict, supplier: str):
    """ Displays the supplies of of specific supplier

    Args:
        leveranties (dict): Dictionary of all supplies (by get_leveranties)
        supplier (str): Name of supplier to request list of supplies from
    """
    logger.info('')
    logger.info(f'[Leverancier: {supplier}]')

    for key in leveranties.keys():
        logger.info(f' - {key}: {leveranties[key]["table"]}')
        info = leveranties[key]

        if info['table'][0] == '*':
            logger.info(f'    {info["table"]}')

        else:
            logger.info(f'    {info["table"]}, {info["records"]} records')
            if info['deliveries'] is not None:
                logger.info(f'    Levering rapportageperiode     Aantal')
                for index, row in info['deliveries'].iterrows():
                    logger.info(f'{row["levering_rapportageperiode"]:>30s} {row["count"]:10d}')

        # if
    # for

    return


def show_database(title: str, config: dict):
    logger.info(title)
    logger.info(f'Server:   {config["POSTGRES_HOST"]}')
    logger.info(f'Port:     {config["POSTGRES_PORT"]}')
    logger.info(f'Database: {config["POSTGRES_DB"]}')
    logger.info(f'Schema:   {config["POSTGRES_SCHEMA"]}')
    logger.info(f'User:     {config["POSTGRES_USER"]}')
    logger.info('')

    return


def dido_remove(header: str):
    cpu = time.time()

    # read commandline parameters
    appname, args = dc.read_cli()

    # read the configuration file
    config_dict = dc.read_config(args.project)

    dc.display_dido_header(header, config_dict)

    # get the database server definitions
    db_servers = config_dict['SERVER_CONFIGS']
    odl_server_config = db_servers['ODL_SERVER_CONFIG']
    data_server_config = db_servers['DATA_SERVER_CONFIG']
    foreign_server_config = db_servers['FOREIGN_SERVER_CONFIG']

    # get project environment
    # project_name = config_dict['PROJECT_NAME']
    # root_dir = config_dict['ROOT_DIR']
    work_dir = config_dict['WORK_DIR']
    leveranciers = config_dict['SUPPLIERS']
    # columns_to_write = config_dict['COLUMNS']
    # table_desc = config_dict['TABLES']
    # report_periods = config_dict['REPORT_PERIODS']
    # parameters = config_dict['PARAMETERS']

    # if there is no supplier that received any supply, there is nothing to remove.
    # The program terminates
    n_suppliers = display_leveranciers(project_name, leveranciers, data_server_config)
    if len(n_suppliers) < 1:
        logger.info('')
        logger.warning('!!! Er zijn geen leveranciers met een of meer leveranties.')
        logger.warning('!!! Er valt niets te verwijderen, het programma wordt beeindigd.')

        sys.exit()

    #if

    # get name of supplier to delete
    logger.info('')
    leverancier = ''
    while leverancier not in leveranciers:
        prompt = 'Typ de naam van de leverancier in: '
        leverancier = input(prompt).lower()
        logger.debug(f'{prompt}{leverancier}')

        # check if it exists
        if leverancier not in leveranciers:
            logger.error(f'*** Geen bestaande leveranciersnaam in dit project: "{leverancier}"')

    # while

    leveranties = get_leveranties(project_name, leverancier, data_server_config)
    display_leveranties(leveranties, leverancier)

    df = leveranties['schema']['deliveries']

    supplies = df['levering_rapportageperiode'].tolist()

    # display alle supplies from the supplier
    logger.info(f'{leverancier} kent de volgende leveranties:')
    for  supply in supplies:
        logger.info(f' - {supply}')

    # and ask which supply should be removed
    leverantie = ''
    while leverantie not in supplies:
        logger.info('Kies de levering rapportageperiode van deleverantie die u wilt vernietigen (hoofdlettergevoelig): ')
        prompt = 'Levering rapportageperiode: '
        leverantie = input(prompt)
        logger.debug(f'{prompt}{leverantie}')

        if leverantie not in supplies:
            logger.error(f'*** geen bestaande leverantie voor {leverancier}: "{leverantie}"')

    # while

    # prepare SQL statements and run these thru simple_table
    logger.info(f'leverancier: {leverancier}, leverantie: {leverantie}')
    table_names = dc.get_table_names(project_name, leverancier, 'data')
    for table_name in table_names.keys():

        if table_name != dc.TAG_TABLE_META:
            name = table_names[table_name]
            sql = f'DELETE FROM {data_server_config["POSTGRES_SCHEMA"]}.{name}\n' \
                  f'WHERE levering_rapportageperiode = \'{leverantie}\'\n' \
                   'RETURNING levering_rapportageperiode;\n'

            result = st.row_count(sql, sql_server_config = data_server_config)
            logger.info(f'{result} records vernietigd for {data_server_config["POSTGRES_SCHEMA"]}.{name}')
    # for

    cpu = time.time() - cpu
    logger.info('')
    logger.info(f'[Ready {appname["name"]} in {cpu:.02f} seconds]')
    logger.info('')

    return


if __name__ == '__main__':
    # read commandline parameters to create log_file from
    cli, args = dc.read_cli()

    # create logger in project directory
    log_file = os.path.join(args.project, 'logs/' + cli['name'] + '.log')
    logger = dc.create_log(log_file, level = 'DEBUG')

    # go
    dido_remove('Removing a delivery')
