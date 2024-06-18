import os
import re
import sys
import json
import time
import locale
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import sqlalchemy

import dido_common as dc
import simple_table as st

# pylint: disable=bare-except, line-too-long, consider-using-enumerate
# pylint: disable=logging-fstring-interpolation, too-many-locals
# pylint: disable=pointless-string-statement, consider-using-dict-items

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
    # Initialize info record
    info = {'table': 'Table exists',
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
        if table_name == tables_name[dc.TAG_TABLE_SCHEMA]:
            query_result = st.sql_select(
                table_name = table_name,
                columns = 'DISTINCT levering_rapportageperiode, count(*) ',
                groupby = 'levering_rapportageperiode ORDER BY levering_rapportageperiode',
                verbose = False,
                sql_server_config = server_config
            )
            if len(query_result) > 0:
                info['deliveries'] = query_result

        elif table_name == tables_name[dc.TAG_TABLE_DELIVERY]:
            query_result = st.sql_select(
                table_name = table_name,
                columns = 'DISTINCT levering_rapportageperiode',
                verbose = False,
                sql_server_config = server_config
            )

    except sqlalchemy.exc.ProgrammingError:
        info['table'] = '*Table does not exist*'

    return info


def get_data_table_names(project_name: str, leverancier: str) -> dict:
    """ Get table names of project for specific supplier.

    Args:
        project_name (str): Name of project
        leverancier (str): Name of leverancier

    Returns:
        dict: dictionary with all data table names
    """
    table_names = dc.get_table_names(project_name, leverancier, 'data')

    return table_names


def get_leveranties(suppliers: dict, supplier_name: str, server_config: dict) -> dict:
    """ get all leveranties for this supplier

    Args:
        project_name (str): Name of project
        leverancier (str): Name of leverancier
        server_config (dict): Database access properties

    Returns:
        dict: for all suppliers and data table a dict of all supplies
    """

    supplier_info = {}
    supplier = suppliers[supplier_name]
    for project_key in supplier.keys():
        tables_info = {}
        project = supplier[project_key]

        tables_name = get_data_table_names(project_key, supplier_name)

        for key in tables_name.keys():
            info = get_info(tables_name[key], tables_name, server_config)
            # info['project'] = project_key
            tables_info[key] = info

        # for

        supplier_info[project_key] = tables_info

    # for

    # print(supplier_info)

    return supplier_info


def display_leveranciers(leveranciers: dict, data_server_config: dict):
    """ Displays the supplies of of specific supplier

    Args:
        leveranties (dict): Dictionary of all supplies (by get_leveranties)
        supplier (str): Name of supplier to request list of supplies from
    """
    logger = logging.getLogger()

    logger.info('')
    logger.info(f'[Overzicht van leveranciers]')

    leverancier_met_data = []

    for key in leveranciers.keys():
        supplier = leveranciers[key]
        for project_name in supplier.keys():
            leveranties = get_leveranties(supplier, data_server_config)
            if leveranties['schema']['records'] > 0:
                leverancier_met_data.append(key)

                logger.info(f' - {key}')

    return leverancier_met_data, leveranciers.keys()


def display_leveranties(leveranties: dict, supplier: str):
    """ Displays the supplies of of specific supplier

    Args:
        leveranties (dict): Dictionary of all supplies (by get_leveranties)
        supplier (str): Name of supplier to request list of supplies from
    """
    logger = logging.getLogger()

    logger.info('')
    logger.info(f'[Leverancier: {supplier}]')
    # print(leveranties)

    for project_key in leveranties.keys():
        print(f'==> Supplier: {supplier}, Project: {project_key}')
        for key in leveranties[project_key].keys():
            info = leveranties[project_key][key]
            exst = 'exists' in info['table']
            if exst:
                logger.info(f' - {key}: {info["table"]}, ' \
                            f'{info["records"]} records')
                if info['table'][0] != '*':
                    if info['deliveries'] is not None:
                        logger.info(f'    Levering rapportageperiode     Aantal')
                        for index, row in info['deliveries'].iterrows():
                            logger.info(f'{row["levering_rapportageperiode"]:>30s} {row["count"]:10d}')

            else:
                logger.info(f'*No tables present*')

            # if
        # for
        logger.info('')
    # for

    return


def show_database(title: str, config: dict):
    logger = logging.getLogger()

    logger.info(title)
    logger.info(f'Server:   {config["POSTGRES_HOST"]}')
    logger.info(f'Port:     {config["POSTGRES_PORT"]}')
    logger.info(f'Database: {config["POSTGRES_DB"]}')
    logger.info(f'Schema:   {config["POSTGRES_SCHEMA"]}')
    logger.info(f'User:     {config["POSTGRES_USER"]}')
    logger.info('')

    return

### show_database ###


def dido_list(header: str = None):
    cpu = time.time()

    logger = logging.getLogger()

    logger.info('')

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
    root_dir = config_dict['ROOT_DIR']
    work_dir = config_dict['WORK_DIR']
    leveranciers = config_dict['SUPPLIERS']

    # if there is no supplier that received any supply, there is nothing to remove.
    # The program terminates
    # n_suppliers, suppliers = display_leveranciers(
    #     leveranciers = leveranciers,
    #     data_server_config = data_server_config,
    # )

    if len(leveranciers) < 1:
        logger.info('')
        logger.warning('Er zijn geen leveranciers of leveranties.')

        sys.exit()

    else:
        for leverancier in leveranciers:
            leveranties = get_leveranties(
                suppliers = leveranciers,
                supplier_name = leverancier,
                server_config = data_server_config,
            )
            display_leveranties(leveranties, leverancier)

        # for

    #if

    cpu = time.time() - cpu
    logger.info('')
    logger.info(f'[Ready {appname["name"]} in {cpu:.0f} seconds]')
    logger.info('')

    return

### dido_list ###


if __name__ == '__main__':
    # read commandline parameters to create log_file from
    cli, args = dc.read_cli()

    # create logger in project directory
    log_file = os.path.join(args.project, 'logs/' + cli['name'] + '.log')
    logger = dc.create_log(log_file, level = 'DEBUG')

    # go
    dido_list('Listing suppliers')
