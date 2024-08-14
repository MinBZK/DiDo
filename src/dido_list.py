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


def display_leveranciers(leveranciers: dict, level = 2):
    """ Displays information on all suppliers found in config.yaml

    Args:
        leveranciers (dict): dictionary with leveranciers information
        level (int, optional): level to which information should be displayed.
            2 means suppliers and projects, 3 = some extra information on tables
            and deliveries, 4 = detailed info on tables and deliveries.
             Defaults to 2.
    """
    logger = logging.getLogger()

    logger.info('')
    logger.info(f'[Overview of suppliers and their projects]')

    leverancier_met_data = []

    for leverancier_naam, leverancier_info in leveranciers.items():
        logger.info(f'Supplier: {leverancier_naam}')
        if level > 1:
            for project_name, project_info in leverancier_info.items():
                logger.info(f' * {project_name}')

                if level > 2:
                    for level3_name, level3_info in project_info.items():
                        if level == 3:
                            logger.info(f'    + {level3_name}: {level3_info}')
                        else:
                            logger.info(f'    + {level3_name}')

                            if level > 3:
                                for level4_name, level4_info in level3_info.items():
                                    logger.info(f'      - {level4_name}: {level4_info}')
                                # for
                            # if
                        # if
                    # for
                # if
            # for
        # if
    # for

    return

### display_leveranciers ###


def dido_list(header: str = None):
    cpu = time.time()

    logger = logging.getLogger()

    logger.info('')

    # read commandline parameters
    appname, args = dc.read_cli()

    # read the configuration file
    config_dict = dc.read_config(args.project)
    dc.display_dido_header(header, config_dict)
    delivery_filename = args.delivery

    delivery_config = dc.read_delivery_config(
        project_path = config_dict['PROJECT_DIR'],
        delivery_filename = args.delivery,
    )

    # get the database server definitions
    db_servers = config_dict['SERVER_CONFIGS']
    odl_server_config = db_servers['ODL_SERVER_CONFIG']
    data_server_config = db_servers['DATA_SERVER_CONFIG']
    foreign_server_config = db_servers['FOREIGN_SERVER_CONFIG']

    # get project environment
    root_dir = config_dict['ROOT_DIR']
    work_dir = config_dict['WORK_DIR']
    leveranciers = config_dict['SUPPLIERS']

    dc.show_database(
        server_config = data_server_config,
        pfun = logger.info,
        title = 'Dido List database',
    )
    supplier_info = dc.get_suppliers(leveranciers)
    supplier_info = dc.add_deliveries_to_suppliers(
        suppliers_dict = supplier_info,
        delivery_dict = delivery_config['DELIVERIES']
    )
    supplier_info = dc.add_table_info_to_deliveries(
        suppliers_dict = supplier_info,
        server_config = data_server_config
    )

    if len(leveranciers) < 1:
        logger.info('')
        logger.warning('No suppliers nor deliveries')

    else:
        display_leveranciers(supplier_info)
        dc.display_leveranties(supplier_info)

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
