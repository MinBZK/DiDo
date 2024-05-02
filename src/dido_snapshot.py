import os
import re
import sys
import json
import time
import logging
import zipfile
from datetime import datetime
import numpy as np
import pandas as pd

import common
import mutate
import simple_table

from dido_common import DiDoError

from dido_common import read_cli, read_config, load_odl_table, load_schema, \
    display_dido_header, get_server, get_table_names, get_supplier_projects, \
    get_current_delivery_seq, get_par, load_parameters, load_sql, load_pgpass, \
    get_par, get_par_par, create_data_types

from dido_common import EXTRA_TEMPLATE, \
    TAG_TABLE_SCHEMA, TAG_TABLE_META, TAG_TABLE_DELIVERY, TAG_TABLE_QUALITY, \
    TAG_TABLES, TAG_PREFIX, TAG_SUFFIX, TAG_SCHEMA, TAG_DATA, \
    DIR_SCHEMAS, DIR_DOCS, DIR_DONE, DIR_TODO, DIR_SQL, \
    VALUE_OK, VALUE_NOT_IN_LIST, VALUE_MANDATORY_NOT_SPECIFIED, \
    VALUE_NOT_BETWEEN_MINMAX, VALUE_OUT_OF_REACH, VALUE_IMPROBABLE, \
    VALUE_WRONG_DATATYPE, VALUE_HAS_WRONG_FORMAT, VALUE_NOT_CONFORM_RE, \
    ODL_RECORDNO, ODL_CODE_BRONBESTAND, ODL_LEVERING_FREK, \
    ODL_DATUM_BEGIN, ODL_DATUM_EINDE, ODL_SYSDATUM, ODL_DELIVERY_DATE

# show all columns of dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# raise exceptions on all numpy warnings
np.seterr(all = 'raise')


def dido_snapshot(header: str):
    cpu = time.time()

    logger = logging.getLogger()

    display_dido_header(header)

    # read commandline parameters
    appname, args = read_cli()

    # read the configuration file
    config_dict = read_config(args.project)

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
    # snapshot_options = get_par(config_dict, 'SNAPSHOTS', {'zip': 'no', 'destroy_todo': 'no'})

    create_zip = get_par_par(config_dict, 'SNAPSHOTS', 'zip', False)
    destroy_todo = get_par_par(config_dict, 'SNAPSHOTS', 'destroy_todo', False)

    # select which suppliers to process
    suppliers_to_process = get_par(config_dict, 'SUPPLIERS_TO_PROCESS', '*')

    # just * means process all
    if suppliers_to_process == '*':
        suppliers_to_process = leveranciers.keys()

    if create_zip:
        # copy table_desc as a template for information each leverancier has to deliver
        for leverancier_id in suppliers_to_process:
            logger.info('')
            logger.info(f'=== {leverancier_id} ===')
            logger.info('')

            # count the number of deliveries and fetch sup[plier and delivery accordingly
            count = get_current_delivery_seq(project_name, leverancier_id, data_server_config)
            delivery_seq = count + 1
            logger.info(f'Current delivery is {delivery_seq}')
            leverancier_config, deliveries = get_supplier_projects(config_dict, leverancier_id, delivery_seq)

            if len(deliveries) > 0:
                logger.info('Delivery configs supplied in config.yaml (x = chosen)')
                for key in deliveries.keys():
                    logger.info(f" - {deliveries[key]['delivery_naam']} " \
                                f"{deliveries[key]['delivery_keus']}")

                # for
                logger.info('')
            # if

            nw = datetime.now()
            today = f'{nw.year}-{nw.month:02d}-{nw.day:02d}_{nw.hour}.{nw.minute:02d}.{nw.second:02d}'
            basename = f'{project_name}_{leverancier_id}'
            basename = common.change_column_name(basename)
            basename = (today + '_' + basename + '.zip')
            zip_filename = os.path.join(work_dir, DIR_DONE, basename)

            # add config file
            snapshots: list = [os.path.join(args.project, 'config', 'config.yaml')]
            arcnames: list = ['config/config.yaml']

            # create empty destroy list
            to_be_destroyed: list = []

            # add schema and meta file
            snapshots.append(os.path.join(work_dir, DIR_SCHEMAS, leverancier_id, leverancier_config['schema_file'] + '.schema.csv'))
            snapshots.append(os.path.join(work_dir, DIR_SCHEMAS, leverancier_id, leverancier_config['schema_file'] + '.meta.csv'))
            arcnames.append(os.path.join(DIR_SCHEMAS, leverancier_id, leverancier_config['schema_file'] + '.schema.csv'))
            arcnames.append(os.path.join(DIR_SCHEMAS, leverancier_id, leverancier_config['schema_file'] + '.meta.csv'))

            # retrieve all files from current supplier in the todo directory

            todo_dir = os.path.join(work_dir, DIR_TODO, leverancier_id)
            todo_files = [f for f in os.listdir(todo_dir) if os.path.isfile(os.path.join(todo_dir, f))]
            for f in todo_files:
                todo_file = os.path.join(todo_dir, f)
                snapshots.append(todo_file)
                arcnames.append(os.path.join(DIR_TODO, leverancier_id, f))

                if destroy_todo:
                    to_be_destroyed.append(todo_file)

                # if
            # for

            # below one line of code will create a 'Zip' in the current working directory
            with zipfile.ZipFile(zip_filename, 'w', allowZip64 = True) as zip_archive:
                logger.info(f'Creating snapshots to {zip_filename}')

                for i in range(len(snapshots)):
                    snapshot = snapshots[i]
                    arcname = arcnames[i]
                    logger.info(f'Snapshotting: {os.path.basename(snapshot)}')
                    zip_archive.write(snapshot, arcname)

                # for

            # with

            correct_test: bool = False
            with zipfile.ZipFile(zip_filename, 'r', allowZip64 = True) as zip_archive:
                test = zip_archive.testzip()
                if test is None:
                    logger.info('Zipfile correctly written')
                    correct_test = True

                else:
                    logger.error(f'Errors while zipping, first erroneous file is {test}')
                # if
            # with

            if not correct_test:
                logger.info(f'No further processing for {leverancier_id}. Files marked for erasure will remain untouched')
                continue

            # delete data files if that was requested
            if destroy_todo and len(to_be_destroyed) > 0 and correct_test:
                logger.info('')
                logger.info('Files to be erased')

                for filename in to_be_destroyed:
                    logger.info(filename)
                    os.remove(filename)
                # for
            # if
        # for
    else:
        logger.info('[No snapshots will be created as SNAPSHOTS:zip is False]')

    # if

    cpu = time.time() - cpu
    logger.info('')
    logger.info(f'[Ready {appname["name"]} in {cpu:.02f} seconds]')
    logger.info('')

    return


if __name__ == '__main__':
    # read commandline parameters
    cli, args = read_cli()

    # create logger in project directory
    log_file = os.path.join(args.project, 'logs', cli['name'] + '.log')
    logger = common.create_log(log_file, level = 'DEBUG')

    # go
    dido_snapshot('Snapshotting Data')
