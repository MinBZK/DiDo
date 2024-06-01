"""
Dido-begin prepareert een working directory op basis van een root directory
en configuratiefiles.

Het uitgangspunt is dat dido alles kan reconstrueren op basis van de gegevens
in de root directory en de configuratiefile, samen met de data. In de schemas en
docs directory van de root directory staat informatie die gelezen, bewerkt en
gekopieerd wordt naar de working directory.

Als de schemafiles ok zijn bevonden dan kunnen ze worden verwerkt met dido-create,
anders moeten de schemafiles in de root directory worden aangepast. Wat in de
work directrory staat wordt altijd overschreven door dido_begin.
"""
import os
import re
import sys
import time
import yaml
import shutil
import pandas as pd

from datetime import datetime

# Don't forget to set PYTHONPATH to your python library files
# export PYTHONPATH=/path/to/dido/helpers/map
# import api_postcode
import dido_common as dc
import simple_table as st
import s3_helper

from dido_common import DiDoError
from dido_list import dido_list

# pylint: disable=bare-except, line-too-long, consider-using-enumerate
# pylint: disable=logging-fstring-interpolation, too-many-locals
# pylint: disable=pointless-string-statement, consider-using-dict-items

# show all columns of dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


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

    result = result[result['table_name'].str.contains('schema_data')]
    tables = result['table_name'].tolist()

    return tables

### get_table_names ###


def find_project_name(string_dict: list,
                      n_splits: int,
                      project_name: str,
                      periode: str,
                     ):

    for string in string_dict[periode]:
        _, text, _ = dc.split_filename(string)
        splits = text.split('_', n_splits)
        project_part = splits[n_splits - 1].replace('_', '').strip().lower()
        if project_part == project_name:
            return string

    # for

    return ''

### find_project_name ###


def collect_files_and_headers(config: dict,
                              periodes: list,
                              table_list: list,
                              pad: str,
                             ):
    file_dict = {}
    header_dict = {}
    for periode in periodes:
        regex = config['FILE_PATTERN']
        file_pattern = re.compile(regex)
        regex = config['HEADER_PATTERN']
        header_pattern = re.compile(regex)

        path = os.path.join(pad, periode) + '/'
        logger.info(f'Fetching files from {path}')
        files = s3_helper.s3_command_ls_return_fullpath(folder = path)

        logger.info(f'Redirecting {len(files)} files')
        data_files = []
        header_files = []
        for file in files:
            _, fn, ext = dc.split_filename(file)
            if file_pattern.match(fn):
                data_files.append(fn + ext)
            elif header_pattern.match(fn):
                header_files.append(fn + ext)
            # if
        # for

        file_dict[periode] = data_files
        header_dict[periode] = header_files

        logger.info(f'{periode}: {len(data_files)} data files and '
                    f'{len(header_files)} header files of a total '
                    f'of {len(files)} files.')

    # for

    return file_dict, header_dict

### collect_files_and_headers ###


def collect_project_info(config: dict,
                         periodes: list,
                         table_list: list,
                         pad: str,
                        ):

    data_dict, header_dict = collect_files_and_headers(
        config = config,
        periodes = periodes,
        table_list = table_list,
        pad = pad,
    )

    projects = {}
    # table list contains expected data files, find associated filenames
    for table_name in table_list:
        splits = table_name.split('_')
        project_name = splits[1].strip().lower()
        delivery = {}
        for periode in periodes:
            project = {}
            project['table_name'] = table_name
            project['project_name'] = project_name

            filename = find_project_name(data_dict, 2, project_name, periode)
            headername = find_project_name(header_dict, 3, project_name, periode)

            if len(filename) < 0:
                logger.warning(f'!!! Project without file name: {project_name}')
            elif len(headername) < 0:
                logger.warning(f'!!! Project without header file: {project_name}')
            else:
                logger.info(filename)
                splits = filename.split('_', 2)
                leveringsdatum = splits[0].strip()
                code_bronbestand = splits[1].upper().strip()
                project['data_name'] = os.path.join(pad, periode, filename)
                project['header_name'] = os.path.join(pad, periode, headername)
                project['levering_rapportageperiode'] = periode
                project['code_bronbestand'] = code_bronbestand
                project['leveringsdatum'] = leveringsdatum

                data_dict[periode].remove(filename)
                header_dict[periode].remove(headername)

            # if

            delivery[periode] = project

        # for

        projects[project_name] = delivery
    # for

    for periode in periodes:
        if len(data_dict[periode]) > 0:
            logger.warning(f'!!! Data files not in tables: {data_dict[periode]}')
        if len(header_dict[periode]) > 0:
            logger.warning(f'!!! Header files not in tables: {header_dict[periode]}')

    return projects

### collect_project_info ###


def create_deliveries(projects: dict,
                      supplier_id: str,
                      periodes: list,
                      template: str,
                      delivery_template: str,
                      filename: str,
                     ):
    deliveries = ''
    for project_key in projects.keys():
        deliveries += '    ' + project_key + ':\n'
        project = projects[project_key]
        for periode in periodes:
            proj_periode = project[periode]
            filled = template.format(**proj_periode)
            deliveries += filled

    variables = {'deliveries': deliveries,
                 'supplier_name': supplier_id
                }

    delivery_output = delivery_template.format(**variables)

    with open(filename, 'w') as outfile:
        outfile.write(delivery_output)

    return

### create_deliveries ###


def ref_deliver(config_dict: dict):
    # get the database server definitions
    db_servers = config_dict['SERVER_CONFIGS']

    # get project environment
    root_dir = config_dict['ROOT_DIR']
    work_dir = config_dict['WORK_DIR']
    leveranciers = config_dict['SUPPLIERS']

    # just * means process all
    suppliers_to_process = config_dict['SUPPLIERS_TO_PROCESS']
    if suppliers_to_process == '*':
        suppliers_to_process = leveranciers.keys()

    # just * means process all
    suppliers_to_process = config_dict['SUPPLIERS_TO_PROCESS']
    if suppliers_to_process == '*':
        suppliers_to_process = leveranciers.keys()

    projects = []
    for leverancier_id in suppliers_to_process:
        dc.subheader(f'Supplier {leverancier_id}', '=')

        # read auto config file
        config_dir = os.path.join(config_dict['PROJECT_DIRECTORY'],
                                  'config', leverancier_id)

        # auto_config = dc.get_config_file(config_dir, 'auto_config.yaml')
        auto_delivery = dc.get_config_file(config_dir, 'auto_delivery.yaml')
        periodes = auto_delivery['LEVERING_RAPPORTAGEPERIODE']

        filename = os.path.join(config_dir, 'project_delivery.yaml')
        with open(filename, 'r') as infile:
            template = infile.read()

        delivery_template_filename = \
            os.path.join(config_dir, 'total_delivery.yaml')
        with open(delivery_template_filename, 'r') as infile:
            delivery_template = infile.read()

        pad = os.path.join(auto_delivery['DATA_PATH'])
        table_list = get_table_names(leverancier_id, db_servers)

        projects = collect_project_info(auto_delivery, periodes, table_list, pad)

        logger.info(f'A total of {len(projects)} files will be delivered')
        filename = os.path.join(root_dir, 'data', config_dict['DELIVERY_FILE'])
        create_deliveries(
            projects = projects,
            supplier_id = leverancier_id,
            periodes = periodes,
            template = template,
            delivery_template = delivery_template,
            filename = filename,
        )

    # for

    return

### ref_deliver ###


def main():
    cpu = time.time()

    # read commandline parameters
    appname, args = dc.read_cli()

    # read the configuration file
    config = dc.read_config(args.project)
    config['DELIVERY_FILE'] = args.delivery

    # display banner
    dc.display_dido_header('Auto Delivery Generator', config)

    # create the tables
    ref_deliver(config)

    # quit with goodbye message
    cpu = time.time() - cpu
    logger.info('')
    logger.info(f'[Ready {appname["name"]} in {cpu:.0f} seconds]')
    logger.info('')

    return

### main ###


if __name__ == '__main__':
    # read commandline parameters to create log_file from
    cli, args = dc.read_cli()

    # create logger in project directory
    log_file = os.path.join(args.project, 'logs/' + cli['name'] + '.log')
    logger = dc.create_log(log_file, level = 'DEBUG')

    # go
    main()