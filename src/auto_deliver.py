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
# import yaml
# import shutil
import pandas as pd

# from datetime import datetime

# Don't forget to set PYTHONPATH to your python library files
# export PYTHONPATH=/path/to/dido/helpers/map
# import api_postcode
import dido_common as dc
import simple_table as st
import s3_helper

from dido_common import DiDoError

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


def find_previous_filename(config: dict,
                        header: bool,
                           project_name: str,
                           periode: str,
                           table_list: list,
                           pad: str,
                           servers: dict,
                          ):
    new_period = dc.compute_periods(periode, -1, servers)

    data_dict, header_dict = collect_files_and_headers(
        config = config,
        periodes = [new_period],
        table_list = table_list,
        pad = pad,
    )

    if header:
        filename = find_project_name(header_dict, 3, project_name, new_period)
    else:
        filename = find_project_name(data_dict, 2, project_name, new_period)

    return filename, new_period

### find_previous_filename ###

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
        logger.debug(f'Fetching files from {path}')
        files = dc.get_files_from_dir(path)

        logger.debug(f'Redirecting {len(files)} files')
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
                         servers: dict,
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

            # find data and header file associated with project name
            filename = find_project_name(data_dict, 2, project_name, periode)
            headername = find_project_name(header_dict, 3, project_name, periode)
            true_period = periode

            # when there is no associated file found, look for it in
            # the previous period
            if len(filename) < 1 and config['USE_PREVIOUS_WHEN_OMITTED']:
                filename, true_period = find_previous_filename(
                    config = config,
                    header = False,
                    project_name = project_name,
                    periode = periode,
                    table_list = table_list,
                    pad = pad,
                    servers = servers,
                )

            if len(headername) < 1 and config['USE_PREVIOUS_WHEN_OMITTED']:
                headername, true_period = find_previous_filename(
                    config = config,
                    header = True,
                    project_name = project_name,
                    periode = periode,
                    table_list = table_list,
                    pad = pad,
                    servers = servers,
                )

            if len(filename) < 1:
                logger.warning(f'!!! Project without file name: {project_name}')
            elif len(headername) < 1:
                logger.warning(f'!!! Project without header file: {project_name}')
            else:
                logger.info(f'{filename} from {true_period}')
                splits = filename.split('_', 2)
                leveringsdatum = splits[0].strip()
                code_bronbestand = splits[1].upper().strip()
                project['data_name'] = os.path.join(pad, true_period, filename)
                project['header_name'] = os.path.join(pad, true_period, headername)
                project['levering_rapportageperiode'] = periode
                project['code_bronbestand'] = code_bronbestand
                project['leveringsdatum'] = leveringsdatum

                try:
                    data_dict[periode].remove(filename)
                except:
                    pass

                try:
                    header_dict[periode].remove(headername)
                except:
                    pass

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
    # enumerate over all projects
    for project_key in projects.keys():
        project = projects[project_key]
        proj_deliv = ''

        # and over all periods
        for periode in periodes:
            proj_periode = project[periode]

            # check if project has a data_name, if not: no data file
            if 'data_name' in proj_periode.keys():
                filled = template.format(**proj_periode)
                proj_deliv += filled

            else:
                logger.warning(f'!!! No data for project {project_key}/{periode}')

        # for

        # Only add delivery when project contains data
        if len(proj_deliv) > 0:
            deliveries += '    ' + project_key + ':\n' + proj_deliv

    # for

    # only when there are deliveries, write to delivery file
    if len(deliveries) > 0:
        variables = {'deliveries': deliveries,
                    'supplier_name': supplier_id
                    }

        delivery_output = delivery_template.format(**variables)

        with open(filename, 'w') as outfile:
            outfile.write(delivery_output)

    else:
        logger.info('')
        logger.error('*** Empty project directory: sure this is the correct one?')

    # if

    return

### create_deliveries ###


def ref_deliver(config_dict: dict):
    # get the database server definitions
    db_servers = config_dict['SERVER_CONFIGS']

    # new_period = dc.compute_periods('2023-Q2', 1, db_servers)
    # print(new_period)
    # new_period = dc.compute_periods('2023-Q2', 4, db_servers)
    # print(new_period)
    # new_period = dc.compute_periods('2023-M2', -15, db_servers)
    # print(new_period)
    # new_period = dc.compute_periods('2023-J', 3, db_servers)
    # print(new_period)
    # new_period = dc.compute_periods('2023-D2', 800, db_servers)
    # print(new_period)
    # new_period = dc.compute_periods('1001-A2', 368, db_servers)
    # print(new_period)
    # new_period = dc.compute_periods('1001-A2', -3, db_servers)
    # print(new_period)
    # new_period = dc.compute_periods('1001-D2', -368, db_servers)
    # print(new_period)

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

        projects = collect_project_info(
            config = auto_delivery,
            periodes = periodes,
            table_list = table_list,
            pad = pad,
            servers = db_servers,
        )

        logger.info('')
        logger.info(f'A total of {len(projects)} files will be delivered')
        logger.info('')

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