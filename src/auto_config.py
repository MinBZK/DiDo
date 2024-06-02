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


def create_config_file(projects: list, config_dir: str, supplier_id):
    auto_dir = os.path.join(config_dir, supplier_id)
    config_snippet_file = os.path.join(auto_dir, 'project_config.yaml')

    with open(config_snippet_file, 'r') as infile:
        snippet = infile.read()

    configs = f'  {supplier_id}:\n'

    for project in projects:
        project_cfg = snippet.format(**project)
        configs += project_cfg + '\n'

    total_config_file = os.path.join(auto_dir, 'total_config.yaml')
    with open(total_config_file, 'r') as infile:
        snippet = infile.read()

    specifiers = {'suppliers': configs}
    total = snippet.format(**specifiers)

    with open(os.path.join(config_dir, 'config.yaml'), 'w') as outfile:
        outfile.write(total)

    return

### create_config_file ###


def create_meta_files(projects: pd.DataFrame,
                      supplier_id: str,
                      config_dir: str,
                      schema_dir: str,
                     ):
    snippet_dir = os.path.join(config_dir, supplier_id)
    config_snippet_file = os.path.join(snippet_dir,
                                      'meta_file_snippet.csv')

    with open(config_snippet_file, 'r') as infile:
        snippet = infile.read()

    for project in projects:
        project_meta = snippet.format(**project)
        _, fn, ext = dc.split_filename(project['filename'])
        fn = fn.replace('_S_', '_')
        meta_filename = os.path.join(schema_dir, fn + '.meta.csv')

        with open(os.path.join(config_dir, meta_filename), 'w') as outfile:
            outfile.write(project_meta)

    return

### create_meta_files ###


def create_code_bronbestand(schema_dir: str,
                            ref_config: dict,
                            supplier_name: str,
                            # data_dict_path: str,
                            # frequency: str,
                            # decimal: str,
                           ):

    # fetch relevant info from the ref_config secxtion
    dd_path = dc.get_par(ref_config, 'DATA_DICT_PATH', schema_dir).strip()
    freq = dc.get_par(ref_config, 'FREQUENCY', 'Q').upper().strip()
    decimal = dc.get_par(ref_config, 'DECIMAL', '.').strip()
    regex = dc.get_par(ref_config, 'HEADER_PATTERN', None).strip()
    pattern = re.compile(regex)

    # get the header files from s3
    # files = s3_helper.s3_command_ls_return_fullpath(folder=dd_path)
    files = dc.get_files_from_dir(dd_path)

    # select those that conform to header_pattern
    result = []
    logger.info('Selecting files that conform to header pattern')
    for file in files:
        _, fn, ext = dc.split_filename(file)
        if pattern.match(fn):
            result.append(fn + ext)
            logger.debug(fn + ext)

    logger.info(f'{len(files)} found, {len(result)} files conform to pattern')
    logger.info('')

    codename_list = []
    for filename in result:
        # test if filename is directory (ends in '/'), if so, ignore
        if filename[-1] == '/':
            continue

        _, fn, ext = dc.split_filename(filename)
        splits = fn.split('_', 3)
        schema_name = fn.replace('_S_', '_')
        pname = splits[3].translate({ord(c): None for c in '_'}).strip()
        data_dict_file = os.path.join(dd_path, fn + ext)
        project = {'code': splits[2],
                   'supplier_name': supplier_name.strip(),
                   'project_name': splits[2].lower().strip(), # pname.lower(),
                   'data_dict_file': data_dict_file,
                   'frequency': freq,
                   'decimal': decimal,
                   'n_records': 1,
                   'schema_filename': schema_name,
                   'filename': fn + ext,
                  }
        codename_list.append(project)

    return codename_list

### create_code_bronbestand ###


def ref_config(config_dict: dict):
    # # get the database server definitions
    # db_servers = config_dict['SERVER_CONFIGS']

    # get project environment
    root_dir = config_dict['ROOT_DIR']
    project_dir = config_dict['PROJECT_DIRECTORY']
    config_dir = os.path.join(project_dir, 'config')
    leveranciers = config_dict['SUPPLIERS']

    # just * means process all
    suppliers_to_process = config_dict['SUPPLIERS_TO_PROCESS']
    if suppliers_to_process == '*':
        suppliers_to_process = leveranciers.keys()

    # process each supplier
    for leverancier_id in suppliers_to_process:
        dc.subheader(f'Supplier {leverancier_id}', '=')

        # create schema_dir
        schema_dir = os.path.join(root_dir, 'schemas', leverancier_id)

        # read auto config file
        configfile = os.path.join(config_dir, leverancier_id, 'auto_config.yaml')

        with open(configfile, encoding = 'utf8', mode = "r") as infile:
            auto_config = yaml.safe_load(infile)

        df = create_code_bronbestand(
            schema_dir = schema_dir,
            ref_config = auto_config,
            supplier_name = leverancier_id,
        )

        logger.info(f'{len(df)} projects will be processed for config')
        create_config_file(df, config_dir, leverancier_id)
        create_meta_files(df, leverancier_id, config_dir, schema_dir)

    # for -- supplier

    return

### ref_config ###


def main():
    cpu = time.time()

    # read commandline parameters
    appname, args = dc.read_cli()

    # read the configuration file
    config = dc.read_config(args.project)

    # display banner
    dc.display_dido_header('Creating Tables and Documentation', config)

    # create the tables
    ref_config(config)

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