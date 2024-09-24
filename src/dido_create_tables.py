"""
Dido-begin prepareert een working directory op basis van een root directory
en configuratiefiles.

Het uitgangspunt i
s dat dido alles kan reconstrueren op basis van de gegevens
in de root directory en de configuratiefile, samen met de data. In de schemas en
docs directory van de root directory staat informatie die gelezen, bewerkt en
gekopieerd wordt naar de working directory.

Als de schemafiles ok zijn bevonden dan kunnen ze worden verwerkt met dido-create,
anders moeten de schemafiles in de root directory worden aangepast. Wat in de
work directrory staat wordt altijd overschreven door dido_create_tables.
"""
import os
import sys
import yaml
import time
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

###############################################################################
#
# Fase 1 - Read and check the schema files
#
###############################################################################

def fetch_schema_from_table(table_name: str, sql_server_config: dict) -> pd.DataFrame:
    """ fetch a schema from the database

    Args:
        table_name (str): name of the table to fetch
        sql_server_config (dict): server information

    Returns:
        pd.DataFrame: the fetched table as a Pandas DataFrame
    """
    result = st.get_structure(
        table_name,
        verbose = False,
        sql_server_config = sql_server_config,
    )

    return result

### fetch_schema_from_table ###


def merge_bootstrap_data(schema: pd.DataFrame, table_name: str, server_config: dict) -> pd.DataFrame:
    """  Merge bootstrap data with the schema data.

    The insertions use the pandas option to use fractional indices.

    Args:
        schema (pd.DataFrame): schema to be adjusted
        filename (str): data to be added

    Returns:
        pd.DataFrame: adjusted schema dataframe
    """

    # Read bootstrap data
    bootstrap_data = dc.load_odl_table(dc.EXTRA_TEMPLATE, server_config)

    # First increase the schema.index by 1
    schema.index = range(1, len(schema) + 1)

    # now use fractional indices to insert bootstrap data *before* the first element
    # check if the column names are present in the schema file
    # In that case, don't copy

    schema_columns = schema['kolomnaam'].tolist()

    # iterate over all rows in bootstrap data
    i = 0
    n = len(bootstrap_data) + 1

    for idx, row in bootstrap_data.iterrows():
        col_name = bootstrap_data.loc[idx, 'kolomnaam']

        # if columnnaam already exists, don't make a copy
        if col_name not in schema_columns:
            schema.loc[i / n] = row # col_name

        i += 1

    # for

    # sort rows in the right order, renumber the index and drop the old one
    schema = schema.sort_index().reset_index(drop = True)

    return schema

### merge_bootstrap_data ###


def substitute_vars(
    schema: pd.DataFrame,
    meta: pd.DataFrame,
    data_dict: dict,
    supplier_config: dict,
) -> pd.DataFrame:

    code_bronbestand = meta.loc['Code bronbestand', 'waarde']
    vars = {'supplier': supplier_config['supplier_id'],
            'code_bronbestand': code_bronbestand}

    for idx in schema.index:
        for col in schema.columns:
            value = schema.loc[idx, col]
            value = value.format(**vars)
            schema.loc[idx, col] = value

        # for
    # for

    return schema

### substitute_vars ###


def apply_schema_odl(template: pd.DataFrame,
                     schema: pd.DataFrame,
                     meta: pd.DataFrame,
                     data_dict: dict,
                     filename: str,
                     allowed_datatypes: list,
                     supplier_config: dict,
                    ) -> pd.DataFrame:
    """ Uses data and meta to fill out the ODL template

    Args:
        template (pd.DataFrame): ODL template to be filled out
        data (pd.DataFrame): data file, usually source analysis
        meta (pd.DataFrame): meta file for some important information

    Returns:
        pd.DataFrame: the filled out template
    """
    errors = False

    # check if template file exists
    schema = substitute_vars(schema, meta, data_dict, supplier_config)
    schema.columns = [col.strip().lower() for col in schema.columns]
    if template is None:
        errmsg = f'*** apply_schema_odl: Schema DataFrame is None. ' \
                  'ODL generatie kan daar niet zonder.'
        logger.error(errmsg)

        raise DiDoError(errmsg)

    # check if meta file exists
    if meta is None:
        errmsg = f'*** apply_schema_odl: Meta DataFrame is None. ' \
                  'ontbreekt. ODL generatie kan daar niet zonder.'
        logger.error(errmsg)

        raise DiDoError(errmsg)

    # check if kolomnaam exists as column name
    if 'kolomnaam' not in schema.columns:
        errmsg = f'* Schema "{filename}" *moet* "kolomnaam" bevatten. '
        errmsg += 'ODL generatie kan daar niet zonder'
        logger.error(errmsg)

        raise DiDoError(errmsg)

    # check if schemna columns names occur in template
    for col_name in schema.columns:
        if col_name not in template.columns:
            errmsg = f'* Schema "{filename}": kolomnaam "{col_name}" komt niet in '
            errmsg += '"bronbestand_attribuut_meta.csv" voor'
            logger.error(errmsg)
            errors = True

    # create a new dataframe with all required columns according to the template
    new_df = pd.DataFrame(columns=template.columns, index=schema.index, dtype=str)

    code_bbs = meta.loc['Code bronbestand', 'waarde']

    # set defaults
    new_df['avg_classificatie'] = 1
    new_df['veiligheid_classificatie'] = 1
    new_df['attribuut_datum_begin'] = meta.loc['Bronbestand datum begin', 'waarde']
    new_df['attribuut_datum_einde'] = meta.loc['Bronbestand datum einde', 'waarde']

    # namen die worden voorgedefinieerd en niet worden overgekopieerd
    names_to_skip = ['kolomnaam', 'code_attribuut', 'code_attribuut_sleutel', f'{dc.ODL_CODE_BRONBESTAND}']

    # generate codes and keys
    for idx, _ in new_df.iterrows():
        col_name = schema.loc[idx, 'kolomnaam']

        # if description is lacking, notify the user and flag it
        beschrijving = schema.loc[idx, 'beschrijving'].strip()
        if len(beschrijving) == 0:
            logger.warning(f'* Schema "{filename}", kolom "{col_name}", beschrijving ontbreekt.')
            schema.loc[idx, 'beschrijving'] = f'*** ERROR!!! NO DESCRIPTION FOR CULUMN {col_name}***'

        # if kolomnaam is omitted copy it from leverancier_kolomnaam
        if len(schema.loc[idx, 'kolomnaam'].strip()) == 0:
            if len(schema.loc[idx, 'leverancier_kolomnaam'].strip()) > 0:
                schema.loc[idx, 'kolomnaam'] = schema.loc[idx, 'leverancier_kolomnaam'].strip().lower()
            else:
                errors = True

        # ensure that kolomnaam is a postgres accepted column name
        new_name = dc.change_column_name(schema.loc[idx, 'kolomnaam'])

        # when no name could be created, create a random one
        if len(new_name) == 0:
            logger.error(f'* Schema {filename}, geen kolomnaam in regel {idx + 1}')

        # replace datatypes if necessary and possible
        dtyp = schema.loc[idx, 'datatype'].strip()
        if  len(dtyp) == 0 and \
            len(schema.loc[idx, 'leverancier_kolomtype'].strip()) > 0 and \
            data_dict is not None:

            sup_type = schema.loc[idx, 'leverancier_kolomtype'].strip()
            if sup_type in data_dict.keys():
                datatype = data_dict[sup_type]
                schema.loc[idx, 'datatype'] = datatype

        # if that does not help, flag it
        datatype = schema.loc[idx, 'datatype'].strip().lower()

        if len(datatype) == 0:
            schema.loc[idx, 'datatype'] = '*NONE*'

        # is datatype allowed?
        if datatype not in allowed_datatypes:
            logger.error(f'* Datatype not allowed: {datatype} in column {col_name}')
            errors = True

        # assign newly created column name to kolomnaam
        new_df.loc[idx, 'kolomnaam'] = new_name

        # create code attribuut
        code_atr = ''

        # check if it occurs in data.columns
        if 'code_attribuut' in schema.columns:
            code_atr = str(schema.loc[idx, 'code_attribuut']).strip()

        # not assign a numeric value?
        if len(code_atr) == 0:
            code_atr = f'{idx + 1:03d}'

        # assign to new_df
        new_df.loc[idx, 'code_attribuut'] = code_atr

        # same for code_attribuut_sleutel
        code_atr_key = code_bbs + code_atr
        if 'code_attribuut_sleutel' in schema.columns:
            code_atr_key = str(schema.loc [idx, 'code_attribuut_sleutel']).strip()

        if len(code_atr_key) == 0:
            code_atr_key = code_bbs + code_atr

        new_df.loc[idx, 'code_attribuut_sleutel'] = code_atr_key

        if 'positie' not in schema.columns or (isinstance(schema.loc[idx, 'positie'], str) and len(schema.loc[idx, 'positie']) == 0):
            new_df.loc[idx, 'positie'] = str(idx + 1)

        # assign rest of data values
        for col_name in schema.columns:
            if col_name not in names_to_skip:
                try:
                    value = schema.loc[idx, col_name].strip()
                except:
                    value = ''

                if len(value) > 0:
                    new_df.loc[idx, col_name] = value

            # if
        # for
    # for

    # some attributes are overruled by meta data
    new_df[f'{dc.ODL_CODE_BRONBESTAND}'] = code_bbs

    # convert all nan values to empty str
    new_df = new_df.fillna('')
    if errors:
        raise DiDoError(f'Schema "{filename}", fatal errors encountered. Programs stops.')

    datatypes = new_df.dtypes['positie']

    return new_df

### apply_schema_odl ###


def apply_meta_odl(meta: pd.DataFrame,
                   n_cols: int,
                   filename: str,
                   bootstrap_data: dict,
                   server_config: pd.DataFrame,
                  ) -> pd.DataFrame:
    """ Checks meta information

    Args:
        meta -- meta information as supplied by user
        n_cols -- number of columns in the data
        filename -- filename to be mentioned in error reporting

    Raises:
        DiDoError: when an error occurs raise this exception

    Returns:
        modified meta file
    """
    errors = False

    template = dc.load_odl_table(dc.META_TEMPLATE, server_config)

    # change meta row index values
    meta.index = [dc.change_column_name(i) for i in meta.index.tolist()]

    # test if name occurs in template column list
    for col_name in meta.index.tolist():
        if col_name not in template['kolomnaam'].tolist():
            logger.error(f'* Naam {col_name} geen metadata kolom')
            errors = True

    meta = meta.reset_index()
    meta.columns = ['attribuut', 'waarde']
    meta = meta.set_index('attribuut')

    if len(meta.loc[f'{dc.ODL_CODE_BRONBESTAND}', 'waarde'].strip()) == 0:
        errmsg = f'* Meta-attribuut "code_bronbestand" in "{filename}" ontbreekt of is niet '
        errmsg += 'gespecificeerd. Vraag dit aan bij Team DWH.'
        logger.error(errmsg)
        errors = True

    meta.loc['created_by', 'waarde'] = 'current_user'

    if len(meta.loc['bronbestand_beschrijving', 'waarde'].strip()) == 0:
        errmsg = f'* Meta-attribuut "bronbestand_beschrijving" in "{filename}" ontbreekt of is '
        errmsg += 'niet gespecificeerd. '
        errmsg += 'Bestanden met ontbrekende beschrijvingen worden niet geaccepteerd.'
        logger.error(errmsg)
        errors = True

    # Check for some rows if an integer is provided: if not assume 1
    int_rows = ['bronbestand_gemiddeld_aantal_records', 'bronbestand_aantal_attributen']
    for int_row in int_rows:
        if int_row in meta.index:
            result = meta.loc[int_row, 'waarde']
            if not result.isnumeric():
                result = 1

        else:
            result = 1

        meta.loc[int_row, 'waarde'] = result

    decimal = meta.loc['bronbestand_decimaal', 'waarde'].strip()
    if decimal not in ['.', ',']:
        errmsg = f'! Het Meta-attribuut "bronbestand_decimaal" in "{filename}" ontbreekt of is niet'
        errmsg += ' correct gespecificeerd. Dit moet "." of "," zijn; "." wordt aangenomen.'
        logger.warning(errmsg)
        meta.loc['bronbestand_decimaal', 'waarde'] = '.'

    if 'bronbestand_expiratie_datum' not in meta.index or \
       len(meta.loc['bronbestand_expiratie_datum', 'waarde'].strip()) == 0:
        meta.loc['bronbestand_expiratie_datum', 'waarde'] = bootstrap_data['END_OF_WORLD']

    if errors:
        raise DiDoError(f'Fout in "{filename}", file aanpassen en opnieuw inleveren.')

    meta.loc['bronbestand_aantal_attributen', 'waarde'] = str(n_cols)
    meta.loc['sysdatum', 'waarde'] = dc.iso_cet_date(datetime.now())

    logger.debug(f'Meta info after having been interpreted:\n{meta}')

    return meta

### apply_meta_odl ###


def create_folder(path: str, folder_to_create: str = ''):
    """and subfolders

    Args:
        path --
        folder_to_create --
    """
    folder_to_create = os.path.join(path, folder_to_create)

    os.makedirs(folder_to_create, exist_ok = True)

    return


def create_workdir_structure(config_vars: dict, server_config):
    """ Fetch the work subdirectories from the ODL database parameter table
        and create the subdirectories

    Args:
        config_vars -- project configuration
    """
    # load the data from the parameters table
    bootstrap_data = dc.load_parameters()

    # get the workdirs string
    subdirectories = bootstrap_data['WORKDIR_STRUCTURE']

    # convert this to a real list using json.loads()
    program_vars = [subdir for subdir in subdirectories.split(',')]

    # Create the subdirectories in the working directory
    work_dir = config_vars['WORK_DIR']
    for subdir in subdirectories.split(','):
        subdir = subdir.strip()
        for supplier in config_vars['SUPPLIERS']:
            create_folder(os.path.join(work_dir, subdir, supplier))

    return

### create_workdir_structure ###


def get_pdirekt_header_file(header_file: str, root_directory: str, supplier_id: str):
    """ Looks for datafiles in the root/data directory

    Args:
        root_directory (str): name of the root directory
        suppliers (_type_): list of suppliers

    Returns:
        dict: dictionary for each supplier with data files
    """

    # check if first character in data_file is a /
    if header_file.startswith('/'):
        # absolute path
        # pad, fn, ext = dc.split_filename(data_file) # Where is data_file coming from????
        pad, fn, ext = dc.split_filename(header_file)
        filename = os.path.join(root_directory, fn + ext)
        shutil.copy2(header_file, filename)


    # check id data reside on s3 bucket
    elif header_file.startswith('s3://'):
        server = 's3'
        pad, fn, ext = dc.split_filename(header_file)
        # server_path = header_file
        fn = fn + ext
        filename = os.path.join(root_directory, 'data', supplier_id, fn)

        s3_helper.s3_command_get_file(
            download_to = filename,
            filepath_s3 = header_file, # server_path,
            force_overwrite = True,
        )

    else:
        # path relative to root_directory/data
        pad = os.path.join(root_directory, 'data', supplier_id)
        filename = os.path.join(pad, header_file) # data_file ???

    # if

    # load the header into a dataframe
    # headers = pd.read_csv(
    #     filename,
    #     sep = ';',
    #     dtype = str,
    #     keep_default_na = False,
    #     skiprows = 5,
    #     engine = 'c',
    #     encoding = 'utf8',
    # )

    return filename

### get_pdirekt_header_file ###


def create_schema_from_pdirekt_datadict(filename: str,
                                        root_directory: str,
                                        supplier_id: str,
                                        data_dict: dict,
                                       ):

    # Process the header file
    filename = get_pdirekt_header_file(
        header_file = filename,
        root_directory = root_directory,
        supplier_id = supplier_id,
    )

    # Read data dictionary into DataFrame
    dd = pd.read_csv(
        filename, # tables[supplier_id]['schema_name'],
        sep = ';',
        skiprows = 3, # Skip first 3 lines
        header = 0,
        dtype = str,
        keep_default_na = False,
        na_values = []
    ).fillna('')

    logger.debug(f'P-Direkt data dictionary just after succesful read\n{dd}')

    # if column_name not in dd.keys():
    #     logger.warning('column_name {column_name} niet gevonden; "Veldnaam" gebruikt.')
    #     column_name = 'Veldnaam'

    df = pd.DataFrame(
        index = dd.index,
        columns = ['kolomnaam', 'datatype', 'leverancier_kolomnaam',
                   'leverancier_kolomtype', 'beschrijving'],
        dtype = str,
    )

    # df['kolomnaam'] = dd['Veldnaam']
    # df['datatype'] = ''
    # df['leverancier_kolomnaam'] = dd['Veldnaam']
    # df['leverancier_kolomtype'] = dd['TYPE']
    # df['beschrijving'] = dd['Unnamed: 8']

    logger.info('')
    logger.info('Columns copied from data dictionary:')
    i = 0
    if 'columns' in data_dict.keys():
        for col in data_dict['columns'].keys():
            i += 1
            new_col = data_dict['columns'][col]
            df[col] = dd[new_col]
            logger.info(f'{i:5d}. {col} <- {new_col}')

        # for
    # if

    # select rows where decimals > 0
    decs = (dd['DECIMALS'].astype('int') > 0)

    # set type these rows in df to numeric
    df.loc[decs, 'datatype'] = 'numeric'

    logger.info('')

    return df.fillna('')

### create_schema_from_pdirekt_datadict ###


def save_schema_file(df: pd.DataFrame, filename: str):
    df.to_csv(filename, index = None, sep =';', encoding = 'utf8')

    return

### save_schema_file ###


def merge_schema(schema: pd.DataFrame, merge_file: str, column_name: str):
    # load schema file to merge
    merger = pd.read_csv(
        merge_file,
        sep = ';',
        header = 0,
        dtype = str,
        keep_default_na = False,
    ).fillna('')

    # check if columns match between schema and merger
    if len(schema) != len(merger):
        raise DiDoError(f'*** length of basic schema ({len(schema)}) '
                        f'and file to merge ({merge_file}, len = {len(merger)}) '
                        'do not match')

    errors = False
    for i in range(len(schema)):
        dd_name = dc.change_column_name(schema.iloc[i]['kolomnaam'])
        if  dd_name != merger.iloc[i]['kolomnaam']:
            errors = True
            logger.error(f'*** Index {i}: kolomnamen komen niet overeen: '
                            f'{dd_name} =/= {merger.iloc[i]["kolomnaam"]}')
        # if
    # for

    if errors:
        raise DiDoError('Schema en merger komen niet overeen op kolom "kolomnaam"')

    for col in merger.columns:
        if col != 'kolomnaam' and 'Unnamed' not in col:
            schema[col] = merger[col]

    return schema

### merge_schema ###


def preprocess_data_dict(
        data_dict_specs: str,
        schema_filename: str,
        schema_dir: str,
        root_directory: str,
        leverancier: pd.DataFrame,
        leverancier_id: str,
    ):
    schema_source = dc.get_par(data_dict_specs, 'schema_source', '')
    merge_file = dc.get_par(data_dict_specs, 'merge_with', '')

    if schema_source == '<P-Direkt>':
        #TODO: Clear message when data_dict is absent
        data_dict_file = data_dict_specs['data_dict']
        column_name = dc.get_par(data_dict_specs, 'column_name', 'Omschrijving')

        # load data dictionary and create a valid schema from it
        # test if data_dict_file already has a .extension
        # _, _, ext = dc.split_filename(data_dict_file)
        # if len(ext) == 0:
        #     data_dict = os.path.join(schema_dir, data_dict_file + '.csv')
        # else:
        #     data_dict = os.path.join(schema_dir, data_dict_file)

        temp_df = create_schema_from_pdirekt_datadict(
            filename = data_dict_file,
            root_directory = root_directory,
            supplier_id = leverancier_id,
            data_dict = data_dict_specs,
        )

        # if additional info available, merge it with the schema
        if len(merge_file):
            merge_filename = os.path.join(schema_dir, merge_file)
            temp_df = merge_schema(temp_df, merge_filename, column_name)

        logger.info(f'Kolom voor kolomnaam: {column_name}')
        logger.debug(f'Processed P-Direkt data dictionary:\n{temp_df}')

        save_schema_file(temp_df, schema_filename)

    else:
        logger.error(f'*** Onbekende schema_source: {schema_source}')
    # if

    return

### preprocess_data_dict ###


def merge_table_and_schema(table: pd.DataFrame, schema: pd.DataFrame) -> pd.DataFrame:
    """ zet de kolomnamen en comments van schema in die van table.

    Als kolomnamen en of beschrijvingen aanwezig zijn in schema, dan worden die in de tabel
    vervangen.

    Args:
        table (pd.dataFrame): tabelbeschrijving zoals die uit de tabel komt
        schema (pd.dataFrame): schema afkomstig root/schemas

    Returns:
        pd.DataFrame: gemergde schema
    """

    # table should contain the same amount of rows as schema
    if len(table) != len(schema):
        raise DiDoError(f'merge_table_and_schema: len(table) {len(table)} niet gelijk aan len(schema) {len(schema)}')

    # keep track of errors being made
    errors = False
    table = table.set_index('kolomnaam')
    schema = schema.set_index('kolomnaam')

    # when schema contains kolomnaam or beschrijving, replace it
    for col_name in table.index:
        if col_name not in schema.index:
            errors = True
            logger.critical(f'Kolom "{col_name}" is onbekend in schema')

        else:
            if schema.loc[col_name, 'beschrijving'] is not None:
                table.loc[col_name, 'beschrijving'] = schema.loc[col_name, 'beschrijving']

    if errors:
        raise DiDoError('Onbekende kolomnamen gevonden, kolomnanmen in schema moeten identiek zijn aan die in de tabel')

    table = table.reset_index()
    schema = schema.reset_index()

    return table


#######################################################################################################################
#
# Fase 2 - Create the CREATE TABLE SQL
#
#######################################################################################################################


def write_markdown_doc(project_name: str,
                       supplier_config: dict,
                       columns_to_write: list,
                       doc_filename: str,
                      ):
    """ Write markdown documentation

    Args:
        outfile (object): file to write documentation to
        suppliers (dict): list of suppliers
        supplier_id (str): specific supplier to handle
        columns_to_write (list): columns to add into documentation
    """
    supplier_id = supplier_config['supplier_id']
    # project_name = supplier_config['config']['PROJECT_NAME']

    # get the meta and schema dataframe
    tables = supplier_config[dc.TAG_TABLES]
    schema = tables[dc.TAG_TABLE_SCHEMA][dc.TAG_SCHEMA]
    meta_table_name = tables[dc.TAG_TABLE_SCHEMA]["table_root"]
    meta_root = tables[dc.TAG_TABLE_META]["table_root"]
    meta_table_name = f'bronbestand_{meta_root}_{dc.TAG_DATA}'
    odl_server_config = supplier_config['config']['SERVER_CONFIGS']['ODL_SERVER_CONFIG']
    odl_meta_data = dc.load_odl_table(
        table_name = meta_table_name, # 'bronbestand_bestand_meta_data',
        server_config = odl_server_config,
    )

    # Get the optional data dataframe
    data = supplier_config[dc.TAG_TABLES][dc.TAG_TABLE_SCHEMA][dc.TAG_DATA]
    if not isinstance(data, pd.DataFrame):
        data = None

    # get version info
    version_odl = odl_meta_data.loc[0, 'odl_version']
    version_odl_date = odl_meta_data.loc[0, 'odl_version_date']
    version_dido = supplier_config['config']['PARAMETERS']['DIDO_VERSION']
    version_dido_date = \
        supplier_config['config']['PARAMETERS']['DIDO_VERSION_DATE'] # odl_meta_data.loc[0, 'dido_version_date']

    # get today
    nu = datetime.now().strftime(dc.DATETIME_FORMAT)

    with open(doc_filename, encoding="utf8", mode='a') as outfile:

        # write name of table or view
        outfile.write(f"# **Tabel: {supplier_id}: {project_name}**\n\n")

        outfile.write(f'This document was created at {nu}  \n')
        outfile.write('Created by DiDo version: '
                      f'{version_dido} ({version_dido_date})  \n')
        outfile.write(f'With ODL version: {version_odl} '
                      f'({version_odl_date})\n\n')

        prefix_text_label = f'{dc.TAG_PREFIX}_text'
        if prefix_text_label in supplier_config.keys() \
            and len(supplier_config[prefix_text_label]) > 0:

            outfile.write(supplier_config[prefix_text_label] + '\n\n')

        table_description: str = \
            supplier_config[dc.TAG_TABLES][dc.TAG_TABLE_META]['comment']
        if len(table_description) == 0:
            table_description = 'DOKUMENTATIE ONTBREEKT!'

        meta_data = \
            supplier_config[dc.TAG_TABLES][dc.TAG_TABLE_META][dc.TAG_DATA]
        if meta_data is None:
            logger.info('* no meta data available for %s', supplier_id)

        else:
            # write meta table header
            outfile.write('## Meta-informatie\n\n')
            outfile.write("| Meta attribuut | Waarde \n")
            outfile.write("| ---------- | ------ |\n")

            # write meta data
            for col in meta_data.columns:
                outfile.write(f'| {col}  | {meta_data.loc[0, col]} |\n')

        # Write the table description
        outfile.write('\n\n## Databeschrijving\n\n')
        outfile.write(table_description)
        outfile.write('\n\n')

        # if len(supplier_config[f'{dc.TAG_SUFFIX}_text']) > 0:
        #     outfile.write(supplier_config[f'{dc.TAG_SUFFIX}_text'] + '\n\n')

        # when columns_to_write is empty or '*' write all columns
        if len(columns_to_write) == 0 or columns_to_write == ['*']:
            columns_to_write = schema.columns

        # write table header
        underscores = ''
        for col in columns_to_write:
            colname = col.replace('_', ' ')
            colname = colname.capitalize()

            outfile.write(f' | {colname} ')
            underscores += ' | ----- '

        # for

        outfile.write(' |\n')
        outfile.write(f'{underscores} |\n')

        # write table contents, iterate over all rows
        for index, row in schema.iterrows():
            outfile.write('| ')

            # iterate over columns
            for col in columns_to_write:

                # if column does not exist, write an error message
                try:
                    cell = str(schema.loc[index, col]).replace('\n', '<br >')
                    cell = cell.replace('\r', '')

                    outfile.write(cell)
                    outfile.write(' | ')

                # Yields an error for each cell in that column, can be overwhelming
                except Exception as e:
                    logger.error(f'Error occurred: {e.args[0]}')
                    logger.error(f'No info written for: {col}')

                # try..except

            # for

            outfile.write('\n')

        # for

        outfile.write('\n')

        # Write the data when present (not None)
        if data is not None:
            outfile.write('\n\n')
            outfile.write('## Data\n\n')

            for col in data.columns:
                outfile.write(f' | {col} ')

            outfile.write(' | \n')

            for col in data.columns:
                outfile.write(' | ------- ')

            outfile.write('| \n')

            for idx, row in data.iterrows():
                for col in data.columns:
                    outfile.write(f' | {data.loc[idx, col] }')

                outfile.write(' | \n')

            outfile.write('\n')

        outfile.write('\n')

        # check if additional markdown exists
        suffix_text_label = f'{dc.TAG_SUFFIX}_text'
        if suffix_text_label in supplier_config.keys() \
            and len(supplier_config[suffix_text_label]) > 0:

            outfile.write('\n\n' + supplier_config[suffix_text_label] + '\n\n')
        # if

    # with_meta_data (

    return

### write_markdown_doc ###


def write_sql(meta_data: pd.DataFrame,
              supplier_config: dict,
              supplier_id: str,
              project_name: str,
              overwrite: bool,
              servers: dict,
              sql_filename: str,
              sql_script: str,
             ):
    """ Iterate over all elements in table and creates a data description

    When a data DataFrame is passed the table itself will be created and the
    data will be stored into the table.

    Args:
        meta_data (pd.DataFrame): meta data for this supplier
        supplier_config (dict): enhanced config dict for supplier
        supplier_id (str): supplier name
        project_name (str): project name
        overwrite (bool): True if existing tables must be overwritten
            Deletes all existing data !!
        servers (dict): dictionary containing all servers
        sql_filename (str): name of the file to write DDL onto
        sql_script (str): custom SQL script to execute after tables have been
            created

    Raises:
        DiDoError: Used when origin <table> is wrong

    @TODO: Origin <table> has not been updated along with other code. \
           This should be tested and probably repaired.
    """
    postgres_schema = servers['DATA_SERVER_CONFIG']['POSTGRES_SCHEMA']

    logger.info('')
    logger.info('[Writing SQL]')
    logger.info(f'>> Writing {supplier_id}')

    with open(sql_filename, encoding="utf8", mode='a') as outfile:

        # get the meta data, they are needed for saome operations
        for table_type in supplier_config[dc.TAG_TABLES]:

            # get schema file
            schema = supplier_config[dc.TAG_TABLES][table_type][dc.TAG_SCHEMA]

            # schema = supplier_config[dc.TAG_TABLES][table_type][dc.TAG_TABLE_SCHEMA]
            data = supplier_config[dc.TAG_TABLES][table_type][dc.TAG_DATA]

            # id data is not a dataframe, there is no data at all
            if not isinstance(data, pd.DataFrame):
                data = None

            # define the prototypical table name
            table_name = dc.get_table_name(project_name, supplier_id, table_type, '')

            # create SQL for the description table
            if supplier_config[dc.TAG_TABLES][table_type]['create_description']:
                description_tag = table_name + dc.TAG_DESC
                sql_code = create_table_description(
                    supplier_config = supplier_config,
                    table_type = table_type,
                    template = meta_data,
                    schema_name = postgres_schema,
                    table_name = description_tag,
                    overwrite = overwrite,
                )

                # define the data for the description
                sql_data = create_table_input(schema, schema,
                                                '',
                                                postgres_schema,
                                                description_tag)
                sql_code += sql_data

            # create the data table if create_data is True
            if supplier_config[dc.TAG_TABLES][table_type]['create_data']:

                data_table_tag = table_name + dc.TAG_DATA
                # schema = supplier_config[dc.TAG_TABLES]['schema'][dc.TAG_TABLE_SCHEMA] = schema

                # when meta, write the meta contents to the data table
                data = None
                if table_type == dc.TAG_TABLE_META:
                    data = supplier_config[dc.TAG_TABLES][dc.TAG_TABLE_META][dc.TAG_DATA]

                # when origin is omitted in config.yaml, supply the default (input: <file>)
                if 'origin' in supplier_config:
                    origin = supplier_config['origin']

                else:
                    origin = {'input': '<file>'}

                # origing is <table>. Probably faulty
                if origin['input'] == '<table>' and table_type == dc.TAG_TABLE_SCHEMA:
                    logger.info(f'  > Creating description for table schema: {data_table_tag}')
                    if 'table_name' not in origin or origin['table_name'] == '':
                        msg = 'No "table_name" was specified for origin in config.yaml'
                        raise DiDoError(msg)

                    if 'code_bronbestand' not in origin or origin['code_bronbestand'] == '':
                        origin['code_bronbestand'] = meta_data.iloc[-1].loc['code_bronbestand']

                    if 'levering_rapportageperiode' not in origin or origin['levering_rapportageperiode'] == '':
                        olp = meta_data.iloc[-1].loc['bronbestand_datum_begin'][:4] + '-I'
                        origin['levering_rapportageperiode'] = olp
                        logger.warning(f'!Geen "levering rapportageperiode" in origin/config.yaml. {olp} verondersteld.')

                    servers['FOREIGN_SERVER_CONFIG']['table'] = origin['table_name']
                    servers['DATA_SERVER_CONFIG']['table'] = data_table_tag

                    logger.info(f'Schema:\n{schema}')

                    sql_code += use_existing_table(
                        origin = origin,
                        schema = schema,
                        server_from = servers['FOREIGN_SERVER_CONFIG'],
                        server_to = servers['DATA_SERVER_CONFIG'],
                        odl_server = servers['ODL_SERVER_CONFIG'],
                    )
                    logger.info(f'  <table> {data_table_tag}')

                # all other situations, mainly <input>
                else:
                    sql_code += create_table(
                        schema = schema,
                        data = data,
                        table_name = data_table_tag,
                        project_name = project_name,
                        overwrite = overwrite,
                        server_config = servers['DATA_SERVER_CONFIG'],
                        supplier_config = supplier_config,
                    )

                    logger.info(f'  <file> {data_table_tag} (default)')

                # if

            # if

            outfile.write(sql_code)
            outfile.write('\n\n')

        # for -- tables

        # add custom SQL to sql_code when present
        if sql_script is not None:
            extra_sql = apply_sql_script(
                supplier_id = supplier_id,
                project_name = project_name,
                sql_script = sql_script,
                servers = servers,
            )

            outfile.write(extra_sql)
            outfile.write('\n\n')
        # if
    # with

    return

### write_sql ###


def apply_sql_script(supplier_id: str,
                     project_name: str,
                     sql_script: str,
                     servers: dict,
                    ) -> str:
    """ Read custom SQL script.

    Reads a custom SQL script yaml file and converts it to real SQL.


    Args:
        supplier_id (str): name of the supplier
        project_name (str): name of the project
        sql_script (str): dict containing the sql script instructions
        servers (dict): server configurations

    Raises:
        DiDoError: when irrecoverable errors occur

    Returns:
        str: generated SQL
    """
    # read sql script as yaml config file
    with open(sql_script, 'r') as infile:
        script = yaml.safe_load(infile)

    logger.info('')
    logger.info(f'[Adding custom SQL code from {sql_script}]')

    # test if required fields are in script
    if 'AFFECTED_TABLES' not in script.keys() or \
        'SQL' not in script.keys():

        raise DiDoError('Script MUST contain AFFECTED_TABLES and SQL fields')

    # if

    # get all names of tables for this supplier and project
    table_names = dc.get_table_names(project_name, supplier_id, 'description')
    table_names = [table_names[k] for k in table_names.keys()]
    data_names = dc.get_table_names(project_name, supplier_id, 'data')
    [table_names.append(data_names[k]) for k in data_names.keys()]

    # determine to which tables to apply the script
    affected_tables = script['AFFECTED_TABLES']
    if affected_tables == ['*']:
        affected_tables = table_names

    # test if affected tables exist
    errors = []
    for table_name in affected_tables:
        if table_name not in table_names:
            errors.append(table_name)

    if len(errors) > 0:
        raise DiDoError('*** Non existing affected tables: ' + str(errors))

    # get popstgres schema for the tables: always in data server
    server = servers['DATA_SERVER_CONFIG']
    schema = server['POSTGRES_SCHEMA']

    # get sql statements
    statements = script['SQL']

    # iterate over all tables, over all sql statements
    logger.info('Tables affected:')
    queries = ''
    for table in affected_tables:
        logger.info(f' - {table}')

        for statement in statements:
            # format variables:
            # - schema: database schema
            # - table: name of the table to apply
            vars = {'schema': schema, 'table': table}
            query = statements[statement].format(**vars)

            # test for terminating ;
            if not query.endswith(';'):
                query += ';'

            # # add qury to queries
            queries += query + '\n'

        # for

    # for

    # db_col = {'schema': schema, 'table': table}
    # query = dc.DB_SHOWD.format(**db_col)

    return queries

### apply_sql_script ###


def test_for_existing_tables(supplier_id: str,
                             project_name: str,
                             supplier_config: dict,
                             sql_server_config: dict,
                            ):

    results = {}
    any_present = False
    # supplier_id = supplier_config['supplier_id']

    # for supplier in supplier_config:
    logger.info(f'[Creating {supplier_id}]')

    # test if tables exist
    for table_type in supplier_config[dc.TAG_TABLES]:
        # test for existence of the table description
        table_name = dc.get_table_name(project_name, supplier_id, table_type, 'description')
        present = st.table_exists(
            table_name = table_name,
            verbose = False,
            sql_server_config  = sql_server_config,
           )
        results[table_name] = [present]
        any_present = any_present or present

        table_name = dc.get_table_name(project_name, supplier_id, table_type, 'data')
        present = st.table_exists(
            table_name = table_name,
            verbose = False,
            sql_server_config  = sql_server_config,
           )
        results[table_name] = [present]
        any_present = any_present or present

    # for

    return results, any_present

### test_for_existing_tables ###


def create_table_description(supplier_config: dict,
                             table_type: str,
                             template: pd.DataFrame,
                             schema_name: str,
                             table_name: str,
                             overwrite: bool,
                             ) -> str:
    """ Generates the description of a table in SQL format

    Args:
        supplier_config (dict): configuration of the supplier
        table_type (str): _description_
        template (pd.DataFrame): dataframe to create description from
        schema_name (str): postgres schema name
        table_name (str): postgres table name

    Returns:
        str: SQL to create the table description from
    """

    supplier_id = supplier_config['supplier_id']
    logger.info(f" > {supplier_id} - {table_type}")

    data_types = ''
    line = ''
    comments = ''
    table_comment = ''

    # get schema file
    schema = supplier_config[dc.TAG_TABLES][table_type][dc.TAG_TABLE_SCHEMA]
    template_property = str(supplier_config[dc.TAG_TABLES][table_type]['template'])

    data = supplier_config[dc.TAG_TABLES][table_type][dc.TAG_DATA]
    if not isinstance(data, pd.DataFrame):
        data = None

    # fetch description from meta data if present and create table comment
    comment = supplier_config[dc.TAG_TABLES][table_type]['comment']
    if len(comment) == 0:
        comment = '*** NO DOCUMENTATION PROVIDED ***'

    table_comment = f"COMMENT ON TABLE {schema_name}.{table_name} IS $${comment}$$;\n\n"

    # get comment
    template = template.set_index('kolomnaam', drop = False)

    # show full supplier schema
    for col in schema.columns:
        line = f'   {col} text'

        data_types += line + ',\n'

        # the if statement is a temporary provision: the meta table is not correctly defined,
        # meta should be data in the meta file
        if template_property != dc.TAG_TABLE_META:
            beschrijving = template.loc[col, 'beschrijving']
            comment = f"COMMENT ON COLUMN {schema_name}.{table_name}.{col} " \
                        f"IS $${beschrijving}$$;\n"

            comments += comment

    # remove last comma and newline
    data_types = data_types[:-2]

    # create a table definition
    tbd = ''
    if overwrite:
        tbd += f'DROP TABLE IF EXISTS {schema_name}.{table_name} CASCADE;\n\n'

    tbd += f'CREATE TABLE {schema_name}.{table_name}\n(\n'
    tbd += data_types + '\n);\n\n'
    tbd += table_comment + comments + '\n\n'
    #tbd += data_sql

    logger.debug(tbd)

    return tbd


def create_table_input(data: pd.DataFrame = None,
                       schema: pd.DataFrame = None,
                       filename: str = '',
                       schema_name: str = '',
                       table_name: str = ''
                       ) -> str:
    """ From DataFrame or data file

    When a filename is provided SQL is generated for the copy command.
    The filename *must* be an absolute path.
    The program does not check that.

    When the filename is the empty string the input is generated from
    the data DataFrame. The schema describes the data and the columns
    in data should exactly match the name and order of the names of
    'kolomnaam' in schema.

    Args:
        data -- to make table of
        schema -- data description
        filename -- name of the file containing the data; if none, 'data' used instead
        schema_name -- postgres schema name
        table_name -- postgres table name

    Returns:
        SQL statements containing the data generation
    """
    # check if a filename is provided; if so, generate COPY instruction
    if len(filename) > 0:
        sql = f"\\COPY {schema_name}.{table_name} FROM {filename} DELIMITER ';' CSV HEADER\n\n"

    # if not, use the data parameter
    else:
        sql = f'INSERT INTO {schema_name}.{table_name} ('
        for col in data.columns:
            sql += col + ', '

        sql = sql[:-2] + ')\nVALUES\n'
        for idx, row in data.iterrows():

            # create a row of values to be inserted
            row_values = '('
            for col in data.columns:
                data_type = schema.loc[idx, 'datatype']

                # when string, surround value by $$, except when kolomnaam == 'created_by'
                if isinstance(data_type, str) and col != 'created_by':
                    row_values += f'$${str(data.loc[idx, col])}$$, '
                else:
                    row_values += f'{str(data.loc[idx, col])}, '

            sql += row_values[:-2] + '),\n'

        # replace last , by ;
        sql = sql[:-2] + ';\n\n'

    return sql


def create_index(schema: pd.DataFrame,
                 data: pd.DataFrame,
                 table_name: str,
                 overwrite: bool,
                 server_config: dict,
                 supplier_config: dict,
                 ) -> str:
    """ Create indices for the table

    Args:
        schema (pd.DataFrame): description of table to be created
        data (pd.DataFrame): data to be stored into table
        table_name (str): name of the table to be created
        server_config(dict): server properties

    Returns:
        str: SQL statements to create the table
    """
    index_supplied = supplier_config['index']
    index_dict: dict = {}

    # for each index collect its columns
    for index_name in index_supplied.keys():
        index_dict[index_name] = {'columns': [], 'order': []}
        cols = index_supplied[index_name]
        for col in cols:
            try:
                order = cols[col].upper()
            except:
                order = ''

            index_dict[index_name]['columns'].append(col.lower()) # (items[0].lower())
            if len(order) > 2:
                if order == 'DESC' or order == 'DESCENDING':
                    index_dict[index_name]['order'].append('DESC')
                else:
                    index_dict[index_name]['order'].append('ASC')
                # if
            # if
        # for
    # for

    # construct the SQL string for creating the index
    pg_schema = server_config['POSTGRES_SCHEMA']
    sql = ''
    for index_name in index_dict.keys():
        if overwrite:
            sql += f'DROP INDEX IF EXISTS {index_name} CASCADE;'

        sql += f"CREATE INDEX {index_name} ON {pg_schema}.{table_name}\n(\n"
        for i in range(len(index_dict[index_name]['columns'])):
            col = index_dict[index_name]['columns'][i]
            order = index_dict[index_name]['order'][i]
            sql += f"    {col} {order},\n"
        # for
        sql = sql[:-2] + '\n);\n\n'

    # for

    return sql


def create_primary_key(schema: pd.DataFrame,
                       supplier_config: dict,
                      ) -> str:
    """ Create a table in the database and fill with data

    Args:
        schema (pd.DataFrame): description of table to be created
        data (pd.DataFrame): data to be stored into table
        table_name (str): name of the table to be created
        server_config(dict): server properties

    Returns:
        str: SQL statements to create the table
    """
    # for each index collect its columns
    kolommen = schema['kolomnaam'].tolist()

    pkeys = supplier_config['primary_key']

    # construct the SQL string for creating the index
    sql = '\n   PRIMARY KEY ('
    for key in pkeys:
        sql += key + ', '

    sql = sql[:-2] + ')'

    return sql


def create_table(schema: pd.DataFrame,
                 data: pd.DataFrame,
                 table_name: str,
                 project_name: str,
                 overwrite: bool,
                 server_config: dict,
                 supplier_config: dict,
                ) -> str:
    """ Create a table in the database and fill with data

    Args:
        schema (pd.DataFrame): description of table to be created
        data (pd.DataFrame): data to be stored into table
        table_name (str): name of the table to be created
        server_config(dict): server properties

    Returns:
        str: SQL statements to create the table
    """

    # initialize variables
    data_types = ''
    line = ''
    table_comment = ''
    comments = ''
    schema_name = server_config['POSTGRES_SCHEMA']
    supplier_id = supplier_config['supplier_id']

    # create variable names and types based on schema
    for idx, row in schema.iterrows():

        line = f'   {row["kolomnaam"]} {row["datatype"]}'
        if len(row['constraints']) > 0:
            line += ' ' + row['constraints']

        data_types += line + ',\n'

        comment = f"COMMENT ON COLUMN {schema_name}.{table_name}." \
                  f"{row['kolomnaam']} IS "
        description = schema.loc[idx, 'beschrijving'].strip()

        if len(description) == 0:
            description = '*** NO DOCUMENTATION PROVIDED ***'

        comment += f"$${description}$$;\n"
        comments += comment

    # for

    names = dc.get_table_names(
        project_name = project_name,
        supplier = supplier_id,
    )

    current_config = supplier_config

    # add primary key when defined
    if 'primary_key' in supplier_config \
        and table_name == names[dc.TAG_TABLE_SCHEMA]:

        primary_key = create_primary_key(schema, supplier_config)
        data_types += primary_key + ', '

    # remove last comma and space
    data_types = data_types[:-2]

    # create a table definition and instruction to read starttabel.csv
    tbd = ''
    if overwrite:
        tbd += f'DROP TABLE IF EXISTS {schema_name}.{table_name} CASCADE;\n\n'

    tbd += f'CREATE TABLE {schema_name}.{table_name}\n(\n'
    tbd += data_types + '\n);\n\n'
    tbd += table_comment + comments + '\n\n'

    # look if a primary key has to be added to the data file
    if 'index' in supplier_config[project_name] \
        and table_name == names[dc.TAG_TABLE_SCHEMA]:

        sql_index = create_index(
            schema = schema,
            data = data,
            table_name = table_name,
            overwrite = overwrite,
            server_config = server_config,
            supplier_config = supplier_config[project_name],
        )
        if len(sql_index) > 0:
            tbd += sql_index

    # if

    if data is not None:
        tbd += create_table_input(
            data = data,
            schema = schema,
            filename = '',
            schema_name = schema_name,
            table_name = table_name,
        )
    # if

    logger.debug(tbd)

    return tbd

### create_table ###


def use_existing_table(origin: dict,
                       schema: pd.DataFrame,
                       server_from: dict,
                       server_to: dict,
                       odl_server: dict
                      ):
    """ Creates a DiDo data table from an existing table

    This is a somewhat elaborate process
    - create new table
    - create the bootstrap columns
    - create the columns of the table to be copied
    - copy the data of the table to be copied
    - fill the bootstrap columns

    Args:
        origin (dict): meta data to use
        server_from (dict): server properties
        server_to (dict): server properties
    """
    # Read bootstrap data
    bootstrap = dc.load_odl_table(dc.EXTRA_TEMPLATE, odl_server)

    # get the table data
    old_table = st.get_structure(server_from['table'],
                                           sql_server_config = server_from,
                                           verbose = False)

    table_to_name = f"{server_to['POSTGRES_SCHEMA']}.{server_to['table']}"
    #sql = f"DROP TABLE IF EXISTS {table_to_name};\n\n"
    sql = f"CREATE TABLE {server_to['POSTGRES_SCHEMA']}.{server_to['table']}\n(\n"
    data = ''
    comments = ''
    select_columns = ''

    # create the new table
    # first copy the bootstrap data, column and comment definitions separately
    for idx, row in bootstrap.iterrows():
        sql += f"   {bootstrap.loc[idx, 'kolomnaam']} "\
               f"{bootstrap.loc[idx, 'datatype']} "\
               f"{bootstrap.loc[idx, 'constraints']},\n"

        comments += f"COMMENT ON COLUMN {table_to_name}." \
                    f"{bootstrap.loc[idx, 'kolomnaam']} IS " \
                    f"$${bootstrap.loc[idx, 'beschrijving']}$$;\n"

    # now copy the table definitions from the old table to the new table definition
    # keep track of the column names, we need the to insert data from old to new
    schema = schema.set_index('kolomnaam')
    for idx, row in old_table.iterrows():
        kolomnaam = old_table.loc[idx, 'kolomnaam']
        sql += f"   {kolomnaam} "\
               f"{old_table.loc[idx, 'datatype']} "\
               f"{old_table.loc[idx, 'constraints']},\n"

        select_columns += f"{old_table.loc[idx, 'kolomnaam']}, "

        beschrijving = schema.loc[kolomnaam, 'beschrijving']
        if beschrijving is not None and len(str(beschrijving)) > 0:
            old_table.loc[idx, 'beschrijving'] = beschrijving

        beschrijving = old_table.loc[idx, 'beschrijving']

        comments += f"COMMENT ON COLUMN {table_to_name}." \
                    f"{old_table.loc[idx, 'kolomnaam']} IS " \
                    f"$${beschrijving}$$;\n"

    # for

    schema = schema.reset_index()

    sql = sql[:-2] + '\n);\n\n' + comments + '\n' # + insert

    # The part of cpying the data is moved to dido_import

    return sql # + data


def load_supplier_schemas(supplier_config: dict,
                          supplier_id: str,
                          projects: dict,
                          project_id: str,
                          root: str,
                          work: str
                         ) -> dict:
    """load data and meta schemas from file and add to suppliers info

    Args:
        suppliers (dict): dictionary with suppliers info
        root (str): root directory to read schemas from
        work (str): working directory to write updated schemas to

    Raises:
        DiDoError: when illegal format of filename is supplied

    Returns:
        updated suppliers dictionary
    """
    tables: dict = {}
    tables[supplier_id] = {}

    logger.info('')
    logger.info('[Loading supplier schemas])')

    # for supplier in supplier_config.keys():
    # supplier_id = supplier_config['supplier_id']
    dc.subheader(f'Supplier: {supplier_id}', '=')
    schema_work = os.path.join(work, 'schemas', supplier_id)
    doc_root = os.path.join(root, 'docs', supplier_id)

    # read the schema description schema
    schema_file = projects[project_id]['schema_file'] + '.schema.csv'
    tables[supplier_id]['schema_name'] = os.path.join(work, 'schemas', supplier_id, schema_file)
    df = pd.read_csv(
        tables[supplier_id]['schema_name'],
        sep = ';',
        dtype = str,
        keep_default_na = False,
        na_values = []
    ).fillna('')

    supplier_config[dc.TAG_TABLES][dc.TAG_TABLE_SCHEMA][dc.TAG_SCHEMA] = df

    # meta information is a bit weird. The input data is kind of user friendly key-value store
    # it is stored in the metadata tag and in a later phase will be merged from the
    # metadata description file fetched from ODL
    meta_file = projects[project_id]['schema_file'] + '.meta.csv'
    tables[supplier_id]['meta_name'] = os.path.join(work, 'schemas', supplier_id, meta_file)
    meta = pd.read_csv(
        tables[supplier_id]['meta_name'],
        sep = ';',
        dtype = str,
        keep_default_na = False
    ).fillna('')

    # strip all spaces left and right
    for idx in meta.index:
        meta.loc[idx, 'waarde'] = meta.loc[idx, 'waarde'].strip()

    meta = meta.set_index(meta.columns[0])
    supplier_config[dc.TAG_TABLES][dc.TAG_TABLE_META][dc.TAG_SCHEMA] = '<odl>'
    supplier_config[dc.TAG_TABLES][dc.TAG_TABLE_META][dc.TAG_DATA] = meta

    # load optional data into dataframe and assign to suppliers info
    if 'data_name' in tables[supplier_id].keys():
        logger.info('[Reading data]')
        supplier_config[dc.TAG_TABLES][dc.TAG_TABLE_SCHEMA][dc.TAG_DATA] = pd.read_csv(
            tables[supplier_id]['data_name'],
            sep = ';',
            dtype = str,
            keep_default_na = False
        ).fillna('')

    return supplier_config

### load_supplier_schemas ###


def merge_meta_data_with_description(desc: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """ converts the metadata input into the format provided by description

    Args:
        desc (pd.DataFrame): description dataframe for metadata
        metadata (pd.DataFrame): metadata in key-value format

    Returns:
        the converted dataframe
    """
    # create new dataframe with kolomnaam as columns
    cols = desc['kolomnaam'].tolist()

    # Create dataframe with empty strings
    df = pd.DataFrame([[''] * len(cols)], columns = cols)

    for idx, row in metadata.iterrows():
        if idx in df.columns:
            value = metadata.loc[idx, 'waarde']
            df.loc[0, idx] = value

    return df

### merge_meta_data_with_description  ###


def merge_meta_data_with_odl_data(config: dict,
                                  metadata: pd.DataFrame,
                                  metadata_data: pd.DataFrame
                                 ):

    """ Assigns default values tot metadata from ODL

    The ODL metadata table contains meta data for general use, especially
     version information. When a cell in metadata is empty it is assigned
     its default value from ODL (metadata_data)

     Note: the version variables must be accessed from PARAMETERS by their name,
           however, they must be accessed from the meta data table by their
           dido_common.py name. That is possible using 'getattr' but it is a
           trick. Be careful using it.

    Args:
        metadata (pd.DataFrame): meta data of current project
        metadata_data (pd.DataFrame): default meta data from ODL

    Returns:
        pd.DataFrame: _description_metadata enhanced with ODL default values
    """
    logger.debug('Assigning ODL metadata to empty cells in DiDo metadata')

    # first preset the DiDo version from parameters in the config dict
    # Get DiDo version from parameters in the config dict
    key = 'DIDO_VERSION'
    value = dc.get_par_par(config, 'config', 'PARAMETERS', '')[key]
    metadata.loc[0, key.lower()] = value
    key += '_DATE'
    vdate = dc.get_par_par(config, 'config', 'PARAMETERS', '')[key]
    metadata.loc[0, key.lower()] = vdate
    logger.info('DiDo version copied from dido.yaml into meta data: '
                f'{value}, {vdate}')

    for col in metadata_data.columns:
        # only assign values if the column occurs in metadata_data
        # and that column ain't empty
        if col in metadata.columns and len(str(metadata.loc[0, col])) == 0:
            logger.debug(f'{col}: {str(metadata.loc[0, col])}')
            metadata.loc[0, col] = metadata_data.loc[0, col]

        # when not, log it
        else:
            logger.debug(f'{col} not assigned')

        # if

    # for

    # return enhanced metada
    return metadata

### merge_meta_data_with_odl_data ###


def load_supplier_odl(supplier_config: dict, project_name: str, server_config: dict):
    """ add additional information to suppliers data from the database

    Args:
        suppliers (dict): all suppliers
        project_name (str): used for naming
        server_config (dict): postgres credentials

    Returns:
        dict: augmented suppliers information
    """
    logger.info('')
    logger.info('[Loading supplier ODL]')

    # enumerate over all suppliers
    supplier_id = supplier_config['supplier_id']
    # for supplier in supplier_config.keys():

    # enumerate over all tables of the supplier
    logger.info(f'[Processing supplier: {supplier_id}]')
    for key in supplier_config[dc.TAG_TABLES].keys():
        logger.info(f'>> Doing table: {key}')
        table_info = supplier_config[dc.TAG_TABLES][key]
        table_root = table_info['table_root']
        new_table_name = dc.get_table_name(
            project_name = project_name,
            supplier = supplier_id,
            table_info = table_root,
            postfix = dc.TAG_DESC,
        )

        # create the table name
        odl_name = f'bronbestand_{table_root}_{dc.TAG_DESC}'

        # if schema is <odl>, fetch the schema from ODL
        if isinstance(table_info[dc.TAG_SCHEMA], str) \
            and table_info[dc.TAG_SCHEMA] == '<odl>':

            # fetch the table from the database and assign as dataframe to schema
            table_info[dc.TAG_SCHEMA] = dc.load_odl_table(odl_name, server_config)

            # change table name as table of this project and this supplier
            table_info['table_name'] = new_table_name

        # if data is <odl>, fetch the data from ODL
        if isinstance(table_info[dc.TAG_SCHEMA], str) \
            and table_info[dc.TAG_DATA] == '<odl>':

            # create the table name
            odl_name = f'bronbestand_{table_root}_{dc.TAG_DATA}'

            # fetch the table from the database and assign as dataframe to schema
            table_info[dc.TAG_DATA] = dc.load_odl_table(odl_name, server_config)

            # change table name as table of this project and this supplier
            table_info[dc.TAG_DATA] = dc.load_odl_table(odl_name, server_config)
            table_info['data_name'] = dc.get_table_name(project_name, supplier_id, table_root, 'data')

        if key == dc.TAG_TABLE_META:
            # merge the metadata into the dataframe
            # x = table_info['table_root']
            meta_name = f'bronbestand_{table_root}_{dc.TAG_DATA}'
            metadata_data = dc.load_odl_table(meta_name, server_config)
            df = merge_meta_data_with_description(table_info[dc.TAG_SCHEMA], table_info[dc.TAG_DATA])
            df = merge_meta_data_with_odl_data(supplier_config, df, metadata_data)

            logger.debug('Final meta data')
            for i, col in enumerate(df.columns):
                logger.debug(f'{col}: {df.iloc[0, i]}')

            table_info[dc.TAG_DATA] = df
        # if

        # if comment is <odl> fetch it from the ODL
        if table_info['comment'] == '<odl>':
            comment = f"obj_description('datamanagement.{odl_name}'::regclass, 'pg_class')"

            # info on where to find ODL is hard coded
            comment_df = st.sql_select(
                columns = comment,
                verbose = False,
                sql_server_config = server_config,
            )

            # Table comment is also stored separately for later use
            table_info['comment'] = comment_df.loc[0, 'obj_description']

        # if
    # for

    logger.info('')

    return supplier_config

### load_supplier_odl ###


def show_supplier_schemas(suppliers: dict):
    """ Does a pretty prnt of the suppliers table

    Args:
        suppliers (dict): the suppliers table
    """

    # show a deeply nested dictionary and indent accordingly
    for supplier in suppliers:
        logger.info(f'=== {supplier} ===')
        for prop in suppliers[supplier]:
            if prop == dc.TAG_TABLES:
                for table_key in suppliers[supplier][prop]:
                    logger.info(f'    --- {table_key} ---')
                    for sub in suppliers[supplier][prop][table_key]:
                        if isinstance(suppliers[supplier][prop][table_key][sub], pd.DataFrame):
                            logger.info(f'      {sub}: {str(type(suppliers[supplier][prop][table_key][sub]))}')
                            logger.info(suppliers[supplier][prop][table_key][sub].columns)

                        else:
                            logger.info(f'      {sub}: {suppliers[supplier][prop][table_key][sub]}')

                    else:
                        logger.info(f'    {table_key}: *{str(type(suppliers[supplier][prop]))}*')

            else:
                try:
                    logger.info(f'  {prop}: { suppliers[supplier][prop]}')

                except:
                    logger.info(f'  {prop}: { suppliers[supplier]}')

                # try..except
            # if
        # for
    # for

    return

### show_supplier_schemas ###


def dido_begin(config_dict: dict):
    # get the database server definitions
    db_servers = config_dict['SERVER_CONFIGS']
    odl_server_config = db_servers['ODL_SERVER_CONFIG']
    foreign_server_config = db_servers['FOREIGN_SERVER_CONFIG']

    # get project environment
    root_dir = config_dict['ROOT_DIR']
    work_dir = config_dict['WORK_DIR']
    leveranciers = config_dict['SUPPLIERS']

    # get data types
    data_types = dc.create_data_types()

    # make location for logger
    create_workdir_structure(config_dict, db_servers['ODL_SERVER_CONFIG'])

    # assign DATATYPES to data_types when present in config_dict,
    # else it will be an empty dictionary
    data_types = {}
    if 'DATATYPES' in config_dict:
        data_types = config_dict['DATATYPES']

    # select which suppliers to process
    suppliers_to_process = dc.get_par(config_dict, 'SUPPLIERS_TO_PROCESS', '*')

    # just * means process all
    if suppliers_to_process == '*':
        suppliers_to_process = leveranciers.keys()

    # process each supplier
    for leverancier_id in suppliers_to_process:
        dc.subheader(f'Supplier {leverancier_id}', '=')

        leverancier, projects = dc.get_supplier_projects(
            config = config_dict,
            supplier = leverancier_id,
            delivery = 1,
            keyword = 'SUPPLIERS',
        )

        for project_key in projects:
            dc.subheader(f'Project {project_key}', '-')
            project = projects[project_key]

            # directory to load files from root_dir
            dir_load = os.path.join(root_dir, 'schemas', leverancier_id)

            # directory to save into work_dir
            dir_save = os.path.join(work_dir, 'schemas', leverancier_id)

            source_file = project['schema_file']
            fname_schema_load = os.path.join(dir_load, source_file + '.schema.csv')
            fname_meta_load = os.path.join(dir_load, source_file + '.meta.csv')
            fname_data_load = os.path.join(dir_load, source_file + '.data.csv')
            fname_schema_save = fname_schema_load.replace(dir_load, dir_save)
            fname_meta_save = fname_meta_load.replace(dir_load, dir_save)
            fname_data_save = fname_data_load.replace(dir_load, dir_save)

            """
            schema_source determines the origin of the schema. Sources are:
            <schema> - a user supplied schema file
            <P-Direkt> - data dictionary from P-Direkt
            <...> - when necessary other sources with their interface can be added

            A source other than <schema> must be converted to schema format.
            For p-direct the function create_schema_from_pdirekt_datadict is provided.
            The function preprocess_data_dict prepocesses a data dictionary when present.
            """
            data_dict = dc.get_par(project, 'data_dictionary', {})
            if len(data_dict) > 0:
                preprocess_data_dict(
                    data_dict_specs = data_dict,
                    schema_filename = fname_schema_load,
                    schema_dir = dir_load,
                    root_directory = root_dir,
                    leverancier = leverancier,
                    leverancier_id = leverancier_id,
                )

            # load templates from database
            schema_template = dc.load_odl_table(dc.SCHEMA_TEMPLATE, odl_server_config)

            # load source files for current supplier: schema and meta file
            if not os.path.exists(fname_schema_load):
                raise DiDoError(f'Schema file not found {fname_schema_load}')
            if not os.path.exists(fname_meta_load):
                raise DiDoError(f'Meta data file not found {fname_meta_load}')

            logger.debug(f'[Input schema file: {fname_schema_load}]')
            schema_leverancier = pd.read_csv(
                fname_schema_load,
                sep = ';', # r'\s*;\s*', # warning regex seps ignore quoted data
                dtype = str,
                keep_default_na = False,
                engine = 'python',
            ).fillna('')

            """
            Origin is an option that indicates the origin of the data.
            <file>  - the data are delivered by file
            <table> - the data are delived by a postgres table,
                      only valid for initialization
            <api>   - data are delived via an internet api,
                      one time initialization and frequent updates
                      when this option is specified, the connection is tested

            Only <table> impacts dido_begin, all three impact
            dido_data_prep and dido_import
            """

            # fetch origin from config when present, else default to <file>
            if 'origin' in leverancier.keys():
                origin = leverancier['origin']

            else:
                origin = {'input': '<file>'}

            if origin['input'] == '<file>':
                logger.info('Origin is <file>')

            elif origin['input'] == '<table>':
                # fetch schema  from table
                table_name = origin['table_name']
                logger.debug(f'[Input schema table: {table_name}]')
                table_leverancier = fetch_schema_from_table(
                    table_name = table_name,
                    sql_server_config = foreign_server_config,
                )
                schema_leverancier = merge_table_and_schema(
                    table = table_leverancier,
                    schema = schema_leverancier,
                )
                logger.info(f'Origin is <tabprintle>, from {table_name}')

            else:
                raise DiDoError(f'*** Unknown origin input: { origin["input"]}. '
                                'Only <file>, <table> or <api> allowed.')

            logger.debug(schema_leverancier)
            schema_leverancier = merge_bootstrap_data(
                schema = schema_leverancier,
                table_name = dc.EXTRA_TEMPLATE,
                server_config = odl_server_config,
            )

            meta_leverancier = pd.read_csv(
                fname_meta_load,
                sep = ';',
                dtype = str,
                keep_default_na = False,
            )

            meta_leverancier = meta_leverancier.fillna('').set_index('attribuut')

            # load the data from the parameters table
            bootstrap_data = dc.load_parameters()

            # fetch the allowed datatypes from dido.yaml
            temp, _ = dc.create_data_types()
            allowed_datatypes = list(temp.keys())

            # preprocess and save the schema
            schema_template = apply_schema_odl(
                template = schema_template,
                schema = schema_leverancier,
                meta = meta_leverancier,
                data_dict = data_types,
                filename = fname_schema_load,
                allowed_datatypes = allowed_datatypes,
                supplier_config = leverancier,
            )

            schema_template.to_csv(fname_schema_save, sep=';', index=False)

            # preprocess and save the meta data
            meta_leverancier = apply_meta_odl(
                meta = meta_leverancier,
                n_cols = len(schema_template),
                filename = fname_meta_load,
                bootstrap_data = bootstrap_data,
                server_config = odl_server_config,
            )
            meta_leverancier.to_csv(fname_meta_save, sep=';', index=True)

        # for -- project
    # for -- supplier

    return

### dido_begin ###


def dido_create(config_dict: dict):
    """ generates markdown docs for wiki and SQL from schema and meta.

    Args:
        config_dict (dict): configuration file
    """
    # get the database server definitions
    db_servers = config_dict['SERVER_CONFIGS']
    odl_server_config = db_servers['ODL_SERVER_CONFIG']
    data_server_config = db_servers['DATA_SERVER_CONFIG']

    # get project environment
    root_dir = config_dict['ROOT_DIR']
    work_dir = config_dict['WORK_DIR']
    leveranciers = config_dict['SUPPLIERS']
    table_desc = config_dict['PARAMETERS']['TABLES']
    write_columns = config_dict['COLUMNS']
    overwrite_tables = dc.get_par(config_dict, 'KILL_EXISTING_TABLES', False)

    # file names
    sql_filename = os.path.join(work_dir, 'sql', 'create-tables.sql')
    doc_filename = os.path.join(work_dir, 'docs', 'create-docs.md')

    # write initial code to sqlfile, write_sql opens file in append mode
    with open(sql_filename, encoding="utf8", mode='w') as sqlfile:
        # create a transaction of tables creation
        sqlfile.write('-- Quit immediately with exit code other '
                      'than 0 when an error occurs\n')
        sqlfile.write('\\set ON_ERROR_STOP true\n\n')
        sqlfile.write('BEGIN; -- Transaction\n\n')

    # select which suppliers to process
    suppliers_to_process = suppliers_to_process = dc.get_par(
        config = config_dict,
        key = 'SUPPLIERS_TO_PROCESS',
        default = '*',
    )

    # just * means process all
    if suppliers_to_process == '*':
        suppliers_to_process = leveranciers.keys()

    sql_filename = os.path.join(work_dir, 'sql', 'create-tables.sql')
    doc_filename = os.path.join(work_dir, 'docs', 'create-docs.md')
    any_present = False

    # create documentation file and write TOC header
    with open(doc_filename, encoding="utf8", mode='w') as docfile:
        docfile.write('[[_TOC_]]\n\n')

    # copy table_desc as template for information each supplier has to deliver
    for leverancier_id in suppliers_to_process:
        dc.subheader(f'Supplier: {leverancier_id}', '=')
        logger.info('')

        # count number of deliveries and fetch supplier and delivery accordingly
        delivery_seq = 1
        logger.info(f'Dido_create always applies delivery {delivery_seq}')
        leverancier_config, projects = dc.get_supplier_projects(
            config = config_dict,
            supplier = leverancier_id,
            delivery = 1,
            keyword = 'SUPPLIERS',
        )

        # Read custom SQL script for supplier, None when not present
        sql_script = dc.get_par(config_dict, 'SQL_SCRIPT', None)
        if sql_script is not None and not sql_script.startswith('/'):
            sql_script = os.path.join(root_dir, 'sql', leverancier_id, sql_script)

        # Iterate over all projects
        for project_key in projects:
            dc.subheader(f'Project: {project_key}', '-')
            project = projects[project_key]

            # copy table info into leverancier_config
            leverancier_config[dc.TAG_TABLES] = {}
            for table_key in table_desc.keys():

                # COPY the table dictionary to the supplier dict, else shared
                # printreference will be copied; use copy() function
                leverancier_config[dc.TAG_TABLES][table_key] = \
                    table_desc[table_key].copy()

                # copy all keys, as they are string, the are correctly copied
                for key in table_desc[table_key].keys():
                    leverancier_config[dc.TAG_TABLES][table_key][key] = \
                        table_desc[table_key][key]
                # for
            # for

            # load schema and documentation files and add to leveranciers info
            leverancier_config = load_supplier_schemas(
                supplier_config = leverancier_config,
                supplier_id = leverancier_id,
                projects = projects,
                project_id = project_key,
                root = root_dir,
                work = work_dir,
            )

            # add ODL info from database
            leverancier_config = load_supplier_odl(
                supplier_config = leverancier_config,
                project_name = project_key,
                server_config = odl_server_config,
            )

            # warn if the tables exist in the database
            present, any_present = test_for_existing_tables(
                supplier_id = leverancier_id,
                project_name = project_key,
                supplier_config = leverancier_config,
                sql_server_config = data_server_config,
            )

            if any_present:
                logger.warning('!!! Tables already exist for supplier '
                               f'{leverancier_id}, project {project_key}. '
                               'Remove these using "dido_kill_supplier"')
                logger.warning('Existing tables:')
                for key in present.keys():
                    if present[key]:
                        logger.info(f' - {key}')

                logger.info('')
            # if

            # create SQL to create tables
            schema_name = 'bronbestand_' + 'attribuutmeta' + '_description'
            meta_table = dc.load_odl_table(
                table_name = schema_name,
                server_config = odl_server_config,
            )
            leverancier_config[dc.TAG_TABLES][dc.TAG_TABLE_META][dc.TAG_TABLE_META] = \
                meta_table

            write_sql(
                meta_data = meta_table,
                supplier_config = leverancier_config,
                supplier_id = leverancier_id,
                # project_config = project,
                project_name = project_key,
                overwrite = overwrite_tables,
                servers = db_servers,
                sql_filename = sql_filename,
                sql_script = sql_script,
            )

            # create documentation
            write_markdown_doc(
                project_name = project_key,
                supplier_config = leverancier_config,
                columns_to_write = write_columns,
                doc_filename = doc_filename,
            )
        # for -- project
    # for -- supplier

    # write the commit statement
    with open(sql_filename, encoding="utf8", mode='a') as sqlfile:
        sqlfile.write('\nCOMMIT; -- Transaction\n')

    dc.report_psql_use(
        table = 'create-tables',
        servers = db_servers,
        tables_exist = any_present,
        overwrite = overwrite_tables,
    )

    return

### dido_create ###


def main():
    cpu = time.time()

    # read commandline parameters
    appname, args = dc.read_cli()

    # read the configuration file
    config = dc.read_config(args.project)

    # display banner
    dc.display_dido_header('Creating Tables and Documentation', config)

    # create the tables
    dido_begin(config)
    dido_create(config)

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