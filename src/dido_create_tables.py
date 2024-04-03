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
import sys
import json
import time
import shutil
import requests
import pandas as pd

from datetime import datetime
from dido_list import dido_list
from requests.auth import HTTPBasicAuth

# Don't forget to set PYTHONPATH to your python library files
# export PYTHONPATH=/path/to/dido/helpers/map
import api_postcode
import dido_common as dc
import simple_table as st

from dido_common import DiDoError

# show all columns of dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# pylint: pointless-string-statement

#######################################################################################################################
#
# Fase 1 - Read and check the schema files
#
#######################################################################################################################


def fetch_schema_from_table(table_name, sql_server_config: dict) -> pd.DataFrame:
    result = st.get_structure(table_name,
                                        verbose = False,
                                        sql_server_config = sql_server_config,
                                       )
    return result


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
        errmsg = f'* Schema "{filename}": de file "bronbestand_attribuut_meta.csv" file ontbreekt. '
        errmsg += 'ODL generatie kan daar niet zonder.'
        logger.error(errmsg)

        raise DiDoError(errmsg)

    # check if meta file exists
    if meta is None:
        errmsg = f'* Schema "{filename}": de file "bronbestand_attribuut_meta.meta.csv" file '
        errmsg += 'ontbreekt. ODL generatie kan daar niet zonder.'
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
    for row, _ in new_df.iterrows():
        # if description is lacking, notify the user and flag it
        beschrijving = schema.loc[row, 'beschrijving'].strip()
        if len(beschrijving) == 0:
            logger.error(f'* Schema "{filename}, kolom "{row}", beschrijving ontbreekt.')
            errors = True

        # if kolomnaam is omitted copy it from leverancier_kolomnaam
        if len(schema.loc[row, 'kolomnaam'].strip()) == 0:
            if len(schema.loc[row, 'leverancier_kolomnaam'].strip()) > 0:
                schema.loc[row, 'kolomnaam'] = schema.loc[row, 'leverancier_kolomnaam'].strip().lower()
            else:
                errors = True

        # ensure that kolomnaam is a postgres accepted column name
        new_name = dc.change_column_name(schema.loc[row, 'kolomnaam'])

        # when no name could be created, create a random one
        if len(new_name) == 0:
            logger.error(f'* Schema {filename}, geen kolomnaam in regel {row + 1}')

        # replace datatypes if necessary and possible
        dtyp = schema.loc[row, 'datatype'].strip()
        if  len(dtyp) == 0 and \
            len(schema.loc[row, 'leverancier_kolomtype'].strip()) > 0 and \
            data_dict is not None:
            if schema.loc[row, 'leverancier_kolomtype'].strip() in data_dict.keys():
                datatype = data_dict[schema.loc[row, 'leverancier_kolomtype'].strip()]
                schema.loc[row, 'datatype'] = datatype

        # if that does not help, flag it
        datatype = schema.loc[row, 'datatype'].strip().lower()

        if len(datatype) == 0:
            schema.loc[row, 'datatype'] = '*NONE*'

        # is datatype allowed?
        if datatype not in allowed_datatypes:
            logger.error(f'* Datatype not allowed: {datatype}')
            errors = True

        # assign newly created column name to kolomnaam
        new_df.loc[row, 'kolomnaam'] = new_name

        # create code attribuut
        code_atr = ''

        # check if it occurs in data.columns
        if 'code_attribuut' in schema.columns:
            code_atr = str(schema.loc[row, 'code_attribuut']).strip()

        # not assign a numeric value?
        if len(code_atr) == 0:
            code_atr = f'{row + 1:03d}'

        # assign to new_df
        new_df.loc[row, 'code_attribuut'] = code_atr

        # same for code_attribuut_sleutel
        code_atr_key = code_bbs + code_atr
        if 'code_attribuut_sleutel' in schema.columns:
            code_atr_key = str(schema.loc [row, 'code_attribuut_sleutel']).strip()

        if len(code_atr_key) == 0:
            code_atr_key = code_bbs + code_atr

        new_df.loc[row, 'code_attribuut_sleutel'] = code_atr_key

        if 'positie' not in schema.columns or (isinstance(schema.loc[row, 'positie'], str) and len(schema.loc[row, 'positie']) == 0):
            new_df.loc[row, 'positie'] = str(row + 1)

        # assign rest of data values
        for col_name in schema.columns:
            if col_name not in names_to_skip:
                try:
                    value = schema.loc[row, col_name].strip()
                except:
                    value = ''

                if len(value) > 0:
                    new_df.loc[row, col_name] = value

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
    # print(template)

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
    #bootstrap_data = load_odl_table('odl_parameters_description', server_config).set_index('kolomnaam')
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


def create_schema_from_pdirekt_datadict(filename: str, data_dict: dict):
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
    # print(dd)
    # print(df)
    # df.to_csv('AAA.csv')

    # select rows where decimals > 0
    decs = (dd['DECIMALS'].astype('int') > 0)

    # set type these rows in df to numeric
    df.loc[decs, 'datatype'] = 'numeric'
    # print(df)

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
        leverancier: pd.DataFrame,
    ):
    schema_source = dc.get_par(data_dict_specs, 'schema_source', '')
    merge_file = dc.get_par(data_dict_specs, 'merge_with', '')

    if schema_source == '<P-Direkt>':
        #TODO: Clear message when data_dict is absent
        data_dict_file = data_dict_specs['data_dict']
        column_name = dc.get_par(data_dict_specs, 'column_name', 'Omschrijving')

        # load data dictionary and create a valid schema from it
        data_dict = os.path.join(schema_dir, data_dict_file + '.csv')
        temp_df = create_schema_from_pdirekt_datadict(data_dict, data_dict_specs)

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


def write_markdown_doc(outfile: object, supplier_config: dict, columns_to_write: list):
    """_summary_

    Args:
        outfile (object): file to write documentation to
        suppliers (dict): list of suppliers
        supplier_id (str): specific supplier to handle
        columns_to_write (list): columns to add into documentation
    """
    supplier_id = supplier_config['supplier_id']

    # get the meta and schema dataframe
    schema = supplier_config[dc.TAG_TABLES][dc.TAG_TABLE_SCHEMA][dc.TAG_SCHEMA]
    # meta = suppliers[dc.TAG_TABLES][dc.TAG_TABLE_META][dc.TAG_SCHEMA]

    # Get the optional data dataframe
    data = supplier_config[dc.TAG_TABLES][dc.TAG_TABLE_SCHEMA][dc.TAG_DATA]
    if not isinstance(data, pd.DataFrame):
        data = None

    # write name of table or view
    outfile.write(f"# **Tabel: {supplier_id}**\n\n")

    prefix_text_label = f'{dc.TAG_PREFIX}_text'
    if prefix_text_label in supplier_config.keys() and len(supplier_config[prefix_text_label]) > 0:
        outfile.write(supplier_config[prefix_text_label] + '\n\n')

    table_description: str = supplier_config[dc.TAG_TABLES][dc.TAG_TABLE_META]['comment']
    # meta.iloc[0]['bronbestand_beschrijving'].strip()
    if len(table_description) == 0:
        table_description = 'DOKUMENTATIE ONTBREEKT!'

    meta_data = supplier_config[dc.TAG_TABLES][dc.TAG_TABLE_META][dc.TAG_DATA]
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

            # Yields an error message for each cell in that column, can be overwhelming
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
    if suffix_text_label in supplier_config.keys() and len(supplier_config[suffix_text_label]) > 0:
        outfile.write('\n\n' + supplier_config[suffix_text_label] + '\n\n')


    return

### write_markdown_doc ###


def write_documentation(filename: str, suppliers: dict, columns_to_write: list):
    """ The documentation is written in markdown format with a __TOC__
    header for the Gitlab wiki.

    Args:
        filename (str): file to write documentation to
        suppliers (dict): dictionary of suppliers
        columns_to_write (list): list of columns to write into documentation
    """

    logger.info('')
    logger.info('[Writing documentatiom]')
    with open(filename, encoding="utf8", mode='w') as outfile:
        outfile.write('[[_TOC_]]\n\n')
        for supplier in suppliers:
            logger.info(f'>> Documenting {supplier}')

            # create and write documentation
            dc.write_markdown_doc(outfile, suppliers, supplier, columns_to_write)

    logger.info('')
    logger.info(f'=== Documentation written to {filename}')

    return

### write_documentation ###


def write_sql(project_name: str,
              outfile: object,
              supplier_config: dict,
              template: pd.DataFrame,
              servers: dict,
              #server_config: dict,
             ):
    """ Iterate over all elements in table and creates a data description

    When a data DataFrame is passed the table itself will be created and the
    data will be stored into the table.

    Args:
        sql_filename -- name of the file to write DDL onto
        tables -- table names as key and per table name points to additional information
        template -- bronbestand_attribuut_meta.csv has column info for create_table_description
        postgres_schema -- table schema name
    """
    postgres_schema = servers['DATA_SERVER_CONFIG']['POSTGRES_SCHEMA']

    #show_supplier_schemas(suppliers)
    supplier_id = supplier_config['supplier_id']
    logger.info('')
    logger.info('[Writing SQL]')
    logger.info(f'>> Writing {supplier_id}')

    # get the meta data, they are needed for saome operations
    meta_data = supplier_config[dc.TAG_TABLES][dc.TAG_TABLE_META]['data']
    for table_type in supplier_config[dc.TAG_TABLES]:

        # get schema file
        schema = supplier_config[dc.TAG_TABLES][table_type][dc.TAG_TABLE_SCHEMA]
        data = supplier_config[dc.TAG_TABLES][table_type][dc.TAG_DATA]

        # id data is not a dataframe, there is no data at all
        if not isinstance(data, pd.DataFrame):
            data = None

        # define the prototypical table name
        table_name = dc.get_table_name(project_name, supplier_id, table_type, '')

        # create SQL for the description table
        if supplier_config[dc.TAG_TABLES][table_type]['create_description']:
            description_tag = table_name + 'description'
            sql_code = create_table_description(
                supplier_config = supplier_config,
                table_type = table_type,
                template = template,
                schema_name = postgres_schema,
                table_name = description_tag,
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
            schema = supplier_config[dc.TAG_TABLES][table_type][dc.TAG_TABLE_SCHEMA]

            # when meta, write the meta contents to the data table
            data = None
            if table_type == dc.TAG_TABLE_META:
                data = supplier_config[dc.TAG_TABLES][dc.TAG_TABLE_META][dc.TAG_DATA]

            # when origin is omitted in config.yaml, supply the default (input: <file>)
            if 'origin' in supplier_config:
                origin = supplier_config['origin']

            else:
                origin = {'input': '<file>'}

            if origin['input'] == '<table>' and table_type == dc.TAG_TABLE_SCHEMA:
                logger.info(f'  > Creating description for table schema: {data_table_tag}')
                if not 'table_name' in origin or origin['table_name'] == '':
                    msg = 'No "table_name" was specified for origin in config.yaml'
                    raise DiDoError(msg)

                if not 'code_bronbestand' in origin or origin['code_bronbestand'] == '':
                    origin['code_bronbestand'] = meta_data.iloc[-1].loc['code_bronbestand']

                if not 'levering_rapportageperiode' in origin or origin['levering_rapportageperiode'] == '':
                    olp = meta_data.iloc[-1].loc['bronbestand_datum_begin'][:4] + '-I'
                    origin['levering_rapportageperiode'] = olp
                    logger.warning(f'!Geen "levering rapportageperiode" in origin/config.yaml. {olp} verondersteld.')

                servers['FOREIGN_SERVER_CONFIG']['table'] = origin['table_name']
                servers['DATA_SERVER_CONFIG']['table'] = data_table_tag

                logger.info(f'Schema:\n{schema}')

                sql_code += use_existing_table(origin,
                                                schema,
                                                servers['FOREIGN_SERVER_CONFIG'],
                                                servers['DATA_SERVER_CONFIG'],
                                                servers['ODL_SERVER_CONFIG'],
                                                )
                logger.info(f'  <table> {data_table_tag}')

            else:
                sql_code += create_table(schema,
                                            data,
                                            data_table_tag,
                                            servers['DATA_SERVER_CONFIG'],
                                            supplier_config,
                                        )

                logger.info(f'  <file> {data_table_tag} (default)')

                # if
            # if

        outfile.write(sql_code)
        outfile.write('\n\n')
    # for

    return

### write_sql ###


def test_for_existing_tables(project_name: str,
                             supplier_config: dict,
                             sql_server_config: dict,
                            ):
    """iterate over all elements in table and creates a data description

    When a data DataFrame is passed the table itself will be created and the
    data will be stored into the table.

    Args:
        project_name (str): name of the project
    """

    results = {}
    any_present = False
    supplier_id = supplier_config['supplier_id']

    # for supplier in supplier_config:
    logger.info(f'[Creating {supplier_id}]')

    for table_type in supplier_config[dc.TAG_TABLES]:
        table_name = dc.get_table_name(project_name, supplier_id, table_type, 'description')
        present, n_rows = st.test_table_presence(table_name, sql_server_config)
        results[table_name] = [present, n_rows]
        any_present = any_present or present

        table_name = dc.get_table_name(project_name, supplier_id, table_type, 'data')
        present, n_rows = st.test_table_presence(table_name, sql_server_config)
        results[table_name] = [present, n_rows]
        any_present = any_present or present

    # for

    return results, any_present

### test_for_existing_tables ###


def create_table_description(supplier_config: dict,
                             table_type: str,
                             template: pd.DataFrame,
                             schema_name: str,
                             table_name: str
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
    logger.info(f"===> {supplier_id} - {table_type}")

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
    # tbd = f'DROP TABLE IF EXISTS {schema_name}.{table_name} CASCADE;\n\n'
    tbd = f'CREATE TABLE {schema_name}.{table_name}\n(\n'
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
            items = col.split(':')
            index_dict[index_name]['columns'].append(items[0].lower())
            if len(items) > 1:
                asc = items[1].upper()
                if asc == 'DESC' or asc == 'DESCENDING':
                    index_dict[index_name]['order'].append(False)
                else:
                    index_dict[index_name]['order'].append(True)
                # if
            # if
        # for
    # for

    # construct the SQL string for creating the index
    pg_schema = server_config['POSTGRES_SCHEMA']
    sql = ''
    for index_name in index_dict.keys():
        sql += f"CREATE INDEX {index_name} ON {pg_schema}.{table_name}\n(\n"
        for i in range(len(index_dict[index_name]['columns'])):
            col = index_dict[index_name]['columns'][i]
            order = index_dict[index_name]['order'][i]
            if order:
                sql += f"    {col} ASC,\n"
            else:
                sql += f"    {col} DESC,\n"
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

    # create variable names and types based on schema
    for idx, row in schema.iterrows():
        line = f'   {row["kolomnaam"]} {row["datatype"]}'
        if len(row['constraints']) > 0:
            line += ' ' + row['constraints']

        data_types += line + ',\n'

        comment = f"COMMENT ON COLUMN {schema_name}.{table_name}.{row['kolomnaam']} IS "
        description = schema.loc[idx, 'beschrijving'].strip()

        if len(description) == 0:
            description = '*** NO DOCUMENTATION PROVIDED ***'

        comment += f"$${description}$$;\n"
        comments += comment

    names = dc.get_table_names(supplier_config['config']['PROJECT_NAME'], supplier_config['supplier_id'])

    # add primary key when defined
    if 'primary_key' in supplier_config and table_name == names[dc.TAG_TABLE_SCHEMA]:
        primary_key = create_primary_key(schema, supplier_config)
        data_types += primary_key + ', '

    # remove last comma and newline
    data_types = data_types[:-2]

    # create a table definition and instruction to read starttabel.csv
    #tbd = f'DROP TABLE IF EXISTS {schema_name}.{table_name} CASCADE;\n\n'
    tbd = f'CREATE TABLE {schema_name}.{table_name}\n(\n'
    tbd += data_types + '\n);\n\n'
    tbd += table_comment + comments + '\n\n'

    # look if a primary key has to be added to the data file
    if 'index' in supplier_config and table_name == names[dc.TAG_TABLE_SCHEMA]:
        sql_index = create_index(schema, data, table_name, server_config, supplier_config)
        if len(sql_index) > 0:
            tbd += sql_index

    if data is not None:
        tbd += create_table_input(data, schema, '', schema_name, table_name)

    logger.debug(tbd)

    return tbd

### create_table ###


def use_existing_table(origin: dict, schema: pd.DataFrame, server_from: dict, server_to: dict, odl_server: dict):
    """ Creates a DiDo data table from an existing table

    This is a somwhat elaborate process
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

    # # now copy the data from the old table to the new table, see template below
    # # insert into tablename (columns) select columns from oldtable;
    # select_columns = select_columns[:-2]

    # insert = f"INSERT INTO {table_to_name} "\
    #          f"({select_columns})\n   SELECT {select_columns}\n" \
    #          f"   FROM {server_from['POSTGRES_SCHEMA']}.{server_from['table']};\n"

    sql = sql[:-2] + '\n);\n\n' + comments + '\n' # + insert

    # The part of cpying the data is moved to dido_import

#     # copy the data for the bootstrap columns

#     # update bronbestand_recordnummer
#     data += f"""
# DROP SEQUENCE IF EXISTS seq;

# CREATE SEQUENCE seq
#     START 1
#     INCREMENT 1;

# UPDATE {table_to_name} SET bronbestand_recordnummer = nextval('seq');
# """

#     # update code_bronbestand
#     data += f"UPDATE {table_to_name} " \
#             f"SET code_bronbestand = \'{origin['code_bronbestand']}\';\n"

#     # fetch levering_rapportageperiode from origin
#     data += f"UPDATE {table_to_name} " \
#             f"SET levering_rapportageperiode = \'{origin['levering_rapportageperiode']}\';\n"

#     # set sysdatum at current year
#     data += f"UPDATE {table_to_name} " \
#             f"SET sysdatum = CURRENT_DATE;\n"

    return sql # + data


def load_supplier_schemas(supplier_config: dict, root: str, work: str) -> dict:
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

    logger.info('')
    logger.info('[Loading supplier schemas])')

    # for supplier in supplier_config.keys():
    supplier_id = supplier_config['supplier_id']
    logger.info(f'=== {supplier_id} ===')
    schema_work = os.path.join(work, 'schemas', supplier_id)
    doc_root = os.path.join(root, 'docs', supplier_id)

    # get the schema files
    files = [f for f in os.listdir(schema_work) if os.path.isfile(os.path.join(schema_work, f))]

    # setup the schemas dictionary by interpreting the read files.
    # Each files has the following format
    # filename.index.csv, index one of [description, meta, data]
    tables[supplier_id] = {}

    for filename in files:
        parts = filename.split('.') # parts being name, index, 'csv'
        # check that extension is .csv and there are exactly 3 parts
        if parts[-1] == 'csv' and len(parts) == 3:

            # index is supplied by the scond part
            index = parts[-2]
            if len(parts) == 3:
                table_idx = f'{index}_name'
                tables[supplier_id][table_idx] = os.path.join(work, 'schemas', supplier_id, filename)

        else:
            errmsg = f' Wrong format of filename: should be name. '
            errmsg += '[description|meta|data].csv'
            raise DiDoError(errmsg)

    # read the data description schema
    df = pd.read_csv(tables[supplier_id]['schema_name'],
                        sep = ';',
                        dtype = str,
                        keep_default_na = False,
                        na_values = []).fillna('')
    supplier_config[dc.TAG_TABLES][dc.TAG_TABLE_SCHEMA][dc.TAG_SCHEMA] = df

    # meta information is a bit weird. The input data is kind of user friendly key-value store
    # it is stored in the metadata tag and in a later phase will be merged from the
    # metadata description file fetched from ODL
    meta = pd.read_csv(tables[supplier_id]['meta_name'],
                        sep = ';',
                        dtype = str,
                        keep_default_na = False).fillna('')
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
        odl_name = f'bronbestand_{table_root}_description'
        table_name = dc.get_table_name(project_name, supplier_id, table_root, 'description')


        # create the table name
        odl_name = 'bronbestand_' + table_root + '_description'

        # if schema is <odl>, fetch the schema from ODL
        if isinstance(table_info[dc.TAG_SCHEMA], str) and table_info[dc.TAG_SCHEMA] == '<odl>':
            # fetch the table from the database and assign as dataframe to schema
            table_info[dc.TAG_SCHEMA] = dc.load_odl_table(odl_name, server_config)

            # change table name as table of this project and this supplier
            table_info['table_name'] = table_name

        # if data is <odl>, fetch the data from ODL
        if isinstance(table_info[dc.TAG_SCHEMA], str) and table_info[dc.TAG_DATA] == '<odl>':
            # create the table name
            odl_name = 'bronbestand_' + table_root + '_data'

            # fetch the table from the database and assign as dataframe to schema
            table_info[dc.TAG_DATA] = dc.load_odl_table(odl_name, server_config)

            # change table name as table of this project and this supplier
            table_info[dc.TAG_DATA] = dc.load_odl_table(odl_name, server_config)
            table_info['data_name'] = dc.get_table_name(project_name, supplier_id, table_root, 'data')

        if key == dc.TAG_TABLE_META:
            # merge the metadata into the dataframe
            #show_supplier_schemas(suppliers)
            df = merge_meta_data_with_description(table_info[dc.TAG_SCHEMA], table_info[dc.TAG_DATA])
            table_info[dc.TAG_DATA] = df

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

        '''
        # !!! PAS OP tables IS NIET BEKEND, MOET NOG AANGEPAST WORDEN!!!
        pad = os.path.join(supplier_config['config']['ROOT_DIR'], dc.DIR_DOCS, supplier_id)
        if len(table_info[dc.TAG_PREFIX]) > 0:
            logger.info(f'   Loading prefix: {table_info[dc.TAG_PREFIX]}')
            prefix = os.path.join(pad, table_info[dc.TAG_PREFIX])

            if os.path.exists(prefix):
                with open(prefix, 'r', encoding="utf8") as infile:
                    supplier_config['prefix_text'] = infile.read().strip()
            else:
                logger.warning(f'!!! Prefix file foes not exist: {prefix}')

        if len(table_info[dc.TAG_SUFFIX]) > 0:
            logger.info(f'   Loading suffix: {table_info[dc.TAG_SUFFIX]}')
            suffix = os.path.join(pad, table_info[dc.TAG_SUFFIX])

            if os.path.exists(suffix):
                with open(suffix, 'r', encoding="utf8") as infile:
                    supplier_config['suffix_text'] = infile.read().strip()
            else:
                logger.warning(f'!!! Suffix file foes not exist: {suffix}')
        '''

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
                logger.info(f'  {prop}: { suppliers[supplier][prop]}')

    return


def dido_begin(config_dict: dict):
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
    # if 'SUPPLIERS_TO_PROCESS' in config_dict.keys():
    #     suppliers_to_process = config_dict['SUPPLIERS_TO_PROCESS']

    # just * means process all
    if suppliers_to_process == '*':
        suppliers_to_process = leveranciers.keys()

    # process each supplier
    for leverancier_id in suppliers_to_process:
        logger.info('')
        logger.info(f'=== {leverancier_id} ===')

        leverancier, deliveries = dc.get_supplier_dict(config_dict, leverancier_id, 1)
        if len(deliveries) > 0:
            logger.info('')
            logger.info('Deliveries supplied (x = chosen)')
            for key in deliveries.keys():
                logger.info(f" - {deliveries[key]['delivery_naam']} " \
                            f"{deliveries[key]['delivery_keus']}")

            # for
            logger.info('')
        # if

        # directories
        # direcory to load files from root_dir
        dir_load = os.path.join(root_dir, 'schemas', leverancier_id)

        # directory to save into work_dir
        dir_save = os.path.join(work_dir, 'schemas', leverancier_id)

        # files
        source_file = leverancier['schema_file']
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
        Fot p-direct the function create_schema_from_pdirekt_datadict is provided.
        """

        data_dict = dc.get_par(leverancier, 'data_dictionary', {})
        if len(data_dict) > 0:
            preprocess_data_dict(data_dict, fname_schema_load, dir_load, leverancier)

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
            sep = ';', # r'\s*;\s*', # warning re seps tend to ignore quoted data
            dtype = str,
            keep_default_na = False,
            engine = 'python',
        ).fillna('')

        """
        Origin is an option that indicates the origin of the data. The options are:
        <file>  - the data are delivered by file
        <table> - the data are delived by a postgres table, only valid for initialization
        <api>   - data are delived via an internet api, one tine initialization and frequent updates
                  when this option is specified, the connection is tested

        Only <table> impacts dido_begin, all three impact dido_data_prep and dido_import
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
            table_leverancier = fetch_schema_from_table(table_name, foreign_server_config)
            schema_leverancier = merge_table_and_schema(table_leverancier, schema_leverancier)
            logger.info(f'Origin is <table>, from {table_name}')

        else:
            raise DiDoError(f'*** Unknown origin input: { origin["input"]}. Only <file>, <table> or <api> allowed.')


        logger.debug(schema_leverancier)
        schema_leverancier = merge_bootstrap_data(schema_leverancier, dc.EXTRA_TEMPLATE, odl_server_config)
        meta_leverancier = pd.read_csv(fname_meta_load, sep=';', dtype=str, keep_default_na=False)
        meta_leverancier = meta_leverancier.fillna('').set_index('attribuut')

        # load the data from the parameters table
        bootstrap_data = dc.load_parameters()

        # fetch the allowed datatypes from dido.yaml
        temp, _ = dc.create_data_types() # bootstrap_data['ALLOWED_POSTGRES_TYPES']
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

        # if data file exists, load and store it
        if os.path.exists(fname_data_load):
            logger.info('[Copying data file]')
            shutil.copy2(fname_data_load, fname_data_save)

    # for

    return

### dido_begin ###


def dido_create(config_dict: dict):
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
    table_desc = config_dict['TABLES']
    write_columns = config_dict['COLUMNS']

    # select which suppliers to process
    suppliers_to_process = suppliers_to_process = dc.get_par(config_dict, 'SUPPLIERS_TO_PROCESS', '*')

    # just * means process all
    if suppliers_to_process == '*':
        suppliers_to_process = leveranciers.keys()

    sql_filename = os.path.join(work_dir, 'sql', 'create-tables.sql')
    doc_filename = os.path.join(work_dir, 'docs', 'create-docs.md')
    any_present = False

    with open(sql_filename, encoding="utf8", mode='w') as sqlfile:
        # create a transaction of tables creation
        sqlfile.write('-- Quit immediately with exit code other than 0 when an error occurs\n')
        sqlfile.write('\\set ON_ERROR_STOP true\n\n')
        sqlfile.write('BEGIN; -- Transaction\n\n')

        # create documentation file and write TOC header
        with open(doc_filename, encoding="utf8", mode='w') as docfile:
            docfile.write('[[_TOC_]]\n\n')

            # copy table_desc as a template for information each leverancier has to deliver
            for leverancier_id in suppliers_to_process:
                logger.info('')
                logger.info(f'=== {leverancier_id} ===')
                logger.info('')

                # count the number of deliveries and fetch sup[plier and delivery accordingly
                delivery_seq = 1
                logger.info(f'Dido_create applies always delivery {delivery_seq}')
                leverancier_config, deliveries = dc.get_supplier_dict(config_dict, leverancier_id, delivery_seq)
                if len(deliveries) > 0:
                    logger.info('Delivery configs supplied in config.yaml (x = chosen)')
                    for key in deliveries.keys():
                        logger.info(f" - {deliveries[key]['delivery_naam']} " \
                                    f"{deliveries[key]['delivery_keus']}")

                    # for
                    logger.info('')
                # if

                # copy table info into leverancier_config
                leverancier_config[dc.TAG_TABLES] = {}
                []
                for table_key in table_desc.keys():
                    # COPY the table dictionary to the supplier dict,
                    # else a shared reference will be copied; use copy() function
                    leverancier_config[dc.TAG_TABLES][table_key]= table_desc[table_key].copy()

                    # copy all keys, as they are string, the are correctly copied
                    for key in table_desc[table_key].keys():
                        leverancier_config[dc.TAG_TABLES][table_key][key] = table_desc[table_key][key]

                # load schema and documentation files and add to leveranciers info
                # leveranciers = load_supplier_schemas(leveranciers, root_dir, work_dir)
                leveranciers = load_supplier_schemas(leverancier_config, root_dir, work_dir)

                # add ODL info from database
                leveranciers = load_supplier_odl(leverancier_config, project_name, odl_server_config)

                # check if the tables exist in the database and warn the user if such is the case
                #@@@@@
                presence, any_present = test_for_existing_tables(project_name, leveranciers, data_server_config)
                if any_present:
                    logger.warning('!!! Er bestaan al tabellen, deze moeten eerst worden vernietigd met "dido_kill_supplier"')
                    dido_list()

                # create SQL to create tables
                meta_table = dc.load_odl_table(table_name = 'bronbestand_attribuut_meta_description',
                                        server_config = odl_server_config)

                write_sql(project_name, sqlfile, leverancier_config, meta_table, db_servers)

                # create documentation
                write_markdown_doc(docfile, leverancier_config, write_columns)

            # for
        # with

        # write the commit statement
        sqlfile.write('\nCOMMIT; -- Transaction\n')

    # with

    dc.report_psql_use('create-tables', db_servers, any_present)

    return

### dido_create ###


def main():
    cpu = time.time()
    dc.display_dido_header('Creating Tables and Documentation')

    # read commandline parameters
    appname, args = dc.read_cli()

    # read the configuration file
    config = dc.read_config(args.project)

    dido_begin(config)
    dido_create(config)

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