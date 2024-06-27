import os
import logging
import psycopg2
import sqlalchemy
import pandas as pd
from dotenv import load_dotenv

# set defaults
DEFAULT_SQL_HOST = 'localhost'
DEFAULT_SQL_PORT = 5432
DEFAULT_SCHEMA_NAME = 'public'
DEFAULT_SQL_USERNAME = ''
DEFAULT_SQL_PASSWORD = ''

logger = logging.getLogger()


def create_sql_engine(host: str,
                      port: str,
                      database: str,
                      username: str = '',
                      password: str = ''
                     ) -> object:
    """to connect with SQL intended only for use within module

    Returns:
        engine object
    """
    prefix = 'postgresql+psycopg2://'
    if len(username) > 0:
        # md5 authentication
        # * not tested on ODC-Noord *
        engine = sqlalchemy.create_engine(f'{prefix}{username}:{password}@{host}:{port}/{database}')

    else:
        # peer id
        engine = sqlalchemy.create_engine(f'{prefix}/{database}')

    return


def dataframe_to_table(data_frame: pd.DataFrame,
                       table_name: str,
                       sql_server_config: dict,
                       ) -> object:
    """ writes a postgres table and returns it as a pandas dataframe.

    Args:
        data_frame (pd.DataFrame): to be written to postgres
        table_name (str): Postgres database table name
        sql_server_config (dict): see description at beginning of this file
    """
    host = sql_server_config.get('host', DEFAULT_SQL_HOST)
    port = sql_server_config.get('port', DEFAULT_SQL_PORT)
    database = sql_server_config.get('POSTGRES_DB', '* No default database *')
    schema = sql_server_config.get('POSTGRES_SCHEMA', DEFAULT_SCHEMA_NAME)
    username = sql_server_config.get('username', DEFAULT_SQL_USERNAME)
    password = sql_server_config.get('password', DEFAULT_SQL_PASSWORD)

    engine = create_sql_engine(
        username = username,
        password = password,
        host = host,
        port = port,
        database = database,
    )

    data_frame.to_sql(
        table_name,
        engine,
        schema = schema,
        if_exists = 'replace',
        index = False,
    )

    logger = logging.getLogger()
    logger.debug('done pushing dataframe to SQL')

    return

### dataframe_to_table ###


def table_to_dataframe(table_name: str,
                       columns: str = None,
                       sql_server_config: dict = {},
                       ) -> pd.DataFrame:
    """ Select a table from PostgreSQL into a Pandas DataFrame

    Args:
        table_name (str, optional): Postgres table name. Defaults to ''.
        columns (str, optional): columns to select. Defaults to '*'.
        sql_server_config (dict): see description at beginning of this file

    Returns:
        pd.DataFrame: postgres table in pandas dataframe
    """
    # set defaults
    host = sql_server_config.get('POSTGRES_HOST', DEFAULT_SQL_HOST)
    port = sql_server_config.get('POSTGRES_PORT', DEFAULT_SQL_PORT)
    database = sql_server_config.get('POSTGRES_DB', '* No default database *')
    schema = sql_server_config.get('POSTGRES_SCHEMA', DEFAULT_SCHEMA_NAME)
    username = sql_server_config.get('POSTGRES_USER', DEFAULT_SQL_USERNAME)
    password = sql_server_config.get('POSTGRES_PW', DEFAULT_SQL_PASSWORD)

    engine = create_sql_engine(username = username,
                               password = password,
                               host = host,
                               port = port,
                               database = database
                              )

    return pd.read_sql_table(
        table_name = table_name,
        con = engine,
        schema = schema,
        coerce_float = True,
        columns = columns
        )


def query_to_dataframe(query: str,
                       columns: str = None,
                       sql_server_config: dict = {},
                       ) -> pd.DataFrame:
    """ Select a table from PostgreSQL into a Pandas DataFrame

    Args:
        table_name (str, optional): Postgres table name. Defaults to ''.
        columns (str, optional): columns to select. Defaults to '*'.
        sql_server_config (dict): see description at beginning of this file

    Returns:
        pd.DataFrame: postgres table in pandas dataframe
    """
    # set defaults
    host = sql_server_config.get('POSTGRES_HOST', DEFAULT_SQL_HOST)
    port = sql_server_config.get('POSTGRES_PORT', DEFAULT_SQL_PORT)
    database = sql_server_config.get('POSTGRES_DB', '* No default database *')
    schema = sql_server_config.get('POSTGRES_SCHEMA', DEFAULT_SCHEMA_NAME)
    username = sql_server_config.get('POSTGRES_USER', DEFAULT_SQL_USERNAME)
    password = sql_server_config.get('POSTGRES_PW', DEFAULT_SQL_PASSWORD)

    engine = create_sql_engine(username = username,
                               password = password,
                               host = host,
                               port = port,
                               database = database
                              )

    try:
        result = pd.read_sql(
            query,
            con = engine,
            coerce_float = True,
        )

    except:
        result = None

    return result

### query_to_dataframe ###


def sql_select(table_name: str = '',
               columns: str = '*',
               where: str = '',
               groupby: str = '',
               limit: int = 0,
               verbose: bool = True,
               sql_server_config: dict = {},
               ) -> pd.DataFrame:
    """ Select a table from PostgreSQL into a Pandas DataFrame

    Args:
        table_name (str, optional): Postgres table name. Defaults to ''.
        columns (str, optional): columns to select. Defaults to '*'.
        where (str, optional): filter statements SQL should start with WHERE.
            Defaults to ''.
        sql_server_config (dict, optional): dictionary containing the following keys:
            Defaults to empty.
        verbose (bool): when True generates extra output. Defaults to True.

    Raises:
        DiDoError: _description_

    Returns:
        pd.DataFrame: postgres table in pandas dataframe
    """
    logger = logging.getLogger()

    # errorprevention
    if not isinstance(table_name, str):
       raise DiDoError('table_name moet string zijn, gegeven:', type(table_name))

    # set defaults
    host = sql_server_config.get('POSTGRES_HOST', DEFAULT_SQL_HOST)
    port = sql_server_config.get('POSTGRES_PORT', DEFAULT_SQL_PORT)
    database = sql_server_config.get('POSTGRES_DB', '* No default database *')
    schema = sql_server_config.get('POSTGRES_SCHEMA', DEFAULT_SCHEMA_NAME)
    username = sql_server_config.get('POSTGRES_USER', DEFAULT_SQL_USERNAME)
    password = sql_server_config.get('POSTGRES_PW', DEFAULT_SQL_PASSWORD)

    engine = create_sql_engine(
        username = username,
        password = password,
        host = host,
        port = port,
        database = database,
    )

    # build query
    sql_query = f"SELECT {columns} "

    if schema and table_name:
        sql_query += f"FROM {schema}.{table_name} "

    if len(where) > 0:
        sql_query += 'WHERE ' + where + ' '

    if len(groupby.strip()) > 0:
        sql_query += 'GROUP BY ' + groupby + ' '

    if limit > 0:
        sql_query += f' LIMIT {limit}'

    sql_query += ';'

    logger.debug('SQL query built: ' + sql_query)

    # read_sql_query does not close the connection and generates an error
    # when repeatedly called. Wrap in with statement to work miracles
    with engine.connect() as connection:
        query_result = pd.read_sql_query(con = connection, # engine.connect(),
                                        sql = sqlalchemy.text(sql_query))


    return query_result

### sql_select ###


def sql_statement(statement: str,
                  verbose = True,
                  sql_server_config: dict = None,
                 ) -> pd.DataFrame:
    """ Executes SQL statement. Does not return result, only
        the number of rows affected.

    Args:
        statement (str): SQL statement to execute
        verbose (bool): when True generates extra output. Defaults to True.
        sql_server_config (dict, optional): dictionary containing the following keys:
            Defaults to empty.

    Raises:
        ValueError: _description_

    Returns:
        pd.DataFrame: postgres table in pandas dataframe
    """
    host = sql_server_config.get('POSTGRES_HOST', DEFAULT_SQL_HOST)
    port = sql_server_config.get('POSTGRES_PORT', DEFAULT_SQL_PORT)
    database = sql_server_config.get('POSTGRES_DB', '* No default database *')
    schema = sql_server_config.get('POSTGRES_SCHEMA', DEFAULT_SCHEMA_NAME)
    username = sql_server_config.get('POSTGRES_USER', DEFAULT_SQL_USERNAME)
    password = sql_server_config.get('POSTGRES_PW', DEFAULT_SQL_PASSWORD)

    # Connect to an existing database
    connection_string = f"host={host} dbname={database} user={username} password={password}"
    connection = psycopg2.connect(connection_string)
    try:
        # Open a cursor to perform database operations
        cursor = connection.cursor()

        # Execute a command: this creates a new table
        cursor.execute(statement)

        # fetch the result
        result = 0
        try:
            cursor.fetchall()
            result = cursor.rowcount
            colnames = [desc[0] for desc in cursor.description]

        except Exception as e:
            logger.error(f'*** sql_statement Exception: {str(e)}')

        # Make the changes to the database persistent
        connection.commit()

        # Close communication with the database

    finally:
        cursor.close()
        connection.close()

    return result

### sql_statement###


def row_count(statement: str,
                  verbose = True,
                  sql_server_config: dict = None,
                 ) -> pd.DataFrame:
    """ Executes SQL statement. Does not return result, only
        the number of rows affected.

    Args:
        statement (str): SQL statement to execute
        verbose (bool): when True generates extra output. Defaults to True.
        sql_server_config (dict, optional): dictionary containing the following keys:
            Defaults to empty.

    Raises:
        ValueError: _description_

    Returns:
        pd.DataFrame: postgres table in pandas dataframe
    """
    host = sql_server_config.get('POSTGRES_HOST', DEFAULT_SQL_HOST)
    port = sql_server_config.get('POSTGRES_PORT', DEFAULT_SQL_PORT)
    database = sql_server_config.get('POSTGRES_DB', '* No default database *')
    schema = sql_server_config.get('POSTGRES_SCHEMA', DEFAULT_SCHEMA_NAME)
    username = sql_server_config.get('POSTGRES_USER', DEFAULT_SQL_USERNAME)
    password = sql_server_config.get('POSTGRES_PW', DEFAULT_SQL_PASSWORD)

    # Connect to an existing database
    connection_string = f"host={host} dbname={database} user={username} password={password}"
    connection = psycopg2.connect(connection_string)
    try:
        # Open a cursor to perform database operations
        cursor = connection.cursor()

        # Execute a command: this creates a new table
        cursor.execute(statement)

        # fetch the result
        result = 0
        cursor.fetchall()
        result = cursor.rowcount
        colnames = [desc[0] for desc in cursor.description]

        # Make the changes to the database persistent
        connection.commit()

        # Close communication with the database

    finally:
        cursor.close()
        connection.close()

    return result


def display_leveranciers(leveranciers: dict):
    """ Displays the supplies of of specific supplier

    Args:
        leveranties (dict): Dictionary of all supplies (by get_leveranties)
        supplier (str): Name of supplier to request list of supplies from
    """
    logger = logging.getLogger()
    logger.info('')
    logger.info(f'[Overzicht van leveranciers]')

    for key in leveranciers.keys():
        logger.info(f' - {key}')

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


def table_exists(table_name: str, verbose: bool = True, sql_server_config: dict = None):
    host = sql_server_config.get('POSTGRES_HOST', DEFAULT_SQL_HOST)
    port = sql_server_config.get('POSTGRES_PORT', DEFAULT_SQL_PORT)
    database = sql_server_config.get('POSTGRES_DB', '* No default database *')
    schema = sql_server_config.get('POSTGRES_SCHEMA', DEFAULT_SCHEMA_NAME)
    username = sql_server_config.get('POSTGRES_USER', DEFAULT_SQL_USERNAME)
    password = sql_server_config.get('POSTGRES_PW', DEFAULT_SQL_PASSWORD)

    statement = f"""SELECT EXISTS(SELECT 1 FROM information_schema.tables
                    WHERE table_catalog='{sql_server_config["POSTGRES_DB"]}' AND
                            table_schema='{sql_server_config["POSTGRES_SCHEMA"]}' AND
                            table_name='{table_name}');"""


    logger = logging.getLogger()
    logger.debug(statement)

    # Connect to an existing database
    connection_string = f"host={host} port={port} dbname={database} user={username} password={password}"
    connection = psycopg2.connect(connection_string)
    try:
        # Open a cursor to perform database operations
        cursor = connection.cursor()

        # Execute a command: this creates a new table
        cursor.execute(statement)

        # fetch the result
        result = cursor.fetchall()
        # result = cursor.rowcount
        colnames = [desc[0] for desc in cursor.description]

        # Make the changes to the database persistent
        connection.commit()

        # Close communication with the database

    finally:
        cursor.close()
        connection.close()

    result = result[0][0]

    return result


def table_contains_data(table_name: str, server_config: dict) -> bool:
    """ Returns True if table contains at least one row, else False

    Args:
        table_name (str): name of the table
        server_config (dict): server and database where the table resides

    Returns:
        bool: True = table contains data, False = no data
    """
    data = sql_select(table_name, limit = 1, sql_server_config = server_config)
    has_data = len(data) > 0

    return has_data


def table_size(table_name: str, verbose: bool = True, sql_server_config: dict = None) -> int:
    """ return number of rows
    """

    host = sql_server_config.get('POSTGRES_HOST', DEFAULT_SQL_HOST)
    port = sql_server_config.get('POSTGRES_PORT', DEFAULT_SQL_PORT)
    database = sql_server_config.get('POSTGRES_DB', '* No default database *')
    schema = sql_server_config.get('POSTGRES_SCHEMA', DEFAULT_SCHEMA_NAME)
    username = sql_server_config.get('POSTGRES_USER', DEFAULT_SQL_USERNAME)
    password = sql_server_config.get('POSTGRES_PW', DEFAULT_SQL_PASSWORD)

    statement = f"SELECT count(*) FROM {schema}.{table_name}"

    logger = logging.getLogger()
    logger.debug(statement)

    # Connect to an existing database
    connection_string = f"host={host} port={port} dbname={database} user={username} password={password}"
    connection = psycopg2.connect(connection_string)
    try:
        # Open a cursor to perform database operations
        cursor = connection.cursor()

        # Execute a command: this creates a new table
        cursor.execute(statement)

        # fetch the result
        result = cursor.fetchall()
        # result = cursor.rowcount
        colnames = [desc[0] for desc in cursor.description]

        # Make the changes to the database persistent
        connection.commit()

        # Close communication with the database

    finally:
        cursor.close()
        connection.close()

    result = result[0][0]

    return result


def test_table_presence(table_name: str, server_config: dict):
    result = table_exists(table_name, verbose = False, sql_server_config = server_config)
    if result:
        rows = table_size(table_name, verbose = False, sql_server_config = server_config)

        return  True, rows

    return False, 0


def duplicate_table(table_name: str, from_server: dict, to_server: dict):
    # create table historie.duplicate as (select * from historie.oud);

    return


def get_structure(table_name: str,
                  verbose = True,
                  sql_server_config: dict = None,
                 ) -> pd.DataFrame:
    """ Returns column name, data type, default value, constraints and comments of a table

    Args:
        statement (str): SQL statement to execute
        verbose (bool): when True generates extra output. Defaults to True.
        sql_server_config (dict, optional): dictionary containing the following keys:
            Defaults to empty.

    Raises:
        ValueError: _description_

    Returns:
        pd.DataFrame: postgres table in pandas dataframe
    """
    host = sql_server_config.get('POSTGRES_HOST', DEFAULT_SQL_HOST)
    port = sql_server_config.get('POSTGRES_PORT', DEFAULT_SQL_PORT)
    database = sql_server_config.get('POSTGRES_DB', '* No default database *')
    schema = sql_server_config.get('POSTGRES_SCHEMA', DEFAULT_SCHEMA_NAME)
    username = sql_server_config.get('POSTGRES_USER', DEFAULT_SQL_USERNAME)
    password = sql_server_config.get('POSTGRES_PW', DEFAULT_SQL_PASSWORD)

    statement = f"""
    SELECT
        cols.column_name as kolomnaam,
        cols.data_type as datatype,
        cols.is_nullable as constraints,
        cols.column_default as verste,
        pg_catalog.col_description(c.oid, cols.ordinal_position::int)
    FROM
        pg_catalog.pg_class c, information_schema.columns cols
    WHERE
        cols.table_catalog = '{database}' AND
        cols.table_schema = '{schema}' AND
        cols.table_name = '{table_name}' AND
        cols.table_name = c.relname;
    """

    logger = logging.getLogger()
    logger.debug(statement)

    # Connect to an existing database
    connection_string = f"host={host} port={port} dbname={database} user={username} password={password}"
    connection = psycopg2.connect(connection_string)
    try:
        # Open a cursor to perform database operations
        cursor = connection.cursor()

        # Execute a command: this creates a new table
        cursor.execute(statement)

        # fetch the result
        result = cursor.fetchall()

        # Make the changes to the database persistent
        connection.commit()

        # Close communication with the database

    finally:
        cursor.close()
        connection.close()

    df = pd.DataFrame(result, columns = ['kolomnaam', 'datatype',
                                         'constraints', 'verstek', 'beschrijving'])

    df.loc[df['constraints'] == 'YES', 'constraints'] = ''
    df.loc[df['constraints'] == 'NO', 'constraints'] = 'NOT NULL'

    return df
