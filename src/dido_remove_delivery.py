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


def get_tables_and_deliveries(supplier_name: str,
                              project_name: str,
                              server_config: dict
                             ):

    # select all tables from schema
    query = "SELECT * FROM information_schema.tables WHERE " \
           f"table_schema = '{server_config['POSTGRES_SCHEMA']}';"
    result = st.query_to_dataframe(query, sql_server_config = server_config)

    # convert all table names into a list
    tables = result['table_name'].tolist()

    # select only tables belonging to this supplier and project
    table_list = []
    identifier = f'{supplier_name}_{project_name}_'
    for table in tables:
        if table.startswith(identifier):
            table_list.append(table)

    data_table_name = f'{identifier}levering_data'
    if data_table_name not in table_list:
        logger.error(f'*** {data_table_name} not found')

        return table_list, None

    else:
        # now select deliveries from leveringen table
        query = f'SELECT {dc.ODL_LEVERING_FREK} \n' \
                f'FROM {server_config["POSTGRES_SCHEMA"]}.{data_table_name};'
        result = st.query_to_dataframe(query, sql_server_config = server_config)
        deliveries = result[dc.ODL_LEVERING_FREK].tolist()

    # if

    return table_list, deliveries

### select_tables_for ###




def dido_remove(header: str):
    cpu = time.time()

    # read commandline parameters
    appname, args = dc.read_cli()

    # read the configuration file
    config_dict = dc.read_config(args.project)

    dc.display_dido_header(header, config_dict)

    # get the database server definitions
    db_servers = config_dict['SERVER_CONFIGS']
    data_server_config = db_servers['DATA_SERVER_CONFIG']

    leveranciers = config_dict['SUPPLIERS']

    dc.show_database(
        server_config = data_server_config,
        pfun = logger.info,
        title = 'Deliveries are removed from the following database',
    )

    # if there is no supplier that received any supply, there is nothing to remove.
    # The program terminates
    supplier_info = dc.get_suppliers(leveranciers)
    supplier_info = dc.add_table_info_to_deliveries(
        suppliers_dict = supplier_info,
        server_config = data_server_config
    )

    # get name of supplier to delete
    supplier_name = dc.get_supplier_name(supplier_info)
    logger.info(f'[Supplier selected is {supplier_name}]')

    # get project name to delete for this supplier
    project_name = dc.get_project_name(supplier_info, supplier_name)
    logger.info(f'[Project selected is {project_name}]')

    condition = 'continue'
    while condition != 'stop':
        tables, deliveries = get_tables_and_deliveries(
            supplier_name = supplier_name,
            project_name = project_name,
            server_config = data_server_config,
        )

        # display alle supplies from the supplier
        logger.info(f'{supplier_name} {project_name} has the following suplies:')
        for  delivery in deliveries:
            logger.info(f' - {delivery}')

        # and ask which supply should be removed
        delivery = ''
        while delivery not in deliveries:
            prompt = 'Select delivery you want to delete (case sensitive): '
            delivery = input(prompt)
            logger.debug(f'{prompt}{delivery}')

            if len(delivery) == 0:
                condition = 'stop'
                break

            elif delivery not in deliveries:
                logger.error(f'*** Delivery does not exist for supplier '
                            f'{supplier_name}: "{delivery}"')
                logger.info('Enter Ctrl-C if you wish to exit')

            else:
                # prepare SQL statements and run these thru simple_table
                logger.info(f'leverancier: {supplier_name}, leverantie: {delivery}')
                table_names = dc.get_table_names(project_name, supplier_name, 'data')
                for table_name in table_names.keys():

                    if table_name != dc.TAG_TABLE_META:
                        name = table_names[table_name]
                        sql = f'DELETE FROM {data_server_config["POSTGRES_SCHEMA"]}.{name}\n' \
                            f'WHERE levering_rapportageperiode = \'{delivery}\'\n' \
                            'RETURNING levering_rapportageperiode;\n'

                        result = st.row_count(sql, sql_server_config = data_server_config)
                        logger.info(f'{result} records destroyed for {data_server_config["POSTGRES_SCHEMA"]}.{name}')
                    # if
                # for
            # if
        # while
    # while

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
