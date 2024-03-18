import os
import sys
import logging
import requests

import pandas as pd

from datetime import datetime, timedelta
from os.path import join, splitext, dirname, basename
from requests.auth import HTTPBasicAuth

# from common import create_log, get_headers_and_types # , read_env, read_schema_file
import dido_common as dc
import simple_table as st

HTTP_OK = 200


def show_account_info(json: list):
    logger = logging.getLogger()
    for dict in json:
        for k in dict:
            logger.info(f'{k}: {dict[k]}')

        # for
        logger.info('')
    # for

    return

### show ###


def get_account_id(url: str, auth):
    # get an accounts request
    response = requests.get(url, auth = auth)

    if response.status_code != HTTP_OK:
        logger.error(f'*** get_account_id response status not ok: {response.status_code}')
        logger.error(response.json())

        return response.status_code, ''

    # if

    # getr account id
    resp = response.json()[0]
    account = resp['accountId']

    return response.status_code, account

### get_account_id ###


def get_account_subscriptions(url: str, auth,
                              account_id: str):

    # get subscriptions for account
    url_subscriptions: str = f"{url}/{account_id}"
    response = requests.get(url_subscriptions,
                            auth = auth,
                           )

    if response.status_code != HTTP_OK:
        logger.error(f'*** get_account_subscriptions response status not ok: {response.status_code}')
        logger.error(response.json())

        return response.status_code, ''

    # if

    return response.status_code, response.json()

### get_account_subscriptions ###


def get_subscriptions(url: str, auth, from_date: str, to_date: str):
    # get a subscription delivery for a certain period
    delivery: dict = {
                    #'accountId': int(account),
                    'deliveryType ': 'mutation',
                    'from': from_date,
                    'to': to_date,
                    # 'after': '20220201',
                    }

    response = requests.get(url,
                            auth = auth,
                            params = delivery,
                           )

    if response.status_code != HTTP_OK:
        logger.error(f'*** get_subscriptions response status not ok: {response.status_code}')
        logger.error(response.json())

        return response.status_code, ''

    # if

    if len(response.json()) <= 0:
        logger.info('*** get_delivery no deliveries pending')
        logger.info(response.json())

        return 0, ''

    # if

    return response.status_code, response.json()

### get_subscriptions ###


def get_deliveries(url: str, auth: object, deliveries: list, download_path: str):
    # initialize downloads
    downloads: list = []

    # iterate over all deliveries
    for delivery in deliveries:
        # create filename from start and end date for each delivery
        begin = delivery['deliverySource']
        end = delivery['deliveryTarget']
        filename = 'mutation_' + begin + '-' + end + '.zip'

        # retrieve download url
        download_url = delivery['downloadUrl']

        # download the stuf
        status = get_download(download_url, auth, filename, download_path)

        # check for errors
        if status != HTTP_OK:
            logger.error('*** get_deliveries response status not ok:', status)

            return status

        # if

        downloads.append(filename)

    # for

    return HTTP_OK, downloads

### get_deliveries ###


def get_download(url: str, auth: object, filename: str, pad: str):

    # download delivery
    response = requests.get(url, auth = auth)

    if response.status_code != HTTP_OK:
        logger.error(f'*** get_download response status not ok: {response.status_code}')
        logger.error(response.json())

        return response.status_code, ''

    # if

    filename = os.path.join(pad, filename)
    with open(filename, 'wb') as outfile:
        logger.info(f'Downloading {url}')
        logger.info(f'to {filename}')
        outfile.write(response.content)

    # if

    return response.status_code

### get_delivery ###


def get_history(data_server: dict):

    history = sql_select('postcodes_history',
                            columns = '*',
                            sql_server_config = data_server,
                        )

    history = history.sort_values(['einddatum'], ascending = True)

    logger.info(history)
    einddatum = history.iloc[-1]['einddatum']

    return einddatum

### get_history ###


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
    bootstrap_data = load_parameters()

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


if __name__ == '__main__':
    # read commandline parameters to create log_file from
    # cli, args = read_cli()

    # read commandline parameters
    app_path, args = read_cli()

    # create logger in project directory
    log_file = os.path.join(args.project, app_path['name'] + '.log')
    logger = create_log(log_file, level = 'DEBUG')

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

    if 'origin' in config_dict['SUPPLIERS']['postcode_nl']:
        origin = config_dict['SUPPLIERS']['postcode_nl']['origin']

        # Read the api keys from environment file
        key = config_dict['KEY']
        secret = config_dict['SECRET']

        # read url's to access the data
        url_account = origin['url_account']
        url_delivery =  origin['url_delivery']

        # authenticate
        authentication = HTTPBasicAuth(key, secret)

        # get other variables: datetime.strptime(dateString, "%d-%B-%Y")
        last_date = origin['start_date']

    else:
        raise ValueError('Origin expected for postcode_nl')

    # make working directory structure
    create_workdir_structure(config_dict, db_servers['ODL_SERVER_CONFIG'])

    # get account id, quit when unsuccesful
    status, number = get_account_id(url_account, authentication)
    if status != HTTP_OK:
        logger.error(f'*** Error ({status}) when trying to login into API account, exit 1')
        sys.exit(1)

    else:
        logger.info(f'==> logged in into account {number}')

    # if

    status, subs = get_account_subscriptions(url_account, authentication, number)
    if status != HTTP_OK:
        logger.error(f'*** Error ({status}) when trying to get subscription from API account, exit 2')
        sys.exit(2)

    else:
        logger.info(f'==> Succesfully obtained subscriptions')
        show_account_info([subs])

    # if

    from_date = (last_date + timedelta(days = 1)).strftime('%Y%m%d')
    to_date = datetime.today().strftime('%Y%m%d')

    logger.info(f'==> Fetching deliveries valid between {from_date} and {to_date}')
    status, delivs = get_subscriptions(url_delivery,
                                       authentication,
                                       from_date,
                                       to_date)
    if status != HTTP_OK:
        logger.error(f'*** Error ({status}) when trying to get pending deliveries, exit 3')
        sys.exit(3)

    else:
        logger.info(f'==> Got info on deliveries to download')
        show_account_info(delivs)

    # if
    print(delivs)
    df = pd.DataFrame(delivs) # .from_dict(delivs, orient='index')
    df.to_csv('deliveries.csv')
    print(df)
    logger.info('[Exit]')

    sys.exit(0)

    logger.info(f'===> Downloading deliveries to {work_dir}')
    status, downloads = get_deliveries(url_delivery, authentication, delivs, work_dir)
    if status != HTTP_OK:
        logger.error(f'*** Error ({status}) downloading deliveries failed, exit 4')
        sys.exit(4)

    else:
        logger.info(f'==> Downloading files succesful: {downloads}')

    # if
