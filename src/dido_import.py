import os
import re
import time
from datetime import datetime
import psutil
import numpy as np
import pandas as pd
import matplotlib

import simple_table as st
import dido_common as dc
from dido_common import DiDoError

import mutate

# pylint: disable=bare-except, line-too-long, consider-using-enumerate
# pylint: disable=logging-fstring-interpolation, too-many-locals
# pylint: disable=pointless-string-statement

# show all columns of dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# raise exceptions on all numpy warnings
np.seterr(all = 'raise')

# some limits
MAX_INSERT: int = 950


def add_error(report, row: int, col: int, col_name: str, err_no: int, explain: str):
    # create empty error report as table imports are not checked
    rows = {'Row': [row],
            'Column': [col],
            'Column Name': [col_name],
            'Error Code': [err_no],
            'Error Message': [explain],
            }

    error_report = pd.DataFrame(rows)

    # existing_report may be none, check
    if report is not None:
        error_report = pd.concat([report, error_report], ignore_index = True)

    # log error
    logger.debug(f'Add error @{row},{col}({col_name}) - {explain} ({err_no})')

    return error_report

### add_error ###


def check_null_iter(data: pd.DataFrame,
                    schema: pd.DataFrame,
                    report: pd.DataFrame,
                    max_errors: int,
                    total_errors: int,
                    report_code: int,
                    column: str,
                   ):
    """ Check NOT NULL for specified  column

    Args:
        data (pd.DataFrame): data to be checked
        schema (pd.DataFrame): schema to check against
        report (pd.DataFrame): existing report to be appended with errors
        report_code (int): code for this error (NOT NULL violated)
        column (str): column to check

    Returns:
        _type_: _description_reate dataframe with dtypes
    """
    if 'NOT NULL' in schema.loc[column, 'constraints']:
        col_index = data.columns.get_loc(column)

        # print('++report++\n', report)
        # print('++row++', report.iloc[0].loc['check_null'])
        for row in range(len(data)):
            # test if an error was already reported, further checks are not very useful
            if len(report) < 1 or report.iloc[row][column] != 0:
                continue

            value = data.iloc[row][column]

            if len(value.strip()) == 0:
                total_errors += 1
                report = add_error(
                    report = report,
                    # mess = messages,
                    row = row,
                    col = col_index,
                    col_name = column,
                    err_no = report_code,
                    explain = 'NULL detected where not allowed'
                )
            # if
        # for

        if total_errors > max_errors:
            raise DiDoError(f'Maximum number of check_null_iter errors {max_errors} '
                            f'exceeded for column {column}. Checking of errors stopped')

    # if

    return report, total_errors

### check_null_iter ###


def check_type(data: pd.DataFrame,
               schema: pd.DataFrame,
               report: pd.DataFrame,
               max_errors: int,
               total_errors: int,
               data_type: str,
               check_function,
               report_code: int,
               column: str
              ):
    """ Checks type of values of a data column

    Args:
        data (pd.DataFrame): imported data
        schema (pd.DataFrame): schema describing the data
        report (pd.DataFrame): current report regarding the data
        total_errors (int): total number of error till now
        check_function (_type_): function to do the type check
        report_code (int): data quality code reflecting a failed test
        column (str): column to check

    Raises:
        DiDoError: when number of errors exceeds a maximum

    Returns:
        pd.DataFrame, int: see description of the parameters
    """
    col_index = data.columns.get_loc(column)

    result = data[column].str.match(check_function)

    if result.sum() != len(result):
        logger.debug(f'{len(result) - result.sum()} errors found in column {column}, type {data_type}')

        idx = result[~result].index
        n_errors = 0
        for row in idx:
            # if total number of errors exceeds the maximum allowed, break the loop
            if total_errors + n_errors > max_errors:
                break

            n_errors += 1

            value = data.iloc[row].loc[column]
            report = add_error(
                report = report,
                row = row,
                col = col_index,
                col_name = column,
                err_no = report_code,
                explain = f'Datatype not {data_type}, value found is "{value}"'
            )
        # for
        total_errors += len(idx)
    # if

    if total_errors > max_errors:
        raise DiDoError(f'Maximum number of check_type errors {max_errors} '
                        f'exceeded for column {column}. Checking of errors stopped')

    return report, total_errors

### check_type ###


def check_data_types(data: pd.DataFrame,
                    schema: pd.DataFrame,
                    report: pd.DataFrame,
                    max_errors: int,
                    total_errors: int,
                   ):
    """"""
    cpu = time.time()
    data_types, _ = dc.create_data_types()
    logger.info('Data type check')

    for col in data.columns:
        check = ''
        n_errors = total_errors
        data_type = schema.loc[col, 'datatype'].lower().strip()
        typ = str(data[col].dtype)
        result = ''

        if typ != 'object':
            logger.debug(f'  data type check skipped for column "{col}", type is: {typ}')
            result = ': skipped'

        elif data_type != 'text':
            if data_type not in data_types.keys():
                logger.error(f'  Onbekend datatype in schema: {data_type}, ignored in check')
                result = ': unknown'

            else:
                check = data_types[data_type]

                report, total_errors = check_type(
                    data = data,
                    schema = schema,
                    report = report,
                    max_errors = max_errors,
                    total_errors = total_errors,
                    data_type = data_type,
                    check_function = check,
                    report_code = dc.VALUE_WRONG_DATATYPE,
                    column = col,
                )
                n_errors = total_errors - n_errors

                result = ': ok'
                if n_errors > 0:
                    result = f': {n_errors} errors'

            # if

        else:
            result = ': ok'

        # if

        logger.info(f'  Data type check "{data_type}" for column {col}{result}')
        if len(check) > 0:
            logger.debug(f'    Tested for: {check}')

    # for

    cpu = time.time() - cpu
    logger.info(f'Checking data types   - errors: {total_errors} CPU: {cpu:.2f} data: [{len(data)}, {len(data.columns)}]')

    return report, total_errors

### check_data_types ###


def check_domain_list(data: pd.DataFrame,
                      schema: pd.DataFrame,
                      report: pd.DataFrame,
                      max_errors: int,
                      total_errors: int,
                      report_code: int,
                      column: str,
                      domain: str
                     ):
    """"""
    # Domain is specified as a list specification: [1, 3.14, 'a'] in the database
    # the code below unpacks the values, spaces and quotes stripped left and right
    domain_list = [x.strip().strip('\"').strip('\'').strip('\"') for x in domain[1:-1].split(',')]
    may_be_null: bool = 'NOT NULL' not in schema.loc[column, 'constraints']
    col_index = data.columns.get_loc(column)

    for row in range(len(data)):
        # test if an error was already reported, further checks are not very useful
        # if report.iloc[row][column] != 0:
        #     continue

        # get the values
        values = data.iloc[row][column].strip()

        # Special case: when empty and NULL is allowed, just don't check
        if values == '' and may_be_null:
            continue

        # values may be a comma separated list, so split at comma
        value_list = values.split(',')

        # test if each value is in the domain
        for value in value_list:
            value = value.strip()
            if not value in domain_list:
                total_errors += 1
                report = add_error( # reate dataframe with dtypes
                    report = report,
                    row = row,
                    col = col_index,
                    col_name = column,
                    err_no = report_code,
                    explain = f'Value not in domain list: "{value}"'
                )

        if total_errors > max_errors:
            raise DiDoError(f'Maximum number of check_domain_list errors {max_errors} '
                            f'exceeded for column {column}. Checking of errors stopped')

    return report, total_errors

### check_domain_list ###


def check_domain_minmax(
    data: pd.DataFrame,
    schema: pd.DataFrame,
    report: pd.DataFrame,
    max_errors: int,
    total_errors: int,
    report_code: int,
    column: str,
    domain: str
):
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        schema (pd.DataFrame): _description_
        report (pd.DataFrame): _description_
        max_errors (int): _description_
        total_errors (int): _description_
        report_code (int): _description_
        column (str): _description_
        domain (str): _description_

    Raises:
        ValueError: _description_
        DiDoError: _description_

    Returns:
        _type_: _description_
    """
    may_be_null: bool = 'NOT NULL' not in schema.loc[column, 'constraints']
    data_type = schema.loc[column, 'datatype'].lower()
    mins, maxs = domain.split(':')
    min_val, max_val = 0, 0
    col_index = data.columns.get_loc(column)

    if data_type in ['integer', 'bigint']:
        min_val, max_val = int(mins), int(maxs)

    elif data_type in ['numeric', 'real', 'double']:
        min_val, max_val = float(mins), float(maxs)

    for row in range(len(data)):
        # test if an error was already reported, further checks are not very useful
        # if report.iloc[row][column] != 0:
        #     continue

        value = data.iloc[row][column]

        # Special case: when empty and NULL is allowed, just don't check
        if value == '' and may_be_null:
            continue

        try:
            if data_type in ['integer', 'bigint']:
                value = int(value)

            elif data_type in ['numeric', 'real', 'double']:
                value = float(value)

            if not min_val <= value <= max_val:
                raise ValueError

        except Exception:
            total_errors += 1
            report = add_error(
                report = report,
                row = row,
                col = col_index,
                col_name = column,
                err_no = report_code,
                explain = f'Value {value} exceeds min {mins} or max {maxs}'
            )

        if total_errors > max_errors:
            raise DiDoError(f'Maximum number of check errors {max_errors} exceeded. Checking of errors stopped')

    return report, total_errors

### check_domain_minmax ###

def check_domain_minmax_2(
    data: pd.DataFrame,
    schema: pd.DataFrame,
    report: pd.DataFrame,
    max_errors: int,
    total_errors: int,
    report_code: int,
    column: str,
    domain: str
):
    """ Check if variables are in thev range [min..max]

    Args:
        data (pd.DataFrame): data frame
        schema (pd.DataFrame): data description
        report (pd.DataFrame): current report (to be appended)
        max_errors (int): maximum number of errors at which the routine stops
        total_errors (int): number of errors detected
        report_code (int): report code of errors detected by this routine
        column (str): column to examine
        domain (str): Domain codedido fatal error

    Raises:
        DiDoError: is raised when total_numebers > max_errors

    Returns:
        pd.DataFrame, int: updated dataframe with errors, updated number of total errors
    """
    # may_be_null: bool = 'NOT NULL' not in schema.loc[column, 'constraints']
    # data_type = schema.loc[column, 'datatype'].lower()
    mins, maxs = domain.split(':')
    min_val, max_val = float(mins), float(maxs)

    temp_s = data[column].astype(float, errors='ignore')
    smaller_min = temp_s < min_val
    greater_max = temp_s > max_val

    ssm = smaller_min.sum()
    sgm = greater_max.sum()
    col_index = data.columns.get_loc(column)

    if ssm > 0:
        ssidx = temp_s[smaller_min.values].index
        for row in ssidx:
            value = temp_s[row]
            total_errors += 1
            report = add_error(
                report = report,
                row = row,
                col = col_index,
                col_name = column,
                err_no = report_code,
                explain = f'Value {value} exceeds min {min_val}'
            )

    if sgm > 0:
        sgidx = temp_s[greater_max.values].index
        for row in sgidx:
            value = temp_s[row]
            total_errors += 1
            report = add_error(
                report = report,
                # mess = messages,
                row = row,
                col = col_index,
                col_name = column,
                err_no = report_code,
                explain = f'Value {value} exceeds max {max_val}'
            )

    if total_errors > max_errors:
        raise DiDoError(f'Maximum number of check_domain_minmax_2 errors {max_errors} '
                        f'exceeded for column {column}. Checking of errors stopped')

    return report, total_errors

### check_domain_minmax_2 ###


def check_domain_re(data: pd.DataFrame,
                    schema: pd.DataFrame,
                    report: pd.DataFrame,
                    max_errors: int,
                    total_errors: int,
                    report_code: int,
                    column: str,
                    domain: str
                   ):
    """ Domain check with regular expression

    Args:
        data (pd.DataFrame): data frame
        schema (pd.DataFrame): schema description
        report (pd.DataFrame): current report (to be appended)
        max_errors (int): maximum number of errors at which the routine stops
        total_errors (int): number of errors detected
        report_code (int): report code of errors detected by this routine
        column (str): column to examine
        domain (str): Domain code

    Raises:
        DiDoError: is raised when total_numebers > max_errors

    Returns:
        pd.DataFrame, int: updated dataframe with errors, updated number of total errors
    """
    may_be_null = 'NOT NULL' not in schema.loc[column, 'constraints']
    pattern = re.compile(domain.strip())
    col_index = data.columns.get_loc(column)
    for row in range(len(data)):
        # test if an error was already reported, further checks are not very useful
        # if report.iloc[row][column] != 0:
        #     continue

        value = data.iloc[row][column]

        # Special case: when empty and NULL is allowed, just don't check
        if value == '' and may_be_null:
            continue

        if not pattern.match(value):
            total_errors += 1
            report = add_error(
                report = report,
                # mess = messages,
                row = row,
                col = col_index,
                col_name = column,
                err_no = report_code,
                explain = f'Value {value} exceeds min {mins} or max {maxs}'
            )

        if total_errors > max_errors:
            raise DiDoError(f'Maximum number of check_domain_re errors {max_errors} '
                            f'exceeded for column {column}. Checking of errors stopped')

    return report, total_errors

### check_domain_re ###


def check_domain(data: pd.DataFrame,
                 schema: pd.DataFrame,
                 report: pd.DataFrame,
                 max_errors: int,
                 total_errors: int,
                ):
    """ Domain check

    Args:
        data (pd.DataFrame): data frame
        schema (pd.DataFrame): schema description
        report (pd.DataFrame): current report (to be appended)
        max_errors (int): maximum number of errors at which the routine stops
        total_errors (int): number of errors detected

    Raises:
        DiDoError: is raised when total_numebers > max_errors

    Returns:
        pd.DataFrame, int: updated dataframe with errors, updated number of total errors
    """
    cpu = time.time()

    for col in data.columns:
        domain = schema.loc[col, 'domein'].strip()

        if len(domain) > 0:
            if domain[0:1] == '[':
                report, total_errors = check_domain_list(
                    data = data,
                    schema = schema,
                    report = report,
                    max_errors = max_errors,
                    total_errors = total_errors,
                    report_code = dc.VALUE_NOT_IN_LIST,
                    column = col,
                    domain = domain,
                )

            elif domain[0:3] == 're:':
                report, total_errors = check_domain_re(
                    data = data,
                    schema = schema,
                    report = report,
                    # messages = messages,
                    max_errors = max_errors,
                    total_errors = total_errors,
                    report_code = dc.VALUE_NOT_CONFORM_RE,
                    column = col,
                    domain = domain[3:],
                )

            elif ':' in domain:
                report, total_errors = check_domain_minmax(
                    data = data,
                    schema = schema,
                    report = report,
                    # messages = messages,
                    max_errors = max_errors,
                    total_errors = total_errors,
                    report_code = dc.VALUE_NOT_BETWEEN_MINMAX,
                    column = col,
                    domain = domain,
                )

            else:
                logger.error(f'*** Unknown domain specification: {domain}')

            # if
        # if
    # for

    cpu = time.time() - cpu
    logger.info(f'Checking domain       - errors: {total_errors} CPU: {cpu:.2f} data: [{len(data)}, {len(data.columns)}]')

    return report, total_errors

### check_domain ###


def check_constraints(data: pd.DataFrame,
                      schema: pd.DataFrame,
                      report: pd.DataFrame,
                      max_errors: int,
                      total_errors: int,
                     ):
    """"""
    # Check for NON NULL
    cpu = time.time()

    for col in data.columns:
        report, total_errors = check_null_iter(
            data = data,
            schema = schema,
            report = report,
            # messages = messages,
            total_errors = total_errors,
            max_errors = max_errors,
            report_code = dc.VALUE_MANDATORY_NOT_SPECIFIED,
            column = col
        )

        if total_errors > max_errors:
            raise DiDoError(f'Maximum number of check_constraints errors {max_errors} exceeded '
                            f'for column {col}. Checking of errors stopped')

    cpu = time.time() - cpu
    logger.info(f'Checking for NON NULL - errors: {total_errors} CPU: {cpu:.2f} data: [{len(data)}, {len(data.columns)}]')

    return report, total_errors

### check_constraints ###


def check_all(data: pd.DataFrame,
              schema: pd.DataFrame,
              supplier_config: dict,
              max_errors: int
             ):
    """ Runs all data checks on data.

    Args:
        data (pd.DataFrame): data to run the checks on
        schema (pd.DataFrame): description of the data
        supplier_config (dict): supply configuration
        max_error (int): maximum # errors before stopping to check for errors

    Returns:
        pd.DataFrame, int: report of all errors, total # of errors
    """
    # create a zero report file
    report = pd.DataFrame(columns = ['Row', 'Column', 'Column Name', 'Error Code', 'Error Message'])
    report['Row'] = report['Row'].astype(int)
    report['Column'] = report['Column'].astype(int)
    report['Error Code'] = report['Error Code'].astype(int)

    dc.report_ram('RAM used before check_all')

    # report = pd.DataFrame(np.zeros(data.shape), columns = data.columns, index = data.index, dtype = np.int32)
    # logger.info(f'[RAM used before messages creation {process.memory_info().rss / (1024 * 1024)}]')
    # messages = pd.DataFrame(columns = data.columns, index = data.index, dtype = str).fillna('')
    # logger.info(f'[RAM used after report creation {process.memory_info().rss / (1024 * 1024)}]')
    total_errors = 0

    # when data_check: no is in the supplier_config, don't check the data
    check = dc.get_par(supplier_config, 'data_check', True)
    if check:
        try:
            report, total_errors = check_constraints(
                data = data,
                schema = schema,
                report = report,
                # messages = messages,
                max_errors = max_errors,
                total_errors = total_errors
            )
            report, total_errors = check_data_types(
                data = data,
                schema = schema,
                report = report,
                # messages = messages,
                max_errors = max_errors,
                total_errors = total_errors
            )
            report, total_errors = check_domain(
                data = data,
                schema = schema,
                report = report,
                # messages = messages,
                max_errors = max_errors,
                total_errors = total_errors
            )

            logger.info(f'Total errors: {total_errors}')

        except DiDoError as error:
            logger.info(error)

    else:
        report = add_error(
            report = report,
            # mess = messages,
            row = 0,
            col = 0,
            col_name = 'Overall message',
            err_no = -3,
            explain = 'Data check is bewust uitgezet, geen datacontrole toegepast',
        )

        logger.warning('!!! Data check is uitgezet, geen datacontrole toegepast')

    # if

    report = add_error(report, 0, 0, '', -1, f'{len(report)} fouten gevonden')

    return report, total_errors

### check_all ###


def generate_statistics(data: pd.DataFrame,
                        stat_config: dict,
                        table: str,
                        data_schema: pd.DataFrame,
                        supplier_config: pd.DataFrame,
                        supplier_id: str,
                       ):

    def get_freqs(variable: pd.Series,
                  var_name: str,
                  db_stats: pd.DataFrame,
                  query: str,
                  plot_pad: str,
                  limit: int,
                 ):
        sql = ''
        freqs = variable.value_counts().sort_values(ascending = False)
        # print(freqs)
        total = freqs.sum()
        if 1 < len(freqs) <= limit:
            db_freqs = st.query_to_dataframe(query, sql_server_config = db_servers['DATA_SERVER_CONFIG'])
            results = {'err': ['-', '-', '-']}
            if db_freqs is not None:
                db_freqs[var_name] = db_freqs[var_name].astype(str).str.replace('-', '')
                db_freqs = db_freqs.set_index(var_name)

                sql += f'\n**Frequency distribution of {len(freqs)} categories for {var_name}**\n\n'
                sql += '| Category | Absolute | Delivery % | Database % |\n'
                sql += '| -------- | -------- | ---------- | ---------- |\n'

                for idx, value in freqs.items():
                    results[idx] = [f'{value}', f'{100 * value / total:.2f}', '']

                for idx, row in db_freqs.iterrows():
                    value = row['percent_total']

                    if idx in results.keys():
                        # print('...', value)
                        results[idx][2] = f'{value:.2f}'

                    else:
                        results[idx] = ['', '', f'{value:.2f}']

                    # if
                # for
            # if

            for key in results.keys():
                s = results[key]
                sql += f'| {key} | {s[0]} | {s[1]} | {s[2]} |\n'

            # for

            """
            # prepare the data for plotting
            results = pd.DataFrame.from_dict(results, orient = 'index')
            results = results.drop([0], axis = 1)
            results = results.reset_index()
            results.columns = [var_name, 'Delivery %', 'Database %']
            results = results.replace(r'^\s*$', np.nan, regex=True)
            results['Delivery %'] = results['Delivery %'].astype(float)
            results['Database %'] = results['Database %'].astype(float)
            # print(results)

            ax = results.plot.bar(x = var_name, y = 'Delivery %', rot = 0)
            filename = os.path.join(plot_pad, var_name) + '.png'
            ax.figure.savefig(filename)
            matplotlib.pyplot.close()

            sql += f'\n![images]({filename})\n\n'
            """

        else:
            sql += f'\nTotal of {len(freqs)} categories thereby not in range [2..{limit}], ignored.\n\n'

        # if

        if db_stats is None:
            logger.info(f'Frequencies for {var_name}')

        else:
            logger.info(f'Statistics for {var_name}, median = {db_stats.loc[0, "median"]:.2f},'
                        f's.d. = {db_stats.loc[0, "std"]:.2f}')

        return sql

    ### get_freqs ###

    # initialize variables
    db_servers = dc.get_par_par(supplier_config, 'config', 'SERVER_CONFIGS', None)
    schema = db_servers['DATA_SERVER_CONFIG']['POSTGRES_SCHEMA']
    extra_schema = dc.load_odl_table(dc.EXTRA_TEMPLATE, db_servers['ODL_SERVER_CONFIG'])
    extra_cols = extra_schema['kolomnaam'].tolist()
    work_dir = dc.get_par_par(supplier_config, 'config', 'WORK_DIR', '')
    plot_pad = os.path.join(work_dir, 'images')
    md = '\n### Statistics\n\n'

    logger.info('')
    logger.info('[Generating statistics]')

    # get which types are reals and integers
    reals = supplier_config['config']['PARAMETERS']['SUB_TYPES']['real']
    ints = supplier_config['config']['PARAMETERS']['SUB_TYPES']['integer']

    # get parameters for statistics
    stat_supplier = stat_config[supplier_id]
    max_categories = dc.get_par(stat_supplier, 'max_categories', 50)
    columns = dc.get_par(stat_supplier, 'columns', '')
    if columns == ['*']:
        columns = data_schema.index.tolist()

    # Iterate over all columns
    for col_name in columns:
        # no statistics for dido meta columns
        if col_name in extra_cols:
            continue

        # setup the markdown variable for this column
        dt = data_schema.loc[col_name, 'datatype']
        md += '----\n'
        md += f'**Statistic analysis for column {col_name} ({dt})**  \n\n'

        # prepare the queries to fetch statistics from the database
        db_col = {'column': col_name, 'schema': schema, 'table': table}
        db_stats = dc.DB_STATS.format(**db_col)
        db_freqs = dc.DB_FREQS.format(**db_col)

        # when a column is interger-like or real-like statistics can be performed
        if dt in reals or dt in ints:
            if dt in reals:
                column = pd.to_numeric(data[col_name], errors = 'coerce')
            elif dt in ints:
                column = pd.to_numeric(data[col_name], errors = 'coerce')
            else:
                column = None
            # if

            stats = st.query_to_dataframe(db_stats, sql_server_config = db_servers['DATA_SERVER_CONFIG'])

            moments = column.agg(['min', 'max', 'mean', 'median', 'std'])

            md += f'\n**Statistics for {col_name}**  \n\n'

            # write markdown table header
            md += '| Statistic | Delivery | Database |  \n'
            md += '| --------- | -------- | -------- |  \n'

            # write statistics
            misdats = column.isna().sum()
            dbval = stats.loc[0, 'n']
            md += f'| N | {len(column)} | {dbval} |\n'
            dbval = stats.loc[0, 'misdat']
            md += f'| Missing data | {misdats} | {dbval} |\n'
            # write markdown info
            for idx, value in moments.items():
                dbval = stats.loc[0, idx]
                md += f'| {idx} | {value:.2f} | {dbval:.2f} |  \n'

            # for

            logger.info(f'Statistics for {col_name}, median = {stats.loc[0, "median"]:.2f}, '
                        f's.d. = {stats.loc[0, "std"]:.2f}')

        # if

        # when data type is not real, frequencies can be computed
        if dt not in reals:
            md += get_freqs(
                variable = data[col_name],
                var_name = col_name,
                db_stats = None,
                query = db_freqs,
                plot_pad = plot_pad,
                limit = max_categories,
            )

    # for

    return md

### generate_statistics ###


def create_markdown(data: pd.DataFrame,
                    table: str,
                    schema: pd.DataFrame,
                    report: pd.DataFrame,
                    pakbon_record: pd.DataFrame,
                    project_name: str,
                    supplier_config: dict,
                    supplier_id: str,
                    report_file: object,
                    filename: str,
                   ):
    """ Generates markdown documentation of errors and save to file

    Args:
        report -- report file
        project_name -- name of the current project
        supplier_id -- name of the supplier
        report_file -- file object for saving all markdown to
        filename -- filename to save only markdown of this specific delivery
    """
    pad, fn, ext = dc.split_filename(filename)
    rapportageperiode = pakbon_record.iloc[0].loc[dc.ODL_LEVERING_FREK]

    md = create_markdown_report(
        report = report,
        rapportageperiode = rapportageperiode,
        project_name = project_name,
        supplier_config = supplier_config,
        supplier_id = supplier_id,
        filename = fn,
    )

    # check if statistics should be generated
    statistics = dc.get_par_par(supplier_config, 'delivery', 'STATISTICS', {})
    if len(statistics) > 0:
        # True, so add statistics to the markdown
        md += generate_statistics(
            data = data,
            stat_config = statistics,
            table = table,
            data_schema = schema,
            supplier_config = supplier_config,
            supplier_id = supplier_id,
        )

    report_file.write(md)

    with open(filename, 'w', encoding="utf8") as outfile:
        outfile.write(md)

    # with

    return

### create_markdown ###


def create_markdown_report(report:pd.DataFrame,
                           rapportageperiode: str,
                           project_name: str,
                           supplier_config: dict,
                           supplier_id: str,
                           filename: str
                          ) -> str:
    """ Generate a string from the error report

    Args:
        report -- dataframe containing the errors
        project_name -- name of the current project
        supplier_id -- name of the current supplier
        filename -- name of the current delivery file

    Returns:
        str: markdown string of the error report
    """
    # get version info
    odl_server_config = supplier_config['config']['SERVER_CONFIGS']['ODL_SERVER_CONFIG']
    odl_meta_data = dc.load_odl_table(
        table_name = 'bronbestand_bestand_meta_data',
        server_config = odl_server_config,
    )

    major_odl = odl_meta_data.loc[0, 'odl_version_major']
    minor_odl = odl_meta_data.loc[0, 'odl_version_minor']
    patch_odl = odl_meta_data.loc[0, 'odl_version_patch']
    odl_date = odl_meta_data.loc[0, 'odl_version_patch_date']
    major_dido = supplier_config['config']['PARAMETERS']['DIDO_VERSION_MAJOR']
    minor_dido = supplier_config['config']['PARAMETERS']['DIDO_VERSION_MINOR']
    patch_dido = supplier_config['config']['PARAMETERS']['DIDO_VERSION_PATCH']
    dido_date = odl_meta_data.loc[0, 'dido_version_patch_date']

    # For markdown: two spaces at end if line garantees a newline
    md = f'# Project: {project_name}\n## Leverancier: {supplier_id}\n'
    md += f'**Levering voor rapportageperiode: {rapportageperiode}**  \n'
    md += f'Data ingelezen van file: {filename}  \n\n'
    md += f'Tijdstip van inlezen: {datetime.now().strftime(dc.DATETIME_FORMAT)}  \n'
    md += f'DiDo versie: {major_dido}.{minor_dido}.{patch_dido} ({dido_date})  \n'
    md += f'ODL versie: {major_odl}.{minor_odl}.{patch_odl} ({odl_date})\n\n'

    # write markdown table header
    for col in report.columns:
        md += f'| {col} '

    md += ' |\n'
    for col in report.columns:
        md += '| ----- '

    md += ' |\n'

    # write markdown info
    for row in report.index:
        for col in report.columns:
            md += f'| {report.loc[row, col]} '

        md += ' |\n'

    return md

### create_markdown_report ###


def create_csv_report(report:pd.DataFrame,
                      report_file:object,
                      filename:str
                     ):
    """"""
    report.to_csv(report_file, sep = ';')
    report.to_csv(filename, sep = ';')

    return

### create_csv_report ###


def create_supply_sql(pakbon_record: pd.DataFrame,
                      schema: pd.DataFrame,
                      schema_name: str,
                      table_name: str
                     ) -> str:
    """ Creates table input from DataFrame or data file.

    The input is generated from the data DataFrame. The schema
    describes the data and the columns in data should exactly match the name and order of the
    names of 'kolomnaam' in schema.

    Args:
        data (pd.DataFrame): DataFrame containing the data
        schema (pd.DataFrame): DataFRame containing description of the data
        schema_name (str): postgres schema name
        table_name (str): postgres table name

    Returns:
        str: SQL statements containing the data generation
    """
    sql = ''
    counter = 0
    schema_cols = schema.index.tolist()

    for idx in pakbon_record.iterrows():
        if isinstance(idx, tuple):
            idx=idx[0]

        # test if max number of INSERT INTO exceeded
        if counter % MAX_INSERT == 0:
            # check if this is the very first time
            if counter > 0:
                sql = sql[:-2] + ';\n\n'

            sql += f'INSERT INTO {schema_name}.{table_name} ('

            # values: str = 'VALUES '
            for col in pakbon_record.columns:
                sql += col + ', '

            sql = sql[:-2] + ')\nVALUES\n'

        row_values = '('

        for col in pakbon_record.columns:
            if col in schema_cols:
                data_type = schema.loc[col, 'datatype']
                if data_type =='text' and col != dc.COL_CREATED_BY:
                    row_values += f'$${str(pakbon_record.loc[idx, col])}$$, '

                elif data_type =='date' or data_type =='timestamp':
                    datum = str(pakbon_record.loc[idx, col])

                    # for date type, empty string is NULL
                    if len(datum) == 0 or datum == 'nan':
                        row_values += 'NULL, '
                    else:
                        if data_type =='date':
                            row_values += 'CURRENT_DATE, '
                        else:
                            row_values += 'CURRENT_TIMESTAMP(0), '

                else:
                    value = str(pakbon_record.loc[idx, col])

                    # for non-text type, empty string is NULL
                    if len(value) == 0:
                        row_values += 'NULL, '
                    else:
                        row_values += f'{value}, '

                # if

            else:
                logger.error(f'*** Data kolomnaam "{col}" komt niet voor in schema: {schema_cols}')
                sql = '* Fouten in de verwerking *\n\n'

                return sql

        # create a row of values to be inserted
        sql += row_values[:-2] + '),\n'

        # increment counter at end of loop
        counter += 1

    sql = sql [:-2] + ';\n\n'

    # print(sql)

    return sql

### create_supply_sql ###


def generate_sql_header(table_name: str,
                        supplier: str,
                        schema: str,
                        key_id: str,
                       ) -> str:

    sql = '-- Quit immediately with exit code other than 0 when an error occurs\n'
    sql += '\\set ON_ERROR_STOP true\n\n'

    # only insert mutation functions when in mutation mode (key_id =/= None)
    if key_id is not None:
        sql += '-- Definition of history helper functions\n'

        values = {'table_name': table_name,
                'supplier': supplier,
                'schema': schema,
                'key_id': key_id,
                }

        in_sql = dc.load_sql()
        sql += in_sql.format(**values) + '\n\n'

    return sql

### generate_sql_header ###


def create_schema_sql(data: pd.DataFrame,
                      supplier_config: dict,
                      filename: str,
                      schema: pd.DataFrame,
                      schema_name: str,
                      pakbon_record: pd.DataFrame,
                      table_name: str,
                      db_servers: dict,
                     ) -> str:
    """ Creates table input from DataFrame or data file.

    The input is generated from the data DataFrame. The schema
    describes the data and the columns in data should exactly match the name and order of the
    names of 'kolomnaam' in schema.

    Args:
        data (pd.DataFrame): DataFrame containing the data
        schema (pd.DataFrame): DataFRame containing description of the data
        schema_name (str): postgres schema name
        table_name (str): postgres table name

    Returns:
        str: SQL statements containing the data generation
    """
    sql = ''
    extra_schema = dc.load_odl_table(dc.EXTRA_TEMPLATE, db_servers['ODL_SERVER_CONFIG'])
    extra_cols = extra_schema['kolomnaam'].tolist()
    schema_cols = data.columns.to_list() # schema.index.tolist()

    # the data only contains the deliverd data, without thge dido meta data cols
    # these cols are in the schema_cols however, so remove the meta data cols
    # from schema_cols
    for item in extra_cols:
        if item in schema_cols:
            schema_cols.remove(item)

    # prepend postgres schema to tablename
    table_to_name = f"{db_servers['DATA_SERVER_CONFIG']['POSTGRES_SCHEMA']}.{table_name}"

    select_columns = ''
    for col in schema_cols:
        select_columns += col + ', '

    # remove last ', ' from the columns string
    if len(select_columns) > 1:
        select_columns = select_columns[:-2]

    # new CSV definition including FORCE_NULL
    # WITH (FORMAT CSV, HEADER, DELIMITER ';', FORCE_NULL(....))
    # First build a list of date, integer and real types
    sub_types = supplier_config['config']['PARAMETERS']['SUB_TYPES']
    numerics = sub_types['integer'] + sub_types['real'] + sub_types['date'] + sub_types['timestamp']

    # build a list with all numeric column names of the data.
    # These should be forced to NULL when field is empty.
    # ODL overhead must be omitted.
    numeric_cols = ''
    for col_name, row in schema.iterrows():
        if col_name not in extra_cols:
            col_type = row['datatype']
            if col_type in numerics:
                #col_name = schema.index[i] # iloc[i].loc['kolomnaam']
                numeric_cols += col_name + ', '

    # if the list of numeric columns, create a force_null addition for the csv
    force_null = ''
    if len(numeric_cols) > 0:
        force_null = ', FORCE_NULL (' + numeric_cols[0:-2] + ')'

    with_clause = f"WITH (FORMAT CSV, DELIMITER ';', HEADER{force_null})"
    sql += f"\copy {table_to_name} ({select_columns}) FROM '{filename}' {with_clause};\n\n"

    # copy the data for the bootstrap columns

    # update bronbestand_recordnummer
    # sql += "DROP SEQUENCE IF EXISTS seq;\n\nCREATE SEQUENCE seq\n    START 1\n    INCREMENT 1;\n\n"

    sql += f"UPDATE {table_to_name}\n" #    SET {dc.ODL_RECORDNO} = nextval('seq'),\n"

    # update code_bronbestand
    code_bronbestand = pakbon_record.iloc[0][dc.ODL_CODE_BRONBESTAND]
    sql += f"    SET {dc.ODL_CODE_BRONBESTAND} = \'{code_bronbestand}\',\n"

    # fetch levering_rapportageperiode from origin
    levering_rapportageperiode = pakbon_record.iloc[0][dc.ODL_LEVERING_FREK]
    sql += f"        {dc.ODL_LEVERING_FREK} = '{levering_rapportageperiode}',\n"

    # initialize record history
    datum_begin = supplier_config['config']['BEGINNING_OF_WORLD']
    datum_einde = supplier_config['config']['END_OF_WORLD']
    sql += f"        {dc.ODL_DATUM_BEGIN} = \'{datum_begin}\',\n" \
           f"        {dc.ODL_DATUM_EINDE} = \'{datum_einde}\',\n"

    # set sysdatum at current time
    sql += f"        sysdatum = CURRENT_TIMESTAMP(0)\n"

    # only process for current added columns
    sql += f"    WHERE ({dc.ODL_LEVERING_FREK} = '') IS NOT FALSE;\n\n"

    return sql

### create_schema_sql ###


def create_table_sql(pakbon_record: pd.DataFrame,
                  sql_file: object,
                  odl_server: dict,
                  server_from: dict,
                  server_to: dict):
    """ Creates a DiDo data table from an existing table

    This is a somwhat elaborate process
    - create new table
    - create the bootstrap columns
    - create the columns of the table to be copied
    - copy the data of the table to be copied
    - fill the bootstrap columns

    and do it all in SQL

    Args:
        origin (dict): meta data to use
        server_from (dict): server properties
        server_to (dict): server properties
    """

    # Read bootstrap data
    bootstrap = dc.load_odl_table(dc.EXTRA_TEMPLATE, odl_server)

    # get the table data
    old_table = st.get_structure(
        table_name = server_from['table'],
        sql_server_config = server_from,
        verbose = False
    )

    table_to_name = f"{server_to['POSTGRES_SCHEMA']}.{server_to['table']}"

    # the DiDo table to receive the data from the table has already been created
    # it has te be filled with data from two sources
    # 1. fill the first 4 columns with dido relevant data
    # 2. fill the successive columns with the import table data
    #
    # we start with 2, copy the import table, determine the columns to copy
    # create a string with all columns
    select_columns = ''
    for col in old_table['kolomnaam'].tolist():
        select_columns += col + ', '

    # remove last ', ' from the columns string
    if len(select_columns) > 1:
        select_columns = select_columns[:-2]

    sql = f"INSERT INTO {table_to_name} "\
             f"({select_columns})\n   SELECT {select_columns}\n" \
             f"   FROM {server_from['POSTGRES_SCHEMA']}.{server_from['table']};\n"

    # copy the data for the bootstrap columns

    # update bronbestand_recordnummer
    data = """\nDROP SEQUENCE IF EXISTS seq;\n\nCREATE SEQUENCE seq\n    START 1\n    INCREMENT 1;\n\n"""

    data += f"UPDATE {table_to_name} SET {dc.ODL_RECORDNO} = nextval('seq');\n"

    # update code_bronbestand
    code_bronbestand = pakbon_record.iloc[0][dc.ODL_CODE_BRONBESTAND]
    data += f"UPDATE {table_to_name} " \
            f"SET {dc.ODL_CODE_BRONBESTAND} = \'{code_bronbestand}\';\n"

    # fetch levering_rapportageperiode from origin
    levering_rapportageperiode = pakbon_record.iloc[0][dc.ODL_LEVERING_FREK]
    data += f"UPDATE {table_to_name} " \
            f"SET {dc.ODL_LEVERING_FREK} = '{levering_rapportageperiode}';\n"

    # set sysdatum at current year, sysdatum is a timestamp
    data += f"UPDATE {table_to_name} " \
            f"SET sysdatum = CURRENT_TIMESTAMP(0);\n"

    return sql + data

### create_table_sql ###


def create_dataquality_sql(dataquality: pd.DataFrame,
                           code_bronbestand: str,
                           rapportageperiode: str,
                           schema: pd.DataFrame,
                           data: pd.DataFrame,
                           data_schema: pd.DataFrame,
                           schema_name: str,
                           table_name: str
                       ) -> str:
    """"""
    # do not modify the dataquality dataframe, use a stand-in
    df = dataquality.copy(deep = True)

    # insert code_bronbestand
    df.insert(1, dc.ODL_CODE_BRONBESTAND, code_bronbestand) # data.loc[0, dc.ODL_CODE_BRONBESTAND])
    df[dc.ODL_SYSDATUM] = 'CURRENT_TIMESTAMP(0)' # datetime.now().strftime('%Y-%m-%d')

    # change column number into column attribute
    df = df.rename(columns = {
        'Row': 'row_number',
        'Column': 'code_attribuut',
        'Column Name': 'column_name',
        'Error Code': 'code_datakwaliteit'}
    )

    # drop onsolete columns and add new columns
    df = df.drop(columns = ['Error Message'])
    df.insert(4, dc.ODL_LEVERING_FREK, value = rapportageperiode, allow_duplicates = True)
    # df.insert(2, 'row_number', df['bronbestand_recordnummer'])
    df.insert(0, dc.ODL_RECORDNO, df['row_number'])

    # let the column order match that of the table (___datakwaliteit_feit_data)
    col_order = ['bronbestand_recordnummer', 'code_bronbestand', 'row_number',
                 'column_name', 'code_attribuut', 'code_datakwaliteit',
                 'levering_rapportageperiode', 'sysdatum']
    df = df[col_order]

    sql_code = create_supply_sql(df, schema, schema_name, table_name)

    return sql_code

### create_dataquality_sql ###


def write_insertion_sql(data: pd.DataFrame,
                        data_filename: str,
                        schema: pd.DataFrame,
                        pakbon_record: pd.DataFrame,
                        dataquality: pd.DataFrame,
                        first_time_table: bool,
                        sql_file: object,
                        single_sql_name: str,
                        tables_name: dict,
                        server_configs: dict,
                        supplier_config: dict,
                       ):
    """
    """

    sql_code = ''
    schema_name = server_configs['DATA_SERVER_CONFIG']['POSTGRES_SCHEMA']
    odl_server_config = server_configs['ODL_SERVER_CONFIG']
    data_server_config = server_configs['DATA_SERVER_CONFIG']
    foreign_server_config = server_configs['FOREIGN_SERVER_CONFIG']

    # fetch the schemas from the database for data quality and delivery
    schema_delivery = dc.load_schema(tables_name[dc.TAG_TABLE_DELIVERY][:-5] + '_description', data_server_config)
    schema_delivery = schema_delivery.set_index('kolomnaam')

    schema_quality = dc.load_schema(tables_name[dc.TAG_TABLE_QUALITY][:-5] + '_description', data_server_config)
    schema_quality = schema_quality.set_index('kolomnaam')

    # generate the SQL code string
    code_bronbestand = pakbon_record.iloc[0].loc[dc.ODL_CODE_BRONBESTAND]
    rapportageperiode = pakbon_record.iloc[0].loc[dc.ODL_LEVERING_FREK]

    sql_code += create_dataquality_sql(
        dataquality = dataquality,
        code_bronbestand = code_bronbestand,
        rapportageperiode = rapportageperiode,
        schema = schema_quality,
        data = data,
        data_schema = schema,
        schema_name = schema_name,
        table_name = tables_name[dc.TAG_TABLE_QUALITY],
    )

    sql_code += create_supply_sql(
        pakbon_record = pakbon_record,
        schema = schema_delivery,
        schema_name = schema_name,
        table_name = tables_name[dc.TAG_TABLE_DELIVERY],
    )
    # when a table has to be ingested, only allowed once, at the first delivery
    if first_time_table:
        sql_code += create_table_sql(
            pakbon_record = pakbon_record,
            sql_file = sql_file,
            odl_server = odl_server_config,
            server_from = foreign_server_config,
            server_to = data_server_config,
        )

    else:
        sql_code += create_schema_sql(
            data = data,
            supplier_config = supplier_config,
            filename = data_filename,
            schema = schema,
            schema_name = schema_name,
            pakbon_record = pakbon_record,
            table_name = tables_name[dc.TAG_TABLE_SCHEMA],
            db_servers = server_configs,
        )
    # if

    # Create sql header, when mode == mutate insert mutation functions
    key_id = dc.get_par_par(supplier_config, 'delivery_type', 'mode', 'insert')
    if key_id != 'mutate':
        key_id = None

    sql_header = generate_sql_header(
        table_name = tables_name[dc.TAG_TABLE_SCHEMA],
        supplier = supplier_config['supplier_id'],
        schema = schema_name,
        key_id = key_id,
    )
    sql_code = sql_header + sql_code

    # write the SQL code to the catch all SQL file
    sql_file.write(sql_code)

    # write code for this delivery only
    with open(single_sql_name, 'w', encoding = "utf8") as outfile:
        outfile.write('BEGIN; -- Transaction\n' + sql_code + 'COMMIT; -- Transaction\n')

    return

### write_insertion_sql ###


def record_insert(table_name: str,
                  data: pd.DataFrame,
                  idx: int,
                  start_datum: datetime,
                  pakbon_record: pd.DataFrame,
                  supplier_config: dict,
                  data_server_config: dict,
                 ) -> str:
    """ creates an sql string to insert a new record into the database

    Args:
        idx (int): index into the database for the values to insert
        data (pd.DataFrame): dataframe containing all mutation instructions
        start_datum (datetime): datetime of the begin time of validity of the record
        pakbon_record (pd.DataFrame): meta data value of the supply

    Returns:
        str: SQL string for the insert
    """
    meta_cols = [dc.ODL_RECORDNO, dc.ODL_CODE_BRONBESTAND, dc.ODL_LEVERING_FREK,
                 dc.ODL_DATUM_BEGIN, dc.ODL_DATUM_EINDE, dc.ODL_SYSDATUM]

    schema = data_server_config['POSTGRES_SCHEMA']

    sql = "-- Insert --\n"
    sql += f"INSERT INTO {schema}.{table_name}("

    # insert column names for all data columns
    for col in data.columns:
        # except for 'bronbestand_recordnummer', that is automatically added (serial)
        if col != dc.ODL_RECORDNO:
            sql += col + ', '
    sql = sql[:-2] + ')\n'

    # add the first six columns as meta data:
    # 'bronbestand_recordnummer', 'code_bronbestand', 'levering_rapportageperiode',
    # 'record_datum_begin', 'record_datum_einde', 'sysdatum'
    sql += "VALUES ($$" + pakbon_record.iloc[0].loc[dc.ODL_CODE_BRONBESTAND] + "$$, "
    sql += "$$" + pakbon_record.iloc[0].loc[dc.ODL_LEVERING_FREK] + "$$, "
    sql += start_datum.strftime("'%Y-%m-%d', ")
    sql += supplier_config['config']['END_OF_WORLD'].strftime("'%Y-%m-%d'")
    sql += ", CURRENT_TIMESTAMP(0), "

    # add the data columns to insert
    for col in data.columns:
        if col not in meta_cols:
            value = data.iloc[idx].loc[col]
            sql += "$$" + value + "$$, "

    sql = sql [:-2] + ");\n\n"

    return sql

### record_insert ###


def record_delete(table_name: str,
                  data: pd.DataFrame,
                  idx: int,
                  eind_datum: datetime,
                  pakbon_record: pd.DataFrame,
                  supplier_config: dict,
                  data_server_config: dict,
                 ) -> str:

    key_col = 'perceelid'
    perceelid = data.iloc[idx].loc[key_col]
    schema = data_server_config['POSTGRES_SCHEMA']
    sql = "-- Delete --\n"
    sql += f"UPDATE {schema}.{table_name}\n"
    sql += f"    SET {dc.ODL_DATUM_EINDE} = '{eind_datum}'\n"
    sql += f"    WHERE {key_col} = {perceelid} AND\n"
    sql += f"          {dc.ODL_DATUM_EINDE} = "
    sql += supplier_config['config']["END_OF_WORLD"].strftime("'%Y-%m-%d'")
    sql += ";\n\n"

    return sql

### record_delete ###


def record_update(table_name: str,
                  data: pd.DataFrame,
                  idx: int,
                  begin_datum: datetime,
                  eind_datum: datetime,
                  pakbon_record: pd.DataFrame,
                  supplier_config: dict,
                  data_server_config: dict,
                 ) -> str:

    sql = "-- Update "
    sql += record_delete(table_name, data, idx, eind_datum, pakbon_record, supplier_config, data_server_config)
    sql += "-- Update "
    sql += record_insert(table_name, data, idx, begin_datum, pakbon_record, supplier_config, data_server_config)

    return sql

### record_update ###


def key_exists(table_name: str,
               mutation_key: str,
               delete: bool,
               value,
               supplier_config: dict,
               server_config: dict,
               ) -> bool:
    """ Tests if a mutatiun_key exists with value

    Args:
        table_name (str): name of the table containing the column
        mutation_key (str): Key
        value (Any): value value to be tested on existence in mutation_key
        server_config (dict): server configuration

    Returns:
        bool: True, value exists, False if not
    """
    schema = server_config['POSTGRES_SCHEMA']
    where = f'{mutation_key} = {value}'
    if delete:
        where += f" AND {dc.ODL_DATUM_EINDE} = '"
        where += f"{supplier_config['config']['END_OF_WORLD']}" + "'"

    result = st.sql_select(
        table_name = table_name,
        columns = mutation_key,
        where = where,
        limit = 1,
        sql_server_config = server_config,
    )

    exists = len(result) > 0

    logger.debug(f'Table {table_name}, column {mutation_key}, value {value} exists: {exists}')

    return exists

### key_exists ###


def write_mutation_sql(data: pd.DataFrame,
                    instructions: pd.DataFrame,
                    data_filename: str,
                    schema: pd.DataFrame,
                    pakbon_record: pd.DataFrame,
                    dataquality: pd.DataFrame,
                    first_time_table: bool,
                    sql_file: object,
                    single_sql_name: str,
                    tables_name: dict,
                    server_configs: dict,
                    supplier_config: dict,
                   ):
    """
    """
    # get mutation parameters
    mutation_info = supplier_config['delivery_type']
    mutation_key = dc.get_par(mutation_info, 'mutation_key')
    mutation_file = os.path.splitext(supplier_config['data_file'])[0]
    start_datum, eind_datum = mutate.generate_start_end_dates(
        method = 'filename',
        base = mutation_file,
        periode = pakbon_record['levering_rapportageperiode']
    )

    # get the server configurations
    schema_name = server_configs['DATA_SERVER_CONFIG']['POSTGRES_SCHEMA']
    odl_server_config = server_configs['ODL_SERVER_CONFIG']
    data_server_config = server_configs['DATA_SERVER_CONFIG']
    foreign_server_config = server_configs['FOREIGN_SERVER_CONFIG']

    # create table_name
    table_name = tables_name[dc.TAG_TABLE_SCHEMA]
    schema_name = data_server_config['POSTGRES_SCHEMA']

    sql_code = ''

    # fetch the schemas from the database for data quality and delivery
    schema_delivery = dc.load_schema(tables_name[dc.TAG_TABLE_DELIVERY][:-5] + '_description',
        data_server_config)
    schema_delivery = schema_delivery.set_index('kolomnaam')

    schema_quality = dc.load_schema(tables_name[dc.TAG_TABLE_QUALITY][:-5] + '_description',
        data_server_config)
    schema_quality = schema_quality.set_index('kolomnaam')

    # generate the SQL code stringhttps://www.programiz.com/python-programming/datetime/strftime
    code_bronbestand = pakbon_record.iloc[0].loc[dc.ODL_CODE_BRONBESTAND]
    rapportageperiode = pakbon_record.iloc[0].loc[dc.ODL_LEVERING_FREK]

    sql_code += create_dataquality_sql(
        dataquality = dataquality,
        code_bronbestand = code_bronbestand,
        rapportageperiode = rapportageperiode,
        schema = schema_quality,
        data = data,
        data_schema = schema,
        schema_name = schema_name,
        table_name = tables_name[dc.TAG_TABLE_QUALITY],
    )

    sql_code += create_supply_sql(
        pakbon_record = pakbon_record,
        schema = schema_delivery,
        schema_name = schema_name,
        table_name = tables_name[dc.TAG_TABLE_DELIVERY],
    )

    # when a table has to be ingested, only allowed once, at the first delivery
    if first_time_table:
        sql_code += create_table_sql(
            pakbon_record = pakbon_record,
            sql_file = sql_file,
            odl_server = odl_server_config,
            server_from = foreign_server_config,
            server_to = data_server_config,
        )
        # sql_code += create_table_sql(pakbon_record, sql_file,
        #                              odl_server_config, foreign_server_config, data_server_config)


    else:
        errors = False
        for idx in instructions.index:
            # instruction contains the mutaion instruction
            # mutation_key is the unique key of each record, may not exist now or in history
            instruction = instructions.iloc[idx, 0].lower()
            value = data.loc[idx, mutation_key]

            # when a record is inserted, its key may not exist
            if instruction == 'insert':
                if not key_exists(
                    table_name = table_name,
                    mutation_key = mutation_key,
                    delete = False,
                    value = value,
                    supplier_config = supplier_config,
                    server_config = data_server_config
                    ):

                    sql_code += record_insert(
                        table_name = table_name,
                        data = data,
                        idx = idx,
                        start_datum = start_datum,
                        pakbon_record = pakbon_record,
                        supplier_config = supplier_config,
                        data_server_config = data_server_config,
                        )

                else:
                    errors = True
                    logger.error(f'*** Insert: value already exists for {mutation_key}: {value}, not inserted')

            # when a record is deleted, its key must exist
            elif instruction == 'delete':
                if key_exists(
                    table_name = table_name,
                    mutation_key = mutation_key,
                    delete = True,
                    value = value,
                    supplier_config = supplier_config,
                    server_config = data_server_config
                    ):

                    sql_code += record_delete(
                        table_name = table_name,
                        data = data,
                        idx = idx,
                        eind_datum = eind_datum,
                        pakbon_record = pakbon_record,
                        supplier_config = supplier_config,
                        data_server_config = data_server_config,
                        )

                else:
                    errors = True
                    logger.error(f'*** Delete: Key {mutation_key} does not exist for value: {value}, not deleted')

            # an record to update must exist
            elif instruction == 'update':
                if key_exists(
                    table_name = table_name,
                    mutation_key = mutation_key,
                    delete = True,
                    value = value,
                    supplier_config = supplier_config,
                    server_config = data_server_config
                    ):

                    sql_code += record_update(
                        table_name = table_name,
                        data = data,
                        idx = idx,
                        begin_datum = start_datum,
                        eind_datum = eind_datum,
                        pakbon_record = pakbon_record,
                        supplier_config = supplier_config,
                        data_server_config = data_server_config,
                        )

                else:
                    errors = True
                    logger.error(f'*** Update: Key {mutation_key} has no value: {value}, not updated')

            elif instruction == 'downdate':
                logger.debug('Downdate: ignored')

            else:
                errors = True
                logger.error(f'*** Onbekende mutatie-instructie: {instruction}')

            # if
        # for
    # if

    logger.info('')

    # write the SQL code to the catch all SQL file
    if errors:
        logger.error('!!! Fouten gevonden, bepaal zelf of je de SQL wilt gebruiken')

    sql_header = generate_sql_header(
        table_name = table_name,
        supplier = supplier_config['supplier_id'],
        schema = schema_name,
        key_id = mutation_key,
    )

    sql_code = sql_header + sql_code
    sql_file.write(sql_code)
    logger.info('[SQL weggeschreven]')

    # write code for this delivery only
    supplier = supplier_config['supplier_id']
    with open(single_sql_name, 'w', encoding = "utf8") as outfile:
        outfile.write(f'BEGIN; -- For supplier {supplier}\n' + sql_code + 'COMMIT;\n')

    return

### write_mutation_sql ###


def test_levering_rapportageperiode(period: str) -> bool:
    """ Test if a string matches the delivery period string using regex: \d{4}-[J|H|Q|M|W|D|A|I]\d*

    Args:
        period (str): the period to match

    Returns:
        bool: True matches, False not
    """

    if re.match(r"\d{4}-[H|Q|M|W|D]\d", period) or \
       re.match(r"\d{4}-[J|A|I]", period):
        return True

    return False

### test_levering_rapportageperiode ###


def prepare_delivery_note(meta_data: pd.DataFrame,
                          n_records: int,
                          first_time_table: bool,
                          supplier_config: dict,
                          supplier_id: str,
                          tables_name: dict,
                          server_config: dict,
                         ) -> pd.DataFrame:
    """ fills a 'pakbon_record' with defaults an overwrites these with actual
        data from delivery note (pakbon).

    The following rows are mandatory in a pakbon: 'levering_rapportageperiode', 'code_bronbestand' and
    'levering_leveringsdatum'. Levering_rapportageperiode may not occur in the database,
    this will be checked. Code_bronbestand must be an existing supplier, will be chcked as well.

    Args:
        pakbon (pd.DataFrame): required delivery note data
        suppliers (list): list of allowable suppliers
        tables_name (dict): Contains the table info
        server_config (dict): dictionary with postgres access info

    Returns:
        pd.DataFrame: adjusted pakbon
    """
    # fetch data on previous deliveries
    vorige_leveringen = st.sql_select(
        table_name = tables_name[dc.TAG_TABLE_DELIVERY],
        columns = '*',
        sql_server_config = server_config,
        verbose = False,
    )

    # build a list of required rows in the delivery note
    required_columns = [dc.ODL_LEVERING_FREK, dc.ODL_CODE_BRONBESTAND, dc.ODL_DELIVERY_DATE]

    # create a pakbon record containing all necessary fields
    pakbon_record = pd.DataFrame(columns = vorige_leveringen.columns)

    errors = False
    critical = False

    # test if levering_rapportageperiode has the correct formay YYYY-<char><number>
    current_period = supplier_config[dc.ODL_LEVERING_FREK]
    if not test_levering_rapportageperiode(current_period):
        logger.error(f'Incorrect format for "{dc.ODL_LEVERING_FREK}" rapportageperiode: {current_period}')
        critical = True

    # test if the letters of levering_rapportageperiode agree
    defined_period = meta_data.iloc[0].loc['bronbestand_frequentielevering']
    try:
        delivery_period = current_period
        delivery_period = delivery_period[5]

    except:
        logger.critical(f'*** Wrong delivery format in delivery file: {current_period}')
        logger.critical(f'*** Format should be: YYYY-{defined_period}<int>')
        raise DiDoError('Further processing not useful')

    if delivery_period != defined_period:
        # it's an error if code_bronbestand does not match, except when first_table
        # is True and code_brombestand == 'I", that is an initial delivery
        if first_time_table:
            if delivery_period != 'I':
                logger.error(f'*** "{dc.ODL_LEVERING_FREK}" "{delivery_period}" does not match definition: "{defined_period}"')
                critical = True

    # Check code_bronbestand
    code_bronbestand = meta_data.iloc[0].loc[dc.ODL_CODE_BRONBESTAND]
    try:
        code_sourcefile = supplier_config[dc.ODL_CODE_BRONBESTAND]

    except KeyError:
        raise DiDoError(f'There is no "{dc.ODL_CODE_BRONBESTAND}" in the delivery.yaml file')

    # try..except

    if code_sourcefile != code_bronbestand:
        logger.error(f'*** "{dc.ODL_CODE_BRONBESTAND}" "{code_sourcefile}" does not match definition: "{code_bronbestand}"')
        critical = True

    # assign maximum to top
    n_rows, _ = vorige_leveringen.shape
    if n_rows < 1:
        top = 1
    else:
        top = max(vorige_leveringen['levering_rapportageperiode_volgnummer']) + 1

    levering_refusal = 'Not applicable'
    if errors:
        levering_refusal = 'Errors in data quality'

    # fetch the contents of the config file
    root_dir = supplier_config['config']['ROOT_DIR']
    config_filename = os.path.join(os.path.dirname(root_dir), 'config', 'config.yaml')
    with open(config_filename, 'r') as infile:
        config_contents = str(infile.read())

    # get the data filename
    data_filename = supplier_config['data_file']

    # fills out all fields of the pakbon
    pakbon = {}
    pakbon[dc.ODL_LEVERING_FREK] = supplier_config[dc.ODL_LEVERING_FREK]
    pakbon[dc.ODL_CODE_BRONBESTAND] = supplier_config[dc.ODL_CODE_BRONBESTAND]
    pakbon[dc.COL_CREATED_BY] = 'current_user'
    pakbon['levering_leveringsdatum'] = supplier_config['levering_leveringsdatum']
    pakbon['levering_rapportageperiode_volgnummer'] = top
    pakbon['levering_goed_voor_verwerking'] = True
    pakbon['levering_reden_niet_verwerken'] = levering_refusal
    pakbon['levering_verwerkingsdatum'] = datetime.now().strftime('%Y-%m-%d')
    pakbon['levering_aantal_records'] = n_records
    pakbon['config_file'] = config_contents
    pakbon['data_filenaam'] = data_filename
    pakbon[dc.ODL_SYSDATUM] = 'CURRENT_TIMESTAMP(0)'

    pakbon_record = pd.DataFrame([pakbon])

    if critical:
        raise DiDoError(f'Errors detected. Delivery for {supplier_id} is not processed.')

    return pakbon_record

### prepare_delivery_note ###


def process_supplier(supplier_config: dict,
                     config: pd.DataFrame,
                     error_codes: pd.DataFrame,
                     project_name: str,
                     origin: dict,
                     workdir: str,
                     csv_file_object: object,
                     doc_file_object: object,
                     sql_file_object: object,
                     server_configs: dict,
                    ):
    """ Function to handle one supplier

    Args:
        supplier (str): key to current supplier
        config (pd.DataFrame): config dictionary
        error_codes (pd.DataFrame): dataframe containing error codes
        project_name (str): name of the project
        workdir (str): current working directory
        csv_file (object): file to store errors in .csv format
        doc_file (object): file to store errors in markdown format
        sql_file (object): file to store the sql storage instructions
        server_config (dict): dictionary containing postgres access data

    Raises:
        DiDoError: when error were detected that violate the integrity of the information
    """
    supplier_id = dc.get_par(supplier_config, 'supplier_id')
    logger.info('')
    logger.info(f'=== {supplier_id} ===')

    max_errors = supplier_config['delivery']['LIMITS']['max_errors']
    sample_size = supplier_config['delivery']['LIMITS']['max_rows']
    tables_name = dc.get_table_names(project_name, supplier_id, 'data')

    data_server = server_configs['DATA_SERVER_CONFIG']
    first_time_table_import = False
    schema_name = dc.get_table_name(project_name, supplier_id, dc.TAG_TABLE_SCHEMA, 'description')

    foreign_server = server_configs['FOREIGN_SERVER_CONFIG']

    supplier_schema = dc.load_schema(schema_name, data_server)
    supplier_schema = supplier_schema.set_index('kolomnaam')

    # schemas_dir is in the workdir directory
    todo_dir = os.path.join(workdir, dc.DIR_TODO, supplier_id)
    docs_dir = os.path.join(workdir, dc.DIR_DOCS, supplier_id)
    sql_dir = os.path.join(workdir, dc.DIR_SQL, supplier_id)

    # retrieve all files from current supplier in the todo directory
    todo_files = [f for f in os.listdir(todo_dir) if os.path.isfile(os.path.join(todo_dir, f))]

    # retrieve data file from config for current supplier
    if origin['input'] == '<table>':
        if st.table_contains_data(tables_name[dc.TAG_TABLE_SCHEMA], data_server):
            tbl_nm = data_server['POSTGRES_DB'] + '::' + \
                data_server['POSTGRES_SCHEMA'] + '.' + tables_name[dc.TAG_TABLE_SCHEMA]
            logger.error('*** A table can only be imported into an empty table.')
            logger.error(f'*** The table to import into: "{tbl_nm}", contains data.')

            return

        first_time_table_import = True
        logger.info(f"Data is in {data_server['POSTGRES_DB']}::{data_server['POSTGRES_SCHEMA']}" \
                    f".{origin['table_name']}")
        filename = os.path.join(todo_dir, origin['table_name']) + '.pakbon'

        # add table names to server configs
        data_server['table'] = tables_name[dc.TAG_TABLE_SCHEMA]
        foreign_server['table'] = origin['table_name']

    else:
        supplier_file = dc.get_par(supplier_config, 'data_file')
        _, sfile, sext = dc.split_filename(supplier_file)
        filename = os.path.join(todo_dir,sfile + sext)

        logger.info(f'Data to be found in {filename}')
        # data_encoding = supplier_data['data_encoding']

        if 'data_file' not in supplier_config:
            logger.warning(f'!!! Geen data file voor leverancier: {supplier_id}')

            return

        # if
    # if

    logger.info('[Reading current state of the database]')
    # fetch a dataframe with deliveries from the data table using distinct
    # on levering_rapportageperiode
    leveringen = st.sql_select(
        table_name = tables_name[dc.TAG_TABLE_DELIVERY],
        columns = f'DISTINCT {dc.ODL_LEVERING_FREK}',
        sql_server_config = server_configs['DATA_SERVER_CONFIG'],
        verbose = False,
    )

    # fetch meta data from database
    meta_data = st.sql_select(
        table_name = tables_name[dc.TAG_TABLE_META],
        columns = '*',
        sql_server_config = server_configs['DATA_SERVER_CONFIG'],
        verbose = False,
    )

    # split filename for reuse
    pad, fn, ext = dc.split_filename(filename)

    # prepare the delivery info
    logger.info('[Preparing the delivery]')
    pakbon_record = prepare_delivery_note(
        meta_data = meta_data,
        n_records = 0, # len(data),
        first_time_table = first_time_table_import,
        supplier_config = supplier_config,
        supplier_id = supplier_id,
        tables_name = tables_name,
        server_config = data_server
    )

    # integrate the levering_rapportageperiode into the file name
    rapportageperiode = pakbon_record.iloc[0].loc[dc.ODL_LEVERING_FREK] + '_'
    single_doc_name = os.path.join(docs_dir, rapportageperiode + fn + '.md')
    single_csv_name = os.path.join(docs_dir, rapportageperiode + fn + '.csv')
    single_sql_name = os.path.join(sql_dir, rapportageperiode + fn + '.sql')

    # process all files in todo for the current supplier
    if first_time_table_import:
        process_table(
            tablename = origin['table_name'],
            supplier_config = supplier_config,
            supplier_data_schema = supplier_schema,
            tables_name = tables_name,
            pakbon_record = pakbon_record,
            error_codes = error_codes,
            max_errors = max_errors,
            project_name = project_name,
            csv_file = csv_file_object,
            doc_file = doc_file_object,
            sql_file = sql_file_object,
            single_csv_name = single_csv_name,
            single_doc_name = single_doc_name,
            single_sql_name = single_sql_name,
            server_configs = server_configs,
        )
    else:
        if os.path.basename(filename) in todo_files:
            pad, fn, ext = dc.split_filename(filename)
            if ext.strip().lower() == '.csv':
                process_file(
                    filename = filename,
                    supplier_config = supplier_config,
                    supplier_data_schema = supplier_schema,
                    tables_name = tables_name,
                    pakbon_record = pakbon_record,
                    error_codes = error_codes,
                    max_errors = max_errors,
                    sample_size = sample_size,
                    project_name = project_name,
                    csv_file = csv_file_object,
                    doc_file = doc_file_object,
                    sql_file = sql_file_object,
                    single_csv_name = single_csv_name,
                    single_doc_name = single_doc_name,
                    single_sql_name = single_sql_name,
                    server_configs = server_configs,
                )

                logger.info(f'[File {filename} processed]')

            else:
                logger.warning(f'!!! File {filename} is not a .csv file, it is not processed')

        else:
            logger.warning(f'!!! File {filename} neither in todo nor a <table>. Is the data prepared ok?')

    # if

    return

### process_supplier ###


def process_table(tablename: str,
                supplier_config: dict,
                supplier_data_schema: pd.DataFrame,
                tables_name: dict,
                pakbon_record: pd.DataFrame,
                error_codes: pd.DataFrame,
                max_errors: int,
                project_name: str,
                csv_file: object,
                doc_file: object,
                sql_file: object,
                single_csv_name: str,
                single_doc_name: str,
                single_sql_name: str,
                server_configs: dict,
            ):
    """ Imports a table from another schema into current schema and
        embeds it in the DiDo functionality.

    Parameter server_configs contains information concerning the whereabouts of
    the tables. server_configs['DATA_SERVER_CONFIG'] contains the necessary
    information for the DiDo server (host, schema, credentials, etc) and
    server_configs['FOREIGN_SERVER_CONFIG'] those for the table
    to be imported.

    Args:
        tablename (str): Name of the table
        supplier_config (dict): configuration dict for this supplier
        supplier_data_schema (pd.DataFrame): schema for this supplier
        tables_name (dict): names of all tables concerned
        pakbon_record (pd.DataFrame): delivery note in record format (i.e. one row of info)
        error_codes (pd.DataFrame): error codes used for error checks
        max_errors (int): maximum number of errors allowed before crashing
        project_name (str): self evident
        csv_file (object): file to write errors in csv format
        doc_file (object): file to write errors in markdown format
        sql_file (object): file to write all SQL instructions
        single_csv_name (str): file name to write errors in csv format for this supplier
        single_doc_name (str): file name to write errors in markdown format for this supplier
        single_sql_name (str): file name to write SQL instructions for this supplier
        server_configs (dict): configuration definition for all servers

    Raises:
        DiDoError: write critical error into logfile and quits application
    """

    foreign_server = server_configs['FOREIGN_SERVER_CONFIG']
    data_server = server_configs['DATA_SERVER_CONFIG']
    import_table_name = foreign_server['POSTGRES_SCHEMA'] + '.' + tablename

    logger.info('')
    logger.info(f"[Table: {import_table_name}")

    try:
        # create empty error report as table imports are not checked
        supplier_config['data_check'] = False
        report, total_errors = check_all(
            data = data,
            schema = supplier_data_schema,
            supplier_config = supplier_config,
            max_errors = max_errors,
        )
        error_report = add_error(
            report = report,
            row = 0,
            col = 0,
            col_name = 'General message',
            err_no = -2,
            explain = f'Initiele levering uit Table {tablename}. Geen controles uitgevoerd.'
        )
        # error_report = convert_errors_to_dataframe(report, messages, error_codes, total_errors)

        # write errors to file
        create_markdown(
            report = error_report,
            pakbon_record = pakbon_record,
            project_name = project_name,
            supplier_config = supplier_config,
            supplier_id = supplier_data_schema,
            report_file = doc_file,
            filename = single_doc_name
        )
        create_csv_report(error_report, csv_file, single_csv_name)

        # write all modifications as SQL
        write_insertion_sql(
            data = None,
            data_filename = supplier_data_schema,
            pakbon = pakbon_record,
            dataquality = error_report,
            first_time_table = True,
            sql_file = sql_file,
            single_sql_name = single_sql_name,
            tables_name = tables_name,
            server_configs = server_configs,
        )

    except DiDoError as e:
        logger.error(f'*** {str(e)}')
        logger.error(f'*** No data processed for {leveranciers_info}')

    # try..except

    return

### process_table ###


def process_file(filename: str,
                supplier_config: dict,
                supplier_data_schema: pd.DataFrame,
                tables_name: dict,
                pakbon_record: pd.DataFrame,
                error_codes: pd.DataFrame,
                max_errors: int,
                sample_size: int,
                project_name: str,
                csv_file: object,
                doc_file: object,
                sql_file: object,
                single_csv_name: str,
                single_doc_name: str,
                single_sql_name: str,
                server_configs: dict,
            ) -> None:

    """ Reads a file for a specific supplier,
        checks its contents and write SQL and markdown.

    Args:
        filename (str): name of the file to read
        supplier_config (dict): configuration dict for this supplier
        supplier_data_schema (pd.DataFrame): schema for this supplier
        tables_name (dict): names of all tables concerned
        pakbon_record (pd.DataFrame): delivery note in record format (i.e. one row of info)
        error_codes (pd.DataFrame): error codes used for error checks
        max_errors (int): maximum number of errors allowed before crashing
        project_name (str): self evident
        csv_file (object): file to write errors in csv format
        doc_file (object): file to write errors in markdown format
        sql_file (object): file to write all SQL instructions
        single_csv_name (str): file name to write errors in csv format for this supplier
        single_doc_name (str): file name to write errors in markdown format for this supplier
        single_sql_name (str): file name to write SQL instructions for this supplier
        server_configs (dict): configuration definition for all servers

    Raises:
        DiDoError: write critical error into logfile and quits application
    """

    logger.info('')
    logger.info(f'[File: {filename}]')

    # get supplier name
    supplier_id = dc.get_par(supplier_config, 'supplier_id')
    delimiter = dc.get_par(supplier_config, 'delimiter', ';')

    # read the data and perform data checks.
    logger.info('[Reading data]')
    if sample_size < 1:
        sample_size = None

    data = pd.read_csv(
        filename,
        sep = ';',
        dtype = str,
        keep_default_na = False,
        header = 0,
        nrows = sample_size,
        engine = 'c',
        encoding = 'utf8',
    )

    process = psutil.Process()
    logger.info(f'[{len(data)} records and {len(data.columns)} columns read]')
    logger.info(f'[RAM used {process.memory_info().rss / (1024 * 1024)}]')

    # get the delivery_type, if omitted use mode: insert as default
    delivery_type = dc.get_par(supplier_config, 'delivery_type', {'mode': 'insert'})

    logger.debug(f'Columns are: {data.columns}')
    # when delivery_type mode is mutate: separate update instructions from data
    if delivery_type['mode'] == 'mutate':
        # get the column that contain the mutation instructions
        instruction_columns = delivery_type['mutation_instructions']
        logger.debug(f'Mutation instructions columns: {instruction_columns}')

        # make a copy of the mutation instructions as a new dataframe
        instruction_set = data[instruction_columns].copy()

        # now create a list of columns not included in instruction_columns
        data_columns = [x for x in data.columns if x not in instruction_columns]

        # select only the data columns from data (without instructions)
        data = data[data_columns]

    # if

    # change the name of the columns of the delivered data file; it was changed for the table columns
    # but must be changhed for each delivery
    data.columns = [dc.change_column_name(col) for col in data.columns]

    # read the delivery note and check its consistency
    try:
        # prepare data for checking and writing
        logger.info('[Data preparation is ready]')

        # check the data
        error_report, total_errors = check_all(
            data = data,
            schema = supplier_data_schema,
            supplier_config = supplier_config,
            max_errors = max_errors,
        )

        # write errors to file
        create_markdown(
            data = data,
            table = tables_name[dc.TAG_TABLE_SCHEMA],
            schema = supplier_data_schema,
            report = error_report,
            pakbon_record = pakbon_record,
            project_name = project_name,
            supplier_config = supplier_config,
            supplier_id = supplier_id,
            report_file = doc_file,
            filename = single_doc_name,
        )
        create_csv_report(error_report, csv_file, single_csv_name)

        # write all modifications as SQL
        if delivery_type['mode'] == 'insert':
            write_insertion_sql(
                data = data,
                data_filename = filename,
                schema = supplier_data_schema,
                pakbon_record = pakbon_record,
                dataquality = error_report,
                first_time_table = False,
                sql_file = sql_file,
                single_sql_name = single_sql_name,
                tables_name = tables_name,
                server_configs = server_configs,
                supplier_config = supplier_config,
            )

        elif delivery_type['mode'] == 'mutate':
            write_mutation_sql(
                data = data,
                instructions = instruction_set,
                data_filename = filename,
                schema = supplier_data_schema,
                pakbon_record = pakbon_record,
                dataquality = error_report,
                first_time_table = False,
                sql_file = sql_file,
                single_sql_name = single_sql_name,
                tables_name = tables_name,
                server_configs = server_configs,
                supplier_config = supplier_config,
            )

        else:
            raise DiDoError(f"Onbekende delivery_type mode: {delivery_type['mode']}")

    except DiDoError as e:
        logger.error(f'*** {str(e)}')
        logger.error(f'*** No data processed for {leveranciers_info}')

    # try..except

    return

### process_file ###


def dido_import(header: str):
    cpu = time.time()

    # read commandline parameters
    appname, args = dc.read_cli()

    # read the configuration file
    config_dict = dc.read_config(args.project)
    dc.display_dido_header(header, config_dict)
    delivery_filename = args.delivery

    delivery_config = dc.read_delivery_config(config_dict['ROOT_DIR'], delivery_filename)
    overwrite = dc.get_par(delivery_config, 'ENFORCE_IMPORT_IF_TABLE_EXISTS', False)

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

    create_zip = dc.get_par_par(delivery_config, 'SNAPSHOTS', 'zip', False)

    # select which suppliers to process
    suppliers_to_process = dc.get_par(config_dict, 'SUPPLIERS_TO_PROCESS', '*')

    # just * means process all
    if suppliers_to_process == '*':
        suppliers_to_process = leveranciers.keys()

    # create the output file names
    _, report_filename, _ = dc.split_filename(delivery_filename)
    report_filename += '_report'
    report_csv_filename = os.path.join(work_dir, dc.DIR_DOCS, report_filename + '.csv')
    report_doc_filename = os.path.join(work_dir, dc.DIR_DOCS, report_filename + '.md')
    sql_filename        = os.path.join(work_dir, dc.DIR_SQL, 'import-all-deliveries.sql')

    # load DiDo parameters to fetch BEGINNING_OF_WORLD and END_OF_WORLD and store into config_dict
    bootstrap_data = dc.load_parameters()
    config_dict['BEGINNING_OF_WORLD'] = bootstrap_data['BEGINNING_OF_WORLD']
    config_dict['END_OF_WORLD'] = bootstrap_data['END_OF_WORLD']

    with open(report_csv_filename, mode = 'w', encoding = "utf8") as csv_file:
        with open(report_doc_filename, mode = 'w', encoding = "utf8") as doc_file:
            with open(sql_filename, mode = 'w', encoding = "utf8") as sql_file:
                # write transaction bracket for SQL
                # sql_file.write(generate_sql_header())
                sql_file.write('BEGIN; -- For all suppliers\n')

                # copy table_desc as a template for information each leverancier has to deliver
                for leverancier_id in suppliers_to_process:
                    logger.info('')
                    logger.info(f'=== {leverancier_id} ===')
                    logger.info('')

                    # get all cargo from the delivery_dict
                    cargo_dict = dc.get_cargo(delivery_config, leverancier_id)

                    # process all deliveries
                    for cargo_name in cargo_dict.keys():
                        logger.info('')
                        print_name = f'--- Delivery {cargo_name} ---'
                        logger.info(len(print_name) * '-')
                        logger.info(print_name)
                        logger.info(len(print_name) * '-')
                        logger.info('')

                        # get cargo associated with the cargo_name
                        cargo = cargo_dict[cargo_name]
                        cargo = dc.enhance_cargo_dict(cargo, cargo_name, leverancier_id)

                        # add config and delivery dicts as they are needed while processing the cargo
                        cargo['config'] = config_dict
                        cargo['delivery'] = delivery_config

                        # present all deliveries and the selected one
                        logger.info('Delivery configs supplied (> is selected)')
                        for key in cargo_dict.keys():
                            if key == cargo_name:
                                logger.info(f" > {key}")
                            else:
                                logger.info(f" - {key}")
                            # if
                        # for

                        # check if delivery exists in the database. If so, skip this delivery
                        if dc.delivery_exists(cargo, leverancier_id, project_name, db_servers):
                            logger.info('')
                            logger.error(f'*** delivery already exists: '
                                         f'{leverancier_id} - {cargo_name}, skipped')

                            if not overwrite:
                                logger.info('Specify ENFORCE_IMPORT_IF_TABLE_EXISTS: yes in your delivery.yaml')
                                logger.info('if you wish to check the data quality')

                                continue

                            else:
                                logger.warning('!!! ENFORCE_IMPORT_IF_TABLE_EXISTS: yes specified, '
                                               'data will be overwritten')
                            # if
                        # if

                        dc.report_ram('At beginning of loop')

                        cargo[dc.TAG_TABLES] = {}

                        for table_key in table_desc.keys():
                            # COPY the table dictionary to the supplier dict, else a shared reference will be copied
                            # to avoid sharing, use the .copy() function of the dictionary
                            cargo[dc.TAG_TABLES][table_key] = table_desc[table_key].copy()

                            # copy all keys, as they are string, they are correctly copied
                            for key in table_desc[table_key].keys():
                                cargo[dc.TAG_TABLES][table_key][key] = table_desc[table_key][key]

                            # for
                        # for

                        error_codes = dc.load_odl_table(
                            table_name = 'bronbestand_datakwaliteit_codes_data',
                            server_config = db_servers['ODL_SERVER_CONFIG']
                        )

                        error_codes = error_codes.set_index('code_datakwaliteit')

                        # for each leverancier, like dji, fmh, etc. create documentation and
                        # DDL generation files
                        # iterate and process each data suplier
                        origin = dc.get_par(cargo, 'origin', {'input': '<file>'})

                        process_supplier(
                            supplier_config = cargo,
                            config = config_dict,
                            error_codes = error_codes,
                            project_name = project_name,
                            origin = origin,
                            workdir = work_dir,
                            csv_file_object = csv_file,
                            doc_file_object = doc_file,
                            sql_file_object = sql_file,
                            server_configs = db_servers,
                        )

                        dc.report_ram('At end of loop')
                    # for
                # for

                sql_file.write('COMMIT;\n')
            # with
        # with
    # with

    dc.report_psql_use('import-all-deliveries', db_servers, False)

    cpu = time.time() - cpu
    logger.info('')
    logger.info(f'[Ready {appname["name"]} in {cpu:.0f} seconds]')
    logger.info('')

    return

### dido_import ###

if __name__ == '__main__':
    # read commandline parameters
    cli, args = dc.read_cli()

    # create logger in project directory
    log_file = os.path.join(args.project, 'logs', cli['name'] + '.log')
    logger = dc.create_log(log_file, level = 'DEBUG')

    # go
    dido_import('Importing Data')
