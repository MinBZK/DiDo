import os
import pandas as pd
import numpy as np

import dido_common as dc
import dido_import

from dido_common import DIR_SCHEMAS, DIR_DOCS, DIR_DONE, DIR_TODO, DIR_SQL
from dido_common import TAG_TABLE_SCHEMA, TAG_TABLE_META, TAG_TABLE_DELIVERY, TAG_TABLE_QUALITY
from dido_common import VALUE_OK, VALUE_NOT_IN_LIST, VALUE_MANDATORY_NOT_SPECIFIED
from dido_common import read_config, get_limits, load_parameters, load_schema

project_path = '/data/aliugurbas/didotest'
bootstrap_data = load_parameters()
allowed_datatypes = bootstrap_data['ALLOWED_POSTGRES_TYPES']

config_dict = dido_common.read_config(project_path)
suppliers = config_dict['SUPPLIERS']
db_servers = config_dict['SERVER_CONFIGS']
odl_server_config = db_servers['ODL_SERVER_CONFIG']
data_server_config = db_servers['DATA_SERVER_CONFIG']
foreign_server_config = db_servers['FOREIGN_SERVER_CONFIG']
work_dir = config_dict['WORK_DIR']
project_name = config_dict['PROJECT_NAME']

def test_test_levering_rapportageperiode():
    period = '2023-J01'
    test_periode = dido_import.test_levering_rapportageperiode(period)
    assert test_periode == True

def test_check_null_iter():
    leveranciers = config_dict['SUPPLIERS']
    for leverancier in leveranciers:
        supplier_data = config_dict['SUPPLIERS'][leverancier]
        todo_dir = os.path.join(work_dir, DIR_TODO, leverancier)
        todo_files = [f for f in os.listdir(todo_dir) if os.path.isfile(os.path.join(todo_dir, f))]
        filename = supplier_data['delivery-1']['data_file']
        pad, fn, ext = dc.split_filename(filename)
        filename = fn + ext
        
        data_filename = os.path.join(todo_dir, filename)
        data = pd.read_csv(data_filename, sep = '\s*;\s*', dtype = str, keep_default_na = False, engine = 'python')
        data.columns = [dc.change_column_name(col) for col in data.columns]
        
        schema_name = f'{project_name}_{leverancier}_{TAG_TABLE_SCHEMA}_description'
        supplier_schema = load_schema(schema_name, data_server_config)
        supplier_schema = supplier_schema.set_index('kolomnaam')

        report = pd.DataFrame(np.zeros(data.shape), columns = data.columns, index = data.index, dtype = np.int32)
        messages = pd.DataFrame(columns = data.columns, index = data.index, dtype = str).fillna('')
        total_errors = 0
        err = 0


        for col in data.columns:
            if supplier_schema.loc[col, 'constraints'] == 'NOT NULL':
                for row in range(len(data)):
                    print(data.iloc[row][col])
                    if (data.iloc[row][col]).strip() == '':
                        err += 1
            report, total_errors = dido_import.check_null_iter(data,supplier_schema, report,  max_errors,total_errors, VALUE_MANDATORY_NOT_SPECIFIED, col)            

        assert total_errors == err

def test_check_boolean_type():
    leveranciers = config_dict['SUPPLIERS']
    for leverancier in leveranciers:
        supplier_data = config_dict['SUPPLIERS'][leverancier]
        todo_dir = os.path.join(work_dir, DIR_TODO, leverancier)
        todo_files = [f for f in os.listdir(todo_dir) if os.path.isfile(os.path.join(todo_dir, f))]
        filename = supplier_data['delivery-1']['data_file']
        pad, fn, ext = common.split_filename(filename)
        filename = fn + ext

        data_filename = os.path.join(todo_dir, filename)
        data = pd.read_csv(data_filename, sep = '\s*;\s*', dtype = str, keep_default_na = False, engine = 'python')
        data.columns = [common.change_column_name(col) for col in data.columns]

        schema_name = f'{project_name}_{leverancier}_{TAG_TABLE_SCHEMA}_description'
        supplier_schema = load_schema(schema_name, data_server_config)
        supplier_schema = supplier_schema.set_index('kolomnaam')

        report = pd.DataFrame(np.zeros(data.shape), columns = data.columns, index = data.index, dtype = np.int32)
        messages = pd.DataFrame(columns = data.columns, index = data.index, dtype = str).fillna('')
        total_errors = 0
        err = 0
        bool_vals = ['true', 'yes', 'on', '1', 'false', 'no', 'off', '0']

        for col in data.columns:
            data_type =  supplier_schema.loc[col, 'datatype'].lower()
            if data_type == 'boolean':
                for row in range(len(data)):
                    print(data.iloc[row][col])
                    if ((data.iloc[row][col]).strip() not in bool_vals and supplier_schema.loc[col, 'constraints'] != 'NOT NULL' 
                        and (data.iloc[row][col]).strip() != ''):
                        err += 1

                        
                    #report, messages, total_errors = dido_import.check_boolean_type(data,supplier_schema, report, messages, total_errors, VALUE_WRONG_DATATYPE, col)

        assert total_errors == err 


def test_check_integer_type():
    leveranciers = config_dict['SUPPLIERS']
    for leverancier in leveranciers:
        supplier_data = config_dict['SUPPLIERS'][leverancier]
        todo_dir = os.path.join(work_dir, DIR_TODO, leverancier)
        todo_files = [f for f in os.listdir(todo_dir) if os.path.isfile(os.path.join(todo_dir, f))]
        filename = supplier_data['delivery-1']['data_file']
        pad, fn, ext = common.split_filename(filename)
        filename = fn + ext

        data_filename = os.path.join(todo_dir, filename)
        data = pd.read_csv(data_filename, sep = '\s*;\s*', dtype = str, keep_default_na = False, engine = 'python')
        data.columns = [common.change_column_name(col) for col in data.columns]

        schema_name = f'{project_name}_{leverancier}_{TAG_TABLE_SCHEMA}_description'
 
        supplier_schema = load_schema(schema_name, data_server_config)
        supplier_schema = supplier_schema.set_index('kolomnaam')

        report = pd.DataFrame(np.zeros(data.shape), columns = data.columns, index = data.index, dtype = np.int32)
        messages = pd.DataFrame(columns = data.columns, index = data.index, dtype = str).fillna('')
        total_errors = 0
        err = 0

        for col in data.columns:
            data_type =  supplier_schema.loc[col, 'datatype'].lower()
            if data_type in ['integer', 'bigint']:
                for row in range(len(data)):
                    value= (data.iloc[row][col]).strip()
                    if (supplier_schema.loc[col, 'constraints'] != 'NOT NULL' 
                        and (value != '')):
                        try:
                            int(str(value)) 
                        except ValueError:
                            err += 1
                        #report, messages, total_errors = dido_import.check_boolean_type(data,supplier_schema, report, messages, total_errors, VALUE_WRONG_DATATYPE, col)
    assert total_errors == err

def test_check_decimal_type():
    leveranciers = config_dict['SUPPLIERS']
    for leverancier in leveranciers:
        supplier_data = config_dict['SUPPLIERS'][leverancier]
        todo_dir = os.path.join(work_dir, DIR_TODO, leverancier)
        todo_files = [f for f in os.listdir(todo_dir) if os.path.isfile(os.path.join(todo_dir, f))]
        filename = supplier_data['delivery-1']['data_file']
        pad, fn, ext = common.split_filename(filename)
        filename = fn + ext

        data_filename = os.path.join(todo_dir, filename)
        data = pd.read_csv(data_filename, sep = '\s*;\s*', dtype = str, keep_default_na = False, engine = 'python')
        data.columns = [common.change_column_name(col) for col in data.columns]
        print('file',data)
        schema_name = f'{project_name}_{leverancier}_{TAG_TABLE_SCHEMA}_description'

        supplier_schema = load_schema(schema_name, data_server_config)
        supplier_schema = supplier_schema.set_index('kolomnaam')
        print('suplier',supplier_schema)
        report = pd.DataFrame(np.zeros(data.shape), columns = data.columns, index = data.index, dtype = np.int32)
        messages = pd.DataFrame(columns = data.columns, index = data.index, dtype = str).fillna('')
        total_errors = 0
        err = 0

        for col in data.columns:
            data_type =  supplier_schema.loc[col, 'datatype'].lower()
            if data_type in ['numeric', 'real', 'double']:
                for row in range(len(data)):
                    value= (data.iloc[row][col]).strip()
                    if (supplier_schema.loc[col, 'constraints'] != 'NOT NULL' 
                        and (value != '')):
                        try:
                            float(str(value)) 
                        except ValueError:
                            err += 1
    assert total_errors == err
    