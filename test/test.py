import os
import re
import pandas as pd
import numpy as np

#import common
import dido_common as dc
#import dido_begin
import dido_import 

from dotenv import load_dotenv

#from common import create_log
from dido_common import load_odl_table,DiDoError,get_limits, read_cli
from dido_common import DIR_SCHEMAS, DIR_DOCS, DIR_DONE, DIR_TODO, DIR_SQL
from dido_common import SCHEMA_TEMPLATE, META_TEMPLATE, EXTRA_TEMPLATE
from dido_common import VALUE_OK, VALUE_NOT_IN_LIST, VALUE_MANDATORY_NOT_SPECIFIED,VALUE_WRONG_DATATYPE
from dido_common import TAG_TABLE_SCHEMA, TAG_TABLE_META, TAG_TABLE_DELIVERY, TAG_TABLE_QUALITY
from dido_common import read_config, get_limits, load_parameters, load_schema

project_path = '/data/aliugurbas/didotest'
bootstrap_data = load_parameters()
allowed_datatypes = bootstrap_data['ALLOWED_POSTGRES_TYPES']

config_dict = dc.read_config(project_path)
max_errors = config_dict['LIMITATIONS']['max_errors']
suppliers = config_dict['SUPPLIERS']
db_servers = config_dict['SERVER_CONFIGS']
odl_server_config = db_servers['ODL_SERVER_CONFIG']
data_server_config = db_servers['DATA_SERVER_CONFIG']
foreign_server_config = db_servers['FOREIGN_SERVER_CONFIG']
work_dir = config_dict['WORK_DIR']
project_name = config_dict['PROJECT_NAME']

def  test_env():
    load_dotenv() 
    project_folder = os.environ['VIRTUAL_ENV'] # .rstrip('.venv')
    print(os.environ['HOME'])
    print(project_folder)

def test_pandas():  
    schema_file = "/data/aliugurbas/Projecten/personeel/ikb/root/schemas/ikb_jopi/Bronanalyse_IKB_JOPI.schema.csv"
    schema_name = pd.read_csv(schema_file, sep = ';', quotechar='"', dtype = str, keep_default_na = False, encoding='UTF-8')
        #print(schema_name['beschrijving'])
    schema = common.read_schema_file(schema_file)
    #print(schema['beschrijving'])
    list_1 = schema_name['beschrijving'] 
    list_2 = schema['beschrijving'] 
    print('zzz',list_1)
    print('xxx',list_2)
    print('ccc',schema_name['beschrijving'].all())
    if schema_name['beschrijving'].all() == schema['beschrijving'].all():
        print('geen error')
    else:
        raise ValueError("Handle the case when the Series is empty")

def test_merge_bootstrap_data():
    file = '/data/aliugurbas/Projecten/personeel/ikb/root/schemas/ikb_jopi/Bronanalyse_IKB_JOPI.schema.csv'
    conf_file = '/data/aliugurbas/d3g/config/bootstrap_data.csv'
    mijn_df = pd.read_csv(file,sep=';',dtype = str,keep_default_na = False)
    mijn_list = mijn_df['kolomnaam'].tolist()
    mijn_list.insert(0,'bronbestand_recordnummer')
    mijn_list.insert(1,'code_bronbestand')
    mijn_list.insert(2,'levering_rapportageperiode')
    mijn_list.insert(len(mijn_df.index)+3,'sysdatum') 
    schema_leverancier = pd.read_csv(file, sep=';').fillna('')
    schema = dido_begin.merge_bootstrap_data(schema_leverancier,conf_file)    
    schema_list = schema['kolomnaam'].tolist()
    print('schema_list', schema_list) 
    print('mijn_list',mijn_list)

def test_merge_bootstrap_data2():
    app_path, args = dido_common.read_cli()    
    config_dict = dido_common.read_config(args.project)
    suppliers = config_dict['SUPPLIERS']
    db_servers = config_dict['SERVER_CONFIGS']
    odl_server_config = db_servers['ODL_SERVER_CONFIG']

    for supplier in suppliers:
        leverancier = suppliers[supplier]
        source_file = leverancier['source_file']
        dir_load = os.path.join(config_dict['ROOT_DIR'], 'schemas', supplier)
        file = os.path.join(dir_load, source_file + '.schema.csv')
        #file = '/data/aliugurbas/Projecten/personeel/ikb/root/schemas/ikb_jopi/Bronanalyse_IKB_JOPI.schema.csv'
        #conf_file = '/data/aliugurbas/d3g/config/bootstrap_data.csv'
        mijn_df = pd.read_csv(file,sep=';',dtype = str,keep_default_na = False)
        mijn_list = mijn_df['kolomnaam'].tolist()
        mijn_list.insert(0,'bronbestand_recordnummer')
        mijn_list.insert(1,'code_bronbestand')
        mijn_list.insert(2,'levering_rapportageperiode')
        #mijn_list.insert(len(mijn_df.index)+3,'sysdatum') 
        mijn_list.insert(3,'sysdatum')    
        schema_leverancier = pd.read_csv(file, sep=';').fillna('')
        schema = dido_begin.merge_bootstrap_data(schema_leverancier, EXTRA_TEMPLATE, odl_server_config)    
        schema_list = schema['kolomnaam'].tolist()
        assert schema_list == mijn_list    

def test_apply_schema_odl():
    username, password, config_vars, program_vars = dido_common.read_config()
    allowed_datatypes = program_vars['POSTGRES_TYPES']

    suppliers = config_vars['SUPPLIERS']

    i = 0 
    for supplier in suppliers:
        if i == 0:
            leverancier = suppliers[supplier] 
            supplier_new = supplier
        i += 1  
    
    source_file = leverancier['source_file']

    dir_load = os.path.join(config_vars['WORK_DIR'], 'schemas', supplier_new)
    fname_schema_load = os.path.join(dir_load, source_file + '.schema.csv')
    #fname_schema_load = '/data/aliugurbas/Projecten/personeel/ikb/work/schemas/ikb_jopi/Bronanalyse_IKB_JOPI.schema.csv'
    print('file',fname_schema_load )    
    mijn_df = pd.read_csv(fname_schema_load,sep=';',dtype = str,keep_default_na = False)
    schema_data_types = {}
    selected_columns = ['leverancier_kolomtype','datatype']
    schema_data_types = mijn_df[selected_columns].to_dict(orient='list')
    data_types = {}
    if 'DATATYPES' in config_vars:
        data_types = config_vars['DATATYPES'] 
    for row,_ in mijn_df.iterrows():
        #print('YYY',mijn_df.loc[row, 'leverancier_kolomtype'].strip(),mijn_df.loc[row, 'datatype'],data_types[mijn_df.loc[row, 'leverancier_kolomtype'].strip()])
        if len(mijn_df.loc[row, 'leverancier_kolomtype']) > 0 and \
            mijn_df.loc[row, 'datatype'] != data_types[mijn_df.loc[row, 'leverancier_kolomtype'].strip()]: 
            print('KKK',mijn_df.loc[row, 'datatype'],data_types[mijn_df.loc[row, 'leverancier_kolomtype'].strip()])
#        if mijn_df.loc[row, 'leverancier_kolomtype'].strip() in data_dict.keys() \
#            data_types[schema.loc[row, 'leverancier_kolomtype'].strip()]:



'''
    for key in schema_data_types:
        value = schema_data_types[key]
        print(f'key: {key} - value: {value}')
        for nkey in allowed_datatypes:
            nvalue = allowed_datatypes[nkey]
            if  value ==  nvalue:
                print('XXX',schema_data_types[key],allowed_datatypes[nkey])
'''
def test_apply_schema_odl2():
    username, password, config_vars, program_vars = dido_common.read_config()
    odl_server_config = config_vars['ODL_SERVER_CONFIG']
    odl_server_config['POSTGRES_USER'] = username
    odl_server_config['POSTGRES_PW'] = password
    schema_template = load_odl_table(SCHEMA_TEMPLATE, odl_server_config)
    meta_template = load_odl_table(META_TEMPLATE, odl_server_config)
    allowed_datatypes = program_vars['POSTGRES_TYPES']
    fname_schema_load = '/data/aliugurbas/Projecten/personeel/ikb/root/schemas/ikb_jopi/Bronanalyse_IKB_JOPI.schema.csv'
    fname_meta_load = '/data/aliugurbas/Projecten/personeel/ikb/root/schemas/ikb_jopi/Bronanalyse_IKB_JOPI.meta.csv'
    schema_leverancier = pd.read_csv(fname_schema_load, sep=';').fillna('')
    schema_leverancier = dido_begin.merge_bootstrap_data(schema_leverancier, EXTRA_TEMPLATE, odl_server_config)
    meta_leverancier = pd.read_csv(fname_meta_load, sep=';', dtype=str, keep_default_na=False)
    mijn_df = pd.read_csv(fname_schema_load,sep=';',dtype = str,keep_default_na = False)
    schema_data_types = {}
    selected_columns = ['leverancier_kolomtype','datatype']
    schema_data_types = mijn_df[selected_columns].to_dict(orient='list')
    data_types = {}
    if 'DATATYPES' in config_vars:
        data_types = config_vars['DATATYPES'] 

    bootstrap_data = load_odl_table(EXTRA_TEMPLATE, server_config)

    print('YYY',meta_leverancier)
    schema_template = dido_begin.apply_schema_odl(
    template = schema_template,
    schema = schema_leverancier,
    meta = meta_leverancier,
    data_dict = data_types,
    filename = fname_schema_load,
    allowed_datatypes = program_vars['POSTGRES_TYPES']
    )   
    
    print('XXX',schema_template)
def test2():
    username, password, config_vars, program_vars = dido_common.read_config()
    odl_server_config = config_vars['ODL_SERVER_CONFIG']
    odl_server_config['POSTGRES_USER'] = username
    odl_server_config['POSTGRES_PW'] = password
    suppliers = config_vars['SUPPLIERS']
    i = 0 
    for supplier in suppliers:
        if i == 0:
            leverancier = suppliers[supplier] 
            supplier_new = supplier
        i += 1    
        print('ASD',leverancier,supplier)

    source_file = leverancier['source_file']
    print('YYY',source_file)    
   
    dir_load = os.path.join(config_vars['ROOT_DIR'], 'schemas', supplier_new)

    bootstrap_data = load_odl_table(EXTRA_TEMPLATE, odl_server_config)
    schema_template = load_odl_table(SCHEMA_TEMPLATE, odl_server_config)
    meta_template = load_odl_table(META_TEMPLATE, odl_server_config)
    allowed_datatypes = program_vars['POSTGRES_TYPES']

    #fname_schema_load = '/data/aliugurbas/Projecten/personeel/ikb/root/schemas/ikb_jopi/Bronanalyse_IKB_JOPI.schema.csv'
    fname_schema_load = os.path.join(dir_load, source_file + '.schema.csv')    
    #fname_meta_load   = '/data/aliugurbas/Projecten/personeel/ikb/root/schemas/ikb_jopi/Bronanalyse_IKB_JOPI.meta.csv'
    fname_meta_load = os.path.join(dir_load, source_file + '.meta.csv')
    schema_leverancier = pd.read_csv(fname_schema_load, sep=';').fillna('')
 
    meta_leverancier = pd.read_csv(fname_meta_load, sep=';', dtype=str, keep_default_na=False) 
    meta_leverancier = meta_leverancier.fillna('').set_index('attribuut')
    data_types = {}
    if 'DATATYPES' in config_vars:
        data_types = config_vars['DATATYPES'] 

    row_count = len(schema_leverancier.index) + len(bootstrap_data.index)
    new_df = pd.DataFrame(columns=['positie'],dtype=int)
    #new_df.insert(0,'kolomnaam','None') 
    column_name = ['kolomnaam','code_attribuut_sleutel','code_attribuut','code_bronbestand','leverancier_kolomnaam','leverancier_kolomtype','datatype','keytype','constraints','domein','verstek']
    initial_values = ['None','None','None','None','None','None','None','None','None','None','None']
    initial_values2 = ['None','None','None','None','None','None']
    column_name2 = ['avg_classificatie','veiligheid_classificatie','kolom_expiratie_datum','datum_start','datum_eind','beschrijving']   
    for col_name, initial_value in zip(column_name, initial_values):
        new_df.insert(len(new_df.columns) - 1, col_name, initial_value)   

    for col_name, initial_value in zip(column_name2, initial_values2):
        new_df.insert(len(new_df.columns) , col_name, initial_value)          
    #new_df.insert(0,'kolomnaam','None') 
    
    #i = 0
    for i in range(0, row_count ):
        new_df.loc[i, 'positie'] = str(i+1)
    #print(new_df)
    #print(new_df['positie'].dtype) 
    schema_leverancier = dido_begin.merge_bootstrap_data(schema_leverancier, EXTRA_TEMPLATE, odl_server_config)   

    schema_template = dido_begin.apply_schema_odl(
    template = schema_template,
    schema = schema_leverancier,
    meta = meta_leverancier,
    data_dict = data_types,
    filename = fname_schema_load,
    allowed_datatypes = program_vars['POSTGRES_TYPES']
    )
    #print('TTT',new_df)    
    #print('SSS',schema_template)
    


    #compare_df = schema_template 
    #print('schema_template',compare_df['positie']]
    print(schema_template.shape)
    print(new_df.shape)
    print(schema_template.columns)
    print(new_df.columns)
    print(schema_template.index)
    print(new_df.index)
    
    
    #comparison = schema_template.compare(new_df)
    kolom_check = schema_template['positie'].equals(new_df['positie'])
    print(kolom_check)
    #comparison.to_csv('compare.csv',sep=';')
    #print(comparison)

def test_check_null_iter():
    leveranciers = config_dict['SUPPLIERS']
    for leverancier in leveranciers:
        supplier_data = config_dict['SUPPLIERS'][leverancier]
        todo_dir = os.path.join(work_dir, DIR_TODO, leverancier)
        todo_files = [f for f in os.listdir(todo_dir) if os.path.isfile(os.path.join(todo_dir, f))]
        filename = supplier_data['delivery-1']['data_file']
        pad, fn, ext = dc.split_filename(filename)
        filename = fn + ext
#        assert filename in todo_files
#        assert ext.lower() == '.csv'
        
        data_filename = os.path.join(todo_dir, filename)
        data = pd.read_csv(data_filename, sep = '\s*;\s*', dtype = str, keep_default_na = False, engine = 'python')
        data.columns = [dc.change_column_name(col) for col in data.columns]
        print('file',data.loc())
        schema_name = f'{project_name}_{leverancier}_{TAG_TABLE_SCHEMA}_description'
        #print('data',data_server_config)
        supplier_schema = load_schema(schema_name, data_server_config)
        supplier_schema = supplier_schema.set_index('kolomnaam')
        print('suplier',supplier_schema)
        report = pd.DataFrame(np.zeros(data.shape), columns = data.columns, index = data.index, dtype = np.int32)
        messages = pd.DataFrame(columns = data.columns, index = data.index, dtype = str).fillna('')
        total_errors = 0
        err = 0
        #print('AAA',report) 
        #print('mes',messages)

        for col in data.columns:
            if supplier_schema.loc[col, 'constraints'] == 'NOT NULL':
                for row in range(len(data)):
                    print(data.iloc[row][col])
                    if (data.iloc[row][col]).strip() == '':
                        err += 1
            print('t_errors',total_errors) 
            #report, messages, total_errors = dido_import.check_null_iter(data,supplier_schema, report, messages, total_errors, max_errors, VALUE_MANDATORY_NOT_SPECIFIED, col)
            report, total_errors = dido_import.check_null_iter(data,supplier_schema, report,  max_errors,total_errors, VALUE_MANDATORY_NOT_SPECIFIED, col)            

#                   assert total_errors == 0 
                    #print(messages) 
        print('err',err)            
        print('total_errors',total_errors)    

def test_check_data_types():
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
        print('file',data)
        schema_name = f'{project_name}_{leverancier}_{TAG_TABLE_SCHEMA}_description'
        #print('data',data_server_config)
        supplier_schema = load_schema(schema_name, data_server_config)
        supplier_schema = supplier_schema.set_index('kolomnaam')
        print('suplier',supplier_schema)
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

                        print('col',supplier_schema.loc[col, 'constraints'])
                    report, messages, total_errors = dido_import.check_data_types(data,supplier_schema, report, messages, total_errors, VALUE_WRONG_DATATYPE, col)

#            assert total_errors == 0 
                    #print(messages)             




def test_check_boolean_type():
    leveranciers = config_dict['SUPPLIERS']
    for leverancier in leveranciers:
        supplier_data = config_dict['SUPPLIERS'][leverancier]
        todo_dir = os.path.join(work_dir, DIR_TODO, leverancier)
        todo_files = [f for f in os.listdir(todo_dir) if os.path.isfile(os.path.join(todo_dir, f))]
        filename = supplier_data['delivery-1']['data_file']
        pad, fn, ext = dc.split_filename(filename)
        filename = fn + ext
#        assert filename in todo_files
#        assert ext.lower() == '.csv'

        data_filename = os.path.join(todo_dir, filename)
        data = pd.read_csv(data_filename, sep = '\s*;\s*', dtype = str, keep_default_na = False, engine = 'python')
        data.columns = [dc.change_column_name(col) for col in data.columns]
        print('file',data)
        schema_name = f'{project_name}_{leverancier}_{TAG_TABLE_SCHEMA}_description'
        #print('data',data_server_config)
        supplier_schema = load_schema(schema_name, data_server_config)
        supplier_schema = supplier_schema.set_index('kolomnaam')
        print('suplier',supplier_schema)
        report = pd.DataFrame(np.zeros(data.shape), columns = data.columns, index = data.index, dtype = np.int32)
        messages = pd.DataFrame(columns = data.columns, index = data.index, dtype = str).fillna('')
        total_errors = 0
        err = 0
        bool_vals = ['true', 'yes', 'on', '1', 'false', 'no', 'off', '0']
        #print('AAA',report) 
        #print('mes',messages)

        for col in data.columns:
            data_type =  supplier_schema.loc[col, 'datatype'].lower()
            if data_type == 'boolean':
                for row in range(len(data)):
                    print(data.iloc[row][col])
                    if ((data.iloc[row][col]).strip() not in bool_vals and supplier_schema.loc[col, 'constraints'] != 'NOT NULL' 
                        and (data.iloc[row][col]).strip() != ''):
                        err += 1

                        print('col',supplier_schema.loc[col, 'constraints'])
                    report, messages, total_errors = dido_import.check_boolean_type(data,supplier_schema, report, messages, total_errors, VALUE_WRONG_DATATYPE, col)

#            assert total_errors == 0 
                    #print(messages)             

def test_check_integer_type():
    leveranciers = config_dict['SUPPLIERS']
    for leverancier in leveranciers:
        supplier_data = config_dict['SUPPLIERS'][leverancier]
        todo_dir = os.path.join(work_dir, DIR_TODO, leverancier)
        todo_files = [f for f in os.listdir(todo_dir) if os.path.isfile(os.path.join(todo_dir, f))]
        filename = supplier_data['delivery-1']['data_file']
        pad, fn, ext = common.split_filename(filename)
        filename = fn + ext
#        assert filename in todo_files
#        assert ext.lower() == '.csv'

        data_filename = os.path.join(todo_dir, filename)
        data = pd.read_csv(data_filename, sep = '\s*;\s*', dtype = str, keep_default_na = False, engine = 'python')
        data.columns = [common.change_column_name(col) for col in data.columns]
        print('file',data)
        schema_name = f'{project_name}_{leverancier}_{TAG_TABLE_SCHEMA}_description'
        #print('data',data_server_config)
        supplier_schema = load_schema(schema_name, data_server_config)
        supplier_schema = supplier_schema.set_index('kolomnaam')
        print('suplier',supplier_schema)
        report = pd.DataFrame(np.zeros(data.shape), columns = data.columns, index = data.index, dtype = np.int32)
        messages = pd.DataFrame(columns = data.columns, index = data.index, dtype = str).fillna('')
        total_errors = 0
        err = 0
        #print('AAA',report) 
        #print('mes',messages)

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
    
    print('Err',err)

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
        #print('data',data_server_config)
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
    print('Err',err)

def test_check_domain_minmax():
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
        #print('data',data_server_config)
        supplier_schema = load_schema(schema_name, data_server_config)
        supplier_schema = supplier_schema.set_index('kolomnaam')
        print('suplier',supplier_schema)
        report = pd.DataFrame(np.zeros(data.shape), columns = data.columns, index = data.index, dtype = np.int32)
        messages = pd.DataFrame(columns = data.columns, index = data.index, dtype = str).fillna('')

        total_errors = 0
        err = 0

        for col in data.columns:
            domain = supplier_schema.loc[col, 'domein'].strip() 
            if ':' in domain:
                data_type =  supplier_schema.loc[col, 'datatype'].lower()
                mins, maxs = domain.split(':')
                min_val, max_val = 0, 0

                if data_type in ['integer', 'bigint']:
                    min_val, max_val = int(mins), int(maxs)

                elif data_type in ['numeric', 'real', 'double']:
                    min_val, max_val = float(mins), float(maxs)
                
                for row in range(len(data)):
                    value= data.iloc[row][col]
                    if (supplier_schema.loc[col, 'constraints'] != 'NOT NULL'  and (value != '')):
                        try:
                            if data_type in ['integer', 'bigint']:
                                value = int(value)

                            elif data_type in ['numeric', 'real', 'double']:
                                value = float(value)

                            if not min_val <= value <= max_val:
                                raise DiDoError(f'Not in range {min_val}..{max_val}')
                        except:
                            err += 1


    print('Err',err)              
  
def test_check_domain_list():
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
        #print('data',data_server_config)
        supplier_schema = load_schema(schema_name, data_server_config)
        supplier_schema = supplier_schema.set_index('kolomnaam')
        print('suplier',supplier_schema)
        report = pd.DataFrame(np.zeros(data.shape), columns = data.columns, index = data.index, dtype = np.int32)
        messages = pd.DataFrame(columns = data.columns, index = data.index, dtype = str).fillna('')

        total_errors = 0
        err = 0

        for col in data.columns:
            domain = supplier_schema.loc[col, 'domein'].strip() 
            domain_list = [x.strip().strip('\"').strip('\'').strip('\"') for x in domain[1:-1].split(',')]
            if domain[0:1] == '[':
                data_type =  supplier_schema.loc[col, 'datatype'].lower()
                
                for row in range(len(data)):
                    values = data.iloc[row][col].strip()
                    value_list = values.split(',')
                    if (supplier_schema.loc[col, 'constraints'] != 'NOT NULL'  and (values != '')):
                        for value in value_list:
                            value = value.strip()
                            if not value in domain_list:
                                err += 1


    print('Err',err)  

def test_check_domain_re():
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
        #print('data',data_server_config)
        supplier_schema = load_schema(schema_name, data_server_config)
        supplier_schema = supplier_schema.set_index('kolomnaam')
        print('suplier',supplier_schema)
        report = pd.DataFrame(np.zeros(data.shape), columns = data.columns, index = data.index, dtype = np.int32)
        messages = pd.DataFrame(columns = data.columns, index = data.index, dtype = str).fillna('')

        total_errors = 0
        err = 0

        for col in data.columns:
            domain = supplier_schema.loc[col, 'domein'].strip()
            pattern = re.compile(domain[3:])
            if domain[0:3] == 're:':
                data_type =  supplier_schema.loc[col, 'datatype'].lower()
                for row in range(len(data)):
                    value= data.iloc[row][col]
                    if (supplier_schema.loc[col, 'constraints'] != 'NOT NULL'  and (value != '')):
                        if not pattern.match(value):
                            err += 1


    print('Err',err) 

'''
if __name__ == '__main__':
    # read commandline parameters
    cli, args = read_cli()

    # create logger in project directory
    log_file = os.path.join(args.project, 'logs', cli['name'] + '.log')
    logger = common.create_log(log_file, level = 'DEBUG')

    test_check_null_iter()
'''

#test_check_null_iter()
test_check_data_types()
#test_check_boolean_type()
#test_check_decimal_type()
#test_check_domain_minmax()
#test_check_domain_list()
#test_check_domain_re()
#test_apply_schema_odl()
#test2()
#test_env()
