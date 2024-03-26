#test_begin.py
import os
import sys
import argparse
import pandas as pd
import dido_begin
import dido_common
from dotenv import load_dotenv

from dido_common import load_odl_table
from dido_common import SCHEMA_TEMPLATE, META_TEMPLATE, EXTRA_TEMPLATE
from dido_common import read_config, get_limits, load_parameters

project_path = '/data/aliugurbas/Projecten/personeel/ikb'
bootstrap_data = load_parameters()
allowed_datatypes = bootstrap_data['ALLOWED_POSTGRES_TYPES']

config_dict = dido_common.read_config(project_path)
suppliers = config_dict['SUPPLIERS']
db_servers = config_dict['SERVER_CONFIGS']
odl_server_config = db_servers['ODL_SERVER_CONFIG']

def test_workdir_is_created():


    # read the configuration file
    config_dict = dido_common.read_config(project_path)
    db_servers = config_dict['SERVER_CONFIGS']
    dido_begin.create_workdir_structure(config_dict, db_servers['ODL_SERVER_CONFIG'])
    bootstrap_data = load_parameters()
    # get the workdirs string
    subdirectories = bootstrap_data['WORKDIR_STRUCTURE']    
    work_dir = config_dict['WORK_DIR']
    for subdir in subdirectories.split(','):
        subdir = subdir.strip()
        for supplier in config_dict['SUPPLIERS']:
            folder_to_create = os.path.join(work_dir, subdir, supplier)
            assert os.path.exists(folder_to_create) == True
    

def test_create_folder():
    current_path = os.getcwd()
    new_folder = 'hopefully_unique_name'

    dido_begin.create_folder(current_path, new_folder)

    folder_to_create = os.path.join(current_path, new_folder)

    assert os.path.exists(folder_to_create)

def test_merge_bootstrap_data():



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
        #assert schema_list == mijn_list 


#   list_lev_kol = mijn_df['leverancier_kolomtype'].tolist() 
#    list_datatype = mijn_df['datatype'].tolist()

def test_datatype():

    i = 0 
    for supplier in suppliers:
        if i == 0:
            leverancier = suppliers[supplier] 
            supplier_new = supplier
        i += 1  
    
    source_file = leverancier['source_file']

    dir_load = os.path.join(config_dict['WORK_DIR'], 'schemas', supplier_new)
    fname_schema_load = os.path.join(dir_load, source_file + '.schema.csv')
    #fname_schema_load = '/data/aliugurbas/Projecten/personeel/ikb/work/schemas/ikb_jopi/Bronanalyse_IKB_JOPI.schema.csv'
    mijn_df = pd.read_csv(fname_schema_load,sep=';',dtype = str,keep_default_na = False)
    schema_data_types = {}
    selected_columns = ['leverancier_kolomtype','datatype']
    schema_data_types = mijn_df[selected_columns].to_dict(orient='list')
    data_types = {}
    if 'DATATYPES' in config_dict:
        data_types = config_dict['DATATYPES'] 
    for row,_ in mijn_df.iterrows():
        #print('YYY',mijn_df.loc[row, 'leverancier_kolomtype'].strip())
        if len(mijn_df.loc[row, 'leverancier_kolomtype']) > 0 and \
            mijn_df.loc[row, 'datatype'] != data_types[mijn_df.loc[row, 'leverancier_kolomtype'].strip()]: 
            assert mijn_df.loc[row, 'datatype'] == data_types[mijn_df.loc[row, 'leverancier_kolomtype'].strip()]
            #print('KKK',data_types[mijn_df.loc[row, 'leverancier_kolomtype'].strip()])


def test_apply_schema_odl():

    i = 0 
    for supplier in suppliers:
        if i == 0:
            leverancier = suppliers[supplier] 
            supplier_new = supplier
        i += 1  
    
    source_file = leverancier['source_file']

    dir_load = os.path.join(config_dict['ROOT_DIR'], 'schemas', supplier_new)

    bootstrap_data = load_odl_table(EXTRA_TEMPLATE, odl_server_config)
    schema_template = load_odl_table(SCHEMA_TEMPLATE, odl_server_config)
    meta_template = load_odl_table(META_TEMPLATE, odl_server_config)
    
    #allowed_datatypes = program_vars['POSTGRES_TYPES']

    #fname_schema_load = '/data/aliugurbas/Projecten/personeel/ikb/root/schemas/ikb_jopi/Bronanalyse_IKB_JOPI.schema.csv'
    fname_schema_load = os.path.join(dir_load, source_file + '.schema.csv')    
    #fname_meta_load   = '/data/aliugurbas/Projecten/personeel/ikb/root/schemas/ikb_jopi/Bronanalyse_IKB_JOPI.meta.csv'
    fname_meta_load = os.path.join(dir_load, source_file + '.meta.csv')
    schema_leverancier = pd.read_csv(fname_schema_load, sep=';').fillna('')
 
    meta_leverancier = pd.read_csv(fname_meta_load, sep=';', dtype=str, keep_default_na=False) 
    meta_leverancier = meta_leverancier.fillna('').set_index('attribuut')
    data_types = {}
    if 'DATATYPES' in config_dict:
        data_types = config_dict['DATATYPES'] 

    row_count = len(schema_leverancier.index) + len(bootstrap_data.index)
    new_df = pd.DataFrame(columns=['positie'],dtype=int)
    #new_df.insert(0,'kolomnaam','None') 

    for i in range(0, row_count ):
        new_df.loc[i, 'positie'] = str(i+1)

    schema_leverancier = dido_begin.merge_bootstrap_data(schema_leverancier, EXTRA_TEMPLATE, odl_server_config)   

    schema_template = dido_begin.apply_schema_odl(
    template = schema_template,
    schema = schema_leverancier,
    meta = meta_leverancier,
    data_dict = data_types,
    filename = fname_schema_load,
    allowed_datatypes = allowed_datatypes
    )   

    #kolom_check = schema_template['positie'].equals(new_df['positie'])
    assert schema_template['positie'].equals(new_df['positie']) 
