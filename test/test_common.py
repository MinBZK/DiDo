import pandas as pd
import common

def test_character_spatie():

    new_kolom = common.change_column_name('my_2#$%3fr y_',)
    uitkomst = 'my_2_3fr_y_'
    
    assert new_kolom ==  uitkomst

def test_nummber_begin_file():

    new_kolom = common.change_column_name('12__ne##$w_',)
    uitkomst = 'ne_w'

    assert new_kolom == uitkomst

def test_uppercase():

    new_kolom = common.change_column_name('AbC  #dF$w_',)
    uitkomst = 'abc_df_w'

    assert new_kolom == uitkomst

def test_math_characters():

    new_kolom = common.change_column_name('A+cF-*?ë.,mN(xy){}[]',)
    uitkomst = 'a_cf_mn_xy'

    assert new_kolom == uitkomst

def test_split_filename():
    full_name ='/data/aliugurbas/Projecten/personeel/ikb/root/schemas/ikb_jopi/Bronanalyse_IKB_JOPI.meta.csv'
    mypath , fname, extname = common.split_filename(full_name)
    c_path = '/data/aliugurbas/Projecten/personeel/ikb/root/schemas/ikb_jopi'
    c_fname = 'Bronanalyse_IKB_JOPI.meta'
    c_extname = '.csv'

    assert mypath == c_path and fname == c_fname and extname ==  c_extname

def test_read_schema_file():
    schema_file = '/data/aliugurbas/Projecten/personeel/ikb/root/schemas/ikb_jopi/Bronanalyse_IKB_JOPI.schema.csv'
    schema_name = pd.read_csv(schema_file, sep = ';', dtype = str, keep_default_na = False, encoding='UTF-8')
    schema = common.read_schema_file(schema_file)
    #list_1 = schema_name['beschrijving'] 
    #list_2 = schema['beschrijving'] 

    #for col in schema_name.columns:
    #    assert schema[col].equals(schema_name[col]) , f"dataframe is not equal for column {col}"
    assert schema['beschrijving'].equals(schema_name['beschrijving']) , "dataframe is not equal for"



