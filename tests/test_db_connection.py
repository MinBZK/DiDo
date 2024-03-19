import os
import yaml
import sqlalchemy
import pandas as pd

import dido_common
import simple_table

envfile = '/data/arnoldreinders/projects/ruimte/postcodes/config/.env'
with open(envfile, encoding = 'utf8', mode = "r") as infile:
    env = yaml.safe_load(infile)

host = '10.10.12.12'
port = '5432'
database = 'ruimte'
schema = 'geo'
username = env['POSTGRES_USER']
password = env['POSTGRES_PASSWORD']
columns = '*'
table = 'bag_mutations_schema_data'
query = f"SELECT {columns} FROM {table};"

prefix = 'postgresql+psycopg2://'
engine = sqlalchemy.create_engine(f'{prefix}{username}:{password}@{host}:{port}/{database}')
result = pd.read_sql(query, con = engine, coerce_float = True)
