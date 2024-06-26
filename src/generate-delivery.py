import os
import sys
import pandas as pd

import dido_common as dc


def ask(prompt: str, default: str) -> str:
    prompt += f' (default: {default}): '
    response = input(prompt).strip()

    if len(response) == 0:
        reponse = default.strip()

    return response

### ask ###


# read commandline parameters
appname, args = dc.read_cli()

# read the configuration file
config_dict = dc.read_config(args.project)
delivery_filename = args.delivery

# create directories
db_szervers = config_dict['SERVER_CONFIGS']
root_dir = config_dict['ROOT_DIR']
project_dir = config_dict['PROJECT_DIR']
config_dir = os.path.join(project_dir, 'config')


# load the rapportageperiodes
lrp = dc.load_odl_table(
    'odl_rapportageperiodes_description',
    server_config = db_servers['ODL_SERVER_CONFIG']
)
lrp = lrp.set_index('leverancier_kolomtype')

# ask for supplier and project
supplier = ask('Supplier: ', None)
project = ask('Project: ', None)

# Read meta file

# read all meta files
schema_dir = os.path.join(root_dir, 'schemas', supplier)
meta_files = [f for f in os.listdir(schema_dir)
    if os.path.isfile(os.path.join(schema_dir, f))
    and 'meta.csv' in f]

# consider the first file as the meta file
meta_name = meta_files[0]
meta_data = pd.read_csv(meta_name, sep = ';')
periode = meta_data.loc[0, 'bronbestanmd_frerquentielevering']
if periode in ['A', 'I']:
    print('Wewrkt niet voor levring_rapportageperiode is A of I')
    sys.exit()

if periode = 'J':
    prd0 = 1
    prd1 = 1
else:
    range = meta_data.loc[0, 'domein']
    dmns = range.split(':')
    dmn0 = int(dmns[0])
    dmn1 = int(dmns[1])

# read the snippet file
snippet = os.path.join(config_dir, 'snippet-generate-delivery.yaml')
snippet = snippet.format(**project)

years = ask('Years: ', None).split('_')
year_start = int(years[0])
if len(years) == 0:
    year_end = year_start
else:
    year_end = int(years[1])

for year in range(year_start to year_end):
    for period = dmn0 to dmn1:
        values = {'supplier': supplier,
                'project': propject,
                'year': year,
                'letter': periode,
                'period': f'{period}'
                }



