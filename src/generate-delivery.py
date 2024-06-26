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
db_servers = config_dict['SERVER_CONFIGS']
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
print('')
supplier = ask('Supplier: ', None)
project = ask('Project: ', None)

# Read meta file

# read all meta files
schema_dir = os.path.join(root_dir, 'schemas', supplier)
meta_files = [f for f in os.listdir(schema_dir)
    if os.path.isfile(os.path.join(schema_dir, f))
    and 'meta.csv' in f]

# consider the first file as the meta file
meta_name = os.path.join(schema_dir, meta_files[0])
meta_data = pd.read_csv(meta_name, sep = ';')
meta_data = meta_data.set_index('attribuut')

periode = meta_data.loc['Bronbestand frequentielevering', 'waarde']

if periode in ['A', 'I']:
    print('Werkt niet voor levering_rapportageperiode A of I')
    sys.exit()

if periode == 'J':
    prd0 = 1
    prd1 = 1

else:
    bereik = lrp.loc[periode, 'domein']
    dmns = bereik.split(':')
    dmn0 = int(dmns[0])
    dmn1 = int(dmns[1]) + 1

# read the snippet file
snippet_name = os.path.join(config_dir, 'snippet-generate-delivery.yaml')
with open(snippet_name, 'r') as infile:
    snippet = infile.read()

# ask which years to process
years = ask('Years: ', None).split('-')
year_start = int(years[0])
if len(years) == 0:
    year_end = year_start
else:
    year_end = int(years[1])

year_end += 1

snippets: str = ''
for year in range(year_start, year_end):
    for period in range(dmn0, dmn1):
        values = {'supplier': supplier.lower(),
                  'supplier_up': supplier.upper(),
                  'project': project.lower(),
                  'project_up': project.upper(),
                  'year': year,
                  'letter': periode,
                  'period': f'{period:02d}'
                 }

        result = snippet.format(**values)
        snippets += result

snippet_name = os.path.join(config_dir, 'snippet-generate-delivery.txt')
with open(snippet_name, 'w') as outfile:
    outfile.write(snippets)


