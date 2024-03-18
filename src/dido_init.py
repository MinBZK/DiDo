import os
import re
import sys
import time
import shutil

import dido_common as dc

template_config = 'dido_template_config.yaml'
template_delivery = 'dido_template_delivery.yaml'
template_schema = 'dido_template_schema.csv'
template_meta = 'dido_template_meta.csv'

snippet_data_dict = 'dido_snippet_data_dict.yaml'
snippet_data_types = 'dido_snippet_pdirekt_datatypes.yaml'

config_dir = 'config'
root_dir = 'root'
work_dir = 'work'
logs_dir = 'logs'
data_dir = 'data'
docs_dir = 'docs'
schemas_dir = 'schemas'

cpu = time.time()

# read commandline parameters
appname, args = dc.read_cli()

# create project directories
project_dir = args.project


def copy_template(template: str, to: str, values):
    with open(template, encoding = 'utf8', mode = "r") as infile:
        config = infile.read()

    config_string = config.format(**values)

    if not os.path.isfile(to):
        with open(to, 'w') as config_file:
            config_file.write(config_string)

        print(f'Example {to} written')

    else:
        print(f"{to} exists, not modified")

    # if

    return

### copy_template ###


def ask(prompt: str, default: str) -> str:
    prompt += f' (default: {default}): '
    response = input(prompt).strip()

    if len(response) == 0:
        reponse = default.strip()

    return response

### ask ###


def pdirekt_data_dict(data_dict_filename: str,
                      proj_dir: str,
                      leverancier: str,
                      values: dict) -> dict:
    with open(data_dict_filename, 'r') as infile:
        data_dict_str = infile.read()

    _, fn, ext = dc.split_filename(data_dict_filename)
    data_dict_name = fn + ext
    values['data_dict_name'] = fn

    dir_to = os.path.join(proj_dir, 'root', 'schemas', leverancier, data_dict_name)
    if not os.path.isfile(dir_to):
        print('[Data dictionary not found and copied]')
        shutil.copyfile(data_dict_filename, dir_to)
    else:
        print('Data dictionary already exists:', dir_to)

    dir_from = os.path.join('config', snippet_data_dict)
    with open(dir_from, 'r') as infile:
        data_dict = infile.read()

    spec = data_dict.format(**values)

    return spec


def pdirekt_data_types(leverancier: str):
    data_types = ''

    if leverancier == 'pdirekt':
        snippet_datatypes = os.path.join('config', snippet_data_types)

        with open(snippet_datatypes, 'r') as infile:
            data_types = infile.read()

        # with
    # if

    return data_types

### pdirekt_data_types ###


# test if project already exists
config_test_name = os.path.join(project_dir, 'config', 'config.yaml')
if os.path.isfile(config_test_name):
    print('')
    print('*** Project directory exists and appears to be a DiDo project:', config_test_name)
    print('*** Quitting immediately')
    print('')

    sys.exit()

# create rudimentary config file
project_name = os.path.basename(project_dir)

print(f'Project name is {project_name}')
supplier = ''
while supplier == '':
    print('')
    supplier_name = ask('Name of supplier', '')
    supplier = dc.change_column_name(supplier_name)
    supplier = supplier.replace('_', '')
    print('Name used for supplier:', supplier)

# while

print('')
dd = ask('Enter full path to data dictionary name when present', '')
while len(dd) > 0 and not os.path.isfile(dd):
    dd = ask('Enter full path to data dictionary name when present', '')

code_bronbestand = supplier.upper() + '_' + project_name.upper()
schema_filename = 'bronbestand_' + supplier.lower() + '_' + project_name.lower()

print('')
decimal = ask('Decimal character, point (.) or (,)', '.')
print('Frequency of deliveries')
print(' J - Yearly')
print(' H - Half yearly')
print(' Q - Quarterly')
print(' M - Monthly')
print(' W - Weekly')
print(' D - Daily')
print(' I - Initial')
print(' A - Other')
freq = ask('Frequency of delivery', 'Q').upper()

values = {'project_name': project_name,
          'root_dir': root_dir,
          'work_dir': work_dir,
          'supplier_name': f'<{supplier_name}>',
          'code': code_bronbestand,
          'frequency': freq,
          'supplier': supplier,
          'decimal': decimal,
          'data_dict_name': dd,
          'schema_filename': schema_filename,
         }

print('')
print('Project directory is: ' + project_dir)
for subdir in [config_dir, root_dir, work_dir, logs_dir]:
    dirname = os.path.join(project_dir, subdir)
    print(f' - {subdir}')

    try:
        os.makedirs(dirname , exist_ok = True)

    except:
        print(f'file already exists: {dirname}')

# Create the minimal root directories
for subsubdir in [data_dir, docs_dir, schemas_dir]:
    dirname = os.path.join(project_dir, root_dir, subsubdir)
    print(f' - {root_dir}/{subsubdir}')

    try:
        os.makedirs(dirname, exist_ok = True)

    except:
        print(f'file already exists: {dirname}')

    dirname = os.path.join(dirname, supplier)
    try:
        os.makedirs(dirname , exist_ok = True)

    except:
        print(f'file already exists: {dirname}')

# for

# Process a P-Direkt data dictionary when one is specified
if len(dd) > 0:
    data_dict_spec = pdirekt_data_dict(dd, project_dir, supplier, values)
    values['data_dict'] = data_dict_spec
else:
    values['data_dict'] = '# No data dictionary provided'

# if

values['data_types'] = pdirekt_data_types(supplier)

# Copy config.yaml
dir_from = os.path.join('config', template_config)
dir_to = os.path.join(project_dir, config_dir, 'config.yaml')
copy_template(dir_from, dir_to, values)

# Copy delivery.yaml
dir_from = os.path.join('config', template_delivery)
dir_to = os.path.join(project_dir, 'root', 'data', 'delivery.yaml')
copy_template(dir_from, dir_to, values)

# Copy example schema file
dir_from = os.path.join('config', template_schema)
dir_to = os.path.join(project_dir, values['root_dir'], 'schemas', supplier,
                      f'{schema_filename}.schema.csv')
copy_template(dir_from, dir_to, values)

# Copy example meta file
dir_from = os.path.join('config', template_meta)
dir_to = os.path.join(project_dir, values['root_dir'], 'schemas', supplier,
                      f'{schema_filename}.meta.csv')
copy_template(dir_from, dir_to, values)

cpu = time.time() - cpu
print('')
print(f'[Ready {appname["name"]} in {cpu:.0f} seconds]')
print('')
