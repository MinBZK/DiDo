import gc
import os
import sys
import time
import psutil
import pandas as pd

def report_ram(message: str):
    mem = psutil.virtual_memory()

    print('')
    print(message)
    print(f'   Total memory:    {mem.total:,}')
    print(f'   Memory used:     {mem.used:,}')
    print(f'   Memory available {mem.available:,}')

    return

### report_ram ###

def fun_call(filename):
    cpu = time.time()

    data = pd.read_csv(
        filename,
        sep = ';',
        dtype = str,
        keep_default_na = False,
        # nrows = 0,
        engine = 'c',
        encoding = 'UTF8',
    )

    report_ram('In fun_call after read_csv')

    ram = data.memory_usage(index=True).sum()
    cpu = time.time() - cpu
    ref = sys.getrefcount(data)

    print(f'[Dataframe({len(data):,}, {len(data.columns)}) RAM:  {ram:,} CPU: {cpu:.0f} Refs: {ref}]')

    del [[data]]
    gc.collect()
    data = pd.DataFrame()

    return

### fun_call ###

offset = '/data/arnoldreinders/projects/personeel/salaris/work/todo/pdirekt'

files = ['20240229_ZBIOSAL01_Salaris_editie_20230630.CSV',
         '20240229_ZBIOSAL01_Salaris_editie_20230930.CSV',
         '20240229_ZBIOSAL01_Salaris_editie_20231231.CSV']

for file in files:
    print('----------------------')
    report_ram('beginning of loop')

    fun_call(os.path.join(offset, file))

    report_ram('@ end of loop')

# for
