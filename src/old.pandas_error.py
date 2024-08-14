from io import StringIO
import pandas as pd

data_string = '''col1;col1;col3
a1;a2;a3;
b1;b2;b3;
'''

print(pd.__version__)

data = pd.read_csv(
    StringIO(data_string),
    mangle_dupe_cols = False,
    sep = ';',
    dtype = str,
    keep_default_na = False,
    engine = 'c',
    encoding = 'utf8',
)

print(data.columns)
