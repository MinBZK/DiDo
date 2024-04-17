BEGIN; -- Transaction
-- Quit immediately with exit code other than 0 when an error occurs
\set ON_ERROR_STOP true

INSERT INTO didotest.dido_test_dido_import_datakwaliteit_feit_data (bronbestand_recordnummer, code_bronbestand, row_number, column_name, code_attribuut, code_datakwaliteit, levering_rapportageperiode, sysdatum)
VALUES
(1, $$DIDO_TEST$$, 1, $$col_integer$$, $$2$$, 6, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(4, $$DIDO_TEST$$, 4, $$col_integer$$, $$2$$, 6, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(5, $$DIDO_TEST$$, 5, $$col_integer$$, $$2$$, 6, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(6, $$DIDO_TEST$$, 6, $$col_integer$$, $$2$$, 6, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(7, $$DIDO_TEST$$, 7, $$col_integer$$, $$2$$, 6, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(8, $$DIDO_TEST$$, 8, $$col_integer$$, $$2$$, 6, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(8, $$DIDO_TEST$$, 8, $$col_decimal$$, $$3$$, 6, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(1, $$DIDO_TEST$$, 1, $$col_date$$, $$4$$, 6, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(3, $$DIDO_TEST$$, 3, $$col_date$$, $$4$$, 6, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(5, $$DIDO_TEST$$, 5, $$col_date$$, $$4$$, 6, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(6, $$DIDO_TEST$$, 6, $$col_date$$, $$4$$, 6, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(8, $$DIDO_TEST$$, 8, $$col_date$$, $$4$$, 6, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(0, $$DIDO_TEST$$, 0, $$check_minmax$$, $$6$$, 6, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(1, $$DIDO_TEST$$, 1, $$check_minmax$$, $$6$$, 6, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(6, $$DIDO_TEST$$, 6, $$check_list_numeric$$, $$7$$, 6, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(0, $$DIDO_TEST$$, 0, $$col_boolean$$, $$1$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(1, $$DIDO_TEST$$, 1, $$col_boolean$$, $$1$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(3, $$DIDO_TEST$$, 3, $$col_boolean$$, $$1$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(5, $$DIDO_TEST$$, 5, $$col_boolean$$, $$1$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(6, $$DIDO_TEST$$, 6, $$col_boolean$$, $$1$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(7, $$DIDO_TEST$$, 7, $$col_boolean$$, $$1$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(8, $$DIDO_TEST$$, 8, $$col_boolean$$, $$1$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(1, $$DIDO_TEST$$, 1, $$col_integer$$, $$2$$, 3, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(4, $$DIDO_TEST$$, 4, $$col_integer$$, $$2$$, 3, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(5, $$DIDO_TEST$$, 5, $$col_integer$$, $$2$$, 3, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(6, $$DIDO_TEST$$, 6, $$col_integer$$, $$2$$, 3, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(7, $$DIDO_TEST$$, 7, $$col_integer$$, $$2$$, 3, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(8, $$DIDO_TEST$$, 8, $$col_integer$$, $$2$$, 3, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(0, $$DIDO_TEST$$, 0, $$check_minmax$$, $$6$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(1, $$DIDO_TEST$$, 1, $$check_minmax$$, $$6$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(2, $$DIDO_TEST$$, 2, $$check_minmax$$, $$6$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(3, $$DIDO_TEST$$, 3, $$check_minmax$$, $$6$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(4, $$DIDO_TEST$$, 4, $$check_minmax$$, $$6$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(5, $$DIDO_TEST$$, 5, $$check_minmax$$, $$6$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(6, $$DIDO_TEST$$, 6, $$check_minmax$$, $$6$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(7, $$DIDO_TEST$$, 7, $$check_minmax$$, $$6$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(8, $$DIDO_TEST$$, 8, $$check_minmax$$, $$6$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(0, $$DIDO_TEST$$, 0, $$check_list_numeric$$, $$7$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(2, $$DIDO_TEST$$, 2, $$check_list_numeric$$, $$7$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(3, $$DIDO_TEST$$, 3, $$check_list_numeric$$, $$7$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(4, $$DIDO_TEST$$, 4, $$check_list_numeric$$, $$7$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(5, $$DIDO_TEST$$, 5, $$check_list_numeric$$, $$7$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(6, $$DIDO_TEST$$, 6, $$check_list_numeric$$, $$7$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(7, $$DIDO_TEST$$, 7, $$check_list_numeric$$, $$7$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(8, $$DIDO_TEST$$, 8, $$check_list_numeric$$, $$7$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(0, $$DIDO_TEST$$, 0, $$check_list_text$$, $$8$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(1, $$DIDO_TEST$$, 1, $$check_list_text$$, $$8$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(2, $$DIDO_TEST$$, 2, $$check_list_text$$, $$8$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(3, $$DIDO_TEST$$, 3, $$check_list_text$$, $$8$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(4, $$DIDO_TEST$$, 4, $$check_list_text$$, $$8$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(5, $$DIDO_TEST$$, 5, $$check_list_text$$, $$8$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(6, $$DIDO_TEST$$, 6, $$check_list_text$$, $$8$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(7, $$DIDO_TEST$$, 7, $$check_list_text$$, $$8$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(8, $$DIDO_TEST$$, 8, $$check_list_text$$, $$8$$, 1, $$2023-Q2$$, CURRENT_TIMESTAMP(0)),
(0, $$DIDO_TEST$$, 0, $$$$, $$0$$, -1, $$2023-Q2$$, CURRENT_TIMESTAMP(0));

INSERT INTO didotest.dido_test_dido_import_levering_feit_data (levering_rapportageperiode, code_bronbestand, created_by, levering_leveringsdatum, levering_rapportageperiode_volgnummer, levering_goed_voor_verwerking, levering_reden_niet_verwerken, levering_verwerkingsdatum, levering_aantal_records, config_file, data_filenaam, sysdatum)
VALUES
($$2023-Q2$$, $$DIDO_TEST$$, current_user, CURRENT_DATE, 1, True, $$Not applicable$$, CURRENT_DATE, 0, $$PROJECT_NAME: dido_import
ROOT_DIR: root
WORK_DIR: work

VERBOSE: False

# database credentials for ODL database. Do not fill out POSTGRES_USER and _PW
# Will be replaced by those of your .pgpass/.env file

SERVER_CONFIGS:
  ODL_SERVER_CONFIG:
    POSTGRES_PORT: 5432
    POSTGRES_DB: techniek
    POSTGRES_SCHEMA: datamanagement

  # database credentials for storing the data. _USER and _PW will be replaced by .env
  DATA_SERVER_CONFIG:
    POSTGRES_PORT: 5432
    POSTGRES_DB: test_database
    POSTGRES_SCHEMA: didotest

  # database credentials for xtracting the data. _USER and _PW will be replaced by .env
  FOREIGN_SERVER_CONFIG:
    POSTGRES_PORT: 5432
    POSTGRES_DB: test_database
    POSTGRES_SCHEMA: historie

# select server to use as POSTGRES_HOST
HOST: datawarehouse_dev

SUPPLIERS:
  dido_test:
    schema_file: Bronanalyse_dido_test

# SAFE_GUARDS faciliteert het instellen van limieten zodat getest kan worden op
# een beperkte hoeveelheid data of dat afgebroken wordt als er teveel fouten zijn gedetekteerd
SAFE_GUARDS:
  data_test_limit: 10_000
  data_test_fraction: 0


LIMITATIONS:
  max_rows: 0 # <1 = read all rows
  max_errors: 1_000

# which columns to write for markdown tables and in which order
COLUMNS:
  - 'kolomnaam'
  - 'datatype'
  - 'beschrijving'
  - 'leverancier_kolomnaam'
  - 'leverancier_kolomtype'
  - 'code_attribuut_sleutel'
  - 'code_attribuut'
  - 'code_bronbestand'
  - 'keytype'
  - 'constraints'
  - 'domein'
  - 'verstek'
  - 'positie'
  - 'avg_classificatie'
  - 'veiligheid_classificatie'
  - 'attribuut_datum_begin'
  - 'attribuut_datum_einde'


# Allows conversion from leverancier_kolomtype to datatype when datatype is empty
DATATYPES:
    CHAR: text
    CUKY: text
    UNIT: text
    NUMC: numeric
    DATS: date     # gaat dit goed met inladen?
    QUAN: numeric
    DEC:  numeric  # float?
    CURR: numeric  # float?


### Parameters for d3g-import


### Parameters for d3g-data-prep

# Rename data on column base
# Conversies voor fake_oud: ontbrekende datums worden aangegeven door 0(0000000)
# In DWH is de conventie 29991231
# Niet nodig voor vertaaltabel
RENAME_DATA:
  fake_oud:
    datefrom:    {"00000000":"29991231", "0": "29991231"}
    dateto:      {"00000000":"29991231", "0": "29991231"}
    cpr_chon:    {"00000000":"29991231", "0": "29991231"}
    amount:      {"re": True, "(.*)-": "-\\1"}
  fake_nieuw:
    datefrom:    {"00000000":"29991231", "0": "29991231"}
    dateto:      {"00000000":"29991231", "0": "29991231"}
    cpr_chon:    {"00000000":"29991231", "0": "29991231"}
    amount:      {"re": True, "(.*)-": "-\\1"}
  faik:

# Parameters for d3g-create

# tables to be created
TABLES:
  schema:
    table_root: attribuut_meta
    template: yes
    comment: <odl>
    schema: <self>
    create_description: yes
    create_data: yes
    table: yes
    data: no
    prefix: ''
    suffix: ''

  meta:
    table_root: bestand_meta
    template: meta
    comment: <odl>
    schema: <odl>
    create_description: yes
    create_data: yes
    table: yes
    data: no
    prefix: ''
    suffix: ''

  datakwaliteit_feit:
    table_root: datakwaliteit_feit
    template: no
    comment: <odl>
    schema: <odl>
    create_description: yes
    create_data: yes
    table: yes
    data: no
    prefix: ''
    suffix: ''

  levering_feit:
    table_root: levering_feit
    template: no
    comment: <odl>
    schema: <odl>
    create_description: yes
    create_data: yes
    table: yes
    data: no
    prefix: ''
    suffix: ''
$$, $$test_data.csv$$, CURRENT_TIMESTAMP(0));

\copy didotest.dido_test_dido_import_schema_data (test_id, col_boolean, col_integer, col_decimal, col_date, check_null, check_minmax, check_list_numeric, check_list_text, check_re) FROM '/data/arnoldreinders/apps/dido/test_project/work/todo/dido_test/test_data.csv' WITH (FORMAT CSV, DELIMITER ';', HEADER, FORCE_NULL (test_id, col_integer, col_decimal, col_date, check_minmax, check_list_numeric));

UPDATE didotest.dido_test_dido_import_schema_data
    SET code_bronbestand = 'DIDO_TEST',
        levering_rapportageperiode = '2023-Q2',
        record_datum_begin = '1970-01-01 00:00:00',
        record_datum_einde = '9999-12-31 23:59:59',
        sysdatum = CURRENT_TIMESTAMP(0)
    WHERE (levering_rapportageperiode = '') IS NOT FALSE;

COMMIT; -- Transaction
