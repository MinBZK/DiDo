set -e # exit at any error = exit with non-0 argument
export LOG=logs/mutation-test.log
export ON_ERROR_STOP=true
export PROJ_DIR=/data/arnoldreinders/projects/ruimte/postcodes

# clear all tables for this test
python3.9 src/dido_kill_supplier.py -s mutations -p $PROJ_DIR --Yes

# Create tables SQL
python3.9 src/dido_create_tables.py -p $PROJ_DIR

# Execute the create tables instruction
psql -h 10.10.12.12 -U arnoldreinders -f $PROJ_DIR/work/sql/create-tables.sql ruimte

### Delivery 1, initial fill ###
# Preparee data
python3.9 src/dido_data_prep.py -p /data/arnoldreinders/projects/ruimte/postcodes

# Create SQL to load the data
python3.9 src/dido_import.py -p /data/arnoldreinders/projects/ruimte/postcodes

# And run it
psql -h 10.10.12.12 -U arnoldreinders -w -f $PROJ_DIR/work/sql/import-all-deliveries.sql ruimte

# dump table to file
python3.9 src/dido_compare.py -c dump -d 2023-01-15 -s mutations -t data-I1.csv -p $PROJ_DIR

### Delivery 2, first mutation ###
python3.9 src/dido_data_prep.py -p $PROJ_DIR
python3.9 src/dido_import.py -p $PROJ_DIR
psql -h 10.10.12.12 -U arnoldreinders -w -f $PROJ_DIR/work/sql/import-all-deliveries.sql ruimte
python3.9 src/dido_compare.py -c dump -d 2023-02-15 -s mutations -t data-M01.csv -p $PROJ_DIR

### Delivery 3, second mutation ###
python3.9 src/dido_data_prep.py -p $PROJ_DIR
python3.9 src/dido_import.py -p $PROJ_DIR
psql -h 10.10.12.12 -U arnoldreinders -w -f $PROJ_DIR/work/sql/import-all-deliveries.sql ruimte
python3.9 src/dido_compare.py -c dump -d 2023-03-15 -s mutations -t data-M02.csv -p $PROJ_DIR

### Delivery 4, third mutation ###
python3.9 src/dido_data_prep.py -p $PROJ_DIR
python3.9 src/dido_import.py -p $PROJ_DIR
psql -h 10.10.12.12 -U arnoldreinders -w -f $PROJ_DIR/work/sql/import-all-deliveries.sql ruimte
python3.9 src/dido_compare.py -c dump -d 2023-04-15 -s mutations -t data-M03.csv -p $PROJ_DIR


