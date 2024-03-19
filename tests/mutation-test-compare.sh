set -e # exit at any error = exit with non-0 argument
export LOG=logs/mutation-test.log
export ON_ERROR_STOP=true
export PROJ_DIR=/data/arnoldreinders/projects/ruimte/postcodes

# compare table to file
python3.9 src/dido_compare.py -c compare -r -v -d 2023-01-15 -s mutations -t data-I1.csv -p $PROJ_DIR

### Delivery 2, first mutation ###
python3.9 src/dido_compare.py -c compare -v -d 2023-02-15 -s mutations -t data-M01.csv -p $PROJ_DIR

### Delivery 3, second mutation ###
python3.9 src/dido_compare.py -c compare -v -d 2023-03-15 -s mutations -t data-M02.csv -p $PROJ_DIR

### Delivery 4, third mutation ###
python3.9 src/dido_compare.py -c compare -v -d 2023-04-15 -s mutations -t data-M03.csv -p $PROJ_DIR
