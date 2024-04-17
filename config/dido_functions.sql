CREATE OR REPLACE FUNCTION {schema}.show_{supplier}_at(datum timestamp)
RETURNS SETOF {schema}.{table_name} AS
$body$
    SELECT *
    FROM {schema}.{table_name}
    WHERE record_datum_begin <= datum AND
          record_datum_einde >= datum;
$body$
LANGUAGE SQL;

CREATE OR REPLACE FUNCTION {schema}.show_history_of_{supplier}(id int)
RETURNS SETOF {schema}.{table_name} AS
$body$
    SELECT *
    FROM {schema}.{table_name}
    WHERE {key_id} = id;
$body$
LANGUAGE SQL;
