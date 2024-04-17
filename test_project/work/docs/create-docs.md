[[_TOC_]]

# **Tabel: dido_test**

## Meta-informatie

| Meta attribuut | Waarde 
| ---------- | ------ |
| code_bronbestand_sleutel  | DIDO |
| code_bronbestand  | DIDO_TEST |
| created_by  | current_user |
| bronbestand_beschrijving  | algemeen test procedures |
| bronbestand_naamconventie  |  |
| bronbestand_leverancier  | dwhteam |
| bronbestand_formaat  | csv |
| bronbestand_decimaal  | . |
| bronbestand_frequentielevering  |  |
| bronbestand_aantal_attributen  | 16 |
| bronbestand_gemiddeld_aantal_records  | 3 |
| bronbestand_voorlooprecord  | No |
| bronbestand_sluitrecord  | No |
| bronbestand_expiratie_datum  | 9999-12-31 |
| bronbestand_datum_begin  | 2022-01-01 |
| bronbestand_datum_einde  | 9999-12-31 |
| sysdatum  | 2024-04-09 09:14:49  |


## Databeschrijving

Tabel met metagegevens over de tabel metagegevens

 | Kolomnaam  | Datatype  | Beschrijving  | Leverancier kolomnaam  | Leverancier kolomtype  | Code attribuut sleutel  | Code attribuut  | Code bronbestand  | Keytype  | Constraints  | Domein  | Verstek  | Positie  | Avg classificatie  | Veiligheid classificatie  | Attribuut datum begin  | Attribuut datum einde  |
 | -----  | -----  | -----  | -----  | -----  | -----  | -----  | -----  | -----  | -----  | -----  | -----  | -----  | -----  | -----  | -----  | -----  |
| bronbestand_recordnummer | bigserial | Uniek recordnummer in de tabel | (dido generated) |  | DIDO_TEST001 | 001 | DIDO_TEST |  |  |  |  | 1 | 1 | 1 | 2022-01-01 | 9999-12-31 | 
| code_bronbestand | text | Unieke code voor identificatie van bronbestanden | (dido generated) |  | DIDO_TEST002 | 002 | DIDO_TEST |  | DEFAULT 'DIDO_TEST' |  |  | 2 | 1 | 1 | 2022-01-01 | 9999-12-31 | 
| levering_rapportageperiode | text | De frekwentie waarmee de bronbestanden worden geleverd (bijvoorbeeld jaarlijks of wekelijks) | (dido generated) |  | DIDO_TEST003 | 003 | DIDO_TEST |  | DEFAULT '' |  |  | 3 | 1 | 1 | 2022-01-01 | 9999-12-31 | 
| record_datum_begin | date | Eerste geldigheidsdatum en tijd van dit record | (dido generated) |  | DIDO_TEST004 | 004 | DIDO_TEST |  | DEFAULT '1970-01-01' |  |  | 4 | 1 | 1 | 2022-01-01 | 9999-12-31 | 
| record_datum_einde | date | Laatste geldigheidsdatum en tijd van dit record | (dido generated) |  | DIDO_TEST005 | 005 | DIDO_TEST |  | DEFAULT '9999-12-31' |  |  | 5 | 1 | 1 | 2022-01-01 | 9999-12-31 | 
| sysdatum | timestamp | De datum waarop dit record is aangemaakt | (dido generated) |  | DIDO_TEST006 | 006 | DIDO_TEST |  | DEFAULT CURRENT_TIMESTAMP(0) |  |  | 6 | 1 | 1 | 2022-01-01 | 9999-12-31 | 
| test_id | integer | SerialId | /TEST/ID | NUMC | DIDO_TEST007 | 007 | DIDO_TEST |  |  |  |  | 7 | 1 | 1 | 2022-01-01 | 9999-12-31 | 
| col_boolean | text | Check Boolean Type | /COL/BOOLEAN | CHAR | DIDO_TEST008 | 008 | DIDO_TEST |  |  | ['','true','false'] |  | 8 | 1 | 1 | 2022-01-01 | 9999-12-31 | 
| col_integer | integer | Check Integer Type | /COL/INTEGER | NUMC | DIDO_TEST009 | 009 | DIDO_TEST |  |  | 1:100 |  | 9 | 1 | 1 | 2022-01-01 | 9999-12-31 | 
| col_decimal | numeric | Check Decimal Type | /COL/DECIMAL | NUMC | DIDO_TEST010 | 010 | DIDO_TEST |  |  |  |  | 10 | 1 | 1 | 2022-01-01 | 9999-12-31 | 
| col_date | date | Check Date Type | /COL/DATE | DATS | DIDO_TEST011 | 011 | DIDO_TEST |  |  |  |  | 11 | 1 | 1 | 2022-01-01 | 9999-12-31 | 
| check_null | text | Check Null | /CHECK/NULL | CHAR | DIDO_TEST012 | 012 | DIDO_TEST |  | NOT NULL |  |  | 12 | 1 | 1 | 2022-01-01 | 9999-12-31 | 
| check_minmax | numeric | Check Min Max | CHECK/MINMAX | NUMC | DIDO_TEST013 | 013 | DIDO_TEST |  |  | [1:7] |  | 13 | 1 | 1 | 2022-01-01 | 9999-12-31 | 
| check_list_numeric | numeric | Check List Numeric values | CHECK/LIST/NUMERIC | NUMC | DIDO_TEST014 | 014 | DIDO_TEST |  |  | [3,5,7,9] |  | 14 | 1 | 1 | 2022-01-01 | 9999-12-31 | 
| check_list_text | text | Check List Text values | CHECK/LIST/TEXT | CHAR | DIDO_TEST015 | 015 | DIDO_TEST |  |  | ['ABC','DEF','XYZ','KLM','ASML',] |  | 15 | 1 | 1 | 2022-01-01 | 9999-12-31 | 
| check_re | text | Check Regular Expression | /CHECK/RE | CHAR | DIDO_TEST016 | 016 | DIDO_TEST |  |  |  |  | 16 | 2 | 2 | 2022-01-01 | 9999-12-31 | 


