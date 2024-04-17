-- Quit immediately with exit code other than 0 when an error occurs
\set ON_ERROR_STOP true

BEGIN; -- Transaction

CREATE TABLE didotest.dido_test_dido_import_schema_description
(
   kolomnaam text,
   code_attribuut_sleutel text,
   code_attribuut text,
   code_bronbestand text,
   leverancier_kolomnaam text,
   leverancier_kolomtype text,
   leverancier_info_1 text,
   leverancier_info_2 text,
   datatype text,
   keytype text,
   constraints text,
   domein text,
   verstek text,
   positie text,
   avg_classificatie text,
   veiligheid_classificatie text,
   gebruiker_info_1 text,
   gebruiker_info_2 text,
   gebruiker_info_3 text,
   kolom_expiratie_datum text,
   attribuut_datum_begin text,
   attribuut_datum_einde text,
   beschrijving text
);

COMMENT ON TABLE didotest.dido_test_dido_import_schema_description IS $$Iedere rij in deze tabel bevat een beschrijving van de kolom die in de tabel wordt aangemaakt. De kolommen worden in deze 'root'-tabel beschreven.$$;

COMMENT ON COLUMN didotest.dido_test_dido_import_schema_description.kolomnaam IS $$Naam van de kolom zoals die bekend is bij de dataleverancier$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_description.code_attribuut_sleutel IS $$Unieke code. Conventie: De sleutel is de code van het attribuut en een volgnummer$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_description.code_attribuut IS $$Unieke code van een attribuut. Conventie: code bronbestand aangevuld met een volgnummer$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_description.code_bronbestand IS $$Unieke code voor identificatie van soorten bronbestanden$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_description.leverancier_kolomnaam IS $$Naam van de kolom zoals die door de leverancier wordt geleverd$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_description.leverancier_kolomtype IS $$Datatype van de kolom zoals opgegeven door de leverancier, vaak volgens het DBMS van de leverancier.$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_description.leverancier_info_1 IS $$Extra informatie over leverancier$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_description.leverancier_info_2 IS $$Extra informatie over leverancier$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_description.datatype IS $$Postgres datatype dat door DWH wordt geaccepteerd (zie domein).$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_description.keytype IS $$PK = Primary Key
FK = Foreign Key
anders geen key$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_description.constraints IS $$Database constraints: NOT NULL = waarde is verplicht, verder geen andere constrints$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_description.domein IS $$lijst met afzonderlijke waarden in datatype
text: ['a', 'b', 'c']
int: [1, 2, 3, 5, 8]
float: [3.14, 2.57]

Voor getallen: min:max volgens de python manier
int: min:max
float: min:max

Datum volgens ISO-8601
date: YYYY-MM-DD

re: <regular expression>$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_description.verstek IS $$default waarde  indien niet ingevuld$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_description.positie IS $$Aanduiding van de positie van het attribuut in het bronbestand$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_description.avg_classificatie IS $$Aanduiding van het datatype van het attribuut
1=Geen persoonsgegeven
2=Persoonsgegeven
3=Bijzonder persoonsgegeven$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_description.veiligheid_classificatie IS $$Aanduiding van de vertrouwelijksclassificatie in het kader van de AVG van het attribuut
1=Niet vertrouwelijk
2=Departementaal vertrouwelijk
3=Staatsgehein
4=Staatsgeheim, zeer geheim
5=Staatsgeheimn, Top Secret$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_description.gebruiker_info_1 IS $$Veld dat de gebruiker kan gebruiken om extra informatie over deze kolom in op te slaan.$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_description.gebruiker_info_2 IS $$Veld dat de gebruiker kan gebruiken om extra informatie over deze kolom in op te slaan.$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_description.gebruiker_info_3 IS $$Veld dat de gebruiker kan gebruiken om extra informatie over deze kolom in op te slaan.$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_description.kolom_expiratie_datum IS $$Datum waarop de kolom niet meer gebruikt mag worden. D3g kontroleert hier niet op, dit is aan de gebruiker of gebruikersapplikatie.$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_description.attribuut_datum_begin IS $$Begindatum waarop het attribuut wordt geleverd met bovenstaande kenmerken$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_description.attribuut_datum_einde IS $$De laatste datum waarop het attribuut wordt geleverd met bovenstaande kenmerken$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_description.beschrijving IS $$Uitgebreide beschrijving van de kolom.$$;


INSERT INTO didotest.dido_test_dido_import_schema_description (kolomnaam, code_attribuut_sleutel, code_attribuut, code_bronbestand, leverancier_kolomnaam, leverancier_kolomtype, leverancier_info_1, leverancier_info_2, datatype, keytype, constraints, domein, verstek, positie, avg_classificatie, veiligheid_classificatie, gebruiker_info_1, gebruiker_info_2, gebruiker_info_3, kolom_expiratie_datum, attribuut_datum_begin, attribuut_datum_einde, beschrijving)
VALUES
($$bronbestand_recordnummer$$, $$DIDO_TEST001$$, $$001$$, $$DIDO_TEST$$, $$(dido generated)$$, $$$$, $$$$, $$$$, $$bigserial$$, $$$$, $$$$, $$$$, $$$$, $$1$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$$$, $$2022-01-01$$, $$9999-12-31$$, $$Uniek recordnummer in de tabel$$),
($$code_bronbestand$$, $$DIDO_TEST002$$, $$002$$, $$DIDO_TEST$$, $$(dido generated)$$, $$$$, $$$$, $$$$, $$text$$, $$$$, $$DEFAULT 'DIDO_TEST'$$, $$$$, $$$$, $$2$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$$$, $$2022-01-01$$, $$9999-12-31$$, $$Unieke code voor identificatie van bronbestanden$$),
($$levering_rapportageperiode$$, $$DIDO_TEST003$$, $$003$$, $$DIDO_TEST$$, $$(dido generated)$$, $$$$, $$$$, $$$$, $$text$$, $$$$, $$DEFAULT ''$$, $$$$, $$$$, $$3$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$$$, $$2022-01-01$$, $$9999-12-31$$, $$De frekwentie waarmee de bronbestanden worden geleverd (bijvoorbeeld jaarlijks of wekelijks)$$),
($$record_datum_begin$$, $$DIDO_TEST004$$, $$004$$, $$DIDO_TEST$$, $$(dido generated)$$, $$$$, $$$$, $$$$, $$date$$, $$$$, $$DEFAULT '1970-01-01'$$, $$$$, $$$$, $$4$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$$$, $$2022-01-01$$, $$9999-12-31$$, $$Eerste geldigheidsdatum en tijd van dit record$$),
($$record_datum_einde$$, $$DIDO_TEST005$$, $$005$$, $$DIDO_TEST$$, $$(dido generated)$$, $$$$, $$$$, $$$$, $$date$$, $$$$, $$DEFAULT '9999-12-31'$$, $$$$, $$$$, $$5$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$$$, $$2022-01-01$$, $$9999-12-31$$, $$Laatste geldigheidsdatum en tijd van dit record$$),
($$sysdatum$$, $$DIDO_TEST006$$, $$006$$, $$DIDO_TEST$$, $$(dido generated)$$, $$$$, $$$$, $$$$, $$timestamp$$, $$$$, $$DEFAULT CURRENT_TIMESTAMP(0)$$, $$$$, $$$$, $$6$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$$$, $$2022-01-01$$, $$9999-12-31$$, $$De datum waarop dit record is aangemaakt$$),
($$test_id$$, $$DIDO_TEST007$$, $$007$$, $$DIDO_TEST$$, $$/TEST/ID$$, $$NUMC$$, $$$$, $$$$, $$integer$$, $$$$, $$$$, $$$$, $$$$, $$7$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$$$, $$2022-01-01$$, $$9999-12-31$$, $$SerialId$$),
($$col_boolean$$, $$DIDO_TEST008$$, $$008$$, $$DIDO_TEST$$, $$/COL/BOOLEAN$$, $$CHAR$$, $$$$, $$$$, $$text$$, $$$$, $$$$, $$['','true','false']$$, $$$$, $$8$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$$$, $$2022-01-01$$, $$9999-12-31$$, $$Check Boolean Type$$),
($$col_integer$$, $$DIDO_TEST009$$, $$009$$, $$DIDO_TEST$$, $$/COL/INTEGER$$, $$NUMC$$, $$$$, $$$$, $$integer$$, $$$$, $$$$, $$1:100$$, $$$$, $$9$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$$$, $$2022-01-01$$, $$9999-12-31$$, $$Check Integer Type$$),
($$col_decimal$$, $$DIDO_TEST010$$, $$010$$, $$DIDO_TEST$$, $$/COL/DECIMAL$$, $$NUMC$$, $$$$, $$$$, $$numeric$$, $$$$, $$$$, $$$$, $$$$, $$10$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$$$, $$2022-01-01$$, $$9999-12-31$$, $$Check Decimal Type$$),
($$col_date$$, $$DIDO_TEST011$$, $$011$$, $$DIDO_TEST$$, $$/COL/DATE$$, $$DATS$$, $$$$, $$$$, $$date$$, $$$$, $$$$, $$$$, $$$$, $$11$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$$$, $$2022-01-01$$, $$9999-12-31$$, $$Check Date Type$$),
($$check_null$$, $$DIDO_TEST012$$, $$012$$, $$DIDO_TEST$$, $$/CHECK/NULL$$, $$CHAR$$, $$$$, $$$$, $$text$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$12$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$$$, $$2022-01-01$$, $$9999-12-31$$, $$Check Null$$),
($$check_minmax$$, $$DIDO_TEST013$$, $$013$$, $$DIDO_TEST$$, $$CHECK/MINMAX$$, $$NUMC$$, $$$$, $$$$, $$numeric$$, $$$$, $$$$, $$[1:7]$$, $$$$, $$13$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$$$, $$2022-01-01$$, $$9999-12-31$$, $$Check Min Max$$),
($$check_list_numeric$$, $$DIDO_TEST014$$, $$014$$, $$DIDO_TEST$$, $$CHECK/LIST/NUMERIC$$, $$NUMC$$, $$$$, $$$$, $$numeric$$, $$$$, $$$$, $$[3,5,7,9]$$, $$$$, $$14$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$$$, $$2022-01-01$$, $$9999-12-31$$, $$Check List Numeric values$$),
($$check_list_text$$, $$DIDO_TEST015$$, $$015$$, $$DIDO_TEST$$, $$CHECK/LIST/TEXT$$, $$CHAR$$, $$$$, $$$$, $$text$$, $$$$, $$$$, $$['ABC','DEF','XYZ','KLM','ASML',]$$, $$$$, $$15$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$$$, $$2022-01-01$$, $$9999-12-31$$, $$Check List Text values$$),
($$check_re$$, $$DIDO_TEST016$$, $$016$$, $$DIDO_TEST$$, $$/CHECK/RE$$, $$CHAR$$, $$$$, $$$$, $$text$$, $$$$, $$$$, $$$$, $$$$, $$16$$, $$2$$, $$2$$, $$$$, $$$$, $$$$, $$$$, $$2022-01-01$$, $$9999-12-31$$, $$Check Regular Expression$$);

CREATE TABLE didotest.dido_test_dido_import_schema_data
(
   bronbestand_recordnummer bigserial,
   code_bronbestand text DEFAULT 'DIDO_TEST',
   levering_rapportageperiode text DEFAULT '',
   record_datum_begin date DEFAULT '1970-01-01',
   record_datum_einde date DEFAULT '9999-12-31',
   sysdatum timestamp DEFAULT CURRENT_TIMESTAMP(0),
   test_id integer,
   col_boolean text,
   col_integer integer,
   col_decimal numeric,
   col_date date,
   check_null text NOT NULL,
   check_minmax numeric,
   check_list_numeric numeric,
   check_list_text text,
   check_re text
);

COMMENT ON COLUMN didotest.dido_test_dido_import_schema_data.bronbestand_recordnummer IS $$Uniek recordnummer in de tabel$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_data.code_bronbestand IS $$Unieke code voor identificatie van bronbestanden$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_data.levering_rapportageperiode IS $$De frekwentie waarmee de bronbestanden worden geleverd (bijvoorbeeld jaarlijks of wekelijks)$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_data.record_datum_begin IS $$Eerste geldigheidsdatum en tijd van dit record$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_data.record_datum_einde IS $$Laatste geldigheidsdatum en tijd van dit record$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_data.sysdatum IS $$De datum waarop dit record is aangemaakt$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_data.test_id IS $$SerialId$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_data.col_boolean IS $$Check Boolean Type$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_data.col_integer IS $$Check Integer Type$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_data.col_decimal IS $$Check Decimal Type$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_data.col_date IS $$Check Date Type$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_data.check_null IS $$Check Null$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_data.check_minmax IS $$Check Min Max$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_data.check_list_numeric IS $$Check List Numeric values$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_data.check_list_text IS $$Check List Text values$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_schema_data.check_re IS $$Check Regular Expression$$;




CREATE TABLE didotest.dido_test_dido_import_meta_description
(
   kolomnaam text,
   code_attribuut_sleutel text,
   code_attribuut text,
   code_bronbestand text,
   leverancier_kolomnaam text,
   leverancier_kolomtype text,
   leverancier_info_1 text,
   leverancier_info_2 text,
   datatype text,
   keytype text,
   constraints text,
   domein text,
   verstek text,
   positie text,
   avg_classificatie text,
   veiligheid_classificatie text,
   gebruiker_info_1 text,
   gebruiker_info_2 text,
   gebruiker_info_3 text,
   kolom_expiratie_datum text,
   attribuut_datum_begin text,
   attribuut_datum_einde text,
   beschrijving text
);

COMMENT ON TABLE didotest.dido_test_dido_import_meta_description IS $$Tabel met metagegevens over de tabel metagegevens$$;



INSERT INTO didotest.dido_test_dido_import_meta_description (kolomnaam, code_attribuut_sleutel, code_attribuut, code_bronbestand, leverancier_kolomnaam, leverancier_kolomtype, leverancier_info_1, leverancier_info_2, datatype, keytype, constraints, domein, verstek, positie, avg_classificatie, veiligheid_classificatie, gebruiker_info_1, gebruiker_info_2, gebruiker_info_3, kolom_expiratie_datum, attribuut_datum_begin, attribuut_datum_einde, beschrijving)
VALUES
($$code_bronbestand_sleutel$$, $$ODLMETA001$$, $$001$$, $$ODLMETA$$, $$$$, $$alfanumeriek$$, $$$$, $$$$, $$text$$, $$PK$$, $$NOT NULL$$, $$$$, $$$$, $$1$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Unieke sleutel van de entiteit. Dit is volgens conventie de code van het bronbestand en een volgnummer$$),
($$code_bronbestand$$, $$ODLMETA002$$, $$002$$, $$ODLMETA$$, $$$$, $$alfanumeriek$$, $$$$, $$$$, $$text$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$2$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Unieke code voor identificatie van bronbestanden (BGFMHLWD: Bezettinggraad data van FMH voor locatie Leeuwarden).$$),
($$created_by$$, $$ODLMETA003$$, $$003$$, $$ODLMETA$$, $$$$, $$alfanumeriek$$, $$$$, $$$$, $$text$$, $$$$, $$$$, $$$$, $$$$, $$3$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Gberuikersnaam van de gebruiker die de tabel aanmaakte$$),
($$bronbestand_beschrijving$$, $$ODLMETA004$$, $$004$$, $$ODLMETA$$, $$$$, $$alfanumeriek$$, $$$$, $$$$, $$text$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$4$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Beschrijving van het bronbestand$$),
($$bronbestand_naamconventie$$, $$ODLMETA005$$, $$005$$, $$ODLMETA$$, $$$$, $$alfanumeriek$$, $$$$, $$$$, $$text$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$5$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$De standaard opbouw van de naam van het bronbestand$$),
($$bronbestand_leverancier$$, $$ODLMETA006$$, $$006$$, $$ODLMETA$$, $$$$, $$alfanumeriek$$, $$$$, $$$$, $$text$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$6$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Omschrijving van de authentieke leverancier/registratie van de brondata$$),
($$bronbestand_formaat$$, $$ODLMETA007$$, $$007$$, $$ODLMETA$$, $$$$, $$alfanumeriek$$, $$$$, $$$$, $$text$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$7$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Beschrijving van het formaat van het bronbestand (bv csv met puntkomma als seperator)$$),
($$bronbestand_decimaal$$, $$ODLMETA008$$, $$008$$, $$ODLMETA$$, $$$$, $$alfanumeriek$$, $$$$, $$$$, $$text$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$8$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Decimale teken dat wordt gebruikt bij de aangeleverde tabellen (komma of punt of iets anders)$$),
($$bronbestand_frequentielevering$$, $$ODLMETA009$$, $$009$$, $$ODLMETA$$, $$$$, $$alfanumeriek$$, $$$$, $$$$, $$text$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$9$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$De frequentie waarmee de beonbestanden worden geleverd (Jaar Halfjaar Quartaal Maand Week Dag of Anders)$$),
($$bronbestand_aantal_attributen$$, $$ODLMETA010$$, $$010$$, $$ODLMETA$$, $$$$, $$integer$$, $$$$, $$$$, $$integer$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$10$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Het aantal attributen in het bronbestand$$),
($$bronbestand_gemiddeld_aantal_records$$, $$ODLMETA011$$, $$011$$, $$ODLMETA$$, $$$$, $$integer$$, $$$$, $$$$, $$integer$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$11$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Het gemiddeld aantal te verwachten records in de levering$$),
($$bronbestand_voorlooprecord$$, $$ODLMETA012$$, $$012$$, $$ODLMETA$$, $$$$, $$waar/niet waar$$, $$$$, $$$$, $$boolean$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$12$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Aanduiding of het bronbestand een voorlooprecord heeft$$),
($$bronbestand_sluitrecord$$, $$ODLMETA013$$, $$013$$, $$ODLMETA$$, $$$$, $$waar/niet waar$$, $$$$, $$$$, $$boolean$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$13$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Aanduiding of het bronbestand een sluitrecord heeft$$),
($$bronbestand_expiratie_datum$$, $$ODLMETA014$$, $$014$$, $$ODLMETA$$, $$$$, $$datum$$, $$$$, $$$$, $$date$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$14$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-10-20$$, $$9999-12-31$$, $$Datum waarop de tabel niet meer gebruikt mag worden. Het is aan de gebruiker om hierop te controleren.$$),
($$bronbestand_datum_begin$$, $$ODLMETA015$$, $$015$$, $$ODLMETA$$, $$$$, $$datum$$, $$$$, $$$$, $$date$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$15$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Begindatum waarop de bronbestand wordt geleverd met bovenstaande kenmerken$$),
($$bronbestand_datum_einde$$, $$ODLMETA016$$, $$016$$, $$ODLMETA$$, $$$$, $$datum$$, $$$$, $$$$, $$date$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$16$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$De laatste datum waarop de bronbestand wordt geleverd met bovenstaande kenmerken$$),
($$sysdatum$$, $$ODLMETA017$$, $$017$$, $$ODLMETA$$, $$$$, $$datum$$, $$$$, $$$$, $$timestamp$$, $$$$, $$$$, $$$$, $$$$, $$17$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Datum inlezen data$$);

CREATE TABLE didotest.dido_test_dido_import_meta_data
(
   code_bronbestand_sleutel text NOT NULL,
   code_bronbestand text NOT NULL,
   created_by text,
   bronbestand_beschrijving text NOT NULL,
   bronbestand_naamconventie text NOT NULL,
   bronbestand_leverancier text NOT NULL,
   bronbestand_formaat text NOT NULL,
   bronbestand_decimaal text NOT NULL,
   bronbestand_frequentielevering text NOT NULL,
   bronbestand_aantal_attributen integer NOT NULL,
   bronbestand_gemiddeld_aantal_records integer NOT NULL,
   bronbestand_voorlooprecord boolean NOT NULL,
   bronbestand_sluitrecord boolean NOT NULL,
   bronbestand_expiratie_datum date NOT NULL,
   bronbestand_datum_begin date NOT NULL,
   bronbestand_datum_einde date NOT NULL,
   sysdatum timestamp
);

COMMENT ON COLUMN didotest.dido_test_dido_import_meta_data.code_bronbestand_sleutel IS $$Unieke sleutel van de entiteit. Dit is volgens conventie de code van het bronbestand en een volgnummer$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_meta_data.code_bronbestand IS $$Unieke code voor identificatie van bronbestanden (BGFMHLWD: Bezettinggraad data van FMH voor locatie Leeuwarden).$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_meta_data.created_by IS $$Gberuikersnaam van de gebruiker die de tabel aanmaakte$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_meta_data.bronbestand_beschrijving IS $$Beschrijving van het bronbestand$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_meta_data.bronbestand_naamconventie IS $$De standaard opbouw van de naam van het bronbestand$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_meta_data.bronbestand_leverancier IS $$Omschrijving van de authentieke leverancier/registratie van de brondata$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_meta_data.bronbestand_formaat IS $$Beschrijving van het formaat van het bronbestand (bv csv met puntkomma als seperator)$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_meta_data.bronbestand_decimaal IS $$Decimale teken dat wordt gebruikt bij de aangeleverde tabellen (komma of punt of iets anders)$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_meta_data.bronbestand_frequentielevering IS $$De frequentie waarmee de beonbestanden worden geleverd (Jaar Halfjaar Quartaal Maand Week Dag of Anders)$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_meta_data.bronbestand_aantal_attributen IS $$Het aantal attributen in het bronbestand$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_meta_data.bronbestand_gemiddeld_aantal_records IS $$Het gemiddeld aantal te verwachten records in de levering$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_meta_data.bronbestand_voorlooprecord IS $$Aanduiding of het bronbestand een voorlooprecord heeft$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_meta_data.bronbestand_sluitrecord IS $$Aanduiding of het bronbestand een sluitrecord heeft$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_meta_data.bronbestand_expiratie_datum IS $$Datum waarop de tabel niet meer gebruikt mag worden. Het is aan de gebruiker om hierop te controleren.$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_meta_data.bronbestand_datum_begin IS $$Begindatum waarop de bronbestand wordt geleverd met bovenstaande kenmerken$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_meta_data.bronbestand_datum_einde IS $$De laatste datum waarop de bronbestand wordt geleverd met bovenstaande kenmerken$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_meta_data.sysdatum IS $$Datum inlezen data$$;


INSERT INTO didotest.dido_test_dido_import_meta_data (code_bronbestand_sleutel, code_bronbestand, created_by, bronbestand_beschrijving, bronbestand_naamconventie, bronbestand_leverancier, bronbestand_formaat, bronbestand_decimaal, bronbestand_frequentielevering, bronbestand_aantal_attributen, bronbestand_gemiddeld_aantal_records, bronbestand_voorlooprecord, bronbestand_sluitrecord, bronbestand_expiratie_datum, bronbestand_datum_begin, bronbestand_datum_einde, sysdatum)
VALUES
($$DIDO$$, $$DIDO_TEST$$, current_user, $$algemeen test procedures$$, $$$$, $$dwhteam$$, $$csv$$, $$.$$, $$$$, $$16$$, $$3$$, $$No$$, $$No$$, $$9999-12-31$$, $$2022-01-01$$, $$9999-12-31$$, $$2024-04-09 09:14:49 $$);



CREATE TABLE didotest.dido_test_dido_import_datakwaliteit_feit_description
(
   kolomnaam text,
   code_attribuut_sleutel text,
   code_attribuut text,
   code_bronbestand text,
   leverancier_kolomnaam text,
   leverancier_kolomtype text,
   leverancier_info_1 text,
   leverancier_info_2 text,
   datatype text,
   keytype text,
   constraints text,
   domein text,
   verstek text,
   positie text,
   avg_classificatie text,
   veiligheid_classificatie text,
   gebruiker_info_1 text,
   gebruiker_info_2 text,
   gebruiker_info_3 text,
   kolom_expiratie_datum text,
   attribuut_datum_begin text,
   attribuut_datum_einde text,
   beschrijving text
);

COMMENT ON TABLE didotest.dido_test_dido_import_datakwaliteit_feit_description IS $$Datakwaliteitscore per bronbestand, voor een specifieke levering per record en attribuut$$;

COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_description.kolomnaam IS $$Naam van de kolom zoals die bekend is bij de dataleverancier$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_description.code_attribuut_sleutel IS $$Unieke code. Conventie: De sleutel is de code van het attribuut en een volgnummer$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_description.code_attribuut IS $$Unieke code van een attribuut. Conventie: code bronbestand aangevuld met een volgnummer$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_description.code_bronbestand IS $$Unieke code voor identificatie van soorten bronbestanden$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_description.leverancier_kolomnaam IS $$Naam van de kolom zoals die door de leverancier wordt geleverd$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_description.leverancier_kolomtype IS $$Datatype van de kolom zoals opgegeven door de leverancier, vaak volgens het DBMS van de leverancier.$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_description.leverancier_info_1 IS $$Extra informatie over leverancier$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_description.leverancier_info_2 IS $$Extra informatie over leverancier$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_description.datatype IS $$Postgres datatype dat door DWH wordt geaccepteerd (zie domein).$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_description.keytype IS $$PK = Primary Key
FK = Foreign Key
anders geen key$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_description.constraints IS $$Database constraints: NOT NULL = waarde is verplicht, verder geen andere constrints$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_description.domein IS $$lijst met afzonderlijke waarden in datatype
text: ['a', 'b', 'c']
int: [1, 2, 3, 5, 8]
float: [3.14, 2.57]

Voor getallen: min:max volgens de python manier
int: min:max
float: min:max

Datum volgens ISO-8601
date: YYYY-MM-DD

re: <regular expression>$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_description.verstek IS $$default waarde  indien niet ingevuld$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_description.positie IS $$Aanduiding van de positie van het attribuut in het bronbestand$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_description.avg_classificatie IS $$Aanduiding van het datatype van het attribuut
1=Geen persoonsgegeven
2=Persoonsgegeven
3=Bijzonder persoonsgegeven$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_description.veiligheid_classificatie IS $$Aanduiding van de vertrouwelijksclassificatie in het kader van de AVG van het attribuut
1=Niet vertrouwelijk
2=Departementaal vertrouwelijk
3=Staatsgehein
4=Staatsgeheim, zeer geheim
5=Staatsgeheimn, Top Secret$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_description.gebruiker_info_1 IS $$Veld dat de gebruiker kan gebruiken om extra informatie over deze kolom in op te slaan.$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_description.gebruiker_info_2 IS $$Veld dat de gebruiker kan gebruiken om extra informatie over deze kolom in op te slaan.$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_description.gebruiker_info_3 IS $$Veld dat de gebruiker kan gebruiken om extra informatie over deze kolom in op te slaan.$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_description.kolom_expiratie_datum IS $$Datum waarop de kolom niet meer gebruikt mag worden. D3g kontroleert hier niet op, dit is aan de gebruiker of gebruikersapplikatie.$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_description.attribuut_datum_begin IS $$Begindatum waarop het attribuut wordt geleverd met bovenstaande kenmerken$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_description.attribuut_datum_einde IS $$De laatste datum waarop het attribuut wordt geleverd met bovenstaande kenmerken$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_description.beschrijving IS $$Uitgebreide beschrijving van de kolom.$$;


INSERT INTO didotest.dido_test_dido_import_datakwaliteit_feit_description (kolomnaam, code_attribuut_sleutel, code_attribuut, code_bronbestand, leverancier_kolomnaam, leverancier_kolomtype, leverancier_info_1, leverancier_info_2, datatype, keytype, constraints, domein, verstek, positie, avg_classificatie, veiligheid_classificatie, gebruiker_info_1, gebruiker_info_2, gebruiker_info_3, kolom_expiratie_datum, attribuut_datum_begin, attribuut_datum_einde, beschrijving)
VALUES
($$bronbestand_recordnummer$$, $$ODLDKF001$$, $$001$$, $$ODLDKF$$, $$$$, $$teller$$, $$$$, $$$$, $$bigint$$, $$FK$$, $$NOT NULL$$, $$$$, $$$$, $$1$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Een sequentiële teller van de records die worden ingelezen in deze entiteit$$),
($$code_bronbestand$$, $$ODLDKF002$$, $$002$$, $$ODLDKF$$, $$$$, $$alfanumeriek$$, $$$$, $$$$, $$text$$, $$FK$$, $$NOT NULL$$, $$$$, $$$$, $$2$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Unieke code voor identificatie van bronbestanden (BGFMHLWD: Bezettinggraad data van FMH voor locatie Leeuwarden).$$),
($$row_number$$, $$ODLDKF003$$, $$003$$, $$ODLDKF$$, $$$$, $$numeriek$$, $$$$, $$$$, $$integer$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$3$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Rijnummer in oorspronkelijke data waar de fout werd gevonden$$),
($$column_name$$, $$ODLDKF004$$, $$004$$, $$ODLDKF$$, $$$$, $$alfanumeriek$$, $$$$, $$$$, $$text$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$4$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Naam van de kolom waar de fout werd gevonden$$),
($$code_attribuut$$, $$ODLDKF005$$, $$005$$, $$ODLDKF$$, $$$$, $$alfanumeriek$$, $$$$, $$$$, $$text$$, $$FK$$, $$NOT NULL$$, $$$$, $$$$, $$5$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Unieke code van attribuut x in het bronbestand. X is gelijk aan het aantal attributen in het bronbestand (zie 'Bronbestand aantal attributen' in 'Meta bronbestand')$$),
($$code_datakwaliteit$$, $$ODLDKF006$$, $$006$$, $$ODLDKF$$, $$$$, $$numeriek$$, $$$$, $$$$, $$integer$$, $$FK$$, $$NOT NULL$$, $$$$, $$$$, $$6$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Unieke code ter indicatie van de datakwaliteit van individuele waarde van een attribuut$$),
($$levering_rapportageperiode$$, $$ODLDKF007$$, $$007$$, $$ODLDKF$$, $$$$, $$alfanumeriek$$, $$$$, $$$$, $$text$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$7$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Unieke beschrijving van de boekings/rapportageperiode (WWYYYY(342023)/MMYYYY(112023)/QQYYYYY(022023)/YYYYYY(232023))$$),
($$sysdatum$$, $$ODLDKF008$$, $$008$$, $$ODLDKF$$, $$$$, $$datum$$, $$$$, $$$$, $$timestamp$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$8$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Datum inlezen data$$);

CREATE TABLE didotest.dido_test_dido_import_datakwaliteit_feit_data
(
   bronbestand_recordnummer bigint NOT NULL,
   code_bronbestand text NOT NULL,
   row_number integer NOT NULL,
   column_name text NOT NULL,
   code_attribuut text NOT NULL,
   code_datakwaliteit integer NOT NULL,
   levering_rapportageperiode text NOT NULL,
   sysdatum timestamp NOT NULL
);

COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_data.bronbestand_recordnummer IS $$Een sequentiële teller van de records die worden ingelezen in deze entiteit$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_data.code_bronbestand IS $$Unieke code voor identificatie van bronbestanden (BGFMHLWD: Bezettinggraad data van FMH voor locatie Leeuwarden).$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_data.row_number IS $$Rijnummer in oorspronkelijke data waar de fout werd gevonden$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_data.column_name IS $$Naam van de kolom waar de fout werd gevonden$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_data.code_attribuut IS $$Unieke code van attribuut x in het bronbestand. X is gelijk aan het aantal attributen in het bronbestand (zie 'Bronbestand aantal attributen' in 'Meta bronbestand')$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_data.code_datakwaliteit IS $$Unieke code ter indicatie van de datakwaliteit van individuele waarde van een attribuut$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_data.levering_rapportageperiode IS $$Unieke beschrijving van de boekings/rapportageperiode (WWYYYY(342023)/MMYYYY(112023)/QQYYYYY(022023)/YYYYYY(232023))$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_datakwaliteit_feit_data.sysdatum IS $$Datum inlezen data$$;




CREATE TABLE didotest.dido_test_dido_import_levering_feit_description
(
   kolomnaam text,
   code_attribuut_sleutel text,
   code_attribuut text,
   code_bronbestand text,
   leverancier_kolomnaam text,
   leverancier_kolomtype text,
   leverancier_info_1 text,
   leverancier_info_2 text,
   datatype text,
   keytype text,
   constraints text,
   domein text,
   verstek text,
   positie text,
   avg_classificatie text,
   veiligheid_classificatie text,
   gebruiker_info_1 text,
   gebruiker_info_2 text,
   gebruiker_info_3 text,
   kolom_expiratie_datum text,
   attribuut_datum_begin text,
   attribuut_datum_einde text,
   beschrijving text
);

COMMENT ON TABLE didotest.dido_test_dido_import_levering_feit_description IS $$Gegevens over iedere levering van het bronbestand$$;

COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_description.kolomnaam IS $$Naam van de kolom zoals die bekend is bij de dataleverancier$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_description.code_attribuut_sleutel IS $$Unieke code. Conventie: De sleutel is de code van het attribuut en een volgnummer$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_description.code_attribuut IS $$Unieke code van een attribuut. Conventie: code bronbestand aangevuld met een volgnummer$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_description.code_bronbestand IS $$Unieke code voor identificatie van soorten bronbestanden$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_description.leverancier_kolomnaam IS $$Naam van de kolom zoals die door de leverancier wordt geleverd$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_description.leverancier_kolomtype IS $$Datatype van de kolom zoals opgegeven door de leverancier, vaak volgens het DBMS van de leverancier.$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_description.leverancier_info_1 IS $$Extra informatie over leverancier$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_description.leverancier_info_2 IS $$Extra informatie over leverancier$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_description.datatype IS $$Postgres datatype dat door DWH wordt geaccepteerd (zie domein).$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_description.keytype IS $$PK = Primary Key
FK = Foreign Key
anders geen key$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_description.constraints IS $$Database constraints: NOT NULL = waarde is verplicht, verder geen andere constrints$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_description.domein IS $$lijst met afzonderlijke waarden in datatype
text: ['a', 'b', 'c']
int: [1, 2, 3, 5, 8]
float: [3.14, 2.57]

Voor getallen: min:max volgens de python manier
int: min:max
float: min:max

Datum volgens ISO-8601
date: YYYY-MM-DD

re: <regular expression>$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_description.verstek IS $$default waarde  indien niet ingevuld$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_description.positie IS $$Aanduiding van de positie van het attribuut in het bronbestand$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_description.avg_classificatie IS $$Aanduiding van het datatype van het attribuut
1=Geen persoonsgegeven
2=Persoonsgegeven
3=Bijzonder persoonsgegeven$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_description.veiligheid_classificatie IS $$Aanduiding van de vertrouwelijksclassificatie in het kader van de AVG van het attribuut
1=Niet vertrouwelijk
2=Departementaal vertrouwelijk
3=Staatsgehein
4=Staatsgeheim, zeer geheim
5=Staatsgeheimn, Top Secret$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_description.gebruiker_info_1 IS $$Veld dat de gebruiker kan gebruiken om extra informatie over deze kolom in op te slaan.$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_description.gebruiker_info_2 IS $$Veld dat de gebruiker kan gebruiken om extra informatie over deze kolom in op te slaan.$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_description.gebruiker_info_3 IS $$Veld dat de gebruiker kan gebruiken om extra informatie over deze kolom in op te slaan.$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_description.kolom_expiratie_datum IS $$Datum waarop de kolom niet meer gebruikt mag worden. D3g kontroleert hier niet op, dit is aan de gebruiker of gebruikersapplikatie.$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_description.attribuut_datum_begin IS $$Begindatum waarop het attribuut wordt geleverd met bovenstaande kenmerken$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_description.attribuut_datum_einde IS $$De laatste datum waarop het attribuut wordt geleverd met bovenstaande kenmerken$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_description.beschrijving IS $$Uitgebreide beschrijving van de kolom.$$;


INSERT INTO didotest.dido_test_dido_import_levering_feit_description (kolomnaam, code_attribuut_sleutel, code_attribuut, code_bronbestand, leverancier_kolomnaam, leverancier_kolomtype, leverancier_info_1, leverancier_info_2, datatype, keytype, constraints, domein, verstek, positie, avg_classificatie, veiligheid_classificatie, gebruiker_info_1, gebruiker_info_2, gebruiker_info_3, kolom_expiratie_datum, attribuut_datum_begin, attribuut_datum_einde, beschrijving)
VALUES
($$levering_rapportageperiode_volgnummer$$, $$ODLLF001$$, $$001$$, $$ODLLF$$, $$$$, $$numeriek$$, $$$$, $$$$, $$integer$$, $$PK$$, $$NOT NULL$$, $$$$, $$$$, $$1$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Uniek nummer voor dit record in de tabel$$),
($$levering_rapportageperiode$$, $$ODLLF002$$, $$002$$, $$ODLLF$$, $$$$, $$alfanumeriek$$, $$$$, $$$$, $$text$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$2$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Unieke beschrijving van de boekings/rapportageperiode (WWYYYY(342023)/MMYYYY(112023)/QQYYYYY(022023)/YYYYYY(232023))$$),
($$code_bronbestand$$, $$ODLLF003$$, $$003$$, $$ODLLF$$, $$$$, $$alfanumeriek$$, $$$$, $$$$, $$text$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$3$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Unieke code voor identificatie van bronbestanden (BGFMHLWD: Bezettinggraad data van FMH voor locatie Leeuwarden).$$),
($$created_by$$, $$ODLLF004$$, $$004$$, $$ODLLF$$, $$$$, $$alfanumeriek$$, $$$$, $$$$, $$text$$, $$$$, $$$$, $$$$, $$$$, $$4$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Naam van de gebruiker die de levering aanmaakte$$),
($$levering_leveringsdatum$$, $$ODLLF005$$, $$005$$, $$ODLLF$$, $$$$, $$datum$$, $$$$, $$$$, $$date$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$5$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$De datum waarop de levering is ontvangen in de staging area$$),
($$levering_goed_voor_verwerking$$, $$ODLLF006$$, $$006$$, $$ODLLF$$, $$$$, $$waar/niet waar$$, $$$$, $$$$, $$boolean$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$6$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Indicatie of de levering geschikt is voor verwerking richting Operationele Data-laag$$),
($$levering_reden_niet_verwerken$$, $$ODLLF007$$, $$007$$, $$ODLLF$$, $$$$, $$alfanumeriek$$, $$$$, $$$$, $$text$$, $$$$, $$$$, $$$$, $$$$, $$7$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Omschrijving waarom de levering niet geschikt is voor verwerking richting Operationele Data-laag$$),
($$levering_verwerkingsdatum$$, $$ODLLF008$$, $$008$$, $$ODLLF$$, $$$$, $$datum$$, $$$$, $$$$, $$date$$, $$$$, $$$$, $$$$, $$$$, $$8$$, $$1$$, $$1$$, $$9999-12-31$$, $$$$, $$$$, $$$$, $$2023-05-10$$, $$9999-12-31$$, $$Datum waarop de levering is verwerkt in de Operationele Data-laag$$),
($$levering_aantal_records$$, $$ODLLF009$$, $$009$$, $$ODLLF$$, $$$$, $$integer$$, $$$$, $$$$, $$integer$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$9$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Het aantal records dat is geleverd in het bronbestand en is verwerkt in de Operationela Data-laag$$),
($$config_file$$, $$ODLLF010$$, $$010$$, $$ODLLF$$, $$$$, $$text$$, $$$$, $$$$, $$text$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$10$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$De configfile die is gebruikt bij het inlezen van deze levering$$),
($$data_filenaam$$, $$ODLLF011$$, $$011$$, $$ODLLF$$, $$$$, $$text$$, $$$$, $$$$, $$text$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$11$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Naam van de datafile van deze levering$$),
($$sysdatum$$, $$ODLLF012$$, $$012$$, $$ODLLF$$, $$$$, $$datum$$, $$$$, $$$$, $$timestamp$$, $$$$, $$NOT NULL$$, $$$$, $$$$, $$12$$, $$1$$, $$1$$, $$$$, $$$$, $$$$, $$9999-12-31$$, $$2023-05-10$$, $$9999-12-31$$, $$Datum inlezen data$$);

CREATE TABLE didotest.dido_test_dido_import_levering_feit_data
(
   levering_rapportageperiode_volgnummer integer NOT NULL,
   levering_rapportageperiode text NOT NULL,
   code_bronbestand text NOT NULL,
   created_by text,
   levering_leveringsdatum date NOT NULL,
   levering_goed_voor_verwerking boolean NOT NULL,
   levering_reden_niet_verwerken text,
   levering_verwerkingsdatum date,
   levering_aantal_records integer NOT NULL,
   config_file text NOT NULL,
   data_filenaam text NOT NULL,
   sysdatum timestamp NOT NULL
);

COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_data.levering_rapportageperiode_volgnummer IS $$Uniek nummer voor dit record in de tabel$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_data.levering_rapportageperiode IS $$Unieke beschrijving van de boekings/rapportageperiode (WWYYYY(342023)/MMYYYY(112023)/QQYYYYY(022023)/YYYYYY(232023))$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_data.code_bronbestand IS $$Unieke code voor identificatie van bronbestanden (BGFMHLWD: Bezettinggraad data van FMH voor locatie Leeuwarden).$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_data.created_by IS $$Naam van de gebruiker die de levering aanmaakte$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_data.levering_leveringsdatum IS $$De datum waarop de levering is ontvangen in de staging area$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_data.levering_goed_voor_verwerking IS $$Indicatie of de levering geschikt is voor verwerking richting Operationele Data-laag$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_data.levering_reden_niet_verwerken IS $$Omschrijving waarom de levering niet geschikt is voor verwerking richting Operationele Data-laag$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_data.levering_verwerkingsdatum IS $$Datum waarop de levering is verwerkt in de Operationele Data-laag$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_data.levering_aantal_records IS $$Het aantal records dat is geleverd in het bronbestand en is verwerkt in de Operationela Data-laag$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_data.config_file IS $$De configfile die is gebruikt bij het inlezen van deze levering$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_data.data_filenaam IS $$Naam van de datafile van deze levering$$;
COMMENT ON COLUMN didotest.dido_test_dido_import_levering_feit_data.sysdatum IS $$Datum inlezen data$$;





COMMIT; -- Transaction
