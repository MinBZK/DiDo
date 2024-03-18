![images](images/DiDo_logo.png)

DiDo (Data in Data out) zet data in het datawarehouse en controleert deze op fouten. Eerst wordt een beschrijving van de data gebruikt om automatisch de tabellen in Postgres en beschrijvingen voor de wiki te genereren. Daarna kan de data worden aangeleverd, periodiek indien nodig.

Het doel van DiDo is de gebruiker zoveel mogelijk te ontzorgen, vandaar de luiaard als logo.

# Gebruik
Kloon deze repository in je `apps` folder en run de applicaties in vscode. Zet ook je PYTHONPATH naar het absolute pad van de helpers directory van dit projekt.
Zie voor meer uitleg de Wiki (werk in uitvoering).

# Bijzondere files

Sommige files staan in de .gitignore maar kun je zelf instellen.

- .pylintrc bevat installingen voor [pylint](https://pylint.pycqa.org/en/latest/user_guide/usage/run.html)
- pyproject.toml bevat instellingen voor [pytest](https://docs.pytest.org/en/stable/reference/customize.html)

# Environment

Vscode ondersteunt environments. Kies F1 > Create Environment > Venv. Dido is ontwikkeld met Python versie 3.9, hogere versie werken waarschijnlijk wel, lagere niet. De instellingen worden geplaatst in de file .venv, deze file zit in .gitignore en wordt niet meegenomen naar Github. De file `requirements.txt` bevat de pakketten die je wilt instellen voor je omgeving.


