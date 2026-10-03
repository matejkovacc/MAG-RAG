# Semantični indeks in generirani odgovori

Način uporablja MongoDB, Qdrant in Azure. Namestite `requirements-live.txt`.
Za novo namestitev pripravite zasebno `.env` po `.env.example`; obstoječe datoteke
ne prepišite. Poverilnic ne vključujte v Git ali izpise.

## Podatkovne storitve

Nova lokalna namestitev uporablja `docker compose -p mag-rag up -d`, imensko
območje `mag_rag` in vrata iz `.env.example`. Pred zagonom preverite, ali na
izbranih vratih že tečejo podatkovne storitve. Obstoječi indeks uporabljajte prek
njegovih obstoječih povezav; nov projekt Compose ustvari druge volumne.

`THESIS_STORAGE_NAMESPACE` privzeto znaša `mag_rag`. Baza mora imeti to ime ali
predpono z dodatnim `_`; vektorska zbirka mora imeti predpono z `_`.
Za že obstoječ namenski indeks nastavite njegovo imensko območje,
`THESIS_MONGO_DB` in `THESIS_QDRANT_COLLECTION` v zasebni `.env`.
To izbere obstoječo shrambo; ničesar ne preimenuje, kopira ali indeksira.
Namenskega območja ne delite z drugimi aplikacijami.

## Priprava in objava

Najprej pripravite ter preglejte korpus po [navodilih za pravilnike](regulations.md).
Šele po izrecni odobritvi omejenega indeksiranja:

```powershell
.venv/Scripts/python.exe -m src.knowledge_base index --corpus data/thesis/regulations/next/corpus.json --allow-external-api
.venv/Scripts/python.exe -m src.knowledge_base verify --corpus data/thesis/regulations/next/corpus.json --output data/thesis/regulations/next/database-verification.json --allow-external-api
```

Spremenjeni ali manjkajoči odlomki potrebujejo nove embeddinge. Nespremenjeni
odlomki lahko ponovno uporabijo vektorje iz združljivega aktivnega posnetka.
Preverjanje `verify` ne kliče modela; `smoke` potrebuje embedding vsake poizvedbe.

## Zagon

Po odobritvi modelnih klicev za izbrano število vprašanj navedite korpus,
ki se ujema z dejansko objavljenim posnetkom:

```powershell
.venv/Scripts/python.exe -m src.rag --live --allow-external-api --max-questions 10 --corpus data/thesis/regulations/next/corpus.json --port 10131
```

Odprite `http://127.0.0.1:10131`. Zagon preveri fingerprint in vektorski profil
ter ob neujemanju zavrne delo. Ne indeksira samodejno. Posamezno vprašanje
potrebuje embedding in največ en generativni klic; neuspešnih klicev ne ponavlja.
Omejitve vprašanj ne obidite s ponovnim zagonom. `--max-questions` je omejitev
procesa, ne denarna omejitev naročnine. Azure prejme vprašanje in uporabljene odlomke.

Shranjeni auditi so dokazi prejšnjih zagonov. Kopiranje kode ali korpusa ne potrdi,
da je trenutna objava v podatkovni bazi ista; to preverite pred nadaljnjo evalvacijo.


## Spletne strani ob vsakem vprašanju

Za trenutne uradne strani dodajte `--current-website --allow-website-fetch`.
Seznam dovoljenih strani, tehnični preizkus in omejitve so opisani v
[sprotnem zajemu in inventarju PDF](current-website.md). Objavljeni korpus PDF
ostane izrecno izbran; spletni zajem ne osveži njegovih vektorjev.
