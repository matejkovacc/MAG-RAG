# MAG-RAG

Prototip magistrske naloge za podporo študentskemu referatu UL FRI. Sistem pripravi
sledljiv korpus javnih dokumentov, poišče odlomke in sestavi odgovor z navedbo virov.

**To je glavni repozitorij aplikacije.** Besedilo naloge, bibliografija in slike so
v ločenem repozitoriju [mag-rag-thesis](https://github.com/matejkovacc/mag-rag-thesis).
Za razvoj odprite to mapo; za skupno delo s kodo in besedilom odprite
`mag-rag.code-workspace`. Podrobnosti so v [organizaciji projekta](docs/organization.md).

## Kaj je vključeno

- priprava PDF, HTML, TXT in Markdown z izvorom, različico, stranjo in členom;
- odkrivanje pravilnikov na uradni strani FRI in shranjevanje nespremenljivih zajemov;
- MongoDB/Qdrant, objava posnetkov, ponovna uporaba nespremenjenih vektorjev in audit;
- lokalni spletni vmesnik, citati in nadaljnja vprašanja z omejeno zgodovino;
- lokalni leksikalni način in izbirni Azure način po izrecni odobritvi;
- evalvacija s sledljivimi referencami, človeškim pregledom in ločenimi podatkovnimi izvori;
- uvoz anonimiziranih parov vprašanj in odgovorov referata, ko bodo pridobljeni.

Priloženih je 26 **sintetičnih, nepregledanih razvojnih primerov**. Zbirka resničnih
vprašanj še ni pridobljena. Implementiran uvoz ne pomeni opravljene evalvacije.

## Namestitev

Python 3.12; ukaze izvajajte iz korena tega repozitorija:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Za teste namestite `requirements-dev.txt`; za Azure `requirements-live.txt`.
Vsaka kopija projekta uporablja svoje okolje `.venv`.

## Lokalni demo brez modelnih klicev

```powershell
.venv/Scripts/python.exe -m src.evaluation prepare-sources --catalog config/demo-sources.json --output data/demo-corpus
.venv/Scripts/python.exe -m src.rag --corpus data/demo-corpus/corpus.json --port 10130
```

Odprite `http://127.0.0.1:10130`. Demo uporablja izmišljeno tehnično gradivo iz
`examples/`, leksikalne vektorje in odgovore iz citiranih odlomkov. Ne meri kakovosti
modela ali poznavanja pravilnikov. Izhodna mapa za pripravo mora biti nova.

## Uradni korpus in semantični način

[Priprava podatkov](docs/data.md) opisuje javne vire. Razširjeno odkrivanje,
pripravo in preverjanje pravilnikov opisuje [postopek za pravilnike](docs/regulations.md).
Lokalni shranjeni kandidat iz 23. 9. 2026 vsebuje 22 dokumentov in 903 odlomke;
ni v Git in njegova vsebinska veljavnost še zahteva pregled. Če ga imate lokalno:

```powershell
.venv/Scripts/python.exe -m src.rag --corpus data/thesis/regulations/2026-09-23-ready/corpus.json --port 10130
```

Za semantično iskanje in generiranje glejte [live način](docs/live-mode.md).
Zagon nikoli samodejno ne indeksira. Modelni klici potrebujejo izrecno odobritev
in ustrezno zastavico; urejanje kode ali izvajanje testov je ne nadomešča.

## Evalvacija

Razvojni nabor v `evaluation/` je vezan na **starejši korpus dveh pravilnikov**,
ne na razširjeni korpus 22 dokumentov. Obstoječa lokalna kopija tega zamrznjenega
korpusa je `data/thesis/development/corpus.json`. Nove priprave ne predstavljajte
kot istega posnetka brez preverjanja hasha.

```powershell
.venv/Scripts/python.exe -m src.evaluation validate-set --dataset evaluation/controlled-silver-v1.json --corpus data/thesis/development/corpus.json
```

[Evalvacijski vodič](docs/evaluation.md) opisuje metrike in človeški pregled.
[Uvoz vprašanj referata](docs/student-office-evaluation.md) ohrani izvirni odgovor
ločeno od referenčne anotacije. Zasebne datoteke hranite pod `data/private/`, zunaj
Gita in zunaj indeksa. Resničnih poizvedb ne nadomeščajte s sintetičnimi brez jasne oznake.

## Testi

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv/Scripts/python.exe -m pytest tests/knowledge_base -q
node --test tests/knowledge_base/chat_ui.test.cjs
```

Node je potreben samo za teste uporabniškega vmesnika. Testi uporabljajo testne
ponudnike; niso plačana modelna evalvacija. Preizkusi javnih lokalnih dokumentov
se preskočijo, kadar ti dokumenti niso na voljo. CI izvede Python in UI teste.

## Zgradba

| Pot | Namen |
| --- | --- |
| `src/knowledge_base/` | priprava, odkrivanje virov, shranjevanje, iskanje in audit |
| `src/rag/` | odgovori, citati, pogovor in spletni vmesnik |
| `src/evaluation/` | validacija, pregled referenc, meritve in poročila |
| `config/` | izrecni javni viri in konfiguracije |
| `evaluation/` | objavljivi razvojni primeri in predloge |
| `tests/` | testi programske opreme |
| `docs/` | arhitektura, podatki, evalvacija in način uporabe |
| `data/` | lokalni korpusi, zajemi, rezultati in arhivi; izključeno iz Gita |

## Omejitve

Pravilni ID citata ne dokazuje pravilnosti razlage ali veljavnosti pravila.
Samodejne metrike prekrivanja besedila niso človeška presoja ali rezultati RAGAs.
Ni še neodvisne končne evalvacije z resničnimi vprašanji referata.
Izvor kode in status gradiva sta opisana v [ATTRIBUTION.md](ATTRIBUTION.md).
