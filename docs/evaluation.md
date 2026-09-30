# Evalvacija

Evalvacija preveri, ali je sistem našel pričakovane odlomke, navedel prave vire in
pripravil odgovor, primerljiv z referenco. Avtomatske metrike ne dokazujejo dejanske
pravilnosti.

## Razvojni nabor

`evaluation/controlled-silver-v1.json` vsebuje 26 sintetičnih primerov, izdelanih
iz javnih dokumentov: neposredna in parafrazirana vprašanja, drugačen besedni red,
vsakdanji izraz, več odlomkov, manjkajoč odgovor, vprašanje izven domene in potrebo
po pojasnilu. Vsi so razvojni in nepreverjeni. Ne predstavljajo komunikacije
resničnih študentov.

Vsak primer hrani vprašanje, referenčni odgovor, ID-je in kopije podpornih odlomkov,
kategorijo, težavnost, odgovorljivost, način nastanka in stanje pregleda. Nabor je
vezan na hash korpusa; sprememba vira zahteva ponovni pregled referenc.

## Človeški pregled referenc

```powershell
.venv/Scripts/python.exe -m src.evaluation review-template --dataset evaluation/controlled-silver-v1.json --corpus data/thesis/development/corpus.json --output data/reference-review.json
.venv/Scripts/python.exe -m src.evaluation apply-review --dataset evaluation/controlled-silver-v1.json --corpus data/thesis/development/corpus.json --reviews data/reference-review.json --output data/reviewed-questions.json
```

Pri vsakem primeru preverite razumljivost vprašanja, pravilnost odgovora, pravi vir,
zadostnost dokazov in odgovorljivost. Vnesite svojo oznako pregledovalca in stanje
`approved`, `rejected` ali `needs_revision`. Odobritev je vezana na hash vsebine.
Odobreni primeri tvorijo gold podnabor, nepregledani pa ostanejo silver.

## Zagon in rezultat

```powershell
.venv/Scripts/python.exe -m src.evaluation evaluate --dataset data/reviewed-questions.json --corpus data/thesis/development/corpus.json --reviewed-only --limit 26 --output data/evaluation/gold-offline
```

Vsak izhod vsebuje metapodatke, uporabljeni nabor, posamezne poskuse, zbirni JSON,
CSV, Markdown poročilo ter predlogo za ocenjevanje odgovorov. Rezultati so ločeni
po izvoru vprašanj, stanju pregleda in odgovorljivosti.

Retrieval metrike so Hit Rate/Recall@1, @3 in @5, MRR ter prisotnost pričakovanega
vira. Odgovorne metrike so normaliziran Exact Match, token F1, ROUGE-L, zaznavanje
neodgovorljivosti in pravilnost citiranih ID-jev. Ročna ocena odgovora od 1 do 5
pokriva pravilnost, relevantnost, popolnost, utemeljenost in jezikovno jasnost.

## Uradni FAQ in javni benchmarki

Uradni FAQ se uvozi kot ločen posnetek. Ker sta objavljeno vprašanje in odgovor
dostopna retrievalu, meri predvsem iskanje in reprodukcijo FAQ. Rezultati morajo
ostati ločeni od dokumentno izpeljanih vprašanj.

Adapter za slovenski SQuAD2 sprejme ročno preneseno datoteko in preverjen licenčni
zapis `config/public-qa-squad-sl-license.json`. QAslovene CSV je podprt šele, ko
uporabnik posebej preveri licenco izvorne zbirke. Ti rezultati merijo splošno QA
sposobnost in niso neposredna ocena uporabnosti za študentski referat.

Ročni pregled zmanjša pristranskost sintetičnih primerov, vendar je ne odpravi.
Brez resničnih uporabniških vprašanj ni mogoče meriti njihove dejanske porazdelitve
ali uporabniške izkušnje. RAGAs in embedding semantična podobnost nista vključena.

## Vprašanja referata

[Postopek uvoza in pregleda](student-office-evaluation.md) uporablja ločen izvor
`real_student_office`, izvirni odgovor ter pregledano referenco. Dejanskih parov
še ni. Vhod, osnutke in preglede hranite v `data/private/`, ne v javni zbirki.
