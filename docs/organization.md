# Organizacija magistrskega projekta

## Dva repozitorija

- `MAG-RAG`: edina aktivna kopija aplikacije, testov in tehničnih navodil;
  `origin` mora kazati na `https://github.com/matejkovacc/MAG-RAG.git`.
- `mag-rag-thesis`: LaTeX, bibliografija in slike; osrednja datoteka je
  `thesis_template.tex`. Ta repozitorij ohrani svojo povezavo z Overleafom.

Mapi naj bosta sosednji. `mag-rag.code-workspace` odpre obe. Če imate dokument
drugje, prilagodite drugo pot v delovnem prostoru. Aplikacija od te mape ni odvisna.
Besedila naloge ne kopirajte v aplikacijski repozitorij in ga ne vzdržujte dvakrat.

## Razvoj in objava

Iz korena ustreznega repozitorija pred commitom preverite:

```powershell
git rev-parse --show-toplevel
git remote -v
git status --short
```

Pri aplikaciji uporabite njen `.venv` in teste v `tests/knowledge_base`.
Objavljajte izbrane izvorne datoteke, konfiguracijske predloge, javne razvojne
primere in tehnična navodila. `.env`, `.venv`, korpusi, zasebna komunikacija,
zbirke podatkov in začasni izpisi ostanejo lokalni. Za rezultate, ki jih citirate
v nalogi, pripravite ločen pregledan izvoz brez osebnih podatkov in z metapodatki
o commitu, korpusu, nastavitvah ter načinu ocenjevanja.

## Lokalna konsolidacija 30. 9. 2026

Manjkajoče funkcije za pravilnike, preverjanje posnetkov in uvoz vprašanj so
združene z obstoječim uporabniškim vmesnikom. Lokalni podatki so bili kopirani
brez spreminjanja zamrznjenih korpusov ali zgodovinskih rezultatov.

- `data/thesis/regulations/2026-09-23-ready/`: razširjeni kandidat in shranjeni izhodi;
- `data/thesis/development/corpus.json`: korpus za priloženi razvojni nabor;
- `data/thesis/corpus.json`: ohranjen prejšnji spletni korpus;
- `data/archive/consolidation-2026-09-30/`: varnostne kopije, seznam kopiranih
  artefaktov s hashi in ohranjen podvojen predlog dokumenta.

Ti lokalni arhivi niso vsebina GitHuba. Starih metapodatkov ne prepisujte zaradi
preimenovanja map: njihove poti so zgodovinski izvor, hashi pa vežejo reference
in objavljene posnetke. Prikazovanje pripravljenega korpusa ne bere izvornih
datotek s teh zgodovinskih poti. Za novo pripravo uporabljajte konfiguracije v
tem repozitoriju ter nove izhodne mape.

Obstoječi indeks lahko uporablja izrecno nastavljeno zasebno imensko območje.
Med konsolidacijo ni bilo izvedeno indeksiranje, selitev podatkovnih baz ali
modelno ocenjevanje. Live dostop potrebuje ločeno preverjanje po običajnem postopku.
