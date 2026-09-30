# Dokumenti in priprava korpusa

## Uradni viri

`config/thesis-sources.json` vsebuje dva javna dokumenta, ki sta bila izbrana na
uradni strani pravilnikov UL FRI 10. 9. 2026. Surove kopije niso v repozitoriju.
Pridobite jih z `scripts/download_sources.ps1` ali ročno z URL-jev v konfiguraciji.
Pred raziskovalno uporabo preverite, ali so še veljavni in ali so objavljene nove
spremembe.

`config/thesis-web-sources.json` je dovoljen seznam uradnih spletnih strani. Ukaz
za refresh sprejme samo FRI HTTPS gostitelje, omeji velikost in preusmeritve ter
ne sledi odkritim povezavam samodejno:

```powershell
.venv/Scripts/python.exe -m src.knowledge_base refresh --manifest config/thesis-web-sources.json --output data/thesis/web/snapshot-v1 --allow-website-fetch --transport curl
```

Rezultat je kandidat za pregled. Ne objavi se samodejno v vektorski bazi.

## Lastni javni dokumenti

Datoteke shranite v `data/source_documents/` in vsako posebej navedite v kopiji
`config/evaluation-sources.json`. Zapis vsebuje institucijo, naslov, URL ali lokalno
pot, datum dostopa, vrsto dokumenta, opombo o pogojih uporabe in potrditev, da je
vir javen.

```powershell
.venv/Scripts/python.exe -m src.evaluation prepare-sources --catalog config/my-sources.json --output data/thesis/my-corpus
```

PDF mora imeti besedilno plast; OCR ni vključen. HTML potrebuje en element z
nastavljenim `selector_id`. TXT in Markdown morata biti UTF-8. `excluded_pages`
in `excluded_lines` omogočata odstranitev znanih naslovnic, kazal in ponavljajočih
se robnih vrstic. Ekstrakcijo in nekaj reprezentativnih strani vedno preglejte.

Ne vključujte zasebnih sporočil, osebnih podatkov, komunikacije brez soglasja ali
evalvacijskih vprašanj. Javna dostopnost dokumenta sama ne pomeni odprte licence.

## Razširjeni korpus pravilnikov

`config/fri-regulations.json` omogoči omejeno odkrivanje na uradni strani FRI.
Podrobnosti, omejitve in audit so v [navodilih za pravilnike](regulations.md).
Razvojni nabor 26 vprašanj ostaja vezan na starejši korpus dveh dokumentov;
lokalni ohranjeni posnetek je `data/thesis/development/corpus.json`.
