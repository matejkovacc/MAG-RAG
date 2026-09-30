# Arhitektura

MAG-RAG je samostojna aplikacija v Pythonu 3.12 z lokalnim spletnim vmesnikom.

## Pot podatkov

1. `src/knowledge_base` prebere izrecno navedene PDF, HTML, TXT ali Markdown vire.
2. Vsak dokument in odlomek dobi stabilen ID, hash različice in lokacijo v viru.
3. Offline način ustvari lokalne leksikalne vektorje ter uporabi Qdrant v pomnilniku
   in MongoDB testni nadomestek.
4. Live način po izrecni odobritvi uporabi Azure embeddinge, MongoDB in Qdrant.
5. `src/rag` pridobi največ pet odlomkov, generator pa sme citirati samo njihove ID-je.
6. Strežnik dopolni citate z zaupanja vrednimi naslovi, lokacijami, URL-ji in hashi.
7. `src/evaluation` zajame isto iskanje in odgovor ter izdela rezultate in poročila.

Offline priprava in prikaz ne ustvarita odjemalcev za Azure. Live način zahteva
konfiguracijo, `--allow-external-api` in vnaprej določen največji obseg vprašanj.

## Pomembne meje

- Evalvacijska vprašanja in pričakovani odgovori niso del knowledge base.
- MongoDB in Qdrant uporabljata skupne ID-je odlomkov in zamrznjene posnetke.
- Leksikalni in Azure vektorji imajo ločene zbirke.
- Spletna vsebina se zajame kot različica; med vprašanjem se ne izvaja spletno iskanje.
- Zgodovina pogovora je omejena in se ne shrani na strežniku.
- Veljaven citat dokazuje izvor odlomka, ne pravilnosti razlage.

Glavne vstopne točke so `python -m src.knowledge_base`, `python -m src.rag` in
`python -m src.evaluation`.
