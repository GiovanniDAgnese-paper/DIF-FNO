# Archivio Zenodo di DIF-FNO

Per citare il software **DIF-FNO 1.2.0**, usa il DOI specifico
[`10.5281/zenodo.24406366`](https://doi.org/10.5281/zenodo.24406366).
Il [record Software](https://zenodo.org/records/24406366) è pubblico, con
licenza MIT, dal 20 novembre 2026. La citazione è in
[`CITATION.cff`](../CITATION.cff) e nel [README](../README.md#10-citation).

| Identificatore | Che cosa identifica |
| --- | --- |
| [`10.5281/zenodo.24406366`](https://doi.org/10.5281/zenodo.24406366) | Il deposito con lo ZIP di **DIF-FNO 1.2.0**; Zenodo lo mostra come `v9`. |
| [`10.5281/zenodo.22814077`](https://doi.org/10.5281/zenodo.22814077) | Il preprint DIF-FNO precedente, mostrato da Zenodo come `v8`. Non identifica lo ZIP di 1.2.0. |
| [`10.5281/zenodo.22071926`](https://doi.org/10.5281/zenodo.22071926) | Il DOI della serie Zenodo che include entrambi i depositi. Non identifica uno ZIP preciso. |

## Scaricare e verificare lo ZIP

Scarica `DIF-FNO-4e9159d68d248dac0de6de6f16f6f7620f173e5a.zip` dalla sezione
**Files** del [record Zenodo](https://zenodo.org/records/24406366). Il file è
l'archivio GitHub del
[commit `4e9159d`](https://github.com/GiovanniDAgnese-paper/DIF-FNO/tree/4e9159d68d248dac0de6de6f16f6f7620f173e5a).
Il commit include il codice scientifico del tag GitHub
[`v1.2.0`](https://github.com/GiovanniDAgnese-paper/DIF-FNO/tree/v1.2.0) e i
metadati preparati prima del deposito. Le correzioni bibliografiche successive
non cambiano lo ZIP già archiviato.

```bash
md5sum DIF-FNO-4e9159d68d248dac0de6de6f16f6f7620f173e5a.zip
# Atteso secondo Zenodo: ba769ded51a2b746ceae5f829a95671f
unzip -tq DIF-FNO-4e9159d68d248dac0de6de6f16f6f7620f173e5a.zip
```

Su macOS, usa `md5` al posto di `md5sum`.

## Contenuto e riproduzione

| Percorso | Contenuto |
| --- | --- |
| `src/`, `scripts/`, `tests/`, `examples/` | Implementazione Python, script, test ed esempio. |
| `data/` | Manifest e metadati dei dataset Darcy. Gli array `.pt`/`.npz` e i checkpoint non sono inclusi. |
| `results/v2/`, `results/study/`, `figures/` | Risultati del benchmark, verifica deterministica e figure. `results/repro/` conserva i risultati storici della linea v1. |
| `paper/` | Sorgenti LaTeX del manoscritto, bibliografia e tabelle, non il PDF del manoscritto. |

Dopo l'estrazione, dalla radice dello ZIP e con Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python examples/quickstart.py
python -m pytest
```

Usa `make data` per rigenerare i dataset canonici e `make stats paper-tables`
per ricostruire statistiche, tabelle e figure del benchmark dai risultati
inclusi. Per rilanciare il benchmark vedi la sezione 4 del [README](../README.md);
per compilare il manoscritto, vedi [`paper/README.md`](../paper/README.md).

## Versioni future

- Non ripubblicare lo stesso ZIP né spostare il tag `v1.2.0` per retroinserire
  il DOI. Per pubblicare file nuovi nella stessa serie Zenodo, parti dal
  [record Software](https://zenodo.org/records/24406366) e usa **New version**.
- Mantieni **Software** come tipo di risorsa e indica la versione del pacchetto
  nel campo **Version**. Se i file cambiano, usa il DOI specifico della nuova
  versione nelle relative citazioni, non quello di un deposito precedente.
- `.zenodo.json` è un file di metadati per l'integrazione GitHub, non la
  citazione del deposito pubblicato. Per una nuova release automatica,
  aggiornalo sul commit da taggare insieme alla versione del package. Non
  usare sia il deposito manuale sia l'integrazione GitHub per la stessa release.

Guide ufficiali: [gestione delle versioni](https://help.zenodo.org/docs/deposit/manage-versions/),
[metadati per l'integrazione GitHub](https://help.zenodo.org/docs/github/describe-software/zenodo-json/).
