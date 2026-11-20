# Pubblicare DIF-FNO 1.2.0 su Zenodo

**Pubblica un nuovo record di tipo Software.** Il DOI
[`10.5281/zenodo.22814077`](https://doi.org/10.5281/zenodo.22814077)
identifica il preprint precedente *DIF-FNO: Diffeomorphic Fourier Neural
Operator with D'Agnese Barrier Loss* (versione Zenodo `v8`, file
`dif-fno-v4.zip`). Non è il DOI del software 1.2.0: non riutilizzarlo e non
premere **New version** su quel preprint per depositare questo software. I due
record si collegano tramite **Related works**.

## 1. Prepara e controlla lo ZIP

Se hai già `DIF-FNO-1.2.0.zip` e `DIF-FNO-1.2.0.zip.sha256`, scaricali nella
stessa cartella e verifica il download con `sha256sum -c DIF-FNO-1.2.0.zip.sha256`
(su macOS: `shasum -a 256 -c DIF-FNO-1.2.0.zip.sha256`). Il file `.sha256` serve
per la verifica locale: non è necessario caricarlo.

Per ricreare il pacchetto, usa il commit del deposito che contiene questa guida
e `.zenodo.json`. Esegui questi comandi dalla radice del repository con quel
commit attivo (Git esporta solo i file già committati, non i file locali):

```bash
mkdir -p zenodo-upload
# Conserva questo SHA: è il commit esatto incluso nel file.
git rev-parse HEAD
git archive --format=zip --prefix=DIF-FNO-1.2.0/ \
  --output=zenodo-upload/DIF-FNO-1.2.0.zip HEAD
(
  cd zenodo-upload &&
  unzip -tq DIF-FNO-1.2.0.zip &&
  sha256sum DIF-FNO-1.2.0.zip > DIF-FNO-1.2.0.zip.sha256
)
```

Il tag GitHub [`v1.2.0`](https://github.com/GiovanniDAgnese-paper/DIF-FNO/tree/v1.2.0)
contiene il codice scientifico di riferimento. Lo ZIP del deposito aggiunge i
file di pubblicazione e corregge i riferimenti bibliografici; **non spostare il
tag** per includere queste modifiche. Il commit esatto è recuperabile anche dallo
ZIP, con `unzip -z DIF-FNO-1.2.0.zip`.

| Nello ZIP | Contenuto |
| --- | --- |
| `src/`, `scripts/`, `tests/`, `examples/` | Implementazione Python, script, test ed esempio. |
| `data/` | Manifest e metadati JSON dei dataset Darcy. Non contiene gli array `.pt`/`.npz` o i checkpoint. |
| `results/v2/`, `results/study/`, `figures/` | Risultati del benchmark, verifica deterministica e figure. `results/repro/` contiene i risultati storici della linea v1. |
| `paper/` | Sorgenti LaTeX del manoscritto, bibliografia e tabelle, non il PDF del manoscritto. |
| `LICENSE`, `CITATION.cff`, `.zenodo.json` | Licenza MIT, citazione del software e metadati per Zenodo. |

Per un controllo rapido dopo l'estrazione, usa un ambiente Python 3.10+ e, dalla
radice estratta:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python examples/quickstart.py
python -m pytest
```

Per rigenerare i dataset canonici usa `make data`; per rigenerare statistiche,
tabelle e figure dei benchmark dai risultati inclusi, usa `make stats paper-tables`.
Per rilanciare i benchmark, vedi la sezione 4 del [README](../README.md). Il PDF
è facoltativo per il deposito del software; per compilare il manoscritto, segui
le istruzioni in [`paper/README.md`](../paper/README.md).

## 2. Crea la bozza e compila i campi

1. Accedi a [Zenodo](https://zenodo.org/) e scegli **+ → New upload**.
2. Carica `DIF-FNO-1.2.0.zip`. Il modulo manuale non legge `.zenodo.json` dallo
   ZIP: copia i campi della tabella qui sotto.
3. In **Digital Object Identifier** rispondi **No** alla domanda se hai già un
   DOI per questo upload. Zenodo lo assegna alla pubblicazione. Se ti serve prima,
   puoi riservarlo con **Get a DOI now!** nella stessa bozza: il DOI sarà
   registrato solo quando la pubblichi.
4. Compila i campi. Verifica il nome; aggiungi ORCID e affiliazione solo se
   confermati. Usa la data effettiva del deposito per **Publication date**.

| Campo Zenodo | Valore |
| --- | --- |
| Resource type | Software |
| Title | The D'Agnese Chart Operator: Certified, Exactly Invertible, Adaptively Learned Charts for Neural Operators |
| Creators | Family name: `D'Agnese`; Given names: `Giovanni` |
| Version | `1.2.0` |
| License | MIT License (`MIT`); selezionala al posto della licenza predefinita, se presente |
| Visibility | Public (file aperti) |
| Language | English (`eng`) |
| Keywords and subjects | neural operator; fourier neural operator; diffeomorphism; certified invertibility; adaptive mesh; equidistribution; moving mesh; scientific machine learning; darcy flow; non-convex domains |
| Related works | **Is derived from** → DOI `10.5281/zenodo.22814077`, tipo **Preprint** |
| Repository URL (se disponibile) | `https://github.com/GiovanniDAgnese-paper/DIF-FNO` |
| Programming language (se disponibile) | Python |

**Description** (da copiare):

> DIF-FNO 1.2.0 implements the D'Agnese Chart Operator, a Fourier neural operator evaluated on an adaptively learned triangular chart of the reference square. The chart is globally injective and has a positive Jacobian determinant and a closed-form inverse by construction. An optional equidistribution loss encourages the mesh to concentrate cells where a field has sharper features.
>
> This software archive contains the Python source, tests, reproduction scripts, dataset generation metadata, benchmark results, figures, and LaTeX manuscript sources. The paired Darcy-flow benchmark covers six geometries and three seeds per configuration. Raw data arrays and trained checkpoints are not included; the generator for the canonical datasets is included. The archived code is based on the GitHub tag v1.2.0.

## 3. Controlla, pubblica e aggiorna la citazione

1. Salva e usa **Preview** per controllare file, titolo, autore, versione,
   licenza, accesso e collegamento al preprint. Controlla di avere il diritto di
   distribuire tutto il contenuto. Poi clicca **Publish** e conferma.
2. Apri il record pubblico, verifica che il file sia scaricabile e copia il
   **DOI specifico della versione 1.2.0**. Per citare esattamente lo ZIP usa
   quello; l'eventuale DOI di tutte le versioni identifica la serie, non lo ZIP.
3. Aggiorna `CITATION.cff` e il BibTeX/badge nel `README.md` sul repository con
   il nuovo DOI di versione, mantenendo il DOI del preprint come riferimento
   storico. Non occorre modificare i file già depositati solo per aggiungerci
   il DOI: la pagina di Zenodo riporta già la citazione.
4. In futuro, per depositare nuovi file mantenendo **la stessa serie software**,
   apri il nuovo record Software e usa **New version** da lì. Non pubblicare
   la stessa versione anche tramite l'integrazione automatica con GitHub:
   creeresti un doppione. `.zenodo.json` è pronto per l'integrazione GitHub,
   ma per eventuali future release automatiche va aggiornato in anticipo sul
   commit da taggare, insieme alla versione nel package e alla citazione. Zenodo
   usa `.zenodo.json` con precedenza su `CITATION.cff`.

Guide ufficiali: [nuovo deposito](https://help.zenodo.org/docs/deposit/create-new-upload/),
[gestione delle versioni](https://help.zenodo.org/docs/deposit/manage-versions/),
[metadati per l'integrazione GitHub](https://help.zenodo.org/docs/github/describe-software/zenodo-json/).
