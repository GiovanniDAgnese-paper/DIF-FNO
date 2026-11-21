# Public research records: Giovanni D'Agnese / DIF-FNO

Checked against public sources on 2026-11-20. This is a project-specific audit,
not a claim to have inventoried every mention of the author. A name match on
its own is not enough to identify the researcher.

## Which link to use

| Record | Use |
| --- | --- |
| [GitHub: GiovanniDAgnese-paper/DIF-FNO](https://github.com/GiovanniDAgnese-paper/DIF-FNO) | Current code and documentation. Use [`CITATION.cff`](../CITATION.cff) for the archived software citation. |
| [10.5281/zenodo.24406366](https://doi.org/10.5281/zenodo.24406366) | Canonical **DIF-FNO 1.2.0 software**: MIT, ZIP of Git commit `4e9159d`. Zenodo calls this deposit `v9`; that is not the package version. |
| [10.5281/zenodo.22814077](https://doi.org/10.5281/zenodo.22814077) and [10.5281/zenodo.22117134](https://doi.org/10.5281/zenodo.22117134) | Historical preprint records, not the current software or independent confirmation of the current benchmark. |
| [10.5281/zenodo.22071926](https://doi.org/10.5281/zenodo.22071926) | DOI for the Zenodo series, not a specific snapshot. |
| [Papers with Code entry](https://paperswithcode.co/paper/109793) and [author page](https://paperswithcode.co/authors/223948) | Third-party index of the August preprint (`22117134`). The source DOI is correct for that entry, but its abstract repeats the historical barrier and accuracy claims. Do not substitute the software DOI for the preprint's source DOI. |

No ORCID, affiliation, arXiv record, Google Scholar profile or external experiment
report was independently verified as belonging to this work in this audit. Do
not attach an identifier or attribute a profile on name similarity alone.

## Corrections for records outside GitHub

These are recommendations for the record owner or index maintainer; they do
not imply that any of these records have been edited. Keep the DOI and files
unchanged when correcting metadata.

- **Current Zenodo deposit:** the [DataCite metadata](https://api.datacite.org/dois/10.5281/zenodo.24406366)
  has `version: null`, a duplicate alternative title, and one subject containing
  a semicolon-delimited list. Set **Version** to `1.2.0`, delete the duplicate
  alternative title, and add the keywords as separate entries. Keep the type
  **Software**, the MIT license and the canonical GitHub URL. For precise
  provenance, add that the ZIP contains commit `4e9159d68d248dac0de6de6f16f6f7620f173e5a`.
- **Historical Zenodo records and Papers with Code:** retain the original
  source and add an update note where the host supports it. The older abstract
  describes a barrier as a guarantee and claims state-of-the-art accuracy;
  neither should be presented as the conclusion of the current benchmark.
  Suggested note:

  > Historical preprint describing an earlier DIF-FNO approach. See the
  > archived D'Agnese Chart Operator software, DIF-FNO 1.2.0
  > (https://doi.org/10.5281/zenodo.24406366), for the constructive chart and
  > the canonical Darcy benchmark. The historical soft penalty does not
  > structurally certify the map. The canonical chart-vs-FNO comparison did
  > not detect a statistically significant pooled accuracy difference.

- **GitHub About:** the public description still uses the older project framing.
  Set the website to `https://doi.org/10.5281/zenodo.24406366` and use, for example:
  “Giovanni D’Agnese’s DIF-FNO research software: PyTorch Fourier neural operators
  on certified, invertible charts, with reproducible Darcy-flow benchmarks.”
  The GitHub integration used for this audit could edit repository files but
  did not have permission to change these settings.
- **Search results:** some indexed snippets returned an older GitHub README.
  Treat the live default branch as authoritative; fixing the source does not
  update a search engine's cached snippet immediately.

Use the [announcement pack](ANNOUNCEMENT.md) for current public wording and
[the archive guide](ZENODO.md) for DOI and file verification.
