# Hosted CPU reproduction evidence

This folder preserves selected evidence from [GitHub Actions run 44746495012](https://github.com/shusi1625/music-caption-gating/actions/runs/44746495012).
The tested source commit is [`c01402ab564ce8208193afd91772c6930986c433`](https://github.com/shusi1625/music-caption-gating/commit/c01402ab564ce8208193afd91772c6930986c433).
It is separate from the earlier Windows-host virtual-environment reproduction.

The run and the downloaded artifact passed the declared checks before this folder
was created. The checks cover fresh official audio/model acquisition, 408 new
feature sets with the original 31 exclusions, development-only calibration replay,
unchanged frozen operating points, evaluation, independent arithmetic checks,
the original-result comparison, and all five main-table exports.
See `SUMMARY.json`, `hosted_validation.json`, `github_run.json` and `environment.json`.
In `reproduction_summary.json`, `new_audio_embedding_inference: false` and the `cache-based` scope describe only the analysis stage consuming the features freshly generated in the preceding stage, which is documented separately in `features/run_report.json`.
This is an **automated reproduction**, not independent human scientific replication,
listening validation, evidence about another model, or a Colab execution claim.

## Included and omitted

Selected numeric JSON/CSV outputs and source metadata preserve their original bytes.
`curation_manifest.json` records their SHA256 values and source artifact paths.
The raw GitHub run response is omitted: only allowlisted run identity, status,
tested commit and timestamps are retained. Actor, triggering-actor and
commit-author/committer objects are not published. Raw execution/system logs are
also omitted. The original requirements list is converted to package/version JSON.

No audio, model weights, tokenizer assets, dense embeddings, or new NPZ files are
included. Some evaluation JSON files reference NPZ hashes; the referenced arrays
are intentionally not copied into this selected evidence folder. The full numeric
Actions artifact and the repository's original scalar reference arrays are
separate resources. Missing arrays here are not a claim that they never existed.

## Caption metadata license and attribution

`features/manifest.json`, `features/exclusions.json` and `evaluation/cases.json`
include selected Song Describer captions and associated identifiers.
These dataset adaptations retain **Creative Commons Attribution-ShareAlike 4.0**:
[license](https://creativecommons.org/licenses/by-sa/4.0/),
[legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en).
Source: Ilaria Manco, Benno Weck, Dmitry Bogdanov, Philip Tovstogan and Minz Won,
with volunteer caption contributors, *The Song Describer Dataset: a Corpus of
Audio Captions for Music-and-Language Evaluation*, version 1.0.0,
[official record](https://doi.org/10.5281/zenodo.10072001).
Changes: a fixed subset was selected, assigned experimental roles/splits, and
reformatted as JSON with computed hashes, timings, provenance and ranks.
Caption strings were not paraphrased. Retain attribution, license links and
the description of changes; ShareAlike applies to adaptations. No creator
endorsement is implied. Original recordings keep their individual licenses.
See the repository's [third-party notices](https://github.com/shusi1625/music-caption-gating/blob/c01402ab564ce8208193afd91772c6930986c433/THIRD_PARTY_NOTICES.md).

The software license does not relicense the caption data, recordings, or model.
