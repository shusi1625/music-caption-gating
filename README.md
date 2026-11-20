# Caption gates for text-to-music retrieval

Reproduction package for an NLP HW2 study of **first-excerpt versus pooled-excerpt caption gates** with a frozen CLAP model. This package preserves the original negative and inconclusive findings. It does not select new settings on the evaluation set.

Release status: the public repository is available, and full source-to-results reproduction passed on a fresh Ubuntu 22.04 CPU runner. [Hosted run 44746495012](https://github.com/shusi1625/music-caption-gating/actions/runs/44746495012) tested commit [`c01402a`](https://github.com/shusi1625/music-caption-gating/commit/c01402ab564ce8208193afd91772c6930986c433); downloaded evidence was independently checked against the frozen results. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for scope and evidence.

## Experiment and expected result

From Song Describer 1.0.0, 439 planned recordings were acquired; 31 recordings unable to provide twelve 10-second windows under the fixed boundary rule were excluded. The retained sets contain 99 development tracks (21 queries) and 309 evaluation tracks (69 queries). Artists do not cross the development/evaluation split.

The frozen model is `laion/clap-htsat-unfused`, revision `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`. Development selects caption weight **0.25** and gate thresholds **0.32523579597473146** (first window) and **0.47426729202270507** (normalized mean of twelve windows). Evaluation uses 0% and 25% caption reassignment, seeds 0–19, eight conditions, and 2,000 artist-cluster bootstrap draws.

| Primary effect | Estimate | 97.5% percentile interval | Interpretation |
|---|---:|---:|---|
| H1: source-caption rejection, first minus mean | -0.093851133 | [-0.150539476, -0.032933639] | Opposite the proposed direction |
| H2: MRR, mean minus first, at 25% reassignment | -0.009737531 | [-0.029924224, 0.000811578] | Improvement not established |

See `reference_results/` for readable tables and `artifacts/input/expected/evaluation/` for the original machine-readable outputs. Source identity is a retrieval label, not a complete musical relevance or caption-factuality judgment. Confidence intervals condition on the fixed catalog, development settings and twenty reassignment seeds.

## Fresh-environment reproduction from official sources

Use Python **3.10**, Git, an internet connection, and a CPU or compatible GPU. The reference workflow uses CPU execution on Ubuntu 22.04. No paid API, private dataset login or training is required. Allow at least **10 GiB of free disk** for packages, audio, weights and temporary files; the original downloaded audio alone is approximately 1.93 GiB. Runtime depends on the machine and network. The verified hosted run took 626.06 seconds for audio acquisition, 691.62 seconds for feature regeneration after model setup, and 2.90 seconds for analysis and verification. These measurements are not runtime guarantees for other machines or networks.

```bash
git clone https://github.com/shusi1625/music-caption-gating.git
cd music-caption-gating
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir -r requirements-source.txt
python -m pip install --no-cache-dir --no-deps torch==2.0.1+cpu torchaudio==2.0.2+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip check
python -m unittest discover -s tests -v

python scripts/download_audio.py --data-dir artifacts/input --audio-dir outputs/source/audio
python scripts/prepare_features.py --data-dir artifacts/input --audio-dir outputs/source/audio --feature-dir outputs/source/features --weights-dir outputs/source/weights --device cpu
python scripts/reproduce.py --data-dir artifacts/input --output-dir outputs/reproduction --regenerated-features outputs/source/features
python scripts/export_paper_tables.py --data-dir artifacts/input --result-dir outputs/reproduction --output-dir outputs/reproduction/paper_tables
```

On Windows, create the environment with `py -3.10 -m venv .venv` and use `.\.venv\Scripts\python.exe` instead of `python`; shell activation is optional. On Linux, a missing libsndfile error can be resolved with the distribution's `libsndfile1` package. The pinned soundfile wheel normally provides a supported decoder. GPU execution requires compatible PyTorch/torchaudio CUDA builds and `--device cuda`; a fresh CPU reproduction does not establish Colab GPU availability or compatibility.

Install ordinary dependencies from PyPI before the two CPU wheels; this avoids a typing-extensions metadata issue observed with the PyTorch index and pip 23.0.1. Do not run with `python -O` or `PYTHONOPTIMIZE`: the integrity and scientific checks deliberately use assertions.

The downloader obtains bounded HTTP ranges from the official archive and verifies original SHA256, size, filename and ZIP CRC. It does not download a substitute recording. The model downloader pins its revision and checks the model/configuration SHA256 values **before loading the weights**. Feature extraction repeats the original mono/48 kHz conversion, twelve consecutive ten-second windows, caption roles and duration exclusions. No original dense feature cache is needed for this route.

Rerun the same commands after an interrupted download or extraction: valid files are reused. A partial feature run has no completed manifest and cannot be evaluated. To repeat the evaluation after changing code, use a new `--output-dir`; the existing run specification is locked to its implementation hashes. Do not relax comparison thresholds merely to obtain a green run.

## Success criteria and numerical policy

Success requires a zero process exit code and `status: passed` in **all** of:

- `outputs/reproduction/development_match.json`
- `outputs/reproduction/development_verification.json`
- `outputs/reproduction/evaluation/independent_verification.json`
- `outputs/reproduction/reference_comparison.json`
- `outputs/reproduction/reproduction_summary.json`

The pipeline independently reconstructs development quantiles/lambda and evaluation similarities, TF–IDF, assignments, gates, ranks and bootstrap intervals. **All 22,080 rank values, all 49,440 gate values, caption assignments, 756,000 bootstrap-weight values, and the selected lambda must match exactly.** The original thresholds are always used for evaluation. Before the full CPU regeneration, cross-platform absolute tolerances were fixed at `1e-5` for feature-based scores/recomputed thresholds, `2e-14` for reported metrics, and `2e-15` for AUROC. A development discrepancy does not authorize overwriting frozen settings.

Different machine/code provenance and NPZ bytes are expected when regenerating embeddings. The original JSON manifests retain their historical hashes; the new run has its own specification, feature hashes and environment record. A failed rank, gate or metric check is a reproducibility failure to investigate, not an accepted alternative result.

Original private-cache evaluation is a separate, faster check. If the author already has the historical 408 NPZ files under `artifacts/input/cache/clap/`, install only `requirements-eval.txt` and omit `--regenerated-features`. These private files are not in Git. This mode does not reproduce audio encoding and is not the public fresh-machine route.

## Reproduce each main paper table

The final export command above produces all five tables from the completed and verified run. Add `--table N` to produce only table N; it does not rerun inference or change any result.

| Paper table | CSV under `outputs/reproduction/paper_tables/` | Content |
|---|---|---|
| 1 | `table1_dataset_counts.csv` | Planned/retained/excluded tracks, artists and queries |
| 2 | `table2_retrieval.csv` | Both reassignment rates, eight conditions, MRR and R@1 |
| 3 | `table3_primary_effects.csv` | H1/H2 estimates and 97.5% intervals |
| 4 | `table4_verifier_statistics.csv` | Source rejection, reassigned-caption acceptance and AUROC |
| 5 | `table5_clean_case_ranks.csv` | Six preselected source-track ranks |

The exporter reads computed results, checks their artifact hashes, and refuses unfinished or unverified runs. It does not copy rounded table entries from the manuscript. Randomness is fixed by reassignment seeds 0–19, matched-random-gate seed `10000+s`, and artist-bootstrap seed `20261120` with 2,000 draws. Feature extraction uses torch/NumPy seed `20261021`.

## Automated reproduction and evidence

`.github/workflows/reproduce.yml` provisions a fresh hosted CPU runner, installs dependencies, tests the implementation, downloads official audio and weights, creates features, and recomputes development settings and all evaluation outputs. It has no restored audio/model/feature cache. Only numeric outputs and execution metadata are uploaded; source audio, model weights and dense embeddings are excluded.

Validation reports and exported tables are retained in [`evidence/hosted-cpu-reproduction/`](evidence/hosted-cpu-reproduction/); the successful [hosted run](https://github.com/shusi1625/music-caption-gating/actions/runs/44746495012) tested `c01402ab564ce8208193afd91772c6930986c433`. GitHub run artifacts expire after 30 days, so the small public evidence files are also committed here. Workflow configuration alone is not evidence of success. The workflow can be run from GitHub's **Actions → Reproduce from source → Run workflow**; pushes affecting implementation, requirements or workflow files also run it. Later documentation/evidence-only commits do not rerun inference and do not change which commit was tested.

## Layout

- `scripts/download_audio.py`: resumable, verified source acquisition.
- `scripts/prepare_features.py`: pinned CLAP extraction; separate regenerated provenance.
- `scripts/reproduce.py`: development replay, evaluation, independent checks and original-result comparison.
- `scripts/verify_development.py`, `scripts/verify_evaluation.py`: arithmetic checks that do not import the evaluated implementation or scikit-learn.
- `tests/`: calibration, permutation, ties, gates, bootstrap, path safety, source validation and download integrity tests.
- `artifacts/input/`: immutable scientific inputs, metadata and original numeric comparison results.
- `reference_results/`: table exports; no audio or dense embeddings.

## Licenses and research limits

Project software is MIT. Adapted Song Describer captions/metadata retain **CC BY-SA 4.0**. Recordings retain their individual source licenses, and the pinned CLAP model card declares Apache-2.0. Read [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`DATA_LICENSE.md`](DATA_LICENSE.md). Audio, model weights and dense embeddings are deliberately not redistributed. Do not upload `outputs/source/` as a public artifact.

This is one frozen encoder, one English dataset and a small fixed split. No human listening audit or training-overlap audit is claimed. The score geometry does not prove a causal mechanism. Code and arithmetic checks were developed with substantial OpenAI Codex assistance; they are not independent human scientific replication. The paper separately discloses writing assistance and author-review requirements.

## References

- Manco et al. (2023), *The Song Describer Dataset: a Corpus of Audio Captions for Music-and-Language Evaluation*. [Paper](https://arxiv.org/abs/2311.10057), [dataset 1.0.0](https://doi.org/10.5281/zenodo.10072001).
- Wu et al. (2023), *Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation*. [ICASSP paper](https://doi.org/10.1109/ICASSP49357.2023.10095969), [pinned checkpoint](https://huggingface.co/laion/clap-htsat-unfused/tree/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a).

