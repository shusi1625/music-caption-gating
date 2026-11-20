# Reproduction evidence

Status on 2026-11-20 (updated when each check actually completes):

| Check | State | Scope |
|---|---|---|
| Original GPU experiment | Passed | Frozen historical results retained in `artifacts/input/expected/evaluation/` |
| New Windows Python 3.10 venv, cached evaluation | Passed | All original ranks/gates/assignments/weights and numerical results reproduced; no encoding in this check |
| Actual CPU feature regeneration, one recording | Passed | Twelve audio windows; max absolute difference from original GPU embeddings about 2.24e-7 |
| Portable implementation tests | Passed | 29 tests in the new venv; pip check reports no broken dependencies |
| Full new-venv CPU regeneration | Passed | All 439 recordings checked; 408 newly encoded, 31 original exclusions, zero reused dense features |
| Fresh hosted runner, complete source-to-results flow | Pending | Workflow provided; no successful hosted run claimed yet |
| Colab | Not tested | No claim of execution |

The cached new-environment replay took 11.38 seconds on the original Windows host. It recalculated development settings without opening evaluation features during calibration and used the unchanged original settings for evaluation. This is isolated-environment evidence, not another-machine evidence.

The full CPU regeneration used the same isolated venv and **previously downloaded official audio/weights**, verified against original hashes and read without modifying them. It produced 4,896 audio-window embeddings and 498 text embeddings from scratch in 857.74 seconds after model setup. Recalibration replay, evaluation and independent checks then took 11.63 seconds. All 22,080 ranks, gate decisions, caption assignments and bootstrap weights matched exactly; all metrics and confidence intervals passed the predeclared numeric comparison. No thresholds or weight were retuned. All five main paper tables were exported from the verified results.

Machine-readable reports, the exact resolved package versions, sanitized commands and exported tables are in [`evidence/windows-cpu-reproduction/`](evidence/windows-cpu-reproduction/). This run establishes full inference and analysis in a new environment on the **same physical Windows host**. It does not establish a new remote-machine run, a complete new download of all recordings, or Colab execution.

The original frozen threshold/weight files are never overwritten. Full regenerated comparisons require exact ranks, gates, assignments and bootstrap weights; absolute tolerances are declared in README and in the run specification before evaluation. Original negative/inconclusive hypotheses are retained.

The public repository is [shusi1625/music-caption-gating](https://github.com/shusi1625/music-caption-gating), created on 2026-11-20. The first hosted source-to-results run is being validated. A successful run URL will be recorded after its completion; repository availability and workflow configuration alone do not count as a completed run.

