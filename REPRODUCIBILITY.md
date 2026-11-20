# Reproduction evidence

Status on 2026-11-20 (updated when each check actually completes):

| Check | State | Scope |
|---|---|---|
| Original GPU experiment | Passed | Frozen historical results retained in `artifacts/input/expected/evaluation/` |
| New Windows Python 3.10 venv, cached evaluation | Passed | All original ranks/gates/assignments/weights and numerical results reproduced; no encoding in this check |
| Actual CPU feature regeneration, one recording | Passed | Twelve audio windows; max absolute difference from original GPU embeddings about 2.24e-7 |
| Portable implementation tests | Passed | 29 tests in the new venv; pip check reports no broken dependencies |
| Full new-venv CPU regeneration | Passed | All 439 recordings checked; 408 newly encoded, 31 original exclusions, zero reused dense features |
| Fresh hosted runner, complete source-to-results flow | Passed | [Run 44746495012](https://github.com/shusi1625/music-caption-gating/actions/runs/44746495012); 439 recordings downloaded, 408 newly encoded, five tables regenerated and independently compared |
| Colab | Not tested | No claim of execution |

The cached new-environment replay took 11.38 seconds on the original Windows host. It recalculated development settings without opening evaluation features during calibration and used the unchanged original settings for evaluation. This is isolated-environment evidence, not another-machine evidence.

The full CPU regeneration used the same isolated venv and **previously downloaded official audio/weights**, verified against original hashes and read without modifying them. It produced 4,896 audio-window embeddings and 498 text embeddings from scratch in 857.74 seconds after model setup. Recalibration replay, evaluation and independent checks then took 11.63 seconds. All 22,080 rank values, all gate decisions, caption assignments and bootstrap weights matched exactly; all metrics and confidence intervals passed the predeclared numeric comparison. No thresholds or weight were retuned. All five main paper tables were exported from the verified results.

Machine-readable reports, the exact resolved package versions, sanitized commands and exported tables are in [`evidence/windows-cpu-reproduction/`](evidence/windows-cpu-reproduction/). This run establishes full inference and analysis in a new environment on the **same physical Windows host**. It does not establish a new remote-machine run, a complete new download of all recordings, or Colab execution.

The original frozen threshold/weight files are never overwritten. Full regenerated comparisons require exact ranks, gates, assignments and bootstrap weights; absolute tolerances are declared in README and in the run specification before evaluation. Original negative/inconclusive hypotheses are retained.

## Completed fresh-machine reproduction

The public repository is [shusi1625/music-caption-gating](https://github.com/shusi1625/music-caption-gating), created on 2026-11-20. [Hosted run 44746495012](https://github.com/shusi1625/music-caption-gating/actions/runs/44746495012) completed successfully on a newly provisioned Ubuntu 22.04 CPU runner with Python 3.10.22. It tested commit [`c01402ab564ce8208193afd91772c6930986c433`](https://github.com/shusi1625/music-caption-gating/commit/c01402ab564ce8208193afd91772c6930986c433). No audio, model or feature cache was restored: all 439 official recordings were reacquired and verified, 408 tracks were newly encoded, and the same 31 duration exclusions remained. The run produced 4,896 audio-window embeddings and 498 text embeddings.

An additional checker inspected the downloaded artifact against the exact tested Git commit and the frozen original results. All 22,080 rank values, 49,440 gate values, caption assignments and 756,000 bootstrap-weight values matched exactly. Development-only settings were replayed; evaluation used the original frozen thresholds and weight. Metrics, confidence intervals and AUROC passed the predeclared tolerances, and all five tables matched at the manuscript's display precision. The largest absolute association-score difference was 3.1888485e-06, below 1e-5. The 29 implementation tests also passed.

| Measured stage | Seconds |
|---|---:|
| Official source audio download and verification | 626.06 |
| CPU feature regeneration after model setup | 691.62 |
| Analysis and verification | 2.90 |

These are measured stage times, not a total wall-time or performance guarantee. The frozen negative/inconclusive findings remain unchanged. Curated validation reports, resolved dependency versions, source metadata and table exports are committed in [`evidence/hosted-cpu-reproduction/`](evidence/hosted-cpu-reproduction/). GitHub's downloadable run artifact expires after 30 days; the committed small evidence files preserve the public record.

The tested scientific code is `c01402ab564ce8208193afd91772c6930986c433`. Subsequent documentation/evidence-only commits are outside the workflow's push-path filters and do not repeat inference. Historical Windows evidence retains its original execution context and then-current publication status. The hosted result demonstrates automated source-to-results reproduction on another machine; it does not establish independent human scientific replication, a listening audit, or Colab execution.

