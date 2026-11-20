# Data and artifact publication scope

The software license applies to this project's software. Third-party captions
and caption-derived metadata retain **CC BY-SA 4.0**, with attribution and changes
described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Original sound
recordings retain their individual licenses. Model weights are not included.

| Artifact | Public release | Source and conditions |
|---|---|---|
| Scripts, tests and experiment protocol | Yes | Project software license; source credits where applicable |
| Caption text, caption/track/artist IDs, split selections and caption-bearing JSON | Yes | Song Describer CC BY-SA 4.0; selected/reformatted subset |
| Original source attribution and license URLs | Yes | Preserved from Song Describer; no change to the sound recordings' licenses |
| Reference ranks, scalar scores, assignment IDs and bootstrap weights | Yes | Computed study outputs; captions in cases retain CC BY-SA 4.0 |
| Dense per-track audio/text NPZ embeddings | No | Excluded from this release; mixed source-license implications are not resolved here |
| MP3/WAV recordings and the source audio ZIP | No | Download directly from the official source when needed; individual terms apply |
| Model weights/tokenizers | No | Obtain pinned CLAP assets from the official Hugging Face repository |
| Personal paths, user email, raw local execution logs | No | Not required for scientific reproduction |

## Input contract

`artifacts/input/` preserves the fixed protocol, development settings and public
metadata. `planned_split.json` is placed at the input root; the historical
`split_file` field is intentionally not rewritten because its serialized hash
belongs to the frozen experiment. The portable loader performs the explicit
path remapping.

`artifacts/input/download_manifest.json` identifies all **439 planned recordings**
and records each original audio SHA256, size, ZIP CRC, archive offset, compressed
size, filename and original license notice. `active_after_length_check` marks the
408 tracks retained in the original experiment; the 31 exclusions are documented
separately. The downloader accesses only bounded ranges of the official archive
and never silently substitutes recordings or versions.

The authoritative dataset record is
[Song Describer 1.0.0](https://doi.org/10.5281/zenodo.10072001). The original
caption CSV, license list and audio metadata checksums are recorded in the
download manifest. Those checksum facts and existing split IDs are not a new
license grant. Caption text has not been paraphrased; JSON organization, subset
selection and experimental roles are this study's changes.

`artifacts/input/expected/evaluation/` contains the six original comparison
artifacts consumed by the portable reproduction script. Its `raw_arrays.npz`
contains ranks, verification scalars, gates, assignment masks and identifiers;
it contains **no per-track dense audio/text representation**. The separate
`bootstrap_weights.npz` contains only numeric resampling weights. The small CSV
tables in `reference_results/` are convenient exports of those results.

## Fresh-machine reproduction

A source-based run obtains the official recordings and pinned model at runtime,
creates a new feature directory, then reruns the fixed analysis. Keep the audio,
weights and dense features outside the tracked tree. An automated workflow may
upload numeric reports and suitable logs, but must not upload its downloaded
audio, model files or dense feature caches. Local exact-cache reproduction and
source-based feature regeneration are different checks and should be reported
separately; platform/decoder/library differences may affect regenerated features.

This avoids distributing the mixed-license audio-derived feature cache. It does
not warrant that any conceivable use of the source music is unrestricted. The
official source attribution and individual terms remain available for each
recording. A different future artifact release needs its own rights review;
labeling all feature vectors MIT, CC0 or CC BY-SA 4.0 would not resolve that issue.

## Licensing evidence

- [Official dataset README, License section](https://github.com/mulab-mir/song-describer-dataset#license): software MIT, dataset CC BY-SA 4.0.
- [Official datasheet, Distribution question 4](https://github.com/mulab-mir/song-describer-dataset/blob/main/docs/datasheet.md): recordings retain individual licenses.
- [CC BY-SA 4.0 legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en): attribution, notices, change identification and ShareAlike provisions.
- [Pinned CLAP model card](https://huggingface.co/laion/clap-htsat-unfused/blob/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/README.md): Apache-2.0 model-card declaration.

These are source checks and a conservative publication boundary, not a legal
determination of the copyright status of embeddings. No third-party endorsement
is implied.
