# Third-party notices

Checked against the official sources on 20 November 2026. The repository's
software license does not replace the licenses of third-party text, audio or
model resources.

## Song Describer captions and dataset metadata

This research uses **The Song Describer Dataset: a Corpus of Audio Captions for
Music-and-Language Evaluation**, version 1.0.0, released by Ilaria Manco, Benno
Weck, Dmitry Bogdanov, Philip Tovstogan and Minz Won, with captions contributed
by the dataset's volunteer annotators.

- Official dataset: <https://doi.org/10.5281/zenodo.10072001>
- Official repository: <https://github.com/mulab-mir/song-describer-dataset>
- Dataset license: [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/)
- [License legal text](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en)

The official repository's License section states:

> The dataset is released under the CC BY-SA 4.0 license.

The upstream repository separately licenses its starter **software** under MIT.
That software license is not used here as permission to relicense the captions.

The caption-bearing JSON files in `artifacts/input/` are an adapted selection of
the dataset: valid captions were selected, associated with track and artist IDs,
assigned to fixed research splits, ordered by numeric caption ID, and placed in
JSON manifests and reference cases. This selection is not the original dataset
or its official benchmark split. Caption text is retained without paraphrasing.
These dataset adaptations retain **CC BY-SA 4.0**. Retain attribution, a license
link and the description of changes when redistributing them; adaptations must
comply with the license's ShareAlike terms. No dataset creator endorsement is
implied.

Please cite the dataset paper:

Ilaria Manco, Benno Weck, Seungheon Doh, Minz Won, Yixiao Zhang, Dmitry Bogdanov,
Yusong Wu, Ke Chen, Philip Tovstogan, Emmanouil Benetos, Elio Quinton, György
Fazekas, and Juhan Nam. 2023. *The Song Describer Dataset: a Corpus of Audio
Captions for Music-and-Language Evaluation*. Machine Learning for Audio Workshop
at NeurIPS 2023. <https://arxiv.org/abs/2311.10057>

## Source recordings: not redistributed

No audio recording is included in this repository. The official Song Describer
[datasheet, Distribution question 4](https://github.com/mulab-mir/song-describer-dataset/blob/main/docs/datasheet.md)
states that recordings retain their respective licenses. The dataset-wide
CC BY-SA 4.0 notice does not replace those individual terms.

The source archive's original per-recording attribution and license statements
are preserved in [`artifacts/source_audio_attribution.json`](artifacts/source_audio_attribution.json)
and in the download manifest. They identify the original title/artist, Jamendo
track URL, original file path and applicable license URL. The source of those
statements is [the official audio_licenses.txt](https://zenodo.org/records/10072001/files/audio_licenses.txt?download=1),
SHA256 `fe9efc05d513b3ddae00ebf0d96a93b5c189e5d7b3fd3ae69a217a6a2ce99aae`.
The file's MD5 agrees with the official Zenodo listing.

Among the 408 recordings retained after length checks, 140 have a source license
containing NoDerivatives, 313 contain NonCommercial, and 7 use License Art Libre.
These counts overlap and are not a partition. Licenses include multiple versions
and jurisdiction-specific ports; consult each preserved link rather than treating
them all as CC BY-SA 4.0. For example, [CC BY-NC-ND 3.0](https://creativecommons.org/licenses/by-nc-nd/3.0/)
permits sharing under its conditions but restricts commercial use and sharing
modified material.

The runtime downloader obtains recordings directly from the official archive
and verifies each original file's SHA256, ZIP CRC and size. Downloading does not
transfer ownership or remove the original conditions. Do not publish generated
audio, downloaded recordings or model-feature caches as CI artifacts.

## CLAP model: not redistributed

The feature extraction configuration identifies the Hugging Face model
[`laion/clap-htsat-unfused`](https://huggingface.co/laion/clap-htsat-unfused), pinned
to revision `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`.

The [model card at that exact revision](https://huggingface.co/laion/clap-htsat-unfused/blob/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/README.md)
declares **Apache-2.0**. Its raw README SHA256, checked during this audit, is
`3542c5c4ce25e7868f2c9c97c16229d876bd8d02049e1e0f4d7675da0e5b5db6`.
See the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
This is distinct from the CC0 license displayed by the original
[LAION-AI/CLAP software repository](https://github.com/LAION-AI/CLAP).
Neither declaration is treated as a license to relicense source recordings.

No model weights, tokenizer assets or third-party inference implementation are
vendored here. Runtime installation/download uses the relevant upstream package
and model terms. Cite Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor
Berg-Kirkpatrick, and Shlomo Dubnov. 2023. *Large-Scale Contrastive Language-Audio
Pretraining with Feature Fusion and Keyword-to-Caption Augmentation*. ICASSP.
<https://doi.org/10.1109/ICASSP49357.2023.10095969>

## Dense embeddings and numeric results are separate artifacts

Dense per-track audio/text embeddings are **not part of the public release**.
Local copies may be used for development but are ignored by Git. This release
does not assert that every feature vector is legally an adaptation of its source,
or that every vector is exempt from the source conditions. That determination
is not established by the resource documentation, so public redistribution of
the mixed-license feature cache is avoided.

Public numeric results contain ranks, scalar similarity/gate scores, caption
assignment IDs and bootstrap weights used to audit the reported experiment.
They are not audio waveforms or dense model embeddings. Caption strings included
in reference cases remain covered by the Song Describer notice above. Public
results do not grant additional rights to the recordings or model.
