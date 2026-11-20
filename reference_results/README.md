# Original reference tables

These CSV files are exports from the completed frozen evaluation:

- `retrieval.csv`: eight conditions at p=0 and p=0.25, with descriptive 95% CIs.
- `primary_effects.csv`: H1 and H2, with the fixed 97.5% primary intervals.
- `secondary_MRR_contrasts.csv`: twelve descriptive contrasts with 95% CIs.

The six machine-readable comparison artifacts are under
[`artifacts/input/expected/evaluation`](../artifacts/input/expected/evaluation).
Those are used by the reproduction script. Their ranks, scalar scores and
resampling weights do not include audio recordings or dense embeddings.

H1 is clean-caption rejection(first) minus rejection(mean12); H2 is
MRR(mean12 gate) minus MRR(first gate) at p=0.25. Fixed catalog, fixed development
settings and the designated twenty corruption seeds limit the interpretation.
These results measure source-track recovery, not listener recommendation quality
or independently established caption truth.
