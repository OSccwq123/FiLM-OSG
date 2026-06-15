# Release and Anonymous-Review Checklist

This checklist records the intended public/review boundary for the FiLM-OSG
reproducibility package.

## Public Review Snapshot

Include these files in the review snapshot or public repository:

- `film_osg/`, `train/`, `eval/`, and `profiling/`.
- `scripts/data_generation/` and the data sanity/plotting helpers referenced in
  `README.md`.
- Manuscript plotting scripts needed to regenerate public figures from completed
  evaluation summaries.
- `requirements.txt`, `docs/environment.md`, `data/README.md`, `NOTICE`,
  `docs/third_party_attribution.md`, and `THIRD_PARTY_LICENSES/`.

Do not include generated `.mat` data files, checkpoints, logs, private queue
scripts, one-shot full-experiment launchers, local figure caches, or manuscript
revision notes. Large data artifacts should be distributed separately through an
approved data host or regenerated from the included scripts when permitted.

## OSF / Zenodo Workflow

For anonymous peer review, prefer an anonymized OSF view-only snapshot. Prepare a
clean archive from the public-review files above, remove author-identifying local
paths or private notes, upload it to an OSF project, and enable an anonymized
view-only link. Use this OSF link in the submitted Data/code availability
statement if the journal review should remain anonymous.

For the final public release, tag the GitHub repository, upload large data files
or generated artifacts to OSF/Zenodo/GitHub Releases as appropriate, and mint a
Zenodo DOI only after the authorship and license decisions are final.

## Data/code Availability Draft

Code for the experiments will be available at the review repository or final DOI.
The repository contains model implementations, data-generation scripts, training
and evaluation entrypoints, random-seed conventions, environment notes, hardware
profiling scripts, and plotting/post-processing scripts. Large `.mat` benchmark
files are not stored in normal git; their expected names, shapes, checksums, and
regeneration or source notes are documented in `data/README.md`.
