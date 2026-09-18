# metricaudit

Indicator-level reconciliation of a classroom analytics reporting view against
the events it claims to summarise.

Quantitative claims about what happened in a lesson are usually read from a
pre-aggregated reporting layer: one tidy row per group, a column per indicator.
Validity work in this area asks at length whether such an indicator means what
it claims. It much more rarely asks whether the indicator counts what it claims.
The second question is the cheaper of the two to answer, it is answerable by
arithmetic, and a negative answer makes the first question moot.

This package answers it. For each column of a reporting view it recomputes the
figure from source events under a rule declared in advance, classifies the
column as reconciled, divergent or unverifiable, and where a divergence appears
it tests that divergence against a set of candidate mechanisms stated as
predictions rather than chosen afterwards.

The package is released so that the procedure can be inspected and re-run. The
session it was written for is a primary-age class, is held under restricted
custody, and is not part of this repository.

## The declared rule comes first

`src/metricaudit/rules.py` holds one counting rule per reported column, written
in the form a reader of the view would most naturally assume from the column's
name. This ordering is the point rather than a formality. Given enough freedom
some rule can nearly always be found that reproduces a reported column, so a
rule discovered after inspecting the data establishes nothing. Each rule is
therefore committed before the recomputation that uses it, the ruleset carries a
version, and the completion gate checks the commit ordering rather than trusting
it.

Where no path from source events to a column exists, the rule records that and
the column is classified unverifiable. A column that cannot be checked is a
finding in its own right, and forcing it into a comparison would hide that.

## What it computes

| Procedure | Question it asks |
|---|---|
| M1 | The declared counting rule for each reported column, versioned |
| M2 | Exact per-group comparison of reported against recomputed, classified three ways |
| M3 | Parity between the two stores, by Jaccard similarity and set difference on identifiers |
| M4 | Per-group ratio and absolute difference, with bias-corrected bootstrap intervals |
| M5 | Agreement visualisation: difference against mean on a logarithmic scale |
| M6 | Candidate mechanism predictions, with the observation matched to the nearest |
| M7 | Falsification: an observation matching no prediction is reported unattributed |
| M8 | Completeness and referential integrity, with exact binomial intervals |
| M9 | Group statistics recomputed under reported and under reconciled values |
| M10 | A representative association estimated once per value set |
| M11 | Reconciliation repeated under each alternative deduplication and rounding assumption |
| M12 | The largest divergence reported on its own and excluded from pooled summaries |

M12 is not a courtesy to the reader. A single counter that accumulates without
bound dominates any mean it enters, so pooling it would describe the counter
rather than the session.

## Mechanisms are predictions, not labels

`src/metricaudit/attribute.py` registers each candidate mechanism as a function
from source events to a predicted value per group. Attribution is then a
comparison rather than a judgement: the prediction either reproduces the
observation exactly or it does not, and a group matched by no candidate is
reported as unattributed. Tolerance is zero, because a mechanism that predicts
an observation to within a few units has been asserted rather than shown.

The registered candidates are cross-store summation, repeated client emission,
absent idempotency, gauge summation, gauge final value, and decision count.

Attribution says which mechanism reproduces an observation. It does not say what
the implicated field is, and the two are separate claims. A counter that
accumulates across a session and one that resets periodically both inflate a
total when summed, but they imply different magnitudes and different remedies,
so `gauge_profile` measures the shape of the field rather than assuming it,
reporting the runs, the highest value reached and the value left at the end.

## Two stores, no shared key

The platform writes each client event to an operational store and to a reporting
store. The two assign their own primary keys, so identifier-set comparison
between them is vacuous and parity has to be established on a composite natural
key within a time window. `src/metricaudit/parity.py` does both and reports
both, since the vacuity of the first is itself a property of the pipeline worth
recording.

## Custody

Text composed by participants never leaves the machine that holds the restricted
session. The payload extractor takes numeric fields by name and ignores the rest
of each object, so pupil writing is not read rather than being read and then
discarded.

What the package writes for release is the column map recording which fields the
analysis was permitted to read, a digest per analysed record, and a run manifest
recording code commit, interpreter, package versions, seed, tolerances and the
ruleset version.

`tests/test_release_guard.py` fails if any released artefact carries participant
text, if the manifest writer is handed a key outside the release policy, or if a
file this policy excludes has been staged into the repository.
`tests/test_synthetic_guard.py` fails if a run on synthetic data could overwrite
results computed from the session, or publish a provenance package describing
records that do not exist.

The session location is supplied through `METRICAUDIT_DATA_DIR` and is recorded
neither in the repository nor in the manifest.

## Running it

```bash
pip install -e ".[test]"
python scripts/make_synthetic.py     # writes synthetic/, already committed
python scripts/run_pipeline.py       # runs against synthetic/ when no session is set
pytest
```

With no `METRICAUDIT_DATA_DIR` set, the pipeline reads the synthetic session in
`synthetic/`. That session has the schema and shape of the real one and nothing
else in common: it is assembled by a seeded generator, and it is built to
exhibit a reconciliation fault so that the procedure demonstrably detects one,
with magnitudes chosen to differ from the observed ones. Every run against it is
stamped `source: synthetic`, it will not overwrite results computed from the
session, and it will not publish provenance.

Against the restricted session:

```bash
export METRICAUDIT_DATA_DIR=/path/to/session
python scripts/run_pipeline.py
python scripts/render_assets.py
```

## Layout

```
src/metricaudit/    rules, ingest, index, parity, reconcile, attribute, audit,
                    simulate, sweep, models, figures, tables, provenance, pipeline
scripts/            pipeline runner, asset renderer, synthetic session generator
synthetic/          the committed stand-in session
provenance/         column map, record digests, run manifest
tests/              analysis tests, release guard, synthetic guard
```

`outputs/` is written at run time and is excluded from the repository, because
results belong to the paper that reports them rather than to the archive of the
procedure that produced them.

## Licence

MIT. See `LICENSE`.
