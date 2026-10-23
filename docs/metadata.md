# Plate metadata

CPTools joins image features to one metadata table for all plates of a run,
with one row per measured well. `cpt.io.read_harmony(..., schema=path)` reads
it from a delimited text file (comma, tab or semicolon; the delimiter is
detected).

## Required for import

| Column | Type | Meaning |
|---|---|---|
| `Batch` | text | Physical plate ID. Must match the `batch` passed to `read_harmony`. |
| `Row` | integer | One-based row: A = 1, B = 2, … |
| `Column` | integer | One-based column. |

`(Batch, Row, Column)` must be unique, and missing or non-numeric values in
these columns raise an error. The importer adds `Batch_row_col` and `Control`
(`Treatment == control_value`). Without a `Treatment` column, treatments are
taken from a `Drug_name` or `Drug_code` column if one exists. Wells without a
treatment are labeled `control_value`, with a warning.

These three columns are enough to join the files, not to analyze them: supply
the columns below as well.

## Recommended for analysis

| Column | Used by | Meaning |
|---|---|---|
| `Well` | observation names, plots | Well address, e.g. `H12`; keep it consistent with `Row`/`Column`. |
| `Treatment` | normalization, feature selection, hit calling, rescue | Compound or vehicle name, e.g. `DMSO`. |
| `line` | genotype-aware functions | The cells in this well, e.g. `WT`, `MUT1`. |
| `role` | reference and control selection | The purpose of a control well (below). |
| `Concentration_uM` | dose grouping | Numeric dose; blank when not applicable. |

The names are conventions. Functions take column names as arguments
(`batch_key`, `treatment_key`, `genotype_key`, `dose_key`, `scale_groupby`,
`replicate_groupby`) and well selections as `obs.query` strings, so other
names work as well.

## Control roles

`Treatment` alone cannot tell apart wells with the same contents but a
different purpose, e.g. WT + DMSO wells that serve as the normalization
reference and WT + DMSO wells placed among the screening wells. A `role` column
makes the selection explicit:

| `role` | Typical wells | Used as |
|---|---|---|
| `reference` | WT + vehicle in dedicated wells on every plate | `reference_query` of the normalization functions |
| `positive` | WT + a positive control at the same dose on every plate | `anchor_queries`, `positive_control_queries` |
| `vehicle` | each plate's genotype + vehicle, at scattered positions | genotype baseline, `disease_control_query`, pseudo-treatment nulls |
| `untreated` | cells without vehicle | tests whether the vehicle itself changes the profile |
| `screen` | genotype + compound | `treatment_query`, compound scores |

```python
reference = "line == 'WT' and role == 'reference'"
positives = {"positive_A": "role == 'positive' and Treatment == 'positive_A'"}
mutant_vehicle = "line == 'MUT1' and role == 'vehicle'"
```

## Genotype

Record the genotype of the cells in each well, not the plate's label: WT
reference wells on a mutant plate have `line == 'WT'`. Plates that mix
genotypes then need no special handling, and a query on `line` always selects
the intended cells.

## Doses

Store doses as numbers and keep the unit in the column name. A blank vehicle
concentration does not mean zero solvent; record the solvent percentage in its
own column if it varies. Do not replace unknown doses with zero. If doses or
exposure times vary, select matching reference conditions explicitly; dose
grouping does not match exposure times.

## Provenance and annotations

Record imaging day, cell lot, differentiation batch and independent run ID
whenever they vary, and make plate IDs unique across runs. Wells on one plate
are technical replicates; identify independent biological runs in their own
column. These columns feed batch diagnostics (`batch_similarity`,
`group_retrieval` with `relation_keys`). Compound annotations such as target,
pathway or MOA are optional; `replicate_map(annotation_key=...)` uses them.

## Example

| Batch | Well | Row | Column | line | role | Treatment | Concentration_uM |
|---|---|---|---|---|---|---|---|
| P01 | H12 | 8 | 12 | WT | reference | DMSO | |
| P01 | C12 | 3 | 12 | WT | positive | positive_A | 1 |
| P01 | D07 | 4 | 7 | MUT1 | vehicle | DMSO | |
| P01 | F15 | 6 | 15 | MUT1 | untreated | untreated | |
| P01 | B02 | 2 | 2 | MUT1 | screen | compound_001 | 10 |

Measurements do not belong in this table: `read_harmony` stores them in
`adata.layers["raw"]`.
