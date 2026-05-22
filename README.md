# MAG-Coder

MAG-Coder is a research prototype for generating Python interaction code from
Multi-Agent Graphs (MAGs), applying deterministic AST/framework refinement, and
evaluating the result with MAG-Bench contract, source-functional, and
agent-interaction-level ABMS checks.

This is a code-only release. It keeps the implementation, tests, provider
templates, small examples, and command-line entry points needed to rebuild and
evaluate MAG-Bench. Large generated benchmarks, frozen experiment bundles,
paper artifacts, claim artifacts, and downloaded official data are intentionally
excluded from Git.

## What This Repository Implements

- A MAG-Coder method pipeline with structured MAG input, graphical DOT+manifest
  input, graph+NL and graph-only modes, LLM provider generation, and
  AST/refinement.
- MAG-Bench construction from official HumanEval/MBPP source tasks, including
  template variant generation and deterministic filtering for the paper-scale
  about-2500-template reproduction.
- Evaluation for framework contract conformance, MAG-Bench-Func
  source-functional correctness, ABMS-level oracle checks, pass@k aggregation,
  and generation-time reporting.
- Ablation and baseline surfaces for zero-shot direct, few-shot direct,
  graph-only versus graph+NL, structured versus graphical, and
  full-with-refinement versus no-refinement runs.
- Provider registry templates and local fake-provider smoke tests.
- DOT/SVG/PNG rendering and a mock/deterministic VLM extraction interface.

## Implemented Paper Functionality

| paper functionality | status | primary entry point |
| --- | --- | --- |
| MAG-Coder full pipeline | implemented | `scripts/run_full_mag_bench_passk.py` |
| Official HumanEval/MBPP normalization | implemented | `scripts/download_official_source_tasks.py` |
| Paper-scale MAG-Bench builder | implemented | `scripts/build_full_mag_bench.py` |
| Package/source/adapter/ABMS validation | implemented | `scripts/validate_full_mag_bench*.py` |
| RQ2 graph-only / graph+NL ablation | implemented | `scripts/analyze_rq2_ablation.py` |
| RQ3 AST refinement ablation | implemented | `full_with_refinement` / `no_refinement` methods |
| MAG-Bench-Func view | implemented | `scripts/export_mag_bench_func_views.py` |
| pass@1/3/5/10/50/100 aggregation | implemented | `scripts/analyze_paper2500_passk.py` |
| Provider templates | interface-only | `configs/providers/*template*.yaml` |
| VLM/PNG extraction | interface-only | `scripts/extract_mag_from_rendered_graph.py` |

## What Is Intentionally Not Included

- Generated large benchmark outputs and case directories.
- `passk_runs`, `frozen_results`, paper artifacts, claim artifacts, and
  submission bundles.
- Official raw or normalized HumanEval/MBPP downloads.
- API keys, credentials, or provider secrets.
- Full 2500-case exhaustive real-model evaluation results.
- Multi-model real validation results.
- A completed arbitrary PNG/VLM recognition experiment.

Provider templates and VLM/PNG extraction are functionality surfaces. They are
not evidence of multi-model generality or arbitrary image recognition unless a
user explicitly configures and runs those experiments.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Quickstart

Run the lightweight release smoke. It uses local examples, the fake provider,
and the mock VLM extractor only.

```bash
.venv/bin/python scripts/run_code_submission_smoke.py \
  --output .release_audit/smoke
```

## Build MAG-Bench From Official HumanEval/MBPP

The downloader allowlists official benchmark source files and writes normalized
SourceTask JSONL. It does not download models and does not call an LLM.

```bash
.venv/bin/python scripts/download_official_source_tasks.py \
  --dataset all \
  --output-dir data/source_tasks/official_raw \
  --normalized-output-dir data/source_tasks \
  --overwrite
```

Downloaded files are ignored by default. Users who need reproducibility should
record their own download manifests.

## Build ~2500-template MAG-Bench

```bash
.venv/bin/python scripts/build_full_mag_bench.py \
  --source data/source_tasks/humaneval_official.jsonl \
  --source data/source_tasks/mbpp_official.jsonl \
  --output benchmarks/full_mag_bench_2500_v1 \
  --target-cases 2500 \
  --variants-per-task 5 \
  --variant-policy paper2500 \
  --formats structured,graphical \
  --split auto \
  --seed 2026 \
  --skip-unsupported \
  --fail-on-oracle-failure \
  --write-source-loader-report \
  --overwrite
```

The about-2500 benchmark is a deterministic reproduction of the construction
procedure from official HumanEval/MBPP tasks. It is not the original authors'
hidden template list.

## Validate MAG-Bench

```bash
.venv/bin/python scripts/validate_full_mag_bench.py \
  --bench benchmarks/full_mag_bench_2500_v1

.venv/bin/python scripts/validate_full_mag_bench_oracles.py \
  --bench benchmarks/full_mag_bench_2500_v1 \
  --output benchmarks/full_mag_bench_2500_v1/oracle_validation_report.json

.venv/bin/python scripts/validate_full_mag_bench_adapters.py \
  --bench benchmarks/full_mag_bench_2500_v1 \
  --output benchmarks/full_mag_bench_2500_v1/adapter_validation_report.json

.venv/bin/python scripts/validate_full_mag_bench_abms_oracles.py \
  --bench benchmarks/full_mag_bench_2500_v1 \
  --output benchmarks/full_mag_bench_2500_v1/abms_oracle_validation_report.json
```

## Run Fake Smoke

```bash
.venv/bin/python scripts/run_full_mag_bench_passk.py \
  --bench benchmarks/full_mag_bench_2500_v1 \
  --provider fake \
  --samples-per-case 1 \
  --modes structured_graph_nl,graphical_graph_nl \
  --methods full_with_refinement,no_refinement \
  --evaluation all \
  --refinement-strategy balanced \
  --output runs/fake_smoke \
  --max-cases 2 \
  --overwrite
```

## Run Local Real-model Evaluation

Real local runs require a model that already exists locally. The code does not
auto-pull `qwen3.5:9b` or any other model.

```bash
.venv/bin/python scripts/check_full_mag_bench_real_run_readiness.py \
  --bench benchmarks/full_mag_bench_2500_v1 \
  --provider ollama_native \
  --model qwen3.5:9b \
  --expected-cases 2500 \
  --planned-max-cases 20 \
  --planned-samples-per-case 1 \
  --output runs/readiness.json
```

Run the provider only if readiness reports `ready`.

```bash
.venv/bin/python scripts/run_full_mag_bench_passk.py \
  --bench benchmarks/full_mag_bench_2500_v1 \
  --provider ollama_native \
  --model qwen3.5:9b \
  --think false \
  --samples-per-case 1 \
  --modes structured_graph_nl,graphical_graph_nl \
  --methods full_with_refinement,no_refinement \
  --evaluation all \
  --refinement-strategy balanced \
  --output runs/local_real_pilot \
  --max-cases 20 \
  --overwrite
```

## Run pass@k

```bash
.venv/bin/python scripts/run_full_mag_bench_passk.py \
  --bench benchmarks/full_mag_bench_2500_v1 \
  --provider fake \
  --samples-per-case 5 \
  --modes structured_graph_nl,graphical_graph_nl \
  --methods full_with_refinement,no_refinement \
  --evaluation all \
  --refinement-strategy balanced \
  --output runs/passk_smoke \
  --max-cases 2 \
  --overwrite

.venv/bin/python scripts/analyze_paper2500_passk.py \
  --run runs/passk_smoke \
  --output runs/passk_analysis
```

pass@k supports `1`, `3`, `5`, `10`, `50`, and `100`. If k is larger than the
number of samples available for a group, the output is `N/A`; users must run
enough samples for pass@10/50/100 claims.

## Baselines And Ablations

Input modes:

- `structured_graph_nl`
- `structured_graph_only`
- `graphical_graph_nl`
- `graphical_graph_only`
- `zero_shot_direct`
- `few_shot_direct`
- `without_nl`

Methods:

- `full_with_refinement`
- `no_refinement`
- `mag_coder_full`
- `without_ast`

RQ2 analysis:

```bash
.venv/bin/python scripts/analyze_rq2_ablation.py \
  --run runs/rq2_ablation \
  --baseline-run runs/local_real_pilot \
  --output runs/rq2_analysis
```

## MAG-Bench-Func / Source-functional View

```bash
.venv/bin/python scripts/export_mag_bench_func_views.py \
  --bench benchmarks/full_mag_bench_2500_v1 \
  --run runs/local_real_pilot \
  --output runs/mag_bench_func
```

MAG-Bench-Func uses the source-functional metric and should be reported
separately from contract and ABMS conformance.

## Provider Registry

Provider templates live in `configs/providers/`. They are placeholders only and
do not contain credentials. Readiness checks report missing credentials without
reading or storing secret values and without calling external APIs.

## Graphical DOT+manifest And Optional VLM/PNG Interface

```bash
.venv/bin/python scripts/render_mag_graphics.py \
  --bench benchmarks/full_mag_bench_2500_v1 \
  --case-id example_case_id \
  --output runs/graphics_smoke

.venv/bin/python scripts/extract_mag_from_rendered_graph.py \
  --input runs/graphics_smoke/example_case_id.png \
  --manifest runs/graphics_smoke/example_case_id_manifest.json \
  --extractor mock \
  --output runs/graphics_smoke/recovered_mag.json
```

The mock extractor recovers MAG semantics from a sidecar manifest. This is an
interface and smoke-test route, not a completed arbitrary PNG/VLM recognition
experiment.

## Tests

```bash
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/mypy
```

Release-tree audit:

```bash
.venv/bin/python scripts/audit_release_tree.py \
  --repo-root . \
  --output .release_audit
```

## Claim Boundaries

- This repository provides code to reproduce the MAG-Coder pipeline and
  MAG-Bench construction.
- Generated large benchmark outputs and frozen experiment results are excluded
  from Git and should be regenerated by users when needed.
- Real-model runs require local model availability and are never auto-pulled.
- Multi-model provider templates are included, but multi-model real validation
  is not included.
- DOT+manifest graphical input is implemented; arbitrary PNG/VLM recognition is
  exposed as an optional/mock interface, not claimed as completed.
- Full 2500-case exhaustive real-model evaluation is not included in this Git
  release.
- Source-functional correctness, contract conformance, and ABMS conformance are
  separate metrics and should not be collapsed into one claim.
