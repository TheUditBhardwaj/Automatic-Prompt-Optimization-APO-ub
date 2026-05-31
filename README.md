# Automated Prompt Optimization (APO)

A configurable, resumable system for automatically optimizing LLM prompts for structured PDF-to-JSON extraction tasks using the [ExtractBench](https://github.com/ContextualAI/extract-bench) benchmark.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the System](#running-the-system)
  - [1. Baseline Extraction](#1-baseline-extraction)
  - [2. Prompt Optimization](#2-prompt-optimization)
  - [3. Resuming an Interrupted Run](#3-resuming-an-interrupted-run)
- [Configuration Guide](#configuration-guide)
  - [Retargeting to a Different Dataset](#retargeting-to-a-different-dataset)
- [Dataset Split Policy](#dataset-split-policy)
- [Scoring Function](#scoring-function)
  - [Per-Field Evaluation Metrics](#per-field-evaluation-metrics)
  - [Array Alignment Policy](#array-alignment-policy)
- [Output Artifacts](#output-artifacts)
- [Project Structure](#project-structure)
- [Video Walkthrough](#video-walkthrough)
- [Constraints and Notes](#constraints-and-notes)

---

## Overview

This system implements an automated prompt optimization loop that:

1. Accepts a **seed prompt**, a **dataset** (PDFs + gold JSON annotations), a **scoring function**, one or more **LLMs**, and a **budget** (iteration count) as configuration inputs.
2. Runs a **greedy optimization process** that iteratively mutates the system instruction using LLM-generated error feedback.
3. **Terminates on budget exhaustion** and produces a final report with seed vs. final scores on a held-out test split, optimization trajectory, and prompt diffs.

All LLM calls, optimization decisions, and predictions are persisted to a SQLite database, enabling full resumability and auditability.

---

## Architecture

```
config/config.yaml
       |
       v
 Dataset Loader  -->  Deterministic Train / Val / Test Split
       |
       v
 Seed Prompt (schema-specific or config override)
       |
       v
 +--------------------------------------------+
 |       Greedy Optimization Loop              |
 |                                             |
 |  Step 0: Evaluate seed on validation set    |
 |  Step 1..N:                                 |
 |    1. Analyze errors from last evaluation   |
 |    2. Mutator LLM proposes revised prompt   |
 |    3. Evaluate mutated prompt on val set    |
 |    4. Accept if F1 improves, reject if not  |
 |    5. Persist step to SQLite                |
 +--------------------------------------------+
       |
       v
 Final: Evaluate seed + best prompt on test split
       |
       v
 Outputs: summary.json, score_curve.png, prompt_diff.txt, final_prompt.txt
```

**LLM Roles**:
- **Extractor** (e.g., `gemini-3.5-flash`, temperature 0.0): Reads the PDF/text and produces structured JSON output.
- **Mutator** (e.g., `gemini-2.5-flash`, temperature 0.7): Receives error feedback and proposes an improved system instruction.

---

## Prerequisites

- Python 3.10+
- A Google Gemini API key (one or more)
- The ExtractBench dataset (cloned or downloaded)

---

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd Automatic-Prompt-Optimization-APO-ub
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If no `requirements.txt` exists, install the following:

```bash
pip install google-genai pydantic pdfplumber PyYAML python-dotenv matplotlib PyPDF2
```

### 3. Configure API keys

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_primary_api_key_here

# Optional: additional keys for rotation (prevents rate limiting)
GEMINI_API_KEY_1=your_second_key
GEMINI_API_KEY_2=your_third_key
```

### 4. Prepare the dataset

Place the ExtractBench data under the `dataset/` directory following this structure:

```
dataset/
  hiring/
    resume/
      pdf+gold/
        candidate1.pdf
        candidate1.gold.json    (or candidate1.json)
        candidate2.pdf
        candidate2.gold.json
        ...
      resume-schema.json
  academic/
    research/
      pdf+gold/
        paper1.pdf
        paper1.gold.json
        ...
      research-schema.json
  sport/
    swimming/
      pdf+gold/
        ...
      swimming-schema.json
```

The system matches PDF files to gold annotations by filename (e.g., `doc1.pdf` pairs with `doc1.gold.json` or `doc1.json`).

---

## Running the System

All commands should be run from the project root directory.

### 1. Baseline Extraction

Runs the seed prompt on the validation split and reports per-document and aggregated scores:

```bash
python3 src/main.py --config config/config.yaml
```

### 2. Prompt Optimization

Runs the full optimization loop (baseline + N mutation iterations + held-out test evaluation):

```bash
python3 src/optimize_cli.py --config config/config.yaml
```

To use a different schema configuration:

```bash
# Academic/Research
python3 src/optimize_cli.py --config config/config_research.yaml

# Sport/Swimming
python3 src/optimize_cli.py --config config/config_swimming.yaml

# Finance/10-K/Q
python3 src/optimize_cli.py --config config/config_10kq.yaml
```

### 3. Resuming an Interrupted Run

If a run is interrupted (API failure, system crash, etc.), resume it without losing completed work:

```bash
python3 src/optimize_cli.py --config config/config.yaml --resume optimize_20260528_140635
```

Replace `optimize_20260528_140635` with the actual run ID (found in the output directory name or console logs). All completed steps and predictions are loaded from the SQLite database.

---

## Configuration Guide

All configuration is in YAML format. Example (`config/config.yaml`):

```yaml
# Model Settings
model:
  name: "gemini-3.5-flash"        # Extraction model
  temperature: 0.0                 # Deterministic extraction
  max_output_tokens: 16384         # Max JSON output length

# Dataset Settings
dataset:
  raw_dir: "dataset/hiring/resume/pdf+gold"   # Directory containing PDFs
  gold_dir: "dataset/hiring/resume/pdf+gold"  # Directory containing gold JSONs
  train_ratio: 0.0                             # Training split (unused currently)
  val_ratio: 0.7                               # Validation split for optimization
  test_ratio: 0.3                              # Held-out test split
  seed: 42                                     # Random seed for deterministic split

# Schema Settings
schema: "resume"                   # ExtractBench schema name

# Optimizer Settings
optimizer:
  iterations: 3                    # Budget: number of mutation attempts
  mutation_model: "gemini-2.5-flash"   # Model used for prompt mutation
  mutation_temperature: 0.7            # Higher temperature for creative mutations

# Seed prompt override (optional)
# seed_prompt: "Your custom seed prompt here..."

# Output settings
output_dir: "output"
```

### Retargeting to a Different Dataset

To run on a new, unseen dataset, **only configuration changes are needed** -- no code edits required:

1. **Place the data**: Put PDFs and gold JSON files in a directory (e.g., `dataset/legal/contracts/pdf+gold/`).

2. **Place the schema**: Put the JSON Schema file as `dataset/legal/contracts/contracts-schema.json` (naming convention: `<schema_name>-schema.json`).

3. **Create a config file** (`config/config_contracts.yaml`):

```yaml
model:
  name: "gemini-3.5-flash"
  temperature: 0.0
  max_output_tokens: 16384

dataset:
  raw_dir: "dataset/legal/contracts/pdf+gold"
  gold_dir: "dataset/legal/contracts/pdf+gold"
  train_ratio: 0.0
  val_ratio: 0.7
  test_ratio: 0.3
  seed: 42

schema: "contracts"

optimizer:
  iterations: 5
  mutation_model: "gemini-2.5-flash"
  mutation_temperature: 0.7

output_dir: "output_contracts"
```

4. **Run**:

```bash
python3 src/optimize_cli.py --config config/config_contracts.yaml
```

The system will automatically:
- Compile a Pydantic model from the JSON Schema (handling `$ref`, `anyOf`, `oneOf`).
- Generate a generic seed prompt for the "contracts" domain.
- Run the full optimization loop.

---

## Dataset Split Policy

The dataset split is **deterministic and reproducible**:

1. All PDF files in `raw_dir` are listed and sorted alphabetically.
2. The sorted list is shuffled using Python's `random.Random(seed)` with the configured seed (default: 42).
3. The shuffled list is divided by ratio:
   - First `train_ratio * N` items go to the training split.
   - Next `val_ratio * N` items go to the validation split.
   - Remaining items go to the test split.

The same seed always produces the same split. The split is performed using a local `Random` instance to avoid contaminating global state.

---

## Scoring Function

The scoring function is **independent of the optimization loop** and can be used standalone.

### Per-Field Evaluation Metrics

The scorer honors ExtractBench's per-field `evaluation_config` embedded in the JSON Schema:

| Metric ID | Description |
|-----------|-------------|
| `string_exact` | Exact string match |
| `string_case_insensitive` | Case-insensitive string match (default fallback) |
| `string_fuzzy` | SequenceMatcher ratio with configurable threshold |
| `string_semantic` | LLM-based semantic equivalence with Jaccard fallback; cached per (pred, gold) pair |
| `integer_exact` | Integer equality |
| `number_exact` | Float equality |
| `number_tolerance` | Absolute tolerance comparison |
| `boolean_exact` | Boolean equality with string coercion |

### Array Alignment Policy

For repeated arrays (e.g., work experience entries, publications), the scorer uses **greedy bipartite matching**:

1. A similarity matrix is computed between all predicted and gold items based on matching leaf fields.
2. Matches are sorted by similarity (descending) and greedily assigned.
3. Evaluation is performed on the aligned pairs.

This avoids O(N!) permutation matching while achieving near-optimal alignment in O(N^2).

### Aggregation

Scores are reported at three levels:
- **Per-document**: Precision, Recall, F1 over all flattened leaf fields.
- **Per-subtree**: Aggregated metrics for each top-level schema key across all documents.
- **Per-leaf**: Aggregated metrics for each individual leaf field path across all documents.

---

## Output Artifacts

Each optimization run produces the following in `<output_dir>/<run_id>/`:

| File | Description |
|------|-------------|
| `summary.json` | Machine-readable run summary with trajectory, test scores, and per-subtree breakdowns |
| `final_prompt.txt` | The optimized system instruction |
| `prompt_diff.txt` | Unified diff between seed and final prompt |
| `score_curve.png` | Validation F1 trajectory plot with accept/reject annotations |
| `config_run.yaml` | Copy of the configuration used for this run |
| `step_N/` | Per-step directories with individual prediction files |

Additionally, `apo_database.db` (SQLite) stores all runs, steps, predictions, and semantic cache persistently.

---

## Project Structure

```
.
|-- config/                     # YAML configuration files
|   |-- config.yaml             # Resume schema (default)
|   |-- config_research.yaml    # Academic/research schema
|   |-- config_swimming.yaml    # Sport/swimming schema
|   |-- config_10kq.yaml        # Finance/10-K/Q schema
|
|-- dataset/                    # ExtractBench data (PDFs, gold JSONs, schemas)
|   |-- academic/research/
|   |-- finance/10kq/
|   |-- hiring/resume/
|   |-- sport/swimming/
|
|-- src/                        # Source code
|   |-- config.py               # Configuration loader and validator
|   |-- dataset.py              # Dataset loading and deterministic splitting
|   |-- pdf_loader.py           # PDF text extraction (pdfplumber)
|   |-- schema.py               # JSON Schema to Pydantic model compiler
|   |-- prompts.py              # Seed prompts (schema-specific + generic fallback)
|   |-- llm_wrapper.py          # Gemini API client with multi-key rotation
|   |-- parser.py               # JSON parsing with hallucination detection
|   |-- evaluator.py            # Scoring function (per-field, per-subtree, global)
|   |-- mutator.py              # Error-driven prompt mutation via LLM
|   |-- optimizer.py            # Greedy optimization loop orchestrator
|   |-- database.py             # SQLite persistence layer
|   |-- main.py                 # Baseline runner CLI
|   |-- optimize_cli.py         # Optimization runner CLI
|
|-- api_key_manager/            # OpenAI-compatible key rotation utility
|-- REPORT.md                   # Final report with results and analysis
|-- README.md                   # This file
```

---


## Constraints and Notes

- **LLM Provider**: Google Gemini is used for all roles. The extractor model must support PDF/image input. Any Gemini model can be substituted via configuration.
- **Libraries**: No third-party library implements a substantial portion of the optimization loop. The optimization algorithm, mutation strategy, scoring function, and pipeline orchestration are implemented from scratch. See the [REPORT.md](REPORT.md) for a detailed library usage table.
- **Coding Assistants**: Coding assistants were used during development. The author can explain any line of submitted code.
- **Karpathy's autoresearch**: Referenced for architectural inspiration. No code was copied.
