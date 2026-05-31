# Automated Prompt Optimization (APO) -- Final Report

---

## 1. Introduction

This report documents the design, implementation, and results of an Automated Prompt Optimization (APO) system built for the ExtractBench PDF-to-JSON structured extraction benchmark. The system accepts a seed prompt, a dataset, a scoring function, one or more LLMs, and a budget, then iteratively mutates the prompt using LLM-generated feedback to maximize extraction quality on a held-out test split.

The system was evaluated on three ExtractBench schemas: **hiring/resume**, **academic/research**, and **sport/swimming**. All runs used Google Gemini models (gemini-3.5-flash for extraction, gemini-2.5-flash for mutation).

---

## 2. System Architecture

### 2.1 High-Level Pipeline

```
Configuration (YAML)
        |
        v
  Dataset Loader  -->  Deterministic Train / Val / Test Split (seed=42)
        |
        v
  Seed Prompt (schema-specific or user-override)
        |
        v
  +-------------------------------------------+
  |       Greedy Optimization Loop             |
  |                                            |
  |  Step 0: Evaluate seed on validation set   |
  |  Step 1..N:                                |
  |    1. Mutator: analyze errors, propose     |
  |       revised system instruction via LLM   |
  |    2. Evaluate mutated prompt on val set   |
  |    3. Greedy accept/reject (F1 improves?)  |
  |    4. Persist step to SQLite               |
  +-------------------------------------------+
        |
        v
  Final Evaluation: seed + final prompt on test split
        |
        v
  Outputs: score_curve.png, prompt_diff.txt, summary.json, final_prompt.txt
```

### 2.2 Module Inventory

| Module | Responsibility |
|--------|---------------|
| `config.py` | YAML-based configuration loader with validation (split ratios, model params) |
| `dataset.py` | PDF/gold-JSON pairing, deterministic seeded shuffle + split |
| `pdf_loader.py` | PDF text extraction via `pdfplumber` |
| `schema.py` | Pydantic model compilation from ExtractBench JSON Schema (dynamic `$ref` resolution, `anyOf`/`oneOf` handling) |
| `prompts.py` | Schema-specific seed prompts with config override and generic fallback |
| `llm_wrapper.py` | Gemini API client with multi-key rotation, exponential backoff, cost tracking |
| `parser.py` | JSON parsing with hallucination detection, truncation recovery, and Pydantic validation |
| `evaluator.py` | Per-field scoring honoring `evaluation_config` metrics; array alignment; per-leaf/subtree/global aggregation |
| `mutator.py` | Error feedback compilation and LLM-driven prompt mutation |
| `optimizer.py` | Greedy optimization loop orchestration with SQLite-backed resumability |
| `database.py` | SQLite persistence for runs, steps, predictions, and semantic evaluation cache |
| `main.py` | Baseline extraction runner CLI |
| `optimize_cli.py` | Optimization loop runner CLI with `--resume` support |
| `api_key_manager/` | Thread-safe OpenAI-compatible key rotation (used for alternative provider support) |

### 2.3 Key Design Decisions

- **Optimization Algorithm**: Greedy accept/reject. A mutated prompt is accepted only if its mean validation F1 strictly exceeds the current best. This avoids overfitting to noise while maintaining monotonic improvement on the validation split.

- **Artifact Structure**: A single system instruction (prompt) is optimized. The prompt is the sole input to the extraction LLM alongside the document. No few-shot examples or prompt chains are used.

- **LLM Roles**: Two distinct roles are used:
  - **Extractor** (gemini-3.5-flash at temperature 0.0): Consumes the PDF/text and produces structured JSON.
  - **Mutator** (gemini-2.5-flash at temperature 0.7): Receives the current prompt and error feedback, proposes a revised prompt.

- **Evaluation Signal to Mutator**: The mutator receives a structured error report containing up to 10 specific field-level mismatches (missing fields, value mismatches, extra fields) from the lowest-scoring documents. This grounds the mutation in concrete failures rather than abstract performance numbers.

- **Resumability**: All optimization state (run metadata, per-step prompts and scores, per-document predictions, semantic evaluation cache) is persisted to SQLite. Interrupted runs resume from the last completed step with zero repeated API calls.

---

## 3. Scoring Function

The scoring function is independent of the optimization loop and implements the following:

### 3.1 Per-Field Evaluation Metrics

The scorer honors ExtractBench's per-field `evaluation_config`, supporting:

| Metric ID | Behavior |
|-----------|----------|
| `string_exact` | Exact string match |
| `string_case_insensitive` | Case-insensitive comparison (default fallback) |
| `string_fuzzy` | SequenceMatcher ratio with configurable threshold |
| `string_semantic` | LLM-based semantic equivalence (Gemini) with Jaccard similarity fallback; results cached per (prediction, gold) pair in SQLite |
| `integer_exact` | Integer equality after float conversion |
| `number_exact` | Float equality |
| `number_tolerance` | Absolute tolerance comparison |
| `boolean_exact` | Boolean equality with string coercion |

### 3.2 Array Alignment Policy

For repeated arrays (e.g., `workExperience`, `publications`), the scorer uses a **greedy bipartite matching** algorithm:

1. Compute a similarity matrix between all predicted and gold items based on matching primitive leaf fields.
2. Sort all (similarity, pred_index, gold_index) triples in descending order.
3. Greedily assign each predicted item to the highest-similarity unmatched gold item.
4. Evaluate field-by-field on the aligned pairs.

This avoids the combinatorial explosion of exhaustive permutation matching (O(N!) reduced to O(N^2)).

### 3.3 Aggregation Levels

The scorer produces:

- **Per-document**: Precision, Recall, F1 over all flattened leaf fields.
- **Per-subtree (top-level key)**: Aggregated TP/FP/FN counts across all documents for each top-level schema field.
- **Per-leaf (schema path)**: Aggregated TP/FP/FN for each individual leaf field across all documents.
- **Global**: Micro-averaged Precision, Recall, F1 across all leaf fields and all documents.

---

## 4. Results

### 4.1 Hiring / Resume Schema

**Configuration**: 7 documents total, 70% validation / 30% test, seed=42, 3 optimization iterations.

#### Test-Set Scores

| Metric | Seed Prompt | Final Prompt | Change |
|--------|-------------|--------------|--------|
| **Mean F1** | **0.5048** | **0.6581** | **+30.4%** |

#### Per-Subtree Breakdown (Test Split)

| Subtree | Seed F1 | Final F1 | Change |
|---------|---------|----------|--------|
| certificationsAndAwards | 0.2320 | 0.3784 | +0.1464 |
| education | 0.6842 | 0.6842 | +0.0000 |
| languages | 0.2000 | 0.2000 | +0.0000 |
| media | 0.4000 | 0.3333 | -0.0667 |
| other | 0.0000 | 0.0000 | +0.0000 |
| personalInfo | 0.7500 | 0.7500 | +0.0000 |
| publications | 0.2857 | 0.9388 | +0.6531 |
| skills | 1.0000 | 1.0000 | +0.0000 |
| socialLinks | 1.0000 | 1.0000 | +0.0000 |
| workExperience | 0.5094 | 0.5878 | +0.0784 |

#### Optimization Trajectory (Validation Split)

| Step | Status | Mean F1 |
|------|--------|---------|
| 0 | BASELINE | 0.5051 |
| 1 | ACCEPT | 0.5962 |
| 2 | ACCEPT | 0.6719 |
| 3 | REJECT | 0.6270 |

---

### 4.2 Academic / Research Schema

**Configuration**: 7 documents total, 70% validation / 30% test, seed=42, 3 optimization iterations.

#### Test-Set Scores

| Metric | Seed Prompt | Final Prompt | Change |
|--------|-------------|--------------|--------|
| **Mean F1 (Global)** | **0.1936** | **0.1991** | **+2.8%** |

#### Per-Subtree Breakdown (Test Split)

| Subtree | Seed F1 | Final F1 | Change |
|---------|---------|----------|--------|
| abstract | 1.0000 | 1.0000 | +0.0000 |
| authors | 0.5200 | 0.4783 | -0.0417 |
| citations | 0.1569 | 0.1221 | -0.0348 |
| keywords | 0.3077 | 0.3077 | +0.0000 |
| number_of_pages | 0.6667 | 1.0000 | +0.3333 |
| publication_type | 0.6667 | 1.0000 | +0.3333 |
| title | 1.0000 | 1.0000 | +0.0000 |

---

### 4.3 Sport / Swimming Schema

**Configuration**: 6 documents total, 70% validation / 30% test, seed=42, 3 optimization iterations.

#### Test-Set Scores

| Metric | Seed Prompt | Final Prompt |
|--------|-------------|--------------|
| **Baseline Validation F1** | **0.9616** | **0.9616 (no accepted mutations)** |

#### Per-Subtree Breakdown (Validation Split)

| Subtree | Baseline F1 |
|---------|-------------|
| age_groups | 0.9625 |
| championship | 1.0000 |
| event_details | 0.9167 |

**Note**: The swimming schema achieved a near-perfect baseline (0.9616 F1). All three mutations were rejected as none exceeded the seed score, demonstrating that the greedy algorithm correctly avoids regressions when the seed is already strong.

---

## 5. Seed Prompt vs. Final Prompt

### 5.1 Resume Schema

**Seed Prompt**:
> You are an expert resume parsing assistant. Your job is to read the candidate's resume text and extract all relevant information into the requested structured JSON format. Ensure names, emails, education history, work experience details, and lists of technical skills are accurately parsed and aligned. Be concise: keep description fields brief (under 200 characters each) and avoid duplicating information already captured in other structured fields. You MUST output a complete, valid JSON object -- never leave the output truncated.

**Final Optimized Prompt**:
> You are an expert resume parsing assistant. Your job is to read the candidate's resume text and extract all relevant information into the requested structured JSON format. Ensure names, emails, education history, work experience details, and lists of technical skills are accurately parsed and aligned. For all textual fields, preserve the original phrasing, specific characters (e.g., hyphens, en-dashes), and formatting unless explicitly instructed otherwise.
>
> Extract the *full and precise* employer name, ensuring no part is omitted. This includes any departments, divisions, specific institutions, or parenthetical abbreviations (e.g., 'Korea Advanced Institute of Science and Technology (KAIST), School of Mechanical Engineering' not just 'Korea Advanced Institute of Science and Technology (KAIST)'). Extract *all* skills comprehensively, ensuring no skills are missed and they are accurately categorized.
>
> For date fields (start/end dates):
> - If a full date (day, month, year) is present in the resume, format it as 'YYYY-MM-DD'.
> - If only month and year are present in the resume, extract the month and year exactly as a string (e.g., 'June 2016', 'Dec 2019').
> - If only a year is present in the resume, extract only the year exactly as a string (e.g., '2020', '2015').
> - Do NOT add default days or months where they are not explicitly present in the source text.
>
> For the `personalInfo.personalStatement`, extract the complete text without truncation. For `workExperience.description` and `education.description` fields, extract the *complete and exact* text for each bullet point or entry. Prioritize preserving original phrasing, all key details, line breaks, and punctuation (e.g., semicolons, bullet points) without rephrasing, summarizing, or omitting content. Do NOT truncate descriptions.
>
> Avoid duplicating information already captured in other structured fields. You MUST output a complete, valid JSON object -- never leave the output truncated.

### 5.2 Summary of Prompt Changes (Diff)

Key mutations accepted during the resume optimization run:

1. **Removed the 200-character description limit** -- the seed prompt's "keep description fields brief (under 200 characters)" instruction was the single largest source of information loss, causing truncated descriptions that failed fuzzy/semantic matching against gold annotations.

2. **Added explicit date formatting rules** -- the mutator identified that dates were being auto-formatted into ISO 8601 when the gold annotations expected the original string representation (e.g., "June 2016" rather than "2016-06-01"). The optimized prompt now specifies three clear rules based on date granularity.

3. **Added employer name precision guidance** -- field mismatches on `workExperience.employer` were traced to the LLM abbreviating institution names. The optimized prompt now explicitly instructs preserving departments, divisions, and parenthetical abbreviations.

4. **Added description completeness instructions** -- the mutator identified that `workExperience.description` and `personalInfo.personalStatement` were being summarized rather than extracted verbatim. Explicit instructions to preserve original phrasing and avoid truncation were added.

---

## 6. Optimization Trajectory Analysis

### 6.1 Resume Schema -- Step-by-Step

- **Step 0 (Baseline, F1=0.5051)**: The seed prompt's 200-character description limit caused systematic truncation across all description fields. Date formatting mismatches were pervasive.

- **Step 1 (Accepted, F1=0.5962, +18.0%)**: The mutator removed the description length constraint and added date formatting rules. This single mutation resolved the majority of description-related false negatives and date mismatches.

- **Step 2 (Accepted, F1=0.6719, +12.7%)**: The mutator added employer name precision guidance and strengthened description completeness instructions. This improved `workExperience.employer` matching and further reduced description truncation.

- **Step 3 (Rejected, F1=0.6270, -6.7%)**: The mutator over-specified date formatting rules (attempting ISO 8601 conversion for certain edge cases), which introduced new mismatches. The greedy algorithm correctly rejected this regression.

### 6.2 Cross-Schema Observations

- **High-baseline schemas resist mutation**: The swimming schema (0.9616 baseline) saw all mutations rejected. When the extraction model already performs near-perfectly, LLM-generated prompt mutations tend to over-specify edge cases that degrade general performance.

- **Citation-heavy schemas are challenging**: The research schema's low F1 is dominated by the `citations` subtree, where the LLM struggles with verbatim reproduction of long reference lists. Prompt mutations improved metadata fields (publication_type, number_of_pages) but had limited impact on citation extraction quality.

---

## 7. API Cost and Token Telemetry

| Schema | Total Prompt Tokens | Total Output Tokens | Total Cost (USD) |
|--------|--------------------|--------------------|------------------|
| Resume | 42,365 | 64,118 | $0.0224 |
| Research | (from run logs) | -- | -- |
| Swimming | 10,261 | 13,907 | $0.0049 |

All costs reflect Google Gemini API pricing (gemini-3.5-flash input: $0.075/M tokens, output: $0.30/M tokens).

---

## 8. Configurability

The system is fully reconfigurable via YAML without any code changes:

```yaml
model:
  name: "gemini-3.5-flash"        # Any Gemini model
  temperature: 0.0
  max_output_tokens: 16384

dataset:
  raw_dir: "dataset/hiring/resume/pdf+gold"
  gold_dir: "dataset/hiring/resume/pdf+gold"
  train_ratio: 0.0
  val_ratio: 0.7
  test_ratio: 0.3
  seed: 42

schema: "resume"                   # Any ExtractBench schema name

optimizer:
  iterations: 3                    # Budget (number of mutation attempts)
  mutation_model: "gemini-2.5-flash"
  mutation_temperature: 0.7

output_dir: "output"
```

To retarget to a new dataset:
1. Place PDFs and gold JSON files in a directory.
2. Place the JSON Schema file in `dataset/<category>/<name>/` with naming convention `<name>-schema.json`.
3. Create a new YAML config pointing to the directories.
4. Run: `python3 src/optimize_cli.py --config config/config_new.yaml`

No source code modifications are required. The schema module dynamically compiles Pydantic models from any valid JSON Schema, resolving `$ref`, `anyOf`, and `oneOf` constructs.

---

## 9. Observability

### 9.1 Per-Iteration Logging

Each optimization step logs:
- The full system instruction (prompt) used.
- Mean validation F1 and accept/reject status.
- Per-document prediction files with gold comparisons.

### 9.2 LLM Call Records

All LLM interactions are tracked with:
- Input/output token counts.
- Computed cost in USD.
- Per-prediction persistence in SQLite for auditability.

### 9.3 Diff and Regression Tooling

- `prompt_diff.txt`: Unified diff between seed and final prompt for each run.
- `score_curve.png`: Matplotlib trajectory plot with accept/reject/baseline annotations.
- `summary.json`: Machine-readable run summary with full trajectory, held-out test results, and per-subtree/per-leaf breakdowns.
- SQLite database: Queryable archive of all runs, steps, predictions, and semantic evaluation cache.

---

## 10. Limitations

1. **Greedy algorithm lacks exploration**: The single-candidate greedy approach cannot escape local optima. A population-based or beam search strategy would enable broader exploration of the prompt space, potentially finding better solutions for difficult schemas like research/citations.

2. **Error feedback is truncated**: The mutator receives at most 10 error examples with 3 field mismatches each. For schemas with hundreds of fields (e.g., citations with 30+ references), this sampling may not represent the dominant failure modes. An adaptive feedback budget proportional to error diversity would improve mutation quality.

3. **No structural prompt evolution**: The current system optimizes a single flat system instruction. It does not explore structural alternatives such as few-shot examples, chain-of-thought decomposition, or prompt ensembles. These could significantly improve extraction on complex schemas.

4. **Semantic evaluation fallback**: The `string_semantic` metric uses a Jaccard word-overlap heuristic (threshold 0.7) when API-based semantic evaluation is disabled (to conserve quota). This is a coarse approximation that may under-count true semantic matches, particularly for paraphrased content.

5. **No overfitting detection**: The system does not explicitly monitor for overfitting to the validation split. While the held-out test evaluation provides a final generalization check, incorporating early stopping or validation-test score divergence monitoring would improve robustness.

6. **Schema compilation limitations**: The dynamic Pydantic model builder collapses `anyOf`/`oneOf` unions to the first non-null type to satisfy Gemini API constraints (which do not support union types in structured output schemas). This may lose information for schemas with genuinely polymorphic fields.

7. **Single-document context**: Each extraction call processes one document independently. Cross-document consistency patterns (e.g., normalized institution names across resumes) are not leveraged.

---

## 11. Third-Party Libraries

| Library | Usage | Custom Extension |
|---------|-------|-----------------|
| `google-genai` | Gemini API client for extraction, mutation, and semantic evaluation | Wrapped with multi-key rotation, exponential backoff, and cost tracking |
| `pydantic` | Schema validation and dynamic model compilation | Extended with custom JSON Schema to Pydantic compiler (`build_pydantic_model_from_json_schema`) |
| `pdfplumber` | PDF text extraction | Used as-is |
| `matplotlib` | Score trajectory visualization | Used as-is |
| `PyYAML` | Configuration parsing | Used as-is |
| `sqlite3` | Persistence layer for resumability and caching | Custom schema and query layer (`APODatabase`) |

No third-party library implements a substantial portion of the optimization loop. The optimization algorithm, mutation strategy, scoring function, and all pipeline orchestration are implemented from scratch.

---

## 12. References

- ExtractBench: https://github.com/ContextualAI/extract-bench (MIT License)
- Karpathy, autoresearch: https://github.com/karpathy/autoresearch (referenced, not copied)
- Google Gemini API Documentation: https://ai.google.dev/docs
