# Evaluation scripts

`unified_evaluator.py` is the common benchmark entry point. Existing HF,
Rosetta, C2C, and two-stage modes retain their legacy loop. The
`training_free_transport` model name is dispatched to `transport_runner.py`,
which reuses the evaluator's dataset formatting and answer parser while writing
versioned per-sample JSONL records through `transport_adapter.py`.

The transport runner has explicit loaders for MMLU-Redux subject configs,
GSM8K `main`, MATH-500, and LongBench task configs. Recipes may bind a local
JSON/Parquet data file to its SHA-256 while retaining the upstream dataset
revision in prompt provenance. LongBench prompts produced by the unified
formatter are already rendered with the source tokenizer and are encoded
directly; they must not be wrapped in a second chat template. LongBench records
preserve the official reference fields and use `external_required` rather than
inventing a boolean accuracy before a task scorer is applied.

Transport evaluation is currently pinned to one visible GPU per invocation.
Interrupted runs skip successful records, retry failed records, and reject
changed prompts or methods. `script.transport.summarize_transport` produces an
atomic summary from the per-sample records.
