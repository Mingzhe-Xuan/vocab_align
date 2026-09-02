# Evaluation scripts

`unified_evaluator.py` is the common benchmark entry point. Existing HF,
Rosetta, C2C, and two-stage modes retain their legacy loop. The
`training_free_transport` model name is dispatched to `transport_runner.py`,
which reuses the evaluator's dataset formatting and answer parser while writing
versioned per-sample JSONL records through `transport_adapter.py`.

Transport evaluation is currently pinned to one visible GPU per invocation.
Interrupted runs skip successful records, retry failed records, and reject
changed prompts or methods. `script.transport.summarize_transport` produces an
atomic summary from the per-sample records.
