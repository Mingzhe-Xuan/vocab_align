# Transport recipes

Each recipe fixes one directed source-to-receiver transport configuration.
`expected_artifact_shape` is always `[target_active_vocab, source_active_vocab]`;
the output path, marginals, tokenizer fingerprints, and artifact must be built
independently for that direction. A reverse experiment must never transpose or
reuse a forward artifact.

Formal recipes explicitly freeze the builder's safe special-token policy:
retain the full source tokenizer support, restrict the target OT support to
ordinary tokens, use exact-kind mapping followed by literal-byte fallback, and
leave BOS/EOS generation boundaries to the receiver model.

The current directed pairs are Qwen3-8B→Mistral-Nemo (primary),
Mistral-Nemo→Qwen3-8B (reverse), and Qwen3-8B→DeepSeek-R1-Distill-Llama-8B
(second heterogeneous tokenizer pair). Recipe presence is not evidence that a
full transport artifact or benchmark result has been produced; those require
separate audited Slurm runs.
