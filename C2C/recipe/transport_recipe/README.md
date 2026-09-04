# Transport recipes

Each recipe fixes one directed source-to-receiver transport configuration.
`expected_artifact_shape` is always `[target_active_vocab, source_active_vocab]`;
the output path, marginals, tokenizer fingerprints, and artifact must be built
independently for a newly solved reverse OT experiment. A bare transpose is
never a valid conditional transport. When the research question instead asks
for the reverse conditional of the *same* learned joint coupling,
`rosetta.transport.reversal.reverse_transport_artifact` may derive it by Bayes
reconditioning with the realized marginal. That artifact must use a separate
path and explicit derivation provenance and is not equivalent to solving a new
reverse-direction OT problem.

The active collaboration contract fixes the source/sender as `planner` and the
target/receiver as `thinker`. The same problem is rendered independently by
both native chat templates with thinking explicitly enabled. The sender first
generates its thought; STT then aligns hidden states for its complete prompt and
thought, and those aligned embeddings are prepended to the receiver's native
prompt. Thus the receiver context order is conceptually `sender_prompt +
sender_think + receiver_prompt`, while the first two segments cross the model
boundary as aligned embeddings rather than target-token text.

Formal recipes explicitly freeze the builder's safe special-token policy:
retain the full source tokenizer support, restrict the target OT support to
ordinary tokens, use exact-kind mapping followed by literal-byte fallback, and
leave BOS/EOS generation boundaries to the receiver model.

The current directed pairs are Qwen3-8B→Mistral-Nemo (primary),
Mistral-Nemo→Qwen3-8B (reverse), and Qwen3-8B→DeepSeek-R1-Distill-Llama-8B
(second heterogeneous tokenizer pair). Recipe presence is not evidence that a
full transport artifact or benchmark result has been produced; those require
separate audited Slurm runs.
