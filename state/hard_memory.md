# Hard Memory

Hard Memory contains verifiable state that can be used as evidence across sessions.

Rules:

- Every entry should be backed by a source such as a Git commit, verifier report, test output, trace, or inspected file.
- Hard Memory may be used for completion and recovery decisions.
- Do not store guesses, unverified causal explanations, or reflections here.

## Entries

- [decision][commit:6f8bd4d] Project baseline initialized as a training-free long-running coding agent harness.
- [decision][commit:391325a] Worker sessions use an artificial 16K estimated token budget and prepare handoff at 70%.
- [decision][commit:32dd612] Context is organized into Always-on, Startup, Just-in-Time, and Persistent layers.

