# Soft Memory

Soft Memory contains language-level hypotheses and guidance. It can help the next Worker decide what to inspect, but it is not evidence.

Rules:

- Soft Memory must not be used as a verified fact.
- Promote an entry to Hard Memory only after verifier output, tests, traces, or inspected files confirm it.
- Remove or revise stale assumptions when contradicted by observations.

## Entries

- [next] A useful next implementation step is to formalize Hard/Soft Memory promotion rules and expose them in handoff.
- [reflection] Evidence gates reduced premature answers, but the model may still over-read before answering.

