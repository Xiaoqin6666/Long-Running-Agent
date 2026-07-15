# Skill Mechanism Evaluation Task

Complete the three preplanned tasks in order. This benchmark evaluates the Skill lifecycle as well as the generated code.

## Required behavior

1. Implement `eval/benchmarks/skill_mechanism/workspace/first_path.py` with:

   ```python
   normalize_path(value: str) -> str
   ```

   It must replace Windows backslashes with forward slashes, collapse repeated `/`, and preserve a leading `/`.

2. After the first implementation receives Verifier PASS, save a reusable Skill named `normalize-portable-path`.

   - Its description must make clear that it applies to converting mixed Windows/POSIX separators into stable forward-slash paths.
   - Its instructions must include implementation steps and an independent verification step.
   - Use `evidence_type="verified_success"` and an `evidence_refs` entry containing the immutable `report_id` returned by T1 Verifier PASS. Do not use the mutable latest report or free-text evidence.

3. For task T3, load `normalize-portable-path` before writing or editing `eval/benchmarks/skill_mechanism/workspace/second_path.py`. Then implement the same public API and behavior in that file by following the loaded Skill.

Do not directly write files under the benchmark state `skills/` or `skill_candidates/` directories; use `save_skill` and `load_skill`.
