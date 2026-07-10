from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from agent.llm import validate_action
from agent.loop import AgentLoop
from agent.planner import create_initial_state
from agent.tools.bash import BashTool
from agent.tools.read import ReadTool
from eval.metrics import summarize


WORKSPACE_TMP = Path(__file__).resolve().parents[1] / ".tmp_tests"


class WorkspaceTemporaryDirectory:
    def __enter__(self) -> str:
        WORKSPACE_TMP.mkdir(exist_ok=True)
        self.path = WORKSPACE_TMP / uuid.uuid4().hex
        self.path.mkdir()
        return str(self.path)

    def __exit__(self, exc_type, exc, tb) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


class HarnessBehaviorTests(unittest.TestCase):
    def test_read_directory_lists_entries(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "alpha.txt").write_text("hello", encoding="utf-8")
            result = ReadTool(root).run({"action": "read", "target": ".", "args": {}})

        self.assertTrue(result.ok)
        self.assertIn("alpha.txt", result.data["content"])

    def test_bash_accepts_args_command(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            result = BashTool(Path(tmp)).run(
                {"action": "bash", "target": ".", "args": {"command": "python --version"}}
            )

        self.assertTrue(result.ok)
        self.assertIn("Python", result.data["output"])

    def test_validate_action_normalizes_non_object_args(self) -> None:
        state = create_initial_state("Inspect and suggest")
        action = validate_action({"action": "bash", "target": "ignored", "args": "python --version"}, state)

        self.assertEqual(action["args"], {"command": "python --version"})

    def test_answer_action_completes_answer_task(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            (root / "agent").mkdir()
            loop = AgentLoop(root=root, task="Inspect and suggest", max_steps=1)
            state = create_initial_state("Inspect and suggest")
            state.evidence_sources.extend(
                [
                    {"action": "read", "target": "README.md", "summary": "read"},
                    {"action": "read", "target": "agent/loop.py", "summary": "read"},
                    {"action": "read", "target": "agent/tools", "summary": "listed"},
                ]
            )
            observation = loop._execute_action(
                {"action": "answer", "target": "", "args": {"answer": "Next: add behavior tests."}},
                state,
            )

        self.assertTrue(observation.ok)
        self.assertEqual(observation.data["answer"], "Next: add behavior tests.")

    def test_answer_requires_key_repository_evidence(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Inspect and suggest", max_steps=1)
            state = create_initial_state("Inspect and suggest")
            state.evidence_sources.append({"action": "read", "target": "README.md", "summary": "read"})

            observation = loop._execute_action(
                {"action": "answer", "target": "", "args": {"answer": "Next: implement bash."}},
                state,
            )

        self.assertFalse(observation.ok)
        self.assertIn("agent/loop.py", observation.summary)
        self.assertIn("agent/tools", observation.summary)

    def test_write_requires_acceptance_contract(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement a feature", max_steps=1)
            state = create_initial_state("Implement a feature")

            observation = loop._execute_action(
                {
                    "action": "write",
                    "target": "feature.py",
                    "args": {"content": "print('hello')"},
                },
                state,
            )

        self.assertFalse(observation.ok)
        self.assertTrue(observation.data["missing_contract"])

    def test_contract_allows_write_for_active_task(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement a feature", max_steps=1)
            state = create_initial_state("Implement a feature")

            contract = loop._execute_action(
                {
                    "action": "contract",
                    "target": "T1",
                    "args": {
                        "task_id": "T1",
                        "summary": "Implement feature with a smoke check.",
                        "checks": ["python -m unittest discover -s tests"],
                    },
                },
                state,
            )
            loop._update_state(state, {"action": "contract", "target": "T1", "args": contract.data}, contract)
            observation = loop._execute_action(
                {
                    "action": "write",
                    "target": "feature.py",
                    "args": {"content": "print('hello')"},
                },
                state,
            )

        self.assertTrue(contract.ok)
        self.assertTrue(observation.ok)

    def test_verifier_rejects_weak_contract_without_behavior_check(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement a feature", max_steps=1)
            state = create_initial_state("Implement a feature")

            contract = loop._execute_action(
                {
                    "action": "contract",
                    "target": "T1",
                    "args": {
                        "task_id": "T1",
                        "summary": "Implement feature.",
                        "checks": ["feature.py exists"],
                    },
                },
                state,
            )

        self.assertFalse(contract.ok)
        self.assertFalse(contract.data["checks"]["behavior_level_checks"])

    def test_metrics_counts_answer_actions(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            trace = Path(tmp) / "run.jsonl"
            events = [
                {
                    "action": {"action": "read"},
                    "observation": {"ok": True},
                },
                {
                    "action": {"action": "answer"},
                    "observation": {"ok": True},
                },
            ]
            trace.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")

            summary = summarize(trace)

        self.assertEqual(summary["steps"], 2)
        self.assertEqual(summary["actions"]["answer"], 1)


if __name__ == "__main__":
    unittest.main()
