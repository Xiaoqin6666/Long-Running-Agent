from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from agent.context import ContextBuilder
from agent.llm import validate_action
from agent.loop import AgentLoop
from agent.orchestrator import count_unlocked_tasks, select_current_task
from agent.planner import create_initial_state
from agent.termination import decide_termination, evaluate_task_graph
from agent.tools.bash import BashTool
from agent.tools.edit import EditTool
from agent.tools.git import GitTool
from agent.tools.list_files import ListFilesTool
from agent.tools.read import ReadTool
from agent.verifier import Verifier
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
    def test_termination_succeeds_only_when_all_project_checks_pass(self) -> None:
        checks = {
            "tasks": {"ok": True, "all_remaining_blocked": False},
            "regression": {"ok": True},
            "hidden_acceptance": {"ok": True},
            "git_clean": {"ok": True},
            "budget": {"ok": True, "exhausted": False},
            "failure_limits": {"ok": True, "exceeded": {}},
            "human_intervention": {"ok": True, "reasons": []},
        }

        result = decide_termination(checks)

        self.assertEqual(result.status, "completed")

    def test_termination_reports_failure_instead_of_fake_completion(self) -> None:
        checks = {
            "tasks": {"ok": False, "all_remaining_blocked": False},
            "regression": {"ok": True},
            "hidden_acceptance": {"ok": True},
            "git_clean": {"ok": True},
            "budget": {"ok": False, "exhausted": True},
            "failure_limits": {"ok": True, "exceeded": {}},
            "human_intervention": {"ok": True, "reasons": []},
        }

        result = decide_termination(checks)

        self.assertEqual(result.status, "stopped_with_failure")
        self.assertIn("total_token_budget_exhausted", result.reasons)

    def test_termination_can_pause_for_human_intervention(self) -> None:
        checks = {
            "tasks": {"ok": False, "all_remaining_blocked": False},
            "regression": {"ok": True},
            "hidden_acceptance": {"ok": False},
            "git_clean": {"ok": True},
            "budget": {"ok": True, "exhausted": False},
            "failure_limits": {"ok": True, "exceeded": {}},
            "human_intervention": {"ok": False, "reasons": ["external_api_key_required"]},
        }

        result = decide_termination(checks)

        self.assertEqual(result.status, "requires_human_intervention")
        self.assertIn("external_api_key_required", result.reasons)

    def test_task_graph_requires_all_required_tasks_completed(self) -> None:
        graph = evaluate_task_graph(
            [
                {"id": "T1", "status": "completed"},
                {"id": "T2", "status": "pending"},
                {"id": "T3", "status": "pending", "optional": True},
            ]
        )

        self.assertFalse(graph["ok"])
        self.assertEqual(graph["remaining_task_ids"], ["T2"])

    def test_orchestrator_prioritizes_latest_verifier_failure(self) -> None:
        tasks = [
            {"id": "T1", "status": "done", "priority": 1, "depends_on": []},
            {"id": "T2", "status": "pending", "priority": 1, "depends_on": ["T1"]},
            {"id": "T3", "status": "pending", "priority": 1, "depends_on": ["T1"]},
        ]

        selection = select_current_task(tasks, failed_task_id="T3")

        self.assertEqual(selection.task["id"], "T3")
        self.assertIn("verifier failure", selection.reason)

    def test_orchestrator_prioritizes_critical_path_before_priority(self) -> None:
        tasks = [
            {"id": "T1", "status": "done", "priority": 1, "depends_on": []},
            {"id": "T2", "status": "pending", "priority": 5, "depends_on": ["T1"]},
            {"id": "T3", "status": "pending", "priority": 1, "depends_on": ["T1"]},
            {"id": "T4", "status": "pending", "priority": 1, "depends_on": ["T2"]},
            {"id": "T5", "status": "pending", "priority": 1, "depends_on": ["T2"]},
        ]

        selection = select_current_task(tasks)

        self.assertEqual(count_unlocked_tasks(tasks[1], tasks), 2)
        self.assertEqual(selection.task["id"], "T2")

    def test_orchestrator_uses_priority_then_stable_id_as_tiebreakers(self) -> None:
        tasks = [
            {"id": "T1", "status": "done", "priority": 1, "depends_on": []},
            {"id": "T3", "status": "pending", "priority": 2, "depends_on": ["T1"]},
            {"id": "T2", "status": "pending", "priority": 2, "depends_on": ["T1"]},
            {"id": "T4", "status": "pending", "priority": 3, "depends_on": ["T1"]},
        ]

        selection = select_current_task(tasks)

        self.assertEqual(selection.task["id"], "T2")

    def test_read_directory_lists_entries(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "alpha.txt").write_text("hello", encoding="utf-8")
            result = ReadTool(root).run({"action": "read", "target": ".", "args": {}})

        self.assertTrue(result.ok)
        self.assertIn("alpha.txt", result.data["content"])

    def test_list_files_returns_structured_entries(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "alpha.txt").write_text("hello", encoding="utf-8")
            (root / "src").mkdir()

            result = ListFilesTool(root).run({"action": "list_files", "target": ".", "args": {}})

        self.assertTrue(result.ok)
        self.assertIn({"path": "alpha.txt", "type": "file"}, result.data["entries"])
        self.assertIn({"path": "src", "type": "dir"}, result.data["entries"])

    def test_edit_replaces_exact_text_once(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "alpha.txt"
            path.write_text("hello world\n", encoding="utf-8")

            result = EditTool(root).run(
                {
                    "action": "edit",
                    "target": "alpha.txt",
                    "args": {"old": "hello", "new": "hi"},
                }
            )
            content = path.read_text(encoding="utf-8")

        self.assertTrue(result.ok)
        self.assertEqual(content, "hi world\n")
        self.assertEqual(result.data["replacements"], 1)

    def test_git_rejects_network_or_destructive_commands(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            result = GitTool(Path(tmp)).run({"action": "git", "target": "push", "args": {}})

        self.assertFalse(result.ok)
        self.assertIn("not allowed", result.summary)

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

    def test_skill_rejects_unverified_reflection(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement a feature", max_steps=1)
            state = create_initial_state("Implement a feature")

            observation = loop._execute_action(
                {
                    "action": "skill",
                    "target": "random-thought",
                    "args": {
                        "skill_id": "random-thought",
                        "title": "Random thought",
                        "body": "Maybe do this next time.",
                        "evidence_type": "verified_success",
                        "evidence": ["I think it worked"],
                    },
                },
                state,
            )

        self.assertFalse(observation.ok)

    def test_skill_accepts_verifier_confirmed_success(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement a feature", max_steps=1)
            state = create_initial_state("Implement a feature")
            state.last_observation = {"ok": True, "summary": "Verifier passed.", "data": {}}

            observation = loop._execute_action(
                {
                    "action": "skill",
                    "target": "verified-debugging",
                    "args": {
                        "skill_id": "verified-debugging",
                        "title": "Verified debugging",
                        "body": "Run tests before claiming completion.",
                        "evidence_type": "verified_success",
                        "evidence": ["verifier_report: Verifier passed"],
                    },
                },
                state,
            )
            skill_path = root / "state" / "skills" / "verified-debugging.md"
            skill_exists = skill_path.exists()

        self.assertTrue(observation.ok)
        self.assertTrue(skill_exists)

    def test_skill_accepts_evidence_confirmed_failure(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement a feature", max_steps=1)
            state = create_initial_state("Implement a feature")

            observation = loop._execute_action(
                {
                    "action": "skill",
                    "target": "avoid-weak-contract",
                    "args": {
                        "skill_id": "avoid-weak-contract",
                        "title": "Avoid weak contracts",
                        "body": "Do not use file existence as the only acceptance check.",
                        "evidence_type": "evidence_confirmed_failure",
                        "evidence": ["trace: contract rejected because behavior_level_checks failed"],
                    },
                },
                state,
            )

        self.assertTrue(observation.ok)

    def test_handoff_ready_blocks_write(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement a feature", max_steps=1)
            state = create_initial_state("Implement a feature")
            state.handoff_ready = True
            state.acceptance_contracts.append(
                {
                    "task_id": "T1",
                    "summary": "Implement feature.",
                    "checks": ["unit tests pass"],
                    "status": "agreed",
                }
            )

            observation = loop._execute_action(
                {
                    "action": "write",
                    "target": "feature.py",
                    "args": {"content": "print('hello')"},
                },
                state,
            )

        self.assertFalse(observation.ok)
        self.assertTrue(observation.data["handoff_ready"])

    def test_handoff_contains_session_budget_and_resume_sections(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            (root / "state" / "hard_memory.md").write_text("# Hard Memory\n\n- [commit:x] fact\n", encoding="utf-8")
            (root / "state" / "soft_memory.md").write_text("# Soft Memory\n\n- [next] inspect\n", encoding="utf-8")
            loop = AgentLoop(root=root, task="Implement a feature", max_steps=1)
            state = create_initial_state("Implement a feature")
            state.session_budget_tokens = 100
            state.handoff_threshold = 0.7
            state.session_used_tokens = 71
            state.handoff_ready = True
            state.evidence_sources.append({"action": "read", "target": "agent/loop.py", "summary": "read"})

            loop._write_handoff(state)
            handoff = (root / "state" / "handoff.md").read_text(encoding="utf-8")

        self.assertIn("# Worker Session Handoff", handoff)
        self.assertIn("## 2. Session Budget", handoff)
        self.assertIn("## 8. Hard Memory", handoff)
        self.assertIn("## 9. Soft Memory", handoff)
        self.assertIn("## 15. Resume Instructions", handoff)
        self.assertIn("threshold_tokens: 70", handoff)

    def test_context_builder_uses_four_context_layers(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "memory.md").write_text("# Memory\n", encoding="utf-8")
            (root / "state" / "hard_memory.md").write_text("# Hard Memory\n", encoding="utf-8")
            (root / "state" / "soft_memory.md").write_text("# Soft Memory\n", encoding="utf-8")
            (root / "state" / "handoff.md").write_text("# Handoff\n", encoding="utf-8")
            (root / "state" / "verifier_report.md").write_text("# Verifier\n", encoding="utf-8")
            (root / "project_spec.md").write_text("# Spec\n", encoding="utf-8")
            (root / "tasks.json").write_text("{}", encoding="utf-8")
            state = create_initial_state("Implement a feature")

            context = ContextBuilder(root).build(state)

        self.assertIn("# Always-on Context", context)
        self.assertIn("# Startup Context", context)
        self.assertIn("# Just-in-Time Context", context)
        self.assertIn("# Persistent Context", context)
        self.assertIn("# Hard Memory", context)
        self.assertIn("# Soft Memory", context)
        self.assertIn("Soft Memory is not evidence", context)

    def test_verifier_writes_latest_report(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            (root / "agent").mkdir()
            state = create_initial_state("Implement a feature")
            state.nodes[0]["evidence"].append("test evidence")

            result = Verifier(root).run("default", state)
            report = (root / "state" / "verifier_report.md").read_text(encoding="utf-8")

        self.assertTrue(result.ok)
        self.assertIn("Latest Verifier Report", report)
        self.assertIn("Verifier passed", report)

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
