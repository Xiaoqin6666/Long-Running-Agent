from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from agent.context import ContextBuilder
from agent.llm import parse_action_json, validate_action
from agent.loop import AgentLoop
from agent.main import build_parser, infer_benchmark_id
from agent.orchestrator import Orchestrator, count_unlocked_tasks, select_current_task
from agent.planner import create_initial_state
from agent.termination import ProjectTerminator, decide_termination, evaluate_task_graph
from agent.tools import ToolResult
from agent.tools.bash import BashTool
from agent.tools.edit import EditTool
from agent.tools.git import GitTool
from agent.tools.list_files import ListFilesTool
from agent.tools.read import ReadTool
from agent.verifier import Verifier
from eval.metrics import load_events, summarize


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
    def test_cli_accepts_custom_tasks_json(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["Task", "--tasks-json", "eval/benchmarks/issue_tracker/tasks.json"])

        self.assertEqual(str(args.tasks_json), "eval\\benchmarks\\issue_tracker\\tasks.json")

    def test_cli_accepts_project_spec(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--project-spec", "eval/benchmarks/todo_counter/project_spec.md"])

        self.assertEqual(str(args.project_spec), "eval\\benchmarks\\todo_counter\\project_spec.md")

    def test_cli_accepts_explicit_benchmark(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["Task", "--benchmark", "todo_counter"])

        self.assertEqual(args.benchmark, "todo_counter")

    def test_benchmark_id_is_inferred_from_benchmark_paths(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--project-spec",
                "eval/benchmarks/todo_counter/project_spec.md",
            ]
        )

        self.assertEqual(infer_benchmark_id(args), "todo_counter")

    def test_project_spec_starts_initializer_without_prebuilt_tasks_json(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = root / "eval" / "benchmarks" / "todo_counter" / "project_spec.md"
            spec.parent.mkdir(parents=True)
            spec.write_text("# Todo Counter\n", encoding="utf-8")
            loop = AgentLoop(
                root=root,
                task=spec.read_text(encoding="utf-8"),
                max_steps=1,
                project_spec_path=spec,
                benchmark_id="todo_counter",
            )
            loop._ensure_state_files()
            loop._prepare_runtime_task_graph()

            state = loop._load_or_create_state()

            self.assertEqual(state.task_id, "INIT")
            self.assertEqual(state.nodes[0]["id"], "INIT")
            self.assertEqual(loop.state_dir, root / "state" / "benchmarks" / "todo_counter")
            self.assertEqual(loop.tasks_path, root / "state" / "benchmarks" / "todo_counter" / "generated_tasks.json")
            self.assertTrue((root / "state" / "benchmarks" / "todo_counter" / "project_spec.md").exists())
            self.assertFalse((root / "state" / "benchmarks" / "todo_counter" / "runtime_tasks.json").exists())

    def test_project_spec_uses_generated_tasks_after_initializer(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = root / "eval" / "benchmarks" / "todo_counter" / "project_spec.md"
            spec.parent.mkdir(parents=True)
            spec.write_text("# Todo Counter\n", encoding="utf-8")
            generated = root / "state" / "benchmarks" / "todo_counter" / "generated_tasks.json"
            generated.parent.mkdir(parents=True)
            generated.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "id": "TC1",
                                "title": "Create skeleton",
                                "status": "pending",
                                "priority": 1,
                                "depends_on": [],
                                "acceptance_criteria": ["Skeleton exists."],
                                "expected_artifacts": ["eval/benchmarks/todo_counter/workspace/README.md"],
                                "verification_commands": ["python -c \"assert True\""],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            loop = AgentLoop(
                root=root,
                task=spec.read_text(encoding="utf-8"),
                max_steps=1,
                project_spec_path=spec,
                benchmark_id="todo_counter",
            )
            loop._ensure_state_files()
            loop._prepare_runtime_task_graph()

            state = loop._load_or_create_state()

            self.assertEqual(state.task_id, "TC1")
            self.assertEqual(state.nodes[0]["id"], "TC1")
            self.assertEqual(state.nodes[0]["status"], "in_progress")

    def test_custom_tasks_json_is_copied_to_runtime_state(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "eval" / "tasks" / "benchmark_tasks.json"
            source.parent.mkdir(parents=True)
            source.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {"id": "T1", "status": "pending", "priority": 1, "depends_on": []},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            loop = AgentLoop(root=root, task="Benchmark task", max_steps=1, tasks_path=source)
            loop._ensure_state_files()
            loop._prepare_runtime_task_graph()

            loop.orchestrator.mark_in_progress("T1", "scheduled")
            source_data = json.loads(source.read_text(encoding="utf-8"))
            runtime_data = json.loads((root / "state" / "runtime_tasks.json").read_text(encoding="utf-8"))

        self.assertEqual(source_data["tasks"][0]["status"], "pending")
        self.assertEqual(runtime_data["tasks"][0]["status"], "in_progress")
        self.assertEqual(runtime_data["tasks"][0]["evidence"], ["scheduled"])

    def test_benchmark_tasks_json_is_copied_to_benchmark_runtime_state(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "eval" / "benchmarks" / "issue_tracker" / "tasks.json"
            source.parent.mkdir(parents=True)
            source.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {"id": "IT1", "status": "pending", "priority": 1, "depends_on": []},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            loop = AgentLoop(root=root, task="Benchmark task", max_steps=1, tasks_path=source, benchmark_id="issue_tracker")
            loop._ensure_state_files()
            loop._prepare_runtime_task_graph()

            loop.orchestrator.mark_in_progress("IT1", "scheduled")
            runtime = root / "state" / "benchmarks" / "issue_tracker" / "runtime_tasks.json"
            source_data = json.loads(source.read_text(encoding="utf-8"))
            runtime_data = json.loads(runtime.read_text(encoding="utf-8"))

        self.assertFalse((root / "state" / "runtime_tasks.json").exists())
        self.assertEqual(source_data["tasks"][0]["status"], "pending")
        self.assertEqual(runtime_data["tasks"][0]["status"], "in_progress")

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

    def test_orchestrator_persists_task_status_transitions(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tasks.json").write_text(
                json.dumps(
                    {
                        "tasks": [
                            {"id": "T1", "status": "pending", "priority": 1, "depends_on": []},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            orchestrator = Orchestrator(root)

            orchestrator.mark_in_progress("T1", "scheduled")
            orchestrator.mark_awaiting_verification("T1", "candidate ready")
            orchestrator.mark_verified("T1", True, "Verifier passed.")
            data = json.loads((root / "tasks.json").read_text(encoding="utf-8"))

        self.assertEqual(data["tasks"][0]["status"], "completed")
        self.assertEqual(data["tasks"][0]["evidence"], ["scheduled", "candidate ready", "Verifier passed."])

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

    def test_list_files_missing_path_recommends_write(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            result = ListFilesTool(Path(tmp)).run({"action": "list_files", "target": "missing", "args": {}})

        self.assertFalse(result.ok)
        self.assertTrue(result.data["missing_path"])
        self.assertEqual(result.data["recommended_action"], "write")

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

    def test_parse_action_json_extracts_object_after_preface(self) -> None:
        action = parse_action_json(
            'I will inspect first. {"action": "list_files", "target": ".", "args": {}, '
            '"thought_summary": "Inspect.", "expected_observation": "Files.", "risk": "low"}'
        )

        self.assertEqual(action["action"], "list_files")
        self.assertEqual(action["target"], ".")

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

    def test_contract_update_deduplicates_same_task_contract(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement a feature", max_steps=1)
            state = create_initial_state("Implement a feature")
            action = {
                "action": "contract",
                "target": "T1",
                "args": {
                    "task_id": "T1",
                    "summary": "Implement feature with a smoke check.",
                    "checks": ["python -m unittest discover -s tests"],
                },
            }

            first = loop._execute_action(action, state)
            second = loop._execute_action(action, state)
            loop._update_state(state, action, first)
            loop._update_state(state, action, second)

        self.assertEqual(len(state.acceptance_contracts), 1)

    def test_guard_rewrites_repeated_missing_list_to_required_write(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement a feature", max_steps=1)
            state = create_initial_state("Implement a feature")
            state.acceptance_contracts.append(
                {
                    "task_id": "T1",
                    "summary": "Create skeleton.",
                    "checks": [
                        "File eval/benchmarks/issue_tracker/workspace/README.md exists.",
                        "File eval/benchmarks/issue_tracker/workspace/issue_tracker/__init__.py exists.",
                    ],
                    "status": "agreed",
                }
            )
            state.last_observation = {
                "ok": False,
                "summary": "missing",
                "data": {"missing_path": True, "recommended_action": "write"},
            }

            guarded = loop._guard_action(
                {
                    "thought_summary": "Check again.",
                    "action": "list_files",
                    "target": "eval/benchmarks/issue_tracker/workspace",
                    "args": {},
                    "expected_observation": "List target.",
                    "risk": "low",
                },
                state,
            )
            observation = loop._execute_action(guarded, state)

        self.assertEqual(guarded["action"], "write")
        self.assertEqual(guarded["target"], "eval/benchmarks/issue_tracker/workspace/README.md")
        self.assertEqual(guarded["guard_override"], "missing_path_list_files_to_write")
        self.assertTrue(observation.ok)

    def test_guard_rewrites_repeated_existing_list_to_next_missing_file(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            existing = root / "eval" / "benchmarks" / "issue_tracker" / "workspace" / "issue_tracker" / "__init__.py"
            existing.parent.mkdir(parents=True)
            existing.write_text('"""Issue tracker package."""\n', encoding="utf-8")
            loop = AgentLoop(root=root, task="Implement a feature", max_steps=1)
            state = create_initial_state("Implement a feature")
            state.acceptance_contracts.append(
                {
                    "task_id": "T1",
                    "summary": "Create skeleton.",
                    "checks": [
                        "File eval/benchmarks/issue_tracker/workspace/issue_tracker/__init__.py exists.",
                        "File eval/benchmarks/issue_tracker/workspace/tests/__init__.py exists.",
                        "File eval/benchmarks/issue_tracker/workspace/README.md exists.",
                    ],
                    "status": "agreed",
                }
            )
            state.last_action = {
                "action": "list_files",
                "target": "eval/benchmarks/issue_tracker/workspace",
                "args": {},
            }
            state.last_observation = {
                "ok": True,
                "summary": "listed",
                "data": {
                    "target": "eval/benchmarks/issue_tracker/workspace",
                    "entries": [{"path": "eval/benchmarks/issue_tracker/workspace/issue_tracker", "type": "dir"}],
                },
            }

            guarded = loop._guard_action(
                {
                    "thought_summary": "List again.",
                    "action": "list_files",
                    "target": "eval/benchmarks/issue_tracker/workspace",
                    "args": {},
                    "expected_observation": "List target.",
                    "risk": "low",
                },
                state,
            )

        self.assertEqual(guarded["action"], "write")
        self.assertEqual(guarded["target"], "eval/benchmarks/issue_tracker/workspace/tests/__init__.py")

    def test_guard_rewrites_repeated_list_to_expected_artifact(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            (root / "workspace" / "src").mkdir(parents=True)
            (root / "workspace" / "tests").mkdir()
            loop = AgentLoop(root=root, task="Implement persistence", max_steps=1)
            state = create_initial_state("Implement persistence")
            state.task_id = "T2"
            state.user_goal = "T2: Implement persistence layer"
            state.acceptance_criteria = [
                "Store can create, list, get, update, and delete issues.",
                "Unit tests cover store behavior.",
            ]
            state.nodes = [
                {
                    "id": "T2",
                    "title": "Persistence",
                    "status": "in_progress",
                    "evidence": [],
                    "expected_artifacts": ["workspace/README.md"],
                }
            ]
            state.acceptance_contracts.append(
                {
                    "task_id": "T2",
                    "summary": "Complete T2.",
                    "checks": state.acceptance_criteria + [
                        "python -m unittest discover -s workspace/tests"
                    ],
                    "status": "agreed",
                }
            )
            state.last_action = {
                "action": "list_files",
                "target": "workspace",
                "args": {},
            }
            state.last_observation = {
                "ok": True,
                "summary": "listed",
                "data": {
                    "target": "workspace",
                    "entries": [{"path": "workspace/src", "type": "dir"}],
                },
            }

            guarded = loop._guard_action(
                {
                    "thought_summary": "Inspect again.",
                    "action": "list_files",
                    "target": "workspace",
                    "args": {},
                    "expected_observation": "List target.",
                    "risk": "low",
                },
                state,
            )

        self.assertEqual(guarded["action"], "write")
        self.assertEqual(guarded["target"], "workspace/README.md")
        self.assertIn("Commands", guarded["args"]["content"])

    def test_guard_rewrites_existing_expected_artifacts_to_test(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            (root / "workspace" / "src").mkdir(parents=True)
            (root / "workspace" / "tests").mkdir()
            (root / "workspace" / "src" / "store.py").write_text("class Store: pass\n", encoding="utf-8")
            (root / "workspace" / "tests" / "test_store.py").write_text("import unittest\n", encoding="utf-8")
            loop = AgentLoop(root=root, task="Implement persistence", max_steps=1)
            state = create_initial_state("Implement persistence")
            state.task_id = "T2"
            state.user_goal = "T2: Implement persistence layer"
            state.acceptance_criteria = ["Unit tests cover store behavior."]
            state.nodes = [
                {
                    "id": "T2",
                    "title": "Persistence",
                    "status": "in_progress",
                    "evidence": [],
                    "expected_artifacts": ["workspace/src/store.py", "workspace/tests/test_store.py"],
                    "verification_commands": ["python -m unittest discover -s workspace/tests"],
                }
            ]
            test_command = "python -m unittest discover -s workspace/tests"
            state.acceptance_contracts.append(
                {
                    "task_id": "T2",
                    "summary": "Complete T2.",
                    "checks": ["Unit tests cover store behavior.", test_command],
                    "status": "agreed",
                }
            )
            state.last_action = {
                "action": "list_files",
                "target": "workspace",
                "args": {},
            }
            state.last_observation = {
                "ok": True,
                "summary": "listed",
                "data": {
                    "target": "workspace",
                    "entries": [{"path": "workspace/src", "type": "dir"}],
                },
            }

            guarded = loop._guard_action(
                {
                    "thought_summary": "Inspect again.",
                    "action": "list_files",
                    "target": "workspace",
                    "args": {},
                    "expected_observation": "List target.",
                    "risk": "low",
                },
                state,
            )

        self.assertEqual(guarded["action"], "bash")
        self.assertEqual(guarded["target"], test_command)
        self.assertEqual(guarded["guard_override"], "implementation_files_exist_to_test")

    def test_guard_rewrites_repeated_inspection_without_contract_to_contract(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement persistence", max_steps=1)
            state = create_initial_state("Implement persistence")
            state.task_id = "IT2"
            state.user_goal = "IT2: Implement JSON persistence layer"
            state.acceptance_criteria = [
                "Store can create, list, get, update, and delete issues.",
                "Issue ids are deterministic increasing integers.",
                "Persistence survives reloading the JSON file.",
                "Unit tests cover store behavior.",
            ]
            state.nodes = [
                {
                    "id": "IT2",
                    "title": "Persistence",
                    "status": "in_progress",
                    "evidence": [],
                    "verification_commands": [
                        "python -m unittest discover -s eval/benchmarks/issue_tracker/workspace/tests"
                    ],
                }
            ]
            state.last_action = {
                "action": "list_files",
                "target": "eval/benchmarks/issue_tracker/workspace",
                "args": {},
            }
            state.last_observation = {
                "ok": True,
                "summary": "listed",
                "data": {
                    "target": "eval/benchmarks/issue_tracker/workspace",
                    "entries": [{"path": "eval/benchmarks/issue_tracker/workspace/issue_tracker", "type": "dir"}],
                },
            }

            guarded = loop._guard_action(
                {
                    "thought_summary": "Inspect again.",
                    "action": "list_files",
                    "target": "eval/benchmarks/issue_tracker/workspace",
                    "args": {},
                    "expected_observation": "List target.",
                    "risk": "low",
                },
                state,
            )

        self.assertEqual(guarded["action"], "contract")
        self.assertEqual(guarded["target"], "IT2")
        self.assertIn("python -m unittest discover", guarded["args"]["checks"][-1])
        self.assertEqual(guarded["guard_override"], "repeated_inspection_to_contract")

    def test_guard_rewrites_protocol_error_recovery_to_contract(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement persistence", max_steps=1)
            state = create_initial_state("Implement persistence")
            state.task_id = "IT2"
            state.user_goal = "IT2: Implement JSON persistence layer"
            state.acceptance_criteria = ["Unit tests cover store behavior."]
            state.nodes = [{"id": "IT2", "title": "Persistence", "status": "in_progress", "evidence": []}]
            state.last_action = {"action": "protocol_error", "target": "decision_maker", "args": {}}
            state.last_observation = {"ok": False, "summary": "invalid JSON", "data": {}}

            guarded = loop._guard_action(
                {
                    "thought_summary": "Inspect again.",
                    "action": "list_files",
                    "target": "eval/benchmarks/issue_tracker/workspace",
                    "args": {},
                    "expected_observation": "List target.",
                    "risk": "low",
                },
                state,
            )

        self.assertEqual(guarded["action"], "contract")
        self.assertEqual(guarded["guard_override"], "repeated_inspection_to_contract")

    def test_guard_rewrites_duplicate_create_to_next_missing_file(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            existing = root / "eval" / "benchmarks" / "issue_tracker" / "workspace" / "issue_tracker" / "__init__.py"
            existing.parent.mkdir(parents=True)
            existing.write_text('"""Issue tracker package."""\n', encoding="utf-8")
            loop = AgentLoop(root=root, task="Implement a feature", max_steps=1)
            state = create_initial_state("Implement a feature")
            state.acceptance_contracts.append(
                {
                    "task_id": "T1",
                    "summary": "Create skeleton.",
                    "checks": [
                        "File eval/benchmarks/issue_tracker/workspace/issue_tracker/__init__.py exists.",
                        "File eval/benchmarks/issue_tracker/workspace/tests/__init__.py exists.",
                    ],
                    "status": "agreed",
                }
            )

            guarded = loop._guard_action(
                {
                    "thought_summary": "Create package init.",
                    "action": "write",
                    "target": "eval/benchmarks/issue_tracker/workspace/issue_tracker/__init__.py",
                    "args": {"mode": "create", "content": "# package\n"},
                    "expected_observation": "Create file.",
                    "risk": "low",
                },
                state,
            )

        self.assertEqual(guarded["target"], "eval/benchmarks/issue_tracker/workspace/tests/__init__.py")
        self.assertEqual(guarded["guard_override"], "duplicate_create_to_next_required_file")

    def test_guard_rewrites_duplicate_create_to_expected_artifact(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            existing = root / "workspace" / "src" / "__init__.py"
            existing.parent.mkdir(parents=True)
            existing.write_text('"""Package."""\n', encoding="utf-8")
            (root / "workspace" / "tests").mkdir(parents=True)
            loop = AgentLoop(root=root, task="Implement skeleton", max_steps=1)
            state = create_initial_state("Implement skeleton")
            state.task_id = "T1"
            state.nodes = [
                {
                    "id": "T1",
                    "title": "Skeleton",
                    "status": "in_progress",
                    "expected_artifacts": [
                        "workspace/src/__init__.py",
                        "workspace/tests/__init__.py",
                        "workspace/README.md",
                    ],
                }
            ]
            state.acceptance_contracts.append(
                {
                    "task_id": "T1",
                    "summary": "Create skeleton.",
                    "checks": ["python -c \"assert True\""],
                    "status": "agreed",
                }
            )

            guarded = loop._guard_action(
                {
                    "thought_summary": "Create package init.",
                    "action": "write",
                    "target": "workspace/src/__init__.py",
                    "args": {"content": "# package\n"},
                    "expected_observation": "Create file.",
                    "risk": "low",
                },
                state,
            )

        self.assertEqual(guarded["target"], "workspace/tests/__init__.py")
        self.assertEqual(guarded["guard_override"], "duplicate_create_to_next_required_file")

    def test_guard_rewrites_duplicate_create_to_verification_when_files_exist(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            readme = root / "workspace" / "README.md"
            readme.parent.mkdir(parents=True)
            readme.write_text("# Project\n\nCLI commands\n", encoding="utf-8")
            loop = AgentLoop(root=root, task="Create skeleton", max_steps=1)
            state = create_initial_state("Create skeleton")
            state.task_id = "T1"
            state.nodes = [
                {
                    "id": "T1",
                    "title": "Skeleton",
                    "status": "in_progress",
                    "expected_artifacts": ["workspace/README.md"],
                    "verification_commands": ["python -c \"assert True\""],
                }
            ]
            state.acceptance_contracts.append(
                {
                    "task_id": "T1",
                    "summary": "Create skeleton.",
                    "checks": ["python -c \"assert True\""],
                    "status": "agreed",
                }
            )

            guarded = loop._guard_action(
                {
                    "thought_summary": "Create README.",
                    "action": "write",
                    "target": "workspace/README.md",
                    "args": {"content": "# Project\n"},
                    "expected_observation": "Create file.",
                    "risk": "low",
                },
                state,
            )

        self.assertEqual(guarded["action"], "bash")
        self.assertEqual(guarded["target"], "python -c \"assert True\"")
        self.assertEqual(guarded["guard_override"], "duplicate_create_to_verification")

    def test_contract_file_extraction_ignores_python_function_names(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement a feature", max_steps=1)
            state = create_initial_state("Implement a feature")
            state.acceptance_contracts.append(
                {
                    "task_id": "T1",
                    "summary": "Create skeleton.",
                    "checks": [
                        "File eval/benchmarks/issue_tracker/workspace/README.md exists.",
                        "Smoke test: python -c \"import os; assert os.path.isdir('eval/benchmarks/issue_tracker/workspace')\"",
                    ],
                    "status": "agreed",
                }
            )

            target = loop._next_contract_file_target(state)

        self.assertEqual(target, "eval/benchmarks/issue_tracker/workspace/README.md")

    def test_guard_rewrites_after_smoke_pass_to_verify(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement a feature", max_steps=1)
            state = create_initial_state("Implement a feature")
            smoke = "python -c \"assert True\""
            state.acceptance_contracts.append(
                {
                    "task_id": "T1",
                    "summary": "Create skeleton.",
                    "checks": [f"Smoke test: {smoke}"],
                    "status": "agreed",
                }
            )
            state.last_action = {"action": "bash", "target": smoke, "args": {}}
            state.last_observation = {
                "ok": True,
                "summary": "Command exited with code 0.",
                "data": {"command": smoke, "output": ""},
            }

            guarded = loop._guard_action(
                {
                    "thought_summary": "Inspect more.",
                    "action": "list_files",
                    "target": ".",
                    "args": {},
                    "expected_observation": "List files.",
                    "risk": "low",
                },
                state,
            )

        self.assertEqual(guarded["action"], "verify")
        self.assertEqual(guarded["guard_override"], "smoke_passed_to_verify")

    def test_guard_rewrites_after_contract_test_pass_to_verify(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement persistence", max_steps=1)
            state = create_initial_state("Implement persistence")
            state.task_id = "IT2"
            state.user_goal = "IT2: Implement JSON persistence layer"
            state.nodes = [{"id": "IT2", "title": "Persistence", "status": "in_progress", "evidence": []}]
            command = "python -m unittest discover -s eval/benchmarks/issue_tracker/workspace/tests"
            state.acceptance_contracts.append(
                {
                    "task_id": "IT2",
                    "summary": "Complete IT2.",
                    "checks": [command],
                    "status": "agreed",
                }
            )
            state.last_action = {"action": "bash", "target": command, "args": {}}
            state.last_observation = {
                "ok": True,
                "summary": "Command exited with code 0.",
                "data": {"command": command, "output": "OK"},
            }

            guarded = loop._guard_action(
                {
                    "thought_summary": "Inspect again.",
                    "action": "list_files",
                    "target": ".",
                    "args": {},
                    "expected_observation": "List files.",
                    "risk": "low",
                },
                state,
            )

        self.assertEqual(guarded["action"], "verify")

    def test_guard_does_not_autocreate_empty_python_artifact(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            (root / "workspace" / "pkg").mkdir(parents=True)
            loop = AgentLoop(root=root, task="Implement store", max_steps=1)
            state = create_initial_state("Implement store")
            state.task_id = "T2"
            state.nodes = [
                {
                    "id": "T2",
                    "title": "Store",
                    "status": "in_progress",
                    "expected_artifacts": ["workspace/pkg/store.py"],
                    "verification_commands": ["python -m unittest discover -s workspace/tests"],
                }
            ]
            state.acceptance_contracts.append(
                {
                    "task_id": "T2",
                    "summary": "Implement store.",
                    "checks": ["python -m unittest discover -s workspace/tests"],
                    "status": "agreed",
                }
            )
            state.last_action = {"action": "list_files", "target": "workspace", "args": {}}
            state.last_observation = {
                "ok": True,
                "summary": "listed",
                "data": {"target": "workspace", "entries": [{"path": "workspace/pkg", "type": "dir"}]},
            }

            guarded = loop._guard_action(
                {
                    "thought_summary": "List again.",
                    "action": "list_files",
                    "target": "workspace",
                    "args": {},
                    "expected_observation": "List files.",
                    "risk": "low",
                },
                state,
            )

        self.assertEqual(guarded["action"], "list_files")

    def test_guard_does_not_reread_same_empty_python_artifact(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            store = root / "workspace" / "pkg" / "store.py"
            store.parent.mkdir(parents=True)
            store.write_text("\n", encoding="utf-8")
            loop = AgentLoop(root=root, task="Implement store", max_steps=1)
            state = create_initial_state("Implement store")
            state.task_id = "T2"
            state.nodes = [
                {
                    "id": "T2",
                    "title": "Store",
                    "status": "in_progress",
                    "expected_artifacts": ["workspace/pkg/store.py"],
                    "verification_commands": ["python -m unittest discover -s workspace/tests"],
                }
            ]
            state.acceptance_contracts.append(
                {
                    "task_id": "T2",
                    "summary": "Implement store.",
                    "checks": ["python -m unittest discover -s workspace/tests"],
                    "status": "agreed",
                }
            )
            state.last_action = {"action": "read", "target": "workspace/pkg/store.py", "args": {}}
            state.last_observation = {
                "ok": True,
                "summary": "Read empty file.",
                "data": {"target": "workspace/pkg/store.py", "content": ""},
            }

            guarded = loop._guard_action(
                {
                    "thought_summary": "List again.",
                    "action": "list_files",
                    "target": "workspace",
                    "args": {},
                    "expected_observation": "List files.",
                    "risk": "low",
                },
                state,
            )

        self.assertEqual(guarded["action"], "required_write")
        self.assertEqual(guarded["target"], "workspace/pkg/store.py")

    def test_context_requires_write_after_empty_expected_code_read(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "memory.md").write_text("# Memory\n", encoding="utf-8")
            state = create_initial_state("Implement store")
            state.task_id = "T2"
            state.nodes = [
                {
                    "id": "T2",
                    "title": "Store",
                    "status": "in_progress",
                    "expected_artifacts": ["workspace/pkg/store.py"],
                }
            ]
            state.last_action = {"action": "read", "target": "workspace/pkg/store.py", "args": {}}
            state.last_observation = {
                "ok": True,
                "summary": "Read empty file.",
                "data": {"target": "workspace/pkg/store.py", "content": ""},
            }

            context = ContextBuilder(root).build(state)

        self.assertIn("write with args.mode='overwrite'", context)
        self.assertIn("workspace/pkg/store.py", context)

    def test_context_requires_write_for_empty_expected_code_even_after_rejection(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "memory.md").write_text("# Memory\n", encoding="utf-8")
            store = root / "workspace" / "pkg" / "store.py"
            store.parent.mkdir(parents=True)
            store.write_text("\n", encoding="utf-8")
            state = create_initial_state("Implement store")
            state.task_id = "T2"
            state.nodes = [
                {
                    "id": "T2",
                    "title": "Store",
                    "status": "in_progress",
                    "expected_artifacts": ["workspace/pkg/store.py"],
                }
            ]
            state.last_action = {"action": "required_write", "target": "workspace/pkg/store.py", "args": {}}
            state.last_observation = {
                "ok": False,
                "summary": "Action rejected.",
                "data": {"required_action": "write", "target": "workspace/pkg/store.py"},
            }

            context = ContextBuilder(root).build(state)

        self.assertIn("Next action must be write target='workspace/pkg/store.py'", context)
        self.assertIn("complete implementation content", context)

    def test_empty_python_artifact_is_not_implementation_complete(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            store = root / "workspace" / "pkg" / "store.py"
            store.parent.mkdir(parents=True)
            store.write_text("\n", encoding="utf-8")
            loop = AgentLoop(root=root, task="Implement store", max_steps=1)
            state = create_initial_state("Implement store")
            state.task_id = "T2"
            state.nodes = [
                {
                    "id": "T2",
                    "title": "Store",
                    "status": "in_progress",
                    "expected_artifacts": ["workspace/pkg/store.py"],
                }
            ]

            complete = loop._implementation_files_exist(state)
            target = loop._next_implementation_file_target(state)

        self.assertFalse(complete)
        self.assertEqual(target, "workspace/pkg/store.py")

    def test_guard_allows_overwrite_of_incomplete_python_artifact(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            store = root / "workspace" / "pkg" / "store.py"
            store.parent.mkdir(parents=True)
            store.write_text("\n", encoding="utf-8")
            loop = AgentLoop(root=root, task="Implement store", max_steps=1)
            state = create_initial_state("Implement store")
            state.task_id = "T2"
            state.nodes = [
                {
                    "id": "T2",
                    "title": "Store",
                    "status": "in_progress",
                    "expected_artifacts": ["workspace/pkg/store.py"],
                }
            ]
            state.acceptance_contracts.append(
                {
                    "task_id": "T2",
                    "summary": "Implement store.",
                    "checks": ["python -m unittest discover -s workspace/tests"],
                    "status": "agreed",
                }
            )

            guarded = loop._guard_action(
                {
                    "thought_summary": "Implement store.",
                    "action": "write",
                    "target": "workspace/pkg/store.py",
                    "args": {"mode": "create", "content": "class Store:\n    pass\n"},
                    "expected_observation": "Write store.",
                    "risk": "low",
                },
                state,
            )

        self.assertEqual(guarded["action"], "write")
        self.assertEqual(guarded["args"]["mode"], "overwrite")
        self.assertEqual(guarded["guard_override"], "incomplete_create_to_overwrite")

    def test_guard_blocks_non_write_until_incomplete_python_artifact_is_written(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            store = root / "workspace" / "pkg" / "store.py"
            store.parent.mkdir(parents=True)
            store.write_text("\n", encoding="utf-8")
            loop = AgentLoop(root=root, task="Implement store", max_steps=1)
            state = create_initial_state("Implement store")
            state.task_id = "T2"
            state.nodes = [
                {
                    "id": "T2",
                    "title": "Store",
                    "status": "in_progress",
                    "expected_artifacts": ["workspace/pkg/store.py"],
                }
            ]
            state.acceptance_contracts.append(
                {
                    "task_id": "T2",
                    "summary": "Implement store.",
                    "checks": ["python -m unittest discover -s workspace/tests"],
                    "status": "agreed",
                }
            )

            guarded = loop._guard_action(
                {
                    "thought_summary": "Inspect again.",
                    "action": "list_files",
                    "target": "workspace",
                    "args": {},
                    "expected_observation": "List files.",
                    "risk": "low",
                },
                state,
            )
            observation = loop._execute_action(guarded, state)

        self.assertEqual(guarded["action"], "required_write")
        self.assertEqual(guarded["target"], "workspace/pkg/store.py")
        self.assertFalse(observation.ok)
        self.assertEqual(observation.data["required_action"], "write")

    def test_failed_contract_command_records_pending_repair_targets(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            store = root / "workspace" / "issue_tracker" / "store.py"
            test_store = root / "workspace" / "tests" / "test_store.py"
            store.parent.mkdir(parents=True)
            test_store.parent.mkdir(parents=True)
            store.write_text("class Store:\n    pass\n", encoding="utf-8")
            test_store.write_text("from issue_tracker.store import IssueStore\n", encoding="utf-8")
            loop = AgentLoop(root=root, task="Implement store", max_steps=1)
            state = create_initial_state("Implement store")
            state.task_id = "IT2"
            state.nodes = [
                {
                    "id": "IT2",
                    "title": "Store",
                    "status": "in_progress",
                    "expected_artifacts": [
                        "workspace/issue_tracker/store.py",
                        "workspace/tests/test_store.py",
                    ],
                    "verification_commands": ["python -m unittest discover -s workspace/tests"],
                }
            ]
            state.acceptance_contracts.append(
                {
                    "task_id": "IT2",
                    "summary": "Implement store.",
                    "checks": ["python -m unittest discover -s workspace/tests"],
                    "status": "agreed",
                }
            )
            output = (
                "File \"C:/tmp/workspace/tests/test_store.py\", line 1, in <module>\n"
                "from issue_tracker.store import IssueStore\n"
                "ImportError: cannot import name 'IssueStore' from 'issue_tracker.store'\n"
            )

            loop._update_state(
                state,
                {"action": "bash", "target": "python -m unittest discover -s workspace/tests", "args": {}},
                ToolResult(
                    False,
                    "Command exited with code 1.",
                    {"command": "python -m unittest discover -s workspace/tests", "output": output},
                ),
            )

        self.assertEqual(state.pending_repair["reason"], "failed_acceptance_command")
        self.assertEqual(state.pending_repair["targets"][0], "workspace/issue_tracker/store.py")
        self.assertIn("workspace/tests/test_store.py", state.pending_repair["targets"])
        self.assertEqual(state.pending_repair["repair_targets"], ["workspace/issue_tracker/store.py"])
        self.assertEqual(state.pending_repair["required_reads"][0], "workspace/tests/test_store.py")

    def test_guard_blocks_no_progress_actions_while_pending_repair_exists(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement store", max_steps=1)
            state = create_initial_state("Implement store")
            state.task_id = "IT2"
            state.nodes = [
                {
                    "id": "IT2",
                    "title": "Store",
                    "status": "in_progress",
                    "expected_artifacts": ["workspace/issue_tracker/store.py", "workspace/tests/test_store.py"],
                }
            ]
            state.acceptance_contracts.append(
                {
                    "task_id": "IT2",
                    "summary": "Implement store.",
                    "checks": ["python -m unittest discover -s workspace/tests"],
                    "status": "agreed",
                }
            )
            state.pending_repair = {
                "reason": "failed_acceptance_command",
                "command": "python -m unittest discover -s workspace/tests",
                "summary": "Command exited with code 1.",
                "targets": ["workspace/issue_tracker/store.py", "workspace/tests/test_store.py"],
            }

            guarded = loop._guard_action(
                {
                    "thought_summary": "Run tests again.",
                    "action": "bash",
                    "target": "python -m unittest discover -s workspace/tests",
                    "args": {},
                    "expected_observation": "Tests pass.",
                    "risk": "low",
                },
                state,
            )
            observation = loop._execute_action(guarded, state)

        self.assertEqual(guarded["action"], "required_repair")
        self.assertEqual(guarded["target"], "workspace/issue_tracker/store.py")
        self.assertEqual(guarded["guard_override"], "failed_contract_requires_repair")
        self.assertFalse(observation.ok)
        self.assertEqual(observation.data["required_action"], "write_or_edit")

    def test_guard_forces_failed_test_read_before_repair(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement store", max_steps=1)
            state = create_initial_state("Implement store")
            state.task_id = "IT2"
            state.nodes = [
                {
                    "id": "IT2",
                    "title": "Store",
                    "status": "in_progress",
                    "expected_artifacts": ["workspace/issue_tracker/store.py", "workspace/tests/test_store.py"],
                }
            ]
            state.pending_repair = {
                "reason": "failed_acceptance_command",
                "command": "python -m unittest discover -s workspace/tests",
                "summary": "Command exited with code 1.",
                "targets": ["workspace/issue_tracker/store.py", "workspace/tests/test_store.py"],
                "required_reads": ["workspace/tests/test_store.py", "workspace/issue_tracker/store.py"],
                "read_targets": [],
                "repaired_targets": [],
            }

            guarded = loop._guard_action(
                {
                    "thought_summary": "Guess a repair.",
                    "action": "write",
                    "target": "workspace/issue_tracker/store.py",
                    "args": {"mode": "overwrite", "content": "class Store:\n    pass\n"},
                    "expected_observation": "Repair store.",
                    "risk": "medium",
                },
                state,
            )

        self.assertEqual(guarded["action"], "read")
        self.assertEqual(guarded["target"], "workspace/tests/test_store.py")
        self.assertEqual(guarded["guard_override"], "failed_contract_requires_read_before_repair")

    def test_guard_prefers_implementation_repair_over_test_rewrite(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement store", max_steps=1)
            state = create_initial_state("Implement store")
            state.task_id = "IT2"
            state.nodes = [
                {
                    "id": "IT2",
                    "title": "Store",
                    "status": "in_progress",
                    "expected_artifacts": ["workspace/issue_tracker/store.py", "workspace/tests/test_store.py"],
                }
            ]
            state.pending_repair = {
                "reason": "failed_acceptance_command",
                "command": "python -m unittest discover -s workspace/tests",
                "summary": "Command exited with code 1.",
                "targets": ["workspace/tests/test_store.py", "workspace/issue_tracker/store.py"],
                "repair_targets": ["workspace/issue_tracker/store.py"],
                "required_reads": ["workspace/tests/test_store.py", "workspace/issue_tracker/store.py"],
                "read_targets": ["workspace/tests/test_store.py", "workspace/issue_tracker/store.py"],
                "repaired_targets": [],
            }

            guarded = loop._guard_action(
                {
                    "thought_summary": "Rewrite the test.",
                    "action": "write",
                    "target": "workspace/tests/test_store.py",
                    "args": {"mode": "overwrite", "content": "import unittest\n"},
                    "expected_observation": "Test rewritten.",
                    "risk": "medium",
                },
                state,
            )
            observation = loop._execute_action(guarded, state)

        self.assertEqual(guarded["action"], "required_repair")
        self.assertEqual(guarded["target"], "workspace/issue_tracker/store.py")
        self.assertEqual(guarded["args"]["targets"], ["workspace/issue_tracker/store.py"])
        self.assertFalse(observation.ok)

    def test_failed_contract_preserves_repair_reads_across_retest_failure(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement store", max_steps=1)
            state = create_initial_state("Implement store")
            state.task_id = "IT2"
            state.nodes = [
                {
                    "id": "IT2",
                    "title": "Store",
                    "status": "in_progress",
                    "expected_artifacts": ["workspace/issue_tracker/store.py", "workspace/tests/test_store.py"],
                    "implementation_artifacts": ["workspace/issue_tracker/store.py"],
                    "acceptance_artifacts": ["workspace/tests/test_store.py"],
                    "frozen_acceptance_artifacts": ["workspace/tests/test_store.py"],
                    "verification_commands": ["python -m unittest discover -s workspace/tests"],
                }
            ]
            state.acceptance_contracts.append(
                {
                    "task_id": "IT2",
                    "summary": "Implement store.",
                    "checks": ["python -m unittest discover -s workspace/tests"],
                    "status": "agreed",
                }
            )
            state.pending_repair = {
                "reason": "failed_acceptance_command",
                "command": "python -m unittest discover -s workspace/tests",
                "summary": "Command exited with code 1.",
                "targets": ["workspace/tests/test_store.py", "workspace/issue_tracker/store.py"],
                "repair_targets": ["workspace/issue_tracker/store.py"],
                "required_reads": ["workspace/tests/test_store.py", "workspace/issue_tracker/store.py"],
                "read_targets": ["workspace/tests/test_store.py", "workspace/issue_tracker/store.py"],
                "repaired_targets": ["workspace/issue_tracker/store.py"],
            }
            output = (
                "TypeError: IssueStore.create() got an unexpected keyword argument 'title'\n"
                "File \"C:/tmp/workspace/tests/test_store.py\", line 22\n"
            )

            loop._update_state(
                state,
                {"action": "bash", "target": "python -m unittest discover -s workspace/tests", "args": {}},
                ToolResult(
                    False,
                    "Command exited with code 1.",
                    {"command": "python -m unittest discover -s workspace/tests", "output": output},
                ),
            )
            guarded = loop._guard_action(
                {
                    "thought_summary": "Read the test again.",
                    "action": "read",
                    "target": "workspace/tests/test_store.py",
                    "args": {},
                    "expected_observation": "Read test.",
                    "risk": "low",
                },
                state,
            )

        self.assertEqual(
            state.pending_repair["read_targets"],
            ["workspace/tests/test_store.py", "workspace/issue_tracker/store.py"],
        )
        self.assertEqual(state.pending_repair["repaired_targets"], [])
        self.assertEqual(guarded["action"], "required_repair")
        self.assertEqual(guarded["target"], "workspace/issue_tracker/store.py")

    def test_guard_blocks_frozen_acceptance_test_write(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement store", max_steps=1)
            state = create_initial_state("Implement store")
            state.task_id = "IT2"
            state.nodes = [
                {
                    "id": "IT2",
                    "title": "Store",
                    "status": "in_progress",
                    "expected_artifacts": ["workspace/issue_tracker/store.py", "workspace/tests/test_store.py"],
                    "implementation_artifacts": ["workspace/issue_tracker/store.py"],
                    "worker_test_artifacts": ["workspace/tests/test_store.py"],
                    "acceptance_artifacts": ["workspace/tests/test_store.py"],
                    "frozen_acceptance_artifacts": ["workspace/tests/test_store.py"],
                    "test_policy": {"acceptance_tests_mutable_by_worker": False},
                }
            ]
            state.acceptance_contracts.append(
                {"task_id": "IT2", "summary": "Implement store.", "checks": ["unit tests"], "status": "agreed"}
            )

            guarded = loop._guard_action(
                {
                    "thought_summary": "Rewrite acceptance test.",
                    "action": "write",
                    "target": "workspace/tests/test_store.py",
                    "args": {"mode": "overwrite", "content": "import unittest\n"},
                    "expected_observation": "Test rewritten.",
                    "risk": "medium",
                },
                state,
            )

        self.assertEqual(guarded["action"], "required_repair")
        self.assertEqual(guarded["target"], "workspace/issue_tracker/store.py")
        self.assertEqual(guarded["guard_override"], "frozen_acceptance_test_write_blocked")

    def test_guard_allows_worker_test_write_when_not_frozen(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement store", max_steps=1)
            state = create_initial_state("Implement store")
            state.task_id = "IT2"
            state.nodes = [
                {
                    "id": "IT2",
                    "title": "Store",
                    "status": "in_progress",
                    "expected_artifacts": ["workspace/issue_tracker/store.py", "workspace/tests/test_store.py"],
                    "implementation_artifacts": ["workspace/issue_tracker/store.py"],
                    "worker_test_artifacts": ["workspace/tests/test_store.py"],
                    "acceptance_artifacts": [],
                    "frozen_acceptance_artifacts": [],
                }
            ]
            state.acceptance_contracts.append(
                {"task_id": "IT2", "summary": "Implement store.", "checks": ["unit tests"], "status": "agreed"}
            )
            action = {
                "thought_summary": "Create worker-owned unit test.",
                "action": "write",
                "target": "workspace/tests/test_store.py",
                "args": {"mode": "overwrite", "content": "import unittest\n"},
                "expected_observation": "Test written.",
                "risk": "medium",
            }

            guarded = loop._guard_action(action, state)

        self.assertEqual(guarded, action)

    def test_test_only_failure_still_adds_implementation_repair_target(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement store", max_steps=1)
            state = create_initial_state("Implement store")
            state.task_id = "IT2"
            state.nodes = [
                {
                    "id": "IT2",
                    "title": "Store",
                    "status": "in_progress",
                    "expected_artifacts": ["workspace/issue_tracker/store.py", "workspace/tests/test_store.py"],
                    "verification_commands": ["python -m unittest discover -s workspace/tests"],
                }
            ]
            state.acceptance_contracts.append(
                {
                    "task_id": "IT2",
                    "summary": "Implement store.",
                    "checks": ["python -m unittest discover -s workspace/tests"],
                    "status": "agreed",
                }
            )
            output = (
                "File \"C:/tmp/workspace/tests/test_store.py\", line 4, in <module>\n"
                "from issue_tracker.store import IssueStore\n"
                "ModuleNotFoundError: No module named 'issue_tracker'\n"
            )

            loop._update_state(
                state,
                {"action": "bash", "target": "python -m unittest discover -s workspace/tests", "args": {}},
                ToolResult(
                    False,
                    "Command exited with code 1.",
                    {"command": "python -m unittest discover -s workspace/tests", "output": output},
                ),
            )

        self.assertIn("workspace/tests/test_store.py", state.pending_repair["targets"])
        self.assertIn("workspace/issue_tracker/store.py", state.pending_repair["targets"])
        self.assertEqual(state.pending_repair["repair_targets"], ["workspace/issue_tracker/store.py"])

    def test_pending_repair_is_cleared_only_after_contract_command_passes(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement store", max_steps=1)
            state = create_initial_state("Implement store")
            state.task_id = "IT2"
            state.nodes = [
                {
                    "id": "IT2",
                    "title": "Store",
                    "status": "in_progress",
                    "expected_artifacts": ["workspace/issue_tracker/store.py"],
                    "verification_commands": ["python -m unittest discover -s workspace/tests"],
                }
            ]
            state.acceptance_contracts.append(
                {
                    "task_id": "IT2",
                    "summary": "Implement store.",
                    "checks": ["python -m unittest discover -s workspace/tests"],
                    "status": "agreed",
                }
            )
            state.pending_repair = {
                "reason": "failed_acceptance_command",
                "command": "python -m unittest discover -s workspace/tests",
                "summary": "Command exited with code 1.",
                "targets": ["workspace/issue_tracker/store.py"],
                "required_reads": ["workspace/issue_tracker/store.py"],
                "read_targets": ["workspace/issue_tracker/store.py"],
                "repaired_targets": [],
            }

            loop._update_state(
                state,
                {"action": "write", "target": "workspace/issue_tracker/store.py", "args": {"mode": "overwrite"}},
                ToolResult(True, "Wrote store.", {"target": "workspace/issue_tracker/store.py"}),
            )
            still_pending = bool(state.pending_repair)
            guarded = loop._guard_action(
                {
                    "thought_summary": "Inspect again.",
                    "action": "list_files",
                    "target": "workspace",
                    "args": {},
                    "expected_observation": "List files.",
                    "risk": "low",
                },
                state,
            )
            loop._update_state(
                state,
                {
                    "action": "bash",
                    "target": "python -m unittest discover -s workspace/tests 2>&1",
                    "args": {},
                },
                ToolResult(
                    True,
                    "Command exited with code 0.",
                    {"command": "python -m unittest discover -s workspace/tests 2>&1", "output": "OK"},
                ),
            )

        self.assertTrue(still_pending)
        self.assertEqual(guarded["action"], "bash")
        self.assertEqual(guarded["target"], "python -m unittest discover -s workspace/tests")
        self.assertEqual(guarded["guard_override"], "failed_contract_repair_to_retest")
        self.assertEqual(state.pending_repair, {})

    def test_pending_repair_write_create_mode_becomes_overwrite(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            store = root / "workspace" / "issue_tracker" / "store.py"
            store.parent.mkdir(parents=True)
            store.write_text("class Store:\n    pass\n", encoding="utf-8")
            loop = AgentLoop(root=root, task="Implement store", max_steps=1)
            state = create_initial_state("Implement store")
            state.task_id = "IT2"
            state.nodes = [
                {
                    "id": "IT2",
                    "title": "Store",
                    "status": "in_progress",
                    "expected_artifacts": ["workspace/issue_tracker/store.py"],
                }
            ]
            state.acceptance_contracts.append(
                {
                    "task_id": "IT2",
                    "summary": "Implement store.",
                    "checks": ["python -m unittest discover -s workspace/tests"],
                    "status": "agreed",
                }
            )
            state.pending_repair = {
                "reason": "failed_acceptance_command",
                "command": "python -m unittest discover -s workspace/tests",
                "summary": "Command exited with code 1.",
                "targets": ["workspace/issue_tracker/store.py"],
            }

            guarded = loop._guard_action(
                {
                    "thought_summary": "Repair store.",
                    "action": "write",
                    "target": "workspace/issue_tracker/store.py",
                    "args": {"mode": "create", "content": "class IssueStore:\n    pass\n"},
                    "expected_observation": "Store repaired.",
                    "risk": "medium",
                },
                state,
            )

        self.assertEqual(guarded["action"], "write")
        self.assertEqual(guarded["args"]["mode"], "overwrite")
        self.assertEqual(guarded["guard_override"], "failed_contract_create_to_overwrite")

    def test_context_mentions_pending_repair_as_required_next_action(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            state = create_initial_state("Implement store")
            state.task_id = "IT2"
            state.nodes = [
                {
                    "id": "IT2",
                    "title": "Store",
                    "status": "in_progress",
                    "expected_artifacts": ["workspace/issue_tracker/store.py"],
                }
            ]
            state.pending_repair = {
                "reason": "failed_acceptance_command",
                "summary": "Command exited with code 1.",
                "output": "ImportError: cannot import name IssueStore",
                "targets": ["workspace/issue_tracker/store.py"],
            }

            context = ContextBuilder(root).build(state)

        self.assertIn("Next action must be write or edit", context)
        self.assertIn("workspace/issue_tracker/store.py", context)

    def test_guard_runs_next_contract_command_before_verify(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Create skeleton", max_steps=1)
            state = create_initial_state("Create skeleton")
            state.task_id = "T1"
            state.nodes = [
                {
                    "id": "T1",
                    "title": "Skeleton",
                    "status": "in_progress",
                    "expected_artifacts": ["README.md"],
                }
            ]
            first = "python -c \"assert True\""
            second = "python -c \"assert 'CLI' in open('README.md').read()\""
            state.acceptance_contracts.append(
                {
                    "task_id": "T1",
                    "summary": "Create skeleton.",
                    "checks": [first, second],
                    "status": "agreed",
                }
            )
            (root / "README.md").write_text("# Project\n\nCLI commands\n", encoding="utf-8")
            state.last_action = {"action": "bash", "target": first, "args": {}}
            state.last_observation = {"ok": True, "summary": "Command exited with code 0.", "data": {"command": first}}

            guarded = loop._guard_action(
                {
                    "thought_summary": "Inspect again.",
                    "action": "list_files",
                    "target": ".",
                    "args": {},
                    "expected_observation": "List files.",
                    "risk": "low",
                },
                state,
            )

        self.assertEqual(guarded["action"], "bash")
        self.assertEqual(guarded["target"], second)
        self.assertEqual(guarded["guard_override"], "implementation_files_exist_to_test")

    def test_guard_repairs_document_after_failed_contract_command(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            readme = root / "workspace" / "README.md"
            readme.parent.mkdir(parents=True)
            readme.write_text("# Project\n", encoding="utf-8")
            loop = AgentLoop(root=root, task="Create skeleton", max_steps=1)
            state = create_initial_state("Create skeleton")
            state.task_id = "T1"
            state.user_goal = "T1: Create project skeleton"
            state.acceptance_criteria = ["README describes intended CLI commands."]
            state.nodes = [
                {
                    "id": "T1",
                    "title": "Skeleton",
                    "status": "in_progress",
                    "expected_artifacts": ["workspace/README.md"],
                }
            ]
            command = "python -c \"assert 'CLI' in open('workspace/README.md').read()\""
            state.acceptance_contracts.append(
                {
                    "task_id": "T1",
                    "summary": "Create skeleton.",
                    "checks": [command],
                    "status": "agreed",
                }
            )
            state.last_action = {"action": "bash", "target": command, "args": {}}
            state.last_observation = {
                "ok": False,
                "summary": "Command exited with code 1.",
                "data": {"command": command, "output": "AssertionError"},
            }

            guarded = loop._guard_action(
                {
                    "thought_summary": "Inspect again.",
                    "action": "list_files",
                    "target": "workspace",
                    "args": {},
                    "expected_observation": "List files.",
                    "risk": "low",
                },
                state,
            )

        self.assertEqual(guarded["action"], "write")
        self.assertEqual(guarded["target"], "workspace/README.md")
        self.assertEqual(guarded["args"]["mode"], "overwrite")
        self.assertIn("CLI command", guarded["args"]["content"])
        self.assertEqual(guarded["guard_override"], "failed_contract_to_artifact_repair")

    def test_readme_initial_content_mentions_commands_and_criteria(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Create skeleton", max_steps=1)
            state = create_initial_state("Create skeleton")
            state.user_goal = "T1: Create project skeleton"
            state.acceptance_criteria = ["README describes intended CLI commands."]

            content = loop._initial_content_for_target("workspace/README.md", state)

        self.assertIn("Commands", content)
        self.assertIn("CLI command", content)
        self.assertIn("README describes intended CLI commands.", content)

    def test_guard_rewrites_when_contract_files_exist_to_smoke(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            readme = root / "eval" / "benchmarks" / "issue_tracker" / "workspace" / "README.md"
            package_init = root / "eval" / "benchmarks" / "issue_tracker" / "workspace" / "issue_tracker" / "__init__.py"
            tests_init = root / "eval" / "benchmarks" / "issue_tracker" / "workspace" / "tests" / "__init__.py"
            readme.parent.mkdir(parents=True)
            package_init.parent.mkdir(parents=True)
            tests_init.parent.mkdir(parents=True)
            readme.write_text("# Issue Tracker\n", encoding="utf-8")
            package_init.write_text('"""Package."""\n', encoding="utf-8")
            tests_init.write_text('"""Tests."""\n', encoding="utf-8")
            loop = AgentLoop(root=root, task="Implement a feature", max_steps=1)
            state = create_initial_state("Implement a feature")
            smoke = "python -c \"assert True\""
            state.acceptance_contracts.append(
                {
                    "task_id": "T1",
                    "summary": "Create skeleton.",
                    "checks": [
                        "File eval/benchmarks/issue_tracker/workspace/README.md exists.",
                        "File eval/benchmarks/issue_tracker/workspace/issue_tracker/__init__.py exists.",
                        "File eval/benchmarks/issue_tracker/workspace/tests/__init__.py exists.",
                        f"Smoke test: {smoke}",
                    ],
                    "status": "agreed",
                }
            )

            guarded = loop._guard_action(
                {
                    "thought_summary": "Inspect again.",
                    "action": "list_files",
                    "target": "eval/benchmarks/issue_tracker/workspace",
                    "args": {},
                    "expected_observation": "List files.",
                    "risk": "low",
                },
                state,
            )

        self.assertEqual(guarded["action"], "bash")
        self.assertEqual(guarded["target"], smoke)
        self.assertEqual(guarded["guard_override"], "contract_files_exist_to_smoke")

    def test_orchestrator_selects_next_task_after_verify_pass(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            (root / "tasks.json").write_text(
                json.dumps(
                    {
                        "tasks": [
                            {"id": "T1", "status": "in_progress", "priority": 1, "depends_on": []},
                            {"id": "T2", "status": "pending", "priority": 2, "depends_on": ["T1"]},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            loop = AgentLoop(root=root, task="Implement a feature", max_steps=1)
            state = create_initial_state("Implement a feature")
            state.task_id = "T1"
            state.nodes = [{"id": "T1", "title": "First", "status": "in_progress", "evidence": ["scheduled"]}]

            observation = loop._execute_action({"action": "verify", "target": "default", "args": {}}, state)
            loop._update_state(state, {"action": "verify", "target": "default", "args": {}}, observation)
            loop._apply_orchestrator_selection(state)
            data = json.loads((root / "tasks.json").read_text(encoding="utf-8"))

        self.assertTrue(observation.ok)
        self.assertEqual(state.task_id, "T2")
        self.assertEqual(state.nodes[0]["status"], "in_progress")
        self.assertEqual(data["tasks"][0]["status"], "completed")
        self.assertEqual(data["tasks"][1]["status"], "in_progress")

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

    def test_resume_resets_session_budget_flags(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            (root / "state" / "current_task.json").write_text(
                json.dumps(
                    {
                        "task_id": "current",
                        "user_goal": "Resume task",
                        "acceptance_criteria": [],
                        "nodes": [],
                        "iterations": 1,
                        "last_action": {},
                        "last_observation": {},
                        "evidence_sources": [],
                        "acceptance_contracts": [],
                        "session_budget_tokens": 16000,
                        "handoff_threshold": 0.7,
                        "session_used_tokens": 12000,
                        "handoff_ready": True,
                    }
                ),
                encoding="utf-8",
            )
            loop = AgentLoop(root=root, task="Resume task", max_steps=1, resume=True)

            state = loop._load_or_create_state()

        self.assertEqual(state.session_used_tokens, 0)
        self.assertFalse(state.handoff_ready)

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
            payload = json.loads((root / "state" / "handoff_payload.json").read_text(encoding="utf-8"))

        self.assertIn("# Worker Session Handoff", handoff)
        self.assertIn("## 2. Session Budget", handoff)
        self.assertIn("## 4. Handoff Data References", handoff)
        self.assertIn("structured_payload: state/handoff_payload.json", handoff)
        self.assertIn("## 14. Resume Instructions", handoff)
        self.assertIn("threshold_tokens: 70", handoff)
        self.assertEqual(payload["schema"], "long-agent.handoff-payload.v1")
        self.assertTrue(payload["session_budget"]["handoff_ready"])

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
        self.assertIn('"task_id": "current"', report)

    def test_hidden_acceptance_config_can_pass_in_temp_project(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "eval").mkdir()
            (root / "eval" / "hidden_acceptance.json").write_text(
                json.dumps({"command": ["python", "-c", "import sys; sys.exit(0)"]}),
                encoding="utf-8",
            )

            result = ProjectTerminator(root)._run_hidden_acceptance()

        self.assertTrue(result["ok"])
        self.assertTrue(result["configured"])

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

    def test_metrics_reads_pretty_multiline_trace_events(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            trace = Path(tmp) / "run.jsonl"
            events = [
                {
                    "step": 1,
                    "action": {"action": "read"},
                    "observation": {"ok": True, "summary": "Read.", "data": {}},
                },
                {
                    "step": 2,
                    "action": {"action": "answer"},
                    "observation": {"ok": True, "summary": "Answered.", "data": {}},
                },
            ]
            trace.write_text("".join(json.dumps(event, indent=2) + "\n" for event in events), encoding="utf-8")

            loaded = load_events(trace)
            summary = summarize(trace)

        self.assertEqual(len(loaded), 2)
        self.assertEqual(summary["steps"], 2)
        self.assertEqual(summary["actions"]["answer"], 1)

    def test_metrics_reports_long_running_harness_fields(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "run.jsonl"
            tasks = root / "tasks.json"
            tasks.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {"id": "T1", "status": "completed"},
                            {"id": "T2", "status": "blocked"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            events = [
                {
                    "action": {"action": "contract"},
                    "observation": {"ok": False, "summary": "Acceptance contract rejected.", "data": {}},
                    "session_used_tokens": 10,
                    "handoff_ready": False,
                    "nodes": [{"id": "T1", "status": "in_progress"}],
                },
                {
                    "action": {"action": "verify"},
                    "observation": {"ok": False, "summary": "Verifier failed.", "data": {}},
                    "session_used_tokens": 20,
                    "handoff_ready": True,
                    "nodes": [{"id": "T1", "status": "in_progress"}],
                },
                {
                    "action": {"action": "skill"},
                    "observation": {"ok": True, "summary": "Skill promoted.", "data": {}},
                    "session_used_tokens": 30,
                    "handoff_ready": False,
                    "nodes": [{"id": "T1", "status": "completed"}],
                },
            ]
            trace.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")

            summary = summarize(trace, tasks)

        self.assertEqual(summary["contract_rejections"], 1)
        self.assertEqual(summary["verifier_failures"], 1)
        self.assertEqual(summary["skill_promotions"], 1)
        self.assertEqual(summary["handoff_count"], 1)
        self.assertEqual(summary["no_progress_sessions"], 0)
        self.assertEqual(summary["max_session_used_tokens"], 30)
        self.assertEqual(summary["completed_tasks"], 1)
        self.assertEqual(summary["blocked_tasks"], 1)


if __name__ == "__main__":
    unittest.main()
