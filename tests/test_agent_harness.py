from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
import uuid
from pathlib import Path

from agent.context import ContextBuilder
from agent.llm import parse_action_json, validate_action
from agent.loop import AgentLoop
from agent.main import build_parser, infer_benchmark_id, resolve_log_path
from agent.memory_retrieval import (
    MemoryRetriever,
    scan_memory_headers,
    truncate_entrypoint_content,
)
from agent.orchestrator import Orchestrator, count_unlocked_tasks, select_current_task
from agent.planner import create_initial_state, create_initializer_state, validate_generated_task_graph, validate_initializer_script
from agent.prompts import MAIN_AGENT_SYSTEM_PROMPT
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
    def test_initializer_script_validator_rejects_python_source_and_state_workspace(self) -> None:
        errors = validate_initializer_script(
            "#!/usr/bin/env python3\n"
            "import os\n"
            "WORKSPACE_ROOT = 'state/benchmarks/todo_counter/workspace'\n"
            "os.makedirs(WORKSPACE_ROOT, exist_ok=True)\n",
            expected_workspace_root="eval/benchmarks/todo_counter/workspace",
            standard_library_only=True,
        )

        combined = " ".join(errors)
        self.assertIn("#!/usr/bin/env sh", combined)
        self.assertIn("Python source code", combined)
        self.assertIn("state/benchmarks", combined)

    def test_generated_task_validator_rejects_semantic_quality_gaps(self) -> None:
        errors = validate_generated_task_graph(
            {
                "tasks": [
                    {
                        "id": "T1",
                        "title": "Implement summary behavior",
                        "priority": 1,
                        "depends_on": [],
                        "status": "pending",
                        "acceptance_criteria": ["Tests pass with pytest."],
                        "expected_artifacts": [],
                        "implementation_artifacts": [],
                        "worker_test_artifacts": [],
                        "acceptance_artifacts": [],
                        "frozen_acceptance_artifacts": [],
                        "verification_commands": ["echo 'not implemented yet'"],
                    }
                ]
            },
            expected_workspace_root="eval/benchmarks/todo_counter/workspace",
            standard_library_only=True,
        )

        combined = " ".join(errors)
        self.assertIn("implementation_artifacts is empty", combined)
        self.assertIn("standard-library-only", combined)
        self.assertIn("placeholder/no-op", combined)

    def test_benchmark_context_does_not_load_repository_task_graph(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tasks.json").write_text(
                json.dumps({"tasks": [{"id": "ROOT-ONLY-TASK"}]}),
                encoding="utf-8",
            )
            state_dir = root / "state" / "benchmarks" / "todo_counter"
            state_dir.mkdir(parents=True)
            (state_dir / "project_spec.md").write_text("# Benchmark Spec\n", encoding="utf-8")

            context = ContextBuilder(root, state_dir=state_dir).build(create_initial_state("Benchmark"))

        self.assertIn("# Benchmark Spec", context)
        self.assertNotIn("ROOT-ONLY-TASK", context)

    def test_initializer_prompt_requires_integer_priority_with_complete_example(self) -> None:
        self.assertIn("priority MUST be an integer", MAIN_AGENT_SYSTEM_PROMPT)
        self.assertIn('"priority":1', MAIN_AGENT_SYSTEM_PROMPT)
        self.assertIn('"implementation_artifacts"', MAIN_AGENT_SYSTEM_PROMPT)
        self.assertIn('"verification_commands"', MAIN_AGENT_SYSTEM_PROMPT)
        self.assertIn("commands run from the repository root", MAIN_AGENT_SYSTEM_PROMPT)
        self.assertIn("Read narrowly instead of preloading the repository", MAIN_AGENT_SYSTEM_PROMPT)
        self.assertIn("continue read has_more pages", MAIN_AGENT_SYSTEM_PROMPT)
        self.assertIn("Avoid Unix-only commands", MAIN_AGENT_SYSTEM_PROMPT)

    def test_generated_task_validator_requires_workspace_import_bootstrap(self) -> None:
        workspace = "eval/benchmarks/todo_counter/workspace"
        graph = {
            "tasks": [
                {
                    "id": "T3",
                    "title": "Implement CLI module",
                    "priority": 1,
                    "depends_on": [],
                    "status": "pending",
                    "acceptance_criteria": ["CLI runs as a module."],
                    "expected_artifacts": [f"{workspace}/todo_counter/cli.py"],
                    "implementation_artifacts": [f"{workspace}/todo_counter/cli.py"],
                    "worker_test_artifacts": [],
                    "acceptance_artifacts": [],
                    "frozen_acceptance_artifacts": [],
                    "verification_commands": [
                        "python -c \"import subprocess, sys; subprocess.run([sys.executable, '-m', 'todo_counter.cli', 'todos.txt'])\""
                    ],
                }
            ]
        }

        errors = validate_generated_task_graph(graph, expected_workspace_root=workspace)

        self.assertIn("without configuring", " ".join(errors))

    def test_generated_task_validator_accepts_workspace_subprocess_cwd(self) -> None:
        workspace = "eval/benchmarks/todo_counter/workspace"
        graph = {
            "tasks": [
                {
                    "id": "T3",
                    "title": "Implement CLI module",
                    "priority": 1,
                    "depends_on": [],
                    "status": "pending",
                    "acceptance_criteria": ["CLI runs as a module."],
                    "expected_artifacts": [f"{workspace}/todo_counter/cli.py"],
                    "implementation_artifacts": [f"{workspace}/todo_counter/cli.py"],
                    "worker_test_artifacts": [],
                    "acceptance_artifacts": [],
                    "frozen_acceptance_artifacts": [],
                    "verification_commands": [
                        f"python -c \"import subprocess, sys; subprocess.run([sys.executable, '-m', 'todo_counter.cli', 'todos.txt'], cwd='{workspace}')\""
                    ],
                }
            ]
        }

        errors = validate_generated_task_graph(graph, expected_workspace_root=workspace)

        self.assertFalse(any("without configuring" in error for error in errors))

    def test_generated_task_validator_rejects_nested_chdir_and_subprocess_cwd(self) -> None:
        workspace = "eval/benchmarks/todo_counter/workspace"
        graph = {
            "tasks": [
                {
                    "id": "T3",
                    "title": "Implement CLI module",
                    "priority": 1,
                    "depends_on": [],
                    "status": "pending",
                    "acceptance_criteria": ["CLI runs as a module."],
                    "criterion_command_map": {
                        "CLI runs as a module.": [
                            f"python -c \"import os, subprocess, sys; os.chdir('{workspace}'); subprocess.run([sys.executable, '-m', 'todo_counter.cli'], cwd='{workspace}')\""
                        ]
                    },
                    "expected_artifacts": [f"{workspace}/todo_counter/cli.py"],
                    "implementation_artifacts": [f"{workspace}/todo_counter/cli.py"],
                    "worker_test_artifacts": [],
                    "acceptance_artifacts": [],
                    "frozen_acceptance_artifacts": [],
                    "verification_commands": [
                        f"python -c \"import os, subprocess, sys; os.chdir('{workspace}'); subprocess.run([sys.executable, '-m', 'todo_counter.cli'], cwd='{workspace}')\""
                    ],
                }
            ]
        }

        errors = validate_generated_task_graph(graph, expected_workspace_root=workspace)

        self.assertIn("mixes os.chdir", " ".join(errors))

    def test_compound_echo_verification_command_is_not_treated_as_placeholder(self) -> None:
        errors = validate_generated_task_graph(
            {
                "tasks": [
                    {
                        "id": "T1",
                        "title": "Run final verification",
                        "priority": 1,
                        "depends_on": [],
                        "status": "pending",
                        "acceptance_criteria": ["Command performs a real assertion."],
                        "expected_artifacts": [],
                        "implementation_artifacts": [],
                        "worker_test_artifacts": [],
                        "acceptance_artifacts": [],
                        "frozen_acceptance_artifacts": [],
                        "verification_commands": ["echo setup && python -c \"assert 2 + 2 == 4\""],
                    }
                ]
            }
        )

        self.assertFalse(any("placeholder/no-op" in error for error in errors))
        self.assertTrue(any("not cross-platform" in error for error in errors))

    def test_generated_task_validator_requires_complete_criterion_command_map(self) -> None:
        first = "python -c \"assert 1 + 1 == 2\""
        second = "python -c \"assert 'ok'.upper() == 'OK'\""
        errors = validate_generated_task_graph(
            {
                "tasks": [
                    {
                        "id": "T1",
                        "title": "Verify behavior",
                        "priority": 1,
                        "depends_on": [],
                        "status": "pending",
                        "acceptance_criteria": ["Arithmetic works.", "Text works."],
                        "criterion_command_map": {"Arithmetic works.": [first]},
                        "expected_artifacts": [],
                        "implementation_artifacts": [],
                        "worker_test_artifacts": [],
                        "acceptance_artifacts": [],
                        "frozen_acceptance_artifacts": [],
                        "verification_commands": [first, second],
                    }
                ]
            }
        )

        combined = " ".join(errors)
        self.assertIn("missing acceptance criteria", combined)
        self.assertIn("does not assign verification commands", combined)

    def test_generated_task_validator_requires_imported_module_artifact_path(self) -> None:
        errors = validate_generated_task_graph(
            {
                "tasks": [
                    {
                        "id": "T1",
                        "title": "Implement core module",
                        "priority": 1,
                        "depends_on": [],
                        "status": "pending",
                        "acceptance_criteria": ["Core import works."],
                        "expected_artifacts": ["eval/benchmarks/todo_counter/workspace/core.py"],
                        "implementation_artifacts": ["eval/benchmarks/todo_counter/workspace/core.py"],
                        "worker_test_artifacts": [],
                        "acceptance_artifacts": [],
                        "frozen_acceptance_artifacts": [],
                        "verification_commands": [
                            "python -c \"import sys; sys.path.insert(0, 'eval/benchmarks/todo_counter/workspace'); from todo_counter.core import parse_todos\""
                        ],
                    }
                ]
            },
            expected_workspace_root="eval/benchmarks/todo_counter/workspace",
        )

        self.assertIn(
            "eval/benchmarks/todo_counter/workspace/todo_counter/core.py",
            " ".join(errors),
        )

    def test_generated_task_validator_requires_subprocess_module_artifact_path(self) -> None:
        errors = validate_generated_task_graph(
            {
                "tasks": [
                    {
                        "id": "T1",
                        "title": "Implement CLI module",
                        "priority": 1,
                        "depends_on": [],
                        "status": "pending",
                        "acceptance_criteria": ["CLI runs as a module."],
                        "expected_artifacts": ["eval/benchmarks/todo_counter/workspace/cli.py"],
                        "implementation_artifacts": ["eval/benchmarks/todo_counter/workspace/cli.py"],
                        "worker_test_artifacts": [],
                        "acceptance_artifacts": [],
                        "frozen_acceptance_artifacts": [],
                        "verification_commands": [
                            "python -c \"import subprocess, sys; subprocess.run([sys.executable, '-m', 'todo_counter.cli', 'todos.txt'])\""
                        ],
                    }
                ]
            },
            expected_workspace_root="eval/benchmarks/todo_counter/workspace",
        )

        self.assertIn(
            "eval/benchmarks/todo_counter/workspace/todo_counter/cli.py",
            " ".join(errors),
        )

    def test_generated_task_validator_rejects_invalid_python_c_syntax(self) -> None:
        errors = validate_generated_task_graph(
            {
                "tasks": [
                    {
                        "id": "T1",
                        "title": "Implement CLI module",
                        "priority": 1,
                        "depends_on": [],
                        "status": "pending",
                        "acceptance_criteria": ["CLI behavior is checked."],
                        "expected_artifacts": ["eval/benchmarks/todo_counter/workspace/todo_counter/cli.py"],
                        "implementation_artifacts": ["eval/benchmarks/todo_counter/workspace/todo_counter/cli.py"],
                        "worker_test_artifacts": [],
                        "acceptance_artifacts": [],
                        "frozen_acceptance_artifacts": [],
                        "verification_commands": [
                            "python -c \"import tempfile; with tempfile.NamedTemporaryFile() as f: pass\""
                        ],
                    }
                ]
            },
            expected_workspace_root="eval/benchmarks/todo_counter/workspace",
        )

        self.assertIn("invalid python -c syntax", " ".join(errors))

    def test_initializer_repeat_count_recovers_from_pretty_trace(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(root=root, task="INIT", max_steps=1, benchmark_id="todo_counter")
            loop._ensure_state_files()
            errors_by_step = [
                ["tasks[0].priority must be an integer."],
                ["tasks[0].priority must be an integer.", "tasks[1].priority must be an integer."],
                ["tasks[0].priority must be an integer.", "tasks[2].priority must be an integer."],
            ]
            trace = loop.trace_dir / "run_20260712_000000.jsonl"
            trace.write_text(
                "".join(
                    json.dumps(
                        {
                            "step": index,
                            "observation": {
                                "ok": False,
                                "data": {"initializer_validation_errors": errors},
                            },
                        },
                        indent=2,
                    )
                    + "\n"
                    for index, errors in enumerate(errors_by_step, start=1)
                ),
                encoding="utf-8",
            )
            signature = loop._initializer_error_signature(errors_by_step[-1])

            count = loop._recent_initializer_error_repeat_count(signature)

        self.assertEqual(count, 3)

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

    def test_cli_accepts_auto_resume_session_limit(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["Task", "--auto-resume", "--max-sessions", "3"])

        self.assertTrue(args.auto_resume)
        self.assertEqual(args.max_sessions, 3)

    def test_cli_defaults_to_unlimited_steps_per_session(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["Task"])

        self.assertIsNone(args.max_steps)

    def test_cli_accepts_explicit_log_file(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["Task", "--log-file", "diagnostics/run.log"])

        self.assertEqual(str(args.log_file), "diagnostics\\run.log")

    def test_debug_context_action_is_validated(self) -> None:
        state = create_initial_state("Inspect context")

        action = validate_action(
            {
                "thought_summary": "Inspect the current model context.",
                "action": "debug_context",
                "target": "current",
                "args": {"include_content": True},
                "expected_observation": "Context snapshot is returned.",
                "risk": "low",
            },
            state,
        )

        self.assertEqual(action["action"], "debug_context")
        self.assertTrue(action["args"]["include_content"])

    def test_debug_context_snapshot_is_written_and_trace_references_it(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(root=root, task="Inspect context", max_steps=1)
            loop._ensure_state_files()
            state = create_initial_state("Inspect context")

            snapshot = loop._record_context_snapshot(1, state, "# User-visible context")
            observation = loop._handle_debug_context_action(
                {"action": "debug_context", "target": "current", "args": {"include_content": True}}
            )
            loop._append_trace(1, {"action": "debug_context", "target": "current", "args": {}}, observation, state, snapshot)

            snapshot_path = root / snapshot["path"]
            snapshot_content = snapshot_path.read_text(encoding="utf-8")
            trace_events = loop._load_trace_events(loop.trace_path)

        self.assertTrue(observation.ok)
        self.assertIn("# Full Model Context", snapshot_content)
        self.assertIn("## System Message", observation.data["content"])
        self.assertEqual(trace_events[0]["context_ref"]["path"], snapshot["path"])
        self.assertEqual(trace_events[0]["tool_return"], trace_events[0]["observation"])

    def test_default_log_path_uses_benchmark_state_directory(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["Task", "--root", str(WORKSPACE_TMP), "--benchmark", "todo_counter"]
        )

        log_path = resolve_log_path(args, infer_benchmark_id(args))

        self.assertEqual(log_path.parent, WORKSPACE_TMP.resolve() / "state" / "benchmarks" / "todo_counter" / "logs")
        self.assertTrue(log_path.name.startswith("run_"))
        self.assertEqual(log_path.suffix, ".log")

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

    def test_initializer_update_plan_does_not_complete_init(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="INIT", max_steps=1)
            state = create_initializer_state(
                "Build benchmark",
                project_spec_artifact="state/project_spec.md",
                generated_tasks_artifact="state/generated_tasks.json",
                init_artifact="state/init.sh",
            )

            loop._update_state(
                state,
                {"action": "update_plan", "target": "current_task", "args": {}},
                ToolResult(True, "Plan updated by harness.", {}),
            )

        self.assertEqual(state.nodes[0]["status"], "in_progress")
        self.assertFalse(state.evidence_sources)

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
                                "criterion_command_map": {
                                    "Skeleton exists.": [
                                        "python -c \"import pathlib; assert pathlib.Path('eval/benchmarks/todo_counter/workspace/README.md').is_file()\""
                                    ]
                                },
                                "expected_artifacts": ["eval/benchmarks/todo_counter/workspace/README.md"],
                                "implementation_artifacts": [
                                    "eval/benchmarks/todo_counter/workspace/README.md"
                                ],
                                "verification_commands": [
                                    "python -c \"import pathlib; assert pathlib.Path('eval/benchmarks/todo_counter/workspace/README.md').is_file()\""
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (generated.parent / "init.sh").write_text(
                "#!/usr/bin/env sh\nset -eu\npython -c \"import sys; assert sys.version_info >= (3, 8)\"\n",
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
            self.assertEqual(state.acceptance_contracts[-1]["status"], "agreed")
            self.assertTrue(state.acceptance_contracts[-1]["frozen"])
            self.assertEqual(state.acceptance_contracts[-1]["source"], "task_graph")

    def test_initializer_remains_active_until_init_script_exists(self) -> None:
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
                                "expected_artifacts": [
                                    "eval/benchmarks/todo_counter/workspace/README.md"
                                ],
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

        self.assertEqual(state.task_id, "INIT")
        self.assertEqual(state.nodes[0]["id"], "INIT")

    def test_initializer_can_write_its_artifacts_without_contract(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = root / "eval" / "benchmarks" / "todo_counter" / "project_spec.md"
            spec.parent.mkdir(parents=True)
            spec.write_text(
                "The generated application should live under `eval/benchmarks/todo_counter/workspace/`.\n",
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
            graph = {
                "tasks": [
                    {
                        "id": "T1",
                        "title": "Implement counter",
                        "priority": 1,
                        "depends_on": [],
                        "status": "pending",
                        "acceptance_criteria": ["Counter works."],
                        "criterion_command_map": {
                            "Counter works.": [
                                "python -m unittest discover -s eval/benchmarks/todo_counter/workspace/tests"
                            ]
                        },
                        "expected_artifacts": [
                            "eval/benchmarks/todo_counter/workspace/todo_counter/core.py"
                        ],
                        "implementation_artifacts": [
                            "eval/benchmarks/todo_counter/workspace/todo_counter/core.py"
                        ],
                        "verification_commands": [
                            "python -m unittest discover -s eval/benchmarks/todo_counter/workspace/tests"
                        ],
                    }
                ]
            }

            observation = loop._execute_action(
                {
                    "action": "write",
                    "target": "state/benchmarks/todo_counter/generated_tasks.json",
                    "args": {"mode": "overwrite", "content": json.dumps(graph)},
                },
                state,
            )
            generated_exists = (root / "state" / "benchmarks" / "todo_counter" / "generated_tasks.json").exists()

        self.assertTrue(observation.ok)
        self.assertTrue(generated_exists)

    def test_initializer_rejects_application_code_write(self) -> None:
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

            observation = loop._execute_action(
                {
                    "action": "write",
                    "target": "eval/benchmarks/todo_counter/workspace/todo_counter/core.py",
                    "args": {"mode": "overwrite", "content": "def count(): return 1\n"},
                },
                state,
            )
            workspace_exists = (root / "eval" / "benchmarks" / "todo_counter" / "workspace").exists()

        self.assertFalse(observation.ok)
        self.assertTrue(observation.data["initializer_restricted"])
        self.assertFalse(workspace_exists)

    def test_initializer_rejects_task_graph_outside_spec_workspace(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = root / "eval" / "benchmarks" / "todo_counter" / "project_spec.md"
            spec.parent.mkdir(parents=True)
            spec.write_text(
                "The generated application should live under `eval/benchmarks/todo_counter/workspace/`.\n",
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
            graph = {
                "tasks": [
                    {
                        "id": "T1",
                        "title": "Implement counter",
                        "priority": 1,
                        "depends_on": [],
                        "status": "pending",
                        "acceptance_criteria": ["Counter works."],
                        "expected_artifacts": [
                            "state/benchmarks/todo_counter/workspace/todo_counter/core.py"
                        ],
                        "verification_commands": ["python -c \"assert True\""],
                    }
                ]
            }

            observation = loop._execute_action(
                {
                    "action": "write",
                    "target": "state/benchmarks/todo_counter/generated_tasks.json",
                    "args": {"mode": "overwrite", "content": json.dumps(graph)},
                },
                state,
            )
            generated_exists = (root / "state" / "benchmarks" / "todo_counter" / "generated_tasks.json").exists()

        self.assertFalse(observation.ok)
        self.assertIn("initializer_validation_errors", observation.data)
        self.assertIn("eval/benchmarks/todo_counter/workspace", " ".join(observation.data["initializer_validation_errors"]))
        self.assertFalse(generated_exists)

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
            "git_clean": {"ok": True},
            "budget": {"ok": True, "exhausted": False},
            "failure_limits": {"ok": True, "exceeded": {}},
            "human_intervention": {"ok": False, "reasons": ["external_api_key_required"]},
        }

        result = decide_termination(checks)

        self.assertEqual(result.status, "requires_human_intervention")
        self.assertIn("external_api_key_required", result.reasons)

    def test_benchmark_termination_ignores_host_git_and_regression_scope(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            terminator = ProjectTerminator(root, benchmark_id="sample")

            git_check = terminator._git_clean()
            regression_check = terminator._run_regression()

        self.assertTrue(git_check["ok"])
        self.assertTrue(git_check["skipped"])
        self.assertIn("outside benchmark scope", git_check["summary"])
        self.assertTrue(regression_check["ok"])
        self.assertTrue(regression_check["skipped"])

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

    def test_read_default_window_is_50_lines(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "large.txt").write_text("\n".join(f"line {index}" for index in range(1, 81)), encoding="utf-8")

            result = ReadTool(root).run({"action": "read", "target": "large.txt", "args": {}})

        self.assertTrue(result.ok)
        self.assertEqual(result.data["start"], 1)
        self.assertEqual(result.data["end"], 50)
        self.assertTrue(result.data["has_more"])
        self.assertEqual(result.data["next_read"]["args"], {"start": 51, "end": 100})
        self.assertIn("Read 50 line(s)", result.summary)

    def test_read_query_returns_matching_code_instead_of_file_head(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "module.py").write_text(
                "\n".join(
                    ["# header"] * 80
                    + [
                        "def target_function():",
                        "    return 'found'",
                    ]
                ),
                encoding="utf-8",
            )

            result = ReadTool(root).run(
                {"action": "read", "target": "module.py", "args": {"query": "target_function", "context": 1}}
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.data["match_line"], 81)
        self.assertEqual(result.data["start"], 80)
        self.assertIn("def target_function", result.data["content"])
        self.assertNotIn("# header\n# header\n# header", result.data["content"])

    def test_read_query_reports_next_read_when_truncated(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "module.py").write_text(
                "\n".join(["def target_function():"] + [f"    value_{index} = {index}" for index in range(20)]),
                encoding="utf-8",
            )

            first = ReadTool(root).run(
                {"action": "read", "target": "module.py", "args": {"query": "target_function", "max_lines": 5}}
            )
            follow_up = ReadTool(root).run(
                {"action": "read", "target": first.data["next_read"]["target"], "args": first.data["next_read"]["args"]}
            )

        self.assertTrue(first.ok)
        self.assertTrue(first.data["has_more"])
        self.assertIn("More lines exist after this window", first.summary)
        self.assertIn("Continue with data.next_read.args only if", first.summary)
        self.assertEqual(first.data["next_read"]["args"]["continue_from"], 6)
        self.assertTrue(follow_up.ok)
        self.assertEqual(follow_up.data["start"], 6)
        self.assertIn("value_4", follow_up.data["content"])
        self.assertNotIn("def target_function", follow_up.data["content"])

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

    def test_benchmark_git_writes_to_isolated_workspace(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "eval" / "benchmarks" / "sample" / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
            tool = GitTool(
                workspace,
                allow_write=True,
                auto_init=True,
                scope_description="benchmark workspace eval/benchmarks/sample/workspace",
            )

            add_result = tool.run({"action": "git", "target": "add --all", "args": {}})
            commit_result = tool.run({"action": "git", "target": "commit -m benchmark", "args": {}})
            root_git_exists = (root / ".git").exists()

        self.assertTrue(add_result.ok, add_result.data.get("output"))
        self.assertTrue(commit_result.ok, commit_result.data.get("output"))
        self.assertTrue((workspace / ".git").is_dir())
        self.assertFalse(root_git_exists)

    def test_benchmark_loop_configures_workspace_git(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(root=root, task="Benchmark", max_steps=1, benchmark_id="sample")

        self.assertTrue(loop.tools["git"].allow_write)
        self.assertEqual(
            loop.tools["git"].root,
            root / "eval" / "benchmarks" / "sample" / "workspace",
        )
        self.assertTrue(loop.tools["git"].auto_init)

    def test_benchmark_context_uses_workspace_git_scope(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess_result = subprocess.run(["git", "init"], cwd=root, capture_output=True, text=True)
            self.assertEqual(subprocess_result.returncode, 0, subprocess_result.stderr)
            workspace = root / "eval" / "benchmarks" / "sample" / "workspace"
            workspace.mkdir(parents=True)
            state_dir = root / "state" / "benchmarks" / "sample"
            state_dir.mkdir(parents=True)
            context = ContextBuilder(root, state_dir=state_dir, git_root=workspace).build(
                create_initial_state("Benchmark")
            )

        self.assertIn("benchmark workspace git (eval/benchmarks/sample/workspace) git status", context)
        self.assertIn("Git workspace is not initialized yet", context)
        self.assertIn("Git commands are scoped to the benchmark workspace", context)

    def test_bash_accepts_args_command(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            result = BashTool(Path(tmp)).run(
                {"action": "bash", "target": ".", "args": {"command": "python --version"}}
            )

        self.assertTrue(result.ok)
        self.assertIn("Python", result.data["output"])

    def test_bash_injects_benchmark_workspace_python_path(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "eval" / "benchmarks" / "sample" / "workspace"
            package = workspace / "sample_package"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("VALUE = 42\n", encoding="utf-8")

            result = BashTool(root, python_path=workspace).run(
                {
                    "action": "bash",
                    "target": "python -c \"import sample_package; assert sample_package.VALUE == 42\"",
                    "args": {},
                }
            )

        self.assertTrue(result.ok, result.data.get("output"))
        self.assertEqual(Path(result.data["python_path"]), workspace.resolve())

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

    def test_loop_executes_model_action_without_rewrite(self) -> None:
        action = {
            "action": "list_files",
            "target": ".",
            "args": {},
            "thought_summary": "Inspect the workspace.",
            "expected_observation": "Workspace entries.",
            "risk": "low",
        }

        class FixedDecisionMaker:
            def next_action(self, context: str, state: object) -> dict[str, object]:
                return action

        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(root=root, task="Inspect workspace", max_steps=1)
            loop._ensure_state_files()
            loop.decision_maker = FixedDecisionMaker()
            state = create_initial_state("Inspect workspace")
            state.last_action = dict(action)
            state.last_observation = {
                "ok": False,
                "summary": "Previous listing failed.",
                "data": {},
            }

            loop._run_one_session(state)

        self.assertEqual(state.last_action, action)
        self.assertTrue(state.last_observation["ok"])

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
            trace = root / "state" / "traces" / "run_test.jsonl"
            trace.write_text(
                json.dumps(
                    {
                        "step": 1,
                        "task_id": "current",
                        "action": {"action": "verify"},
                        "observation": {"ok": False, "summary": "Verifier failed.", "data": {}},
                    }
                ),
                encoding="utf-8",
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

    def test_managed_contract_allows_verification_procedure_update_only(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement generated task", max_steps=1)
            criterion = "CLI counts open todos."
            old_command = "python -m unittest tests.test_cli"
            new_command = "python -m unittest discover -s tests"
            state = create_initial_state("Implement generated task")
            state.task_id = "T1"
            state.acceptance_criteria = [criterion]
            state.nodes = [
                {
                    "id": "T1",
                    "title": "Todo CLI",
                    "status": "in_progress",
                    "evidence": [],
                    "acceptance_criteria": [criterion],
                    "criterion_command_map": {criterion: [old_command]},
                    "expected_artifacts": ["workspace/todo_counter/cli.py"],
                    "verification_commands": [old_command],
                    "contract_managed": True,
                }
            ]
            state.acceptance_contracts = [
                {
                    "task_id": "T1",
                    "summary": "Frozen task-graph acceptance contract for T1: Todo CLI",
                    "scope": ["workspace/todo_counter/cli.py"],
                    "frozen_requirements": [criterion],
                    "verification_procedure": {"command": old_command},
                    "checks": [old_command],
                    "criterion_command_map": {criterion: [old_command]},
                    "required_evidence": [criterion],
                    "status": "agreed",
                    "source": "task_graph",
                    "frozen": True,
                }
            ]

            result = loop._execute_action(
                {
                    "action": "contract",
                    "target": "T1",
                    "args": {
                        "task_id": "T1",
                        "summary": "Frozen task-graph acceptance contract for T1: Todo CLI",
                        "frozen_requirements": [criterion],
                        "verification_procedure": {"command": new_command},
                    },
                },
                state,
            )

        self.assertTrue(result.ok)
        self.assertEqual(state.acceptance_contracts[-1]["checks"], [new_command])
        self.assertEqual(state.acceptance_contracts[-1]["frozen_requirements"], [criterion])
        self.assertEqual(state.acceptance_contracts[-1]["criterion_command_map"], {criterion: [new_command]})

    def test_managed_contract_rejects_nested_chdir_and_subprocess_cwd_update(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement generated task", max_steps=1)
            criterion = "CLI creates and lists issues."
            old_command = "python -m unittest discover -s tests"
            workspace = "eval/benchmarks/issue_tracker/workspace"
            bad_command = (
                f"python -c \"import os, subprocess, sys; os.chdir('{workspace}'); "
                f"subprocess.run([sys.executable, '-m', 'issue_tracker.cli', 'list'], cwd='{workspace}')\""
            )
            state = create_initial_state("Implement generated task")
            state.task_id = "T1"
            state.acceptance_criteria = [criterion]
            state.nodes = [
                {
                    "id": "T1",
                    "title": "Issue CLI",
                    "status": "in_progress",
                    "evidence": [],
                    "acceptance_criteria": [criterion],
                    "criterion_command_map": {criterion: [old_command]},
                    "expected_artifacts": [f"{workspace}/issue_tracker/cli.py"],
                    "verification_commands": [old_command],
                    "contract_managed": True,
                }
            ]
            state.acceptance_contracts = [
                {
                    "task_id": "T1",
                    "summary": "Frozen task-graph acceptance contract for T1: Issue CLI",
                    "scope": [f"{workspace}/issue_tracker/cli.py"],
                    "frozen_requirements": [criterion],
                    "verification_procedure": {"command": old_command},
                    "checks": [old_command],
                    "criterion_command_map": {criterion: [old_command]},
                    "required_evidence": [criterion],
                    "status": "agreed",
                    "source": "task_graph",
                    "frozen": True,
                }
            ]

            result = loop._execute_action(
                {
                    "action": "contract",
                    "target": "T1",
                    "args": {
                        "task_id": "T1",
                        "summary": "Frozen task-graph acceptance contract for T1: Issue CLI",
                        "frozen_requirements": [criterion],
                        "verification_procedure": {"command": bad_command},
                    },
                },
                state,
            )

        self.assertFalse(result.ok)
        self.assertFalse(result.data["checks"]["portable_executable_checks"])

    def test_managed_contract_rejects_frozen_requirement_change(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement generated task", max_steps=1)
            criterion = "CLI counts open todos."
            command = "python -m unittest discover -s tests"
            state = create_initial_state("Implement generated task")
            state.task_id = "T1"
            state.nodes = [
                {
                    "id": "T1",
                    "title": "Todo CLI",
                    "status": "in_progress",
                    "evidence": [],
                    "acceptance_criteria": [criterion],
                    "criterion_command_map": {criterion: [command]},
                    "expected_artifacts": ["workspace/todo_counter/cli.py"],
                    "verification_commands": [command],
                    "contract_managed": True,
                }
            ]
            state.acceptance_contracts = [
                {
                    "task_id": "T1",
                    "summary": "Frozen task-graph acceptance contract for T1: Todo CLI",
                    "scope": ["workspace/todo_counter/cli.py"],
                    "frozen_requirements": [criterion],
                    "verification_procedure": {"command": command},
                    "checks": [command],
                    "criterion_command_map": {criterion: [command]},
                    "required_evidence": [criterion],
                    "status": "agreed",
                    "source": "task_graph",
                    "frozen": True,
                }
            ]

            result = loop._execute_action(
                {
                    "action": "contract",
                    "target": "T1",
                    "args": {
                        "task_id": "T1",
                        "frozen_requirements": ["CLI returns any JSON."],
                        "verification_procedure": {"command": command},
                    },
                },
                state,
            )

        self.assertFalse(result.ok)
        self.assertIn("frozen_requirements", result.summary)

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

        self.assertIn("## Recent Step Trace", context)
        self.assertIn("### Detailed Tool Observations", context)
        self.assertIn("- action: read", context)
        self.assertIn("workspace/pkg/store.py", context)

    def test_context_lists_missing_owned_artifacts(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            state = create_initial_state("Implement reports")
            state.task_id = "T9"
            state.nodes = [
                {
                    "id": "T9",
                    "title": "Reports",
                    "status": "in_progress",
                    "expected_artifacts": [
                        "workspace/reports.py",
                        "workspace/tests/test_reports.py",
                    ],
                    "implementation_artifacts": ["workspace/reports.py"],
                    "worker_test_artifacts": ["workspace/tests/test_reports.py"],
                }
            ]

            context = ContextBuilder(root).build(state)

        self.assertIn(
            "- missing_owned_artifacts: workspace/reports.py, workspace/tests/test_reports.py",
            context,
        )

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

    def test_nested_cwd_failure_records_command_environment_repair(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement CLI", max_steps=1)
            state = create_initial_state("Implement CLI")
            state.task_id = "T1"
            command = (
                "python -c \"import os, subprocess, sys; os.chdir('workspace'); "
                "subprocess.run([sys.executable, '-m', 'issue_tracker.cli'], cwd='workspace')\""
            )
            state.nodes = [
                {
                    "id": "T1",
                    "title": "CLI",
                    "status": "in_progress",
                    "expected_artifacts": ["workspace/issue_tracker/cli.py"],
                    "verification_commands": [command],
                }
            ]
            state.acceptance_contracts.append(
                {
                    "task_id": "T1",
                    "summary": "Implement CLI.",
                    "checks": [command],
                    "verification_procedure": {"command": command},
                    "status": "agreed",
                }
            )
            output = (
                "Traceback (most recent call last):\n"
                "  File \"<string>\", line 1, in <module>\n"
                "NotADirectoryError: [WinError 267] The directory name is invalid\n"
            )

            loop._update_state(
                state,
                {"action": "bash", "target": command, "args": {}},
                ToolResult(False, "Command exited with code 1.", {"command": command, "output": output}),
            )

        self.assertEqual(state.pending_repair["reason"], "failed_acceptance_command")
        self.assertEqual(state.pending_repair["command_failure_type"], "command_environment_error")
        self.assertEqual(state.pending_repair["targets"], [])
        self.assertEqual(state.pending_repair["repair_targets"], [])

    def test_repeated_failed_bash_command_is_rejected(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement store", max_steps=1)
            state = create_initial_state("Implement store")
            command = "python -m unittest discover -s workspace/tests"
            state.last_action = {"action": "bash", "target": command, "args": {}}
            state.last_observation = {
                "ok": False,
                "summary": "Command exited with code 1.",
                "data": {"command": command, "output": "ImportError: Start directory is not importable"},
            }

            result = loop._execute_action(
                {"action": "bash", "target": f"{command} 2>&1", "args": {"timeout": 60}},
                state,
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.data["required_action"], "repair_or_update_verification")
        self.assertIn("same command just failed", result.summary)

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

    def test_pending_repair_infers_package_module_path_from_import_failure(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement core", max_steps=1)
            state = create_initial_state("Implement core")
            state.task_id = "T2"
            state.nodes = [
                {
                    "id": "T2",
                    "title": "Core",
                    "status": "in_progress",
                    "expected_artifacts": ["eval/benchmarks/todo_counter/workspace/core.py"],
                    "implementation_artifacts": ["eval/benchmarks/todo_counter/workspace/core.py"],
                    "verification_commands": [
                        "python -c \"import sys; sys.path.insert(0, 'eval/benchmarks/todo_counter/workspace'); from todo_counter.core import parse_todos\""
                    ],
                }
            ]
            state.acceptance_contracts.append(
                {
                    "task_id": "T2",
                    "summary": "Implement core.",
                    "checks": [
                        "python -c \"import sys; sys.path.insert(0, 'eval/benchmarks/todo_counter/workspace'); from todo_counter.core import parse_todos\""
                    ],
                    "status": "agreed",
                }
            )
            command = state.acceptance_contracts[-1]["checks"][0]
            output = "ModuleNotFoundError: No module named 'todo_counter'\n"

            loop._update_state(
                state,
                {"action": "bash", "target": command, "args": {}},
                ToolResult(False, "Command exited with code 1.", {"command": command, "output": output}),
            )

        self.assertIn("eval/benchmarks/todo_counter/workspace/todo_counter/core.py", state.pending_repair["targets"])
        self.assertIn("eval/benchmarks/todo_counter/workspace/todo_counter/__init__.py", state.pending_repair["targets"])
        self.assertEqual(
            state.pending_repair["repair_targets"][0],
            "eval/benchmarks/todo_counter/workspace/todo_counter/core.py",
        )

    def test_context_mentions_pending_repair_details(self) -> None:
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

        self.assertIn("# Repair Details", context)
        self.assertIn("- repair_targets: ['workspace/issue_tracker/store.py']", context)
        self.assertIn("workspace/issue_tracker/store.py", context)

    def test_repair_details_omits_failure_output_body(self) -> None:
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
                "command": "python -m unittest discover -s workspace/tests",
                "summary": "Command exited with code 1.",
                "output": "LONG_FAILURE_OUTPUT_MARKER",
                "targets": ["workspace/issue_tracker/store.py"],
            }

            context = ContextBuilder(root).build(state)
            repair_details = context.split("# Repair Details", 1)[1].split("## Recent Step Trace", 1)[0]

        self.assertNotIn("failure_output", repair_details)
        self.assertNotIn("LONG_FAILURE_OUTPUT_MARKER", repair_details)
        self.assertIn("- summary: Command exited with code 1.", repair_details)
        self.assertIn("- repair_targets: ['workspace/issue_tracker/store.py']", repair_details)

    def test_context_handles_pending_repair_without_mutable_targets(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            state = create_initial_state("Write public tests")
            state.task_id = "T4"
            state.nodes = [
                {
                    "id": "T4",
                    "title": "Public tests",
                    "status": "in_progress",
                    "expected_artifacts": ["workspace/tests/test_core.py"],
                    "implementation_artifacts": [],
                    "worker_test_artifacts": ["workspace/tests/test_core.py"],
                    "acceptance_artifacts": ["workspace/tests/test_core.py"],
                    "frozen_acceptance_artifacts": ["workspace/tests/test_core.py"],
                    "test_policy": {
                        "acceptance_tests_mutable_by_worker": False,
                        "acceptance_test_repair_requires_verifier_approval": True,
                    },
                }
            ]
            state.acceptance_contracts = [
                {
                    "task_id": "T4",
                    "summary": "Run public tests.",
                    "checks": ["python -m unittest discover -s workspace/tests"],
                    "status": "agreed",
                }
            ]
            state.pending_repair = {
                "reason": "failed_acceptance_command",
                "command": "python -m unittest discover -s workspace/tests",
                "summary": "Command exited with code 1.",
                "output": "ModuleNotFoundError: No module named 'package'",
                "targets": ["workspace/tests/test_core.py"],
                "repair_targets": ["workspace/tests/test_core.py"],
            }

            context = ContextBuilder(root).build(state)

        self.assertIn("# Repair Details", context)
        self.assertIn("- repair_targets: []", context)
        self.assertIn("workspace/tests/test_core.py", context)

    def test_worker_test_repair_after_agreed_contract_records_repair_and_requests_retest(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state" / "traces").mkdir(parents=True)
            test_path = root / "workspace" / "tests" / "test_cli.py"
            test_path.parent.mkdir(parents=True)
            test_path.write_text("EXPECTED = 'old'\n", encoding="utf-8")
            loop = AgentLoop(root=root, task="Repair worker tests", max_steps=1)
            command = "python -m unittest discover -s workspace/tests"
            target = "workspace/tests/test_cli.py"
            state = create_initial_state("Repair worker tests")
            state.task_id = "T3"
            state.nodes = [
                {
                    "id": "T3",
                    "title": "Repair worker tests",
                    "status": "in_progress",
                    "expected_artifacts": [target],
                    "implementation_artifacts": [],
                    "worker_test_artifacts": [target],
                    "acceptance_artifacts": [],
                    "frozen_acceptance_artifacts": [],
                    "verification_commands": [command],
                }
            ]
            state.acceptance_contracts = [
                {
                    "task_id": "T3",
                    "summary": "Worker tests pass.",
                    "checks": [command],
                    "status": "agreed",
                }
            ]
            state.pending_repair = {
                "reason": "failed_acceptance_command",
                "command": command,
                "summary": "Command exited with code 1.",
                "output": "AssertionError",
                "targets": [target],
                "repair_targets": [target],
                "required_reads": [target],
                "read_targets": [target],
                "repaired_targets": [],
            }
            action = {
                "action": "edit",
                "target": target,
                "args": {"old": "'old'", "new": "'new'"},
            }

            observation = loop._execute_action(action, state)
            loop._update_state(state, action, observation)
            context = ContextBuilder(root).build(state)

        self.assertTrue(observation.ok)
        self.assertEqual(state.pending_repair["repaired_targets"], [target])
        self.assertIn(command, context)
        self.assertIn(f"- repaired_targets: ['{target}']", context)

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
            state.evidence_sources.append(
                {
                    "action": "bash",
                    "target": "python -c \"assert True\"",
                    "summary": "Command exited with code 0.",
                    "task_id": "T1",
                    "evidence_type": "acceptance_command_passed",
                    "ok": True,
                }
            )

            observation = loop._execute_action({"action": "verify", "target": "default", "args": {}}, state)
            loop._update_state(state, {"action": "verify", "target": "default", "args": {}}, observation)
            loop._apply_orchestrator_selection(state)
            data = json.loads((root / "tasks.json").read_text(encoding="utf-8"))

        self.assertTrue(observation.ok)
        self.assertEqual(state.task_id, "T2")
        self.assertEqual(state.nodes[0]["status"], "in_progress")
        self.assertEqual(data["tasks"][0]["status"], "completed")
        self.assertEqual(data["tasks"][1]["status"], "in_progress")

    def test_finish_does_not_run_manual_hidden_acceptance_or_create_repair_tasks(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            source_tasks = root / "input_tasks.json"
            source_tasks.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "id": "T1",
                                "title": "Implement CLI",
                                "priority": 1,
                                "depends_on": [],
                                "status": "completed",
                                "acceptance_criteria": ["CLI works."],
                                "criterion_command_map": {
                                    "CLI works.": ["python -c \"assert True\""],
                                },
                                "expected_artifacts": [
                                    "eval/benchmarks/sample/workspace/issue_tracker/cli.py",
                                ],
                                "implementation_artifacts": [
                                    "eval/benchmarks/sample/workspace/issue_tracker/cli.py",
                                ],
                                "worker_test_artifacts": [],
                                "acceptance_artifacts": [],
                                "frozen_acceptance_artifacts": [],
                                "test_policy": {
                                    "acceptance_tests_mutable_by_worker": False,
                                    "acceptance_test_repair_requires_verifier_approval": True,
                                },
                                "verification_commands": ["python -c \"assert True\""],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            hidden = root / "eval" / "benchmarks" / "sample" / "hidden_acceptance.py"
            hidden.parent.mkdir(parents=True)
            hidden.write_text(
                "import pathlib, sys\n"
                "pathlib.Path('hidden-was-run.txt').write_text('unexpected', encoding='utf-8')\n"
                "sys.exit(1)\n",
                encoding="utf-8",
            )
            loop = AgentLoop(
                root=root,
                task="Benchmark project",
                max_steps=1,
                benchmark_id="sample",
                tasks_path=source_tasks,
            )
            loop._ensure_state_files()
            loop._prepare_runtime_task_graph()
            state = create_initial_state("Benchmark project")
            state.task_id = "T1"
            state.nodes = [{"id": "T1", "title": "Implement CLI", "status": "completed", "evidence": []}]

            result = loop._execute_action({"action": "finish", "target": "current_task", "args": {}}, state)
            data = json.loads(loop.tasks_path.read_text(encoding="utf-8"))

        self.assertTrue(result.ok)
        self.assertEqual(result.summary, "Project completed.")
        self.assertNotIn("hidden_acceptance", result.data["checks"])
        self.assertFalse((root / "hidden-was-run.txt").exists())
        self.assertEqual([task["id"] for task in data["tasks"]], ["T1"])
        self.assertEqual(state.task_id, "T1")

    def test_generated_task_graph_rejects_unknown_contract_task_id(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            source_tasks = root / "input_tasks.json"
            source_tasks.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "id": "T5",
                                "title": "README",
                                "priority": 1,
                                "depends_on": [],
                                "status": "completed",
                                "acceptance_criteria": ["README exists."],
                                "expected_artifacts": ["README.md"],
                                "verification_commands": ["python -c \"assert True\""],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            loop = AgentLoop(root=root, task="Benchmark project", max_steps=1, tasks_path=source_tasks)
            loop._ensure_state_files()
            loop._prepare_runtime_task_graph()
            state = create_initial_state("Benchmark project")
            state.task_id = "T5"
            state.nodes = [{"id": "T5", "title": "README", "status": "completed", "evidence": []}]

            result = loop._execute_action(
                {
                    "action": "contract",
                    "target": "T5-fix",
                    "args": {
                        "task_id": "T5-fix",
                        "summary": "Ad-hoc repair",
                        "checks": ["python -c \"assert True\""],
                    },
                },
                state,
            )

        self.assertFalse(result.ok)
        self.assertIn("task_id must refer to an active task graph node", result.summary)

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
        self.assertFalse(contract.data["checks"]["portable_executable_checks"])

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
                        "evidence_refs": [
                            {"type": "trace", "path": "state/traces/missing.jsonl", "step": 1}
                        ],
                    },
                },
                state,
            )
            candidate = json.loads(
                (root / "state" / "skill_candidates" / "SC-0001.json").read_text(encoding="utf-8")
            )

        self.assertFalse(observation.ok)
        self.assertEqual(candidate["status"], "rejected_missing_evidence")

    def test_skill_accepts_verifier_confirmed_success(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement a feature", max_steps=1)
            state = create_initial_state("Implement a feature")
            trace = root / "state" / "traces" / "run_test.jsonl"
            trace.write_text(
                json.dumps(
                    {
                        "step": 1,
                        "task_id": "current",
                        "action": {"action": "verify"},
                        "observation": {"ok": True, "summary": "Verifier passed.", "data": {}},
                    }
                ),
                encoding="utf-8",
            )

            observation = loop._execute_action(
                {
                    "action": "skill",
                    "target": "verified-debugging",
                    "args": {
                        "skill_id": "verified-debugging",
                        "title": "Verified debugging",
                        "body": "Run tests before claiming completion.",
                        "evidence_type": "verified_success",
                        "evidence_refs": [
                            {"type": "trace", "path": "state/traces/run_test.jsonl", "step": 1}
                        ],
                    },
                },
                state,
            )
            skill_path = root / "state" / "skills" / "verified-debugging.md"
            skill_exists = skill_path.exists()
            candidate = json.loads(
                (root / "state" / "skill_candidates" / "SC-0001.json").read_text(encoding="utf-8")
            )

        self.assertTrue(observation.ok)
        self.assertTrue(skill_exists)
        self.assertFalse((root / "state" / "skills" / "verified-debugging.md.tmp").exists())
        self.assertEqual(candidate["status"], "promoted")
        self.assertEqual(
            [item["status"] for item in candidate["status_history"]],
            ["proposed", "evidence_validated", "content_validated", "approved", "promoted"],
        )

    def test_skill_accepts_evidence_confirmed_failure(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement a feature", max_steps=1)
            state = create_initial_state("Implement a feature")
            trace = root / "state" / "traces" / "run_test.jsonl"
            trace.write_text(
                json.dumps(
                    {
                        "step": 2,
                        "task_id": "current",
                        "action": {"action": "contract"},
                        "observation": {"ok": False, "summary": "Contract rejected.", "data": {}},
                    }
                ),
                encoding="utf-8",
            )

            observation = loop._execute_action(
                {
                    "action": "skill",
                    "target": "avoid-weak-contract",
                    "args": {
                        "skill_id": "avoid-weak-contract",
                        "title": "Avoid weak contracts",
                        "body": "Do not use file existence as the only acceptance check.",
                        "evidence_type": "evidence_confirmed_failure",
                        "evidence_refs": [
                            {"type": "trace", "path": "state/traces/run_test.jsonl", "step": 2}
                        ],
                    },
                },
                state,
            )

        self.assertTrue(observation.ok)

    def test_save_skill_accepts_real_verifier_report_reference(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state" / "traces").mkdir(parents=True)
            report = {
                "time": "2026-07-15T00:00:00+00:00",
                "ok": True,
                "summary": "Verifier passed.",
                "data": {"task_id": "current"},
            }
            (root / "state" / "verifier_report.md").write_text(
                "# Latest Verifier Report\n\n```json\n" + json.dumps(report) + "\n```\n",
                encoding="utf-8",
            )
            loop = AgentLoop(root=root, task="Implement a feature", max_steps=1)
            state = create_initial_state("Implement a feature")
            observation = loop._execute_action(
                {
                    "action": "save_skill",
                    "target": "verify-before-finish",
                    "args": {
                        "name": "verify-before-finish",
                        "description": "Require independent verification before finishing a coding task.",
                        "instruction": "Run the mapped verification command before finish.",
                        "evidence_type": "verified_success",
                        "evidence_refs": [{"type": "verifier_report", "task_id": "current"}],
                    },
                },
                state,
            )

        self.assertTrue(observation.ok)
        self.assertEqual(observation.data["candidate_status"], "promoted")

    def test_skill_catalog_injects_metadata_only(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "state" / "skills"
            skill_dir.mkdir(parents=True)
            (skill_dir / "locate-error.md").write_text(
                "---\nname: locate-error\n"
                "description: Locate repeated errors in long logs.\n"
                "---\n\n# Instructions\n\nSECRET FULL PROCEDURE\n",
                encoding="utf-8",
            )
            context = ContextBuilder(root).build(create_initial_state("Debug tests"))

        self.assertIn("locate-error: Locate repeated errors in long logs.", context)
        self.assertNotIn("SECRET FULL PROCEDURE", context)

    def test_skill_reflection_does_not_trigger_after_ordinary_verifier_pass(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state" / "traces").mkdir(parents=True)
            loop = AgentLoop(root=root, task="Implement feature", max_steps=1)
            state = create_initial_state("Implement feature")
            state.task_session_ids["T1"] = ["run-1", "run-2"]
            observation = ToolResult(
                True,
                "Verifier passed.",
                {"report_id": "VR-T1-test", "archived_verifier_report": "state/verifier_reports/VR-T1-test.json"},
            )
            loop._maybe_create_pending_skill_review(state, "T1", observation)

        self.assertFalse(state.pending_skill_review)

    def test_skill_reflection_triggers_only_after_more_than_five_sessions(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state" / "traces").mkdir(parents=True)
            loop = AgentLoop(root=root, task="Implement feature", max_steps=1)
            state = create_initial_state("Implement feature")
            observation = ToolResult(
                True,
                "Verifier passed.",
                {"report_id": "VR-T1-test", "archived_verifier_report": "state/verifier_reports/VR-T1-test.json"},
            )
            state.task_session_ids["T1"] = [f"run-{index}" for index in range(5)]
            loop._maybe_create_pending_skill_review(state, "T1", observation)
            at_five = dict(state.pending_skill_review)
            state.task_session_ids["T1"].append("run-5")
            loop._maybe_create_pending_skill_review(state, "T1", observation)

        self.assertFalse(at_five)
        self.assertEqual(state.pending_skill_review["trigger_reasons"][0]["type"], "high_cost_success")
        self.assertEqual(state.pending_skill_review["trigger_reasons"][0]["session_count"], 6)

    def test_skill_reflection_triggers_after_three_matching_errors_are_resolved(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state" / "traces").mkdir(parents=True)
            loop = AgentLoop(root=root, task="Implement feature", max_steps=1)
            state = create_initial_state("Implement feature")
            fingerprint = "execution_error:ModuleNotFoundError"
            state.error_patterns[fingerprint] = {
                "count": 3,
                "failure_type": "execution_error",
                "task_ids": ["T1"],
            }
            state.task_error_fingerprints["T1"] = [fingerprint]
            observation = ToolResult(
                True,
                "Verifier passed.",
                {"report_id": "VR-T1-test", "archived_verifier_report": "state/verifier_reports/VR-T1-test.json"},
            )
            loop._maybe_create_pending_skill_review(state, "T1", observation)
            context = ContextBuilder(root).build(state)

        reason = state.pending_skill_review["trigger_reasons"][0]
        self.assertEqual(reason["type"], "repeated_error_resolved")
        self.assertEqual(reason["patterns"][0]["count"], 3)
        self.assertIn("# Pending Skill Reflection", context)
        self.assertIn("save_skill or dismiss_skill", context)

    def test_pending_skill_reflection_blocks_work_until_dismissed(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state" / "traces").mkdir(parents=True)
            loop = AgentLoop(root=root, task="Implement feature", max_steps=1)
            state = create_initial_state("Implement feature")
            state.pending_skill_review = {"task_id": "T1", "report_id": "VR-T1-test"}
            blocked = loop._execute_action({"action": "read", "target": "README.md", "args": {}}, state)
            action = {
                "action": "dismiss_skill",
                "target": "VR-T1-test",
                "args": {"reason": "The change was task-specific and not reusable."},
            }
            dismissed = loop._execute_action(action, state)
            loop._update_state(state, action, dismissed)

        self.assertFalse(blocked.ok)
        self.assertEqual(blocked.data["required_action"], "save_skill_or_dismiss_skill")
        self.assertTrue(dismissed.ok)
        self.assertFalse(state.pending_skill_review)
        self.assertEqual(state.skill_review_history[-1]["decision"], "dismissed")

    def test_immutable_verifier_report_resolves_by_report_id(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state" / "traces").mkdir(parents=True)
            loop = AgentLoop(root=root, task="Implement feature", max_steps=1)
            loop._current_trace_step = 7
            archived = loop._archive_verifier_success(
                "T1", ToolResult(True, "Verifier passed.", {"checks": {"unit_tests": True}})
            )
            state = create_initial_state("Implement feature")
            result = loop.verifier.validate_skill_promotion(
                {
                    "name": "verified-procedure",
                    "description": "Reuse a verified procedure.",
                    "instruction": "Execute the procedure and independently verify it.",
                    "evidence_type": "verified_success",
                    "evidence_refs": [
                        {"type": "verifier_report", "report_id": archived["report_id"], "task_id": "T1"}
                    ],
                },
                state,
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.data["resolved_evidence"][0]["report_id"], archived["report_id"])

    def test_load_skill_returns_full_content_and_tracks_pending_validation(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state" / "traces").mkdir(parents=True)
            skill_dir = root / "state" / "skills"
            skill_dir.mkdir()
            (skill_dir / "locate-error.md").write_text(
                "---\nname: locate-error\n"
                "description: Locate repeated errors in long logs.\n"
                "---\n\n# Instructions\n\nSearch the traceback.\n",
                encoding="utf-8",
            )
            loop = AgentLoop(root=root, task="Debug tests", max_steps=1)
            state = create_initial_state("Debug tests")
            observation = loop._execute_action(
                {"action": "load_skill", "target": "locate-error", "args": {}}, state
            )
            duplicate = loop._execute_action(
                {"action": "load_skill", "target": "locate-error", "args": {}}, state
            )
            loaded_context = ContextBuilder(root).build(state)
            (skill_dir / "locate-error.md").write_text(
                "---\nname: locate-error\n"
                "description: Locate repeated errors in long logs.\n"
                "---\n\n# Instructions\n\nChanged procedure.\n",
                encoding="utf-8",
            )
            invalidated_context = ContextBuilder(root).build(state)

        self.assertTrue(observation.ok)
        self.assertIn("Search the traceback.", observation.data["content"])
        self.assertEqual(state.loaded_skills[0]["status"], "loaded")
        self.assertEqual(len(observation.data["content_hash"]), 64)
        self.assertTrue(duplicate.ok)
        self.assertTrue(duplicate.data["already_loaded"])
        self.assertEqual(len(state.loaded_skills), 1)
        self.assertIn("# Loaded Skills", loaded_context)
        self.assertIn("Search the traceback.", loaded_context)
        self.assertNotIn("Search the traceback.", invalidated_context)
        self.assertIn("Invalidated Skills (reload before use): locate-error", invalidated_context)

    def test_save_skill_writes_yaml_structure_and_rejects_duplicate(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state" / "traces").mkdir(parents=True)
            loop = AgentLoop(root=root, task="Debug tests", max_steps=1)
            state = create_initial_state("Debug tests")
            trace = root / "state" / "traces" / "run_test.jsonl"
            trace.write_text(
                json.dumps(
                    {
                        "step": 3,
                        "task_id": "current",
                        "action": {"action": "verify"},
                        "observation": {"ok": True, "summary": "Verifier passed.", "data": {}},
                    }
                ),
                encoding="utf-8",
            )
            action = {
                "action": "save_skill",
                "target": "locate-errors",
                "args": {
                    "name": "locate-errors",
                    "description": "Locate repeated errors in long logs.",
                    "instruction": ["Run the failing command", "Inspect the final workspace frame"],
                    "examples": [{"input": "Traceback", "result": "Relevant source frame"}],
                    "evidence_type": "verified_success",
                    "evidence_refs": [
                        {"type": "trace", "path": "state/traces/run_test.jsonl", "step": 3}
                    ],
                },
            }
            first = loop._execute_action(action, state)
            second = loop._execute_action(action, state)
            content = (root / "state" / "skills" / "locate-errors.md").read_text(encoding="utf-8")

        self.assertTrue(first.ok)
        self.assertFalse(second.ok)
        self.assertTrue(content.startswith('---\nname: "locate-errors"\n'))
        self.assertIn("# Instructions", content)
        self.assertIn("# Examples", content)
        self.assertNotIn("run_test.jsonl", content)

    def test_save_memory_writes_typed_yaml_and_updates_index(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(root=root, task="Remember preference", max_steps=1)
            loop._ensure_state_files()
            state = create_initial_state("Remember preference")
            action = {
                "action": "save_memory",
                "target": "real-db-integration-tests",
                "args": {
                    "name": "real-db-integration-tests",
                    "description": "Integration tests must use a real database",
                    "type": "feedback",
                    "content": "Integration tests must use a real database, not mocks.",
                    "why": "Mock-backed tests passed while production migrations failed.",
                    "how_to_apply": "Connect integration tests to the real test database.",
                },
            }

            result = loop._execute_action(action, state)
            duplicate = loop._execute_action(action, state)
            content = (root / "state" / "memories" / "real-db-integration-tests.md").read_text(encoding="utf-8")
            index = (root / "state" / "memory.md").read_text(encoding="utf-8")
            hard_memory_exists = (root / "state" / "hard_memory.md").exists()
            soft_memory_exists = (root / "state" / "soft_memory.md").exists()

        self.assertTrue(result.ok)
        self.assertFalse(duplicate.ok)
        self.assertTrue(content.startswith('---\nname: "real-db-integration-tests"\n'))
        self.assertIn("type: feedback", content)
        self.assertIn("**Why:** Mock-backed tests passed", content)
        self.assertIn("**How to apply:** Connect integration tests", content)
        self.assertIn("[feedback] real-db-integration-tests", index)
        self.assertFalse(hard_memory_exists)
        self.assertFalse(soft_memory_exists)

    def test_save_memory_rejects_invalid_type_feedback_without_why_and_project_relative_dates(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(root=root, task="Remember constraints", max_steps=1)
            loop._ensure_state_files()
            state = create_initial_state("Remember constraints")

            invalid_type = loop._execute_action(
                {
                    "action": "save_memory",
                    "target": "custom-kind",
                    "args": {
                        "name": "custom-kind",
                        "description": "A custom memory kind",
                        "type": "architecture",
                        "content": "Do not allow this.",
                    },
                },
                state,
            )
            missing_feedback_fields = loop._execute_action(
                {
                    "action": "save_memory",
                    "target": "no-mocks",
                    "args": {
                        "name": "no-mocks",
                        "description": "Do not mock integration tests",
                        "type": "feedback",
                        "content": "Do not mock integration tests.",
                    },
                },
                state,
            )
            relative_project_date = loop._execute_action(
                {
                    "action": "save_memory",
                    "target": "freeze-date",
                    "args": {
                        "name": "freeze-date",
                        "description": "Freeze merges by next Thursday",
                        "type": "project",
                        "content": "Freeze merges by next Thursday.",
                    },
                },
                state,
            )

        self.assertFalse(invalid_type.ok)
        self.assertIn("type must be one of", invalid_type.summary)
        self.assertFalse(missing_feedback_fields.ok)
        self.assertIn("feedback memory must include Why", missing_feedback_fields.summary)
        self.assertIn("feedback memory must include How to apply", missing_feedback_fields.summary)
        self.assertFalse(relative_project_date.ok)
        self.assertIn("relative dates", relative_project_date.summary)

    def test_memory_index_is_truncated_by_lines_and_bytes(self) -> None:
        by_lines = truncate_entrypoint_content("\n".join(f"- item {index}" for index in range(250)))
        by_bytes = truncate_entrypoint_content("x" * 30_000)

        self.assertTrue(by_lines.was_line_truncated)
        self.assertIn("WARNING: memory.md was truncated", by_lines.content)
        self.assertTrue(by_bytes.was_byte_truncated)
        self.assertLessEqual(len(by_bytes.content.encode("utf-8")), 25_200)

    def test_memory_retriever_scans_headers_and_loads_relevant_memory_with_local_fallback(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "state" / "memories"
            memory_dir.mkdir(parents=True)
            (memory_dir / "feedback_no_mock_db.md").write_text(
                "---\n"
                "name: no-mock-database\n"
                "description: Integration tests must use real database\n"
                "type: feedback\n"
                "---\n\n"
                "Integration tests must use the real database.\n\n"
                "**Why:** Mock tests missed migration failures.\n"
                "**How to apply:** Connect to the real test DB.\n",
                encoding="utf-8",
            )
            (memory_dir / "user_preferences.md").write_text(
                "---\n"
                "name: user-preferences\n"
                "description: User prefers terse responses\n"
                "type: user\n"
                "---\n\n"
                + ("body line that should not affect header scanning\n" * 40),
                encoding="utf-8",
            )

            headers = scan_memory_headers(memory_dir)
            retrieved = MemoryRetriever(root / "state").retrieve("write integration tests against database")

        self.assertEqual(len(headers), 2)
        self.assertEqual(headers[0].filename.endswith(".md"), True)
        self.assertEqual(retrieved.source, "local")
        self.assertEqual(retrieved.selected_filenames, ["feedback_no_mock_db.md"])
        self.assertIn("real database", retrieved.memories[0].content)

    def test_agent_loop_injects_relevant_memories_into_context(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(root=root, task="Write integration tests", max_steps=1)
            loop._ensure_state_files()
            memory_dir = root / "state" / "memories"
            (memory_dir / "feedback_no_mock_db.md").write_text(
                "---\n"
                "name: no-mock-database\n"
                "description: Integration tests must use real database\n"
                "type: feedback\n"
                "---\n\n"
                "Integration tests must use the real database.\n\n"
                "**Why:** Mock tests missed migration failures.\n"
                "**How to apply:** Connect to the real test DB.\n",
                encoding="utf-8",
            )
            (root / "state" / "memory.md").write_text(
                "# Memory Index\n\n"
                "## Entries\n"
                "- [feedback] no-mock-database: Integration tests must use real database (`memories/feedback_no_mock_db.md`)\n",
                encoding="utf-8",
            )
            state = create_initial_state("Write integration tests against the database")

            context = loop.context_builder.build(state, relevant_memories=loop._relevant_memory_context(state))

        self.assertIn("# Relevant Memories", context)
        self.assertIn("feedback_no_mock_db.md", context)
        self.assertIn("Mock tests missed migration failures", context)
        self.assertEqual(loop._last_memory_selection["source"], "local")

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

    def test_budget_handoff_uses_current_turn_tokens_not_session_total(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = AgentLoop(root=root, task="Implement a feature", max_steps=1)
            state = create_initial_state("Implement a feature")
            state.session_budget_tokens = 100
            state.handoff_threshold = 1.0
            action = {"action": "read", "target": "a.py", "args": {}}
            observation = ToolResult(True, "ok", {})

            loop._record_budget_usage(state, "x" * 220, action, observation)
            first_turn_tokens = state.session_used_tokens
            loop._record_budget_usage(state, "x" * 220, action, observation)

        self.assertLess(first_turn_tokens, 100)
        self.assertEqual(state.session_used_tokens, first_turn_tokens)
        self.assertFalse(state.handoff_ready)

    def test_auto_resume_continues_after_handoff_until_session_limit(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "state"
            (state_dir / "traces").mkdir(parents=True)
            (state_dir / "current_task.json").write_text(
                json.dumps(
                    {
                        "task_id": "current",
                        "user_goal": "Resume task",
                        "acceptance_criteria": [],
                        "nodes": [{"id": "T1", "title": "Inspect", "status": "pending", "evidence": []}],
                        "iterations": 0,
                        "last_action": {},
                        "last_observation": {},
                        "evidence_sources": [],
                        "acceptance_contracts": [],
                        "session_budget_tokens": 1,
                        "handoff_threshold": 0.5,
                        "session_used_tokens": 0,
                        "handoff_ready": False,
                    }
                ),
                encoding="utf-8",
            )
            loop = AgentLoop(
                root=root,
                task="Resume task",
                max_steps=1,
                resume=True,
                auto_resume=True,
                max_sessions=2,
            )

            result = loop.run()
            trace_files = list((state_dir / "traces").glob("run_*.jsonl"))

        self.assertFalse(result.completed)
        self.assertEqual(result.sessions, 2)
        self.assertEqual(result.steps, 2)
        self.assertEqual(len(trace_files), 2)
        self.assertIn("Session handoff threshold reached", result.message)

    def test_handoff_contains_session_budget_and_resume_sections(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement a feature", max_steps=1)
            state = create_initial_state("Implement a feature")
            state.session_budget_tokens = 100
            state.handoff_threshold = 0.7
            state.session_used_tokens = 71
            state.handoff_ready = True
            state.evidence_sources.append({"action": "read", "target": "agent/loop.py", "summary": "read"})
            state.task_id = "T2"
            state.nodes = [
                {"id": "T1", "title": "Old feature", "status": "done"},
                {"id": "T2", "title": "Active feature", "status": "in_progress"},
                {"id": "T3", "title": "Future feature", "status": "pending"},
            ]
            state.acceptance_contracts.extend(
                [
                    {"task_id": "T1", "summary": "old contract", "checks": ["old-check"]},
                    {"task_id": "T2", "summary": "active contract", "checks": ["active-check"]},
                    {"task_id": "T3", "summary": "future contract", "checks": ["future-check"]},
                ]
            )

            loop._write_handoff(state)
            handoff = (root / "state" / "handoff.md").read_text(encoding="utf-8")
            payload = json.loads((root / "state" / "handoff_payload.json").read_text(encoding="utf-8"))

        self.assertIn("# Worker Session Handoff", handoff)
        self.assertIn("## Critical Context", handoff)
        self.assertIn("### Session Budget", handoff)
        self.assertIn("## Working Context", handoff)
        self.assertIn("## Reference Context", handoff)
        self.assertIn("### Handoff Data References", handoff)
        self.assertIn("structured_payload: state/handoff_payload.json", handoff)
        self.assertIn("memory_index: state/memory.md", handoff)
        self.assertIn("memories: state/memories/", handoff)
        self.assertNotIn("latest_verifier_report", handoff)
        self.assertIn("### Active Acceptance Contract", handoff)
        self.assertIn("active contract", handoff)
        self.assertNotIn("old contract", handoff)
        self.assertNotIn("future contract", handoff)
        self.assertIn("## Resume Guidance", handoff)
        self.assertIn("### Resume Instructions", handoff)
        self.assertIn("threshold_tokens: 70", handoff)
        self.assertNotIn("hard_memory", handoff)
        self.assertNotIn("soft_memory", handoff)
        self.assertNotIn("Soft Memory", handoff)
        self.assertEqual(payload["schema"], "long-agent.handoff-payload.v1")
        self.assertTrue(payload["session_budget"]["handoff_ready"])
        self.assertEqual([item["task_id"] for item in payload["acceptance_contracts"]], ["T2"])

    def test_handoff_includes_command_only_pending_repair(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Implement CLI", max_steps=1)
            state = create_initial_state("Implement CLI")
            state.task_id = "T3"
            bad_command = (
                "python -c \"import sys,tempfile,subprocess; sys.path.insert(0,'eval/benchmarks/todo_counter/workspace'); "
                "with tempfile.NamedTemporaryFile(mode='w') as f: pass; "
                "subprocess.run([sys.executable,'-m','todo_counter.cli','x'])\""
            )
            state.nodes = [
                {
                    "id": "T3",
                    "title": "CLI",
                    "status": "in_progress",
                    "expected_artifacts": ["eval/benchmarks/todo_counter/workspace/todo_counter/cli.py"],
                    "implementation_artifacts": ["eval/benchmarks/todo_counter/workspace/todo_counter/cli.py"],
                    "verification_commands": [bad_command],
                }
            ]
            state.pending_repair = {
                "reason": "failed_acceptance_command",
                "command": bad_command,
                "summary": "Command exited with code 1.",
                "output": "SyntaxError: invalid syntax",
                "targets": [],
                "repair_targets": [],
                "required_reads": [],
                "read_targets": [],
                "repaired_targets": [],
                "command_failure_type": "command_syntax_error",
            }

            loop._write_handoff(state)
            handoff = (root / "state" / "handoff.md").read_text(encoding="utf-8")

        self.assertIn("### Pending Repair", handoff)
        self.assertIn("- command_failure_type: command_syntax_error", handoff)
        self.assertIn("- suggested_command:", handoff)
        self.assertNotIn("### Pending Repair\n- none", handoff)

    def test_context_builder_uses_reorganized_context_layers(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "memories").mkdir()
            (root / "state" / "memory.md").write_text(
                "# Memory Index\n\n"
                "## Entries\n"
                "- [user] user-react: User is experienced in Go and new to React (`memories/user-react.md`)\n",
                encoding="utf-8",
            )
            (root / "state" / "memories" / "user-react.md").write_text(
                "---\n"
                "name: user-react\n"
                "description: User is experienced in Go and new to React\n"
                "type: user\n"
                "---\n\n"
                "User has ten years of Go backend experience and is new to React.\n",
                encoding="utf-8",
            )
            (root / "state" / "handoff.md").write_text("# Handoff\n", encoding="utf-8")
            (root / "state" / "verifier_report.md").write_text("# Verifier\n", encoding="utf-8")
            (root / "project_spec.md").write_text("# Spec\n", encoding="utf-8")
            (root / "tasks.json").write_text("{}", encoding="utf-8")
            state = create_initial_state("Implement a feature")

            context = ContextBuilder(root).build(state)

        self.assertIn("# Critical Context", context)
        self.assertIn("# Working Context", context)
        self.assertIn("# Session Startup Context", context)
        self.assertNotIn("# Just-in-Time Discovery", context)
        self.assertIn("# Persistent Context", context)
        self.assertNotIn("# Tail Guard", context)
        self.assertIn("# Available Tools And Calling Format", context)
        self.assertIn('"action":"<one action>"', context)
        self.assertIn("- list_files: inspect a directory or file entry", context)
        self.assertIn("- search: grep-style literal text search", context)
        self.assertIn("Use this before read when locating T7, validation errors", context)
        self.assertNotIn("read target='<file>' args={'query': '\"id\": \"T7\"'}", context)
        self.assertIn("- write: create/overwrite/append file", context)
        self.assertIn("- verify: ask harness verifier", context)
        self.assertIn("# Relevant Memories", context)
        self.assertIn("[user] user-react", context)
        self.assertNotIn("User has ten years of Go", context)
        self.assertNotIn("# Hard Memory", context)
        self.assertNotIn("# Soft Memory", context)
        self.assertNotIn("Soft Memory is not evidence", context)
        self.assertNotIn("# Always-on Context", context)
        self.assertNotIn("## Non-Negotiable Rules", context)
        self.assertLess(context.index("# Critical Context"), context.index("# Working Context"))
        self.assertLess(context.index("# Working Context"), context.index("# Session Startup Context"))

    def test_context_builder_includes_only_active_acceptance_contract(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            state = create_initial_state("Implement active feature")
            state.task_id = "T2"
            state.nodes = [
                {"id": "T1", "title": "Old feature", "status": "done"},
                {"id": "T2", "title": "Active feature", "status": "in_progress"},
                {"id": "T3", "title": "Future feature", "status": "pending"},
            ]
            state.acceptance_contracts = [
                {"task_id": "T1", "summary": "old contract", "checks": ["old-check"], "status": "agreed"},
                {"task_id": "T2", "summary": "active contract", "checks": ["active-check"], "status": "agreed"},
                {"task_id": "T3", "summary": "future contract", "checks": ["future-check"], "status": "agreed"},
            ]

            context = ContextBuilder(root).build(state)

        self.assertIn("# Active Acceptance Contract", context)
        self.assertIn("active contract", context)
        self.assertNotIn("old contract", context)
        self.assertNotIn("future contract", context)

    def test_context_builder_keeps_startup_reference_without_later_handoff(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "state"
            state_dir.mkdir()
            (state_dir / "project_spec.md").write_text(
                "# Spec\nVERY_DETAILED_SPEC_BODY\n" + ("details\n" * 200),
                encoding="utf-8",
            )
            (state_dir / "generated_tasks.json").write_text(
                json.dumps({"tasks": [{"id": "T1", "status": "in_progress"}]}),
                encoding="utf-8",
            )
            (state_dir / "runtime_tasks.json").write_text(
                json.dumps({"tasks": [{"id": "T2", "status": "pending"}]}),
                encoding="utf-8",
            )
            (state_dir / "handoff.md").write_text(
                "# Worker Session Handoff\n\n## 15. Suggested Next Action\nRun verifier.\n",
                encoding="utf-8",
            )
            (state_dir / "verifier_report.md").write_text("# Verifier\nVerifier passed.\n", encoding="utf-8")
            state = create_initial_state("Implement a feature")
            state.session_used_tokens = 120
            state.last_verified_at = "2026-07-14T00:00:00+00:00"

            context = ContextBuilder(root).build(state)

        self.assertIn("# Session Startup Context", context)
        self.assertIn("VERY_DETAILED_SPEC_BODY", context)
        self.assertIn("Task graph: state/generated_tasks.json", context)
        self.assertIn("Task graph: state/runtime_tasks.json", context)
        self.assertNotIn("Verifier passed.", context)
        self.assertNotIn("## state/verifier_report.md", context)
        self.assertNotIn("handoff.md focus", context)
        self.assertNotIn("Run verifier.", context)

    def test_context_builder_includes_handoff_only_on_explicit_session_start(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "state"
            state_dir.mkdir()
            (state_dir / "handoff.md").write_text(
                "# Worker Session Handoff\n\n## 15. Suggested Next Action\nHANDOFF_ONLY_ON_SESSION_START\n",
                encoding="utf-8",
            )
            state = create_initial_state("Resume work")
            state.session_used_tokens = 250

            first_step = ContextBuilder(root).build(state, include_handoff=True)
            later_step = ContextBuilder(root).build(state, include_handoff=False)

        self.assertIn("# Session Startup Context", first_step)
        self.assertIn("HANDOFF_ONLY_ON_SESSION_START", first_step)
        self.assertIn("# Session Startup Context", later_step)
        self.assertNotIn("handoff.md focus", later_step)
        self.assertNotIn("HANDOFF_ONLY_ON_SESSION_START", later_step)

    def test_context_builder_normalizes_legacy_handoff_focus_sections(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "state"
            state_dir.mkdir()
            (state_dir / "handoff.md").write_text(
                "# Worker Session Handoff\n\n"
                "## 10a. Pending Repair\n"
                "- none\n\n"
                "## 10b. Initializer Repair\n"
                "- none\n\n"
                "## 12. Known Risks And Failed Attempts\n"
                "- Do not repeat failed actions unchanged.\n\n"
                "## 14. Resume Instructions\n"
                "1. Read this handoff first.\n\n"
                "## 15. Suggested Next Action\n"
                "Resume T9 with a small evidence-backed action.\n",
                encoding="utf-8",
            )
            state = create_initial_state("Resume work")

            context = ContextBuilder(root).build(state, include_handoff=True)

        self.assertIn("## Critical Context", context)
        self.assertIn("### Pending Repair", context)
        self.assertIn("## Resume Guidance", context)
        self.assertIn("### Suggested Next Action", context)
        self.assertIn("Resume T9 with a small evidence-backed action.", context)
        self.assertNotIn("## 10a. Pending Repair", context)
        self.assertNotIn("## 15. Suggested Next Action", context)

    def test_context_builder_omits_working_context_evidence_sources(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "state"
            state_dir.mkdir()
            (state_dir / "handoff.md").write_text(
                "# Worker Session Handoff\n\n"
                "## Working Context\n"
                "### Active Acceptance Contract\n"
                "- T2: active contract\n\n"
                "### Evidence Sources\n"
                "- read: old.py -- OLD_HANDOFF_EVIDENCE\n\n"
                "### Active Verification Commands\n"
                "- python -m unittest\n\n"
                "## Resume Guidance\n"
                "### Suggested Next Action\n"
                "Continue with verification.\n",
                encoding="utf-8",
            )
            state = create_initial_state("Resume work")
            state.evidence_sources.append({"action": "read", "target": "new.py", "summary": "CURRENT_CONTEXT_EVIDENCE"})

            context = ContextBuilder(root).build(state, include_handoff=True)

        self.assertIn("### Active Acceptance Contract", context)
        self.assertIn("### Active Verification Commands", context)
        self.assertNotIn("# Evidence Sources", context)
        self.assertNotIn("CURRENT_CONTEXT_EVIDENCE", context)
        self.assertNotIn("OLD_HANDOFF_EVIDENCE", context)

    def test_session_startup_context_summarizes_task_graph_without_full_json(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "state"
            state_dir.mkdir()
            (state_dir / "generated_tasks.json").write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "id": "T1",
                                "title": "Completed marker SHOULD_NOT_APPEAR_IN_CONTEXT",
                                "status": "completed",
                                "depends_on": [],
                            },
                            {"id": "T2", "title": "Current", "status": "in_progress", "depends_on": ["T1"]},
                            {"id": "T3", "title": "Next", "status": "pending", "depends_on": ["T2"]},
                            {"id": "T4", "title": "Blocked", "status": "pending", "depends_on": ["T9"]},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            state = create_initial_state("Implement a feature")
            state.task_id = "T2"
            state.nodes = [{"id": "T2", "title": "Current", "status": "in_progress"}]

            context = ContextBuilder(root).build(state)

        self.assertIn("Task graph: state/generated_tasks.json", context)
        self.assertIn("Total: 4", context)
        self.assertIn("Done: 1", context)
        self.assertIn("Current task: T2", context)
        self.assertIn("In progress: T2", context)
        self.assertIn("Ready after current completion: T3", context)
        self.assertIn("Blocked: 1", context)
        self.assertNotIn("SHOULD_NOT_APPEAR_IN_CONTEXT", context)

    def test_working_context_omits_full_plan_list(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            state = create_initial_state("Implement a feature")
            state.nodes = [
                {"id": "T1", "title": "Current task", "status": "in_progress"},
                {"id": "T2", "title": "FUTURE_PLAN_MARKER", "status": "pending"},
            ]

            context = ContextBuilder(root).build(state)
            working = context.split("# Working Context", 1)[1].split("# Session Startup Context", 1)[0]

        self.assertNotIn("# Plan", working)
        self.assertNotIn("FUTURE_PLAN_MARKER", working)

    def test_working_context_includes_recent_read_content(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            state = create_initial_state("Inspect implementation")
            state.last_action = {
                "action": "read",
                "target": "agent/example.py",
                "args": {"start": 20, "end": 40},
            }
            state.last_observation = {
                "ok": True,
                "summary": "Read 21 lines.",
                "data": {
                    "target": "agent/example.py",
                    "start": 20,
                    "end": 40,
                    "content": "def calculate_total(items):\n    return sum(items)",
                },
            }

            context = ContextBuilder(root).build(state)

        self.assertIn("## Recent Step Trace", context)
        self.assertIn("### Detailed Tool Observations", context)
        self.assertIn("- range: 20-40", context)
        self.assertIn("def calculate_total(items):", context)
        self.assertIn("return sum(items)", context)
        self.assertLess(context.index("## Recent Step Trace"), context.index("### Detailed Tool Observations"))

    def test_working_context_includes_recent_successful_bash_output(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            state = create_initial_state("Run tests")
            state.last_action = {
                "action": "bash",
                "target": "python -m unittest",
                "args": {},
            }
            state.last_observation = {
                "ok": True,
                "summary": "Command exited with code 0.",
                "data": {
                    "command": "python -m unittest",
                    "output": "Ran 12 tests in 0.5s\nOK",
                    "cwd": str(root),
                },
            }

            context = ContextBuilder(root).build(state)

        self.assertIn("## Recent Step Trace", context)
        self.assertIn("### Detailed Tool Observations", context)
        self.assertIn("- action: bash", context)
        self.assertIn("- command: python -m unittest", context)
        self.assertIn("Ran 12 tests", context)

    def test_working_context_includes_failed_bash_output_from_session_observations(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            state = create_initial_state("Run tests")
            state.last_action = {"action": "bash", "target": "python -m pytest", "args": {}}
            state.last_observation = {
                "ok": False,
                "summary": "Command exited with code 1.",
                "data": {"command": "python -m pytest", "output": "FAILURE_MARKER"},
            }

            context = ContextBuilder(root).build(state)

        self.assertIn("### Detailed Tool Observations", context)
        self.assertIn("FAILURE_MARKER", context)

    def test_working_context_includes_recent_list_entries(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            state = create_initial_state("Inspect files")
            state.last_action = {"action": "list_files", "target": "src", "args": {}}
            state.last_observation = {
                "ok": True,
                "summary": "Listed 2 items.",
                "data": {
                    "target": "src",
                    "entries": [
                        {"path": "src/app.py", "type": "file"},
                        {"path": "src/tests", "type": "dir"},
                    ],
                },
            }

            context = ContextBuilder(root).build(state)

        self.assertIn('"path": "src/app.py"', context)
        self.assertIn('"path": "src/tests"', context)
        self.assertIn('"type": "dir"', context)

    def test_working_context_includes_recent_search_matches_with_limit(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            state = create_initial_state("Find implementation")
            state.last_action = {"action": "search", "target": "calculate_total", "args": {"path": "."}}
            state.last_observation = {
                "ok": True,
                "summary": "Found matches.",
                "data": {
                    "matches": [
                        {
                            "path": "src/app.py",
                            "line": 42,
                            "text": "def calculate_total(items): " + ("x" * 13000),
                        }
                    ]
                },
            }

            context = ContextBuilder(root).build(state)
            observation_section = context.split("### Detailed Tool Observations", 1)[1].split(
                "# Session Startup Context", 1
            )[0]

        self.assertIn('"path": "src/app.py"', observation_section)
        self.assertIn('"line": 42', observation_section)
        self.assertIn("[tool output truncated]", observation_section)
        self.assertLessEqual(len(observation_section), 8400)

    def test_working_context_includes_all_current_session_tool_observations(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_dir = root / "state" / "traces"
            trace_dir.mkdir(parents=True)
            trace = trace_dir / "run_20260714_000000.jsonl"
            events = [
                {
                    "step": 1,
                    "action": {"action": "read", "target": "src/app.py", "args": {}},
                    "observation": {
                        "ok": True,
                        "summary": "Read old app.",
                        "data": {"start": 1, "end": 20, "content": "OLD_APP_CONTENT"},
                    },
                },
                {
                    "step": 2,
                    "action": {"action": "read", "target": "tests/test_app.py", "args": {}},
                    "observation": {
                        "ok": True,
                        "summary": "Read tests.",
                        "data": {"start": 1, "end": 30, "content": "TEST_CONTENT"},
                    },
                },
                {
                    "step": 3,
                    "action": {"action": "search", "target": "calculate_total", "args": {"path": "src"}},
                    "observation": {
                        "ok": True,
                        "summary": "Found match.",
                        "data": {
                            "matches": [
                                {"path": "src/helpers.py", "line": 9, "text": "def calculate_total(items):"}
                            ]
                        },
                    },
                },
                {
                    "step": 4,
                    "action": {"action": "list_files", "target": "src/pkg", "args": {}},
                    "observation": {
                        "ok": True,
                        "summary": "Listed package.",
                        "data": {
                            "entries": [
                                {"path": "src/pkg/core.py", "type": "file"},
                            ]
                        },
                    },
                },
                {
                    "step": 5,
                    "action": {"action": "read", "target": "README.md", "args": {}},
                    "observation": {
                        "ok": True,
                        "summary": "Read README.",
                        "data": {"start": 1, "end": 10, "content": "README_CONTENT"},
                    },
                },
            ]
            for step in range(6, 12):
                events.append(
                    {
                        "step": step,
                        "action": {"action": "read", "target": f"src/file_{step}.py", "args": {}},
                        "observation": {
                            "ok": True,
                            "summary": f"Read file {step}.",
                            "data": {"start": 1, "end": 5, "content": f"FILE_{step}_CONTENT"},
                        },
                    }
                )
            trace.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            state = create_initial_state("Compare implementation and tests")
            state.last_action = {"action": "read", "target": "src/app.py", "args": {}}
            state.last_observation = {
                "ok": True,
                "summary": "Read current app.",
                "data": {"start": 1, "end": 20, "content": "NEW_APP_CONTENT"},
            }

            builder = ContextBuilder(root)
            builder.current_trace_path = trace
            context = builder.build(state)

        self.assertIn("OLD_APP_CONTENT", context)
        self.assertIn("TEST_CONTENT", context)
        self.assertIn("src/helpers.py", context)
        self.assertIn("src/pkg/core.py", context)
        self.assertIn("README_CONTENT", context)
        self.assertIn("FILE_11_CONTENT", context)
        self.assertIn("tool_return=", context)
        self.assertEqual(context.count("### Tool Observation "), 11)

    def test_context_builder_preserves_last_action_when_reference_context_is_large(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "state" / "benchmarks" / "todo_counter"
            (state_dir / "rejected_candidates").mkdir(parents=True)
            (state_dir / "memory.md").write_text("# Memory\n", encoding="utf-8")
            (state_dir / "project_spec.md").write_text("# Spec\n" + ("spec\n" * 500), encoding="utf-8")
            (state_dir / "handoff.md").write_text(
                "# Worker Session Handoff\n\n"
                "## 9. Evidence Sources\n"
                + ("- noisy evidence\n" * 500)
                + "\n## 10. Last Step Summary\nHANDOFF_STALE_LAST_STEP_MARKER\n"
                + "\n## 15. Suggested Next Action\nRepair the saved INIT candidate.\n",
                encoding="utf-8",
            )
            (state_dir / "rejected_candidates" / "generated_tasks.json").write_text(
                "{\"tasks\": [\n" + (" " * 6000),
                encoding="utf-8",
            )
            state = create_initializer_state(
                "INIT",
                project_spec_artifact="state/benchmarks/todo_counter/project_spec.md",
                generated_tasks_artifact="state/benchmarks/todo_counter/generated_tasks.json",
                init_artifact="state/benchmarks/todo_counter/init.sh",
            )
            state.last_action = {"action": "write", "target": "state/benchmarks/todo_counter/generated_tasks.json"}
            state.last_observation = {
                "ok": False,
                "summary": "INIT write rejected: generated_tasks.json is invalid JSON.",
                "data": {"initializer_validation_errors": ["Expecting ',' delimiter"]},
            }
            state.initializer_repair = {
                "candidate_path": "state/benchmarks/todo_counter/rejected_candidates/generated_tasks.json",
                "validation_errors": ["Expecting ',' delimiter"],
                "repeat_count": 1,
            }

            context = ContextBuilder(root, max_chars=3500, state_dir=state_dir).build(state)

        self.assertIn("# Critical Context", context)
        self.assertIn("## Last Step Summary", context)
        self.assertIn("## Repair Summary", context)
        self.assertIn("## Safety Boundary", context)
        self.assertIn("INIT write rejected", context)
        self.assertIn("Repair the saved INIT candidate", context)
        self.assertNotIn("HANDOFF_STALE_LAST_STEP_MARKER", context)
        self.assertNotIn("# Tail Guard", context)
        self.assertNotIn("reference context omitted", context)
        self.assertNotIn("[context truncated by harness]", context)

    def test_context_builder_does_not_truncate_by_default(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "state"
            state_dir.mkdir()
            (state_dir / "memory.md").write_text("# Memory\n", encoding="utf-8")
            (state_dir / "handoff.md").write_text("# Handoff\n", encoding="utf-8")
            (state_dir / "project_spec.md").write_text("# Spec\n" + ("details\n" * 3000), encoding="utf-8")
            state = create_initial_state("Implement a feature")

            context = ContextBuilder(root).build(state)

        self.assertNotIn("[context truncated by harness]", context)
        self.assertIn("# Critical Context", context)
        self.assertIn("# Session Startup Context", context)

    def test_context_builder_ignores_env_budget_without_truncating(self) -> None:
        previous = os.environ.get("LONG_AGENT_CONTEXT_MAX_CHARS")
        os.environ["LONG_AGENT_CONTEXT_MAX_CHARS"] = "3000"
        try:
            with WorkspaceTemporaryDirectory() as tmp:
                root = Path(tmp)
                state_dir = root / "state"
                state_dir.mkdir()
                (state_dir / "memory.md").write_text("# Memory\n", encoding="utf-8")
                (state_dir / "handoff.md").write_text("# Handoff\n", encoding="utf-8")
                (state_dir / "project_spec.md").write_text("# Spec\n" + ("details\n" * 3000), encoding="utf-8")
                state = create_initial_state("Implement a feature")

                context = ContextBuilder(root).build(state)
        finally:
            if previous is None:
                os.environ.pop("LONG_AGENT_CONTEXT_MAX_CHARS", None)
            else:
                os.environ["LONG_AGENT_CONTEXT_MAX_CHARS"] = previous

        self.assertIn("# Critical Context", context)
        self.assertNotIn("# Tail Guard", context)
        self.assertNotIn("[context truncated by harness]", context)

    def test_context_builder_infers_package_repair_targets_from_import_failure(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "state"
            state_dir.mkdir()
            state = create_initial_state("Implement todo counter")
            state.nodes = [
                {
                    "id": "T2",
                    "title": "Core",
                    "status": "in_progress",
                    "expected_artifacts": ["eval/benchmarks/todo_counter/workspace/core.py"],
                    "verification_commands": [
                        "python -c \"import sys; sys.path.insert(0, 'eval/benchmarks/todo_counter/workspace'); "
                        "from todo_counter.core import count_todos\""
                    ],
                }
            ]
            state.pending_repair = {
                "command": state.nodes[0]["verification_commands"][0],
                "output": "ModuleNotFoundError: No module named 'todo_counter'",
                "targets": ["eval/benchmarks/todo_counter/workspace/core.py"],
                "repair_targets": ["eval/benchmarks/todo_counter/workspace/core.py"],
            }

            context = ContextBuilder(root, state_dir=state_dir).build(state)

        self.assertIn("# Repair Details", context)
        self.assertIn("inferred_import_targets", context)
        self.assertIn("eval/benchmarks/todo_counter/workspace/todo_counter/core.py", context)
        self.assertIn("eval/benchmarks/todo_counter/workspace/todo_counter/__init__.py", context)

    def test_verifier_writes_latest_report(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            (root / "agent").mkdir()
            state = create_initial_state("Implement a feature")
            state.evidence_sources.append(
                {
                    "action": "bash",
                    "target": "python -c \"assert True\"",
                    "summary": "Command exited with code 0.",
                    "task_id": "current",
                    "evidence_type": "acceptance_command_passed",
                    "ok": True,
                }
            )

            result = Verifier(root).run("default", state)
            report = (root / "state" / "verifier_report.md").read_text(encoding="utf-8")

        self.assertTrue(result.ok)
        self.assertIn("Latest Verifier Report", report)
        self.assertIn("Verifier passed", report)
        self.assertIn('"task_id": "current"', report)

    def test_verifier_does_not_use_evidence_as_an_independent_gate(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            state = create_initial_state("Implement a feature")
            state.nodes[0]["evidence"].extend(["scheduled by orchestrator", "Verifier failed."])
            state.evidence_sources.append(
                {
                    "action": "verify",
                    "target": "current",
                    "summary": "Verifier failed.",
                    "task_id": "current",
                    "evidence_type": "verifier_failed",
                    "ok": False,
                }
            )

            result = Verifier(root).run("default", state)

        self.assertTrue(result.ok)
        self.assertNotIn("has_evidence", result.data["checks"])

    def test_benchmark_verifier_does_not_run_host_agent_tests(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "state" / "benchmarks" / "sample"
            (state_dir / "traces").mkdir(parents=True)
            workspace = root / "eval" / "benchmarks" / "sample" / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "feature.py").write_text("VALUE = 1\n", encoding="utf-8")
            host_tests = root / "tests"
            host_tests.mkdir()
            (host_tests / "test_host_failure.py").write_text(
                "import unittest\nclass HostFailure(unittest.TestCase):\n    def test_failure(self):\n        self.fail('host-only failure')\n",
                encoding="utf-8",
            )
            state = create_initial_state("Benchmark feature")
            state.task_id = "T1"
            state.nodes = [{"id": "T1", "title": "Feature", "status": "in_progress", "evidence": []}]
            state.evidence_sources.append(
                {
                    "action": "bash",
                    "target": "python -c \"assert True\"",
                    "summary": "Command exited with code 0.",
                    "task_id": "T1",
                    "evidence_type": "acceptance_command_passed",
                    "ok": True,
                }
            )

            result = Verifier(root, state_dir=state_dir).run("default", state)

        self.assertTrue(result.ok)
        self.assertTrue(result.data["checks"]["unit_tests"])
        self.assertIn("frozen requirement procedures run separately", result.data["test_output"])

    def test_verifier_executes_frozen_commands_and_generates_evidence(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "state" / "benchmarks" / "sample"
            (state_dir / "traces").mkdir(parents=True)
            workspace = root / "eval" / "benchmarks" / "sample" / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "feature.py").write_text("VALUE = 2\n", encoding="utf-8")
            criterion = "Feature value is verified."
            command = (
                "python -c \"import pathlib; ns={}; "
                "exec(pathlib.Path('eval/benchmarks/sample/workspace/feature.py').read_text(), ns); "
                "assert ns['VALUE'] == 2\""
            )
            state = create_initial_state("Benchmark feature")
            state.task_id = "T1"
            state.acceptance_criteria = [criterion]
            state.nodes = [
                {
                    "id": "T1",
                    "title": "Feature",
                    "status": "in_progress",
                    "evidence": [],
                    "acceptance_criteria": [criterion],
                    "criterion_command_map": {criterion: [command]},
                    "expected_artifacts": ["eval/benchmarks/sample/workspace/feature.py"],
                    "verification_commands": [command],
                    "contract_managed": True,
                }
            ]
            state.acceptance_contracts = [
                {
                    "task_id": "T1",
                    "summary": "Frozen task-graph acceptance contract for T1: Feature",
                    "scope": ["eval/benchmarks/sample/workspace/feature.py"],
                    "frozen_requirements": [criterion],
                    "verification_procedure": {"command": command},
                    "checks": [command],
                    "criterion_command_map": {criterion: [command]},
                    "required_evidence": [criterion],
                    "status": "agreed",
                    "source": "task_graph",
                    "frozen": True,
                }
            ]

            result = Verifier(root, state_dir=state_dir).run("default", state)

        self.assertTrue(result.ok)
        self.assertTrue(result.data["checks"]["contract_frozen"])
        self.assertTrue(result.data["checks"]["verification_commands"])
        self.assertEqual(result.data["verification"]["commands"][0]["returncode"], 0)
        self.assertTrue(
            any(item.get("evidence_type") == "verification_command_passed" for item in state.evidence_sources)
        )
        self.assertNotIn("has_evidence", result.data["checks"])

    def test_verifier_freezes_requirements_but_uses_updated_procedure(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "state" / "benchmarks" / "sample"
            (state_dir / "traces").mkdir(parents=True)
            workspace = root / "eval" / "benchmarks" / "sample" / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "feature.py").write_text("VALUE = 2\n", encoding="utf-8")
            criterion = "Feature value is verified."
            wrong_command = "python -m unittest tests.test_feature"
            corrected_command = (
                "python -c \"import pathlib; ns={}; "
                "exec(pathlib.Path('feature.py').read_text(), ns); "
                "assert ns['VALUE'] == 2\""
            )
            state = create_initial_state("Benchmark feature")
            state.task_id = "T1"
            state.acceptance_criteria = [criterion]
            state.nodes = [
                {
                    "id": "T1",
                    "title": "Feature",
                    "status": "in_progress",
                    "evidence": [],
                    "acceptance_criteria": [criterion],
                    "criterion_command_map": {criterion: [wrong_command]},
                    "expected_artifacts": ["eval/benchmarks/sample/workspace/feature.py"],
                    "verification_commands": [wrong_command],
                    "contract_managed": True,
                }
            ]
            state.acceptance_contracts = [
                {
                    "task_id": "T1",
                    "summary": "Frozen task-graph acceptance contract for T1: Feature",
                    "scope": ["eval/benchmarks/sample/workspace/feature.py"],
                    "frozen_requirements": [criterion],
                    "verification_procedure": {
                        "command": corrected_command,
                        "working_directory": "eval/benchmarks/sample/workspace",
                    },
                    "checks": [corrected_command],
                    "criterion_command_map": {criterion: [corrected_command]},
                    "required_evidence": [criterion],
                    "status": "agreed",
                    "source": "task_graph",
                    "frozen": True,
                }
            ]

            result = Verifier(root, state_dir=state_dir).run("default", state)

        self.assertTrue(result.ok)
        self.assertTrue(result.data["contract_validation"]["requirements_match_task_graph"])
        self.assertNotIn(wrong_command, result.data["verification"]["commands"][0]["command"])
        self.assertEqual(result.data["verification"]["commands"][0]["working_directory"], "eval/benchmarks/sample/workspace")

    def test_successful_contract_command_records_structured_task_evidence(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            (root / "state" / "traces").mkdir()
            loop = AgentLoop(root=root, task="Final verification", max_steps=1)
            state = create_initial_state("Final verification")
            state.task_id = "T5"
            state.nodes = [{"id": "T5", "title": "Final", "status": "in_progress", "evidence": []}]
            command = "python -c \"assert True\""
            state.acceptance_contracts = [
                {
                    "task_id": "T5",
                    "summary": "Final verification.",
                    "checks": [command],
                    "status": "agreed",
                }
            ]

            loop._update_state(
                state,
                {"action": "bash", "target": command, "args": {}},
                ToolResult(True, "Command exited with code 0.", {"command": command}),
            )

        evidence = state.evidence_sources[-1]
        self.assertEqual(evidence["task_id"], "T5")
        self.assertEqual(evidence["evidence_type"], "acceptance_command_passed")
        self.assertTrue(evidence["ok"])
        self.assertIn("Acceptance command passed.", state.nodes[0]["evidence"])

    def test_verifier_does_not_run_manual_hidden_acceptance(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "state" / "benchmarks" / "sample"
            (state_dir / "traces").mkdir(parents=True)
            hidden = root / "eval" / "benchmarks" / "sample" / "hidden_acceptance.py"
            hidden.parent.mkdir(parents=True)
            hidden.write_text(
                "import pathlib, sys\n"
                "pathlib.Path('hidden-verifier-was-run.txt').write_text('unexpected', encoding='utf-8')\n"
                "sys.exit(1)\n",
                encoding="utf-8",
            )
            state = create_initial_state("Final verification")
            state.task_id = "T5"
            state.acceptance_criteria = ["Public verification passes"]
            state.nodes = [{"id": "T5", "title": "Final", "status": "in_progress", "evidence": []}]
            state.evidence_sources.append(
                {
                    "action": "bash",
                    "target": "python -c \"assert True\"",
                    "summary": "Command exited with code 0.",
                    "task_id": "T5",
                    "evidence_type": "acceptance_command_passed",
                    "ok": True,
                }
            )

            result = Verifier(root, state_dir=state_dir).run("default", state)
            report = (state_dir / "verifier_report.md").read_text(encoding="utf-8")

        self.assertTrue(result.ok)
        self.assertNotIn("hidden_acceptance", result.data["checks"])
        self.assertNotIn("hidden_acceptance", result.data)
        self.assertFalse((root / "hidden-verifier-was-run.txt").exists())
        self.assertNotIn("hidden_acceptance", report)

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
        self.assertEqual(summary["max_turn_used_tokens"], 30)
        self.assertEqual(summary["completed_tasks"], 1)
        self.assertEqual(summary["blocked_tasks"], 1)

    def test_metrics_reports_skill_loading_and_validation(self) -> None:
        with WorkspaceTemporaryDirectory() as tmp:
            trace = Path(tmp) / "run.jsonl"
            events = [
                {
                    "action": {"action": "load_skill"},
                    "observation": {"ok": True, "summary": "Skill loaded.", "data": {}},
                    "skill_catalog_size": 3,
                    "nodes": [],
                },
                {
                    "action": {"action": "load_skill"},
                    "observation": {
                        "ok": True,
                        "summary": "Skill already loaded.",
                        "data": {"already_loaded": True},
                    },
                    "skill_catalog_size": 3,
                    "nodes": [],
                },
                {
                    "action": {"action": "verify"},
                    "observation": {
                        "ok": True,
                        "summary": "Verifier passed.",
                        "data": {
                            "skill_validation": [
                                {"name": "locate-error", "status": "verified_pass", "tool_calls_since_load": 2}
                            ]
                        },
                    },
                    "skill_catalog_size": 3,
                    "nodes": [],
                },
            ]
            trace.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
            summary = summarize(trace)

        self.assertEqual(summary["skill_metadata_impressions"], 9)
        self.assertEqual(summary["skill_loads"], 2)
        self.assertEqual(summary["duplicate_skill_loads_avoided"], 1)
        self.assertEqual(summary["skill_validation_passes"], 1)
        self.assertEqual(summary["average_tool_calls_from_skill_load_to_validation"], 2)


if __name__ == "__main__":
    unittest.main()
