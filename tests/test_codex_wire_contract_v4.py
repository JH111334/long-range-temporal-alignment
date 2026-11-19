"""Tests for Codex CLI 0.157.1 Wire Contract (Hotfix V4).
Enforces exact byte-level stdout format, exit codes, and matcher filtering.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ADAPTER_PATH = Path(r"C:\Users\GF63\.skills\global\agent-system-v2\adapters\codex.py")
HOOKS_CONFIG_PATH = Path(__file__).resolve().parent.parent / ".codex" / "hooks.json"

# Import direct functions from codex adapter
sys.path.insert(0, str(ADAPTER_PATH.parent))
from codex import format_codex_stdout, handle_parse_failure_fallback, adapt_codex_event


class TestCodexWireContractDirect:
    """Unit tests verifying format_codex_stdout directly."""

    def test_pre_tool_use_allow_emits_empty_stdout(self):
        engine_res = {"decision": "allow"}
        exit_code, stdout, stderr = format_codex_stdout("PreToolUse", engine_res)
        assert exit_code == 0
        assert stdout == b""
        assert stderr == b""

    def test_pre_tool_use_deny_emits_hook_specific_output(self):
        engine_res = {"decision": "deny", "reason": "Path outside writable root"}
        exit_code, stdout, stderr = format_codex_stdout("PreToolUse", engine_res)
        assert exit_code == 0
        assert stderr == b""
        data = json.loads(stdout.decode("utf-8"))
        assert "hookSpecificOutput" in data
        assert data["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
        assert data["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert data["hookSpecificOutput"]["permissionDecisionReason"] == "Path outside writable root"

    def test_post_tool_use_allow_emits_empty_stdout(self):
        engine_res = {"decision": "allow"}
        exit_code, stdout, stderr = format_codex_stdout("PostToolUse", engine_res)
        assert exit_code == 0
        assert stdout == b""
        assert stderr == b""

    def test_post_tool_use_block_emits_structured_decision(self):
        engine_res = {"decision": "block", "reason": "Integrity violation"}
        exit_code, stdout, stderr = format_codex_stdout("PostToolUse", engine_res)
        assert exit_code == 0
        assert stderr == b""
        data = json.loads(stdout.decode("utf-8"))
        assert data["decision"] == "block"
        assert data["reason"] == "Integrity violation"
        assert data["hookSpecificOutput"]["hookEventName"] == "PostToolUse"

    def test_stop_allow_emits_continue_true(self):
        engine_res = {"decision": "allow"}
        exit_code, stdout, stderr = format_codex_stdout("Stop", engine_res)
        assert exit_code == 0
        assert stderr == b""
        data = json.loads(stdout.decode("utf-8"))
        assert data == {"continue": True}

    def test_subagent_stop_allow_emits_continue_true(self):
        engine_res = {"decision": "allow"}
        exit_code, stdout, stderr = format_codex_stdout("SubagentStop", engine_res)
        assert exit_code == 0
        assert stderr == b""
        data = json.loads(stdout.decode("utf-8"))
        assert data == {"continue": True}

    def test_permission_request_allow(self):
        engine_res = {"decision": "allow"}
        exit_code, stdout, stderr = format_codex_stdout("PermissionRequest", engine_res)
        assert exit_code == 0
        assert stderr == b""
        data = json.loads(stdout.decode("utf-8"))
        assert data["hookSpecificOutput"]["hookEventName"] == "PermissionRequest"
        assert data["hookSpecificOutput"]["decision"]["behavior"] == "allow"

    def test_permission_request_deny(self):
        engine_res = {"decision": "deny", "reason": "Restricted command"}
        exit_code, stdout, stderr = format_codex_stdout("PermissionRequest", engine_res)
        assert exit_code == 0
        assert stderr == b""
        data = json.loads(stdout.decode("utf-8"))
        assert data["hookSpecificOutput"]["hookEventName"] == "PermissionRequest"
        assert data["hookSpecificOutput"]["decision"]["behavior"] == "deny"
        assert data["hookSpecificOutput"]["decision"]["message"] == "Restricted command"

    def test_user_prompt_submit_system_message(self):
        engine_res = {"decision": "allow", "system_hint": "[WorkflowHint] Phase: AUDIT"}
        exit_code, stdout, stderr = format_codex_stdout("UserPromptSubmit", engine_res)
        assert exit_code == 0
        assert stderr == b""
        data = json.loads(stdout.decode("utf-8"))
        assert data["systemMessage"] == "[WorkflowHint] Phase: AUDIT"

    def test_session_start_additional_context(self):
        engine_res = {"decision": "allow", "system_hint": "[AgentSystemV2] Project=demo"}
        exit_code, stdout, stderr = format_codex_stdout("SessionStart", engine_res)
        assert exit_code == 0
        assert stderr == b""
        data = json.loads(stdout.decode("utf-8"))
        assert data["hookSpecificOutput"]["hookEventName"] == "SessionStart"
        assert data["hookSpecificOutput"]["additionalContext"] == "[AgentSystemV2] Project=demo"


class TestCodexParseFailureFallbacks:
    """Verifies fail-closed behavior on corrupted/unparseable stdin."""

    def test_pre_tool_use_parse_failure_is_fail_closed(self):
        exit_code, stdout, stderr = handle_parse_failure_fallback("PreToolUse")
        assert exit_code == 2
        assert stdout == b""
        assert b"FAIL_CLOSED" in stderr

    def test_post_tool_use_parse_failure_exits_zero_cleanly(self):
        exit_code, stdout, stderr = handle_parse_failure_fallback("PostToolUse")
        assert exit_code == 0
        assert stdout == b""

    def test_stop_parse_failure_allows_continuation(self):
        exit_code, stdout, stderr = handle_parse_failure_fallback("Stop")
        assert exit_code == 0
        assert stdout == b'{"continue": true}'

    def test_unknown_event_parse_failure_exits_zero(self):
        exit_code, stdout, stderr = handle_parse_failure_fallback("UnknownEvent")
        assert exit_code == 0
        assert stdout == b""


class TestCodexSubprocessWireContract:
    """Subprocess integration tests with actual stdin and CLI invocations."""

    def run_adapter(self, event: str, stdin_payload: bytes) -> subprocess.CompletedProcess:
        cmd = [sys.executable, str(ADAPTER_PATH), "--event", event]
        return subprocess.run(
            cmd,
            input=stdin_payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_subprocess_pre_tool_use_allow_exact_bytes(self):
        payload = json.dumps({
            "toolName": "write_to_file",
            "toolArgs": {"target_file": r"c:\Users\GF63\Documents\long-range-temporal-alignment\test.txt"},
            "cwd": r"c:\Users\GF63\Documents\long-range-temporal-alignment",
            "sessionId": "test-session-wire-01",
        }).encode("utf-8")

        res = self.run_adapter("PreToolUse", payload)
        assert res.returncode == 0
        assert res.stdout == b"", f"Expected empty stdout on allow, got: {res.stdout}"

    def test_subprocess_post_tool_use_receipt_exact_bytes(self):
        payload = json.dumps({
            "toolName": "write_to_file",
            "cwd": r"c:\Users\GF63\Documents\long-range-temporal-alignment",
            "sessionId": "test-session-wire-01",
        }).encode("utf-8")

        res = self.run_adapter("PostToolUse", payload)
        assert res.returncode == 0
        assert res.stdout == b"", f"Expected empty stdout on receipt, got: {res.stdout}"

    def test_subprocess_stop_exact_bytes(self):
        payload = json.dumps({
            "cwd": r"c:\Users\GF63\Documents\long-range-temporal-alignment",
            "sessionId": "test-session-wire-01",
        }).encode("utf-8")

        res = self.run_adapter("Stop", payload)
        assert res.returncode == 0
        data = json.loads(res.stdout.decode("utf-8"))
        assert data == {"continue": True}

    def test_subprocess_pre_tool_use_corrupted_stdin_fails_closed(self):
        res = self.run_adapter("PreToolUse", b"INVALID_NON_JSON_DATA{{{")
        assert res.returncode == 2
        assert res.stdout == b""
        assert b"FAIL_CLOSED" in res.stderr

    def test_subprocess_stop_corrupted_stdin_recovers(self):
        res = self.run_adapter("Stop", b"INVALID_NON_JSON_DATA{{{")
        assert res.returncode == 0
        assert res.stdout == b'{"continue": true}'


class TestMatcherExclusion:
    """Verifies matcher regex eliminates Bash and read-only tools while preserving write/subagent tools."""

    @pytest.fixture
    def hooks_config(self):
        with open(HOOKS_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_pre_and_post_tool_matcher_excludes_bash_and_read_tools(self, hooks_config):
        pre_matcher = hooks_config["hooks"]["PreToolUse"][0]["matcher"]
        post_matcher = hooks_config["hooks"]["PostToolUse"][0]["matcher"]

        assert pre_matcher == post_matcher
        pattern = re.compile(pre_matcher)

        # Mutating tools MUST match
        assert pattern.match("apply_patch")
        assert pattern.match("write_to_file")
        assert pattern.match("edit_file")
        assert pattern.match("replace_file_content")
        assert pattern.match("invoke_subagent")

        # Non-mutating and shell tools MUST NOT match
        assert not pattern.match("Bash")
        assert not pattern.match("bash")
        assert not pattern.match("read_file")
        assert not pattern.match("list_directory")
        assert not pattern.match("view_file")
        assert not pattern.match("grep")
        assert not pattern.match("read_url_content")
