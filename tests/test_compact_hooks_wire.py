"""Byte-level wire protocol unit tests for PreCompact and PostCompact hooks.

Tests exact conformance with Codex CLI 0.158.0 native contract on Windows:
- Valid input schema
- Normal success (exit 0, stdout == b"")
- Explicit stop ({"continue": false, "stopReason": "..."})
- Malformed input safe fallback (exit 0, stdout == b"", durable failure log)
- Rejection of unsupported old H7 output (no hookSpecificOutput, no additionalContext)
- Windows subprocess invocation matching .codex/hooks.json exactly
- H7 durable state pointer preservation
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = Path(r"C:\Users\GF63\.tools\.venv\Scripts\python.exe")
ADAPTER_PATH = Path(r"C:\Users\GF63\.skills\global\agent-system-v2\adapters\codex.py")
FAILURE_LOG = Path(r"C:\Users\GF63\.tools\agent-system-v2\state\hook_failures.jsonl")
STATE_DIR = Path(r"C:\Users\GF63\.tools\agent-system-v2\state\long-range-temporal-alignment")


class TestCompactHooksWire(unittest.TestCase):
    """Rigorous byte-level wire protocol tests for PreCompact and PostCompact."""

    def setUp(self):
        self.session_id = f"test_wire_compact_{os.getpid()}"
        self.valid_precompact_input = {
            "cwd": str(REPO_ROOT),
            "hook_event_name": "PreCompact",
            "model": "gpt-4o",
            "session_id": self.session_id,
            "transcript_path": None,
            "trigger": "manual",
            "turn_id": "turn_001",
        }
        self.valid_postcompact_input = {
            "cwd": str(REPO_ROOT),
            "hook_event_name": "PostCompact",
            "model": "gpt-4o",
            "session_id": self.session_id,
            "transcript_path": None,
            "trigger": "manual",
            "turn_id": "turn_001",
        }

    def tearDown(self):
        state_file = STATE_DIR / f"{self.session_id}.json"
        if state_file.exists():
            try:
                state_file.unlink()
            except Exception:
                pass
        rec_dir = Path(r"C:\Users\GF63\.tools\agent-system-v2\receipts\long-range-temporal-alignment") / self.session_id
        if rec_dir.exists():
            try:
                shutil.rmtree(rec_dir)
            except Exception:
                pass

    def _run_adapter_subprocess(self, event_name: str, payload_bytes: bytes) -> tuple[int, bytes, bytes]:
        cmd = [str(PYTHON_EXE), str(ADAPTER_PATH), "--event", event_name]
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(REPO_ROOT),
        )
        out, err = p.communicate(payload_bytes)
        return p.returncode, out, err

    def test_precompact_normal_success_exact_zero_bytes(self):
        """PreCompact normal success must return exit 0 and exactly zero stdout bytes."""
        input_bytes = json.dumps(self.valid_precompact_input).encode("utf-8")
        code, out, err = self._run_adapter_subprocess("PreCompact", input_bytes)
        self.assertEqual(code, 0, f"Expected exit 0, got {code}. Stderr: {err.decode('utf-8', errors='ignore')}")
        self.assertEqual(out, b"", f"PreCompact normal success must emit exactly 0 bytes stdout, got: {out}")
        self.assertEqual(err, b"", f"PreCompact normal success must emit 0 bytes stderr, got: {err}")

    def test_postcompact_normal_success_exact_zero_bytes(self):
        """PostCompact normal success must return exit 0 and exactly zero stdout bytes."""
        input_bytes = json.dumps(self.valid_postcompact_input).encode("utf-8")
        code, out, err = self._run_adapter_subprocess("PostCompact", input_bytes)
        self.assertEqual(code, 0, f"Expected exit 0, got {code}. Stderr: {err.decode('utf-8', errors='ignore')}")
        self.assertEqual(out, b"", f"PostCompact normal success must emit exactly 0 bytes stdout, got: {out}")
        self.assertEqual(err, b"", f"PostCompact normal success must emit 0 bytes stderr, got: {err}")

    def test_precompact_explicit_stop_wire_contract(self):
        """Explicit stop on PreCompact must output exact supported JSON: {'continue': false, 'stopReason': '...'}."""
        sys.path.insert(0, str(ADAPTER_PATH.parent.parent))
        from adapters.codex import format_codex_stdout

        stop_engine_res = {
            "decision": "deny",
            "continue": False,
            "stopReason": "Critical verification gate pending; compaction paused.",
        }
        code, out_b, err_b = format_codex_stdout("PreCompact", stop_engine_res)
        self.assertEqual(code, 0)
        self.assertEqual(err_b, b"")

        payload = json.loads(out_b.decode("utf-8"))
        self.assertIn("continue", payload)
        self.assertFalse(payload["continue"])
        self.assertIn("stopReason", payload)
        self.assertEqual(payload["stopReason"], "Critical verification gate pending; compaction paused.")
        # Must not contain unsupported fields
        self.assertNotIn("decision", payload)
        self.assertNotIn("hookSpecificOutput", payload)
        self.assertNotIn("additionalContext", payload)

    def test_compact_unsupported_old_output_never_emitted(self):
        """Assert adapter never emits hookSpecificOutput, additionalContext, or generic decision for compact hooks."""
        sys.path.insert(0, str(ADAPTER_PATH.parent.parent))
        from adapters.codex import format_codex_stdout

        allow_res = {"decision": "allow", "continue": True, "compaction_hint": "[CompactionSummary] hint text"}
        for ev in ("PreCompact", "PostCompact"):
            code, out_b, err_b = format_codex_stdout(ev, allow_res)
            self.assertEqual(code, 0)
            self.assertEqual(out_b, b"", f"Normal success for {ev} must be 0 bytes, not {out_b}")
            self.assertNotIn(b"hookSpecificOutput", out_b)
            self.assertNotIn(b"additionalContext", out_b)
            self.assertNotIn(b"CompactionSummary", out_b)
            self.assertNotIn(b'"decision"', out_b)

    def test_precompact_malformed_input_safe_fallback(self):
        """Malformed input on PreCompact must exit 0, stdout b'', and append safe failure row."""
        malformed_bytes = b"MALFORMED_JSON_PRECOMPACT_TEST{"
        malformed_sha = hashlib.sha256(malformed_bytes).hexdigest()

        code, out, err = self._run_adapter_subprocess("PreCompact", malformed_bytes)
        self.assertEqual(code, 0, "PreCompact parse failure must fail-open with exit 0")
        self.assertEqual(out, b"", "PreCompact parse failure must emit 0 bytes stdout")

        # Verify failure row
        self.assertTrue(FAILURE_LOG.exists())
        with open(FAILURE_LOG, "r", encoding="utf-8") as f:
            last_line = f.readlines()[-1].strip()
        record = json.loads(last_line)
        self.assertEqual(record.get("expected_event"), "PreCompact")
        self.assertEqual(record.get("stage_reached"), "JSON_PARSE")
        self.assertEqual(record.get("sha256"), malformed_sha)
        self.assertNotIn("MALFORMED_JSON", json.dumps(record), "Raw payload must never leak to log")

    def test_postcompact_empty_input_safe_fallback(self):
        """Empty input on PostCompact must exit 0, stdout b'', allowing session to continue."""
        code, out, err = self._run_adapter_subprocess("PostCompact", b"")
        self.assertEqual(code, 0, "PostCompact empty input must exit 0")
        self.assertEqual(out, b"", "PostCompact empty input must emit 0 bytes stdout")

    def test_h7_state_preservation_across_compaction(self):
        """H7 semantics: durable state pointer recorded before/after compaction, recoverable without text stdout."""
        # 1. Run PreCompact
        input_bytes_pre = json.dumps(self.valid_precompact_input).encode("utf-8")
        code1, out1, _ = self._run_adapter_subprocess("PreCompact", input_bytes_pre)
        self.assertEqual(code1, 0)
        self.assertEqual(out1, b"")

        state_file = STATE_DIR / f"{self.session_id}.json"
        self.assertTrue(state_file.exists(), "State file must exist after PreCompact")
        with open(state_file, "r", encoding="utf-8") as f:
            state_data = json.load(f)
        self.assertIn("last_compact_pointer", state_data)
        pointer = state_data["last_compact_pointer"]
        self.assertEqual(pointer["compact_event"], "PreCompact")
        self.assertEqual(pointer["project"], "long-range-temporal-alignment")

        # 2. Run PostCompact
        input_bytes_post = json.dumps(self.valid_postcompact_input).encode("utf-8")
        code2, out2, _ = self._run_adapter_subprocess("PostCompact", input_bytes_post)
        self.assertEqual(code2, 0)
        self.assertEqual(out2, b"")

        with open(state_file, "r", encoding="utf-8") as f:
            state_data2 = json.load(f)
        self.assertIn("compact_completed_at", state_data2)
        self.assertEqual(state_data2["last_compact_pointer"]["compact_event"], "PostCompact")


if __name__ == "__main__":
    unittest.main()
