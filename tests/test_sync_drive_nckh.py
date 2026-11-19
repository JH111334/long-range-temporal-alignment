"""Kiem thu contract va hanh vi cho script dong bo Google Drive NCKH."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    script = Path(__file__).resolve().parents[1] / "scripts/sync_drive_nckh.py"
    spec = importlib.util.spec_from_file_location("sync_drive_nckh", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_active_allowlist_and_filter():
    sync = _module()
    manifest_path = (
        Path(__file__).resolve().parents[1] / "configs/Shared/sync_nckh_manifest.json"
    )
    data = sync.load_manifest(manifest_path)

    # File hop le duoc phep
    assert sync.allowed("src/temporal_alignment/contracts.py", 100, data)
    assert sync.allowed("Documents/Decision-Making/04_Quyet_dinh_thiet_ke.md", 100, data)
    assert sync.allowed("configs/smoke.json", 100, data)
    assert sync.allowed("AGENTS.md", 100, data)

    # File cache / rac bi loai tru
    assert not sync.allowed("src/temporal_alignment/__pycache__/module.pyc", 100, data)
    assert not sync.allowed(".venv/Scripts/python.exe", 100, data)
    assert not sync.allowed("Documents/Archive/old.md", 100, data)
    assert not sync.allowed("configs/Archive/old.yaml", 100, data)

    # File qua kich thuoc 100MB bi chan mac dinh
    large_path = "src/temporal_alignment/huge_file.bin"
    assert not sync.allowed(large_path, 101 * sync.MIB, data)


def test_build_plan_simulation(tmp_path):
    sync = _module()
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()

    # Gia lap file nguon
    active_file = source / "configs/smoke.json"
    active_file.parent.mkdir(parents=True)
    active_file.write_text('{"test": true}', encoding="utf-8")

    manifest_data = {
        "schema_version": "nckh-active-sync-v1",
        "include_roots": ["configs"],
        "include_files": ["configs/smoke.json"],
        "exclude_roots": ["Documents/Archive", "configs/Archive"],
        "retired_drive_roots": [],
        "protected_drive_roots": ["configs"],
        "large_files": {},
    }

    plan = sync.build_plan(source, target, manifest_data)
    assert "configs/smoke.json" in plan["copy"]
    assert plan["active_file_count"] == 1
    assert len(plan["update"]) == 0
    assert len(plan["delete_stale"]) == 0
