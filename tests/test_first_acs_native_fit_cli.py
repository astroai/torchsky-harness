"""Bounded CLI contract; synthetic checks never require the real FITS input."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parents[1] / "campaigns" / "cosmos"
RUNNER = HERE / "first_acs_native_fit.py"
PROTOCOL = HERE / "first_acs_native_fit_protocol_2026-10-18.json"


def test_output_directory_is_required():
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--selftest-only"], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "--output-dir" in result.stderr


def test_selftest_only_records_actual_identity_without_real_io(tmp_path):
    protocol = json.loads(PROTOCOL.read_text())
    # An attempted real stat/hash/open would fail on this deliberately absent file.
    protocol["input"]["path"] = str(tmp_path / "absent-real-input.fits")
    protocol_path = tmp_path / "test-protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    output = tmp_path / "results"
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--output-dir",
            str(output),
            "--protocol",
            str(protocol_path),
            "--selftest-only",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["real_input_accessed"] is False
    subject = json.loads((output / "subject.json").read_text())
    assert subject["runner_path"] == str(RUNNER)
    assert subject["runner_sha256"] == hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    assert subject["protocol_path"] == str(protocol_path)
    assert (
        subject["protocol_sha256"]
        == hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    )
    assert subject["output_dir"] == str(output)
    assert subject["execution_mode"] == "selftest-only"
    assert subject["torchsky_path"]
    assert subject["core_epsf_sha256"]
    assert json.loads((output / "selftest.json").read_text())["passed"] is True
    assert set(p.name for p in output.iterdir()) == {"subject.json", "selftest.json"}


def test_default_protocol_selftest(tmp_path):
    output = tmp_path / "default-protocol-results"
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--output-dir", str(output), "--selftest-only"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    subject = json.loads((output / "subject.json").read_text())
    assert subject["protocol_path"] == str(PROTOCOL)
    assert not (output / "real_attempt_started.json").exists()
    assert not (output / "real_result.json").exists()


def test_changed_recipe_is_rejected_before_output_or_real_io(tmp_path):
    protocol = json.loads(PROTOCOL.read_text())
    protocol["input"]["path"] = str(tmp_path / "absent-real-input.fits")
    protocol["proposal"]["rank_count"] = 3
    protocol_path = tmp_path / "unsupported-protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    output = tmp_path / "must-not-be-created"
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--output-dir",
            str(output),
            "--protocol",
            str(protocol_path),
            "--selftest-only",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "ValueError: Unsupported protocol modifications" in result.stderr
    assert "proposal.rank_count" in result.stderr
    assert "only input.path relocation is supported" in result.stderr
    assert not output.exists()
