"""The evaluation runner must never implicitly install its containing repo."""
import subprocess

import pytest

from torchsky_harness.scripts import run_local_sphere_bench as runner


def test_missing_native_subject_fails_without_install(monkeypatch):
    commands = []

    def fail(command, **kwargs):
        commands.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(runner.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        runner._ensure_local_cpp()
    assert len(commands) == 1
    assert commands[0][1:] == ["-c", "import torchsky; import torchsky.cpp"]
