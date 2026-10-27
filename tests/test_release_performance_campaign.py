import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaigns/release_performance"


def test_canfar_campaign_has_fail_closed_resource_contract() -> None:
    payload = json.loads((CAMPAIGN / "campaign.json").read_text(encoding="utf-8"))
    assert payload["container_image_source"] == (
        "campaigns/release_performance/image.sh:TORCHSKY_RELEASE_PERFORMANCE_IMAGE"
    )
    assert "container_image" not in payload
    assert "container_digest" not in payload
    assert payload["resources"] == {
        "cpu": 8,
        "memory_gb": 32,
        "gpu": 0,
        "walltime_hours": 6,
    }
    assert payload["concurrency"]["jobs"] == 1
    assert payload["concurrency"]["subjects_per_process"] == 1
    assert payload["retry"] == {
        "max_attempts": 1,
        "unit": "complete_campaign",
        "reuse_partial_pairs": False,
    }


def test_canfar_scripts_enforce_headless_hygiene_and_persistence() -> None:
    runner = (CAMPAIGN / "run_canfar.sh").read_text(encoding="utf-8")
    launcher = (CAMPAIGN / "launch_canfar.sh").read_text(encoding="utf-8")
    image_policy = (CAMPAIGN / "image.sh").read_text(encoding="utf-8")
    assert image_policy.count("sha256:") == 1
    assert 'source "${script_root}/image.sh"' in runner
    assert 'source "${script_root}/image.sh"' in launcher
    assert 'container_image} == "${TORCHSKY_RELEASE_PERFORMANCE_IMAGE}"' in runner
    assert "export PYTHONNOUSERSITE=1" in runner
    assert "unset PYTHONPATH" in runner
    assert "/scratch/code" in runner
    assert "/scratch/env" in runner
    assert "/scratch/data" in runner
    assert "[[ ! -e ${persistent_dest} ]]" in runner
    assert "artifact_root=/scratch/data/release-performance-output" in runner
    assert 'cp -a "${artifact_root}/." "${persistent_dest}/"' in runner
    assert 'cmp -s "${source_file}" "${persistent_dest}/${relative}"' in runner
    assert "exit 70" in runner
    assert "PERSISTENCE_FAILED.json" in runner
    assert (
        "manifest.json release_performance_claims.json release_performance_readiness.json"
        in runner
    )
    assert "trap on_exit EXIT" in runner
    assert "create headless" in launcher
    assert "--submit" in launcher
    assert "container_digest" in launcher
    assert "--container-digest" not in launcher
    assert "container_digest=${image##*@}" in launcher
    assert 'create headless "${image}"' in launcher
