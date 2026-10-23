from __future__ import annotations

import os

from torchsky_harness.subject import identify_subject


def main() -> int:
    subject = identify_subject()
    source = subject.get("source")
    if not isinstance(source, dict):
        raise SystemExit("optimization evidence requires source identity")
    if source.get("dirty") is not False:
        raise SystemExit("optimization evidence refuses a dirty torchsky subject")
    expected = os.environ.get("TORCHSKY_EXPECTED_COMMIT") or os.environ.get(
        "TORCHSKY_COMMIT"
    )
    revision = source.get("revision")
    if expected and revision != expected:
        raise SystemExit(
            f"optimization evidence expected torchsky {expected}, found {revision}"
        )
    print(f"torchsky_subject_revision={revision}")
    print(f"torchsky_subject_import_path={subject['import_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
