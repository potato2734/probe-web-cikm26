"""Generate the static baseline manifest consumed by PROBEWeb.

Only JSON files inside n_infos/<model>/<dataset>/ are included. Run with
--check in CI or before release to verify that the checked-in manifest matches
the baseline files.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINE_ROOT = ROOT / "n_infos"
MANIFEST_PATH = ROOT / "baseline-manifest.json"


def build_datasets() -> dict[str, dict[str, list[str]]]:
    """Return the stable dataset/model/path mapping used by the web app."""
    datasets: dict[str, dict[str, list[str]]] = {}
    for model_dir in sorted(path for path in BASELINE_ROOT.iterdir() if path.is_dir()):
        for dataset_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            files = sorted(path for path in dataset_dir.glob("*.json") if path.is_file())
            if files:
                datasets.setdefault(dataset_dir.name, {})[model_dir.name] = [
                    file.relative_to(ROOT).as_posix() for file in files
                ]
    return dict(sorted(datasets.items()))


def load_manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the checked-in manifest is stale")
    args = parser.parse_args()

    datasets = build_datasets()
    if args.check:
        current = load_manifest()
        if current.get("datasets") != datasets:
            print("baseline-manifest.json is stale; run: python generate_baseline_manifest.py")
            return 1
        print("baseline-manifest.json matches n_infos/")
        return 0

    manifest = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datasets": datasets,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST_PATH.name} for {len(datasets)} datasets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
