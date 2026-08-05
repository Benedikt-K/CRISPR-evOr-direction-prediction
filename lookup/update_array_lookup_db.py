#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def normalize_dna(sequence: str) -> str:
    sequence = sequence.upper().replace("U", "T")
    cleaned: list[str] = []
    for char in sequence:
        if char.isspace():
            continue
        cleaned.append(char if char in {"A", "C", "G", "T"} else "N")
    return "".join(cleaned)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_splits_dir() -> Path:
    return _repo_root() / "direction_learning" / "carbon-model" / "outputs" / "carbon-500m-direction" / "splits"


def _default_output_path() -> Path:
    return _repo_root() / "Standalone" / "lookup" / "array_lookup_db.json"


def _latest_splits_dir(outputs_root: Path) -> Path:
    candidates = [path for path in outputs_root.glob("**/splits") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No splits directories found under: {outputs_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _extract_entries_from_jsonl(path: Path, split_name: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        payload = json.loads(line)
        if isinstance(payload.get("example"), dict):
            payload = payload["example"]

        repeats = payload.get("repeats")
        spacers = payload.get("spacers")
        if not isinstance(repeats, list) or not all(isinstance(x, str) for x in repeats):
            continue
        if not isinstance(spacers, list) or not all(isinstance(x, str) for x in spacers):
            continue

        entries.append(
            {
                "split": split_name,
                "array_name": str(payload.get("array_name", f"{split_name}_{line_number}")),
                "repeats": [normalize_dna(sequence) for sequence in repeats if sequence],
                "spacers": [normalize_dna(sequence) for sequence in spacers if sequence],
            }
        )
    return entries


def _split_plan(mode: str) -> list[tuple[str, str]]:
    if mode == "train_val":
        return [("train", "train"), ("val", "val")]
    if mode == "train_all":
        return [("train_all", "train")]
    if mode == "all":
        return [("train", "train"), ("val", "val"), ("test", "test")]
    raise ValueError(f"Unsupported mode: {mode}")


def build_lookup_db(
    *,
    splits_dir: Path,
    mode: str,
    output_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    source_files: dict[str, str] = {}

    for file_stem, split_name in _split_plan(mode):
        split_path = splits_dir / f"{file_stem}.jsonl"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")

        current_entries = _extract_entries_from_jsonl(split_path, split_name=split_name)
        entries.extend(current_entries)
        source_files[file_stem] = str(split_path)

    db = {
        "version": 1,
        "sources": {
            "splits_dir": str(splits_dir),
            "mode": mode,
            "files": source_files,
        },
        "entries": entries,
    }

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(db, indent=2, sort_keys=True))

    return db


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Standalone lookup DB from Carbon split JSONL files."
    )
    parser.add_argument(
        "--splits_dir",
        type=Path,
        default=None,
        help="Path to a splits directory containing train/val/test JSONLs. "
        "Default: direction_learning/carbon-model/outputs/carbon-500m-direction/splits",
    )
    parser.add_argument(
        "--outputs_root",
        type=Path,
        default=None,
        help="Root directory searched when --latest is used. "
        "Default: direction_learning/carbon-model/outputs",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recently modified splits directory under outputs root.",
    )
    parser.add_argument(
        "--mode",
        choices=["train_val", "train_all", "all"],
        default="train_val",
        help="Which split files to include in the lookup DB.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_output_path(),
        help="Output path for array_lookup_db.json in Standalone/lookup.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Parse and summarize only; do not write output file.",
    )

    args = parser.parse_args()

    if args.latest:
        outputs_root = args.outputs_root or (
            _repo_root() / "direction_learning" / "carbon-model" / "outputs"
        )
        splits_dir = _latest_splits_dir(outputs_root)
    else:
        splits_dir = args.splits_dir or _default_splits_dir()

    db = build_lookup_db(
        splits_dir=splits_dir,
        mode=args.mode,
        output_path=args.output,
        dry_run=args.dry_run,
    )

    entry_count = len(db["entries"])
    split_counts: dict[str, int] = {}
    for row in db["entries"]:
        split_name = str(row.get("split", "unknown"))
        split_counts[split_name] = split_counts.get(split_name, 0) + 1

    print(f"splits_dir: {splits_dir}")
    print(f"mode: {args.mode}")
    print(f"entries: {entry_count}")
    print(f"split_counts: {split_counts}")

    if args.dry_run:
        print("dry_run: no file written")
    else:
        print(f"wrote: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
