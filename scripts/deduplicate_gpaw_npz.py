#!/usr/bin/env python3
"""Report or remove duplicate GPAW NPZ files for the same QM9 ID."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete duplicate files. Without this flag the command is a dry-run.",
    )
    return parser.parse_args()


def parse_path(path: Path) -> tuple[int, str]:
    parts = path.stem.split("_", 1)
    if len(parts) != 2 or not parts[0].isdigit():
        raise ValueError(f"Unexpected GPAW NPZ filename: {path.name}")
    return int(parts[0]), parts[1]


def main() -> None:
    args = parse_args()
    grouped: dict[str, list[Path]] = defaultdict(list)
    ignored: list[Path] = []
    for path in sorted(args.dataset_dir.glob("*.npz")):
        try:
            _, qm9_id = parse_path(path)
        except ValueError:
            ignored.append(path)
            continue
        grouped[qm9_id].append(path)

    duplicate_groups = {
        qm9_id: sorted(paths, key=lambda path: (parse_path(path)[0], path.name))
        for qm9_id, paths in grouped.items()
        if len(paths) > 1
    }
    removable = sum(len(paths) - 1 for paths in duplicate_groups.values())
    print(
        f"files={sum(len(paths) for paths in grouped.values())} "
        f"unique_qm9_ids={len(grouped)} duplicate_groups={len(duplicate_groups)} "
        f"removable_files={removable}"
    )
    for qm9_id, paths in duplicate_groups.items():
        print(f"[keep]   {qm9_id}: {paths[0].name}")
        for path in paths[1:]:
            print(f"[{'delete' if args.delete else 'would delete'}] {path.name}")
            if args.delete:
                path.unlink()
                xyz_path = args.dataset_dir / "xyz" / f"{path.stem}.xyz"
                if xyz_path.exists():
                    xyz_path.unlink()
    if ignored:
        print(f"ignored_nonstandard_files={len(ignored)}")
    if removable and not args.delete:
        print("Dry-run only. Re-run with --delete after reviewing the keep/delete list.")


if __name__ == "__main__":
    main()
