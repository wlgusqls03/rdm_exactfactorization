from __future__ import annotations

import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from transferable_rdm.systems import build_npz_mmap_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Build persistent mmap caches for an NPZ corpus.")
    parser.add_argument("--npz-glob", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--num-systems", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.npz_glob))
    if args.num_systems > 0:
        paths = paths[: args.num_systems]
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.npz_glob}")

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["RDM_NPZ_MMAP_CACHE_DIR"] = str(cache_dir)

    workers = max(args.workers, 1)
    start = time.perf_counter()
    print(f"[NPZ mmap cache] files={len(paths)} workers={workers} dir={cache_dir}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(build_npz_mmap_cache, path): path for path in paths}
        for completed, future in enumerate(as_completed(futures), start=1):
            future.result()
            if (
                completed == 1
                or completed % max(args.progress_every, 1) == 0
                or completed == len(paths)
            ):
                elapsed = time.perf_counter() - start
                print(
                    f"[NPZ mmap cache] {completed}/{len(paths)} | "
                    f"elapsed {elapsed:.1f}s | "
                    f"rate {completed / max(elapsed, 1e-9):.2f} systems/s"
                )


if __name__ == "__main__":
    main()
