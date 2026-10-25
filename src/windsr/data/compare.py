"""Compare a built data directory with a reference: every variable and coordinate of the five stores must have the
same dims, dtype and values, NaN equal to NaN. Attribute differences are printed as notes.

usage: python -m windsr.data.compare BUILT REFERENCE
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from .store import STORES, store_path


def same(a: np.ndarray, b: np.ndarray) -> bool:
    if a.dtype.kind == "f":
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


def compare(built: Path, ref: Path) -> list[str]:
    """Mismatch lines, empty when every store equals the reference."""
    bad = []
    for store in STORES:
        r = store_path(ref, store)
        name = r.name
        rf, bt = xr.open_zarr(r), xr.open_zarr(store_path(built, store))
        names = set(rf.variables)
        if names != set(bt.variables):
            bad.append(f"{name}: variables differ {sorted(names ^ set(bt.variables))}")
            continue
        for v in sorted(names):
            a, b = rf[v], bt[v]
            if a.dims != b.dims or a.dtype != b.dtype:
                bad.append(f"{name}:{v} dims/dtype {a.dims} {a.dtype} vs {b.dims} {b.dtype}")
            elif not same(a.values, b.values):
                d = np.nanmax(np.abs(a.values.astype(float) - b.values.astype(float))) if a.dtype.kind == "f" else "n/a"
                bad.append(f"{name}:{v} values differ, max abs diff {d}")
            if a.attrs != b.attrs:
                print(f"note {name}:{v} attrs differ: {a.attrs} vs {b.attrs}")
        if rf.attrs != bt.attrs:
            print(f"note {name} store attrs differ: {rf.attrs} vs {bt.attrs}")
        print(f"{name}: {len(names)} variables and coordinates compared with {r}")
    return bad


def main(argv: list[str]) -> None:
    built, ref = map(Path, argv)
    bad = compare(built, ref)
    if bad:
        print("MISMATCH")
        print("\n".join(bad))
        sys.exit(1)
    print("BUILD EQUALS REFERENCE")


if __name__ == "__main__":
    main(sys.argv[1:])
