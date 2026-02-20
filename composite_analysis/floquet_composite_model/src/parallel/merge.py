from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Optional, Tuple, Dict

import numpy as np
import h5py
import datetime as dt

today = dt.date.today()

def merge_axis0_chunks(
    files: Sequence[str | Path],
    out_path: str | Path,
    *,
    x_key: str,
    y_key: str,
    data_keys: Optional[Sequence[str]] = None,
    extra_axis0_keys: Sequence[str] = (),
    require_same_keys: bool = True,
) -> str:

    files = [Path(p).expanduser().resolve() for p in files]
    out_path = Path(out_path).expanduser().resolve()
    if not files:
        raise ValueError("No chunk files to merge.")

    chunks = []
    for fp in files:
        with h5py.File(fp, "r") as f:
            x = np.asarray(f[x_key][:], dtype=float)
            y = np.asarray(f[y_key][:], dtype=float)
            # discover keys
            if data_keys is None:
                keys = []
                for k, item in f.items():
                    if k in (x_key, y_key):
                        continue
                    if isinstance(item, h5py.Dataset):
                        keys.append(k)
                keys = sorted(keys)
            else:
                keys = list(data_keys)
            # sanity check: datasets shapes
            shapes = {k: tuple(f[k].shape) for k in keys}
            dtypes = {k: f[k].dtype for k in keys}
            extra_shapes = {k: tuple(f[k].shape) for k in extra_axis0_keys if k in f}
            extra_dtypes = {k: f[k].dtype for k in extra_axis0_keys if k in f}

            # metadata snapshot
            meta = {}
            if "_metadata" in f:
                meta_group = f["_metadata"]
                meta["attrs"] = dict(meta_group.attrs.items())
                meta["datasets"] = {k: meta_group[k][:] for k in meta_group.keys()}

            chunks.append((float(x[0]), fp, x, y, keys, shapes, dtypes, extra_shapes, extra_dtypes, meta))

    chunks.sort(key=lambda t: t[0])

    _, fp0, x0, y0, keys0, shapes0, dtypes0, extra_shapes0, extra_dtypes0, meta0 = chunks[0]

    for _, fp, x, y, keys, shapes, dtypes, extra_shapes, extra_dtypes, _ in chunks[1:]:
        if len(y) != len(y0) or not np.allclose(y, y0):
            raise ValueError(f"Chunk {fp} has different y grid than {fp0}.")
        if require_same_keys and set(keys) != set(keys0):
            raise ValueError(f"Chunk {fp} has different dataset keys than {fp0}.")
        # check shape compatibility along axis 0
        for k in keys0:
            if k not in shapes:
                raise ValueError(f"Chunk {fp} missing dataset {k}")
            if len(shapes[k]) != len(shapes0[k]):
                raise ValueError(f"Dataset {k} rank mismatch between chunks")
            if shapes[k][1:] != shapes0[k][1:]:
                raise ValueError(f"Dataset {k} non-axis dims mismatch: {shapes[k]} vs {shapes0[k]}")
        for k in extra_axis0_keys:
            if k in extra_shapes0 and k in extra_shapes:
                if extra_shapes[k] != extra_shapes0[k] and not (len(extra_shapes[k])==1 and len(extra_shapes0[k])==1):
                    raise ValueError(f"Extra axis0 dataset {k} shape mismatch: {extra_shapes[k]} vs {extra_shapes0[k]}")

    x_all = np.concatenate([c[2] for c in chunks])
    y_all = y0
    Nx = len(x_all)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as out:
        out.create_dataset(x_key, data=x_all)
        out.create_dataset(y_key, data=y_all)

        #allocate merged datasets
        for k in keys0:
            shape0 = shapes0[k]
            full_shape = (Nx,) + shape0[1:]
            out.create_dataset(
                k,
                shape=full_shape,
                dtype=dtypes0[k],
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                chunks=(min(64, Nx),) + shape0[1:],
            )

        for k in extra_axis0_keys:
            if k in extra_shapes0:
                out.create_dataset(
                    k,
                    shape=(Nx,),
                    dtype=extra_dtypes0.get(k, "f8"),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                    chunks=(min(1024, Nx),),
                )

        #copy metadata from first chunk if present
        if meta0:
            g = out.create_group("_metadata")
            for ak, av in meta0.get("attrs", {}).items():
                g.attrs[ak] = av
            for dk, dv in meta0.get("datasets", {}).items():
                g.create_dataset(dk, data=dv)

        # write data
        cursor = 0
        for _, fp, x_chunk, _, keys, *_rest in chunks:
            nx = len(x_chunk)
            with h5py.File(fp, "r") as f:
                for k in keys0:
                    out[k][cursor:cursor+nx, ...] = f[k][:]
                for k in extra_axis0_keys:
                    if k in out and k in f:
                        out[k][cursor:cursor+nx] = f[k][:]

            cursor += nx

    return str(out_path)
