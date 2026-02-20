from __future__ import annotations

import json
import os
import pickle
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional
import datetime as dt
import numpy as np
import h5py


def ensure_dir(path: str | os.PathLike[str]) -> None:
    p = Path(path)
    if str(p) == "":
        return
    p.mkdir(parents=True, exist_ok=True)


def _json_convert(obj: Any) -> Any:
    if is_dataclass(obj):
        return _json_convert(asdict(obj))
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (dict,)):
        return {str(k): _json_convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_convert(v) for v in obj]
    if isinstance(obj, (Path,)):
        return str(obj)
    return obj


def save_json(data: Any, path: str | os.PathLike[str], *, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_convert(data), f, indent=indent, sort_keys=True)

def save_pickle(data: Any, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def save_npz(path: str | os.PathLike[str], **arrays: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez(path, **arrays)


def save_dict_h5(
    data: Dict[str, Any],
    path: str | os.PathLike[str],
    *,
    compression: str | None = "gzip",
    compression_opts: int | None = 4,
    shuffle: bool = True,
) -> str:
    """Save a nested dict to HDF5.
    - numpy arrays -> datasets
    - dict -> groups
    - scalars/strings -> attrs on their group
    """
    path = Path(path).expanduser().resolve()
    ensure_dir(path.parent)

    def _write_group(g: h5py.Group, d: Dict[str, Any]) -> None:
        for k, v in d.items():
            key = str(k)
            if isinstance(v, dict):
                sg = g.create_group(key)
                _write_group(sg, v)
            elif isinstance(v, np.ndarray):
                g.create_dataset(
                    key,
                    data=v,
                    compression=compression,
                    compression_opts=compression_opts,
                    shuffle=shuffle,
                )
            elif v is None:
                g.attrs[key] = "__none__"
            elif isinstance(v, (str, int, float, bool, np.integer, np.floating)):
                g.attrs[key] = v
            else:
                # fallback to JSON string
                g.attrs[key] = json.dumps(_json_convert(v))

    with h5py.File(path, "w") as f:
        _write_group(f, data)

    return str(path)


def load_dict_h5(path: str | os.PathLike[str]) -> Dict[str, Any]:
    """Load a nested dict saved by save_dict_h5."""
    path = Path(path).expanduser().resolve()

    def _read_group(g: h5py.Group) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        # attrs
        for k, v in g.attrs.items():
            if isinstance(v, (bytes, bytearray)):
                v = v.decode("utf-8")
            if v == "__none__":
                out[k] = None
            else:
                out[k] = v
        # children
        for k, item in g.items():
            if isinstance(item, h5py.Dataset):
                out[k] = item[()]
            else:
                out[k] = _read_group(item)
        return out

    with h5py.File(path, "r") as f:
        return _read_group(f)
