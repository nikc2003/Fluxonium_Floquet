from __future__ import annotations

import json
import time
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Dict, Iterator, Optional, Any, List, Tuple


@dataclass
class StageTiming:
    seconds: float = 0.0
    calls: int = 0
    def add(self, dt: float) -> None:
        self.seconds += float(dt)
        self.calls += 1
    @property
    def avg_seconds(self) -> float:
        return self.seconds / self.calls if self.calls else 0.0
class Timing:
    def __init__(self) -> None:
        self._stats: Dict[str, StageTiming] = {}
        self._t0 = time.perf_counter()
    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            stat = self._stats.get(name)
            if stat is None:
                stat = StageTiming()
                self._stats[name] = stat
            stat.add(dt)

    def wall_seconds(self) -> float:
        return time.perf_counter() - self._t0

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for k, st in self._stats.items():
            out[k] = {
                "seconds": float(st.seconds),
                "calls": float(st.calls),
                "avg_seconds": float(st.avg_seconds),
            }
        out["__wall__"] = {"seconds": float(self.wall_seconds()), "calls": 1.0, "avg_seconds": float(self.wall_seconds())}
        return out

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def top(self, n: int = 8) -> List[Tuple[str, float, int]]:
        items = [(k, st.seconds, st.calls) for k, st in self._stats.items()]
        items.sort(key=lambda t: t[1], reverse=True)
        return items[: int(n)]

    def report(self, n: int = 10) -> str:
        rows = self.top(n)
        lines = ["stage                               seconds    calls   avg"]
        for name, sec, calls in rows:
            avg = sec / calls if calls else 0.0
            lines.append(f"{name:<35} {sec:>8.3f}  {calls:>6d}  {avg:>7.4f}")
        lines.append(f"{'__wall__':<35} {self.wall_seconds():>8.3f}")
        return "\n".join(lines)


def fmt_seconds(x: float) -> str:
    x = float(x)
    if x < 60:
        return f"{x:.1f}s"
    if x < 3600:
        return f"{x/60:.1f}m"
    return f"{x/3600:.2f}h"