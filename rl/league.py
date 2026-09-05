"""Light historical checkpoint league for self-play opponents."""

from __future__ import annotations

import random
import re
from pathlib import Path

from sb3_contrib import MaskablePPO

from rl.policy_opponent import FrozenPolicyOpponent

_SNAP_RE = re.compile(r"snap_(\d+)\.zip$")


class League:
    """Keep up to ``max_size`` frozen snapshots; sample uniformly as opponents."""

    def __init__(self, directory: Path, max_size: int = 3):
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.directory = Path(directory)
        self.max_size = max_size
        self.directory.mkdir(parents=True, exist_ok=True)
        self._next_index = self._discover_next_index()

    def _discover_next_index(self) -> int:
        indices = []
        for path in self.paths():
            m = _SNAP_RE.search(path.name)
            if m:
                indices.append(int(m.group(1)))
        return (max(indices) + 1) if indices else 0

    def paths(self) -> list[Path]:
        snaps = sorted(self.directory.glob("snap_*.zip"))
        return snaps

    def oldest_path(self) -> Path | None:
        paths = self.paths()
        return paths[0] if paths else None

    def add_snapshot(self, model: MaskablePPO) -> Path:
        path = self.directory / f"snap_{self._next_index:04d}"
        self._next_index += 1
        model.save(str(path))
        zip_path = path.with_suffix(".zip")
        self._trim()
        return zip_path

    def _trim(self) -> None:
        paths = self.paths()
        while len(paths) > self.max_size:
            oldest = paths.pop(0)
            oldest.unlink(missing_ok=True)

    def sample(self, rng: random.Random) -> FrozenPolicyOpponent:
        paths = self.paths()
        if not paths:
            raise RuntimeError("League is empty; add a snapshot before sampling")
        path = paths[rng.randrange(len(paths))]
        return FrozenPolicyOpponent.load(path, deterministic=True)

    def __len__(self) -> int:
        return len(self.paths())
