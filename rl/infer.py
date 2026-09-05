"""Production RL inference for live CPU games.

Loads ``production_model.zip`` once and returns legal Actions via FrozenPolicyOpponent.
Falls back gracefully when the checkpoint or optional RL deps are missing.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from rl.policy_opponent import FrozenPolicyOpponent

logger = logging.getLogger(__name__)

# server/ root (parent of rl/)
_SERVER_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CHECKPOINT = _SERVER_ROOT / "rl" / "checkpoints" / "production_model.zip"

_lock = threading.Lock()
_opponent: FrozenPolicyOpponent | None = None
_load_attempted = False
_load_error: str | None = None


def default_checkpoint_path() -> Path:
    return _DEFAULT_CHECKPOINT


def get_rl_opponent(checkpoint: Path | str | None = None) -> FrozenPolicyOpponent | None:
    """Return a process-wide frozen policy, or None if unavailable."""
    global _opponent, _load_attempted, _load_error
    with _lock:
        if _opponent is not None:
            return _opponent
        if _load_attempted:
            return None
        _load_attempted = True
        path = Path(checkpoint) if checkpoint else _DEFAULT_CHECKPOINT
        if not path.is_absolute():
            path = _SERVER_ROOT / path
        if not path.exists():
            _load_error = f"checkpoint not found: {path}"
            logger.warning("RL inference disabled: %s", _load_error)
            return None
        try:
            _opponent = FrozenPolicyOpponent.load(path, deterministic=True)
            logger.info("Loaded RL production checkpoint from %s", path)
            return _opponent
        except Exception as exc:
            _load_error = str(exc)
            logger.exception("Failed to load RL checkpoint from %s", path)
            return None


def reset_rl_opponent_for_tests() -> None:
    """Clear cached opponent (tests only)."""
    global _opponent, _load_attempted, _load_error
    with _lock:
        _opponent = None
        _load_attempted = False
        _load_error = None
