"""Frozen MaskablePPO used as a WhotEnv opponent (no gradient updates)."""

from __future__ import annotations

import random
from pathlib import Path
from uuid import UUID

import numpy as np
from sb3_contrib import MaskablePPO

from api.models.game import GameState
from rl.actions import Action, legal_actions
from rl.encode import encode_observation

_MAX_ACTIONS = 256


class FrozenPolicyOpponent:
    """Play from a seat using a loaded MaskablePPO checkpoint."""

    def __init__(self, model: MaskablePPO, *, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    @classmethod
    def load(cls, checkpoint: Path | str, *, deterministic: bool = True) -> FrozenPolicyOpponent:
        model = MaskablePPO.load(str(checkpoint))
        return cls(model, deterministic=deterministic)

    def choose(self, state: GameState, player_id: UUID, rng: random.Random) -> Action:
        legal = legal_actions(state, player_id)
        if not legal:
            return Action(kind="PICK")

        obs = np.asarray(encode_observation(state, player_id), dtype=np.float32)
        mask = np.zeros(_MAX_ACTIONS, dtype=np.bool_)
        n = min(len(legal), _MAX_ACTIONS)
        mask[:n] = True

        try:
            action_idx, _ = self.model.predict(
                obs,
                action_masks=mask,
                deterministic=self.deterministic,
            )
            idx = int(action_idx)
        except Exception:
            idx = -1

        if idx < 0 or idx >= len(legal):
            pick = next((a for a in legal if a.kind == "PICK"), legal[0])
            return pick
        return legal[idx]
