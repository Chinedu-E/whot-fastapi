"""Bridge RL Action objects to the sync Whot engine."""

from __future__ import annotations

import random
from uuid import UUID

from api.models.game import GameState
from rl.actions import Action
from rl.engine import ApplyResult, apply_pick, apply_play


def apply_action(
    state: GameState,
    player_id: UUID,
    action: Action,
    *,
    require_eligible: bool = False,
    rng: random.Random | None = None,
) -> ApplyResult:
    if action.kind == "PICK":
        return apply_pick(
            state,
            player_id,
            require_eligible=require_eligible,
            rng=rng,
        )
    if action.kind == "PLAY":
        return apply_play(
            state,
            player_id,
            list(action.cards),
            requested_shape=action.requested_shape,
            require_eligible=require_eligible,
            rng=rng,
        )
    return ApplyResult(success=False, error=f"Unknown action kind: {action.kind}")
