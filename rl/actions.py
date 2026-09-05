"""Legal action enumeration for Whot RL agents."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal
from uuid import UUID

from api.models.game import Card, GameState, Player
from rl.engine import WHOT_SHAPES, is_valid_play_stack


@dataclass(frozen=True)
class Action:
    kind: Literal["PICK", "PLAY"]
    cards: tuple[Card, ...] = ()
    requested_shape: str | None = None


def action_key(action: Action) -> tuple:
    """Stable sort key for deterministic tests and masking."""
    card_keys = tuple((c.shape, c.number) for c in action.cards)
    return (action.kind, card_keys, action.requested_shape or "")


def _player_by_id(state: GameState, player_id: UUID) -> Player | None:
    return next((p for p in state.players if p.id == player_id), None)


def _expand_play(cards: list[Card]) -> list[Action]:
    """Emit PLAY actions; Whot plays expand to one action per requested shape."""
    if cards and cards[-1].number == 20:
        return [
            Action(kind="PLAY", cards=tuple(cards), requested_shape=shape)
            for shape in WHOT_SHAPES
        ]
    return [Action(kind="PLAY", cards=tuple(cards), requested_shape=None)]


def _same_number_stacks(player: Player, number: int) -> list[list[Card]]:
    """All non-empty subsets of the player's cards with the given number (singles included)."""
    matching = [c for c in player.cards if c.number == number]
    stacks: list[list[Card]] = []
    for size in range(1, len(matching) + 1):
        for combo in combinations(range(len(matching)), size):
            stacks.append([matching[i] for i in combo])
    return stacks


def legal_actions(state: GameState, player_id: UUID) -> list[Action]:
    """Enumerate legal actions for ``player_id`` on their turn.

    Always includes PICK when it is that player's turn. Whot plays expand into
    one PLAY per shape in WHOT_SHAPES.
    """
    if state.current_player_id != player_id:
        return []

    player = _player_by_id(state, player_id)
    if not player:
        return []

    actions: list[Action] = [Action(kind="PICK")]

    if state.pick_chain_count > 0:
        chain_type = state.pick_chain_type
        if chain_type is None:
            return sorted(actions, key=action_key)
        for stack in _same_number_stacks(player, chain_type):
            if is_valid_play_stack(stack, state.current_face_card, state, player):
                actions.extend(_expand_play(stack))
        return sorted(actions, key=action_key)

    # Singles and same-number stacks
    seen_stacks: set[tuple[tuple[str, int], ...]] = set()
    numbers_in_hand = {c.number for c in player.cards}
    for number in numbers_in_hand:
        for stack in _same_number_stacks(player, number):
            stack_key = tuple((c.shape, c.number) for c in stack)
            if stack_key in seen_stacks:
                continue
            seen_stacks.add(stack_key)
            if is_valid_play_stack(stack, state.current_face_card, state, player):
                actions.extend(_expand_play(stack))

    return sorted(actions, key=action_key)
