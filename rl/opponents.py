"""Sync heuristic opponents for WhotEnv (no Redis / async)."""

from __future__ import annotations

import random
from typing import Literal
from uuid import UUID

from api.models.game import Card, GameState, Player
from rl.actions import Action, legal_actions
from rl.engine import ACTION_NUMBERS, is_valid_play

OpponentName = Literal["random", "easy", "normal"]
EnvOpponent = Literal["random", "easy", "normal", "policy"]


def _player(state: GameState, player_id: UUID) -> Player | None:
    return next((p for p in state.players if p.id == player_id), None)


def _pick_action() -> Action:
    return Action(kind="PICK")


def _play_cards(cards: list[Card], requested_shape: str | None = None) -> Action:
    return Action(kind="PLAY", cards=tuple(cards), requested_shape=requested_shape)


def _choose_whot_shape(player: Player) -> str:
    shape_counts: dict[str, int] = {}
    for card in player.cards:
        if card.number != 20:
            shape_counts[card.shape] = shape_counts.get(card.shape, 0) + 1
    if shape_counts:
        return max(shape_counts, key=shape_counts.get)  # type: ignore[arg-type]
    return "circle"


def _match_legal(desired: Action, legal: list[Action]) -> Action | None:
    """Find a legal action matching kind/cards/shape (Whot shape must match)."""
    for action in legal:
        if action.kind != desired.kind:
            continue
        if action.kind == "PICK":
            return action
        if len(action.cards) != len(desired.cards):
            continue
        desired_keys = sorted((c.shape, c.number) for c in desired.cards)
        action_keys = sorted((c.shape, c.number) for c in action.cards)
        if desired_keys != action_keys:
            continue
        if desired.cards and desired.cards[-1].number == 20:
            if action.requested_shape == desired.requested_shape:
                return action
        else:
            return action
    return None


def _ensure_legal(desired: Action, legal: list[Action], rng: random.Random) -> Action:
    matched = _match_legal(desired, legal)
    if matched is not None:
        return matched
    if legal:
        return legal[rng.randrange(len(legal))]
    return _pick_action()


def choose_random(state: GameState, player_id: UUID, rng: random.Random) -> Action:
    legal = legal_actions(state, player_id)
    if not legal:
        return _pick_action()
    return legal[rng.randrange(len(legal))]


def choose_easy(state: GameState, player_id: UUID, rng: random.Random) -> Action:
    legal = legal_actions(state, player_id)
    if not legal:
        return _pick_action()

    player = _player(state, player_id)
    if not player:
        return _pick_action()

    if state.pick_chain_count > 0:
        singles = [
            a for a in legal
            if a.kind == "PLAY" and len(a.cards) == 1
        ]
        if singles:
            return singles[rng.randrange(len(singles))]
        return _pick_action()

    singles = [
        a for a in legal
        if a.kind == "PLAY" and len(a.cards) == 1
    ]
    if not singles:
        return _pick_action()
    return singles[rng.randrange(len(singles))]


def choose_normal(state: GameState, player_id: UUID, rng: random.Random) -> Action:
    legal = legal_actions(state, player_id)
    if not legal:
        return _pick_action()

    player = _player(state, player_id)
    if not player:
        return _pick_action()

    if state.pick_chain_count > 0:
        defend = [a for a in legal if a.kind == "PLAY"]
        if not defend:
            return _pick_action()
        # Prefer largest stack (all defenders), matching CPU normal
        defend.sort(key=lambda a: len(a.cards), reverse=True)
        best_len = len(defend[0].cards)
        candidates = [a for a in defend if len(a.cards) == best_len]
        return candidates[rng.randrange(len(candidates))]

    playable = [
        c for c in player.cards
        if is_valid_play(c, state.current_face_card, state)
    ]
    if not playable:
        return _pick_action()

    # Best same-number stack among playable
    by_number: dict[int, list[Card]] = {}
    for card in playable:
        by_number.setdefault(card.number, []).append(card)

    best_number = None
    max_count = 1
    for number, cards in by_number.items():
        if len(cards) > max_count:
            max_count = len(cards)
            best_number = number

    if best_number is not None and max_count > 1:
        stack = list(by_number[best_number])
        shape = _choose_whot_shape(player) if stack[-1].number == 20 else None
        desired = _play_cards(stack, shape)
        return _ensure_legal(desired, legal, rng)

    action_cards = [c for c in playable if c.number in ACTION_NUMBERS]
    if action_cards:
        pick_cards = [c for c in action_cards if c.number in (2, 5)]
        card = rng.choice(pick_cards if pick_cards else action_cards)
    else:
        card = rng.choice(playable)

    shape = _choose_whot_shape(player) if card.number == 20 else None
    desired = _play_cards([card], shape)
    return _ensure_legal(desired, legal, rng)


def choose_action(
    name: OpponentName,
    state: GameState,
    player_id: UUID,
    rng: random.Random,
) -> Action:
    if name == "easy":
        return choose_easy(state, player_id, rng)
    if name == "normal":
        return choose_normal(state, player_id, rng)
    return choose_random(state, player_id, rng)
