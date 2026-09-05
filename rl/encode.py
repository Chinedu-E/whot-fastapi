"""Seat-relative observation encoding for Whot RL agents.

Observation layout (float32-compatible list of floats):

1. Hand — multiplicity over CARD_CATALOG (one slot per distinct shape/number
   in DECK_DICT, plus one slot for Whot count). Values are counts / DECK_SIZE.
2. Face — one-hot over FACE_SHAPES + normalized face number + is_whot flag.
3. Meta — pick_chain_count / DECK_SIZE, pick_chain_type one-hot (none/2/5),
   direction in {-1, +1} mapped to [0, 1], market_count / DECK_SIZE,
   discard_count / DECK_SIZE.
4. Opponents — up to MAX_PLAYERS-1 seats in turn order after the agent;
   each is card_count / DECK_SIZE. Unused seats are zeros.
"""

from __future__ import annotations

from uuid import UUID

from api.core.constants import DECK_DICT
from api.models.game import Card, GameState
from rl.engine import DECK_SIZE, WHOT_SHAPES

MAX_PLAYERS = 4
FACE_SHAPES = WHOT_SHAPES + ("whot",)

# Distinct non-whot catalog entries, then one Whot count slot.
CARD_CATALOG: tuple[tuple[str, int], ...] = tuple(
    (shape, number)
    for shape, numbers in sorted(DECK_DICT.items())
    for number in numbers
)
HAND_SIZE = len(CARD_CATALOG) + 1  # +1 for Whot count
FACE_SIZE = len(FACE_SHAPES) + 2  # one-hot shapes + number + is_whot
META_SIZE = 1 + 3 + 1 + 1 + 1  # chain count, type OH, direction, market, discard
OPPONENT_SLOTS = MAX_PLAYERS - 1
OPPONENT_SIZE = OPPONENT_SLOTS  # one normalized count each

_OBS_SIZE = HAND_SIZE + FACE_SIZE + META_SIZE + OPPONENT_SIZE


def observation_size() -> int:
    return _OBS_SIZE


def _hand_features(cards: list[Card]) -> list[float]:
    feats = [0.0] * HAND_SIZE
    catalog_index = {key: i for i, key in enumerate(CARD_CATALOG)}
    whot_count = 0
    for card in cards:
        if card.number == 20:
            whot_count += 1
            continue
        idx = catalog_index.get((card.shape, card.number))
        if idx is not None:
            feats[idx] += 1.0
    for i in range(HAND_SIZE - 1):
        feats[i] /= float(DECK_SIZE)
    feats[-1] = whot_count / float(DECK_SIZE)
    return feats


def _face_features(face: Card) -> list[float]:
    feats = [0.0] * FACE_SIZE
    shape_index = {s: i for i, s in enumerate(FACE_SHAPES)}
    shape = face.shape if face.shape in shape_index else "whot"
    if face.number == 20:
        shape = "whot"
    feats[shape_index[shape]] = 1.0
    feats[len(FACE_SHAPES)] = face.number / 20.0
    feats[len(FACE_SHAPES) + 1] = 1.0 if face.number == 20 else 0.0
    return feats


def _meta_features(state: GameState) -> list[float]:
    feats = [0.0] * META_SIZE
    feats[0] = state.pick_chain_count / float(DECK_SIZE)
    # pick_chain_type one-hot: [none, 2, 5]
    if state.pick_chain_count <= 0 or state.pick_chain_type is None:
        feats[1] = 1.0
    elif state.pick_chain_type == 2:
        feats[2] = 1.0
    else:
        feats[3] = 1.0
    feats[4] = 1.0 if state.direction >= 0 else 0.0
    feats[5] = len(state.market.cards) / float(DECK_SIZE)
    feats[6] = len(state.discard_pile) / float(DECK_SIZE)
    return feats


def _opponent_features(state: GameState, agent_id: UUID) -> list[float]:
    feats = [0.0] * OPPONENT_SIZE
    n = len(state.players)
    if n == 0:
        return feats

    agent_idx = next((i for i, p in enumerate(state.players) if p.id == agent_id), None)
    if agent_idx is None:
        return feats

    direction = state.direction if state.direction != 0 else 1
    slot = 0
    for step in range(1, n):
        if slot >= OPPONENT_SLOTS:
            break
        idx = (agent_idx + direction * step) % n
        if state.players[idx].id == agent_id:
            continue
        feats[slot] = len(state.players[idx].cards) / float(DECK_SIZE)
        slot += 1
    return feats


def encode_observation(state: GameState, player_id: UUID) -> list[float]:
    """Encode ``state`` from ``player_id``'s seat. Does not leak opponent hands."""
    player = next((p for p in state.players if p.id == player_id), None)
    hand = player.cards if player else []
    return (
        _hand_features(hand)
        + _face_features(state.current_face_card)
        + _meta_features(state)
        + _opponent_features(state, player_id)
    )
