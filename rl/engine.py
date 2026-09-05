"""Sync Whot rules engine — no Redis, WebSockets, or async I/O."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import UUID

from api.core.constants import DECK_DICT
from api.models.game import Card, GameState, GameStatus, Player

ACTION_NUMBERS = frozenset({1, 2, 5, 8, 14, 20})
WHOT_SHAPES = ("circle", "triangle", "cross", "square", "star")
DECK_SIZE = sum(len(nums) for nums in DECK_DICT.values()) + 5


@dataclass(frozen=True)
class ApplyResult:
    success: bool
    error: str | None = None
    game_ended: bool = False
    winner_id: UUID | None = None
    winner_name: str | None = None


def create_deck(rng: random.Random | None = None) -> tuple[list[Card], Card]:
    """Build a shuffled deck and pop the initial face card.

    The opening face card is never a Whot (20) — that would leave shape="whot"
    with no called suit and only another Whot could match.
    """
    shuffle = rng.shuffle if rng is not None else random.shuffle
    cards: list[Card] = []
    for shape, numbers in DECK_DICT.items():
        for number in numbers:
            cards.append(Card(shape=shape, number=number))
    for _ in range(5):
        cards.append(Card(shape="whot", number=20))
    shuffle(cards)
    init_card = cards.pop()
    if init_card.number == 20:
        for i, card in enumerate(cards):
            if card.number != 20:
                cards[i] = init_card
                init_card = card
                break
    return cards, init_card


def is_eligible_turn(player: Player) -> bool:
    return player.is_cpu or player.is_connected


def get_current_player_index(state: GameState) -> int:
    for i, player in enumerate(state.players):
        if player.id == state.current_player_id:
            return i
    return 0


def get_next_player_index(state: GameState) -> int:
    current_idx = get_current_player_index(state)
    return (current_idx + state.direction) % len(state.players)


def next_turn(state: GameState, *, require_eligible: bool = True) -> None:
    if not state.players:
        return
    n = len(state.players)
    current_idx = get_current_player_index(state)
    next_idx = (current_idx + state.direction) % n
    state.current_player_id = state.players[next_idx].id
    if not require_eligible:
        return
    for _ in range(n - 1):
        player = state.players[get_current_player_index(state)]
        if is_eligible_turn(player):
            return
        next_idx = (get_current_player_index(state) + state.direction) % n
        state.current_player_id = state.players[next_idx].id


def can_defend_pick_chain(player: Player, state: GameState) -> bool:
    return any(c.number == state.pick_chain_type for c in player.cards)


def reset_market(state: GameState, rng: random.Random | None = None) -> None:
    if len(state.discard_pile) <= 1:
        return
    shuffle = rng.shuffle if rng is not None else random.shuffle
    current_face = state.discard_pile[-1]
    market_cards = state.discard_pile[:-1]
    shuffle(market_cards)
    state.market.cards = market_cards
    state.discard_pile = [current_face]


def remove_card_from_player(
    player: Player,
    card: Card,
    card_index: Optional[int] = None,
) -> None:
    if card_index is not None and 0 <= card_index < len(player.cards):
        if (
            player.cards[card_index].shape == card.shape
            and player.cards[card_index].number == card.number
        ):
            player.cards.pop(card_index)
            return

    for i, c in enumerate(player.cards):
        if c.shape == card.shape and c.number == card.number:
            player.cards.pop(i)
            return


def is_valid_first_card(card: Card, face_card: Card, state: GameState) -> bool:
    if card.number == 20:
        return True
    if face_card.number == 20:
        return card.shape == state.current_face_card.shape
    return card.shape == face_card.shape or card.number == face_card.number


def is_valid_play(card: Card, face_card: Card, state: GameState) -> bool:
    if state.pick_chain_count > 0:
        if state.pick_chain_type == 2:
            return card.number == 2
        if state.pick_chain_type == 5:
            return card.number == 5
        return False
    if card.number == 20:
        return True
    if face_card.number == 20:
        return card.shape == state.current_face_card.shape
    return card.shape == face_card.shape or card.number == face_card.number


def is_valid_play_stack(
    cards: list[Card],
    face_card: Card,
    state: GameState,
    player: Player,
) -> bool:
    if not cards:
        return False

    if state.pick_chain_count > 0:
        if state.pick_chain_type == 2:
            return all(c.number == 2 for c in cards)
        if state.pick_chain_type == 5:
            return all(c.number == 5 for c in cards)
        return False

    first_card = cards[0]
    if not is_valid_first_card(first_card, face_card, state):
        return False

    if len(cards) > 1:
        first_number = first_card.number
        for card in cards[1:]:
            if card.number != first_number:
                return False

    player_card_counts: dict[tuple[str, int], int] = {}
    for c in player.cards:
        key = (c.shape, c.number)
        player_card_counts[key] = player_card_counts.get(key, 0) + 1

    for card in cards:
        key = (card.shape, card.number)
        if player_card_counts.get(key, 0) == 0:
            return False
        player_card_counts[key] -= 1

    return True


def _apply_whot_shape(state: GameState, requested_shape: str, *, require_eligible: bool) -> None:
    state.current_face_card = Card(shape=requested_shape, number=20)
    next_turn(state, require_eligible=require_eligible)


def _apply_pick_2(state: GameState, *, require_eligible: bool) -> None:
    if state.pick_chain_type == 2:
        state.pick_chain_count += 1
    else:
        state.pick_chain_count = 1
        state.pick_chain_type = 2
    next_turn(state, require_eligible=require_eligible)


def _apply_pick_5(state: GameState, *, require_eligible: bool) -> None:
    if state.pick_chain_type == 5:
        state.pick_chain_count += 1
    else:
        state.pick_chain_count = 1
        state.pick_chain_type = 5
    next_turn(state, require_eligible=require_eligible)


def _apply_suspension(state: GameState, *, require_eligible: bool) -> None:
    next_turn(state, require_eligible=require_eligible)
    next_turn(state, require_eligible=require_eligible)


def _apply_general_market(
    state: GameState,
    rng: random.Random | None = None,
) -> None:
    for player in state.players:
        if player.id != state.current_player_id:
            if len(state.market.cards) == 0:
                reset_market(state, rng=rng)
            if len(state.market.cards) > 0:
                player.cards.append(state.market.cards.pop())
                player.cards_drawn += 1


def _end_game(state: GameState, player: Player) -> ApplyResult:
    state.winner_id = player.id
    state.status = GameStatus.COMPLETED
    state.ended_at = datetime.utcnow()
    return ApplyResult(
        success=True,
        game_ended=True,
        winner_id=player.id,
        winner_name=player.name,
    )


def apply_play(
    state: GameState,
    player_id: UUID,
    cards: list[Card],
    requested_shape: str | None = None,
    card_indices: list[int] | None = None,
    card_index: int | None = None,
    *,
    require_eligible: bool = True,
    rng: random.Random | None = None,
) -> ApplyResult:
    player = next((p for p in state.players if p.id == player_id), None)
    if not player:
        return ApplyResult(success=False, error="Player not found")

    if not cards:
        return ApplyResult(success=False, error="No card specified")

    if not is_valid_play_stack(cards, state.current_face_card, state, player):
        if state.pick_chain_count > 0:
            need = "Pick Two (2)" if state.pick_chain_type == 2 else "Pick Three (5)"
            return ApplyResult(
                success=False,
                error=f"Pick chain active — play a {need} or draw from the market",
            )
        return ApplyResult(success=False, error="That card doesn’t match the face card")

    last_card = cards[-1]
    will_win = len(player.cards) == len(cards)

    if not state.settings.can_win_with_action and will_win:
        if last_card.number in ACTION_NUMBERS:
            return ApplyResult(success=False, error="Cannot win with action card")

    indices = card_indices or []
    for idx, card in enumerate(cards):
        if idx < len(indices):
            remove_idx: int | None = indices[idx]
        elif idx == 0:
            remove_idx = card_index
        else:
            remove_idx = None
        remove_card_from_player(player, card, remove_idx)
        state.discard_pile.append(card)
        state.current_face_card = card

    player.cards_played += len(cards)
    state.last_action = f"PLAY_CARD:{len(cards)}_cards@{player.name}"

    if len(player.cards) == 0:
        return _end_game(state, player)

    if last_card.number == 20:
        if requested_shape:
            _apply_whot_shape(state, requested_shape, require_eligible=require_eligible)
        else:
            next_turn(state, require_eligible=require_eligible)
    elif last_card.number == 2:
        _apply_pick_2(state, require_eligible=require_eligible)
    elif last_card.number == 5:
        _apply_pick_5(state, require_eligible=require_eligible)
    elif last_card.number == 1:
        pass  # hold on: keep turn
    elif last_card.number == 8:
        _apply_suspension(state, require_eligible=require_eligible)
    elif last_card.number == 14:
        _apply_general_market(state, rng=rng)
    else:
        next_turn(state, require_eligible=require_eligible)

    return ApplyResult(success=True)


def apply_pick(
    state: GameState,
    player_id: UUID,
    *,
    require_eligible: bool = True,
    rng: random.Random | None = None,
) -> ApplyResult:
    player = next((p for p in state.players if p.id == player_id), None)
    if not player:
        return ApplyResult(success=False, error="Player not found")

    picked = 0
    if state.pick_chain_count > 0:
        cards_to_pick = state.pick_chain_count * (2 if state.pick_chain_type == 2 else 3)
        for _ in range(cards_to_pick):
            if len(state.market.cards) == 0:
                reset_market(state, rng=rng)
            if len(state.market.cards) > 0:
                player.cards.append(state.market.cards.pop())
                picked += 1
        state.pick_chain_count = 0
        state.pick_chain_type = None
    else:
        if len(state.market.cards) == 0:
            reset_market(state, rng=rng)
        if len(state.market.cards) > 0:
            player.cards.append(state.market.cards.pop())
            picked = 1

    player.cards_drawn += picked
    state.last_action = f"PICK_CARD:{picked}_cards@{player.name}"
    next_turn(state, require_eligible=require_eligible)
    return ApplyResult(success=True)


def apply_use_whot(
    state: GameState,
    player_id: UUID,
    requested_shape: str,
    *,
    require_eligible: bool = True,
) -> ApplyResult:
    if not requested_shape:
        return ApplyResult(success=False, error="Must specify requested shape")

    player = next((p for p in state.players if p.id == player_id), None)
    if not player:
        return ApplyResult(success=False, error="Player not found")

    whot_card = next((c for c in player.cards if c.number == 20), None)
    if not whot_card:
        return ApplyResult(success=False, error="No whot card in hand")

    remove_card_from_player(player, whot_card)
    state.discard_pile.append(whot_card)
    state.current_face_card = Card(shape=requested_shape, number=20)
    player.cards_played += 1
    state.last_action = f"USE_WHOT:{requested_shape}@{player.name}"

    if len(player.cards) == 0:
        return _end_game(state, player)

    next_turn(state, require_eligible=require_eligible)
    return ApplyResult(success=True)
