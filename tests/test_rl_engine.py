"""Unit tests for the sync Whot rules engine (no Redis)."""

import random
from datetime import datetime
from uuid import uuid4

from api.core.constants import DECK_DICT
from api.models.game import Card, Deck, GameCreate, GameState, GameStatus, Player
from rl import engine as whot_engine


def _settings(**kwargs) -> GameCreate:
    base = dict(
        num_players=2,
        fill_with_computers=False,
        num_starting_cards=4,
        can_win_with_action=False,
    )
    base.update(kwargs)
    return GameCreate(**base)


def _two_player_state(
    p1_cards: list[Card],
    p2_cards: list[Card],
    face: Card,
    *,
    market: list[Card] | None = None,
    settings: GameCreate | None = None,
) -> tuple[GameState, Player, Player]:
    p1 = Player(id=uuid4(), name="Alice", cards=list(p1_cards))
    p2 = Player(id=uuid4(), name="Bob", cards=list(p2_cards))
    market_cards = market if market is not None else [
        Card(shape="star", number=4),
        Card(shape="star", number=7),
        Card(shape="triangle", number=10),
        Card(shape="cross", number=11),
        Card(shape="square", number=13),
        Card(shape="circle", number=12),
    ]
    state = GameState(
        game_id=uuid4(),
        deck=Deck(cards=[]),
        market=Deck(cards=market_cards),
        discard_pile=[face],
        players=[p1, p2],
        spectators=[],
        num_spectators=0,
        status=GameStatus.IN_PROGRESS,
        current_face_card=face,
        current_player_id=p1.id,
        time_elapsed=0,
        created_at=datetime.utcnow(),
        started_at=datetime.utcnow(),
        settings=settings or _settings(),
    )
    return state, p1, p2


def test_deck_composition():
    cards, face = whot_engine.create_deck()
    all_cards = cards + [face]
    expected = sum(len(nums) for nums in DECK_DICT.values()) + 5
    assert len(all_cards) == expected
    assert sum(1 for c in all_cards if c.number == 20) == 5
    assert whot_engine.DECK_SIZE == expected


def test_opening_face_never_whot():
    for seed in range(200):
        cards, face = whot_engine.create_deck(rng=random.Random(seed))
        assert face.number != 20
        assert face.shape != "whot"
        all_cards = cards + [face]
        assert sum(1 for c in all_cards if c.number == 20) == 5
        assert len(all_cards) == whot_engine.DECK_SIZE


def test_valid_play_and_stack():
    state, p1, _ = _two_player_state(
        [
            Card(shape="circle", number=7),
            Card(shape="triangle", number=7),
            Card(shape="star", number=4),
        ],
        [Card(shape="square", number=3)],
        Card(shape="circle", number=3),
    )
    face = state.current_face_card
    assert whot_engine.is_valid_play(Card(shape="circle", number=7), face, state)
    assert whot_engine.is_valid_play_stack(
        [Card(shape="circle", number=7)],
        face,
        state,
        p1,
    )
    assert whot_engine.is_valid_play_stack(
        [Card(shape="circle", number=7), Card(shape="triangle", number=7)],
        face,
        state,
        p1,
    )
    assert not whot_engine.is_valid_play_stack(
        [Card(shape="circle", number=7), Card(shape="star", number=4)],
        face,
        state,
        p1,
    )


def test_hold_on_keeps_turn():
    state, p1, _ = _two_player_state(
        [Card(shape="circle", number=1), Card(shape="star", number=4)],
        [Card(shape="square", number=3)],
        Card(shape="circle", number=3),
    )
    result = whot_engine.apply_play(
        state, p1.id, [Card(shape="circle", number=1)], require_eligible=False
    )
    assert result.success
    assert state.current_player_id == p1.id
    assert state.current_face_card.number == 1


def test_suspension_two_players_returns_to_same():
    state, p1, _ = _two_player_state(
        [Card(shape="triangle", number=8), Card(shape="star", number=4)],
        [Card(shape="square", number=3)],
        Card(shape="triangle", number=3),
    )
    result = whot_engine.apply_play(
        state, p1.id, [Card(shape="triangle", number=8)], require_eligible=False
    )
    assert result.success
    assert state.current_player_id == p1.id


def test_suspension_three_players_skips_next():
    settings = _settings(num_players=3)
    a, b, c = uuid4(), uuid4(), uuid4()
    players = [
        Player(id=a, name="A", cards=[Card(shape="circle", number=8), Card(shape="star", number=4)]),
        Player(id=b, name="B", cards=[Card(shape="star", number=3)]),
        Player(id=c, name="C", cards=[Card(shape="star", number=7)]),
    ]
    state = GameState(
        game_id=uuid4(),
        deck=Deck(cards=[]),
        market=Deck(cards=[Card(shape="star", number=4)] * 10),
        discard_pile=[Card(shape="circle", number=3)],
        players=players,
        spectators=[],
        num_spectators=0,
        status=GameStatus.IN_PROGRESS,
        current_face_card=Card(shape="circle", number=3),
        current_player_id=a,
        time_elapsed=0,
        created_at=datetime.utcnow(),
        started_at=datetime.utcnow(),
        settings=settings,
    )
    result = whot_engine.apply_play(
        state, a, [Card(shape="circle", number=8)], require_eligible=False
    )
    assert result.success
    assert state.current_player_id == c
    assert state.current_face_card.number == 8


def test_general_market_others_pick_keep_turn():
    state, p1, p2 = _two_player_state(
        [Card(shape="circle", number=14), Card(shape="star", number=4)],
        [Card(shape="square", number=3)],
        Card(shape="circle", number=3),
    )
    p2_before = len(p2.cards)
    market_before = len(state.market.cards)
    result = whot_engine.apply_play(
        state, p1.id, [Card(shape="circle", number=14)], require_eligible=False
    )
    assert result.success
    assert len(p2.cards) == p2_before + 1
    assert len(state.market.cards) == market_before - 1
    assert state.current_player_id == p1.id


def test_pick_two_chain_and_draw():
    state, p1, p2 = _two_player_state(
        [Card(shape="cross", number=2), Card(shape="star", number=4)],
        [Card(shape="square", number=3)],
        Card(shape="cross", number=7),
    )
    result = whot_engine.apply_play(
        state, p1.id, [Card(shape="cross", number=2)], require_eligible=False
    )
    assert result.success
    assert state.pick_chain_count == 1
    assert state.pick_chain_type == 2
    assert state.current_player_id == p2.id

    before = len(p2.cards)
    result = whot_engine.apply_pick(state, p2.id, require_eligible=False)
    assert result.success
    assert len(p2.cards) == before + 2
    assert state.pick_chain_count == 0


def test_pick_five_chain_count():
    state, p1, p2 = _two_player_state(
        [Card(shape="star", number=5), Card(shape="circle", number=4)],
        [Card(shape="square", number=3)],
        Card(shape="star", number=3),
    )
    result = whot_engine.apply_play(
        state, p1.id, [Card(shape="star", number=5)], require_eligible=False
    )
    assert result.success
    assert state.pick_chain_count == 1
    assert state.pick_chain_type == 5
    assert state.current_player_id == p2.id


def test_cannot_win_with_action_card():
    state, p1, _ = _two_player_state(
        [Card(shape="circle", number=1)],
        [Card(shape="square", number=3)],
        Card(shape="circle", number=3),
    )
    result = whot_engine.apply_play(
        state, p1.id, [Card(shape="circle", number=1)], require_eligible=False
    )
    assert not result.success
    assert result.error == "Cannot win with action card"
    assert state.status == GameStatus.IN_PROGRESS
    assert len(p1.cards) == 1


def test_can_win_with_normal_card():
    state, p1, _ = _two_player_state(
        [Card(shape="circle", number=7)],
        [Card(shape="square", number=3)],
        Card(shape="circle", number=3),
    )
    result = whot_engine.apply_play(
        state, p1.id, [Card(shape="circle", number=7)], require_eligible=False
    )
    assert result.success
    assert result.game_ended
    assert result.winner_id == p1.id
    assert state.status == GameStatus.COMPLETED


def test_whot_sets_requested_shape():
    state, p1, p2 = _two_player_state(
        [Card(shape="whot", number=20), Card(shape="star", number=4)],
        [Card(shape="square", number=3)],
        Card(shape="circle", number=3),
    )
    result = whot_engine.apply_play(
        state,
        p1.id,
        [Card(shape="whot", number=20)],
        requested_shape="triangle",
        require_eligible=False,
    )
    assert result.success
    assert state.current_face_card.number == 20
    assert state.current_face_card.shape == "triangle"
    assert state.current_player_id == p2.id


def test_reset_market_reshuffles_discard_except_face():
    face = Card(shape="circle", number=3)
    state, _, _ = _two_player_state(
        [Card(shape="star", number=4)],
        [Card(shape="square", number=3)],
        face,
        market=[],
    )
    state.discard_pile = [
        Card(shape="cross", number=1),
        Card(shape="triangle", number=7),
        face,
    ]
    whot_engine.reset_market(state)
    assert state.discard_pile == [face]
    assert len(state.market.cards) == 2
    assert all(c.number != 3 or c.shape != "circle" for c in state.market.cards)


def test_next_turn_skips_disconnected_when_require_eligible():
    state, p1, p2 = _two_player_state(
        [Card(shape="star", number=4)],
        [Card(shape="square", number=3)],
        Card(shape="circle", number=3),
    )
    p2.is_connected = False
    p2.is_cpu = False
    whot_engine.next_turn(state, require_eligible=True)
    assert state.current_player_id == p1.id


def test_apply_use_whot():
    state, p1, p2 = _two_player_state(
        [Card(shape="whot", number=20), Card(shape="star", number=4)],
        [Card(shape="square", number=3)],
        Card(shape="circle", number=3),
    )
    result = whot_engine.apply_use_whot(
        state, p1.id, "cross", require_eligible=False
    )
    assert result.success
    assert state.current_face_card.shape == "cross"
    assert state.current_face_card.number == 20
    assert state.current_player_id == p2.id
    assert len(p1.cards) == 1
