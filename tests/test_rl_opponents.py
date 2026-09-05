"""Tests for sync heuristic opponents."""

from datetime import datetime
from uuid import uuid4

from api.models.game import Card, Deck, GameCreate, GameState, GameStatus, Player
from rl.actions import action_key, legal_actions
from rl.opponents import choose_easy, choose_normal, choose_random


def _settings() -> GameCreate:
    return GameCreate(
        num_players=2,
        fill_with_computers=False,
        num_starting_cards=4,
        can_win_with_action=False,
    )


def _state(
    p1_cards: list[Card],
    p2_cards: list[Card],
    face: Card,
    *,
    pick_chain_count: int = 0,
    pick_chain_type: int | None = None,
) -> tuple[GameState, Player, Player]:
    p1 = Player(id=uuid4(), name="A", cards=list(p1_cards))
    p2 = Player(id=uuid4(), name="B", cards=list(p2_cards))
    state = GameState(
        game_id=uuid4(),
        deck=Deck(cards=[]),
        market=Deck(cards=[Card(shape="star", number=4)] * 8),
        discard_pile=[face],
        players=[p1, p2],
        spectators=[],
        num_spectators=0,
        status=GameStatus.IN_PROGRESS,
        current_face_card=face,
        current_player_id=p1.id,
        time_elapsed=0,
        pick_chain_count=pick_chain_count,
        pick_chain_type=pick_chain_type,  # type: ignore[arg-type]
        created_at=datetime.utcnow(),
        started_at=datetime.utcnow(),
        settings=_settings(),
    )
    return state, p1, p2


def test_choose_random_in_legal_set():
    import random

    state, p1, _ = _state(
        [Card(shape="circle", number=7), Card(shape="star", number=4)],
        [Card(shape="square", number=3)],
        Card(shape="circle", number=3),
    )
    legal_keys = {action_key(a) for a in legal_actions(state, p1.id)}
    rng = random.Random(0)
    for _ in range(20):
        action = choose_random(state, p1.id, rng)
        assert action_key(action) in legal_keys


def test_easy_no_stacking_prefers_singles():
    import random

    state, p1, _ = _state(
        [
            Card(shape="circle", number=7),
            Card(shape="triangle", number=7),
        ],
        [Card(shape="square", number=3)],
        Card(shape="circle", number=3),
    )
    rng = random.Random(1)
    for _ in range(30):
        action = choose_easy(state, p1.id, rng)
        legal = legal_actions(state, p1.id)
        assert action_key(action) in {action_key(a) for a in legal}
        if action.kind == "PLAY":
            assert len(action.cards) == 1


def test_normal_pick_chain_stacks_defenders():
    import random

    state, p1, _ = _state(
        [
            Card(shape="cross", number=2),
            Card(shape="circle", number=2),
            Card(shape="star", number=5),
        ],
        [Card(shape="square", number=3)],
        Card(shape="cross", number=7),
        pick_chain_count=1,
        pick_chain_type=2,
    )
    action = choose_normal(state, p1.id, random.Random(0))
    assert action.kind == "PLAY"
    assert len(action.cards) == 2
    assert all(c.number == 2 for c in action.cards)
    legal_keys = {action_key(a) for a in legal_actions(state, p1.id)}
    assert action_key(action) in legal_keys


def test_normal_returns_legal_action():
    import random

    state, p1, _ = _state(
        [
            Card(shape="circle", number=7),
            Card(shape="whot", number=20),
            Card(shape="star", number=4),
        ],
        [Card(shape="square", number=3)],
        Card(shape="circle", number=3),
    )
    legal_keys = {action_key(a) for a in legal_actions(state, p1.id)}
    action = choose_normal(state, p1.id, random.Random(42))
    assert action_key(action) in legal_keys
