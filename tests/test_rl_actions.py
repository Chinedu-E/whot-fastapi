"""Tests for legal action enumeration."""

from datetime import datetime
from uuid import uuid4

from api.models.game import Card, Deck, GameCreate, GameState, GameStatus, Player
from rl.actions import legal_actions
from rl.engine import WHOT_SHAPES, is_valid_play_stack


def _settings(**kwargs) -> GameCreate:
    base = dict(
        num_players=2,
        fill_with_computers=False,
        num_starting_cards=4,
        can_win_with_action=False,
    )
    base.update(kwargs)
    return GameCreate(**base)


def _state(
    p1_cards: list[Card],
    p2_cards: list[Card],
    face: Card,
    *,
    current_is_p1: bool = True,
    pick_chain_count: int = 0,
    pick_chain_type: int | None = None,
) -> tuple[GameState, Player, Player]:
    p1 = Player(id=uuid4(), name="Alice", cards=list(p1_cards))
    p2 = Player(id=uuid4(), name="Bob", cards=list(p2_cards))
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
        current_player_id=p1.id if current_is_p1 else p2.id,
        time_elapsed=0,
        pick_chain_count=pick_chain_count,
        pick_chain_type=pick_chain_type,  # type: ignore[arg-type]
        created_at=datetime.utcnow(),
        started_at=datetime.utcnow(),
        settings=_settings(),
    )
    return state, p1, p2


def test_not_your_turn_returns_empty():
    state, p1, p2 = _state(
        [Card(shape="circle", number=7)],
        [Card(shape="square", number=3)],
        Card(shape="circle", number=3),
        current_is_p1=True,
    )
    assert legal_actions(state, p2.id) == []


def test_always_includes_pick_and_valid_plays():
    state, p1, _ = _state(
        [
            Card(shape="circle", number=7),
            Card(shape="triangle", number=7),
            Card(shape="star", number=4),
        ],
        [Card(shape="square", number=3)],
        Card(shape="circle", number=3),
    )
    actions = legal_actions(state, p1.id)
    assert any(a.kind == "PICK" for a in actions)

    play_actions = [a for a in actions if a.kind == "PLAY"]
    assert play_actions
    for action in play_actions:
        assert is_valid_play_stack(
            list(action.cards),
            state.current_face_card,
            state,
            p1,
        )

    # star-4 does not match circle-3
    assert not any(
        len(a.cards) == 1 and a.cards[0].shape == "star" and a.cards[0].number == 4
        for a in play_actions
    )


def test_whot_expands_to_five_shapes():
    state, p1, _ = _state(
        [Card(shape="whot", number=20), Card(shape="star", number=4)],
        [Card(shape="square", number=3)],
        Card(shape="circle", number=3),
    )
    actions = legal_actions(state, p1.id)
    whot_plays = [
        a for a in actions
        if a.kind == "PLAY" and len(a.cards) == 1 and a.cards[0].number == 20
    ]
    shapes = {a.requested_shape for a in whot_plays}
    assert shapes == set(WHOT_SHAPES)
    assert len(whot_plays) == 5


def test_pick_chain_only_allows_matching_number_or_pick():
    state, p1, _ = _state(
        [
            Card(shape="cross", number=2),
            Card(shape="circle", number=2),
            Card(shape="star", number=5),
            Card(shape="circle", number=7),
        ],
        [Card(shape="square", number=3)],
        Card(shape="cross", number=7),
        pick_chain_count=1,
        pick_chain_type=2,
    )
    actions = legal_actions(state, p1.id)
    assert any(a.kind == "PICK" for a in actions)
    play_actions = [a for a in actions if a.kind == "PLAY"]
    assert play_actions
    assert all(all(c.number == 2 for c in a.cards) for a in play_actions)
    assert not any(any(c.number == 5 for c in a.cards) for a in play_actions)
    assert not any(any(c.number == 7 for c in a.cards) for a in play_actions)


def test_stack_of_same_number_included():
    state, p1, _ = _state(
        [
            Card(shape="circle", number=7),
            Card(shape="triangle", number=7),
        ],
        [Card(shape="square", number=3)],
        Card(shape="circle", number=3),
    )
    actions = legal_actions(state, p1.id)
    stacks = [a for a in actions if a.kind == "PLAY" and len(a.cards) == 2]
    assert len(stacks) == 1
    assert {c.shape for c in stacks[0].cards} == {"circle", "triangle"}
