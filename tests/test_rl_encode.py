"""Tests for seat-relative observation encoding."""

from datetime import datetime
from uuid import uuid4

from api.models.game import Card, Deck, GameCreate, GameState, GameStatus, Player
from rl.encode import (
    HAND_SIZE,
    encode_observation,
    observation_size,
)


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
    players: list[Player],
    face: Card,
    *,
    current_player_id=None,
    market: list[Card] | None = None,
) -> GameState:
    return GameState(
        game_id=uuid4(),
        deck=Deck(cards=[]),
        market=Deck(cards=market or [Card(shape="star", number=4)] * 6),
        discard_pile=[face],
        players=players,
        spectators=[],
        num_spectators=0,
        status=GameStatus.IN_PROGRESS,
        current_face_card=face,
        current_player_id=current_player_id or players[0].id,
        time_elapsed=0,
        created_at=datetime.utcnow(),
        started_at=datetime.utcnow(),
        settings=_settings(num_players=len(players)),
    )


def test_observation_fixed_length():
    p1 = Player(
        id=uuid4(),
        name="A",
        cards=[Card(shape="circle", number=7), Card(shape="whot", number=20)],
    )
    p2 = Player(id=uuid4(), name="B", cards=[Card(shape="square", number=3)] * 3)
    state = _state([p1, p2], Card(shape="circle", number=3))
    obs = encode_observation(state, p1.id)
    assert len(obs) == observation_size()
    assert all(isinstance(x, float) for x in obs)


def test_swapping_opponent_hands_same_counts_unchanged():
    agent = Player(
        id=uuid4(),
        name="Agent",
        cards=[Card(shape="circle", number=7)],
    )
    opp_a = Player(
        id=uuid4(),
        name="OppA",
        cards=[Card(shape="cross", number=1), Card(shape="star", number=5)],
    )
    opp_b = Player(
        id=uuid4(),
        name="OppB",
        cards=[Card(shape="triangle", number=14), Card(shape="square", number=2)],
    )
    face = Card(shape="circle", number=3)

    state1 = _state([agent, opp_a, opp_b], face)
    state2 = _state(
        [
            agent,
            Player(id=opp_a.id, name="OppA", cards=list(opp_b.cards)),
            Player(id=opp_b.id, name="OppB", cards=list(opp_a.cards)),
        ],
        face,
    )
    # Same counts in the same seats → identical observations
    assert encode_observation(state1, agent.id) == encode_observation(state2, agent.id)


def test_changing_agent_hand_changes_obs():
    agent_id = uuid4()
    opp = Player(id=uuid4(), name="Opp", cards=[Card(shape="square", number=3)] * 2)
    face = Card(shape="circle", number=3)

    state1 = _state(
        [Player(id=agent_id, name="A", cards=[Card(shape="circle", number=7)]), opp],
        face,
    )
    state2 = _state(
        [Player(id=agent_id, name="A", cards=[Card(shape="circle", number=1)]), opp],
        face,
    )
    assert encode_observation(state1, agent_id) != encode_observation(state2, agent_id)


def test_no_opponent_card_identities_in_encoding():
    """Opponent hand identities must not appear; only counts in opponent block."""
    agent = Player(
        id=uuid4(),
        name="Agent",
        cards=[Card(shape="circle", number=7)],
    )
    # Distinct unique cards in opponent hands, same length
    opp1 = Player(
        id=uuid4(),
        name="O1",
        cards=[Card(shape="cross", number=14)],
    )
    opp2_state_a = Player(
        id=uuid4(),
        name="O2",
        cards=[Card(shape="star", number=1)],
    )
    opp2_state_b = Player(
        id=opp2_state_a.id,
        name="O2",
        cards=[Card(shape="triangle", number=8)],
    )
    face = Card(shape="circle", number=3)

    state_a = _state([agent, opp1, opp2_state_a], face)
    state_b = _state([agent, opp1, opp2_state_b], face)
    assert encode_observation(state_a, agent.id) == encode_observation(state_b, agent.id)

    # Hand block should only reflect agent's cards — opponent unique numbers
    # must not create extra non-zero hand slots beyond agent hand.
    obs = encode_observation(state_a, agent.id)
    hand = obs[:HAND_SIZE]
    # Exactly one non-whot card in agent hand → exactly one positive catalog slot
    nonzero = sum(1 for v in hand[:-1] if v > 0)
    assert nonzero == 1
    assert hand[-1] == 0.0  # no whot in agent hand


def test_opponent_count_difference_changes_obs():
    agent = Player(id=uuid4(), name="A", cards=[Card(shape="circle", number=7)])
    face = Card(shape="circle", number=3)
    state_small = _state(
        [agent, Player(id=uuid4(), name="B", cards=[Card(shape="star", number=4)])],
        face,
    )
    state_large = _state(
        [
            agent,
            Player(
                id=uuid4(),
                name="B",
                cards=[Card(shape="star", number=4), Card(shape="star", number=7)],
            ),
        ],
        face,
    )
    assert encode_observation(state_small, agent.id) != encode_observation(
        state_large, agent.id
    )
