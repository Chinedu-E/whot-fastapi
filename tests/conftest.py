import os

os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

import pytest
from datetime import datetime
from uuid import uuid4

from api.models.game import Card, Deck, GameCreate, GameState, GameStatus, Player
from api.services.game import WhotGame
from tests.fakes import FakeRedisService


@pytest.fixture
def redis_service():
    return FakeRedisService()


@pytest.fixture
def game_settings():
    return GameCreate(
        num_players=2,
        fill_with_computers=False,
        num_starting_cards=4,
        can_win_with_action=False,
    )


@pytest.fixture
async def two_player_game(redis_service, game_settings):
    """WhotGame with two humans and deterministic hands."""
    game_id = uuid4()
    p1_id = uuid4()
    p2_id = uuid4()
    game = WhotGame(game_id, redis_service)

    face = Card(shape="circle", number=3)
    market = [
        Card(shape="star", number=4),
        Card(shape="star", number=7),
        Card(shape="triangle", number=10),
        Card(shape="cross", number=11),
        Card(shape="square", number=13),
        Card(shape="circle", number=12),
        Card(shape="triangle", number=4),
        Card(shape="cross", number=7),
    ]

    p1 = Player(
        id=p1_id,
        name="Alice",
        cards=[
            Card(shape="circle", number=7),
            Card(shape="circle", number=1),
            Card(shape="triangle", number=8),
            Card(shape="circle", number=14),
            Card(shape="whot", number=20),
            Card(shape="whot", number=20),
            Card(shape="cross", number=2),
            Card(shape="star", number=5),
        ],
    )
    p2 = Player(
        id=p2_id,
        name="Bob",
        cards=[
            Card(shape="square", number=3),
            Card(shape="triangle", number=2),
            Card(shape="circle", number=8),
            Card(shape="star", number=1),
        ],
    )

    state = GameState(
        game_id=game_id,
        deck=Deck(cards=[]),
        market=Deck(cards=market),
        discard_pile=[face],
        players=[p1, p2],
        spectators=[],
        num_spectators=0,
        status=GameStatus.IN_PROGRESS,
        current_face_card=face,
        current_player_id=p1_id,
        time_elapsed=0,
        created_at=datetime.utcnow(),
        started_at=datetime.utcnow(),
        settings=game_settings,
    )
    await redis_service.add_active_game(game_id)
    await redis_service.add_player_to_game(game_id, p1_id)
    await redis_service.add_player_to_game(game_id, p2_id)
    await redis_service.set_game_state(game_id, state)
    await redis_service.set_player_session(p1_id, game_id)
    await redis_service.set_player_session(p2_id, game_id)

    return {
        "game": game,
        "redis": redis_service,
        "game_id": game_id,
        "p1_id": p1_id,
        "p2_id": p2_id,
        "settings": game_settings,
    }
