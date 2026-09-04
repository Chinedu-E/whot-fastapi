import os

os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

import pytest
from uuid import uuid4
from datetime import datetime

from api.core.config import settings
from api.models.game import Card, Deck, GameCreate, GameState, GameStatus
from api.services.redis_service import RedisService


@pytest.fixture
async def fake_redis():
    fakeredis = pytest.importorskip("fakeredis")
    server = fakeredis.FakeServer()
    client = fakeredis.aioredis.FakeRedis(server=server)
    yield client
    await client.aclose()


@pytest.mark.asyncio
async def test_set_game_state_sets_ttl(fake_redis):
    svc = RedisService(fake_redis)
    game_id = uuid4()
    state = GameState(
        game_id=game_id,
        deck=Deck(cards=[]),
        market=Deck(cards=[]),
        discard_pile=[Card(shape="circle", number=3)],
        players=[],
        spectators=[],
        num_spectators=0,
        status=GameStatus.WAITING_FOR_PLAYERS,
        current_face_card=Card(shape="circle", number=3),
        time_elapsed=0,
        created_at=datetime.utcnow(),
        settings=GameCreate(num_players=2),
    )
    await svc.add_player_to_game(game_id, uuid4())
    await svc.set_game_state(game_id, state)

    ttl = await fake_redis.ttl(f"game:{game_id}")
    assert 0 < ttl <= settings.game_ttl_seconds

    state.status = GameStatus.COMPLETED
    await svc.add_active_game(game_id)
    await svc.set_game_state(game_id, state)
    ttl = await fake_redis.ttl(f"game:{game_id}")
    assert 0 < ttl <= settings.completed_game_ttl_seconds
    active = await svc.get_active_game_ids()
    assert game_id not in active


@pytest.mark.asyncio
async def test_delete_game_removes_all(fake_redis):
    svc = RedisService(fake_redis)
    game_id = uuid4()
    player_id = uuid4()
    state = GameState(
        game_id=game_id,
        deck=Deck(cards=[]),
        market=Deck(cards=[]),
        discard_pile=[Card(shape="circle", number=3)],
        players=[],
        spectators=[],
        num_spectators=0,
        status=GameStatus.IN_PROGRESS,
        current_face_card=Card(shape="circle", number=3),
        time_elapsed=0,
        created_at=datetime.utcnow(),
        settings=GameCreate(num_players=2),
    )
    await svc.set_game_state(game_id, state)
    await svc.add_player_to_game(game_id, player_id)
    await svc.add_active_game(game_id)
    await svc.set_player_session(player_id, game_id)

    await svc.delete_game(game_id)
    assert await svc.get_game_state(game_id) is None
    assert await svc.get_game_players(game_id) == []
    assert game_id not in await svc.get_active_game_ids()
    assert await svc.get_player_game(player_id) is None
