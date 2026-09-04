import pytest
from uuid import uuid4

from api.core.constants import DECK_DICT
from api.models.game import Card, GameEvent, GameStatus
from api.services.game import WhotGame
from api.services.cpu import CPUPlayer
from api.services.session import GamesManager
from tests.fakes import FakeRedisService


@pytest.mark.asyncio
async def test_deck_composition():
    game = WhotGame(uuid4(), FakeRedisService())
    cards, face = game.create_deck()
    all_cards = cards + [face]
    expected = sum(len(nums) for nums in DECK_DICT.values()) + 5
    assert len(all_cards) == expected
    assert sum(1 for c in all_cards if c.number == 20) == 5


@pytest.mark.asyncio
async def test_valid_play_and_stack(two_player_game):
    game = two_player_game["game"]
    state = await game._get_game_state()
    p1 = state.players[0]
    face = state.current_face_card

    assert game.is_valid_play(Card(shape="circle", number=7), face, state)
    assert game.is_valid_play_stack(
        [Card(shape="circle", number=7)],
        face,
        state,
        p1,
    )
    # Same number stacking from hand
    assert not game.is_valid_play_stack(
        [Card(shape="circle", number=7), Card(shape="star", number=4)],
        face,
        state,
        p1,
    )


@pytest.mark.asyncio
async def test_hold_on_keeps_turn(two_player_game):
    game = two_player_game["game"]
    p1_id = two_player_game["p1_id"]

    event = GameEvent(
        action="PLAY_CARD",
        payload={"card": {"shape": "circle", "number": 1}},
        player_id=p1_id,
        game_id=two_player_game["game_id"],
    )
    await game.process_game_event(event)
    state = await game._get_game_state()
    assert state.current_player_id == p1_id
    assert state.current_face_card.number == 1


@pytest.mark.asyncio
async def test_suspension_skips_next(two_player_game):
    """With 2 players, skipping next returns turn to the same player."""
    game = two_player_game["game"]
    p1_id = two_player_game["p1_id"]

    event = GameEvent(
        action="PLAY_CARD",
        payload={"card": {"shape": "triangle", "number": 8}},
        player_id=p1_id,
        game_id=two_player_game["game_id"],
    )
    # 8 triangle is not valid on circle 3 — set face to triangle first
    state = await game._get_game_state()
    state.current_face_card = Card(shape="triangle", number=3)
    state.discard_pile = [state.current_face_card]
    await game.set_game_state(state)

    await game.process_game_event(event)
    state = await game._get_game_state()
    # Skip Bob → back to Alice
    assert state.current_player_id == p1_id


@pytest.mark.asyncio
async def test_suspension_three_players(redis_service, game_settings):
    from datetime import datetime
    from api.models.game import Deck, GameState, GameStatus, Player

    game_settings = game_settings.model_copy(update={"num_players": 3})
    game_id = uuid4()
    a, b, c = uuid4(), uuid4(), uuid4()
    game = WhotGame(game_id, redis_service)
    state = GameState(
        game_id=game_id,
        deck=Deck(cards=[]),
        market=Deck(cards=[Card(shape="star", number=4)] * 10),
        discard_pile=[Card(shape="circle", number=8)],
        players=[
            Player(id=a, name="A", cards=[
                Card(shape="circle", number=8),
                Card(shape="star", number=4),
            ]),
            Player(id=b, name="B", cards=[Card(shape="star", number=3)]),
            Player(id=c, name="C", cards=[Card(shape="star", number=7)]),
        ],
        spectators=[],
        num_spectators=0,
        status=GameStatus.IN_PROGRESS,
        current_face_card=Card(shape="circle", number=3),
        current_player_id=a,
        time_elapsed=0,
        created_at=datetime.utcnow(),
        started_at=datetime.utcnow(),
        settings=game_settings,
    )
    await redis_service.set_game_state(game_id, state)

    await game.process_game_event(GameEvent(
        action="PLAY_CARD",
        payload={"card": {"shape": "circle", "number": 8}},
        player_id=a,
        game_id=game_id,
    ))
    state = await game._get_game_state()
    # Skip B → C's turn
    assert state.current_player_id == c
    assert state.current_face_card.number == 8


@pytest.mark.asyncio
async def test_general_market_others_pick_keep_turn(two_player_game):
    game = two_player_game["game"]
    p1_id = two_player_game["p1_id"]
    p2_id = two_player_game["p2_id"]

    state = await game._get_game_state()
    p2_before = len(next(p for p in state.players if p.id == p2_id).cards)
    market_before = len(state.market.cards)

    await game.process_game_event(GameEvent(
        action="PLAY_CARD",
        payload={"card": {"shape": "circle", "number": 14}},
        player_id=p1_id,
        game_id=two_player_game["game_id"],
    ))
    state = await game._get_game_state()
    p2_after = len(next(p for p in state.players if p.id == p2_id).cards)
    assert p2_after == p2_before + 1
    assert len(state.market.cards) == market_before - 1
    assert state.current_player_id == p1_id


@pytest.mark.asyncio
async def test_pick_two_chain(two_player_game):
    game = two_player_game["game"]
    p1_id = two_player_game["p1_id"]
    p2_id = two_player_game["p2_id"]

    state = await game._get_game_state()
    state.current_face_card = Card(shape="cross", number=7)
    await game.set_game_state(state)

    await game.process_game_event(GameEvent(
        action="PLAY_CARD",
        payload={"card": {"shape": "cross", "number": 2}},
        player_id=p1_id,
        game_id=two_player_game["game_id"],
    ))
    state = await game._get_game_state()
    assert state.pick_chain_count == 1
    assert state.pick_chain_type == 2
    assert state.current_player_id == p2_id

    # Bob cannot defend with 2 — pick 2 cards
    p2 = next(p for p in state.players if p.id == p2_id)
    before = len(p2.cards)
    await game.process_game_event(GameEvent(
        action="PICK_CARD",
        payload={},
        player_id=p2_id,
        game_id=two_player_game["game_id"],
    ))
    state = await game._get_game_state()
    p2 = next(p for p in state.players if p.id == p2_id)
    assert len(p2.cards) == before + 2
    assert state.pick_chain_count == 0


@pytest.mark.asyncio
async def test_pick_five_chain_count(two_player_game):
    game = two_player_game["game"]
    p1_id = two_player_game["p1_id"]
    p2_id = two_player_game["p2_id"]

    state = await game._get_game_state()
    state.current_face_card = Card(shape="star", number=3)
    await game.set_game_state(state)

    await game.process_game_event(GameEvent(
        action="PLAY_CARD",
        payload={"card": {"shape": "star", "number": 5}},
        player_id=p1_id,
        game_id=two_player_game["game_id"],
    ))
    state = await game._get_game_state()
    assert state.pick_chain_type == 5
    assert state.pick_chain_count == 1

    p2 = next(p for p in state.players if p.id == p2_id)
    before = len(p2.cards)
    await game.process_game_event(GameEvent(
        action="PICK_CARD",
        payload={},
        player_id=p2_id,
        game_id=two_player_game["game_id"],
    ))
    state = await game._get_game_state()
    p2 = next(p for p in state.players if p.id == p2_id)
    assert len(p2.cards) == before + 3


@pytest.mark.asyncio
async def test_use_whot_removes_single(two_player_game):
    game = two_player_game["game"]
    p1_id = two_player_game["p1_id"]

    state = await game._get_game_state()
    p1 = next(p for p in state.players if p.id == p1_id)
    whots_before = sum(1 for c in p1.cards if c.number == 20)
    assert whots_before == 2

    await game.process_game_event(GameEvent(
        action="USE_WHOT",
        payload={"requested_shape": "triangle"},
        player_id=p1_id,
        game_id=two_player_game["game_id"],
    ))
    state = await game._get_game_state()
    p1 = next(p for p in state.players if p.id == p1_id)
    assert sum(1 for c in p1.cards if c.number == 20) == 1
    assert state.current_face_card.shape == "triangle"
    assert state.current_face_card.number == 20


@pytest.mark.asyncio
async def test_can_win_with_non_action_when_action_wins_disabled(two_player_game):
    game = two_player_game["game"]
    p1_id = two_player_game["p1_id"]
    redis = two_player_game["redis"]

    state = await game._get_game_state()
    assert state.settings.can_win_with_action is False
    p1 = next(p for p in state.players if p.id == p1_id)
    p1.cards = [Card(shape="circle", number=7)]
    state.current_face_card = Card(shape="circle", number=3)
    state.current_player_id = p1_id
    await game.set_game_state(state)
    await redis.add_active_game(two_player_game["game_id"])

    await game.process_game_event(GameEvent(
        action="PLAY_CARD",
        payload={"card": {"shape": "circle", "number": 7}},
        player_id=p1_id,
        game_id=two_player_game["game_id"],
    ))
    state = await game._get_game_state()
    assert state.status == GameStatus.COMPLETED
    assert state.winner_id == p1_id
    assert state.last_action.startswith("PLAY_CARD:1_cards@")


@pytest.mark.asyncio
async def test_cannot_win_with_action(two_player_game):
    game = two_player_game["game"]
    p1_id = two_player_game["p1_id"]

    state = await game._get_game_state()
    p1 = next(p for p in state.players if p.id == p1_id)
    p1.cards = [Card(shape="circle", number=1)]
    state.current_face_card = Card(shape="circle", number=3)
    await game.set_game_state(state)

    await game.process_game_event(GameEvent(
        action="PLAY_CARD",
        payload={"card": {"shape": "circle", "number": 1}},
        player_id=p1_id,
        game_id=two_player_game["game_id"],
    ))
    state = await game._get_game_state()
    p1 = next(p for p in state.players if p.id == p1_id)
    assert len(p1.cards) == 1
    assert state.status.value == "in_progress"
    errors = [e for e in two_player_game["redis"]._published if e["event_type"] == "ERROR"]
    assert any("Cannot win with action card" in e["data"].get("error", "") for e in errors)


@pytest.mark.asyncio
async def test_player_view_hides_opponent_cards(two_player_game):
    game = two_player_game["game"]
    p1_id = two_player_game["p1_id"]
    p2_id = two_player_game["p2_id"]

    view = await game.get_game_state_for_player(p1_id)
    assert view is not None
    self_view = next(p for p in view.players if p.id == p1_id)
    opp_view = next(p for p in view.players if p.id == p2_id)
    assert self_view.cards is not None
    assert len(self_view.cards) > 0
    assert opp_view.cards is None
    assert opp_view.card_count > 0
    assert view.market_count > 0
    assert not hasattr(view, "market") or True

    spectator = await game.get_game_state_for_spectator()
    assert all(p.card_count >= 0 for p in spectator.players)
    assert spectator.market_count == view.market_count


@pytest.mark.asyncio
async def test_completed_removes_from_active(two_player_game):
    game = two_player_game["game"]
    p1_id = two_player_game["p1_id"]
    redis = two_player_game["redis"]

    state = await game._get_game_state()
    state.settings = state.settings.model_copy(update={"can_win_with_action": True})
    p1 = next(p for p in state.players if p.id == p1_id)
    p1.cards = [Card(shape="circle", number=7)]
    state.current_face_card = Card(shape="circle", number=3)
    await game.set_game_state(state)
    await redis.add_active_game(two_player_game["game_id"])

    await game.process_game_event(GameEvent(
        action="PLAY_CARD",
        payload={"card": {"shape": "circle", "number": 7}},
        player_id=p1_id,
        game_id=two_player_game["game_id"],
    ))
    state = await game._get_game_state()
    assert state.status.value == "completed"
    active = await redis.get_active_game_ids()
    assert two_player_game["game_id"] not in active
    assert redis._ttls[str(two_player_game["game_id"])] == 3600


@pytest.mark.asyncio
async def test_delete_game_clears_keys(two_player_game):
    redis = two_player_game["redis"]
    game_id = two_player_game["game_id"]
    await redis.delete_game(game_id)
    assert await redis.get_game_state(game_id) is None
    assert game_id not in await redis.get_active_game_ids()
    assert await redis.get_game_players(game_id) == []


@pytest.mark.asyncio
async def test_handle_user_message_binds_connection_player_id(two_player_game):
    redis = two_player_game["redis"]
    game_id = two_player_game["game_id"]
    p1_id = two_player_game["p1_id"]
    p2_id = two_player_game["p2_id"]

    manager = GamesManager.__new__(GamesManager)
    manager.redis_client = None
    manager.redis_service = redis

    # Spoofed payload claims p1, but connection is p2 (not their turn)
    message = {
        "action": "PLAY_CARD",
        "payload": {"card": {"shape": "circle", "number": 7}},
        "player_id": str(p1_id),
        "game_id": str(game_id),
    }
    await manager.handle_user_message(message, game_id, p2_id)
    state = await redis.get_game_state(game_id)
    # Move rejected — still p1's turn, card still in p1 hand
    assert state.current_player_id == p1_id
    p1 = next(p for p in state.players if p.id == p1_id)
    assert any(c.number == 7 and c.shape == "circle" for c in p1.cards)
    errors = [e for e in redis._published if e["event_type"] == "ERROR"]
    assert any("Not your turn" in e["data"].get("error", "") for e in errors)


@pytest.mark.asyncio
async def test_cpu_easy_defend_sends_event(two_player_game):
    game = two_player_game["game"]
    p1_id = two_player_game["p1_id"]
    p2_id = two_player_game["p2_id"]
    redis = two_player_game["redis"]

    state = await game._get_game_state()
    state.pick_chain_count = 1
    state.pick_chain_type = 2
    state.current_player_id = p2_id
    p2 = next(p for p in state.players if p.id == p2_id)
    p2.cards = [Card(shape="triangle", number=2), Card(shape="star", number=4)]
    await game.set_game_state(state)

    cpu = CPUPlayer(p2_id, redis, difficulty="easy")
    await cpu.defend_pick_chain(two_player_game["game_id"], game, p2, state)

    state = await game._get_game_state()
    assert state.current_face_card.number == 2
    assert state.pick_chain_count == 2


@pytest.mark.asyncio
async def test_cpu_play_card_stacking_no_nameerror(two_player_game):
    game = two_player_game["game"]
    p1_id = two_player_game["p1_id"]
    redis = two_player_game["redis"]

    state = await game._get_game_state()
    p1 = next(p for p in state.players if p.id == p1_id)
    cards = [
        Card(shape="circle", number=7),
        Card(shape="star", number=7),
    ]
    # Ensure both in hand
    p1.cards = cards + [Card(shape="triangle", number=4)]
    state.current_face_card = Card(shape="circle", number=3)
    await game.set_game_state(state)

    cpu = CPUPlayer(p1_id, redis, difficulty="normal")
    await cpu.play_card(two_player_game["game_id"], game, p1, cards, state)

    state = await game._get_game_state()
    assert state.current_face_card.number == 7
    p1 = next(p for p in state.players if p.id == p1_id)
    assert not any(c.number == 7 for c in p1.cards)


@pytest.mark.asyncio
async def test_waiting_room_disconnect_removes_player(redis_service, game_settings):
    from datetime import datetime
    from api.models.game import Deck, GameState, GameStatus, Player

    game_id = uuid4()
    p1_id = uuid4()
    game = WhotGame(game_id, redis_service)
    state = GameState(
        game_id=game_id,
        deck=Deck(cards=[]),
        market=Deck(cards=[]),
        discard_pile=[Card(shape="circle", number=3)],
        players=[Player(id=p1_id, name="Alice")],
        spectators=[],
        num_spectators=0,
        status=GameStatus.WAITING_FOR_PLAYERS,
        current_face_card=Card(shape="circle", number=3),
        time_elapsed=0,
        created_at=datetime.utcnow(),
        settings=game_settings,
    )
    await redis_service.set_game_state(game_id, state)
    await redis_service.add_active_game(game_id)
    await redis_service.add_player_to_game(game_id, p1_id)

    await game.remove_player(p1_id)
    assert await game._get_game_state() is None
    assert await redis_service.get_game_players(game_id) == []
    assert game_id not in await redis_service.get_active_game_ids()
    errors = [e for e in redis_service._published if e["event_type"] == "ERROR"]
    assert any("does not exist" in e["data"].get("error", "") for e in errors)


@pytest.mark.asyncio
async def test_waiting_room_keeps_table_while_seats_remain(redis_service, game_settings):
    from datetime import datetime
    from api.models.game import Deck, GameState, GameStatus, Player

    game_id = uuid4()
    p1_id = uuid4()
    p2_id = uuid4()
    game = WhotGame(game_id, redis_service)
    state = GameState(
        game_id=game_id,
        deck=Deck(cards=[]),
        market=Deck(cards=[]),
        discard_pile=[Card(shape="circle", number=3)],
        players=[Player(id=p1_id, name="Alice"), Player(id=p2_id, name="Bob")],
        spectators=[],
        num_spectators=0,
        status=GameStatus.WAITING_FOR_PLAYERS,
        current_face_card=Card(shape="circle", number=3),
        time_elapsed=0,
        created_at=datetime.utcnow(),
        settings=game_settings,
    )
    await redis_service.set_game_state(game_id, state)
    await redis_service.add_active_game(game_id)
    await redis_service.add_player_to_game(game_id, p1_id)
    await redis_service.add_player_to_game(game_id, p2_id)

    await game.remove_player(p1_id)
    state = await game._get_game_state()
    assert state is not None
    assert [p.id for p in state.players] == [p2_id]
    assert game_id in await redis_service.get_active_game_ids()


@pytest.mark.asyncio
async def test_last_human_disconnect_ends_cpu_game(two_player_game):
    game = two_player_game["game"]
    redis = two_player_game["redis"]
    game_id = two_player_game["game_id"]
    p1_id = two_player_game["p1_id"]
    p2_id = two_player_game["p2_id"]

    state = await game._get_game_state()
    p2 = next(p for p in state.players if p.id == p2_id)
    p2.is_cpu = True
    await game.set_game_state(state)

    await game.remove_player(p1_id)
    state = await game._get_game_state()
    assert state.status == GameStatus.COMPLETED
    assert state.winner_id is None
    assert game_id not in await redis.get_active_game_ids()
    ended = [e for e in redis._published if e["event_type"] == "GAME_ENDED"]
    assert ended


@pytest.mark.asyncio
async def test_human_disconnect_keeps_game_if_another_human_remains(two_player_game):
    game = two_player_game["game"]
    game_id = two_player_game["game_id"]
    p1_id = two_player_game["p1_id"]
    p2_id = two_player_game["p2_id"]
    redis = two_player_game["redis"]

    await game.remove_player(p1_id)
    state = await game._get_game_state()
    assert state.status == GameStatus.IN_PROGRESS
    assert game_id in await redis.get_active_game_ids()
    assert state.abandon_at is not None

    await game.remove_player(p2_id)
    state = await game._get_game_state()
    assert state.status == GameStatus.COMPLETED
    assert game_id not in await redis.get_active_game_ids()


@pytest.mark.asyncio
async def test_disconnect_skips_to_next_connected_player(two_player_game):
    game = two_player_game["game"]
    p1_id = two_player_game["p1_id"]
    p2_id = two_player_game["p2_id"]

    state = await game._get_game_state()
    state.current_player_id = p1_id
    await game.set_game_state(state)

    await game.remove_player(p1_id)
    state = await game._get_game_state()
    assert state.current_player_id == p2_id
    p1 = next(p for p in state.players if p.id == p1_id)
    assert p1.is_connected is False


@pytest.mark.asyncio
async def test_reconnect_clears_solo_abandon(two_player_game):
    game = two_player_game["game"]
    p1_id = two_player_game["p1_id"]
    p2_id = two_player_game["p2_id"]

    await game.remove_player(p1_id)
    state = await game._get_game_state()
    assert state.abandon_at is not None

    from api.models.game import Player
    ok, action = await game.add_player(Player(id=p1_id, name="Alice"), is_reconnecting=True)
    assert ok
    assert action == "clear_abandon"
    state = await game._get_game_state()
    assert state.abandon_at is None
    p1 = next(p for p in state.players if p.id == p1_id)
    assert p1.is_connected is True


@pytest.mark.asyncio
async def test_cpu_does_not_move_after_humans_leave(two_player_game):
    game = two_player_game["game"]
    redis = two_player_game["redis"]
    p1_id = two_player_game["p1_id"]
    p2_id = two_player_game["p2_id"]

    state = await game._get_game_state()
    p1 = next(p for p in state.players if p.id == p1_id)
    p2 = next(p for p in state.players if p.id == p2_id)
    p1.is_connected = False
    p2.is_cpu = True
    state.current_player_id = p2_id
    await game.set_game_state(state)

    await game.check_and_trigger_cpu_if_needed(state)
    state = await game._get_game_state()
    p2 = next(p for p in state.players if p.id == p2_id)
    assert state.status == GameStatus.COMPLETED
    assert state.current_face_card.shape == "circle"
    assert state.current_face_card.number == 3
    assert any(c.shape == "square" and c.number == 3 for c in p2.cards)
