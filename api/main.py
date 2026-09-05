import json
import asyncio
import time
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import ConnectionPool

from api.core.config import settings
from api.core.constants import WS_PING_INTERVAL_SECONDS
from api.middleware.dependencies import get_games_manager, get_games_manager_ws, get_player_id
from api.models.game import GameCreate, GameResponse, GameStatus, GameListItem, EventType
from api.services.session import GamesManager
from api.services.game import WhotGame
from api.services.redis_service import RedisService


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.redis_pool = ConnectionPool.from_url(settings.redis_url)
    yield
    await app.redis_pool.aclose()


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/games", response_model=GameResponse)
async def create_game(
    game_settings: GameCreate,
    game_manager: GamesManager = Depends(get_games_manager)
):
    if game_settings.num_players < 2:
        raise HTTPException(status_code=400, detail="Game must have at least 2 players")
    return await game_manager.create_game(game_settings)


@app.get("/games/{game_id}", response_model=GameListItem)
async def get_game(
    game_id: UUID,
    game_manager: GamesManager = Depends(get_games_manager),
):
    game = await game_manager.get_game_list_item(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    return game


async def _send_player_game_state(websocket: WebSocket, game: WhotGame, game_id: UUID, player_id: UUID) -> None:
    player_state = await game.get_game_state_for_player(player_id)
    if player_state:
        await websocket.send_json({
            "event_type": EventType.GAME_STATE.value,
            "data": {"game_state": player_state.model_dump(mode='json')},
            "game_id": str(game_id),
            "player_id": str(player_id),
        })


async def _maybe_send_ping(websocket: WebSocket, last_ping_at: float) -> float:
    now = time.monotonic()
    if now - last_ping_at < WS_PING_INTERVAL_SECONDS:
        return last_ping_at
    await websocket.send_json({"event_type": EventType.PING.value})
    return now


@app.websocket("/join/{game_id}")
async def join_game(
    websocket: WebSocket,
    game_id: UUID,
    player_id: UUID = Depends(get_player_id),
    game_manager: GamesManager = Depends(get_games_manager_ws)
):
    await websocket.accept()
    display_name = websocket.query_params.get("display_name", "Player")

    game_status = await game_manager.get_game_status(game_id)
    if game_status is None:
        await websocket.send_json({
            "event_type": EventType.ERROR.value,
            "data": {"error": "This table does not exist"},
            "game_id": str(game_id)
        })
        await websocket.close(reason="Game not found")
        return

    if game_status == GameStatus.COMPLETED:
        await websocket.send_json({
            "event_type": EventType.ERROR.value,
            "data": {"error": "This table has finished"},
            "game_id": str(game_id)
        })
        await websocket.close(reason="Game has ended")
        return

    is_reconnecting = await game_manager.is_player_reconnecting(game_id, player_id)
    if game_status == GameStatus.IN_PROGRESS and not is_reconnecting:
        await websocket.send_json({
            "event_type": EventType.ERROR.value,
            "data": {"error": "Game has already started"},
            "game_id": str(game_id)
        })
        await websocket.close(reason="Game has started")
        return

    added = await game_manager.add_to_game(game_id, player_id, display_name, is_reconnecting)
    if not added:
        await websocket.send_json({
            "event_type": EventType.ERROR.value,
            "data": {"error": "Could not join game"},
            "game_id": str(game_id)
        })
        await websocket.close(reason="Could not join game")
        return

    redis_service = RedisService(game_manager.redis_client)
    pubsub = game_manager.redis_client.pubsub()
    channel_name = f"game:{game_id}"
    await pubsub.subscribe(channel_name)

    game = WhotGame(game_id, redis_service)
    await _send_player_game_state(websocket, game, game_id, player_id)

    intentional_leave = False
    last_ping_at = time.monotonic()

    try:
        while True:
            try:
                user_message = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
                still_connected = await game_manager.handle_user_message(
                    user_message, game_id, player_id
                )
                if not still_connected:
                    intentional_leave = True
                    break
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break

            game_message = await pubsub.get_message(ignore_subscribe_messages=True)
            if game_message:
                data = json.loads(game_message['data'].decode('utf-8'))
                if data.get('event_type') == EventType.GAME_STATE.value:
                    await _send_player_game_state(websocket, game, game_id, player_id)
                else:
                    await websocket.send_json(data)
                if data.get('event_type') == EventType.GAME_ENDED.value:
                    break

            last_ping_at = await _maybe_send_ping(websocket, last_ping_at)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.aclose()

    if intentional_leave:
        # LEAVE already called remove_from_game
        return

    await game_manager.handle_socket_disconnect(game_id, player_id)


@app.websocket("/spectate/{game_id}")
async def spectate_game(
    websocket: WebSocket,
    game_id: UUID,
    spectator_id: UUID = Depends(get_player_id),
    game_manager: GamesManager = Depends(get_games_manager_ws)
):
    await websocket.accept()

    game_status = await game_manager.get_game_status(game_id)
    if game_status is None:
        await websocket.send_json({
            "event_type": EventType.ERROR.value,
            "data": {"error": "This table does not exist"},
            "game_id": str(game_id)
        })
        await websocket.close(reason="Game not found")
        return

    if game_status == GameStatus.COMPLETED:
        await websocket.send_json({
            "event_type": EventType.ERROR.value,
            "data": {"error": "This table has finished"},
            "game_id": str(game_id)
        })
        await websocket.close(reason="Game has ended")
        return

    added = await game_manager.add_spectator(game_id, spectator_id)
    if not added:
        await websocket.send_json({
            "event_type": EventType.ERROR.value,
            "data": {"error": "Could not spectate game"},
            "game_id": str(game_id)
        })
        await websocket.close(reason="Could not spectate game")
        return

    redis_service = RedisService(game_manager.redis_client)
    pubsub = game_manager.redis_client.pubsub()
    channel_name = f"game:{game_id}"
    await pubsub.subscribe(channel_name)

    game = WhotGame(game_id, redis_service)
    spectator_state = await game.get_game_state_for_spectator()
    if spectator_state:
        await websocket.send_json({
            "event_type": EventType.GAME_STATE.value,
            "data": {"game_state": spectator_state.model_dump(mode='json')},
            "game_id": str(game_id)
        })

    last_ping_at = time.monotonic()

    try:
        while True:
            game_message = await pubsub.get_message(ignore_subscribe_messages=True)
            if game_message:
                data = json.loads(game_message['data'].decode('utf-8'))
                if data.get('event_type') == EventType.GAME_STATE.value:
                    spectator_state = await game.get_game_state_for_spectator()
                    if spectator_state:
                        await websocket.send_json({
                            "event_type": EventType.GAME_STATE.value,
                            "data": {"game_state": spectator_state.model_dump(mode='json')},
                            "game_id": str(game_id)
                        })
                else:
                    await websocket.send_json(data)
                if data.get('event_type') == EventType.GAME_ENDED.value:
                    break
            last_ping_at = await _maybe_send_ping(websocket, last_ping_at)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.aclose()
        await redis_service.remove_spectator_from_game(game_id, spectator_id)


@app.websocket("/games/list")
async def list_games(
    websocket: WebSocket,
    game_manager: GamesManager = Depends(get_games_manager_ws)
):
    await websocket.accept()

    try:
        while True:
            games = await game_manager.get_all_games()
            await websocket.send_json({
                "games": [game.model_dump(mode='json') for game in games]
            })
            await asyncio.sleep(3)
    except WebSocketDisconnect:
        pass
