from fastapi import Request, WebSocket
from uuid import UUID, uuid4
from redis.asyncio import Redis

from api.core.config import settings
from api.services.session import GamesManager


async def get_player_id(websocket: WebSocket) -> UUID:
    cookie_name = settings.guest_session_cookie_name
    existing_session = websocket.cookies.get(cookie_name)
    if existing_session:
        try:
            return UUID(existing_session)
        except ValueError:
            pass
    
    new_player_id = uuid4()
    return new_player_id


async def get_redis_client(websocket: WebSocket | Request) -> Redis:
    if isinstance(websocket, WebSocket):
        return Redis(connection_pool=websocket.app.redis_pool)
    return Redis(connection_pool=websocket.app.redis_pool)


def get_games_manager(request: Request) -> GamesManager:
    redis_client = Redis(connection_pool=request.app.redis_pool)
    return GamesManager(redis_client)


def get_games_manager_ws(websocket: WebSocket) -> GamesManager:
    redis_client = Redis(connection_pool=websocket.app.redis_pool)
    return GamesManager(redis_client)