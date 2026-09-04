import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator, Optional
from uuid import UUID

from redis.asyncio import Redis
from redis.asyncio.lock import Lock

from api.core.config import settings
from api.models.game import GameState, GameStatus


class RedisService:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    def _ttl_for_state(self, state: GameState) -> int:
        if state.status == GameStatus.COMPLETED:
            return settings.completed_game_ttl_seconds
        return settings.game_ttl_seconds

    async def _expire_game_keys(self, game_id: UUID, ttl: int) -> None:
        pipe = self.redis.pipeline()
        pipe.expire(f"game:{game_id}", ttl)
        pipe.expire(f"game:{game_id}:players", ttl)
        pipe.expire(f"game:{game_id}:spectators", ttl)
        await pipe.execute()

    async def get_game_state(self, game_id: UUID) -> Optional[GameState]:
        key = f"game:{game_id}"
        data = await self.redis.get(key)
        if not data:
            return None
        return GameState.model_validate_json(data)

    async def set_game_state(self, game_id: UUID, state: GameState) -> None:
        key = f"game:{game_id}"
        ttl = self._ttl_for_state(state)
        pipe = self.redis.pipeline()
        pipe.set(key, state.model_dump_json(), ex=ttl)
        pipe.expire(f"game:{game_id}:players", ttl)
        pipe.expire(f"game:{game_id}:spectators", ttl)
        await pipe.execute()

        if state.status == GameStatus.COMPLETED:
            await self.remove_active_game(game_id)

    @asynccontextmanager
    async def game_lock(self, game_id: UUID) -> AsyncIterator[Lock]:
        lock = self.redis.lock(
            f"game:{game_id}:lock",
            timeout=settings.game_lock_timeout_seconds,
            blocking_timeout=settings.game_lock_timeout_seconds,
        )
        await lock.acquire()
        try:
            yield lock
        finally:
            try:
                await lock.release()
            except Exception:
                pass

    async def add_player_to_game(self, game_id: UUID, player_id: UUID) -> None:
        ttl = settings.game_ttl_seconds
        pipe = self.redis.pipeline()
        pipe.sadd(f"game:{game_id}:players", str(player_id))
        pipe.expire(f"game:{game_id}:players", ttl)
        pipe.set(f"player:{player_id}:game", str(game_id), ex=ttl)
        await pipe.execute()

    async def remove_player_from_game(self, game_id: UUID, player_id: UUID) -> None:
        await self.redis.srem(f"game:{game_id}:players", str(player_id))
        await self.redis.delete(f"player:{player_id}:game")

    async def get_game_players(self, game_id: UUID) -> list[UUID]:
        members = await self.redis.smembers(f"game:{game_id}:players")
        return [UUID(member.decode() if isinstance(member, bytes) else member) for member in members]

    async def add_spectator_to_game(self, game_id: UUID, spectator_id: UUID) -> None:
        ttl = settings.game_ttl_seconds
        pipe = self.redis.pipeline()
        pipe.sadd(f"game:{game_id}:spectators", str(spectator_id))
        pipe.expire(f"game:{game_id}:spectators", ttl)
        await pipe.execute()

    async def remove_spectator_from_game(self, game_id: UUID, spectator_id: UUID) -> None:
        await self.redis.srem(f"game:{game_id}:spectators", str(spectator_id))

    async def get_game_spectators(self, game_id: UUID) -> list[UUID]:
        members = await self.redis.smembers(f"game:{game_id}:spectators")
        return [UUID(member.decode() if isinstance(member, bytes) else member) for member in members]

    async def add_active_game(self, game_id: UUID) -> None:
        await self.redis.sadd("games:active", str(game_id))

    async def remove_active_game(self, game_id: UUID) -> None:
        await self.redis.srem("games:active", str(game_id))

    async def get_active_game_ids(self) -> list[UUID]:
        members = await self.redis.smembers("games:active")
        return [UUID(member.decode() if isinstance(member, bytes) else member) for member in members]

    async def publish_game_event(self, game_id: UUID, event_type: str, data: dict, player_id: Optional[UUID] = None) -> None:
        channel = f"game:{game_id}"
        payload = {
            "event_type": event_type,
            "data": data,
            "game_id": str(game_id),
            "player_id": str(player_id) if player_id else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.redis.publish(channel, json.dumps(payload))

    async def get_player_game(self, player_id: UUID) -> Optional[UUID]:
        game_id_str = await self.redis.get(f"player:{player_id}:game")
        if not game_id_str:
            return None
        raw = game_id_str.decode() if isinstance(game_id_str, bytes) else game_id_str
        return UUID(raw)

    async def set_player_session(self, player_id: UUID, game_id: UUID) -> None:
        await self.redis.set(f"player:{player_id}:session", str(game_id), ex=86400)

    async def get_player_session(self, player_id: UUID) -> Optional[UUID]:
        game_id_str = await self.redis.get(f"player:{player_id}:session")
        if not game_id_str:
            return None
        raw = game_id_str.decode() if isinstance(game_id_str, bytes) else game_id_str
        return UUID(raw)

    async def delete_game(self, game_id: UUID) -> None:
        player_ids = await self.get_game_players(game_id)
        pipe = self.redis.pipeline()
        pipe.delete(f"game:{game_id}")
        pipe.delete(f"game:{game_id}:players")
        pipe.delete(f"game:{game_id}:spectators")
        pipe.srem("games:active", str(game_id))
        for player_id in player_ids:
            pipe.delete(f"player:{player_id}:game")
            pipe.delete(f"player:{player_id}:session")
        await pipe.execute()
