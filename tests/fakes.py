from contextlib import asynccontextmanager
from datetime import datetime
import json
from typing import AsyncIterator, Optional
from uuid import UUID
import asyncio

from api.models.game import GameState, GameStatus


class FakeRedisService:
    """In-memory Redis stand-in for unit tests (no live Redis required)."""

    def __init__(self):
        self._games: dict[str, str] = {}
        self._ttls: dict[str, int] = {}
        self._sets: dict[str, set[str]] = {}
        self._strings: dict[str, str] = {}
        self._string_ttls: dict[str, int] = {}
        self._published: list[dict] = []
        self._locks: dict[str, asyncio.Lock] = {}

    def _ttl_for_state(self, state: GameState) -> int:
        if state.status == GameStatus.COMPLETED:
            return 3600
        return 86400

    async def get_game_state(self, game_id: UUID) -> Optional[GameState]:
        data = self._games.get(str(game_id))
        if not data:
            return None
        return GameState.model_validate_json(data)

    async def set_game_state(self, game_id: UUID, state: GameState) -> None:
        ttl = self._ttl_for_state(state)
        gid = str(game_id)
        self._games[gid] = state.model_dump_json()
        self._ttls[gid] = ttl
        self._ttls[f"{gid}:players"] = ttl
        self._ttls[f"{gid}:spectators"] = ttl
        if state.status == GameStatus.COMPLETED:
            await self.remove_active_game(game_id)

    @asynccontextmanager
    async def game_lock(self, game_id: UUID) -> AsyncIterator[None]:
        key = str(game_id)
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        async with self._locks[key]:
            yield

    async def add_player_to_game(self, game_id: UUID, player_id: UUID) -> None:
        key = f"game:{game_id}:players"
        self._sets.setdefault(key, set()).add(str(player_id))
        self._ttls[f"{game_id}:players"] = 86400
        self._strings[f"player:{player_id}:game"] = str(game_id)
        self._string_ttls[f"player:{player_id}:game"] = 86400

    async def remove_player_from_game(self, game_id: UUID, player_id: UUID) -> None:
        key = f"game:{game_id}:players"
        if key in self._sets:
            self._sets[key].discard(str(player_id))
        self._strings.pop(f"player:{player_id}:game", None)

    async def get_game_players(self, game_id: UUID) -> list[UUID]:
        members = self._sets.get(f"game:{game_id}:players", set())
        return [UUID(m) for m in members]

    async def add_spectator_to_game(self, game_id: UUID, spectator_id: UUID) -> None:
        key = f"game:{game_id}:spectators"
        self._sets.setdefault(key, set()).add(str(spectator_id))
        self._ttls[f"{game_id}:spectators"] = 86400

    async def remove_spectator_from_game(self, game_id: UUID, spectator_id: UUID) -> None:
        key = f"game:{game_id}:spectators"
        if key in self._sets:
            self._sets[key].discard(str(spectator_id))

    async def get_game_spectators(self, game_id: UUID) -> list[UUID]:
        members = self._sets.get(f"game:{game_id}:spectators", set())
        return [UUID(m) for m in members]

    async def add_active_game(self, game_id: UUID) -> None:
        self._sets.setdefault("games:active", set()).add(str(game_id))

    async def remove_active_game(self, game_id: UUID) -> None:
        if "games:active" in self._sets:
            self._sets["games:active"].discard(str(game_id))

    async def get_active_game_ids(self) -> list[UUID]:
        return [UUID(m) for m in self._sets.get("games:active", set())]

    async def publish_game_event(self, game_id: UUID, event_type: str, data: dict, player_id: Optional[UUID] = None) -> None:
        self._published.append({
            "event_type": event_type,
            "data": data,
            "game_id": str(game_id),
            "player_id": str(player_id) if player_id else None,
            "timestamp": datetime.utcnow().isoformat(),
        })

    async def get_player_game(self, player_id: UUID) -> Optional[UUID]:
        val = self._strings.get(f"player:{player_id}:game")
        return UUID(val) if val else None

    async def set_player_session(self, player_id: UUID, game_id: UUID) -> None:
        self._strings[f"player:{player_id}:session"] = str(game_id)
        self._string_ttls[f"player:{player_id}:session"] = 86400

    async def get_player_session(self, player_id: UUID) -> Optional[UUID]:
        val = self._strings.get(f"player:{player_id}:session")
        return UUID(val) if val else None

    async def delete_game(self, game_id: UUID) -> None:
        player_ids = await self.get_game_players(game_id)
        self._games.pop(str(game_id), None)
        self._ttls.pop(str(game_id), None)
        self._sets.pop(f"game:{game_id}:players", None)
        self._sets.pop(f"game:{game_id}:spectators", None)
        await self.remove_active_game(game_id)
        for player_id in player_ids:
            self._strings.pop(f"player:{player_id}:game", None)
            self._strings.pop(f"player:{player_id}:session", None)
