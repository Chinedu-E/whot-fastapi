import asyncio
from datetime import datetime
from uuid import UUID, uuid4

from redis.asyncio import Redis

from api.models.game import GameCreate, GameResponse, GameStatus, GameListItem, GameEvent, EventType
from api.services.game import DisconnectAction, WhotGame
from api.services.redis_service import RedisService

# Process-wide so timers survive per-request GamesManager instances.
_abandon_tasks: dict[UUID, asyncio.Task] = {}


class GamesManager:
    def __init__(self, redis_client: Redis):
        self.redis_client = redis_client
        self.redis_service = RedisService(redis_client)
    
    async def create_game(self, game_settings: GameCreate) -> GameResponse:
        game_id = uuid4()
        game = WhotGame(game_id, self.redis_service)
        await game.initialize(game_settings)
        await self.redis_service.add_active_game(game_id)
        return GameResponse(game_id=game_id, created_at=datetime.utcnow())
        
    async def delete_game(self, game_id: UUID) -> None:
        self.cancel_abandon(game_id)
        await self.redis_service.delete_game(game_id)

    def cancel_abandon(self, game_id: UUID) -> None:
        task = _abandon_tasks.pop(game_id, None)
        if task and not task.done():
            task.cancel()

    def schedule_abandon(self, game_id: UUID, abandon_at: datetime) -> None:
        self.cancel_abandon(game_id)
        delay = max(0.0, (abandon_at - datetime.utcnow()).total_seconds())
        _abandon_tasks[game_id] = asyncio.create_task(
            self._run_abandon(game_id, delay),
            name=f"abandon-{game_id}",
        )

    async def _run_abandon(self, game_id: UUID, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            game = WhotGame(game_id, self.redis_service)
            await game.finish_if_still_solo()
        except asyncio.CancelledError:
            return
        finally:
            existing = _abandon_tasks.get(game_id)
            if existing is asyncio.current_task():
                _abandon_tasks.pop(game_id, None)

    async def _apply_abandon_action(
        self,
        game_id: UUID,
        action: DisconnectAction,
        abandon_at: datetime | None = None,
    ) -> None:
        if action == "abandon":
            if abandon_at is None:
                state = await self.redis_service.get_game_state(game_id)
                abandon_at = state.abandon_at if state else None
            if abandon_at:
                self.schedule_abandon(game_id, abandon_at)
        elif action in ("clear_abandon", "finished", "deleted"):
            self.cancel_abandon(game_id)
        
    async def is_player_reconnecting(self, game_id: UUID, player_id: UUID) -> bool:
        session_game_id = await self.redis_service.get_player_session(player_id)
        if session_game_id == game_id:
            player_game_id = await self.redis_service.get_player_game(player_id)
            return player_game_id == game_id
        return False
        
    async def add_to_game(self, game_id: UUID, player_id: UUID, player_name: str, is_reconnecting: bool) -> bool:
        game = WhotGame(game_id, self.redis_service)
        state = await game._get_game_state()
        if not state:
            return False
        
        if state.status == GameStatus.COMPLETED:
            return False
        
        if state.status == GameStatus.IN_PROGRESS and not is_reconnecting:
            return False
        
        from api.models.game import Player
        player = Player(id=player_id, name=player_name)
        success, action = await game.add_player(player, is_reconnecting)
        if success:
            await self.redis_service.set_player_session(player_id, game_id)
            latest = await self.redis_service.get_game_state(game_id)
            await self._apply_abandon_action(
                game_id,
                action,
                latest.abandon_at if latest else None,
            )
        return success
    
    async def remove_from_game(self, game_id: UUID, player_id: UUID) -> None:
        game = WhotGame(game_id, self.redis_service)
        action = await game.remove_player(player_id)
        latest = await self.redis_service.get_game_state(game_id)
        await self._apply_abandon_action(
            game_id,
            action,
            latest.abandon_at if latest else None,
        )
        
    async def handle_user_message(self, message: dict, game_id: UUID, player_id: UUID) -> None:
        try:
            event = GameEvent.model_validate(message)
            if event.game_id != game_id:
                return
            
            # Always bind identity to the WebSocket connection — ignore client-supplied player_id
            event.player_id = player_id
            
            game = WhotGame(game_id, self.redis_service)
            await game.process_game_event(event)
        except Exception as e:
            await self.redis_service.publish_game_event(
                game_id,
                EventType.ERROR.value,
                {"error": str(e), "message": "Invalid message format"},
                player_id
            )
        
    async def get_game_status(self, game_id: UUID) -> GameStatus | None:
        state = await self.redis_service.get_game_state(game_id)
        if not state:
            return None
        return state.status

    async def get_game_list_item(self, game_id: UUID) -> GameListItem | None:
        state = await self.redis_service.get_game_state(game_id)
        if not state:
            return None
        player_ids = await self.redis_service.get_game_players(game_id)
        return GameListItem(
            game_id=game_id,
            status=state.status,
            player_count=len(player_ids),
            max_players=state.settings.num_players,
            created_at=state.created_at,
            started_at=state.started_at,
        )
    
    async def get_all_games(self) -> list[GameListItem]:
        game_ids = await self.redis_service.get_active_game_ids()
        games = []
        for game_id in game_ids:
            state = await self.redis_service.get_game_state(game_id)
            if state and state.status != GameStatus.COMPLETED:
                player_ids = await self.redis_service.get_game_players(game_id)
                games.append(GameListItem(
                    game_id=game_id,
                    status=state.status,
                    player_count=len(player_ids),
                    max_players=state.settings.num_players,
                    created_at=state.created_at,
                    started_at=state.started_at
                ))
        return games
    
    async def add_spectator(self, game_id: UUID, spectator_id: UUID) -> bool:
        state = await self.redis_service.get_game_state(game_id)
        if not state:
            return False
        game = WhotGame(game_id, self.redis_service)
        from api.models.game import Spectator
        spectator = Spectator(id=spectator_id)
        return await game.add_spectator(spectator)
