import asyncio
import logging
from datetime import datetime, timedelta
from typing import Literal, Optional
from uuid import UUID, uuid4

from api.core.constants import CPU_MOVE_DELAY_SECONDS, SOLO_ABANDON_SECONDS
from api.models.game import (
    Card, Deck, GameCreate, GameEvent, GameState, GameStateForPlayer,
    GameStateForSpectator, GameStatus, Player, PlayerForPlayerView,
    PlayerForSpectator, Spectator, EventType
)
from api.services.redis_service import RedisService
from rl import engine as whot_engine

logger = logging.getLogger(__name__)

DisconnectAction = Literal["deleted", "finished", "abandon", "clear_abandon", "ok"]


class WhotGame:
    def __init__(self, game_id: UUID, redis_service: RedisService):
        self.game_id = game_id
        self.redis_service = redis_service
        
    async def initialize(self, settings: GameCreate) -> None:
        deck_cards, init_card = self.create_deck()
        state = GameState(
            game_id=self.game_id,
            deck=Deck(cards=[]),
            market=Deck(cards=deck_cards),
            discard_pile=[init_card],
            players=[],
            spectators=[],
            num_spectators=0,
            status=GameStatus.WAITING_FOR_PLAYERS,
            current_face_card=init_card,
            time_elapsed=0,
            created_at=datetime.utcnow(),
            settings=settings
        )
        await self.set_game_state(state)
        
    async def set_game_state(self, state: GameState) -> None:
        await self.redis_service.set_game_state(self.game_id, state)
        
    async def _get_game_state(self) -> GameState | None:
        return await self.redis_service.get_game_state(self.game_id)
        
    def create_deck(self) -> tuple[list[Card], Card]:
        return whot_engine.create_deck()
 
    async def add_player(self, player: Player, is_reconnecting: bool = False) -> tuple[bool, DisconnectAction]:
        abandon_action: DisconnectAction = "ok"
        async with self.redis_service.game_lock(self.game_id):
            state = await self._get_game_state()
            if not state:
                return False, "ok"
            
            if state.status == GameStatus.IN_PROGRESS and not is_reconnecting:
                return False, "ok"
            
            player_exists = any(p.id == player.id for p in state.players)
            if player_exists:
                for p in state.players:
                    if p.id == player.id:
                        p.is_connected = True
                        p.name = player.name
                        break
            else:
                if len(state.players) >= state.settings.num_players:
                    return False, "ok"
                state.players.append(player)
            
            await self.redis_service.add_player_to_game(self.game_id, player.id)

            if state.status == GameStatus.IN_PROGRESS and is_reconnecting:
                abandon_action = self._sync_abandon_window(state)
            
            await self.set_game_state(state)
            if state.status == GameStatus.IN_PROGRESS:
                await self.broadcast_game_state()
            
            if not is_reconnecting:
                await self.broadcast_message(EventType.PLAYER_JOINED, {
                    "player_id": str(player.id),
                    "player_name": player.name
                })
            
            should_start = False
            if state.settings.fill_with_computers:
                human_players = [p for p in state.players if not p.is_cpu]
                should_start = len(human_players) >= 1 and state.status == GameStatus.WAITING_FOR_PLAYERS
            else:
                should_start = (
                    len(state.players) == state.settings.num_players
                    and state.status == GameStatus.WAITING_FOR_PLAYERS
                )
        
        if should_start:
            await self.start_game()
        
        return True, abandon_action
        
    async def add_spectator(self, spectator: Spectator) -> bool:
        async with self.redis_service.game_lock(self.game_id):
            state = await self._get_game_state()
            if not state:
                return False
            
            spectator_exists = any(s.id == spectator.id for s in state.spectators)
            if not spectator_exists:
                state.spectators.append(spectator)
                state.num_spectators = len(state.spectators)
                await self.redis_service.add_spectator_to_game(self.game_id, spectator.id)
                await self.set_game_state(state)
            
            return True
        
    async def remove_player(self, player_id: UUID) -> DisconnectAction:
        trigger_cpu = False
        abandon_action: DisconnectAction = "ok"
        async with self.redis_service.game_lock(self.game_id):
            state = await self._get_game_state()
            if not state:
                return "ok"
            
            if state.status == GameStatus.WAITING_FOR_PLAYERS:
                state.players = [p for p in state.players if p.id != player_id]
                await self.redis_service.remove_player_from_game(self.game_id, player_id)
            else:
                for player in state.players:
                    if player.id == player_id:
                        player.is_connected = False
                        break
            
            await self.broadcast_message(EventType.PLAYER_LEFT, {
                "player_id": str(player_id)
            })

            if state.status == GameStatus.WAITING_FOR_PLAYERS and not state.players:
                await self.broadcast_message(EventType.ERROR, {
                    "error": "This table does not exist"
                })
                await self.redis_service.delete_game(self.game_id)
                return "deleted"

            if state.status == GameStatus.IN_PROGRESS:
                if not self._has_connected_human(state):
                    state.abandon_at = None
                    await self._finish_cpu_only_game(state)
                    return "finished"

                self._ensure_turn_eligible(state)
                abandon_action = self._sync_abandon_window(state)
                await self.set_game_state(state)
                await self.broadcast_game_state()
                trigger_cpu = True
            else:
                await self.set_game_state(state)

        if trigger_cpu:
            latest = await self._get_game_state()
            if latest:
                self.schedule_cpu_check(latest)

        return abandon_action

    @staticmethod
    def _has_connected_human(state: GameState) -> bool:
        return any(player.is_connected and not player.is_cpu for player in state.players)

    @staticmethod
    def _connected_human_count(state: GameState) -> int:
        return sum(1 for player in state.players if player.is_connected and not player.is_cpu)

    @staticmethod
    def _disconnected_human_count(state: GameState) -> int:
        return sum(1 for player in state.players if not player.is_connected and not player.is_cpu)

    @staticmethod
    def _is_eligible_turn(player: Player) -> bool:
        return whot_engine.is_eligible_turn(player)

    def _sync_abandon_window(self, state: GameState) -> DisconnectAction:
        """Update abandon_at for solo-human reconnect window. Mutates state."""
        if state.status != GameStatus.IN_PROGRESS:
            if state.abandon_at is not None:
                state.abandon_at = None
                return "clear_abandon"
            return "ok"

        connected = self._connected_human_count(state)
        disconnected = self._disconnected_human_count(state)

        if connected == 1 and disconnected >= 1:
            if state.abandon_at is None:
                state.abandon_at = datetime.utcnow() + timedelta(seconds=SOLO_ABANDON_SECONDS)
                return "abandon"
            return "ok"

        if state.abandon_at is not None:
            state.abandon_at = None
            return "clear_abandon"
        return "ok"

    def _ensure_turn_eligible(self, state: GameState) -> None:
        """If current seat is a disconnected human, advance to a connected human or CPU."""
        if not state.players or not state.current_player_id:
            return
        current = next((p for p in state.players if p.id == state.current_player_id), None)
        if current and self._is_eligible_turn(current):
            return
        self.next_turn(state)

    async def _finish_cpu_only_game(self, state: GameState) -> None:
        """Caller must hold the game lock."""
        state.abandon_at = None
        state.ended_at = datetime.utcnow()
        state.status = GameStatus.COMPLETED
        await self.set_game_state(state)
        await self.broadcast_game_state()
        await self.broadcast_message(EventType.GAME_ENDED, {
            "winner_id": None,
            "winner_name": None,
        })

    async def finish_if_still_solo(self) -> bool:
        """End the table if the solo-human abandon window has elapsed. Returns True if finished."""
        async with self.redis_service.game_lock(self.game_id):
            state = await self._get_game_state()
            if not state or state.status != GameStatus.IN_PROGRESS:
                return False
            if self._connected_human_count(state) != 1:
                return False
            if self._disconnected_human_count(state) < 1:
                return False
            if state.abandon_at and state.abandon_at > datetime.utcnow():
                return False
            await self._finish_cpu_only_game(state)
            return True
        
    async def broadcast_message(self, event_type: EventType, data: dict, player_id: UUID | None = None) -> None:
        await self.redis_service.publish_game_event(self.game_id, event_type.value, data, player_id)
        
    async def process_game_event(self, event: GameEvent) -> None:
        async with self.redis_service.game_lock(self.game_id):
            state = await self._get_game_state()
            if not state:
                return
            
            if state.status != GameStatus.IN_PROGRESS:
                await self.broadcast_message(EventType.ERROR, {
                    "error": "Game is not in progress"
                }, event.player_id)
                return
            
            if state.current_player_id != event.player_id:
                await self.broadcast_message(EventType.ERROR, {
                    "error": "Not your turn"
                }, event.player_id)
                return
            
            if event.action == "PLAY_CARD":
                await self.handle_play_card(event, state)
            elif event.action == "PICK_CARD":
                await self.handle_pick_card(event, state)
            elif event.action == "USE_WHOT":
                await self.handle_use_whot(event, state)
        
        state = await self._get_game_state()
        if state:
            self.schedule_cpu_check(state)
        
    async def handle_play_card(self, event: GameEvent, state: GameState) -> None:
        payload = event.payload
        cards_to_play: list[Card] = []

        if "cards" in payload and payload["cards"]:
            for card_data in payload["cards"]:
                cards_to_play.append(Card(shape=card_data["shape"], number=card_data["number"]))
        elif "card" in payload and payload["card"]:
            card_data = payload["card"]
            cards_to_play.append(Card(shape=card_data["shape"], number=card_data["number"]))
        else:
            await self.broadcast_message(EventType.ERROR, {
                "error": "No card specified"
            }, event.player_id)
            return

        result = whot_engine.apply_play(
            state,
            event.player_id,
            cards_to_play,
            requested_shape=payload.get("requested_shape"),
            card_indices=payload.get("card_indices") or None,
            card_index=payload.get("card_index"),
            require_eligible=True,
        )
        if not result.success:
            await self.broadcast_message(EventType.ERROR, {
                "error": result.error or "Invalid play"
            }, event.player_id)
            return

        if result.game_ended:
            await self.broadcast_message(EventType.GAME_ENDED, {
                "winner_id": str(result.winner_id) if result.winner_id else None,
                "winner_name": result.winner_name,
            })

        await self.set_game_state(state)
        await self.broadcast_game_state()

    def remove_card_from_player(self, player: Player, card: Card, card_index: Optional[int] = None) -> None:
        whot_engine.remove_card_from_player(player, card, card_index)

    async def handle_pick_card(self, event: GameEvent, state: GameState) -> None:
        result = whot_engine.apply_pick(
            state,
            event.player_id,
            require_eligible=True,
        )
        if not result.success:
            await self.broadcast_message(EventType.ERROR, {
                "error": result.error or "Invalid pick"
            }, event.player_id)
            return

        await self.set_game_state(state)
        await self.broadcast_game_state()

    async def handle_use_whot(self, event: GameEvent, state: GameState) -> None:
        result = whot_engine.apply_use_whot(
            state,
            event.player_id,
            event.payload.get("requested_shape") or "",
            require_eligible=True,
        )
        if not result.success:
            await self.broadcast_message(EventType.ERROR, {
                "error": result.error or "Invalid whot"
            }, event.player_id)
            return

        if result.game_ended:
            await self.broadcast_message(EventType.GAME_ENDED, {
                "winner_id": str(result.winner_id) if result.winner_id else None,
                "winner_name": result.winner_name,
            })

        await self.set_game_state(state)
        await self.broadcast_game_state()

    def is_valid_play_stack(self, cards: list[Card], face_card: Card, state: GameState, player: Player) -> bool:
        return whot_engine.is_valid_play_stack(cards, face_card, state, player)

    def is_valid_first_card(self, card: Card, face_card: Card, state: GameState) -> bool:
        return whot_engine.is_valid_first_card(card, face_card, state)

    def is_valid_play(self, card: Card, face_card: Card, state: GameState) -> bool:
        return whot_engine.is_valid_play(card, face_card, state)

    async def handle_whot_card(self, state: GameState, requested_shape: str) -> None:
        whot_engine._apply_whot_shape(state, requested_shape, require_eligible=True)

    async def handle_pick_2(self, state: GameState) -> None:
        whot_engine._apply_pick_2(state, require_eligible=True)

    async def handle_pick_5(self, state: GameState) -> None:
        whot_engine._apply_pick_5(state, require_eligible=True)

    async def handle_hold_on(self, state: GameState) -> None:
        return

    async def handle_suspension(self, state: GameState) -> None:
        whot_engine._apply_suspension(state, require_eligible=True)

    async def handle_general_market(self, state: GameState) -> None:
        whot_engine._apply_general_market(state)

    def next_turn(self, state: GameState) -> None:
        whot_engine.next_turn(state, require_eligible=True)

    def get_current_player_index(self, state: GameState) -> int:
        return whot_engine.get_current_player_index(state)

    def get_next_player_index(self, state: GameState) -> int:
        return whot_engine.get_next_player_index(state)

    def can_defend_pick_chain(self, player: Player, state: GameState) -> bool:
        return whot_engine.can_defend_pick_chain(player, state)

    async def reset_market(self, state: GameState) -> None:
        whot_engine.reset_market(state)
 
    async def start_game(self) -> None:
        async with self.redis_service.game_lock(self.game_id):
            state = await self._get_game_state()
            if not state:
                return
            
            if state.status != GameStatus.WAITING_FOR_PLAYERS:
                return
            
            human_players = [p for p in state.players if not p.is_cpu]
            if not human_players:
                await self.broadcast_message(EventType.ERROR, {
                    "error": "Game must have at least one human player"
                })
                return
            
            if state.settings.fill_with_computers:
                num_cpus_needed = state.settings.num_players - len(state.players)
                
                for i in range(num_cpus_needed):
                    cpu_id = uuid4()
                    cpu_name = f"CPU {i + 1}"
                    cpu_player = Player(
                        id=cpu_id,
                        name=cpu_name,
                        is_connected=True,
                        is_cpu=True
                    )
                    state.players.append(cpu_player)
                    await self.redis_service.add_player_to_game(self.game_id, cpu_id)
            
            deck_cards, init_card = self.create_deck()
            state.market = Deck(cards=deck_cards)
            state.discard_pile = [init_card]
            state.current_face_card = init_card
            
            cards_per_player = state.settings.num_starting_cards
            for player in state.players:
                for _ in range(cards_per_player):
                    if len(state.market.cards) > 0:
                        player.cards.append(state.market.cards.pop())
            
            state.status = GameStatus.IN_PROGRESS
            state.started_at = datetime.utcnow()
            state.current_player_id = state.players[0].id if state.players else None
            
            await self.set_game_state(state)
            await self.broadcast_message(EventType.GAME_STARTED, {
                "message": "Game has started"
            })
            await self.broadcast_game_state()
        
        state = await self._get_game_state()
        if state:
            self.schedule_cpu_check(state)
        
    def schedule_cpu_check(self, state: GameState) -> None:
        """Run CPU turns in the background so WS handlers stay responsive."""
        asyncio.create_task(
            self._run_cpu_check(state),
            name=f"cpu-check-{self.game_id}",
        )

    async def _run_cpu_check(self, state: GameState) -> None:
        try:
            await self.check_and_trigger_cpu_if_needed(state)
        except Exception:
            logger.exception("CPU check failed for game %s", self.game_id)

    async def check_and_trigger_cpu_if_needed(self, state: GameState) -> None:
        if state.status != GameStatus.IN_PROGRESS:
            return

        if not self._has_connected_human(state):
            async with self.redis_service.game_lock(self.game_id):
                latest = await self._get_game_state()
                if (
                    latest
                    and latest.status == GameStatus.IN_PROGRESS
                    and not self._has_connected_human(latest)
                ):
                    await self._finish_cpu_only_game(latest)
            return
        
        if not state.current_player_id:
            return
        
        current_player = next((p for p in state.players if p.id == state.current_player_id), None)
        if not current_player:
            return

        if not self._is_eligible_turn(current_player):
            async with self.redis_service.game_lock(self.game_id):
                latest = await self._get_game_state()
                if not latest or latest.status != GameStatus.IN_PROGRESS:
                    return
                self._ensure_turn_eligible(latest)
                if not self._has_connected_human(latest):
                    await self._finish_cpu_only_game(latest)
                    return
                await self.set_game_state(latest)
                await self.broadcast_game_state()
            latest = await self._get_game_state()
            if latest:
                await self.check_and_trigger_cpu_if_needed(latest)
            return
        
        if current_player.is_cpu:
            cpu_id = current_player.id
            await asyncio.sleep(CPU_MOVE_DELAY_SECONDS)

            latest = await self._get_game_state()
            if (
                not latest
                or latest.status != GameStatus.IN_PROGRESS
                or latest.current_player_id != cpu_id
            ):
                return

            from api.services.cpu import CPUPlayer
            cpu_difficulty = (
                latest.settings.cpu_difficulty
                if latest.settings.fill_with_computers
                else "normal"
            )
            cpu_player = CPUPlayer(cpu_id, self.redis_service, difficulty=cpu_difficulty)
            await cpu_player.make_move(self.game_id, latest)
    
    async def broadcast_game_state(self) -> None:
        """Publish a lightweight signal; each WS connection builds a filtered view."""
        await self.broadcast_message(EventType.GAME_STATE, {"changed": True})
        
    async def get_game_state_for_player(self, viewer_id: UUID) -> GameStateForPlayer | None:
        state = await self._get_game_state()
        if not state:
            return None
        
        players_for_view = []
        for p in state.players:
            if p.id == viewer_id:
                players_for_view.append(PlayerForPlayerView(
                    id=p.id,
                    name=p.name,
                    is_connected=p.is_connected,
                    is_cpu=p.is_cpu,
                    card_count=len(p.cards),
                    cards=p.cards,
                    cards_played=p.cards_played,
                    cards_drawn=p.cards_drawn,
                ))
            else:
                players_for_view.append(PlayerForPlayerView(
                    id=p.id,
                    name=p.name,
                    is_connected=p.is_connected,
                    is_cpu=p.is_cpu,
                    card_count=len(p.cards),
                    cards=None,
                    cards_played=p.cards_played,
                    cards_drawn=p.cards_drawn,
                ))
        
        return GameStateForPlayer(
            game_id=state.game_id,
            players=players_for_view,
            spectators=state.spectators,
            num_spectators=state.num_spectators,
            status=state.status,
            current_face_card=state.current_face_card,
            current_player_id=state.current_player_id,
            winner_id=state.winner_id,
            time_elapsed=state.time_elapsed,
            direction=state.direction,
            pick_chain_count=state.pick_chain_count,
            pick_chain_type=state.pick_chain_type,
            last_action=state.last_action,
            market_count=len(state.market.cards),
            discard_pile_count=len(state.discard_pile),
            created_at=state.created_at,
            started_at=state.started_at,
            ended_at=state.ended_at,
            abandon_at=state.abandon_at,
            settings=state.settings
        )
        
    async def get_game_state_for_spectator(self) -> GameStateForSpectator | None:
        state = await self._get_game_state()
        if not state:
            return None
        
        players_for_spectator = [
            PlayerForSpectator(
                id=p.id,
                name=p.name,
                is_connected=p.is_connected,
                card_count=len(p.cards),
                is_cpu=p.is_cpu,
                cards_played=p.cards_played,
                cards_drawn=p.cards_drawn,
            )
            for p in state.players
        ]
        
        return GameStateForSpectator(
            game_id=state.game_id,
            players=players_for_spectator,
            spectators=state.spectators,
            num_spectators=state.num_spectators,
            status=state.status,
            current_face_card=state.current_face_card,
            current_player_id=state.current_player_id,
            winner_id=state.winner_id,
            time_elapsed=state.time_elapsed,
            direction=state.direction,
            pick_chain_count=state.pick_chain_count,
            pick_chain_type=state.pick_chain_type,
            last_action=state.last_action,
            market_count=len(state.market.cards),
            discard_pile_count=len(state.discard_pile),
            created_at=state.created_at,
            started_at=state.started_at,
            ended_at=state.ended_at,
            abandon_at=state.abandon_at,
            settings=state.settings
        )
