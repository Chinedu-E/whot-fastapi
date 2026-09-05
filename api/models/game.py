from datetime import datetime
from enum import Enum
from uuid import UUID
from pydantic import BaseModel
from typing import Optional, Literal


class Card(BaseModel):
    shape: str
    number: int


class Deck(BaseModel):
    cards: list[Card]


class Player(BaseModel):
    id: UUID
    name: str
    is_connected: bool = True
    cards: list[Card] = []
    is_cpu: bool = False
    cards_played: int = 0
    cards_drawn: int = 0


class PlayerForSpectator(BaseModel):
    id: UUID
    name: str
    is_connected: bool = True
    card_count: int
    is_cpu: bool = False
    cards_played: int = 0
    cards_drawn: int = 0


class Spectator(BaseModel):
    id: UUID
    name: Optional[str] = None


class GameStatus(Enum):
    WAITING_FOR_PLAYERS = "waiting_for_players"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class GameCreate(BaseModel):
    num_players: int
    fill_with_computers: bool = False
    cpu_difficulty: Literal["easy", "normal", "hard"] = "normal"
    search_time_seconds: int = 60
    num_starting_cards: int = 4
    can_win_with_action: bool = False
    time_per_move: Optional[int] = None


class GameResponse(BaseModel):
    game_id: UUID
    created_at: datetime


class GameState(BaseModel):
    game_id: UUID
    deck: Deck
    market: Deck
    discard_pile: list[Card]
    players: list[Player]
    spectators: list[Spectator]
    num_spectators: int
    status: GameStatus
    current_face_card: Card
    current_player_id: Optional[UUID] = None
    winner_id: Optional[UUID] = None
    time_elapsed: int
    direction: int = 1
    pick_chain_count: int = 0
    pick_chain_type: Optional[Literal[2, 5]] = None
    last_action: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    abandon_at: Optional[datetime] = None
    settings: GameCreate


class GameStateForSpectator(BaseModel):
    game_id: UUID
    players: list[PlayerForSpectator]
    spectators: list[Spectator]
    num_spectators: int
    status: GameStatus
    current_face_card: Card
    current_player_id: Optional[UUID] = None
    winner_id: Optional[UUID] = None
    time_elapsed: int
    direction: int = 1
    pick_chain_count: int = 0
    pick_chain_type: Optional[Literal[2, 5]] = None
    last_action: Optional[str] = None
    market_count: int
    discard_pile_count: int
    created_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    abandon_at: Optional[datetime] = None
    settings: GameCreate


class PlayerForPlayerView(BaseModel):
    """Player as seen by another player: count only, or self with full hand."""
    id: UUID
    name: str
    is_connected: bool = True
    is_cpu: bool = False
    card_count: int
    cards: Optional[list[Card]] = None
    cards_played: int = 0
    cards_drawn: int = 0


class GameStateForPlayer(BaseModel):
    game_id: UUID
    players: list[PlayerForPlayerView]
    spectators: list[Spectator]
    num_spectators: int
    status: GameStatus
    current_face_card: Card
    current_player_id: Optional[UUID] = None
    winner_id: Optional[UUID] = None
    time_elapsed: int
    direction: int = 1
    pick_chain_count: int = 0
    pick_chain_type: Optional[Literal[2, 5]] = None
    last_action: Optional[str] = None
    market_count: int
    discard_pile_count: int
    created_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    abandon_at: Optional[datetime] = None
    settings: GameCreate


class GameListItem(BaseModel):
    game_id: UUID
    status: GameStatus
    player_count: int
    max_players: int
    created_at: datetime
    started_at: Optional[datetime] = None


class EventType(str, Enum):
    PLAY_CARD = "PLAY_CARD"
    PICK_CARD = "PICK_CARD"
    USE_WHOT = "USE_WHOT"
    GAME_STATE = "GAME_STATE"
    ERROR = "ERROR"
    GAME_STARTED = "GAME_STARTED"
    GAME_ENDED = "GAME_ENDED"
    PLAYER_JOINED = "PLAYER_JOINED"
    PLAYER_LEFT = "PLAYER_LEFT"
    TURN_CHANGED = "TURN_CHANGED"
    PING = "PING"
    LEAVE = "LEAVE"


class GamePayload(BaseModel):
    event_type: EventType
    data: dict
    game_id: UUID
    player_id: Optional[UUID] = None
    timestamp: datetime


class PlayCardEvent(BaseModel):
    card: Optional[Card] = None
    cards: Optional[list[Card]] = None
    card_index: Optional[int] = None
    requested_shape: Optional[str] = None


class PickCardEvent(BaseModel):
    pass


class UseWhotEvent(BaseModel):
    requested_shape: str


class GameEvent(BaseModel):
    action: Literal["PLAY_CARD", "PICK_CARD", "USE_WHOT", "LEAVE"]
    payload: dict
    player_id: UUID
    game_id: UUID