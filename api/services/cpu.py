import random
import logging
from typing import Literal
from uuid import UUID

from api.models.game import Card, GameEvent, GameState, Player
from api.services.redis_service import RedisService
from api.services.game import WhotGame
from api.core.config import settings

logger = logging.getLogger(__name__)

CPUDifficulty = Literal["easy", "normal", "hard"]


class CPUPlayer:
    def __init__(self, player_id: UUID, redis_service: RedisService, difficulty: CPUDifficulty = "normal"):
        self.player_id = player_id
        self.redis_service = redis_service
        self.difficulty = difficulty
    
    async def make_move(self, game_id: UUID, state: GameState) -> None:
        player = next((p for p in state.players if p.id == self.player_id), None)
        if not player or state.current_player_id != self.player_id:
            return
        
        game = WhotGame(game_id, self.redis_service)
        
        if state.pick_chain_count > 0:
            if self.can_defend_pick_chain(player, state):
                await self.defend_pick_chain(game_id, game, player, state)
            else:
                await self.pick_cards_from_chain(game_id, game)
        else:
            if self.difficulty == "easy":
                await self._make_move_easy(game_id, game, player, state)
            elif self.difficulty == "normal":
                await self._make_move_normal(game_id, game, player, state)
            elif self.difficulty == "hard":
                await self._make_move_hard(game_id, game, player, state)
    
    def can_defend_pick_chain(self, player: Player, state: GameState) -> bool:
        return any(c.number == state.pick_chain_type for c in player.cards)
    
    async def defend_pick_chain(self, game_id: UUID, game: WhotGame, player: Player, state: GameState) -> None:
        state = await game._get_game_state()
        if not state:
            return
        
        defend_cards = [c for c in player.cards if c.number == state.pick_chain_type]
        if not defend_cards:
            return
        
        # For easy difficulty, only play one card. For others, can stack
        if self.difficulty == "easy":
            card = defend_cards[0]
            event = GameEvent(
                action="PLAY_CARD",
                payload={"card": {"shape": card.shape, "number": card.number}},
                player_id=self.player_id,
                game_id=game_id
            )
        else:
            # Stack all matching cards
            cards_data = [{"shape": c.shape, "number": c.number} for c in defend_cards]
            event = GameEvent(
                action="PLAY_CARD",
                payload={"cards": cards_data},
                player_id=self.player_id,
                game_id=game_id
            )
        
        await game.process_game_event(event)
    
    async def pick_cards_from_chain(self, game_id: UUID, game: WhotGame) -> None:
        event = GameEvent(
            action="PICK_CARD",
            payload={},
            player_id=self.player_id,
            game_id=game_id
        )
        await game.process_game_event(event)
    
    # Easy Difficulty: Random valid card, no stacking
    async def _make_move_easy(self, game_id: UUID, game: WhotGame, player: Player, state: GameState) -> None:
        playable_cards = self._get_playable_cards(player, state)
        
        if not playable_cards:
            await self.pick_card(game_id, game)
            return
        
        # Randomly select one card (no stacking)
        card = random.choice(playable_cards)
        await self.play_card(game_id, game, player, [card], state)
    
    # Normal Difficulty: Basic strategy with stacking
    async def _make_move_normal(self, game_id: UUID, game: WhotGame, player: Player, state: GameState) -> None:
        playable_cards = self._get_playable_cards(player, state)
        
        if not playable_cards:
            await self.pick_card(game_id, game)
            return
        
        # Try to stack cards of the same number
        cards_to_play = self._find_best_stack_normal(player, playable_cards, state)
        
        if cards_to_play:
            await self.play_card(game_id, game, player, cards_to_play, state)
        else:
            # Play a single card with basic strategy
            card = self._choose_card_normal(player, playable_cards, state)
            await self.play_card(game_id, game, player, [card], state)
    
    # Hard Difficulty: Gemini AI with fallback
    async def _make_move_hard(self, game_id: UUID, game: WhotGame, player: Player, state: GameState) -> None:
        try:
            cards_to_play = await self._get_gemini_move(game_id, game, player, state)
            if cards_to_play:
                await self.play_card(game_id, game, player, cards_to_play, state)
            else:
                # Fallback to normal if Gemini returns nothing
                await self._make_move_normal(game_id, game, player, state)
        except Exception as e:
            logger.error(f"Gemini API error for player {self.player_id}: {e}")
            # Fallback to normal difficulty
            await self._make_move_normal(game_id, game, player, state)
    
    def _get_playable_cards(self, player: Player, state: GameState) -> list[Card]:
        """Get all cards that can be played"""
        playable = []
        for card in player.cards:
            if self.is_valid_play(card, state.current_face_card, state):
                playable.append(card)
        return playable
    
    def is_valid_play(self, card: Card, face_card: Card, state: GameState) -> bool:
        if card.number == 20:
            return True
        if face_card.number == 20:
            return card.shape == state.current_face_card.shape
        return card.shape == face_card.shape or card.number == face_card.number
    
    def _find_best_stack_normal(self, player: Player, playable_cards: list[Card], state: GameState) -> list[Card] | None:
        """Find the best stack of cards to play (normal difficulty)"""
        if not playable_cards:
            return None
        
        # Group playable cards by number (stacking requires same number)
        cards_by_number = {}
        for card in playable_cards:
            if card.number not in cards_by_number:
                cards_by_number[card.number] = []
            cards_by_number[card.number].append(card)
        
        # Find the largest stack we can make
        best_number = None
        max_count = 1
        
        for number, cards in cards_by_number.items():
            # Count how many cards of this number the player has (all shapes)
            # Only count cards that are actually playable
            player_count = len(cards)
            
            if player_count > max_count:
                max_count = player_count
                best_number = number
        
        if best_number and max_count > 1:
            # Return all playable cards of this number
            # playable_cards already contains references to cards in player.cards
            return [c for c in playable_cards if c.number == best_number]
        
        return None
    
    def _choose_card_normal(self, player: Player, playable_cards: list[Card], state: GameState) -> Card:
        """Choose a card with basic strategy (normal difficulty)"""
        # Prefer action cards (2, 5, 1, 8, 14, 20) when beneficial
        action_cards = [c for c in playable_cards if c.number in [1, 2, 5, 8, 14, 20]]
        
        if action_cards:
            # Prefer pick cards (2, 5) to put pressure on opponents
            pick_cards = [c for c in action_cards if c.number in [2, 5]]
            if pick_cards:
                return random.choice(pick_cards)
            return random.choice(action_cards)
        
        # Otherwise random
        return random.choice(playable_cards)
    
    async def _get_gemini_move(self, game_id: UUID, game: WhotGame, player: Player, state: GameState) -> list[Card] | None:
        """Get move from Gemini API"""
        if not settings.gemini_api_key:
            logger.warning("Gemini API key not configured, falling back to normal difficulty")
            return None
        
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=settings.gemini_api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            # Format game state for Gemini
            game_context = self._format_game_state_for_gemini(player, state)
            
            prompt = f"""You are playing Whot, a card game. Here's the current game state:

{game_context}

Your hand: {[f"{c.shape}-{c.number}" for c in player.cards]}

Current face card: {state.current_face_card.shape}-{state.current_face_card.number}

Rules:
- You can play a card if it matches the face card's shape OR number
- Whot cards (number 20) can be played anytime
- You can stack multiple cards of the same number
- Action cards: 1 (hold on), 2 (pick 2), 5 (pick 3), 8 (suspension), 14 (general market), 20 (whot - choose shape)
- Goal: Play all your cards to win

Respond with JSON only in this exact format:
{{
  "cards": [{{"shape": "circle", "number": 5}}, {{"shape": "circle", "number": 5}}],
  "requested_shape": "circle" (only if playing whot card)
}}

If you need to pick a card, respond with: {{"action": "pick"}}
If playing a single card, use "cards" array with one card.
If stacking, include all cards of the same number in "cards" array.
"""
            
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Parse JSON from response (might have markdown code blocks)
            import json
            import re
            
            # Remove markdown code blocks if present
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0)
            
            move_data = json.loads(response_text)
            
            if move_data.get("action") == "pick":
                await self.pick_card(game_id, game)
                return None
            
            cards_data = move_data.get("cards", [])
            if not cards_data:
                return None
            
            # Convert to Card objects and validate
            cards = []
            for card_data in cards_data:
                card = Card(shape=card_data["shape"], number=card_data["number"])
                # Validate the card is in player's hand and is playable
                card_in_hand = any(
                    c.shape == card.shape and c.number == card.number 
                    for c in player.cards
                )
                if card_in_hand and self.is_valid_play(card, state.current_face_card, state):
                    cards.append(card)
            
            if not cards:
                return None
            
            # If whot card, add requested shape
            if cards[0].number == 20 and "requested_shape" in move_data:
                # Store requested shape for later use
                self._last_whot_shape = move_data["requested_shape"]
            
            return cards
            
        except ImportError:
            logger.error("google-generativeai package not installed")
            return None
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return None
    
    def _format_game_state_for_gemini(self, player: Player, state: GameState) -> str:
        """Format game state as a string for Gemini"""
        lines = []
        lines.append(f"Number of players: {len(state.players)}")
        lines.append(f"Current player: {state.current_player_id}")
        lines.append(f"Direction: {'clockwise' if state.direction == 1 else 'counterclockwise'}")
        
        # Other players' card counts (not their actual cards for fairness)
        for p in state.players:
            if p.id != player.id:
                lines.append(f"Player {p.name}: {len(p.cards)} cards")
        
        lines.append(f"Market cards remaining: {len(state.market.cards)}")
        lines.append(f"Discard pile size: {len(state.discard_pile)}")
        
        if state.pick_chain_count > 0:
            lines.append(f"Active pick chain: {state.pick_chain_count} x {state.pick_chain_type}")
        
        if state.last_action:
            lines.append(f"Last action: {state.last_action}")
        
        return "\n".join(lines)
    
    async def play_card(self, game_id: UUID, game: WhotGame, player: Player, cards: list[Card], state: GameState) -> None:
        """Play one or more cards"""
        if not cards:
            await self.pick_card(game_id, game)
            return
        
        if len(cards) == 1:
            card = cards[0]
            payload: dict = {"card": {"shape": card.shape, "number": card.number}}
            if card.number == 20:
                requested_shape = getattr(self, '_last_whot_shape', None) or self.choose_shape_for_whot(player, state)
                payload["requested_shape"] = requested_shape
                self._last_whot_shape = None
        else:
            cards_data = [{"shape": c.shape, "number": c.number} for c in cards]
            payload = {"cards": cards_data}
            if cards[0].number == 20:
                requested_shape = getattr(self, '_last_whot_shape', None) or self.choose_shape_for_whot(player, state)
                payload["requested_shape"] = requested_shape
                self._last_whot_shape = None
        
        event = GameEvent(
            action="PLAY_CARD",
            payload=payload,
            player_id=self.player_id,
            game_id=game_id
        )
        await game.process_game_event(event)
    
    def choose_shape_for_whot(self, player: Player, state: GameState) -> str:
        """Choose shape for whot card based on hand composition"""
        shape_counts = {}
        for card in player.cards:
            if card.number != 20:
                shape_counts[card.shape] = shape_counts.get(card.shape, 0) + 1
        
        if shape_counts:
            return max(shape_counts, key=shape_counts.get)
        return "circle"
    
    async def pick_card(self, game_id: UUID, game: WhotGame) -> None:
        event = GameEvent(
            action="PICK_CARD",
            payload={},
            player_id=self.player_id,
            game_id=game_id
        )
        await game.process_game_event(event)
