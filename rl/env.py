"""Gymnasium environment for 2-player Whot (agent vs heuristic or policy opponent)."""

from __future__ import annotations

import random
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from api.models.game import Deck, GameCreate, GameState, GameStatus, Player
from rl.actions import Action, legal_actions
from rl.apply_action import apply_action
from rl.encode import encode_observation, observation_size
from rl.engine import create_deck
from rl.opponents import EnvOpponent, choose_action

if TYPE_CHECKING:
    from rl.league import League
    from rl.policy_opponent import FrozenPolicyOpponent

MAX_ACTIONS = 256
MAX_STEPS = 500


class WhotEnv(gym.Env):
    """Single-agent Whot: seat 0 is the learner; seat 1 uses a sync opponent policy."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        opponent: EnvOpponent = "random",
        policy_opponent: FrozenPolicyOpponent | None = None,
        league: League | None = None,
        num_starting_cards: int = 4,
        can_win_with_action: bool = False,
        max_steps: int = MAX_STEPS,
        max_actions: int = MAX_ACTIONS,
    ):
        super().__init__()
        self.opponent: EnvOpponent = opponent
        self.policy_opponent = policy_opponent
        self.league = league
        self.num_starting_cards = num_starting_cards
        self.can_win_with_action = can_win_with_action
        self.max_steps = max_steps
        self.max_actions = max_actions

        obs_dim = observation_size()
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(max_actions)

        self._np_random: np.random.Generator | None = None
        self._rng = random.Random()
        self.state: GameState | None = None
        self.agent_id: UUID | None = None
        self.opponent_id: UUID | None = None
        self._steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._np_random is not None:
            self._rng = random.Random(int(self._np_random.integers(0, 2**31 - 1)))

        if self.league is not None and len(self.league) > 0:
            self.policy_opponent = self.league.sample(self._rng)
            self.opponent = "policy"

        self.agent_id = uuid4()
        self.opponent_id = uuid4()
        self._steps = 0

        market_cards, face = create_deck(rng=self._rng)
        agent = Player(
            id=self.agent_id,
            name="Agent",
            is_connected=True,
            is_cpu=True,
            cards=[],
        )
        opponent = Player(
            id=self.opponent_id,
            name="Opponent",
            is_connected=True,
            is_cpu=True,
            cards=[],
        )

        for _ in range(self.num_starting_cards):
            if market_cards:
                agent.cards.append(market_cards.pop())
            if market_cards:
                opponent.cards.append(market_cards.pop())

        settings = GameCreate(
            num_players=2,
            fill_with_computers=False,
            num_starting_cards=self.num_starting_cards,
            can_win_with_action=self.can_win_with_action,
        )
        self.state = GameState(
            game_id=uuid4(),
            deck=Deck(cards=[]),
            market=Deck(cards=market_cards),
            discard_pile=[face],
            players=[agent, opponent],
            spectators=[],
            num_spectators=0,
            status=GameStatus.IN_PROGRESS,
            current_face_card=face,
            current_player_id=agent.id,
            time_elapsed=0,
            direction=1,
            created_at=datetime.utcnow(),
            started_at=datetime.utcnow(),
            settings=settings,
        )

        self._advance_to_agent_turn()
        return self._obs(), self._info()

    def step(self, action: int):
        assert self.state is not None and self.agent_id is not None
        assert self.state.status == GameStatus.IN_PROGRESS

        legal = legal_actions(self.state, self.agent_id)
        if not legal:
            return self._obs(), 0.0, False, True, self._info()

        idx = int(action)
        if idx < 0 or idx >= len(legal) or idx >= self.max_actions:
            idx = 0

        result = apply_action(
            self.state,
            self.agent_id,
            legal[idx],
            require_eligible=False,
            rng=self._rng,
        )
        if not result.success:
            pick = next((a for a in legal if a.kind == "PICK"), legal[0])
            result = apply_action(
                self.state,
                self.agent_id,
                pick,
                require_eligible=False,
                rng=self._rng,
            )

        self._steps += 1

        reward, terminated, truncated = self._outcome_after_move(result)
        if terminated or truncated:
            return self._obs(), reward, terminated, truncated, self._info()

        self._advance_to_agent_turn()
        reward, terminated, truncated = self._check_terminal()
        if not terminated and not truncated and self._steps >= self.max_steps:
            truncated = True
            reward = 0.0

        return self._obs(), reward, terminated, truncated, self._info()

    def action_masks(self) -> np.ndarray:
        """Boolean mask over Discrete(max_actions) for sb3-contrib MaskablePPO."""
        assert self.state is not None and self.agent_id is not None
        legal = legal_actions(self.state, self.agent_id)
        return self._action_mask(legal)

    def _advance_to_agent_turn(self) -> None:
        assert self.state is not None and self.agent_id is not None
        guard = 0
        while (
            self.state.status == GameStatus.IN_PROGRESS
            and self.state.current_player_id != self.agent_id
            and guard < self.max_steps * 4
        ):
            self._play_opponent(self.state.current_player_id)
            guard += 1

    def _play_opponent(self, player_id: UUID | None) -> None:
        assert self.state is not None
        if player_id is None:
            return
        legal = legal_actions(self.state, player_id)
        if not legal:
            return

        if self.opponent == "policy":
            if self.policy_opponent is None:
                raise RuntimeError("opponent='policy' requires policy_opponent or a non-empty league")
            action = self.policy_opponent.choose(self.state, player_id, self._rng)
        else:
            action = choose_action(self.opponent, self.state, player_id, self._rng)

        result = apply_action(
            self.state,
            player_id,
            action,
            require_eligible=False,
            rng=self._rng,
        )
        if not result.success:
            pick = next((a for a in legal if a.kind == "PICK"), None)
            if pick is not None:
                apply_action(
                    self.state,
                    player_id,
                    pick,
                    require_eligible=False,
                    rng=self._rng,
                )

    def _outcome_after_move(self, result) -> tuple[float, bool, bool]:
        assert self.state is not None and self.agent_id is not None
        if result.game_ended or self.state.status == GameStatus.COMPLETED:
            if self.state.winner_id == self.agent_id:
                return 1.0, True, False
            return -1.0, True, False
        if self._steps >= self.max_steps:
            return 0.0, False, True
        return 0.0, False, False

    def _check_terminal(self) -> tuple[float, bool, bool]:
        assert self.state is not None and self.agent_id is not None
        if self.state.status == GameStatus.COMPLETED:
            if self.state.winner_id == self.agent_id:
                return 1.0, True, False
            return -1.0, True, False
        return 0.0, False, False

    def _obs(self) -> np.ndarray:
        assert self.state is not None and self.agent_id is not None
        return np.asarray(
            encode_observation(self.state, self.agent_id),
            dtype=np.float32,
        )

    def _action_mask(self, legal: list[Action]) -> np.ndarray:
        mask = np.zeros(self.max_actions, dtype=np.bool_)
        n = min(len(legal), self.max_actions)
        if n > 0:
            mask[:n] = True
        return mask

    def _info(self) -> dict:
        assert self.state is not None and self.agent_id is not None
        legal = legal_actions(self.state, self.agent_id)
        return {
            "action_mask": self._action_mask(legal),
            "legal_count": len(legal),
            "steps": self._steps,
        }
