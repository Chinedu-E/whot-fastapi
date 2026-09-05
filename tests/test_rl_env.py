"""Tests for the Gymnasium WhotEnv."""

import numpy as np

from api.models.game import Card, Deck, GameStatus
from rl.encode import observation_size
from rl.env import MAX_ACTIONS, WhotEnv


def test_reset_obs_and_mask_shape():
    env = WhotEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (observation_size(),)
    assert obs.dtype == np.float32
    mask = info["action_mask"]
    assert mask.shape == (MAX_ACTIONS,)
    assert mask.dtype == np.bool_
    assert mask.any()
    assert info["legal_count"] >= 1


def test_step_legal_action():
    env = WhotEnv()
    obs, info = env.reset(seed=1)
    action = int(np.flatnonzero(info["action_mask"])[0])
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert obs2.shape == (observation_size(),)
    assert reward in (-1.0, 0.0, 1.0)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    if not (terminated or truncated):
        assert info2["action_mask"].any()


def test_invalid_action_fallback():
    env = WhotEnv()
    _, info = env.reset(seed=2)
    assert info["legal_count"] < MAX_ACTIONS
    obs, reward, terminated, truncated, info2 = env.step(MAX_ACTIONS - 1)
    assert obs.shape == (observation_size(),)
    assert reward in (-1.0, 0.0, 1.0)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_smoke_random_episodes():
    env = WhotEnv(max_steps=200)
    for ep in range(200):
        obs, info = env.reset(seed=ep)
        done = False
        steps = 0
        while not done:
            mask = info["action_mask"]
            legal_idxs = np.flatnonzero(mask)
            assert len(legal_idxs) > 0
            action = int(legal_idxs[ep % len(legal_idxs)])
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            assert steps <= env.max_steps + 5
        assert reward in (-1.0, 0.0, 1.0)


def test_agent_win_reward():
    env = WhotEnv(can_win_with_action=False)
    env.reset(seed=0)
    assert env.state is not None and env.agent_id is not None

    agent = next(p for p in env.state.players if p.id == env.agent_id)
    opponent = next(p for p in env.state.players if p.id == env.opponent_id)
    agent.cards = [Card(shape="circle", number=7)]
    opponent.cards = [Card(shape="square", number=3), Card(shape="star", number=4)]
    env.state.current_face_card = Card(shape="circle", number=3)
    env.state.discard_pile = [env.state.current_face_card]
    env.state.current_player_id = env.agent_id
    env.state.market = Deck(cards=[Card(shape="star", number=7)] * 5)
    env.state.status = GameStatus.IN_PROGRESS
    env._steps = 0

    from rl.actions import legal_actions

    legal = legal_actions(env.state, env.agent_id)
    play_idx = next(
        i
        for i, a in enumerate(legal)
        if a.kind == "PLAY"
        and len(a.cards) == 1
        and a.cards[0].number == 7
        and a.cards[0].shape == "circle"
    )
    _, reward, terminated, truncated, _ = env.step(play_idx)
    assert terminated is True
    assert truncated is False
    assert reward == 1.0
