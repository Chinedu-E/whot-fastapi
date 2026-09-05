"""Smoke tests for self-play policy opponents and league."""

from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from rl.actions import action_key, legal_actions
from rl.env import WhotEnv
from rl.league import League
from rl.policy_opponent import FrozenPolicyOpponent


def _mask_fn(env: WhotEnv) -> np.ndarray:
    return env.action_masks()


def _tiny_model(tmp_path: Path) -> tuple[MaskablePPO, Path]:
    env = ActionMasker(WhotEnv(opponent="random"), _mask_fn)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=32,
        batch_size=16,
        verbose=0,
        seed=0,
    )
    model.learn(total_timesteps=64, progress_bar=False)
    path = tmp_path / "policy"
    model.save(str(path))
    zip_path = path.with_suffix(".zip")
    return model, zip_path


def test_frozen_policy_opponent_legal_actions(tmp_path):
    _, zip_path = _tiny_model(tmp_path)
    frozen = FrozenPolicyOpponent.load(zip_path)
    env = WhotEnv(opponent="policy", policy_opponent=frozen)
    env.reset(seed=0)
    assert env.state is not None and env.opponent_id is not None

    # Force a few opponent turns by stepping agent with legal actions
    for _ in range(8):
        if env.state.status.name != "IN_PROGRESS":
            break
        mask = env.action_masks()
        idxs = np.flatnonzero(mask)
        if len(idxs) == 0:
            break
        obs, reward, terminated, truncated, info = env.step(int(idxs[0]))
        if terminated or truncated:
            break
        # After agent move, opponent may have acted; when it's agent turn again,
        # verify last constructed policy action path didn't leave illegal state.
        assert env.state is not None

    # Direct choose must be legal
    env2 = WhotEnv(opponent="policy", policy_opponent=frozen)
    env2.reset(seed=1)
    assert env2.state is not None and env2.opponent_id is not None
    # Give opponent the turn
    env2.state.current_player_id = env2.opponent_id
    legal = legal_actions(env2.state, env2.opponent_id)
    action = frozen.choose(env2.state, env2.opponent_id, env2._rng)
    assert action_key(action) in {action_key(a) for a in legal}


def test_league_max_size_and_sample(tmp_path):
    model, _ = _tiny_model(tmp_path)
    league_dir = tmp_path / "league"
    league = League(league_dir, max_size=3)
    for _ in range(5):
        league.add_snapshot(model)
    assert len(league) == 3
    assert len(list(league_dir.glob("snap_*.zip"))) == 3

    import random

    opp = league.sample(random.Random(0))
    assert isinstance(opp, FrozenPolicyOpponent)
    oldest = league.oldest_path()
    assert oldest is not None
    assert oldest.exists()


def test_selfplay_env_learn_smoke(tmp_path):
    model, zip_path = _tiny_model(tmp_path)
    league = League(tmp_path / "league_sp", max_size=2)
    league.add_snapshot(model)

    env = ActionMasker(WhotEnv(opponent="policy", league=league), _mask_fn)
    learner = MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=32,
        batch_size=16,
        verbose=0,
        seed=1,
    )
    learner.learn(total_timesteps=512, progress_bar=False)
    out = tmp_path / "selfplay_final"
    learner.save(str(out))
    assert out.with_suffix(".zip").exists()
