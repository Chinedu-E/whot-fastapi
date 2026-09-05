"""Smoke tests for MaskablePPO training plumbing (short runs only)."""

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from rl.env import MAX_ACTIONS, WhotEnv
from rl.eval import evaluate_policy, make_eval_env


def _mask_fn(env: WhotEnv) -> np.ndarray:
    return env.action_masks()


def test_action_masks_normal_opponent():
    env = WhotEnv(opponent="normal")
    env.reset(seed=0)
    mask = env.action_masks()
    assert mask.shape == (MAX_ACTIONS,)
    assert mask.dtype == np.bool_
    assert mask.any()


def test_maskable_ppo_smoke_learn(tmp_path):
    env = ActionMasker(WhotEnv(opponent="random"), _mask_fn)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=64,
        batch_size=32,
        verbose=0,
        seed=0,
    )
    model.learn(total_timesteps=2048, progress_bar=False)
    path = tmp_path / "smoke_model"
    model.save(str(path))
    assert path.with_suffix(".zip").exists() or (tmp_path / "smoke_model.zip").exists()


def test_evaluate_policy_smoke():
    env = make_eval_env("random")
    model = MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=64,
        batch_size=32,
        verbose=0,
        seed=1,
    )
    # Untrained / briefly trained is fine — just exercise the harness
    model.learn(total_timesteps=256, progress_bar=False)
    stats = evaluate_policy(model, opponent="random", n_games=4, seed=0)
    assert stats["games"] == 4
    assert 0.0 <= stats["win_rate"] <= 1.0
    assert stats["wins"] + stats["losses"] + stats["draws"] == 4
