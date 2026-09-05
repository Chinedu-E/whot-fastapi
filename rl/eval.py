"""Evaluate a MaskablePPO checkpoint against a WhotEnv opponent.

Usage:
  uv run python -m rl.eval --checkpoint rl/checkpoints/best_model.zip \\
      --opponent random --games 200
  uv run python -m rl.eval --checkpoint rl/checkpoints/best_model.zip \\
      --opponent policy --opponent-checkpoint rl/checkpoints/league/snap_0000.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from rl.env import WhotEnv
from rl.opponents import EnvOpponent
from rl.policy_opponent import FrozenPolicyOpponent


def _mask_fn(env: WhotEnv) -> np.ndarray:
    return env.action_masks()


def make_eval_env(
    opponent: EnvOpponent = "random",
    *,
    opponent_checkpoint: Path | str | None = None,
    policy_opponent: FrozenPolicyOpponent | None = None,
) -> ActionMasker:
    if opponent == "policy":
        frozen = policy_opponent
        if frozen is None:
            if opponent_checkpoint is None:
                raise ValueError("policy opponent requires opponent_checkpoint or policy_opponent")
            frozen = FrozenPolicyOpponent.load(opponent_checkpoint)
        env = WhotEnv(opponent="policy", policy_opponent=frozen)
    else:
        env = WhotEnv(opponent=opponent)
    return ActionMasker(env, _mask_fn)


def evaluate_policy(
    model: MaskablePPO,
    opponent: EnvOpponent = "random",
    n_games: int = 50,
    seed: int = 0,
    deterministic: bool = True,
    opponent_checkpoint: Path | str | None = None,
    policy_opponent: FrozenPolicyOpponent | None = None,
) -> dict[str, Any]:
    """Play ``n_games`` and return win/loss/draw stats from the agent's view."""
    env = make_eval_env(
        opponent,
        opponent_checkpoint=opponent_checkpoint,
        policy_opponent=policy_opponent,
    )
    wins = losses = draws = 0
    lengths: list[int] = []

    for i in range(n_games):
        obs, _ = env.reset(seed=seed + i)
        done = False
        steps = 0
        last_reward = 0.0
        while not done:
            masks = env.action_masks()
            action, _ = model.predict(
                obs,
                action_masks=masks,
                deterministic=deterministic,
            )
            obs, reward, terminated, truncated, _ = env.step(int(action))
            last_reward = float(reward)
            done = terminated or truncated
            steps += 1
        lengths.append(steps)
        if last_reward > 0:
            wins += 1
        elif last_reward < 0:
            losses += 1
        else:
            draws += 1

    env.close()
    decided = wins + losses
    win_rate = (wins / decided) if decided else 0.0
    label: str = opponent
    if opponent == "policy" and opponent_checkpoint is not None:
        label = f"policy:{Path(opponent_checkpoint).name}"
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "games": n_games,
        "win_rate": win_rate,
        "mean_episode_length": float(np.mean(lengths)) if lengths else 0.0,
        "opponent": label,
    }


def evaluate_suite(
    model: MaskablePPO,
    *,
    n_games: int = 40,
    seed: int = 0,
    league_oldest: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Eval vs random, normal, and optionally the oldest league snapshot."""
    results: dict[str, dict[str, Any]] = {}
    for name in ("random", "normal"):
        results[name] = evaluate_policy(
            model,
            opponent=name,  # type: ignore[arg-type]
            n_games=n_games,
            seed=seed,
        )
    if league_oldest is not None and league_oldest.exists():
        results["league_oldest"] = evaluate_policy(
            model,
            opponent="policy",
            n_games=n_games,
            seed=seed + 10_000,
            opponent_checkpoint=league_oldest,
        )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Whot MaskablePPO checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--opponent",
        choices=["random", "easy", "normal", "policy"],
        default="random",
    )
    parser.add_argument("--opponent-checkpoint", type=Path, default=None)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.opponent == "policy" and args.opponent_checkpoint is None:
        parser.error("--opponent policy requires --opponent-checkpoint")

    model = MaskablePPO.load(str(args.checkpoint))
    stats = evaluate_policy(
        model,
        opponent=args.opponent,
        n_games=args.games,
        seed=args.seed,
        opponent_checkpoint=args.opponent_checkpoint,
    )
    print(
        f"opponent={stats['opponent']} games={stats['games']} "
        f"W/L/D={stats['wins']}/{stats['losses']}/{stats['draws']} "
        f"win_rate={stats['win_rate']:.3f} "
        f"mean_len={stats['mean_episode_length']:.1f}"
    )


if __name__ == "__main__":
    main()
