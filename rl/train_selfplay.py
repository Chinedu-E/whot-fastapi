"""Self-play training with a light historical league of frozen policies.

Usage:

  uv run python -m rl.train_selfplay \\
      --resume rl/checkpoints/best_model.zip \\
      --timesteps 100000 \\
      --league-size 3 \\
      --snapshot-every 10000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.env import WhotEnv
from rl.eval import evaluate_suite
from rl.league import League


def _mask_fn(env: WhotEnv) -> np.ndarray:
    return env.action_masks()


def make_selfplay_env(league: League, seed: int):
    def _thunk():
        env = WhotEnv(opponent="policy", league=league)
        env = ActionMasker(env, _mask_fn)
        env.reset(seed=seed)
        return env

    return _thunk


class SelfPlayCallback(BaseCallback):
    def __init__(
        self,
        league: League,
        save_dir: Path,
        snapshot_every: int,
        eval_every: int,
        eval_games: int,
        seed: int,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.league = league
        self.save_dir = save_dir
        self.snapshot_every = snapshot_every
        self.eval_every = eval_every
        self.eval_games = eval_games
        self.seed = seed
        self.best_normal_win_rate = -1.0
        self._last_snapshot = 0
        self._last_eval = 0

    def _on_step(self) -> bool:
        assert isinstance(self.model, MaskablePPO)

        if self.num_timesteps - self._last_snapshot >= self.snapshot_every:
            self._last_snapshot = self.num_timesteps
            path = self.league.add_snapshot(self.model)
            if self.verbose:
                print(f"[league t={self.num_timesteps}] snapshot -> {path} (size={len(self.league)})")

        if self.num_timesteps - self._last_eval >= self.eval_every:
            self._last_eval = self.num_timesteps
            oldest = self.league.oldest_path()
            suite = evaluate_suite(
                self.model,
                n_games=self.eval_games,
                seed=self.seed + self.num_timesteps,
                league_oldest=oldest,
            )
            if self.verbose:
                parts = [
                    f"{k}={v['win_rate']:.3f}({v['wins']}/{v['losses']}/{v['draws']})"
                    for k, v in suite.items()
                ]
                print(f"[eval t={self.num_timesteps}] " + " ".join(parts))

            normal_wr = suite.get("normal", {}).get("win_rate", -1.0)
            if normal_wr > self.best_normal_win_rate:
                self.best_normal_win_rate = normal_wr
                path = self.save_dir / "best_model"
                self.model.save(str(path))
                if self.verbose:
                    print(f"  saved best_model (normal win_rate={normal_wr:.3f})")

        return True


def train_selfplay(
    *,
    timesteps: int = 100_000,
    n_envs: int = 4,
    league_size: int = 3,
    snapshot_every: int = 10_000,
    eval_every: int = 10_000,
    eval_games: int = 40,
    save_dir: Path = Path("rl/checkpoints"),
    seed: int = 0,
    resume: Path | None = None,
) -> MaskablePPO:
    save_dir.mkdir(parents=True, exist_ok=True)
    league = League(save_dir / "league", max_size=league_size)

    # League must be non-empty before vec envs reset (they sample on reset).
    if len(league) == 0:
        if resume is not None:
            bootstrap = MaskablePPO.load(str(resume))
        else:
            boot_env = DummyVecEnv(
                [
                    lambda: ActionMasker(WhotEnv(opponent="random"), _mask_fn),
                ]
            )
            bootstrap = MaskablePPO(
                "MlpPolicy",
                boot_env,
                verbose=0,
                seed=seed,
                n_steps=64,
                batch_size=32,
            )
            boot_env.close()
        path = league.add_snapshot(bootstrap)
        print(f"seeded league with initial snapshot -> {path}")

    vec = DummyVecEnv([make_selfplay_env(league, seed + i) for i in range(n_envs)])

    if resume is not None:
        model = MaskablePPO.load(str(resume), env=vec)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec,
            verbose=1,
            seed=seed,
            n_steps=256,
            batch_size=64,
            learning_rate=3e-4,
        )

    callback = SelfPlayCallback(
        league=league,
        save_dir=save_dir,
        snapshot_every=snapshot_every,
        eval_every=eval_every,
        eval_games=eval_games,
        seed=seed,
    )
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)
    final_path = save_dir / "final_model"
    model.save(str(final_path))
    print(f"saved final_model -> {final_path}.zip")
    return model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Self-play MaskablePPO on Whot")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--league-size", type=int, default=3)
    parser.add_argument("--snapshot-every", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--save-dir", type=Path, default=Path("rl/checkpoints"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args(argv)

    train_selfplay(
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        league_size=args.league_size,
        snapshot_every=args.snapshot_every,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        save_dir=args.save_dir,
        seed=args.seed,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
