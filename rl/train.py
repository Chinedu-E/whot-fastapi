"""Train MaskablePPO on WhotEnv with action masking.

Curriculum (recommended):

  uv run python -m rl.train --opponent random --timesteps 100000
  uv run python -m rl.train --opponent normal --timesteps 100000 \\
      --resume rl/checkpoints/best_model.zip

Eval target (manual): >=55% win rate over 200 games vs random.
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
from rl.eval import evaluate_policy
from rl.opponents import OpponentName


def _mask_fn(env: WhotEnv) -> np.ndarray:
    return env.action_masks()


def make_env(opponent: OpponentName, seed: int):
    def _thunk():
        env = WhotEnv(opponent=opponent)
        env = ActionMasker(env, _mask_fn)
        env.reset(seed=seed)
        return env

    return _thunk


class EvalCallback(BaseCallback):
    def __init__(
        self,
        opponent: OpponentName,
        save_dir: Path,
        eval_every: int,
        eval_games: int,
        seed: int,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.opponent = opponent
        self.save_dir = save_dir
        self.eval_every = eval_every
        self.eval_games = eval_games
        self.seed = seed
        self.best_win_rate = -1.0
        self._last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.eval_every:
            return True
        self._last_eval = self.num_timesteps
        assert isinstance(self.model, MaskablePPO)
        stats = evaluate_policy(
            self.model,
            opponent=self.opponent,
            n_games=self.eval_games,
            seed=self.seed + self.num_timesteps,
        )
        wr = stats["win_rate"]
        if self.verbose:
            print(
                f"[eval t={self.num_timesteps}] win_rate={wr:.3f} "
                f"W/L/D={stats['wins']}/{stats['losses']}/{stats['draws']}"
            )
        if wr > self.best_win_rate:
            self.best_win_rate = wr
            path = self.save_dir / "best_model"
            self.model.save(str(path))
            if self.verbose:
                print(f"  saved best_model (win_rate={wr:.3f})")
        return True


def train(
    *,
    opponent: OpponentName = "random",
    timesteps: int = 100_000,
    n_envs: int = 4,
    eval_every: int = 10_000,
    eval_games: int = 50,
    save_dir: Path = Path("rl/checkpoints"),
    seed: int = 0,
    resume: Path | None = None,
) -> MaskablePPO:
    save_dir.mkdir(parents=True, exist_ok=True)
    vec = DummyVecEnv([make_env(opponent, seed + i) for i in range(n_envs)])

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

    callback = EvalCallback(
        opponent=opponent,
        save_dir=save_dir,
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
    parser = argparse.ArgumentParser(description="Train MaskablePPO on Whot")
    parser.add_argument(
        "--opponent",
        choices=["random", "easy", "normal"],
        default="random",
    )
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--save-dir", type=Path, default=Path("rl/checkpoints"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args(argv)

    train(
        opponent=args.opponent,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        save_dir=args.save_dir,
        seed=args.seed,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
