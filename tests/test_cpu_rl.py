"""Tests for live RL CPU difficulty wiring."""

import pytest

from api.models.game import Card
from api.services.cpu import CPUPlayer
from rl.actions import Action
from rl.infer import get_rl_opponent, reset_rl_opponent_for_tests


@pytest.fixture(autouse=True)
def _reset_infer_cache():
    reset_rl_opponent_for_tests()
    yield
    reset_rl_opponent_for_tests()


def test_get_rl_opponent_loads_production_checkpoint():
    opp = get_rl_opponent()
    assert opp is not None
    # Second call returns cached instance
    assert get_rl_opponent() is opp


@pytest.mark.asyncio
async def test_cpu_rl_plays_via_policy(two_player_game, monkeypatch):
    game = two_player_game["game"]
    redis = two_player_game["redis"]
    p1_id = two_player_game["p1_id"]
    game_id = two_player_game["game_id"]

    state = await game._get_game_state()
    # Face is circle-3; Alice has circle-7 playable
    state.current_player_id = p1_id
    await game.set_game_state(state)

    class FakeOpp:
        def choose(self, st, player_id, rng):
            return Action(
                kind="PLAY",
                cards=(Card(shape="circle", number=7),),
                requested_shape=None,
            )

    monkeypatch.setattr("rl.infer.get_rl_opponent", lambda checkpoint=None: FakeOpp())

    cpu = CPUPlayer(p1_id, redis, difficulty="hard")
    await cpu.make_move(game_id, state)

    state = await game._get_game_state()
    assert state.current_face_card.shape == "circle"
    assert state.current_face_card.number == 7


@pytest.mark.asyncio
async def test_cpu_rl_falls_back_when_unavailable(two_player_game, monkeypatch):
    game = two_player_game["game"]
    redis = two_player_game["redis"]
    p1_id = two_player_game["p1_id"]
    game_id = two_player_game["game_id"]

    state = await game._get_game_state()
    state.current_player_id = p1_id
    await game.set_game_state(state)

    monkeypatch.setattr("rl.infer.get_rl_opponent", lambda checkpoint=None: None)

    called = {"normal": False}

    async def fake_normal(self, game_id, game, player, state):
        called["normal"] = True
        # Play a safe card so the game advances
        await self.play_card(
            game_id,
            game,
            player,
            [Card(shape="circle", number=7)],
            state,
        )

    monkeypatch.setattr(CPUPlayer, "_make_move_normal", fake_normal)

    cpu = CPUPlayer(p1_id, redis, difficulty="hard")
    await cpu.make_move(game_id, state)
    assert called["normal"] is True
