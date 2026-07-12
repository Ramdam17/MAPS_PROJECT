"""Production MARL runner — corrected engine (fix a: metacognition acts) on MeltingPot MAPPO.

One invocation trains ONE cell = (substrate × setting × seed). It:
  1. loads config from ``config/`` (training.yaml + factorial_marl.yaml + env/<substrate>.yaml),
  2. probes env shapes in a throwaway subprocess so the torch nets are built BEFORE TF/meltingpot
     load (torch-after-TF segfaults — see docs/reviews / memory),
  3. builds the corrected per-agent engine (R_MAPPOPolicy + R_MAPPO + SeparatedReplayBuffer),
  4. trains to ``num_env_steps`` with periodic **checkpoint/resume** (cascade cells at 1M steps
     can exceed a single wall-time window — resume + --requeue survive preemption),
  5. writes ``metrics.json`` in the SAME schema as the April pipeline (Table-7 ready).

Requires the py3.11 ``.venv-marl`` (meltingpot); see docs/install_marl_drac.md.

Usage
-----
    python scripts/run_marl.py --substrate territory_inside_out --setting maps --seed 42 \
        --num-env-steps 1000000 --device cuda --output-dir <dir> --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# torch + tensorflow(meltingpot) segfault if TF loads before the torch nets are built
# (orthogonal init → LAPACK vs TF's BLAS/OpenMP). Force single-threaded OpenMP + build torch first.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from gymnasium import spaces
from omegaconf import OmegaConf

torch.set_num_threads(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_marl")

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "config"
sys.path.insert(0, str(REPO / "src"))

GAMMA = 0.99       # standard MAPPO (not in training.yaml; matches the April pipeline default)
GAE_LAMBDA = 0.95


def load_configs(substrate: str, setting_id: str):
    train = OmegaConf.load(CFG / "domains" / "marl" / "training.yaml")
    fact = OmegaConf.load(CFG / "experiments" / "factorial_marl.yaml")
    envc = OmegaConf.load(CFG / "domains" / "marl" / "env" / f"{substrate}.yaml")
    try:
        setting = next(s for s in fact.settings if s.id == setting_id)
    except StopIteration as e:
        valid = [s.id for s in fact.settings]
        raise SystemExit(f"unknown setting {setting_id!r}; valid: {valid}") from e
    return train, setting, envc


def build_engine(train, setting, obs_space, share_obs_space, action_space, episode_length, num_agents, device):
    from maps.domains.marl.data import SeparatedReplayBuffer
    from maps.domains.marl.rmappo_policy import R_MAPPOPolicy
    from maps.domains.marl.trainer import R_MAPPO

    m, o, p = train.model, train.optimizer, train.ppo
    policies, trainers, buffers = [], [], []
    for _ in range(num_agents):
        policy = R_MAPPOPolicy(
            obs_space=obs_space,
            cent_obs_space=share_obs_space,
            act_space=action_space,
            hidden_size=int(m.hidden_size),
            recurrent_n=int(m.recurrent_n),
            cascade_iterations_1=int(setting.cascade_iterations1),
            cascade_iterations_2=int(setting.cascade_iterations2),
            lr=float(o.actor_lr),
            critic_lr=float(o.critic_lr),
            opti_eps=float(o.opti_eps),
            weight_decay=float(o.weight_decay),
            optimizer=str(o.name).upper(),
            use_orthogonal=bool(m.use_orthogonal),
            gain=float(m.gain),
            device=device,
        )
        trainer = R_MAPPO(
            policy,
            hidden_size=int(m.hidden_size),
            clip_param=float(p.clip_param),
            ppo_epoch=int(p.ppo_epoch),
            num_mini_batch=int(p.num_mini_batch),
            data_chunk_length=int(p.data_chunk_length),
            value_loss_coef=float(p.value_loss_coef),
            entropy_coef=float(p.entropy_coef),
            max_grad_norm=float(p.max_grad_norm),
            huber_delta=float(p.huber_delta),
            use_valuenorm=bool(p.use_valuenorm),
            device=device,
        )
        buffer = SeparatedReplayBuffer(
            episode_length=episode_length,
            n_rollout_threads=1,
            hidden_size=int(m.hidden_size),
            recurrent_n=int(m.recurrent_n),
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            obs_shape=tuple(obs_space.shape),
            share_obs_shape=tuple(share_obs_space.shape),
            num_actions=int(action_space.n),
            use_valuenorm=bool(p.use_valuenorm),
        )
        policies.append(policy)
        trainers.append(trainer)
        buffers.append(buffer)
    return policies, trainers, buffers


def _probe_shapes(substrate, episode_length):
    """Subprocess in --probe-shapes mode → env obs/act shapes WITHOUT loading TF here."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--probe-shapes",
           "--substrate", substrate, "--episode-length", str(episode_length)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("SHAPES_JSON:"):
            return json.loads(line[len("SHAPES_JSON:") :])
    raise RuntimeError(f"probe failed (rc={out.returncode})\nSTDOUT:\n{out.stdout}\nSTDERR:\n{out.stderr}")


def save_checkpoint(path: Path, next_episode, all_infos, policies, trainers):
    ckpt = {
        "next_episode": next_episode,
        "all_infos": all_infos,
        "agents": [
            {
                "actor": p.actor.state_dict(),
                "critic": p.critic.state_dict(),
                "actor_opt": p.actor_optimizer.state_dict(),
                "critic_opt": p.critic_optimizer.state_dict(),
                "value_normalizer": (
                    t.value_normalizer.state_dict() if t.value_normalizer is not None else None
                ),
            }
            for p, t in zip(policies, trainers, strict=True)
        ],
        "numpy_rng": np.random.get_state(),  # noqa: NPY002  (global RNG state for exact resume)
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)  # atomic


def load_checkpoint(path: Path, policies, trainers, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    for agent, p, t in zip(ckpt["agents"], policies, trainers, strict=True):
        p.actor.load_state_dict(agent["actor"])
        p.critic.load_state_dict(agent["critic"])
        p.actor_optimizer.load_state_dict(agent["actor_opt"])
        p.critic_optimizer.load_state_dict(agent["critic_opt"])
        if agent["value_normalizer"] is not None and t.value_normalizer is not None:
            t.value_normalizer.load_state_dict(agent["value_normalizer"])
    np.random.set_state(ckpt["numpy_rng"])  # noqa: NPY002  (global RNG state for exact resume)
    # S-C1: coerce RNG state to a CPU uint8 tensor (torch.set_rng_state is strict).
    torch.set_rng_state(ckpt["torch_rng"].cpu().to(torch.uint8))
    if ckpt.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    return int(ckpt["next_episode"]), list(ckpt["all_infos"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--substrate", required=True)
    ap.add_argument("--setting", default="baseline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-env-steps", type=int, default=None, help="override training.num_env_steps")
    ap.add_argument("--episode-length", type=int, default=None, help="override env episode_length")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", default=None, help="required except in --probe-shapes mode")
    ap.add_argument("--resume", action="store_true", help="resume from <output-dir>/checkpoint.pt if present")
    ap.add_argument("--probe-shapes", action="store_true", help="internal: emit env shapes JSON and exit")
    args = ap.parse_args()

    from maps.domains.marl.env import env_creator
    from maps.utils.seeding import set_all_seeds

    train, setting, envc = load_configs(args.substrate, args.setting)
    substrate_name = str(envc.substrate_name)
    roles = list(envc.roles)
    num_agents = int(envc.num_agents)
    scaled = int(envc.get("downsample_scale", 8))
    episode_length = args.episode_length or int(envc.episode_length)
    num_env_steps = args.num_env_steps or int(train.training.num_env_steps)
    save_interval = int(train.training.save_interval)
    log_interval = int(train.training.log_interval)

    # ── probe subprocess: build env, emit shapes, exit (isolates TF from the main process) ──
    if args.probe_shapes:
        env = env_creator(substrate_name, roles=roles, scaled=scaled, max_cycles=episode_length)
        p0 = "player_0"
        shapes = dict(
            obs_shape=list(env.observation_space[p0]["RGB"].shape),
            share_obs_shape=list(env.share_observation_space[p0].shape),
            num_actions=int(env.action_space[p0].n),
        )
        env.close()
        print("SHAPES_JSON:" + json.dumps(shapes))  # noqa: T201  (stdout IPC to _probe_shapes)
        return

    if args.output_dir is None:
        raise SystemExit("--output-dir is required (except with --probe-shapes)")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.pt"

    log.info(
        "MARL cell: substrate=%s setting=%s(meta=%s casc=%d/%d) seed=%d agents=%d steps=%d device=%s",
        args.substrate, setting.id, setting.meta, setting.cascade_iterations1,
        setting.cascade_iterations2, args.seed, num_agents, num_env_steps, args.device,
    )
    set_all_seeds(args.seed)

    # Build the torch engine BEFORE loading TF (probe shapes in a TF-isolated subprocess).
    shapes = _probe_shapes(args.substrate, episode_length)
    obs_space = spaces.Box(0, 255, tuple(shapes["obs_shape"]), np.float32)
    share_obs_space = spaces.Box(0, 255, tuple(shapes["share_obs_shape"]), np.float32)
    action_space = spaces.Discrete(int(shapes["num_actions"]))
    policies, trainers, buffers = build_engine(
        train, setting, obs_space, share_obs_space, action_space, episode_length, num_agents, args.device
    )

    # Resume?
    start_episode, all_infos = 0, []
    if args.resume and checkpoint_path.is_file():
        start_episode, all_infos = load_checkpoint(checkpoint_path, policies, trainers, args.device)
        log.info("resumed from %s at episode %d", checkpoint_path, start_episode)

    # NOW build the env (TF loads AFTER the torch nets exist).
    env = env_creator(substrate_name, roles=roles, scaled=scaled, max_cycles=episode_length)

    def warmup():
        obs_dict, _ = env.reset()
        for aid in range(num_agents):
            pk = f"player_{aid}"
            buffers[aid].obs[0] = np.asarray(obs_dict[pk]["RGB"])
            buffers[aid].share_obs[0] = np.asarray(obs_dict[pk]["WORLD.RGB"])

    num_episodes = max(1, num_env_steps // episode_length)
    t0 = time.time()
    try:
        for episode in range(start_episode, num_episodes):
            warmup()
            episode_reward_sum = np.zeros((num_agents, 1), dtype=np.float32)

            for step in range(episode_length):
                rollout = []
                for aid in range(num_agents):
                    trainers[aid].prep_rollout()
                    buf = buffers[aid]
                    with torch.no_grad():
                        values, actions, log_probs, rnn_a, rnn_c = policies[aid].get_actions(
                            buf.share_obs[step], buf.obs[step],
                            buf.rnn_states[step], buf.rnn_states_critic[step], buf.masks[step],
                        )
                    rollout.append(
                        dict(
                            values=values.detach().cpu().numpy(),
                            actions=actions.detach().cpu().numpy(),
                            log_probs=log_probs.detach().cpu().numpy(),
                            rnn_a=rnn_a.detach().cpu().numpy(),
                            rnn_c=rnn_c.detach().cpu().numpy(),
                        )
                    )
                action_dict = {f"player_{aid}": rollout[aid]["actions"] for aid in range(num_agents)}
                obs2, rew, done, _ = env.step(action_dict)
                for aid in range(num_agents):
                    pk = f"player_{aid}"
                    r = np.asarray(rew[pk], dtype=np.float32).reshape(1, 1)
                    mask = 1.0 - np.asarray(done[pk]).reshape(1, 1).astype(np.float32)
                    buffers[aid].insert(
                        share_obs=obs2[pk]["WORLD.RGB"], obs=obs2[pk]["RGB"],
                        rnn_states=rollout[aid]["rnn_a"], rnn_states_critic=rollout[aid]["rnn_c"],
                        actions=rollout[aid]["actions"], action_log_probs=rollout[aid]["log_probs"],
                        value_preds=rollout[aid]["values"], rewards=r, masks=mask, active_masks=mask,
                    )
                    episode_reward_sum[aid, 0] += float(r[0, 0])

            for aid in range(num_agents):
                trainers[aid].prep_rollout()
                buf = buffers[aid]
                with torch.no_grad():
                    next_values = policies[aid].get_values(buf.share_obs[-1], buf.rnn_states_critic[-1], buf.masks[-1])
                buf.compute_returns(next_values.detach().cpu().numpy(), trainers[aid].value_normalizer)

            infos = []
            for aid in range(num_agents):
                trainers[aid].prep_training()
                info = trainers[aid].train(buffers[aid], update_actor=True, meta=bool(setting.meta))
                info["wager_loss_actor"] = info.pop("wager_loss", 0.0)
                info["wager_loss_critic"] = 0.0  # fix (a): no wager on the critic side
                infos.append(info)
                buffers[aid].after_update()

            total_steps = (episode + 1) * episode_length
            mean_return = float(episode_reward_sum.mean())
            all_infos.append(
                dict(
                    episode=episode, per_agent=infos, total_steps=total_steps,
                    episode_return_mean=mean_return,
                    episode_return_per_agent=episode_reward_sum.mean(axis=1).tolist(),
                )
            )
            if episode % log_interval == 0:
                log.info("episode %d/%d steps=%d return=%.4f elapsed=%.1fs",
                         episode + 1, num_episodes, total_steps, mean_return, time.time() - t0)
            if save_interval > 0 and (episode + 1) % save_interval == 0:
                save_checkpoint(checkpoint_path, episode + 1, all_infos, policies, trainers)
    finally:
        env.close()

    save_checkpoint(checkpoint_path, num_episodes, all_infos, policies, trainers)
    elapsed = time.time() - t0
    payload = dict(
        meta=dict(
            engine="new-fixa", substrate=args.substrate,
            setting=dict(id=setting.id, label=str(setting.get("label", "")), meta=bool(setting.meta),
                         cascade_iterations1=int(setting.cascade_iterations1),
                         cascade_iterations2=int(setting.cascade_iterations2)),
            seed=args.seed, num_env_steps=num_env_steps, episode_length=episode_length,
            n_rollout_threads=1, num_agents=num_agents, elapsed_s=elapsed,
        ),
        episodes=all_infos,
    )
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    log.info("done: episodes=%d elapsed=%.1fs → %s", len(all_infos), elapsed, metrics_path)


if __name__ == "__main__":
    main()
