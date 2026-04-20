"""Helpers for evaluation on observation-built RGB images."""

import os
import pickle
import numpy as np
import supersuit as ss
from typing import Dict, Iterable, List, Optional, Tuple

from pettingzoo.sisl import pursuit_v4
from pettingzoo.utils.wrappers import BaseParallelWrapper
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from scripts.train import train_pursuit_mappo_obs_images as train_mod


def sort_agent_ids(agent_ids: Iterable[str]) -> List[str]:
    def sort_key(agent_id: str):
        suffix = agent_id.rsplit("_", 1)[-1]
        return (0, int(suffix)) if suffix.isdigit() else (1, agent_id)

    return sorted(agent_ids, key=sort_key)


def get_restored_module_ids(trainer) -> List[str]:
    return sort_agent_ids(trainer.env_runner_group.local_env_runner.module.keys())


def infer_cell_scale_from_checkpoint(checkpoint_dir: str) -> Optional[int]:
    candidate_paths = [
        os.path.join(checkpoint_dir, "class_and_ctor_args.pkl"),
        os.path.join(checkpoint_dir, "learner_group", "class_and_ctor_args.pkl"),
        os.path.join(checkpoint_dir, "learner_group", "learner", "class_and_ctor_args.pkl"),
    ]
    for path in candidate_paths:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception:
            continue

        ctor_args = data.get("ctor_args_and_kwargs", ())
        cfg = None
        if isinstance(ctor_args, tuple) and len(ctor_args) >= 1:
            first = ctor_args[0]
            if isinstance(first, (list, tuple)) and first:
                cfg = first[0]
        if not isinstance(cfg, dict):
            cfg = {}

        rl_mod_spec = cfg.get("_rl_module_spec") or cfg.get("rl_module_spec")
        model_config = getattr(rl_mod_spec, "model_config", None) if rl_mod_spec else None
        conv_filters = getattr(model_config, "conv_filters", None) if model_config else None
        if not conv_filters:
            continue

        first = conv_filters[0]
        try:
            kernel = first[1][0] if isinstance(first[1], (list, tuple)) else first[1]
            stride = first[2] if len(first) > 2 else kernel
            if kernel == stride:
                return int(kernel)
            return int(stride)
        except Exception:
            continue
    return None


class EvaluationObservationImageWrapper(BaseParallelWrapper):
    """Wrap pursuit env to emit image observations and keep raw obs in info."""

    def __init__(
        self,
        env,
        num_agents: int,
        *,
        cell_scale: int = 24,
        normalize: bool = True,
        draw_counts: bool = True,
    ):
        super().__init__(env)
        self._num_agents = num_agents
        self.normalize = normalize
        self.renderer = train_mod.ObservationImageRenderer(cell_scale, draw_counts)
        self._agent_ids = [f"pursuer_{i}" for i in range(num_agents)]

        sample_obs, _ = self.env.reset()
        sample_agent = next(iter(sample_obs))
        sample_img = self._obs_to_image(sample_obs[sample_agent])
        img_shape = sample_img.shape

        low, high = (0.0, 1.0) if normalize else (0, 255)
        dtype = np.float32 if normalize else np.uint8
        from gymnasium import spaces

        self._obs_space = spaces.Box(low=low, high=high, shape=img_shape, dtype=dtype)

        self._zero_image = np.zeros(img_shape, dtype=dtype)
        self._last_images: Dict[str, np.ndarray] = {
            agent_id: self._zero_image.copy() for agent_id in self._agent_ids
        }
        self._state_shape = (self._num_agents,) + img_shape

    def observation_space(self, agent):
        return self._obs_space

    def _obs_to_image(self, local_obs: np.ndarray) -> np.ndarray:
        img = self.renderer.render(local_obs)
        if self.normalize:
            return (img.astype(np.float32) / 255.0).copy()
        return img.copy()

    def _stack_state(self) -> np.ndarray:
        frames = [
            self._last_images.get(agent_id, self._zero_image)
            for agent_id in self._agent_ids
        ]
        return np.stack(frames, axis=0)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        obs, info = self.env.reset(seed=seed, options=options)
        obs_imgs: Dict[str, np.ndarray] = {}
        for agent, agent_obs in obs.items():
            img = self._obs_to_image(agent_obs)
            obs_imgs[agent] = img
            self._last_images[agent] = img

        global_state = self._stack_state()
        info_dict: Dict[str, dict] = {}
        for agent in obs_imgs:
            base = info.get(agent, {}) if info is not None else {}
            info_dict[agent] = dict(base)
            info_dict[agent]["state"] = global_state
            info_dict[agent]["raw_obs"] = obs[agent]
        return obs_imgs, info_dict

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        obs_imgs: Dict[str, np.ndarray] = {}
        for agent, agent_obs in obs.items():
            img = self._obs_to_image(agent_obs)
            obs_imgs[agent] = img
            self._last_images[agent] = img

        global_state = self._stack_state()
        info_dict: Dict[str, dict] = {}
        for agent in obs_imgs:
            base = infos.get(agent, {}) if infos is not None else {}
            info_dict[agent] = dict(base)
            info_dict[agent]["state"] = global_state
            info_dict[agent]["raw_obs"] = obs[agent]
        return obs_imgs, rewards, terminations, truncations, info_dict


def make_image_env(
    num_agents: int,
    num_evaders: int,
    x_size: int,
    y_size: int,
    n_catch: int,
    max_cycles: int,
    freeze_evaders: bool,
    *,
    cell_scale: int = 24,
    normalize: bool = True,
    draw_counts: bool = True,
    render: bool = False,
) -> ParallelPettingZooEnv:
    env = pursuit_v4.parallel_env(
        n_pursuers=num_agents,
        n_evaders=num_evaders,
        freeze_evaders=freeze_evaders,
        x_size=x_size,
        y_size=y_size,
        n_catch=n_catch,
        surround=False,
        max_cycles=max_cycles,
        shared_reward=False,
        render_mode="human" if render else None,
    )
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = EvaluationObservationImageWrapper(
        env,
        num_agents=num_agents,
        cell_scale=cell_scale,
        normalize=normalize,
        draw_counts=draw_counts,
    )
    return ParallelPettingZooEnv(env)
