from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AsyncStreamingRunner:
    """Shared runtime-side scheduling helper for async streaming rollout."""

    runtime: object
    obs_stride_env_steps: int
    control_dt_ms: float
    obs_counter: int = 0
    formal_obs_count: int = 0
    last_formal_obs_index: Optional[int] = None
    last_formal_submit_env_step: Optional[int] = None

    def __post_init__(self) -> None:
        if int(self.obs_stride_env_steps) <= 0:
            raise ValueError("`obs_stride_env_steps` must be positive.")
        self.obs_stride_env_steps = int(self.obs_stride_env_steps)
        self.control_dt_ms = float(self.control_dt_ms)

    def start_formal_phase(self, *, obs_index_start: int = 0) -> None:
        self.obs_counter = int(obs_index_start)
        self.formal_obs_count = 0
        self.last_formal_obs_index = None
        self.last_formal_submit_env_step = None

    def prime_formal_observation(
        self,
        *,
        input_image: torch.Tensor,
        proprio: Optional[torch.Tensor],
        env_step: int,
    ) -> bool:
        t = int(env_step)
        if t % self.obs_stride_env_steps != 0:
            return False
        obs_index = int(self.obs_counter)
        self.runtime.submit_observation(
            input_image=input_image,
            proprio=proprio,
            env_step=t,
            obs_index=obs_index,
            obs_timestamp_ms=self._obs_timestamp_ms(obs_index),
            trigger_job=False,
        )
        self.runtime.wait_until_idle()
        self.last_formal_obs_index = int(obs_index)
        self.last_formal_submit_env_step = int(t)
        self.obs_counter += 1
        self.formal_obs_count += 1
        return True

    def _obs_timestamp_ms(self, obs_index: int) -> float:
        return float(obs_index) * self.control_dt_ms * float(self.obs_stride_env_steps)

    def maybe_submit_formal_observation(
        self,
        *,
        input_image: torch.Tensor,
        proprio: Optional[torch.Tensor],
        env_step: int,
    ) -> bool:
        t = int(env_step)
        if self.last_formal_submit_env_step is not None and t == int(self.last_formal_submit_env_step):
            return False
        if t % self.obs_stride_env_steps != 0:
            return False

        obs_index = int(self.obs_counter)
        self.runtime.submit_observation(
            input_image=input_image,
            proprio=proprio,
            env_step=t,
            obs_index=obs_index,
            obs_timestamp_ms=self._obs_timestamp_ms(obs_index),
            trigger_job=False,
        )
        self.last_formal_obs_index = int(obs_index)
        self.last_formal_submit_env_step = int(t)
        self.obs_counter += 1
        self.formal_obs_count += 1
        return True

    def wait_for_action(
        self,
        *,
        env_step: int,
        proprio: Optional[torch.Tensor],
    ):
        del proprio
        t = int(env_step)
        action = self.runtime.wait_for_action_available(t)
        if action is None:
            raise RuntimeError(f"Timed out waiting for action at env_step={t}.")
        return action
