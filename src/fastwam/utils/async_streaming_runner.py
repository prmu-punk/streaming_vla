from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import time


@dataclass
class AsyncStreamingRunner:
    """Shared runtime-side scheduling helper for async streaming rollout."""

    runtime: object
    obs_stride_env_steps: int
    control_dt_ms: float
    force_first_job: bool
    obs_counter: int = 0
    formal_obs_count: int = 0
    first_formal_triggered: bool = False
    last_formal_obs_index: Optional[int] = None
    last_formal_submit_env_step: Optional[int] = None

    def __post_init__(self) -> None:
        if int(self.obs_stride_env_steps) <= 0:
            raise ValueError("`obs_stride_env_steps` must be positive.")
        self.obs_stride_env_steps = int(self.obs_stride_env_steps)
        self.control_dt_ms = float(self.control_dt_ms)
        self.force_first_job = bool(self.force_first_job)

    def start_formal_phase(self, *, obs_index_start: int = 0) -> None:
        self.obs_counter = int(obs_index_start)
        self.formal_obs_count = 0
        self.first_formal_triggered = False
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

    def run_warmup(
        self,
        *,
        input_image: torch.Tensor,
        proprio: Optional[torch.Tensor],
        warmup_action_jobs: int,
        start_env_step: int = 0,
        start_obs_index: int = 0,
    ) -> int:
        target_jobs = int(warmup_action_jobs)
        if target_jobs <= 0:
            return int(start_obs_index)

        warmup_t = int(start_env_step)
        obs_index = int(start_obs_index)

        # Under the continuous action loop, "completed jobs" is no longer a stable
        # semantic boundary for warmup. We instead warm up over a fixed negative
        # env-step prefix and let the action worker denoise continuously in the
        # background while we feed observations and advance the runtime clock.
        while warmup_t < 0:
            if warmup_t % self.obs_stride_env_steps == 0:
                self.runtime.submit_observation(
                    input_image=input_image,
                    proprio=proprio,
                    env_step=warmup_t,
                    obs_index=obs_index,
                    obs_timestamp_ms=self._obs_timestamp_ms(obs_index),
                    trigger_job=False,
                )
                obs_index += 1
            self.runtime.get_action(warmup_t, count_miss=False)
            self.runtime.poll()
            warmup_t += 1
        self.runtime.wait_until_idle()
        return obs_index

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
        t = int(env_step)
        action = self.runtime.get_action(t)
        while action is None:
            self.runtime.poll()
            time.sleep(0.001)
            action = self.runtime.get_action(t, count_miss=False)
        return action
