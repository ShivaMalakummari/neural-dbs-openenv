# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Neural Dbs Env Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from uuid import uuid4
import numpy as np

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import NeuralDbsAction, NeuralDbsObservation
except ImportError:
    from models import NeuralDbsAction, NeuralDbsObservation


class NeuralDbsEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

        # Neural system state
        self.beta_power = 0.8
        self.phase = np.random.rand()
        self.energy_used = 0.0
        self.time_step = 0

        # Task parameters (for difficulty control)
        self.drift = 0.05
        self.noise_level = 0.02

    def reset(self, task_config=None) -> NeuralDbsObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        # Apply task settings if provided
        if task_config:
            self.drift = task_config.get("drift", 0.05)
            self.noise_level = task_config.get("noise", 0.02)

        # Reset brain state
        self.beta_power = 0.8
        self.phase = np.random.rand()
        self.energy_used = 0.0
        self.time_step = 0

        return NeuralDbsObservation(
            beta_power=self.beta_power,
            phase=self.phase,
            energy_used=self.energy_used,
            time_step=self.time_step,
            done=False,
            reward=0.0,
        )

    def step(self, action: NeuralDbsAction) -> NeuralDbsObservation:
        self._state.step_count += 1
        self.time_step += 1

        # Clamp action values (safety)
        amp = max(0.0, min(1.0, action.amplitude))
        freq = max(0.0, min(1.0, action.frequency))
        pw = max(0.0, min(1.0, action.pulse_width))

        # Energy consumption
        energy = amp * freq * pw
        self.energy_used += energy

        # Brain dynamics
        drift = self.drift
        stim_effect = amp * (0.6 + 0.4 * freq)
        noise = np.random.normal(0, self.noise_level)

        self.beta_power = self.beta_power + drift - stim_effect + noise
        self.beta_power = max(0.0, min(1.0, self.beta_power))

        # Phase update
        self.phase = (self.phase + freq + np.random.normal(0, 0.01)) % 1.0

        # Reward calculation
        beta_reduction = 1.0 - self.beta_power

        safety_penalty = 0.0
        if amp > 0.9:
            safety_penalty += 0.5

        reward = beta_reduction - 0.3 * energy - safety_penalty

        # Episode termination
        done = self.time_step >= 50

        return NeuralDbsObservation(
            beta_power=self.beta_power,
            phase=self.phase,
            energy_used=self.energy_used,
            time_step=self.time_step,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state