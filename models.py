# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Neural Dbs Env Environment.

The neural_dbs_env environment is a simple test environment that echoes back messages.
"""
from pydantic import BaseModel


class NeuralDbsObservation(BaseModel):
    beta_power: float
    phase: float
    energy_used: float
    time_step: int
    done: bool
    reward: float


class NeuralDbsAction(BaseModel):
    amplitude: float
    frequency: float
    pulse_width: float