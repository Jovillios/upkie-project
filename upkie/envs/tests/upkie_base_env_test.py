#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria

"""Test UpkieBaseEnv."""

import unittest
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from gymnasium import spaces

from upkie.envs import UpkieBaseEnv
from upkie.envs.tests.mock_spine import MockSpine


class UpkieTestEnv(UpkieBaseEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = spaces.Box(
            -1.0,
            +1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=(1,),
            dtype=np.float32,
        )

    def parse_first_observation(self, spine_observation: dict) -> None:
        pass

    def get_env_observation(self, spine_observation: dict) -> np.ndarray:
        return np.full((1,), 0.5, dtype=self.observation_space.dtype)

    def get_spine_action(self, action: np.ndarray) -> dict:
        return {"test": action}

    def get_reward(self, observation: np.ndarray, action: np.ndarray) -> float:
        return 1.0


class TestUpkieBaseEnv(unittest.TestCase):
    def setUp(self):
        shared_memory = SharedMemory(name=None, size=42, create=True)
        self.env = UpkieTestEnv(
            fall_pitch=1.0,
            frequency=100.0,
            shm_name=shared_memory._name,
            spine_config=None,
        )
        shared_memory.close()
        self.env._spine = MockSpine()

    def test_reset(self):
        _, info = self.env.reset()
        spine_observation = info["spine_observation"]
        self.assertGreaterEqual(spine_observation["number"], 1)

    def test_spine_config(self):
        """Check that runtime and default configs are merged properly."""
        shared_memory = SharedMemory(name=None, size=42, create=True)
        env = UpkieTestEnv(
            fall_pitch=1.0,
            frequency=100.0,
            shm_name=shared_memory._name,
            spine_config={"some_value": 12, "bullet": {"gui": False}},
        )
        shared_memory.close()
        self.assertEqual(env._spine_config["some_value"], 12)
        self.assertEqual(env._spine_config["some_value"], 12)
        self.assertEqual(env._spine_config["bullet"]["gui"], False)

    def test_check_env(self):
        try:
            from stable_baselines3.common.env_checker import check_env

            check_env(self.env)
        except ImportError:
            pass

    def test_fall_detected(self):
        spine_observation = {
            "base_orientation": {
                "pitch": 2.0 * self.env.fall_pitch,
            }
        }
        self.assertTrue(self.env.detect_fall(spine_observation))


if __name__ == "__main__":
    unittest.main()