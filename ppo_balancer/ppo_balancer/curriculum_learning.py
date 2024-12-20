#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria
"""
Created on Mon Dec  9 16:57:56 2024

@author: kevin
"""
from typing import Tuple

import gymnasium
import numpy as np
import time

import upkie.envs

upkie.envs.register()
from upkie.utils.spdlog import logging


def get_sagittal_force(intensity: float, duration=None):

    mass = 5.34 #mass in [kg]
    force  = intensity*mass*9.81 #Force in [N]
    if duration is not None and duration > 1e9:
        force = 0.0
    return force

class CurriculumLearning(gymnasium.Wrapper):
    """!
    Model that add a sagitall effort one second with a gradually during the train

    """

    # ## \var filtered_action
    # ## Wrapped action after low-pass filtering.
    # filtered_action: np.ndarray

    # ## \var time_constant
    # ## Cutoff period in seconds of a low-pass filter applied to the action.
    # time_constant: float

    # ## \var time_constant_box
    # ## Box space from which the time constant is sampled at every reset of the
    # ## environment.
    # time_constant_box: Box

    def __init__(self, env, start_apply_force_step, increase_intensity_step):
        r"""!
        Initialize wrapper.

        \param env Environment to wrap.
        \param start_apply_force_step Step after an sagittal force is apply
        \param increase_intensity_step Gap of steps between two increase of intensity
        """
        super().__init__(env)
        
        self.nb_step = 0
        self.nb_step_with_current_magnitude = 0
        
        self.intensity_step = 0.0
        self.current_magnitude = 0.0
        
        self.start_apply_force_step = start_apply_force_step
        self.increase_intensity_step = increase_intensity_step
        
        self.start_episode = 0
        
        self.torso_force_in_world = np.zeros(3)
        self.sens_application = 0
        self.bullet_action = {
        "external_forces": {
            "torso": {
                "force": self.torso_force_in_world,
                "local": False,
            }
        }
    }
        
    def reset(self, **kwargs):
        r"""!
        Reset the environment.

        \param kwargs Keyword arguments forwarded to the wrapped environment.
        """
        self.start_episode = time.time_ns()
        self.sens_application = np.random.choice([-1, 1])
        
        if self.nb_step > self.start_apply_force_step:
            if self.nb_step_with_current_magnitude > self.increase_intensity_step :
                self.intensity_step += 0.01
                self.current_magnitude = get_sagittal_force(self.intensity_step)
                self.nb_step_with_current_magnitude = 0
                logging.info("Current apply magnitude=%f", self.current_magnitude)
            
        return self.env.reset(**kwargs)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        r"""!
        Step the environment.

        \param action Action from the agent.
        \return Tuple with (observation, reward, terminated, truncated,info).
            See \ref upkie.envs.upkie_base_env.UpkieBaseEnv.step for details.
        """
        current_time = time.time_ns()
        duration = current_time - self.start_episode
        
        self.torso_force_in_world[0] = self.sens_application*get_sagittal_force(self.intensity_step, duration=duration)
        
        self.nb_step += 1
        self.nb_step_with_current_magnitude +=1
        
        self.env.bullet_extra(self.bullet_action)
        
        return self.env.step(action)
