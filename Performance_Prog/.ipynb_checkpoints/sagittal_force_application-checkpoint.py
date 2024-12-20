#### !/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Inria

"""Lift the simulated robot while it balances in place."""

import gymnasium as gym
import numpy as np
import time

import upkie.envs

upkie.envs.register()

#Vérifier si le calcul d'intensité est bon
def get_sagittal_force(intensity: float, duration=None):

    mass = 5.34 #mass in [kg]
    force  = intensity*mass*9.81 #Force in [N]
    if duration is not None and duration > 1e9:
        force = 0.0
    return force

def run(env: upkie.envs.UpkieGroundVelocity):
    torso_force_in_world = np.zeros(3)
    bullet_action = {
        "external_forces": {
            "torso": {
                "force": torso_force_in_world,
                "local": False,
            }
        }
    }
    observation, _ = env.reset()
    gain = np.array([10.0, 1.0, 0.0, 0.1])

    #Values of intensity sagitall force that is apply 
    intensity_force = np.arange(0.0,0.1,0.01)
    isFall = False

    magnitude = get_sagittal_force(intensity_force[0])
    previous_magnitude = magnitude
    
    torso_force_in_world[0] = magnitude
    
    best_magnitude = 0.0
    
    index_intensity = 0
    start = time.time_ns()
    
    while isFall==False and index_intensity < intensity_force.shape[0]:
        
        action = gain.dot(observation).reshape((1,))
        
        now = time.time_ns()
        duration = now-start
        magnitude = get_sagittal_force(intensity_force[index_intensity],duration = duration)
        
        torso_force_in_world[0] = magnitude
        env.bullet_extra(bullet_action)  # call before env.step
        observation, _, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated or duration>3e9:
            if terminated or truncated:
                print('est tombé')
                isFall = True
                best_magnitude = get_sagittal_force(intensity_force[index_intensity-1])
            start = time.time_ns()
            if index_intensity < intensity_force.shape[0]:
                index_intensity += 1
            torso_force_in_world[0] = get_sagittal_force(intensity_force[index_intensity])
            print(f'Monitoring : intensity:{intensity_force[index_intensity]}')
            observation, _ = env.reset()
    return best_magnitude


if __name__ == "__main__":
    with gym.make("UpkieGroundVelocity-v3", frequency=200.0) as env:
        best_magnitude2 = run(env)
        print (best_magnitude2)
