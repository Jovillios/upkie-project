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

def run(env: upkie.envs.UpkieGroundVelocity, gain):
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

    #Values of intensity sagitall force that is apply 
    intensity_force = 0.0
    isFall = False
    
    torso_force_in_world[0] = get_sagittal_force(intensity_force)
    
    best_magnitude = 0.0

    start = time.time_ns()
    
    while isFall==False:
        
        action = gain.dot(observation).reshape((1,))
        
        now = time.time_ns()
        duration = now-start
        torso_force_in_world[0] = get_sagittal_force(intensity_force,duration = duration)
        
        env.bullet_extra(bullet_action)  # call before env.step
        observation, _, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated or duration>3e9:
            if terminated or truncated:
                print('est tombé')
                isFall = True
                best_magnitude = get_sagittal_force(intensity_force)
            start = time.time_ns()
            intensity_force += 0.01
            torso_force_in_world[0] = get_sagittal_force(intensity_force)
            observation, _ = env.reset()
    return best_magnitude


if __name__ == "__main__":
    G1 = [5,10,15,20]
    G2 = [5,10,15,20]
    G3 = [0.01,0.05,0.1,0.2]
    G4 = [0.01,0.05,0.1,0.2]
    
    best_magnitude = []
    gain = []
    
    with gym.make("UpkieGroundVelocity-v3", frequency=200.0) as env:
        
        for g1 in G1:
            for g2 in G2:
                for g3 in G3:
                    for g4 in G4:
                        gain.append([g1, g2, g3, g4])
                        best_magnitude.append(run(env,np.array([g1, g2, g3, g4])))
        
        index_best = np.argmax(best_magnitude)
        
        print(f"Best MSFOS:{best_magnitude[index_best]} N for gain = {gain[index_best]} ")
