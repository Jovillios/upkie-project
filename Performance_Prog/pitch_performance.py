#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:04:51 2024

@author: kevin
"""

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
    pitch = np.array(observation[0])
    gain = np.array([gain])

    #Values of intensity sagitall force that is apply 
    intensity_force = 0.0
    isFall = False
    
    torso_force_in_world[0] = get_sagittal_force(intensity_force)
    
    best_magnitude = 0.0

    start = time.time_ns()
    
    while isFall==False:
        
        action = gain.dot(pitch).reshape((1,))
        
        now = time.time_ns()
        duration = now-start
        torso_force_in_world[0] = get_sagittal_force(intensity_force,duration = duration)
        
        env.bullet_extra(bullet_action)  # call before env.step
        observation, _, terminated, truncated, _ = env.step(action)
        pitch = np.array(observation[0])
        
        if terminated or truncated or duration>3e9:
            if terminated or truncated:
                print('est tombé')
                isFall = True
                best_magnitude = get_sagittal_force(intensity_force)
            start = time.time_ns()
            intensity_force += 0.01
            torso_force_in_world[0] = get_sagittal_force(intensity_force)
            observation, _ = env.reset()
            pitch = np.array(observation[0])
    return best_magnitude


gain_list = [5.0,10.0,15.0,20.0,25.0]
best_magnitude = []
if __name__ == "__main__":
    with gym.make("UpkieGroundVelocity-v3", frequency=200.0) as env:
        for i in range(len(gain_list)):
            best_magnitude.append(run(env,gain_list[i]))
                                  
        for i in range(len(gain_list)):
            bm= best_magnitude[i]
            print(f'Gain : {gain_list[i]} => MFOS = {bm}')
            
