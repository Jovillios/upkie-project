import argparse
import logging
from typing import Tuple
import time
import gin
import gymnasium as gym
import numpy as np
import upkie.envs
from upkie.utils.raspi import configure_agent_process, on_raspi
from upkie.utils.robot_state import RobotState
from upkie.utils.robot_state_randomization import RobotStateRandomization
import os

from envs import make_ppo_balancer_env
from stable_baselines3 import PPO


# %%


# %%


upkie.envs.register()

# %%


def get_sagittal_force(intensity: float, duration=None):

    mass = 5.34  # mass in [kg]
    force = intensity*mass*9.81  # Force in [N]
    if duration is not None and duration > 1e9:
        force = 0.0
    return force

# %%


def run_policy(env: gym.Wrapper, policy) -> None:
    """!
    Run the policy on a given environment.

    @param env Upkie environment, wrapped by the agent.
    @param policy MLP policy to follow.
    """

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

    # Values of intensity sagitall force that is apply
    intensity_force = np.arange(0.0, 0.1, 0.01)
    isFall = False

    magnitude = get_sagittal_force(intensity_force[0])
    previous_magnitude = magnitude

    torso_force_in_world[0] = magnitude

    best_magnitude = 0.0

    index_intensity = 0
    start = time.time_ns()

    action = np.zeros(env.action_space.shape)
    observation, info = env.reset()
    reward = 0.0
    while isFall == False and index_intensity < intensity_force.shape[0]:

        action, _ = policy.predict(observation, deterministic=True)
        now = time.time_ns()
        duration = now-start
        magnitude = get_sagittal_force(
            intensity_force[index_intensity], duration=duration)

        torso_force_in_world[0] = magnitude
        env.bullet_extra(bullet_action)  # call before env.step

        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated or duration > 3e9:
            if terminated or truncated:
                print('est tombé')
                isFall = True
                best_magnitude = get_sagittal_force(
                    intensity_force[index_intensity-1])
            start = time.time_ns()
            if index_intensity < intensity_force.shape[0]:
                index_intensity += 1
            torso_force_in_world[0] = get_sagittal_force(
                intensity_force[index_intensity])
            print(f'Monitoring : intensity:{intensity_force[index_intensity]}')
            observation, _ = env.reset()
    return best_magnitude

# %%


def main(policy_path: str, training: bool) -> None:
    """!
    Load environment and policy, and run the latter on the former.

    @param policy_path Path to policy parameters.
    @param training If True, add training noise and domain randomization.
    """
    env_settings = EnvSettings()
    init_state = None
    if training:
        training_settings = TrainingSettings()
        init_state = RobotState(
            randomization=RobotStateRandomization(
                **training_settings.init_rand
            ),
        )
    with gym.make(
        env_settings.env_id,
        frequency=env_settings.agent_frequency,
        init_state=init_state,
        max_ground_velocity=env_settings.max_ground_velocity,
        regulate_frequency=True,
        spine_config=env_settings.spine_config,
    ) as velocity_env:
        env = make_ppo_balancer_env(
            velocity_env,
            env_settings,
            training=training,
        )
        ppo_settings = PPOSettings()
        policy = PPO(
            "MlpPolicy",
            env,
            policy_kwargs={
                "net_arch": {
                    "pi": ppo_settings.net_arch_pi,
                    "vf": ppo_settings.net_arch_vf,
                },
            },
            verbose=0,
        )
        policy.set_parameters(policy_path)
        run_policy(env, policy)


# %%
if on_raspi():
    configure_agent_process()

policy_path = f"policy/params.zip"
# Configuration
config_path = f"policy/operative_config.gin"
logging.info("Loading policy configuration from %s", config_path)
gin.parse_config_file(config_path)

try:
    main(policy_path, training=False)
except KeyboardInterrupt:
    logging.info("Caught a keyboard interrupt")
