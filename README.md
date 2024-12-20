# Upkie wheeled biped robot

<img src="https://github.com/upkie/upkie/assets/1189580/2fc5ee4a-81b0-425c-83df-558c7147cc59" align="right" width="200" />

[![CI](https://img.shields.io/github/actions/workflow/status/upkie/upkie/ci.yml?branch=main)](https://github.com/upkie/upkie/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/github/actions/workflow/status/upkie/upkie/docs.yml?branch=main&label=docs)](https://upkie.github.io/upkie/)
[![Coverage](https://coveralls.io/repos/github/upkie/upkie/badge.svg?branch=main)](https://coveralls.io/github/upkie/upkie?branch=main)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/upkie.svg)](https://anaconda.org/conda-forge/upkie)
[![PyPI version](https://img.shields.io/pypi/v/upkie)](https://pypi.org/project/upkie/)

**Upkie** is a fully open source self-balancing wheeled biped robot. It has wheels for balancing and legs to negotiate uneven terrains. Upkies are designed to be buildable at home using only tools and components ordered online, like mjbots actuators. You can develop in Python or C++, on Linux or macOS, then deploy your software to the robot's Raspberry Pi.

This repository contains all the materials needed to build and animate an Upkie:

- Hardware:
  - [Bill of materials](https://github.com/upkie/upkie/wiki/Bill-of-materials)
  - [Build instructions](https://github.com/upkie/upkie/wiki)
  - [Project log](https://hackaday.io/project/185729-upkie-wheeled-biped-robots)
- Software:
  - [Installation](https://github.com/upkie/upkie#installation)
  - [Getting started](https://github.com/upkie/upkie#getting-started)
  - [Documentation](https://upkie.github.io/upkie/)
- Going further:
  - [Examples](https://github.com/upkie/upkie/tree/main/examples)
  - [Gymnasium environments](https://upkie.github.io/upkie/environments.html)
  - [Agents](https://github.com/upkie/upkie#agents)

Questions are welcome in the [Chat](https://app.element.io/#/room/#upkie:matrix.org) and [Discussions forum](https://github.com/upkie/upkie/discussions).

## Installation

### From conda-forge

```console
conda install -c conda-forge upkie
```

### From PyPI

```console
pip install upkie
```

## Getting started

Let's start a Bullet simulation spine:

<img src="https://github.com/upkie/upkie/blob/main/docs/images/bullet-spine.png" height="100" align="right" />

```console
./start_simulation.sh
```

Click on the robot in the simulator window to apply external forces. Once the simulation spine is running, we can control the robot using one of its Gymnasium environments, for instance:

```python
import gymnasium as gym
import numpy as np
import upkie.envs

upkie.envs.register()

with gym.make("UpkieGroundVelocity-v3", frequency=200.0) as env:
    observation, _ = env.reset()
    gain = np.array([10.0, 1.0, 0.0, 0.1])
    for step in range(1_000_000):
        action = gain.dot(observation).reshape((1,))
        observation, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset()
```

The Python code is the same whether we run in simulation or on a real Upkie. Head over to the [examples](https://github.com/upkie/upkie/tree/main/examples) directory for more use cases.

## Gymnasium environments

Upkie has environments compatible with the [Gymnasium API](https://gymnasium.farama.org/), for instance:

- `UpkieGroundVelocity`: keep legs straight and balance with the wheels.
- `UpkieServos`: control joint servos directly (position, velocity, torque)
  - `UpkieServoPositions`: control joint positions.
  - `UpkieServoTorques`: control joint torques.

Check out the full [list of environments](https://upkie.github.io/upkie/environments.html) for details.

## Agents

Larger Upkie agents have their own repositories:

- [MPC balancer](https://github.com/upkie/mpc_balancer): balance in place using model predictive control.
- [Pink balancer](https://github.com/upkie/pink_balancer): a more advanced agent that can crouch and stand up while balancing.
- [PPO balancer](https://github.com/upkie/ppo_balancer): balance in place with a policy trained by reinforcement learning.
- [PID balancer](https://github.com/upkie/pid_balancer): legacy agent used to test new Upkies with minimal dependencies.

Head over to the [new_agent](https://github.com/upkie/new_agent) template to create your own, and feel free to open a PR here to add your agent to the list.

## How can I participate?

Contributions are welcome to both the hardware and software of Upkies! If you are a developer/maker with some robotics experience looking to hack on open source, check out the [contribution guidelines](CONTRIBUTING.md). On the software side, you can also report any bug you encounter in the [issue tracker](https://github.com/upkie/upkie/issues).

## Citation

If you built an Upkie or use parts of this project in your works, please cite the project and its contributors:

```bibtex
@software{upkie,
  title = {{Upkie open source wheeled biped robot}},
  author = {Caron, St\'{e}phane and Perrin-Gilbert, Nicolas and Ledoux, Viviane and G\"{o}kbakan, \"{Umit} Bora and Raverdy, Pierre-Guillaume and Raffin, Antonin and Tordjman--Levavasseur, Valentin},
  url = {https://github.com/upkie/upkie},
  license = {Apache-2.0},
  version = {6.0.0},
  year = {2024}
}
```

## See also

- [Awesome Open Source Robots](https://github.com/stephane-caron/awesome-open-source-robots): Upkies are one among many open-source open-hardware robot initiative: check out the others!
- [Open Dynamic Robot Initiative](https://open-dynamic-robot-initiative.github.io/): An open torque-controlled modular robot architecture for legged locomotion research.

. . .

# Project : RL to balance a wheeled biped robot

Upkie is a wheeled biped robot developed in the Willow team at Inria. It can balance itself upright and traverse various terrains. One metric for the robustness of a balancing policy is the maximum push that can be applied to the robot before it falls. In this topic, we focus on the maximum sagittal force over one second, which we will write MSFOS for short.

## Goal

Evaluate the balancing performance of a linear feedback controller, then train a more robust behavior by reinforcement learning of a neural-network policy.

## Project plan

1. Start up a simulation following the instructions from the upkie repository
2. Try out a first balancing policy using only pure linear feedback of the base pitch angle
3. Using the environment API, apply a sagittal force to the robot for 1 second. What is the resulting MSFOS (in N) that you can apply before the robot falls ?
4. Update your policy to linear feedback of the full observation vector. How much improvement in MSFOS can you achieve ?
5. From there on, let us switch to training a neural-network policy by proximal policy optimization (PPO). Clone the PPO balancer and run the pre-trained policy. How well does it perform in MSFOS ?
6. Run the training script to train a new policy, adjusting the number of processes to your computer. What is the best number of processes for your computer ?
7. Run your trained policy and check its MSFOS
8. Improve the policy using one of the techniques seen in class that is not implemented in the PPO balancer : either curriculum learning, or reward shaping, or teacher-student distillation.
9. Extension : the UpkieServos Gymnasium environment allows us to send position or velocity commands to each joint of the robot. Check out the new observations and actions from this environment : do they contain everything we need for balancing ? 10. Train a balancing policy for the UpkieServos environment . Does it improve MSFOS outofthebox?
10. Action shaping is a way to make the policy less end-to-end by adding some
    engineering to the environment. For instance, since the two limbs of the robot have roughly the same length, it is usually a good idea to set knee ankle = ±2 \* hip angle so that the action becomes [crouching, wheel velocity] for each leg. Add some action shaping to your environment. Can you improve the MSFOS of the resulting policies ?

## Advancement

- [x] Start up a simulation following the instructions from the upkie repository
- [x] Try out a first balancing policy using only pure linear feedback of the base pitch angle
- [ ] Using the environment API, apply a sagittal force to the robot for 1 second. What is the resulting MSFOS (in N) that you can apply before the robot falls ?
- [ ] Update your policy to linear feedback of the full observation vector. How much improvement in MSFOS can you achieve ?
- [ ] From there on, let us switch to training a neural-network policy by proximal policy optimization (PPO). Clone the PPO balancer and run the pre-trained policy. How well does it perform in MSFOS ?
