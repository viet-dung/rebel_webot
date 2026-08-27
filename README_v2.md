# NeRF-Based Object Representation and Reinforcement Learning for Robotic Manipulation

> Master's thesis, Chair of Knowledge-Based Systems, RWTH Aachen University
> Advisor: Tarik Viehmann, M.Sc. — Examiners: Prof. Gerhard Lakemeyer, Ph.D., Prof. Sebastian Trimpe, Ph.D.

A complete **sim-to-real framework** for teaching a robotic arm a manipulation task with
minimal real-world data. The system combines a parallelized Webots + ROS 2 digital twin for
fast, safe policy training with a NeRF-based (BundleSDF) 6D object pose estimation pipeline
(FoundationPose + Azure Kinect) for real-world perception, driving a Soft Actor-Critic (SAC)
agent on a physical igus ReBeL arm.

**Full thesis PDF:** [add link or file]
**Demo video:** [add embedded video link — see note in Setup]

## Results

| Stage | Success rate |
|---|---|
| Simulation (randomized environment) | 80% |
| Real robot (initial transfer) | lower — overshoot/oscillation, see below |
| Real robot (after tuning) | **90%** |

Two failure modes were diagnosed and fixed during this work:

- **Training instability:** an unconstrained, high-magnitude reward signal caused the SAC
  critic network to diverge. Fixed with reward clipping, which stabilized training and
  produced the 80%-success simulation policy.
- **Sim-to-real gap:** transferring the policy directly caused the arm to overshoot the
  target and oscillate. This turned out to be a system response-time mismatch between sim
  and real hardware, not a policy generalization failure — fixing the underlying control
  timing brought performance up to 90% on the physical robot.

## System Architecture

```
                     ┌─────────────────────────┐
                     │     Digital Twin (Sim)   │
                     │  Webots + ROS 2, parallel│
                     │  data collection         │
                     └───────────┬─────────────┘
                                 │
                     ┌───────────▼─────────────┐
                     │   SAC Agent (+ HER)      │
                     │   reward-clipped, reach  │
                     │   task, 80% sim success  │
                     └───────────┬─────────────┘
                                 │ policy transfer
                     ┌───────────▼─────────────┐
        Azure Kinect │   Real-World Environment │  igus ReBeL
        RGB-D  ─────▶│  ROS 2 hardware interface│◀── arm
                     │  velocity control        │
                     └───────────┬─────────────┘
                                 │
                     ┌───────────▼─────────────┐
                     │  Perception Pipeline     │
                     │  BundleSDF (NeRF) model  │
                     │  → FoundationPose 6D pose│
                     │  → Kalman filter tracking│
                     └──────────────────────────┘
```

## Perception Pipeline (external)

The 6D pose estimation pipeline builds on two open-source projects, used as-is and not
vendored in this repo:

- [BundleSDF](https://github.com/NVlabs/BundleSDF) — builds a NeRF-style 3D object model from a short video
- [FoundationPose](https://github.com/NVlabs/FoundationPose) — 6D pose estimation from the model + Azure Kinect RGB-D input

This repo contains the RL agent, the digital twin, and the real-robot integration that
consumes their output (pose + tracking, stabilized with a Kalman filter) as part of the
observation pipeline. See the thesis PDF (Section 4.4, 5.2–5.6) for the full perception
methodology and evaluation.

## Repository Structure

```
├── agent/                       # SAC agent, custom HER replay buffer, reward shaping
│   ├── agent.py
│   ├── custom_her.py
│   └── custom_her_replay_buffer.py
├── sim_env/                     # Webots digital twin + ROS 2 sync
│   ├── simulationEnv.py
│   ├── simulationEnvPush.py
│   └── box_spawner.py
├── real_env/                    # Real robot hardware interface
│   ├── realEnv.py
│   └── real_arm_controller.py
├── eval/                        # Evaluation scripts (sim + real)
│   ├── eval_sim.py
│   └── eval_real.py
├── best_model/                  # Best trained SAC checkpoint
├── plots/                       # Training curves, evaluation figures
├── requirements.txt             # Agent-side Python dependencies
└── README.md
```

## Setup

This project requires **two separate environments**, because ROS 2 and the RL agent
stack (Stable-Baselines3 / PyTorch) have conflicting dependencies.

```bash
git clone https://github.com/viet-dung/rebel_webot.git
cd rebel_webot
```

**1. Agent environment** (training, SAC/HER, Webots simulation)
```bash
conda create -n sb3-env python=3.10
conda activate sb3-env
pip install -r requirements.txt
```
Used for: `agent/`, `sim_env/`, `eval/eval_sim.py`.
Requires [Webots](https://cyberbotics.com/) for the simulation.

**2. ROS 2 environment** (real robot control + perception)
```bash
# source /opt/ros/<distro>/setup.bash
# build and source your ROS2 workspace containing the igus ReBeL driver
# clone and set up BundleSDF and FoundationPose separately per their own instructions
```
Used for: `real_env/`, `eval/eval_real.py`, and the external BundleSDF/FoundationPose
pipelines — anything talking to the physical arm, camera, or running pose estimation.

> Keep the two environments isolated (separate conda envs/virtualenvs) — training and
> real-hardware control are not run together.

## Demo

The video below shows the full pipeline running simultaneously: the Webots digital twin
(left), the live camera feed of the physical arm (top right), and the ROS 2/RViz state
visualization (bottom right).

To embed a playable video in this README on GitHub: open a new Issue on this repo, drag
`NERF_RL.mp4` into the comment box (don't submit it), copy the resulting
`https://github.com/user-attachments/assets/...` URL, and paste it here in place of this note.

## Method Summary

- **Perception:** a per-object NeRF-style 3D model is built from a short video using
  BundleSDF (external), then used by FoundationPose (external) for 6D pose estimation
  from Azure Kinect RGB-D input, tracked frame-to-frame and stabilized with a Kalman
  filter (this repo's integration).
- **Simulation:** a parallelized Webots digital twin, synchronized with ROS 2, collects
  training data far faster and more safely than the real robot could alone.
- **Learning:** a Soft Actor-Critic agent with a custom Hindsight Experience Replay buffer
  learns a table-top reaching task; reward clipping was required to stabilize critic
  training under a high-magnitude reward signal.
- **Transfer:** the trained policy is deployed on the real igus ReBeL arm via a custom
  velocity-control hardware interface; the initial sim-to-real gap was traced to a system
  response-time mismatch (not policy generalization) and resolved through targeted control
  tuning.

Full technical detail, related work, and evaluation methodology are in the thesis PDF linked above.

## Citation / Contact

Viet Dung Nguyen — [email or LinkedIn]
Master's Thesis, Chair of Knowledge-Based Systems, RWTH Aachen University, 2025
