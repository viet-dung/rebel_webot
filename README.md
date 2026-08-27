# REBEL Webot — Reinforcement Learning for Robotic Arm Manipulation

> Master's thesis project: training a robotic arm (simulated in Webots, deployed on real hardware)
> to [pick/push/stack — fill in the actual task] using Soft Actor-Critic (SAC) with
> Hindsight Experience Replay (HER).

## Overview

[2-3 sentences: what problem does this solve, what's the core approach,
what's the headline result? e.g. "Sparse-reward robotic manipulation tasks
are hard to learn directly. This project combines SAC with a custom HER replay
buffer to train a robotic arm to stack boxes, first in Webots simulation, then
transferred to a real arm."]

**Full thesis:** [link to PDF, or add thesis.pdf to this repo]

## Key Results

[Insert 1-2 of your best plots/GIFs from `plots/` here, e.g.]

![Training curve](plots/your_best_plot.png)

- [Headline metric, e.g. "Success rate improved from X% to Y% after Z training steps"]
- [Sim-to-real transfer result if applicable]

## Repository Structure

```
├── agent.py                  # SAC agent definition
├── custom_her.py              # Custom Hindsight Experience Replay logic
├── custom_her_replay_buffer.py
├── baseEnv.py                 # Base Gym environment
├── simulationEnv.py           # Webots simulation environment
├── simulationEnvPush.py       # Variant for pushing task
├── realEnv.py                  # Real-world environment wrapper
├── real_arm_controller.py     # Real robotic arm controller interface
├── robotic_arm.py              # Robotic arm kinematics/control
├── box_spawner.py              # Spawns boxes in simulation
├── eval_sim.py                 # Evaluate trained policy in simulation
├── eval_real.py                 # Evaluate trained policy on real hardware
├── generate_latex.py           # Generates thesis figures/tables
├── td_plot.py                   # Training/TD-error plotting
├── utils.py
├── best_model/                  # Best trained model checkpoint
├── plots/                       # Result figures
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/viet-dung/rebel_webot.git
cd rebel_webot
pip install -r requirements.txt
```

You'll also need [Webots](https://cyberbotics.com/) installed to run the simulation environment.

## Usage

**Train in simulation:**
```bash
python simulationEnv.py   # adjust as needed — fill in actual entry point/args
```

**Evaluate a trained policy:**
```bash
python eval_sim.py --model best_model/<checkpoint>
```

**Deploy / evaluate on real hardware:**
```bash
python eval_real.py
```

## Method

[Short technical summary: state/action space, reward shaping, SAC + HER
configuration, sim-to-real approach if relevant. Link to relevant thesis
section for full detail.]

## Citation / Contact

[Your name] — [email or LinkedIn]
Master's thesis, [University], [Year]
