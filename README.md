# X30 SuperMimic Locomotion

Simulation project adapting a SuperMimic-style differentiable world-model training loop to a DeepRobotics X30 quadruped model in mjlab/MuJoCo.

> **Status:** flat-ground velocity tracking and diagonal-trot policies have been evaluated in simulation. No learned policy has been deployed to the physical X30. The model uses nominal assumptions and is not a calibrated digital twin.

## What this project contains

- X30 12-joint model integration and ordered actuator configuration for FL, FR, HL, HR legs.
- Flat-ground velocity tracking and diagonal-trot tasks.
- Replay-buffer data collection, differentiable world-model fitting, short imagined rollouts, BPTT actor updates, and TD(lambda) critic learning.
- Evaluation for planar velocity error, turning error, foot-contact phase, body motion, falls, and short-horizon world-model prediction.
- A parameter template. Hardware-specific actuator values are intentionally not included; fill the required fields for your own model before training.

This is an independent research implementation, not official SuperMimic source code. The project borrows the method described in the [SuperMimic paper](https://arxiv.org/abs/2610.28480).
Experimental curriculum and handstand modules are also present in the source tree, but they are outside the results reported here.

## Training loop

1. Run the current policy in MuJoCo and store transitions in a replay buffer.
2. Fit the differentiable world model on sampled transition windows.
3. Start imagined rollouts from replay states and optimize the actor through the frozen learned model.
4. Train the critic with TD(lambda) targets to estimate returns beyond the short imagination horizon.

For X30-Trot-Flat, the actor receives five frames of proprioception, command and gait phase history. Its 12 outputs are normalized joint-position offsets, mapped to targets as:

    q_target = q_default + action_scale * action

The actor does not output motor torque. The simulation applies the configured joint position controller and effort limits.

## Simulation results

The 15,000-iteration X30-Trot-Flat checkpoint was evaluated on three seeds in flat-ground simulation:

| Forward command | Planar velocity RMSE across seeds | Falls |
|---:|---:|---:|
| 0.3 m/s | 0.0379–0.0393 m/s | 0 / 3 |
| 0.5 m/s | 0.0208–0.0211 m/s | 0 / 3 |

A separate command-transition evaluation covered start, stop, restart, speed changes, and left/right turning. All 27 tested segments met the project’s initial simulation tolerances across three seeds. These results are limited to the tested nominal simulation setup; they are not real-robot results or a guarantee for other commands and conditions.

See [simulation results](docs/SIMULATION_RESULTS.md) for test conditions, additional metrics, and limitations.

### 50,000-iteration omnidirectional curriculum checkpoint

A separate `X30-Omnidirectional` checkpoint at iteration 50,000 was evaluated in flat-ground simulation on one seed (101009), with 22 command and transition cases of 24 seconds each. The evaluation report marked 21/22 cases as passed, reported no falls, and passed the world-model and retention checks. Forward, backward, and lateral mastered/probe groups passed; the yaw mastered group passed, while the yaw probe group did not. The ordinary left-turn and right-turn cases both passed with zero falls; the left-turn case had a velocity RMSE of 0.0315 m/s and a yaw-rate RMSE of 0.0190 rad/s.

The failed case was a left U-turn: the simulated turn completed, but the final heading error was 0.108 rad, above the evaluator's 0.08 rad threshold. This result is from one seed and should be read alongside the failure, not as a general success claim. The checkpoint's curriculum caps were forward 2.4 m/s, backward 0.6 m/s, lateral 1.0 m/s, and yaw 0.2 rad/s for mastered commands; probe caps were 3.0 m/s, 0.6 m/s, 1.0 m/s, and 0.4 rad/s, respectively. These are simulation evaluation settings, not X30 hardware safety limits.

See [50,000-iteration evaluation details](docs/OMNIDIRECTIONAL_50000_RESULTS.md). The checkpoint and raw training/evaluation logs are not included.

## Setup

The project targets Linux or WSL2 with a compatible Python, PyTorch, mjlab, MuJoCo and GPU/Warp installation.

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev]"

mjlab and MuJoCo Warp can require platform-specific system and CUDA dependencies. Follow their installation instructions for the machine being used.

## Configure and run

Copy the template and fill every required null field using a documented source appropriate for the robot model:

    cp configs/x30/robot.template.json configs/x30/robot.json

The template is not a ready-to-train hardware configuration. Do not interpret example or nominal simulation values as validated X30 motor settings. The private working configuration, raw robot captures, training logs and checkpoints are not included.

Inspect the compiled model:

    python scripts/x30/inspect_model.py --backend mjlab --robot-config configs/x30/robot.json --output-dir logs/x30_indices

Train the diagonal-trot task:

    python scripts/x30/train.py --task X30-Trot-Flat --robot-config configs/x30/robot.json --num-envs 64 --iterations 15000 --warmup-iterations 500 --wm-window 16 --wm-batch 128 --imag-batch 128 --horizon 16 --compile false --log-root logs/x30_trot --run-name x30_trot

Evaluate a checkpoint after training:

    python scripts/x30/eval_trot.py --checkpoint logs/x30_trot/RUN_NAME/model_15000.pt --command 0.3 0 0 --num-envs 1 --seconds 22 --seed 42 --wm-check

Exact evaluation numbers above came from the private working configuration and checkpoint, which are not distributed. Reproducing them requires a compatible filled configuration and a new training run.

## Scope and limitations

- Flat terrain only for the reported policy results.
- Domain randomization and hardware calibration are not represented by these results.
- No stair traversal, ball avoidance, external-push robustness, or physical-robot deployment is claimed.
- The public parameter template intentionally omits captured joint pose and machine-specific settings.
- Reported command ranges and acceptance tolerances are experiment settings, not X30 safety limits.


## Third-party assets and references

- X30 geometry is based on the low-resolution model published by DeepRoboticsLab. Its BSD 3-Clause license is preserved beside the model files; see [third-party notices](THIRD_PARTY_NOTICES.md).
- Go2 model assets retain their own upstream license files.
- SuperMimic: [*Scaling Humanoid Whole-Body Control with One Differentiable World Model*](https://arxiv.org/abs/2610.28480). This repository is an independent adaptation/reproduction effort, not the paper authors’ official implementation.
