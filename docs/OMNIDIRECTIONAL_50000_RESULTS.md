# X30 Omnidirectional Curriculum: 50,000-Iteration Evaluation

This note records the evaluation of the separate `X30-Omnidirectional` curriculum checkpoint at iteration 50,000. It complements the 15,000-iteration `X30-Trot-Flat` results in the main README; the two checkpoints and tasks are different experiments.

## Evaluation setup

- Environment: flat-ground MuJoCo simulation.
- Checkpoint: iteration 50,000.
- Seed: 101009.
- Duration: 24 seconds per command or transition case.
- Cases: 22.
- Evaluation result: 21 of 22 cases passed; every case reported zero falls.
- World-model checks at horizons 1, 8, and 16 and the retention checks passed.

The curriculum report marked the forward, backward, and lateral mastered/probe groups as passed. The yaw mastered group passed; the yaw probe group did not. The ordinary left-turn case (`vx=0.4 m/s`, `wz=0.15 rad/s`) and right-turn case (`vx=0.4 m/s`, `wz=-0.15 rad/s`) both passed with zero falls. For the ordinary left turn, velocity RMSE was 0.0315 m/s and yaw-rate RMSE was 0.0190 rad/s.

## Command ranges in this evaluation

| Curriculum range | Forward | Backward | Lateral | Yaw rate |
|---|---:|---:|---:|---:|
| Mastered cap | 2.4 m/s | 0.6 m/s | 1.0 m/s | 0.2 rad/s |
| Probe cap | 3.0 m/s | 0.6 m/s | 1.0 m/s | 0.4 rad/s |

The evaluation included both lateral and yaw directions, retention cases, stopping, command switches, curved motion, and U-turns. These command caps describe this simulation curriculum and are not validated hardware limits or deployment recommendations.

## Unpassed case

The left U-turn probe completed the simulated turn and had no fall, but its final heading error was 0.108 rad. The evaluator requires a U-turn heading error no greater than 0.08 rad, so this case failed. This is why the overall result is 21/22 and the yaw probe group remains unpassed.

## Scope

This is one checkpoint evaluation on one seed. It does not establish performance across random seeds, hardware configurations, terrain, or real-robot deployment. The checkpoint and raw run logs are not distributed in this public repository; the summary above is taken from the saved iteration-50,000 evaluation record.
