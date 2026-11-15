# Simulation Results

This page summarizes selected local evaluation reports for the 15,000-iteration X30-Trot-Flat checkpoint. The checkpoint and the exact machine-specific configuration are not distributed in this public portfolio copy.

## Fixed forward commands

Three seeds (1, 42, 2026) were evaluated for each command in a single MuJoCo environment. The first two seconds were excluded from motion and gait statistics; falls were still counted.

| Target vx | Actual mean speed across seeds | Planar velocity RMSE | Gait phase match | Falls |
|---:|---:|---:|---:|---:|
| 0.3 m/s | 0.2622–0.2635 m/s | 0.0379–0.0393 m/s | 100% | 0 / 3 |
| 0.5 m/s | 0.4844–0.4847 m/s | 0.0208–0.0211 m/s | 100% | 0 / 3 |

The 0.3 m/s case has a noticeable underspeed. The reported phase score excludes samples near gait-phase boundaries, so it should not be read as perfect instantaneous contact tracking.

## Command transitions

A separate 38-second sequence evaluated start, stop, restart, speed-up, slow-down, left turn, right turn, straight motion and final stop. Across three seeds, all 27 action segments met the project’s initial simulation tolerances. No falls or four-feet-airborne intervals were observed during the valid measurement windows.

The tolerances are project-level simulation checks, not certified robot safety limits. The policy was not tested on the physical robot.

## World-model prediction

On one held-out flat-ground forward trajectory (0.3 m/s, seed 42), velocity RMSE was:

| Open-loop horizon | World-model RMSE | Persistence baseline |
|---:|---:|---:|
| 1 step | 0.00350 m/s | 0.01265 m/s |
| 8 steps | 0.00623 m/s | 0.01887 m/s |
| 16 steps | 0.00884 m/s | 0.01901 m/s |

This is a narrow evaluation on one trajectory distribution. It does not establish generalization to different terrain, payloads, friction, hardware dynamics or commands.

## Limits

These numbers describe the local nominal simulation experiment only. The public copy excludes the trained checkpoint, raw robot PCAP files, private actuator configuration, TensorBoard logs and detailed traces. A new run with a user-supplied configuration may produce different results.
