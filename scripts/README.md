# Script entry points

Run commands from the repository root.

- scripts/train.py and scripts/eval.py: shared training and evaluation entry points.
- scripts/x30/train.py and scripts/x30/eval.py: X30 task entry points.
- scripts/x30/eval_trot.py: trot metrics, command sequences, and world-model prediction checks.
- scripts/x30/play.py: interactively view a policy in the MuJoCo simulation.
- scripts/x30/inspect_model.py: compile and inspect the X30 model.

Training commands must specify a task and a filled robot configuration. The public template contains unfilled hardware-dependent fields; see the root README before starting a run.

This public portfolio copy omits raw packet-capture analysis and live robot monitoring tools. No policy deployment to a physical robot is included or claimed.
