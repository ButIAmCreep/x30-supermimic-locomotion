# Tests in the public copy

The retained tests focus on simulation code contracts, robot model parsing, task rewards, observation shapes, training state handling and evaluation metrics.

Tests that depend on the private filled X30 configuration, raw robot packet captures, or live/SDK-oriented helpers are not included. The trained checkpoint and local run logs are also not distributed.

After installing the project development dependencies, run from the repository root:

    python -m pytest tests

The tests have not been run as part of preparing this public portfolio copy.
