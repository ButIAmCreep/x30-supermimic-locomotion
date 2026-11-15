"""Train the independent X30 2 m/s lateral and 2 rad/s yaw course."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from go2_supermimic.robots.x30.speed_train import main

if __name__ == '__main__':
    main()
