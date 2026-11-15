"""Train the independent X30 front-foot handstand and recovery skill."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from go2_supermimic.robots.x30.handstand_train import main

if __name__ == '__main__':
    main()
