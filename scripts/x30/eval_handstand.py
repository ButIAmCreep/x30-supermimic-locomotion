"""Measure actual front-support hold, falls and walking recovery."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from go2_supermimic.robots.x30.handstand_eval import main

if __name__ == '__main__':
    main()
