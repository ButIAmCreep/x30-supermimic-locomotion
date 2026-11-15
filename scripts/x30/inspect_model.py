"""Inspect X30 model and actuator indices."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from go2_supermimic.robots.x30.inspect import main

if __name__ == "__main__":
    main()
