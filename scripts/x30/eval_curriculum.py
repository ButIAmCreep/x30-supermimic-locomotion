"""Evaluate an automatic-curriculum checkpoint, optionally recording video."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from go2_supermimic.robots.x30.locomotion_eval import main
if __name__=='__main__':main()
