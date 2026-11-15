"""Evaluate a learned X30 trot with physical gait metrics and optional world-model audit."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from go2_supermimic.robots.x30.trot_eval import main
if __name__=='__main__':main()
