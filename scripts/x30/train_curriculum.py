"""Start or resume the automatic X30 omnidirectional curriculum."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from go2_supermimic.robots.x30.locomotion_train import main
if __name__=='__main__':main()
