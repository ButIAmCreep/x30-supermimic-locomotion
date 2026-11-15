"""Names and ordering shared by the X30 model, checks, and SDK mapping."""

from pathlib import Path


ROBOT_DIR = Path(__file__).resolve().parent / "model"
X30_XML = ROBOT_DIR / "x30.xml"
X30_ASSET_DIR = ROBOT_DIR / "x30_assets"

LEG_ORDER = ("FL", "FR", "HL", "HR")
JOINT_TYPES = ("HipX", "HipY", "Knee")
SDK_JOINT_ORDER = tuple(f"{leg}_{joint}_joint" for leg in LEG_ORDER for joint in JOINT_TYPES)
FOOT_SITE_NAMES = LEG_ORDER
FOOT_GEOM_NAMES = tuple(f"{leg}_foot_collision" for leg in LEG_ORDER)
HIP_BODY_NAMES = tuple(f"{leg}_THIGH" for leg in LEG_ORDER)
KNEE_BODY_NAMES = tuple(f"{leg}_SHANK" for leg in LEG_ORDER)
BASE_BODY_NAME = "base"
BASE_JOINT_NAME = "floating_base_joint"


def load_x30_spec():
  """Load MJCF and meshes from Python bytes, safe for non-ASCII install paths."""
  import mujoco

  assets = {
    f"x30_assets/{path.name}": path.read_bytes()
    for path in X30_ASSET_DIR.glob("*.STL")
  }
  xml = X30_XML.read_text(encoding="utf-8")
  return mujoco.MjSpec.from_string(xml, assets=assets)
