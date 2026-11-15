"""Compile X30 through MuJoCo or mjlab and export actual joint/actuator IDs.

Run from the project root with:
  python -m go2_supermimic.robots.x30.inspect --backend mujoco
  python -m go2_supermimic.robots.x30.inspect --backend mjlab

The direct MuJoCo backend validates the structural MJCF. The mjlab backend
compiles the entity after mjlab has applied its actuator configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from go2_supermimic.robots.x30.model import BASE_JOINT_NAME, SDK_JOINT_ORDER, load_x30_spec


from go2_supermimic.paths import PROJECT_ROOT
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resources"


def compile_model(backend: str = "mjlab", robot_config: str = ""):
  """Compile the configured model; mjlab mode includes entity-side edits."""
  import mujoco

  if backend == "mujoco":
    if robot_config:
      raise ValueError("--robot-config requires --backend mjlab")
    return load_x30_spec().compile()
  if backend == "mjlab":
    from mjlab.entity.entity import Entity

    from go2_supermimic.robots.x30.structural import get_x30_structural_cfg

    if robot_config:
      from go2_supermimic.robots.x30.robot import load_parameters, robot_cfg
      return Entity(robot_cfg(load_parameters(robot_config))).spec.compile()
    return Entity(get_x30_structural_cfg()).spec.compile()
  raise ValueError(f"unknown backend: {backend}")


def _name(model, obj_type, obj_id: int) -> str | None:
  import mujoco

  return mujoco.mj_id2name(model, obj_type, obj_id)


def index_model(model, backend: str) -> dict[str, Any]:
  """Build SDK-ordered indices from a compiled MuJoCo model."""
  import mujoco

  joints_by_name: dict[str, dict[str, Any]] = {}
  for joint_id in range(model.njnt):
    name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    if name is None:
      continue
    joints_by_name[name] = {
      "joint_id": joint_id,
      "qpos_address": int(model.jnt_qposadr[joint_id]),
      "dof_address": int(model.jnt_dofadr[joint_id]),
      "joint_type": int(model.jnt_type[joint_id]),
    }

  missing = [name for name in (BASE_JOINT_NAME, *SDK_JOINT_ORDER) if name not in joints_by_name]
  if missing:
    raise ValueError(f"compiled X30 model is missing joints: {missing}")

  actuators_by_joint: dict[str, tuple[int, str | None]] = {}
  for actuator_id in range(model.nu):
    if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
      continue
    target_joint_id = int(model.actuator_trnid[actuator_id, 0])
    target_joint_name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, target_joint_id)
    if target_joint_name in SDK_JOINT_ORDER:
      if target_joint_name in actuators_by_joint:
        raise ValueError(f"multiple actuators target {target_joint_name}")
      actuators_by_joint[target_joint_name] = (
        actuator_id,
        _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id),
      )

  if model.nu not in (0, len(SDK_JOINT_ORDER)):
    raise ValueError(f"expected 0 structural or 12 configured actuators, got {model.nu}")
  if model.nu and set(actuators_by_joint) != set(SDK_JOINT_ORDER):
    missing_actuators = sorted(set(SDK_JOINT_ORDER) - set(actuators_by_joint))
    raise ValueError(f"actuator mapping is incomplete; missing {missing_actuators}")

  base = joints_by_name[BASE_JOINT_NAME]
  rows = []
  for sdk_index, joint_name in enumerate(SDK_JOINT_ORDER):
    row = {"sdk_index": sdk_index, "name": joint_name, **joints_by_name[joint_name]}
    actuator = actuators_by_joint.get(joint_name)
    row["actuator_id"] = None if actuator is None else actuator[0]
    row["actuator_name"] = None if actuator is None else actuator[1]
    rows.append(row)

  return {
    "backend": backend,
    "model_name": _name(model, mujoco.mjtObj.mjOBJ_MODEL, 0) or "X30_mjlab",
    "dimensions": {"nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu), "njnt": int(model.njnt)},
    "floating_base": {"name": BASE_JOINT_NAME, **base},
    "joints": rows,
    "actuator_mapping_ready": len(actuators_by_joint) == len(SDK_JOINT_ORDER),
    "actuator_note": (
      "Actuator indices are not available: X30 actuator parameters/configuration are pending."
      if not model.nu
      else "Actuator IDs are read from the compiled model transmission targets."
    ),
    "source_note": (
      "Candidate X30 model; verify that this model revision matches the physical X30 PRO "
      "before using it for dynamics training or sim-to-real."
    ),
  }


def render_markdown(result: dict[str, Any]) -> str:
  dims = result["dimensions"]
  lines = [
    "# X30 mjlab joint and actuator indices",
    "",
    f"- Compiled backend: `{result['backend']}`",
    f"- Dimensions: `nq={dims['nq']}`, `nv={dims['nv']}`, `nu={dims['nu']}`, `njnt={dims['njnt']}`",
    f"- Floating base: `{result['floating_base']['name']}` — joint ID `{result['floating_base']['joint_id']}`, "
    f"qpos `{result['floating_base']['qpos_address']}`, dof `{result['floating_base']['dof_address']}`.",
    f"- Actuator mapping: {'ready' if result['actuator_mapping_ready'] else 'pending'} — {result['actuator_note']}",
    f"- Model caveat: {result['source_note']}",
    "",
    "| SDK index | Joint name | Joint ID | qpos address | dof address | Actuator ID | Actuator name |",
    "|---:|---|---:|---:|---:|---:|---|",
  ]
  for row in result["joints"]:
    actuator_id = "pending" if row["actuator_id"] is None else str(row["actuator_id"])
    actuator_name = "pending" if row["actuator_name"] is None else row["actuator_name"]
    lines.append(
      f"| {row['sdk_index']} | `{row['name']}` | {row['joint_id']} | {row['qpos_address']} | "
      f"{row['dof_address']} | {actuator_id} | {actuator_name} |"
    )
  lines += ["", "The SDK order is FL, FR, HL, HR; within each leg: HipX, HipY, Knee.", ""]
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--backend", choices=("mjlab", "mujoco"), default="mjlab")
  parser.add_argument("--robot-config", default="", help="Optional verified training motor JSON; mjlab backend only.")
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
  args = parser.parse_args()

  model = compile_model(args.backend, robot_config=args.robot_config)
  result = index_model(model, backend=args.backend)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  json_path = args.output_dir / "x30_mjlab_indices.json"
  markdown_path = args.output_dir / "x30_mjlab_indices.md"
  json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
  markdown_path.write_text(render_markdown(result), encoding="utf-8")
  print(f"compiled X30: nq={model.nq}, nv={model.nv}, nu={model.nu}, njnt={model.njnt}")
  print(f"wrote {json_path}")
  print(f"wrote {markdown_path}")


if __name__ == "__main__":
  main()
