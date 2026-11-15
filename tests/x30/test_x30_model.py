"""Structural checks for the X30 MJCF integration."""

from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
ROBOT_DIR = ROOT / "go2_supermimic" / "robot"
X30_XML = ROBOT_DIR / "x30.xml"
SDK_JOINT_ORDER = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "HL", "HR")
    for joint in ("HipX", "HipY", "Knee")
)


class X30ModelStructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = ET.parse(X30_XML).getroot()

    def test_has_free_base_and_expected_sdk_joint_order(self):
        base = self.root.find("./worldbody/body[@name='base']")
        self.assertIsNotNone(base)
        self.assertIsNotNone(base.find("freejoint[@name='floating_base_joint']"))
        names = tuple(j.attrib["name"] for j in self.root.findall(".//joint"))
        self.assertEqual(names, SDK_JOINT_ORDER)

    def test_has_four_named_foot_sites_and_collision_geometries(self):
        site_names = {site.attrib["name"] for site in self.root.findall(".//site")}
        geom_names = {geom.attrib.get("name") for geom in self.root.findall(".//geom")}
        for leg in ("FL", "FR", "HL", "HR"):
            self.assertIn(leg, site_names)
            self.assertIn(f"{leg}_foot_collision", geom_names)

    def test_has_imu_sensor_frame_and_no_embedded_ground_plane(self):
        site_names = {site.attrib["name"] for site in self.root.findall(".//site")}
        self.assertIn("imu", site_names)
        sensor_names = {sensor.attrib.get("name") for sensor in self.root.findall("./sensor/*")}
        self.assertTrue({"imu_ang_vel", "imu_lin_vel", "imu_lin_acc", "imu_upvector"}.issubset(sensor_names))
        self.assertIsNone(self.root.find("./worldbody/geom[@name='floor']"))

    def test_mesh_references_resolve_inside_robot_package(self):
        mesh_files = {
            mesh.attrib["file"]
            for mesh in self.root.findall("./asset/mesh")
        }
        self.assertTrue(mesh_files)
        for filename in mesh_files:
            self.assertTrue((ROBOT_DIR / "x30_assets" / filename).is_file(), filename)

    def test_x30_assets_and_license_are_in_package_data(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        package_data = config["tool"]["setuptools"]["package-data"]["go2_supermimic"]
        self.assertIn("robot/x30.xml", package_data)
        self.assertIn("robot/x30_assets/*.STL", package_data)
        self.assertIn("robot/LICENSE_deep_robotics_model.txt", package_data)

    def test_actuators_are_not_fabricated_without_x30_motor_parameters(self):
        self.assertEqual(self.root.findall("./actuator/*"), [])
        self.assertTrue(all("actuatorfrcrange" not in joint.attrib for joint in self.root.findall(".//joint")))

    def test_compiled_model_index_uses_sdk_order_and_reports_actuators_pending(self):
        from go2_supermimic.robot.x30_inspect import compile_model, index_model

        model = compile_model("mujoco")
        result = index_model(model, backend="mujoco MJCF")
        self.assertEqual(result["dimensions"], {"nq": 19, "nv": 18, "nu": 0, "njnt": 13})
        self.assertEqual(result["floating_base"]["name"], "floating_base_joint")
        self.assertEqual(result["floating_base"]["qpos_address"], 0)
        self.assertEqual(result["floating_base"]["dof_address"], 0)
        self.assertEqual(tuple(row["name"] for row in result["joints"]), SDK_JOINT_ORDER)
        self.assertEqual(tuple(row["qpos_address"] for row in result["joints"]), tuple(range(7, 19)))
        self.assertEqual(tuple(row["dof_address"] for row in result["joints"]), tuple(range(6, 18)))
        self.assertTrue(all(row["actuator_id"] is None for row in result["joints"]))
        self.assertFalse(result["actuator_mapping_ready"])

    def test_mjlab_entity_compiles_the_structural_model(self):
        from go2_supermimic.robot.x30_inspect import compile_model, index_model

        model = compile_model("mjlab")
        result = index_model(model, backend="mjlab Entity")
        self.assertEqual(result["dimensions"], {"nq": 19, "nv": 18, "nu": 0, "njnt": 13})
        self.assertEqual(tuple(row["name"] for row in result["joints"]), SDK_JOINT_ORDER)

    def test_actuator_ids_are_resolved_from_compiled_transmissions_not_assumed_order(self):
        import mujoco

        from go2_supermimic.robot.x30_inspect import index_model

        root = ET.fromstring(ET.tostring(self.root))
        actuators = ET.SubElement(root, "actuator")
        for name in reversed(SDK_JOINT_ORDER):
            ET.SubElement(actuators, "motor", {"name": f"test_{name}", "joint": name})
        assets = {
            f"x30_assets/{path.name}": path.read_bytes()
            for path in (ROBOT_DIR / "x30_assets").glob("*.STL")
        }
        model = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"), assets=assets).compile()
        result = index_model(model, backend="synthetic actuator order check")
        self.assertEqual(model.nu, 12)
        self.assertTrue(result["actuator_mapping_ready"])
        for row in result["joints"]:
            self.assertEqual(row["actuator_id"], 11 - row["sdk_index"])
            self.assertEqual(row["actuator_name"], f"test_{row['name']}")


if __name__ == "__main__":
    unittest.main()
