from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
EMBODIMENT_DIR = ROOT / "assets" / "embodiments" / "nero"
SRC = EMBODIMENT_DIR / "nero_with_gripper_description.urdf"
DST = EMBODIMENT_DIR / "nero_step5_tmp.urdf"


def main():
    tree = ET.parse(SRC)
    robot = tree.getroot()

    for link in robot.findall("link"):
        for collision in list(link.findall("collision")):
            link.remove(collision)

    for joint in robot.findall("joint"):
        if joint.get("type") == "revolute":
            joint.set("type", "continuous")
            limit = joint.find("limit")
            if limit is None:
                limit = ET.SubElement(joint, "limit")
            effort = limit.get("effort", "100")
            velocity = limit.get("velocity", "5")
            limit.clear()
            limit.set("effort", effort)
            limit.set("velocity", velocity)

    tree.write(DST, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {DST}")


if __name__ == "__main__":
    main()
