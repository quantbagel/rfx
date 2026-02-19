"""
rfx.urdf - URDF parser and kinematic model

Parse URDF files into structured data, compute forward kinematics,
and auto-generate RobotConfig from URDF descriptions.

    >>> model = rfx.URDF.load("go2.urdf")
    >>> model.actuated_joints
    ['FL_hip_joint', 'FL_thigh_joint', ...]
    >>> fk = model.forward_kinematics([0.0] * model.num_actuated)
    >>> fk["FL_foot"]
    array([...])  # 4x4 homogeneous transform
    >>> config = model.to_robot_config()
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import RobotConfig


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------


@dataclass
class Origin:
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Box:
    size: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Cylinder:
    radius: float = 0.0
    length: float = 0.0


@dataclass
class Sphere:
    radius: float = 0.0


@dataclass
class Mesh:
    filename: str = ""
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class Geometry:
    box: Box | None = None
    cylinder: Cylinder | None = None
    sphere: Sphere | None = None
    mesh: Mesh | None = None


@dataclass
class Material:
    name: str = ""
    color: tuple[float, float, float, float] | None = None
    texture: str | None = None


@dataclass
class Inertial:
    origin: Origin = field(default_factory=Origin)
    mass: float = 0.0
    ixx: float = 0.0
    ixy: float = 0.0
    ixz: float = 0.0
    iyy: float = 0.0
    iyz: float = 0.0
    izz: float = 0.0


@dataclass
class Visual:
    origin: Origin = field(default_factory=Origin)
    geometry: Geometry | None = None
    material: Material | None = None
    name: str = ""


@dataclass
class Collision:
    origin: Origin = field(default_factory=Origin)
    geometry: Geometry | None = None
    name: str = ""


# ---------------------------------------------------------------------------
# Links and joints
# ---------------------------------------------------------------------------


@dataclass
class URDFLink:
    name: str
    inertial: Inertial | None = None
    visuals: list[Visual] = field(default_factory=list)
    collisions: list[Collision] = field(default_factory=list)


@dataclass
class JointLimit:
    lower: float = 0.0
    upper: float = 0.0
    effort: float = 0.0
    velocity: float = 0.0


@dataclass
class URDFJoint:
    name: str
    type: str  # "revolute", "continuous", "prismatic", "fixed", "floating", "planar"
    parent: str
    child: str
    origin: Origin = field(default_factory=Origin)
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    limit: JointLimit | None = None

    @property
    def is_actuated(self) -> bool:
        return self.type in ("revolute", "continuous", "prismatic")

    @property
    def is_fixed(self) -> bool:
        return self.type == "fixed"


# ---------------------------------------------------------------------------
# Transform math (pure Python, no deps)
# ---------------------------------------------------------------------------


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    """Euler angles (XYZ convention) to 3x3 rotation matrix (row-major)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _origin_to_matrix(origin: Origin) -> list[list[float]]:
    """Convert Origin(xyz, rpy) to a 4x4 homogeneous transform (row-major)."""
    rot = _rpy_to_matrix(*origin.rpy)
    x, y, z = origin.xyz
    return [
        [rot[0][0], rot[0][1], rot[0][2], x],
        [rot[1][0], rot[1][1], rot[1][2], y],
        [rot[2][0], rot[2][1], rot[2][2], z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _axis_rotation(axis: tuple[float, float, float], angle: float) -> list[list[float]]:
    """Rotation matrix (4x4) around an arbitrary unit axis by angle (radians)."""
    ax, ay, az = axis
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm < 1e-12:
        return _identity4()
    ax, ay, az = ax / norm, ay / norm, az / norm
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return [
        [t * ax * ax + c, t * ax * ay - s * az, t * ax * az + s * ay, 0.0],
        [t * ax * ay + s * az, t * ay * ay + c, t * ay * az - s * ax, 0.0],
        [t * ax * az - s * ay, t * ay * az + s * ax, t * az * az + c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _axis_translation(axis: tuple[float, float, float], distance: float) -> list[list[float]]:
    """Translation matrix (4x4) along an arbitrary axis by distance."""
    ax, ay, az = axis
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm < 1e-12:
        return _identity4()
    ax, ay, az = ax / norm, ay / norm, az / norm
    return [
        [1.0, 0.0, 0.0, ax * distance],
        [0.0, 1.0, 0.0, ay * distance],
        [0.0, 0.0, 1.0, az * distance],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _identity4() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matmul4(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Multiply two 4x4 matrices."""
    result = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            s = 0.0
            for k in range(4):
                s += a[i][k] * b[k][j]
            result[i][j] = s
    return result


# ---------------------------------------------------------------------------
# XML parsing helpers
# ---------------------------------------------------------------------------


def _parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split())


def _parse_origin(elem: ET.Element | None) -> Origin:
    if elem is None:
        return Origin()
    xyz = _parse_floats(elem.get("xyz", "0 0 0"))
    rpy = _parse_floats(elem.get("rpy", "0 0 0"))
    return Origin(xyz=(xyz[0], xyz[1], xyz[2]), rpy=(rpy[0], rpy[1], rpy[2]))


def _parse_geometry(elem: ET.Element | None) -> Geometry | None:
    if elem is None:
        return None
    geom = Geometry()
    box_elem = elem.find("box")
    if box_elem is not None:
        size = _parse_floats(box_elem.get("size", "0 0 0"))
        geom.box = Box(size=(size[0], size[1], size[2]))
    cyl_elem = elem.find("cylinder")
    if cyl_elem is not None:
        geom.cylinder = Cylinder(
            radius=float(cyl_elem.get("radius", "0")),
            length=float(cyl_elem.get("length", "0")),
        )
    sph_elem = elem.find("sphere")
    if sph_elem is not None:
        geom.sphere = Sphere(radius=float(sph_elem.get("radius", "0")))
    mesh_elem = elem.find("mesh")
    if mesh_elem is not None:
        scale_str = mesh_elem.get("scale", "1 1 1")
        scale = _parse_floats(scale_str)
        geom.mesh = Mesh(
            filename=mesh_elem.get("filename", ""),
            scale=(scale[0], scale[1], scale[2]),
        )
    return geom


def _parse_material(elem: ET.Element | None) -> Material | None:
    if elem is None:
        return None
    mat = Material(name=elem.get("name", ""))
    color_elem = elem.find("color")
    if color_elem is not None:
        rgba = _parse_floats(color_elem.get("rgba", "1 1 1 1"))
        mat.color = (rgba[0], rgba[1], rgba[2], rgba[3])
    tex_elem = elem.find("texture")
    if tex_elem is not None:
        mat.texture = tex_elem.get("filename", "")
    return mat


def _parse_inertial(elem: ET.Element | None) -> Inertial | None:
    if elem is None:
        return None
    origin = _parse_origin(elem.find("origin"))
    mass_elem = elem.find("mass")
    mass = float(mass_elem.get("value", "0")) if mass_elem is not None else 0.0
    inertia_elem = elem.find("inertia")
    if inertia_elem is not None:
        return Inertial(
            origin=origin,
            mass=mass,
            ixx=float(inertia_elem.get("ixx", "0")),
            ixy=float(inertia_elem.get("ixy", "0")),
            ixz=float(inertia_elem.get("ixz", "0")),
            iyy=float(inertia_elem.get("iyy", "0")),
            iyz=float(inertia_elem.get("iyz", "0")),
            izz=float(inertia_elem.get("izz", "0")),
        )
    return Inertial(origin=origin, mass=mass)


def _parse_visual(elem: ET.Element) -> Visual:
    return Visual(
        origin=_parse_origin(elem.find("origin")),
        geometry=_parse_geometry(elem.find("geometry")),
        material=_parse_material(elem.find("material")),
        name=elem.get("name", ""),
    )


def _parse_collision(elem: ET.Element) -> Collision:
    return Collision(
        origin=_parse_origin(elem.find("origin")),
        geometry=_parse_geometry(elem.find("geometry")),
        name=elem.get("name", ""),
    )


def _parse_link(elem: ET.Element) -> URDFLink:
    name = elem.get("name", "")
    inertial = _parse_inertial(elem.find("inertial"))
    visuals = [_parse_visual(v) for v in elem.findall("visual")]
    collisions = [_parse_collision(c) for c in elem.findall("collision")]
    return URDFLink(name=name, inertial=inertial, visuals=visuals, collisions=collisions)


def _parse_joint(elem: ET.Element) -> URDFJoint:
    name = elem.get("name", "")
    joint_type = elem.get("type", "fixed")
    parent = elem.find("parent")
    child = elem.find("child")
    origin = _parse_origin(elem.find("origin"))

    axis_elem = elem.find("axis")
    if axis_elem is not None:
        axis_vals = _parse_floats(axis_elem.get("xyz", "1 0 0"))
        axis = (axis_vals[0], axis_vals[1], axis_vals[2])
    else:
        axis = (1.0, 0.0, 0.0)

    limit_elem = elem.find("limit")
    limit = None
    if limit_elem is not None:
        limit = JointLimit(
            lower=float(limit_elem.get("lower", "0")),
            upper=float(limit_elem.get("upper", "0")),
            effort=float(limit_elem.get("effort", "0")),
            velocity=float(limit_elem.get("velocity", "0")),
        )

    return URDFJoint(
        name=name,
        type=joint_type,
        parent=parent.get("link", "") if parent is not None else "",
        child=child.get("link", "") if child is not None else "",
        origin=origin,
        axis=axis,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# URDF model
# ---------------------------------------------------------------------------


class URDF:
    """Parsed URDF robot model.

    Provides kinematic tree, forward kinematics, and RobotConfig generation.

    Examples:
        >>> model = URDF.load("go2.urdf")
        >>> print(model.name, model.num_actuated, "DOF")
        go2_description 12 DOF

        >>> fk = model.forward_kinematics([0.0] * 12)
        >>> fk["FL_foot"]  # 4x4 homogeneous transform
        [[...], [...], [...], [...]]

        >>> config = model.to_robot_config()
        >>> config.action_dim
        12
    """

    def __init__(
        self,
        name: str,
        links: list[URDFLink],
        joints: list[URDFJoint],
        materials: dict[str, Material],
        source_path: str | None = None,
    ):
        self.name = name
        self.links = {link.name: link for link in links}
        self.joints = {joint.name: joint for joint in joints}
        self.materials = materials
        self.source_path = source_path

        # Build parent→children map and child→parent map
        self._children: dict[str, list[str]] = {}  # link_name → [joint_names]
        self._parent_joint: dict[str, str] = {}  # child_link → joint_name
        for jname, joint in self.joints.items():
            self._children.setdefault(joint.parent, []).append(jname)
            self._parent_joint[joint.child] = jname

        # Find root link (link that is never a child)
        child_links = {j.child for j in self.joints.values()}
        roots = [name for name in self.links if name not in child_links]
        self.root = roots[0] if roots else ""

        # Cache actuated joint list in tree-traversal order
        self._actuated: list[URDFJoint] = []
        self._walk_tree(self.root)

    def _walk_tree(self, link_name: str) -> None:
        for jname in self._children.get(link_name, []):
            joint = self.joints[jname]
            if joint.is_actuated:
                self._actuated.append(joint)
            self._walk_tree(joint.child)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> URDF:
        """Load and parse a URDF file."""
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"URDF file not found: {path}")
        model = cls.from_string(path.read_text())
        model.source_path = str(path)
        return model

    @classmethod
    def from_string(cls, xml_string: str) -> URDF:
        """Parse URDF from an XML string."""
        root = ET.fromstring(xml_string)
        name = root.get("name", "robot")

        # Top-level materials
        materials: dict[str, Material] = {}
        for mat_elem in root.findall("material"):
            mat = _parse_material(mat_elem)
            if mat is not None:
                materials[mat.name] = mat

        links = [_parse_link(elem) for elem in root.findall("link")]
        joints = [_parse_joint(elem) for elem in root.findall("joint")]
        return cls(name=name, links=links, joints=joints, materials=materials)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def actuated_joints(self) -> list[str]:
        """Names of all actuated (non-fixed) joints in tree order."""
        return [j.name for j in self._actuated]

    @property
    def num_actuated(self) -> int:
        return len(self._actuated)

    @property
    def joint_names(self) -> list[str]:
        """All joint names."""
        return list(self.joints.keys())

    @property
    def link_names(self) -> list[str]:
        """All link names."""
        return list(self.links.keys())

    @property
    def joint_limits(self) -> dict[str, JointLimit]:
        """Joint limits for actuated joints that have them."""
        return {j.name: j.limit for j in self._actuated if j.limit is not None}

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------

    def forward_kinematics(
        self,
        joint_positions: Sequence[float],
        base_transform: list[list[float]] | None = None,
    ) -> dict[str, list[list[float]]]:
        """Compute forward kinematics for all links.

        Args:
            joint_positions: Positions for each actuated joint (in tree order).
                Length must equal ``num_actuated``.
            base_transform: Optional 4x4 base transform. Defaults to identity.

        Returns:
            Dict mapping link names to 4x4 homogeneous transforms (row-major
            nested lists). Use numpy to convert: ``np.array(fk["link"])``.
        """
        if len(joint_positions) != self.num_actuated:
            raise ValueError(
                f"Expected {self.num_actuated} joint positions, got {len(joint_positions)}"
            )

        # Map actuated joint name → position
        q_map: dict[str, float] = {}
        for joint, q in zip(self._actuated, joint_positions, strict=True):
            q_map[joint.name] = q

        result: dict[str, list[list[float]]] = {}
        root_tf = base_transform if base_transform is not None else _identity4()
        self._fk_recurse(self.root, root_tf, q_map, result)
        return result

    def _fk_recurse(
        self,
        link_name: str,
        parent_tf: list[list[float]],
        q_map: dict[str, float],
        result: dict[str, list[list[float]]],
    ) -> None:
        result[link_name] = parent_tf
        for jname in self._children.get(link_name, []):
            joint = self.joints[jname]
            # Joint origin transform
            joint_tf = _origin_to_matrix(joint.origin)
            tf = _matmul4(parent_tf, joint_tf)

            # Apply joint motion
            if joint.name in q_map:
                q = q_map[joint.name]
                if joint.type in ("revolute", "continuous"):
                    motion = _axis_rotation(joint.axis, q)
                elif joint.type == "prismatic":
                    motion = _axis_translation(joint.axis, q)
                else:
                    motion = _identity4()
                tf = _matmul4(tf, motion)

            self._fk_recurse(joint.child, tf, q_map, result)

    def link_position(
        self,
        link_name: str,
        joint_positions: Sequence[float],
    ) -> tuple[float, float, float]:
        """Get the world-frame position of a single link.

        Args:
            link_name: Name of the link.
            joint_positions: Positions for all actuated joints.

        Returns:
            (x, y, z) position tuple.
        """
        fk = self.forward_kinematics(joint_positions)
        if link_name not in fk:
            raise KeyError(f"Link '{link_name}' not found in model")
        tf = fk[link_name]
        return (tf[0][3], tf[1][3], tf[2][3])

    def link_chain(self, from_link: str, to_link: str) -> list[str]:
        """Find the chain of links from ``from_link`` to ``to_link``.

        Walks up from both links to the root, then finds the common ancestor
        and returns the path.
        """

        def _ancestors(link: str) -> list[str]:
            path = [link]
            while link in self._parent_joint:
                jname = self._parent_joint[link]
                link = self.joints[jname].parent
                path.append(link)
            return path

        ancestors_from = _ancestors(from_link)
        ancestors_to = _ancestors(to_link)
        set_from = set(ancestors_from)

        # Find common ancestor
        common = None
        for link in ancestors_to:
            if link in set_from:
                common = link
                break
        if common is None:
            raise ValueError(f"No path between '{from_link}' and '{to_link}'")

        # Build path: from → common → to
        up = []
        for link in ancestors_from:
            up.append(link)
            if link == common:
                break
        down = []
        for link in ancestors_to:
            if link == common:
                break
            down.append(link)
        down.reverse()
        return up + down

    # ------------------------------------------------------------------
    # Config generation
    # ------------------------------------------------------------------

    def to_robot_config(
        self,
        *,
        control_freq_hz: int = 50,
        max_state_dim: int = 64,
        max_action_dim: int = 64,
    ) -> RobotConfig:
        """Generate a RobotConfig from this URDF.

        Uses actuated joints for state/action dims and extracts joint limits.
        """
        from .config import JointConfig, RobotConfig

        joints = []
        for i, j in enumerate(self._actuated):
            kw: dict[str, Any] = {"name": j.name, "index": i}
            if j.limit is not None:
                kw["position_min"] = j.limit.lower
                kw["position_max"] = j.limit.upper
                kw["velocity_max"] = j.limit.velocity
                kw["effort_max"] = j.limit.effort
            joints.append(JointConfig(**kw))

        n = self.num_actuated
        return RobotConfig(
            name=self.name,
            urdf_path=self.source_path,
            state_dim=n,
            action_dim=n,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            joints=joints,
            control_freq_hz=control_freq_hz,
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"URDF(name='{self.name}', "
            f"links={len(self.links)}, "
            f"joints={len(self.joints)}, "
            f"actuated={self.num_actuated})"
        )

    def print_tree(self, link: str | None = None, indent: int = 0) -> None:
        """Print the kinematic tree to stdout."""
        if link is None:
            link = self.root
        prefix = "  " * indent
        print(f"{prefix}{link}")
        for jname in self._children.get(link, []):
            joint = self.joints[jname]
            jtype = joint.type
            extra = ""
            if joint.limit is not None:
                extra = f" [{joint.limit.lower:.3f}, {joint.limit.upper:.3f}]"
            print(f"{prefix}  \u2514\u2500 {jname} ({jtype}){extra}")
            self.print_tree(joint.child, indent + 2)

    @property
    def total_mass(self) -> float:
        """Sum of all link masses."""
        return sum(link.inertial.mass for link in self.links.values() if link.inertial is not None)
