"""Ideal side chains on a wild-type backbone.

Chi angles are the staggered Dunbrack wells (60, 180, -60 degrees), not
values fit to affinity labels. Bond lengths and angles are fixed ideal
values. The second tetrahedral branch sits 120 degrees from the first.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

# C-CA-CB geometry. The dihedral is N-C-CA-CB for an L amino acid.
CB_LENGTH = 1.530
CB_ANGLE = 110.5
CB_DIHEDRAL = 122.5
SECOND_BRANCH = 120.0

Step = namedtuple("Step", "name a b c length angle chi offset")

# chi >= 0: dihedral = chis[chi] + offset
# chi < 0: dihedral = offset (a fixed planar or gauche angle)
STEPS = {
    "SER": [Step("OG", "N", "CA", "CB", 1.417, 112.0, 0, 0.0)],
    "CYS": [Step("SG", "N", "CA", "CB", 1.808, 114.0, 0, 0.0)],
    "THR": [
        Step("OG1", "N", "CA", "CB", 1.433, 109.5, 0, 0.0),
        Step("CG2", "N", "CA", "CB", 1.521, 110.5, 0, -SECOND_BRANCH),
    ],
    "VAL": [
        Step("CG1", "N", "CA", "CB", 1.521, 110.5, 0, 0.0),
        Step("CG2", "N", "CA", "CB", 1.521, 110.5, 0, SECOND_BRANCH),
    ],
    "LEU": [
        Step("CG", "N", "CA", "CB", 1.530, 116.0, 0, 0.0),
        Step("CD1", "CA", "CB", "CG", 1.521, 110.5, 1, 0.0),
        Step("CD2", "CA", "CB", "CG", 1.521, 110.5, 1, SECOND_BRANCH),
    ],
    "ILE": [
        Step("CG1", "N", "CA", "CB", 1.530, 110.4, 0, 0.0),
        Step("CG2", "N", "CA", "CB", 1.521, 110.5, 0, -SECOND_BRANCH),
        Step("CD1", "CA", "CB", "CG1", 1.521, 113.8, 1, 0.0),
    ],
    "MET": [
        Step("CG", "N", "CA", "CB", 1.530, 114.0, 0, 0.0),
        Step("SD", "CA", "CB", "CG", 1.810, 112.7, 1, 0.0),
        Step("CE", "CB", "CG", "SD", 1.820, 100.9, 2, 0.0),
    ],
    "ASP": [
        Step("CG", "N", "CA", "CB", 1.516, 112.6, 0, 0.0),
        Step("OD1", "CA", "CB", "CG", 1.249, 118.4, 1, 0.0),
        Step("OD2", "CA", "CB", "CG", 1.249, 118.4, 1, 180.0),
    ],
    "ASN": [
        Step("CG", "N", "CA", "CB", 1.516, 112.6, 0, 0.0),
        Step("OD1", "CA", "CB", "CG", 1.231, 120.8, 1, 0.0),
        Step("ND2", "CA", "CB", "CG", 1.328, 116.5, 1, 180.0),
    ],
    "GLU": [
        Step("CG", "N", "CA", "CB", 1.520, 113.8, 0, 0.0),
        Step("CD", "CA", "CB", "CG", 1.516, 112.6, 1, 0.0),
        Step("OE1", "CB", "CG", "CD", 1.249, 118.4, 2, 0.0),
        Step("OE2", "CB", "CG", "CD", 1.249, 118.4, 2, 180.0),
    ],
    "GLN": [
        Step("CG", "N", "CA", "CB", 1.520, 113.8, 0, 0.0),
        Step("CD", "CA", "CB", "CG", 1.516, 112.6, 1, 0.0),
        Step("OE1", "CB", "CG", "CD", 1.231, 120.8, 2, 0.0),
        Step("NE2", "CB", "CG", "CD", 1.328, 116.5, 2, 180.0),
    ],
    "LYS": [
        Step("CG", "N", "CA", "CB", 1.520, 113.8, 0, 0.0),
        Step("CD", "CA", "CB", "CG", 1.520, 111.8, 1, 0.0),
        Step("CE", "CB", "CG", "CD", 1.520, 111.8, 2, 0.0),
        Step("NZ", "CG", "CD", "CE", 1.489, 111.9, 3, 0.0),
    ],
    "ARG": [
        Step("CG", "N", "CA", "CB", 1.520, 113.8, 0, 0.0),
        Step("CD", "CA", "CB", "CG", 1.520, 111.8, 1, 0.0),
        Step("NE", "CB", "CG", "CD", 1.460, 112.0, 2, 0.0),
        Step("CZ", "CG", "CD", "NE", 1.329, 124.2, 3, 0.0),
        Step("NH1", "CD", "NE", "CZ", 1.326, 120.0, -1, 0.0),
        Step("NH2", "CD", "NE", "CZ", 1.326, 120.0, -1, 180.0),
    ],
    "HIS": [
        Step("CG", "N", "CA", "CB", 1.497, 113.8, 0, 0.0),
        Step("ND1", "CA", "CB", "CG", 1.378, 122.7, 1, 0.0),
        Step("CD2", "CA", "CB", "CG", 1.354, 130.8, 1, 180.0),
        Step("CE1", "CB", "CG", "ND1", 1.321, 108.5, -1, 180.0),
        Step("NE2", "CG", "ND1", "CE1", 1.374, 109.2, -1, 0.0),
    ],
    "PRO": [
        Step("CG", "N", "CA", "CB", 1.492, 104.0, 0, 0.0),
        Step("CD", "CA", "CB", "CG", 1.503, 104.5, 1, 0.0),
    ],
    "TRP": [
        Step("CG", "N", "CA", "CB", 1.498, 113.6, 0, 0.0),
        Step("CD1", "CA", "CB", "CG", 1.365, 126.8, 1, 0.0),
        Step("CD2", "CA", "CB", "CG", 1.433, 126.6, 1, 180.0),
        Step("NE1", "CB", "CG", "CD1", 1.374, 110.2, -1, 180.0),
        Step("CE2", "CG", "CD1", "NE1", 1.410, 108.8, -1, 0.0),
        Step("CE3", "NE1", "CE2", "CD2", 1.404, 121.5, -1, 180.0),
        Step("CZ2", "CD1", "NE1", "CE2", 1.394, 122.3, -1, 180.0),
        Step("CZ3", "CE2", "CD2", "CE3", 1.392, 120.5, -1, 0.0),
        Step("CH2", "NE1", "CE2", "CZ2", 1.368, 117.5, -1, 180.0),
    ],
}

# Three staggered wells. Phenyl and carboxyl terminals use the angles that
# actually change the heavy-atom placement (90 / 0 rather than a full grid).
_G, _T, _M = 60.0, 180.0, -60.0
ROTAMERS = {
    "ALA": [()],
    "SER": [(_G,), (_T,), (_M,)],
    "CYS": [(_G,), (_T,), (_M,)],
    "THR": [(_G,), (_T,), (_M,)],
    "VAL": [(_G,), (_T,), (_M,)],
    "LEU": [(_M, _T), (_T, _G), (_T, _T), (_G, _T), (_M, _G)],
    "ILE": [(_M, _T), (_M, _M), (_G, _T), (_T, _T), (_T, _G)],
    "MET": [(_M, _T, _T), (_M, _T, _M), (_T, _T, _T), (_T, _T, _G), (_M, _M, _T), (_G, _T, _T)],
    "PHE": [(_M, 90.0), (_M, -90.0), (_T, 90.0), (_G, 90.0)],
    "TYR": [(_M, 90.0), (_M, -90.0), (_T, 90.0), (_G, 90.0)],
    "TRP": [(_M, 90.0), (_M, -90.0), (_T, 90.0), (_G, -90.0)],
    "HIS": [(_M, 80.0), (_M, -100.0), (_T, 80.0), (_G, 80.0)],
    "ASP": [(_M, 0.0), (_M, 90.0), (_T, 0.0), (_T, 90.0), (_G, 0.0)],
    "ASN": [(_M, -40.0), (_M, 40.0), (_T, 20.0), (_G, 30.0)],
    "GLU": [(_M, _T, 0.0), (_M, _T, 90.0), (_T, _T, 0.0), (_M, _M, 0.0), (_G, _T, 0.0)],
    "GLN": [(_M, _T, 0.0), (_M, _T, 90.0), (_T, _T, 0.0), (_M, _M, 0.0), (_G, _T, 0.0)],
    "LYS": [(_M, _T, _T, _T), (_T, _T, _T, _T), (_M, _T, _T, _M), (_T, _T, _G, _T)],
    "ARG": [(_M, _T, _T, _T), (_T, _T, _T, _T), (_M, _T, _T, 90.0), (_T, _T, _G, _T)],
    "PRO": [(30.0, -35.0), (-30.0, 35.0)],
    "GLY": [],
}

CHI_ATOMS = {
    "CYS": [("N", "CA", "CB", "SG")],
    "SER": [("N", "CA", "CB", "OG")],
    "THR": [("N", "CA", "CB", "OG1")],
    "VAL": [("N", "CA", "CB", "CG1")],
    "LEU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "ILE": [("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")],
    "MET": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"), ("CB", "CG", "SD", "CE")],
    "PHE": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "TYR": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "TRP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "HIS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")],
    "ASP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "ASN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "GLU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")],
    "GLN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")],
    "LYS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "CE"), ("CG", "CD", "CE", "NZ")],
    "ARG": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ")],
    "PRO": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")],
}

# United-atom radii: rmin/2 in Angstrom, epsilon in kcal/mol.
# These are softened element parameters, not a reproduction of a force field.
LJ = {
    "C": (2.00, 0.08),
    "N": (1.85, 0.15),
    "O": (1.75, 0.15),
    "S": (2.00, 0.35),
    "P": (2.15, 0.40),
    "F": (1.65, 0.10),
    "CL": (1.95, 0.25),
    "BR": (2.05, 0.30),
    "I": (2.15, 0.35),
    "X": (2.00, 0.08),
}

# Formal charges split over the charged heavy atoms.
ATOM_CHARGE = {
    ("ASP", "OD1"): -0.5, ("ASP", "OD2"): -0.5,
    ("GLU", "OE1"): -0.5, ("GLU", "OE2"): -0.5,
    ("LYS", "NZ"): 1.0,
    ("ARG", "NH1"): 0.5, ("ARG", "NH2"): 0.5,
}

DONORS = {
    ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"), ("TRP", "NE1"),
    ("ASN", "ND2"), ("GLN", "NE2"), ("LYS", "NZ"),
    ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"),
    ("HIS", "ND1"), ("HIS", "NE2"), ("CYS", "SG"),
}
ACCEPTORS = {
    ("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2"),
    ("ASN", "OD1"), ("GLN", "OE1"), ("SER", "OG"), ("THR", "OG1"),
    ("TYR", "OH"), ("HIS", "ND1"), ("HIS", "NE2"),
}

BACKBONE = {"N", "CA", "C", "O", "OXT"}
ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def canon(aa):
    aa = str(aa).strip().upper()
    if len(aa) == 1:
        if aa not in ONE_TO_THREE:
            raise KeyError(aa)
        return ONE_TO_THREE[aa]
    return aa


def _unit(vec):
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return vec / norm


def place(a, b, c, length, angle_deg, dihedral_deg):
    """Place d given a-b-c. Angle is b-c-d. Dihedral is a-b-c-d. Degrees."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    theta = np.deg2rad(float(angle_deg))
    phi = np.deg2rad(float(dihedral_deg))
    bc = _unit(c - b)
    ab = _unit(b - a)
    if float(np.linalg.norm(bc)) < 0.5:
        return c + np.array([length, 0.0, 0.0])
    normal = np.cross(ab, bc)
    nnorm = float(np.linalg.norm(normal))
    if nnorm < 1e-8:
        tmp = np.array([1.0, 0.0, 0.0]) if abs(bc[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        normal = np.cross(tmp, bc)
        nnorm = float(np.linalg.norm(normal))
    normal = normal / nnorm
    binormal = np.cross(normal, bc)
    return (
        c
        - length * np.cos(theta) * bc
        + length * np.sin(theta) * np.cos(phi) * binormal
        + length * np.sin(theta) * np.sin(phi) * normal
    )


def dihedral(p0, p1, p2, p3):
    b0 = np.asarray(p0, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    b1 = np.asarray(p2, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    b2 = np.asarray(p3, dtype=np.float64) - np.asarray(p2, dtype=np.float64)
    b1u = _unit(b1)
    v = b0 - np.dot(b0, b1u) * b1u
    w = b2 - np.dot(b2, b1u) * b1u
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1u, v), w))
    return float(np.degrees(np.arctan2(y, x)))


def angle_deg(a, b, c):
    u = _unit(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    v = _unit(np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    cos = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def build_cb(n, ca, c):
    return place(n, c, ca, CB_LENGTH, CB_ANGLE, CB_DIHEDRAL)


def _ring_from_cd1(cb, cg, cd1, with_oh):
    """Regular hexagon. The ring lies in the CB-CG-CD1 plane."""
    bond = 1.390
    cg = np.asarray(cg, dtype=np.float64)
    cb = np.asarray(cb, dtype=np.float64)
    cd1 = np.asarray(cd1, dtype=np.float64)
    toward_cb = _unit(cb - cg)
    normal = np.cross(cg - cb, cd1 - cg)
    if float(np.linalg.norm(normal)) < 1e-8:
        tmp = np.array([1.0, 0.0, 0.0]) if abs(float(toward_cb[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
        normal = np.cross(toward_cb, tmp)
    normal = _unit(normal)
    center = cg - toward_cb * bond
    radial = cd1 - center
    radial = radial - normal * float(np.dot(radial, normal))
    if float(np.linalg.norm(radial)) < 0.2:
        radial = np.cross(normal, toward_cb)
    radial = _unit(radial) * bond

    def spun(degrees):
        th = np.deg2rad(degrees)
        return radial * np.cos(th) + np.cross(normal, radial) * np.sin(th)

    toward_cg = 60.0 if np.linalg.norm(center + spun(60.0) - cg) < np.linalg.norm(center + spun(-60.0) - cg) else -60.0
    atoms = {
        "CG": cg,
        "CD1": center + radial,
        "CE1": center + spun(-toward_cg),
        "CZ": center + spun(-2.0 * toward_cg),
        "CE2": center + spun(-3.0 * toward_cg),
        "CD2": center + spun(-4.0 * toward_cg),
    }
    if with_oh:
        cz = atoms["CZ"]
        atoms["OH"] = cz + _unit(cz - center) * 1.376
    return atoms


def build_sidechain(aa, n, ca, c, chis, branch=SECOND_BRANCH):
    """Return side-chain heavy atoms, excluding N/CA/C/O. Glycine returns {}."""
    aa = canon(aa)
    atoms = {"N": np.asarray(n, dtype=np.float64), "CA": np.asarray(ca, dtype=np.float64), "C": np.asarray(c, dtype=np.float64)}
    atoms["CB"] = build_cb(atoms["N"], atoms["CA"], atoms["C"])
    if aa == "GLY":
        return {}
    if aa == "ALA":
        return {"CB": atoms["CB"]}
    chis = tuple(chis)
    if aa in ("PHE", "TYR"):
        cg = place(atoms["N"], atoms["CA"], atoms["CB"], 1.502, 113.8, chis[0])
        cd1 = place(atoms["CA"], atoms["CB"], cg, 1.390, 120.0, chis[1] if len(chis) > 1 else 90.0)
        built = _ring_from_cd1(atoms["CB"], cg, cd1, with_oh=(aa == "TYR"))
        built["CB"] = atoms["CB"]
        return built
    steps = STEPS.get(aa)
    if not steps:
        return {"CB": atoms["CB"]}
    for step in steps:
        offset = step.offset
        if abs(abs(offset) - SECOND_BRANCH) < 1e-6:
            offset = np.sign(offset) * branch if offset != 0 else branch
        if step.chi >= 0:
            if step.chi >= len(chis):
                continue
            dih = chis[step.chi] + offset
        else:
            dih = offset
        atoms[step.name] = place(atoms[step.a], atoms[step.b], atoms[step.c], step.length, step.angle, dih)
    return {name: xyz for name, xyz in atoms.items() if name not in ("N", "CA", "C")}


def measure_chis(aa, atom_map):
    aa = canon(aa)
    defs = CHI_ATOMS.get(aa, [])
    chis = []
    for names in defs:
        if any(name not in atom_map for name in names):
            return None
        chis.append(dihedral(*(atom_map[name] for name in names)))
    return tuple(chis)


def reconstruction_rmsd(aa, atom_map, branch=SECOND_BRANCH):
    needed = ("N", "CA", "C")
    if any(name not in atom_map for name in needed):
        return None
    aa = canon(aa)
    if aa in ("GLY", "ALA"):
        if "CB" not in atom_map or aa == "GLY":
            return 0.0
        built = build_cb(atom_map["N"], atom_map["CA"], atom_map["C"])
        return float(np.linalg.norm(built - atom_map["CB"]))
    chis = measure_chis(aa, atom_map)
    if not chis:
        return None
    built = build_sidechain(aa, atom_map["N"], atom_map["CA"], atom_map["C"], chis, branch=branch)
    errs = []
    for name, xyz in built.items():
        if name in atom_map and name not in BACKBONE:
            errs.append(np.sum((xyz - atom_map[name]) ** 2))
    if not errs:
        return None
    return float(np.sqrt(np.mean(errs)))


def element_params(element):
    return LJ.get(element, LJ["X"])


def ligand_charge(element):
    if element == "O":
        return -0.40
    if element == "N":
        return -0.35
    if element == "S":
        return -0.15
    if element in ("F", "CL", "BR", "I"):
        return -0.15
    return 0.0


def sidechain_charge(aa, name):
    return ATOM_CHARGE.get((aa, name), 0.0)


def score_sidechain(aa, atoms, lig_xyz, lig_elem, prot_xyz, prot_elem, prot_q):
    """Score one side chain against the ligand and nearby protein atoms.

    Returns repulsive and attractive Lennard-Jones sums, a Coulomb sum,
    clash and heavy-atom contact counts, a hydrogen-bond proxy count, and
    the closest ligand approach. Empty side chains score as zeros.
    """
    empty = {
        "rep_lig": 0.0, "att_lig": 0.0, "elec_lig": 0.0,
        "rep_prot": 0.0, "att_prot": 0.0,
        "clash": 0.0, "contact": 0.0, "hbond": 0.0, "min_dist": 8.0,
    }
    if not atoms:
        return empty
    aa = canon(aa)
    names = list(atoms.keys())
    xyz = np.vstack([atoms[name] for name in names])
    elem = np.array([_atom_element(name) for name in names])
    q = np.array([sidechain_charge(aa, name) for name in names], dtype=np.float64)
    donor = np.array([(aa, name) in DONORS for name in names])
    acceptor = np.array([(aa, name) in ACCEPTORS for name in names])

    out = dict(empty)
    if len(lig_xyz):
        part = _pair_terms(xyz, elem, q, donor, acceptor, lig_xyz, lig_elem, None, ligand=True)
        out["rep_lig"] = part["rep"]
        out["att_lig"] = part["att"]
        out["elec_lig"] = part["elec"]
        out["clash"] = part["clash"]
        out["contact"] = part["contact"]
        out["hbond"] = part["hbond"]
        out["min_dist"] = part["min_dist"]
    if len(prot_xyz):
        part = _pair_terms(xyz, elem, q, donor, acceptor, prot_xyz, prot_elem, prot_q, ligand=False)
        out["rep_prot"] = part["rep"]
        out["att_prot"] = part["att"]
    out["total"] = out["rep_lig"] + out["att_lig"] + out["elec_lig"] + out["rep_prot"] + out["att_prot"]
    return out


def _atom_element(name):
    if name.startswith(("OG", "OD", "OE", "OH")):
        return "O"
    if name.startswith(("ND", "NE", "NH", "NZ")):
        return "N"
    if name.startswith("SG") or name.startswith("SD"):
        return "S"
    return "C"


def _pair_terms(xyz, elem, q, donor, acceptor, env_xyz, env_elem, env_q, ligand):
    diff = xyz[:, None, :] - env_xyz[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    dist = np.maximum(dist, 0.8)
    ri = np.array([element_params(e)[0] for e in elem], dtype=np.float64)
    ei = np.array([element_params(e)[1] for e in elem], dtype=np.float64)
    rj = np.array([element_params(e)[0] for e in env_elem], dtype=np.float64)
    ej = np.array([element_params(e)[1] for e in env_elem], dtype=np.float64)
    rmin = ri[:, None] + rj[None, :]
    eps = np.sqrt(ei[:, None] * ej[None, :])
    use = dist < 8.0
    r_use = np.maximum(dist, 0.85 * rmin)
    sr = rmin / r_use
    lj = eps * (sr ** 12 - 2.0 * sr ** 6)
    lj = np.where(use, np.clip(lj, -0.6, 8.0), 0.0)
    rep = float(np.sum(np.maximum(lj, 0.0)))
    att = float(np.sum(np.minimum(lj, 0.0)))
    if env_q is None:
        env_charge = np.array([ligand_charge(e) for e in env_elem], dtype=np.float64)
    else:
        env_charge = np.asarray(env_q, dtype=np.float64)
    elec_pairs = 332.0 * q[:, None] * env_charge[None, :] / (4.0 * dist ** 2)
    elec_pairs = np.where(use, np.clip(elec_pairs, -3.0, 3.0), 0.0)
    clash = float(np.sum((dist < 0.80 * rmin) & use))
    contact = float(np.sum((dist < 4.5) & use)) if ligand else 0.0
    polar_env = np.isin(env_elem, ["N", "O"])
    hbond = 0.0
    if ligand and np.any(polar_env):
        close = (dist < 3.3) & polar_env[None, :]
        polar_side = donor | acceptor
        hbond = float(np.sum(close[polar_side])) if np.any(polar_side) else 0.0
    finite = dist[np.isfinite(dist)]
    min_dist = float(finite.min()) if len(finite) else 8.0
    return {
        "rep": rep, "att": att, "elec": float(np.sum(elec_pairs)),
        "clash": clash, "contact": contact, "hbond": hbond, "min_dist": min_dist,
    }


# Temperatures are in the same arbitrary units as the softened Lennard-Jones
# score. They are a fixed basis, not values chosen on the test set.
TEMPERATURES = (0.5, 1.0, 2.0, 4.0)


def _temp_tag(temperature):
    return f"{float(temperature):.1f}".replace(".", "p")


def ensemble_feature_names():
    names = []
    for temperature in TEMPERATURES:
        tag = _temp_tag(temperature)
        names.extend([f"d_lig_{tag}", f"d_tot_{tag}"])
    names.extend(["d_lig_min", "d_tot_min", "d_hb_ens", "d_ct_ens", "d_att_ens"])
    return names


def free_energy(energies, temperature):
    values = np.asarray(list(energies), dtype=np.float64)
    if values.size == 0:
        return 0.0
    shift = float(values.min())
    log_z = np.log(np.sum(np.exp(-(values - shift) / float(temperature))))
    return float(shift - float(temperature) * log_z)


def boltzmann_mean(observables, energies, temperature):
    obs = np.asarray(list(observables), dtype=np.float64)
    values = np.asarray(list(energies), dtype=np.float64)
    if values.size == 0:
        return 0.0
    shift = float(values.min())
    weights = np.exp(-(values - shift) / float(temperature))
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    return float(np.dot(weights, obs) / total)


def ensemble_state(aa, n, ca, c, lig_xyz, lig_elem, prot_xyz, prot_elem, prot_q, branch=SECOND_BRANCH):
    """Free energy of the staggered rotamer ensemble on a fixed backbone.

    The ligand term excludes the protein packing energy. Expectations use
    Boltzmann weights of the total energy at temperature 1.
    """
    aa = canon(aa)
    rotamers = ROTAMERS.get(aa, [()])
    ligand_energies = []
    total_energies = []
    hbonds = []
    contacts = []
    attractions = []
    if aa != "GLY":
        for chis in rotamers:
            try:
                atoms = build_sidechain(aa, n, ca, c, chis, branch=branch)
            except Exception:
                continue
            if not atoms:
                continue
            scored = score_sidechain(aa, atoms, lig_xyz, lig_elem, prot_xyz, prot_elem, prot_q)
            ligand_energies.append(scored["rep_lig"] + scored["att_lig"] + scored["elec_lig"])
            total_energies.append(scored["total"])
            hbonds.append(scored["hbond"])
            contacts.append(scored["contact"])
            attractions.append(scored["att_lig"])
    state = {}
    for temperature in TEMPERATURES:
        tag = _temp_tag(temperature)
        state[f"lig_{tag}"] = free_energy(ligand_energies, temperature)
        state[f"tot_{tag}"] = free_energy(total_energies, temperature)
    state["lig_min"] = float(min(ligand_energies)) if ligand_energies else 0.0
    state["tot_min"] = float(min(total_energies)) if total_energies else 0.0
    state["hb_ens"] = boltzmann_mean(hbonds, total_energies, 1.0)
    state["ct_ens"] = boltzmann_mean(contacts, total_energies, 1.0)
    state["att_ens"] = boltzmann_mean(attractions, total_energies, 1.0)
    return state


def best_rotamer(aa, n, ca, c, lig_xyz, lig_elem, prot_xyz, prot_elem, prot_q, branch=SECOND_BRANCH):
    """Pack the amino acid. Returns the score dict of the lowest total energy."""
    aa = canon(aa)
    rotamers = ROTAMERS.get(aa, [()])
    if aa == "GLY" or not rotamers:
        return score_sidechain(aa, {}, lig_xyz, lig_elem, prot_xyz, prot_elem, prot_q), 0
    best = None
    best_total = None
    built = 0
    for chis in rotamers:
        try:
            atoms = build_sidechain(aa, n, ca, c, chis, branch=branch)
        except Exception:
            continue
        built += 1
        scored = score_sidechain(aa, atoms, lig_xyz, lig_elem, prot_xyz, prot_elem, prot_q)
        total = scored["total"]
        if best_total is None or total < best_total:
            best_total = total
            best = scored
    if best is None:
        best = score_sidechain(aa, {}, lig_xyz, lig_elem, prot_xyz, prot_elem, prot_q)
    return best, built
