#!/usr/bin/env python3
"""Terminal viewer for PyMOL session files (.pse).

Draws what the session shows -- cartoon (helices as spiral ribbons, sheets as
flat arrows, loops as thin tubes), sticks and lines -- in the session's own
colours, starting from the camera angle the session was saved with.

Usage:
    python3 pseview.py analysis.pse
    python3 pseview.py analysis.pse --png view.png    # save a picture instead

Keys: arrows/WASD rotate, +/- zoom, space spin, b braille/blocks, f fit whole molecule,
      n next object, r reset to the session's view, q quit.

Needs numpy. A .pse file is a Python pickle; it is loaded with a restricted
unpickler that refuses to run code from the file.
"""
import argparse
import math
import os
import pickle
import select
import shutil
import struct
import sys
import zlib

import numpy as np

# ---------------------------------------------------------------- loading


class _SessionStorage:
    """Stand-in for pymol.Session_Storage, an empty container class."""


def _latin1_bytes(text, encoding):
    # Protocol-2 pickles rebuild byte strings with _codecs.encode(text, "latin1")
    if encoding != "latin1":
        raise pickle.UnpicklingError(f"unexpected encoding {encoding!r}")
    return text.encode("latin1")


class SafeUnpickler(pickle.Unpickler):
    ALLOWED = {("pymol", "Session_Storage"): _SessionStorage, ("_codecs", "encode"): _latin1_bytes}

    def find_class(self, module, name):
        try:
            return self.ALLOWED[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name} from the session") from None


# Atom record fields (PyMOL 1.8+ session format)
CHAIN, RESI, SEGI, NAME, SS, VISREP, COLOR = 1, 3, 4, 6, 10, 20, 21
REP_STICKS, REP_CARTOON, REP_LINES = 1 << 0, 1 << 5, 1 << 7
OBJECT_MOLECULE = 1


class Molecule:
    def __init__(self, entry, colours):
        self.name, self.visible = entry[0], bool(entry[2])
        data = entry[5]
        self.atoms, csets = data[7], data[4]
        if not self.atoms or not isinstance(self.atoms[0], list) or not isinstance(self.atoms[0][NAME], str):
            sys.exit(f"{self.name}: unsupported session format. In PyMOL run "
                     "'set pse_binary_dump, 0' and save the session again.")
        cs = next(c for c in csets if c)  # first state
        coords = np.array(cs[2], dtype=float).reshape(-1, 3)
        idx_to_atm = cs[3] if cs[3] is not None else range(len(coords))
        self.xyz = np.full((len(self.atoms), 3), np.nan)
        self.xyz[list(idx_to_atm)] = coords
        self.bonds = [(b[0], b[1]) for b in data[6]]
        self.rgb = np.array([colours.rgb(a[COLOR]) for a in self.atoms])


def load_session(path):
    with open(path, "rb") as f:
        session = SafeUnpickler(f).load()
    colours = Colours(session)
    mols = [Molecule(e, colours) for e in session["names"] if e and e[4] == OBJECT_MOLECULE]
    if not mols:
        sys.exit("No molecule objects in this session.")
    return session, mols


# ---------------------------------------------------------------- colours
# Ported from PyMOL's layer1/Color.cpp (ColorReset), so the colour indices
# stored in a session map to the same RGB values PyMOL shows.

# Named colours, in PyMOL registration order (index = position)
_NAMED = (
    "1 1 1;0 0 0;0 0 1;0 1 0;1 0 0;0 1 1;"
    "1 1 0;1 1 0;1 0 1;1 0.6 0.6;0.5 1 0.5;0.5 0.5 1;"
    "1 0 0.5;1 0.5 0;0.5 1 0;0 1 0.5;0.5 0 1;0 0.5 1;"
    "0.77 0.7 0;0.75 0 0.75;0 0.75 0.75;0.6 0.2 0.2;0.2 0.6 0.2;0.25 0.25 0.65;"
    "0.5 0.5 0.5;0.5 0.5 0.5;0.2 1 0.2;0.2 0.2 1;1 0.3 0.3;0.9 0.9 0.9;"
    "1 0.7 0.2;0.9 0.775 0.25;1 0.2 0.2;0.2 1 0.2;0.3 0.3 1;1 1 0.2;"
    "1 0.87 0.37;1 0.55 0.15;0.1 0.1 1;0.2 0.1 0.9;0.3 0.1 0.8;0.4 0.1 0.7;"
    "0.5 0.1 0.6;0.6 0.1 0.5;0.7 0.1 0.4;0.8 0.1 0.3;0.9 0.1 0.2;1 0.1 0.1;"
    "1 0.65 0.85;0.698 0.13 0.13;0.555 0.222 0.111;0.65 0.32 0.17;0.99 0.82 0.65;1 0.5 1;"
    "1 0.2 0.8;0.1 0.1 0.6;1 1 0.5;0.5 1 1;1 0.5 0.5;0.65 0.9 0.65;"
    "0.6 0.6 0.1;0.6 0.1 0.6;0.1 0.6 0.6;0.75 0.75 1;1 0.8 0.5;0.8 1 1;"
    "0.4 0.7 0.7;0.52 0.75 0;0.7 0.3 0.4;0.72 0.55 0.3;0.55 0.7 0.4;0.55 0.25 0.6;"
    "0.7 0.5 0.5;1 0.42 0.42;1 0.75 0.87;0.25 1 0.75;0.75 1 0.25;0.2 0.5 0.8;"
    "0.85 0.85 1;0.85 0.2 0.5;0.73 0.55 0.52;0.851 1 1;0.8 0.502 1;0.7608 1 0;"
    "1 0.7098 0.7098;0.702 1 1;0.702 0.8902 0.9608;0.6706 0.3608 0.949;0.5412 1 0;0.749 0.651 0.651;"
    "0.9412 0.7843 0.6275;1 0.502 0;0.1216 0.9412 0.1216;0.502 0.8196 0.8902;0.5608 0.251 0.8314;0.2392 1 0;"
    "0.902 0.902 0.902;0.749 0.7608 0.7804;0.651 0.651 0.6706;0.5412 0.6 0.7804;0.6118 0.4784 0.7804;0.8784 0.4 0.2;"
    "0.9412 0.5647 0.6275;0.3137 0.8157 0.3137;0.7843 0.502 0.2;0.4902 0.502 0.6902;0.7608 0.5608 0.5608;0.4 0.5608 0.5608;"
    "0.7412 0.502 0.8902;1 0.6314 0;0.651 0.1608 0.1608;0.3608 0.7216 0.8196;0.4392 0.1804 0.6902;0 1 0;"
    "0.5804 1 1;0.5804 0.8784 0.8784;0.451 0.7608 0.7882;0.3294 0.7098 0.7098;0.2314 0.6196 0.6196;0.1412 0.5608 0.5608;"
    "0.0392 0.4902 0.549;0 0.4118 0.5216;0.7529 0.7529 0.7529;1 0.851 0.5608;0.651 0.4588 0.451;0.4 0.502 0.502;"
    "0.6196 0.3882 0.7098;0.8314 0.4784 0;0.5804 0 0.5804;0.2588 0.6196 0.6902;0.3412 0.0902 0.5608;0 0.7882 0;"
    "0.4392 0.8314 1;1 1 0.7804;0.851 1 0.7804;0.7804 1 0.7804;0.6392 1 0.7804;0.5608 1 0.7804;"
    "0.3804 1 0.7804;0.2706 1 0.7804;0.1882 1 0.7804;0.1216 1 0.7804;0 1 0.6118;0 0.902 0.4588;"
    "0 0.8314 0.3216;0 0.749 0.2196;0 0.6706 0.1412;0.302 0.7608 1;0.302 0.651 1;0.1294 0.5804 0.8392;"
    "0.149 0.4902 0.6706;0.149 0.4 0.5882;0.0902 0.3294 0.5294;0.8157 0.8157 0.8784;1 0.8196 0.1373;0.7216 0.7216 0.8157;"
    "0.651 0.3294 0.302;0.3412 0.349 0.3804;0.6196 0.3098 0.7098;0.6706 0.3608 0;0.4588 0.3098 0.2706;0.2588 0.5098 0.5882;"
    "0.2588 0 0.4;0 0.4902 0;0.4392 0.6706 0.9804;0 0.7294 1;0 0.6314 1;0 0.5608 1;"
    "0 0.502 1;0 0.4196 1;0.3294 0.3608 0.949;0.4706 0.3608 0.8902;0.5412 0.3098 0.8902;0.6314 0.2118 0.8314;"
    "0.702 0.1216 0.8314;0.702 0.1216 0.7294;0.702 0.051 0.651;0.7412 0.051 0.5294;0.7804 0 0.4;0.8 0 0.349;"
    "0.8196 0 0.3098;0.851 0 0.2706;0.8784 0 0.2196;0.902 0 0.1804;0.9216 0 0.149;0.9 0.9 0.9;"
    "0.5 0.5 0.5;0.9 0.9 0.9;"
)
_SPECTRUM_S = [(1.0, 0.0, 1.0),
    (0.5, 0.0, 1.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.5, 1.0),
    (0.0, 1.0, 1.0),
    (0.0, 1.0, 0.5),
    (0.0, 1.0, 0.0),
    (0.5, 1.0, 0.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.5, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 0.0, 0.5),
    (1.0, 0.0, 1.0)]
_SPECTRUM_R = [(1.0, 1.0, 0.0),
    (0.5, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 0.5),
    (0.0, 1.0, 1.0),
    (0.0, 0.5, 1.0),
    (0.0, 0.0, 1.0),
    (0.5, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 0.0, 0.5),
    (1.0, 0.0, 0.0),
    (1.0, 0.5, 0.0),
    (1.0, 1.0, 0.0)]
_SPECTRUM_C = [(1.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 1.0)]
_SPECTRUM_W = [(1.0, 1.0, 0.0),
    (1.0, 1.0, 1.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 0.0),
    (1.0, 1.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
    (1.0, 1.0, 0.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 0.0),
    (1.0, 1.0, 1.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (1.0, 1.0, 0.0),
    (1.0, 1.0, 1.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0)]
_SPECTRUM_O = [(1.0, 0.0, 1.0),
    (0.8, 0.0, 1.0),
    (0.5, 0.0, 1.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.2, 1.0),
    (0.0, 0.5, 1.0),
    (0.0, 0.8, 1.0),
    (0.0, 1.0, 1.0),
    (0.0, 1.0, 0.8),
    (0.0, 1.0, 0.5),
    (0.0, 1.0, 0.2),
    (0.0, 1.0, 0.0),
    (0.2, 1.0, 0.0),
    (0.5, 1.0, 0.0),
    (0.8, 1.0, 0.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.9, 0.0),
    (1.0, 0.75, 0.0),
    (1.0, 0.6, 0.0),
    (1.0, 0.5, 0.0),
    (1.0, 0.4, 0.0),
    (1.0, 0.3, 0.0),
    (1.0, 0.2, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 0.0, 0.5),
    (1.0, 0.0, 0.8),
    (1.0, 0.0, 1.0)]

TRGB_BITS, TRGB_MASK = 0x40000000, 0xC0000000


def _ramp(spectrum, div):
    out = []
    for a in range(1000):
        s = int(a / div)
        f = 1.0 - (a - s * div) / div
        out.append(tuple(f * spectrum[s][i] + (1 - f) * spectrum[s + 1][i] for i in range(3)))
    return out


class Colours:
    def __init__(self, session):
        greys = [(a / 99,) * 3 for a in range(100)]
        t = [tuple(map(float, c.split())) for c in "".join(_NAMED).split(";") if c]
        t += greys
        t += _ramp(_SPECTRUM_S, 83.333333333) + _ramp(_SPECTRUM_R, 83.333333333)
        t += _ramp(_SPECTRUM_C, 83.333333333) + _ramp(_SPECTRUM_W, 41.666666667)
        t += greys + _ramp(_SPECTRUM_O, 35.7143)
        for entry in session.get("colors") or []:  # colours made with set_color
            try:
                index, rgb = int(entry[1]), tuple(float(v) for v in entry[2][:3])
            except (TypeError, ValueError, IndexError):
                continue
            t += [(0.7, 0.7, 0.7)] * (index + 1 - len(t))
            t[index] = rgb
        self.table = t

    def rgb(self, index):
        if index & TRGB_MASK == TRGB_BITS:  # colour stored directly as 0x40RRGGBB
            return ((index >> 16) & 255) / 255, ((index >> 8) & 255) / 255, (index & 255) / 255
        if 0 <= index < len(self.table):
            return self.table[index]
        return (0.7, 0.7, 0.7)


# ---------------------------------------------------------------- geometry
# Every shape is turned into surface points with normals; the renderer
# projects them and keeps the nearest point per pixel.

SPACINGS = (0.3, 0.2, 0.13, 0.08)  # Angstrom between surface points; finer when zoomed in
LOOP_R = 0.3        # loop tube radius
HELIX_W, HELIX_T = 1.35, 0.25   # half-width / half-thickness (PyMOL cartoon_oval_*)
SHEET_W, SHEET_T = 1.4, 0.3     # (PyMOL cartoon_rect_*)
ARROW_W = 2.2       # half-width at the base of a sheet arrowhead
STICK_R = 0.25


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 1e-9)


_section_cache = {}


def _section(kind, a, b, spacing):
    """Cross-section outline as offsets (u along width, v along normal) and outline normals."""
    key = (kind, round(a, 2), round(b, 2), spacing)
    if key not in _section_cache:
        if kind == "ellipse":
            m = max(8, math.ceil(2 * math.pi * max(a, b) / spacing))
            th = np.linspace(0, 2 * math.pi, m, endpoint=False)
            u, v = a * np.cos(th), b * np.sin(th)
            nu, nv = np.cos(th) / a, np.sin(th) / b
        else:  # rectangle
            xs = np.linspace(-a, a, max(2, math.ceil(2 * a / spacing) + 1))
            ys = np.linspace(-b, b, max(2, math.ceil(2 * b / spacing) + 1))
            one, zero = np.ones_like, np.zeros_like
            u = np.concatenate([xs, xs, -a * one(ys), a * one(ys)])
            v = np.concatenate([b * one(xs), -b * one(xs), ys, ys])
            nu = np.concatenate([zero(xs), zero(xs), -one(ys), one(ys)])
            nv = np.concatenate([one(xs), -one(xs), zero(ys), zero(ys)])
        n = np.hypot(nu, nv)
        _section_cache[key] = (u, v, nu / n, nv / n)
    return _section_cache[key]


def cartoon_residues(mol):
    """Residues with a visible cartoon, split into continuous segments."""
    by_res = {}
    for i, a in enumerate(mol.atoms):
        by_res.setdefault((a[SEGI], a[CHAIN], a[RESI]), {})[a[NAME]] = i
    segments, seg, last = [], [], None
    for (segi, chain, _), names in by_res.items():
        ca = names.get("CA")
        if ca is None or not mol.atoms[ca][VISREP] & REP_CARTOON or np.isnan(mol.xyz[ca][0]):
            continue
        ss = mol.atoms[ca][SS]
        res = dict(chain=(segi, chain), ca=ca, c=names.get("C"), o=names.get("O"),
                   ss="H" if ss in ("H", "G", "I") else "S" if ss == "S" else "L")
        if last and (last["chain"] != res["chain"] or np.linalg.norm(mol.xyz[ca] - mol.xyz[last["ca"]]) > 4.6):
            segments.append(seg)
            seg = []
        seg.append(res)
        last = res
    segments.append(seg)
    return [s for s in segments if len(s) >= 2]


def _spline(P, sub):
    """Catmull-Rom curve through P; returns parameter t (in residues) and points."""
    n = len(P)
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    t = np.arange((n - 1) * sub + 1) / sub
    i = np.minimum(t.astype(int), n - 2)
    f = (t - i)[:, None]
    p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
    pts = 0.5 * (2 * p1 + (p2 - p0) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f ** 2
                 + (3 * p1 - p0 - 3 * p2 + p3) * f ** 3)
    return t, pts


def cartoon_segment(mol, seg, out):
    n = len(seg)
    ss = [r["ss"] for r in seg]
    P = mol.xyz[[r["ca"] for r in seg]]
    rgb = mol.rgb[[r["ca"] for r in seg]]

    # Sheets: average out the pleat so strands come out flat and straight
    path = P.copy()
    for i in range(1, n - 1):
        if ss[i] == "S":
            path[i] = 0.25 * P[i - 1] + 0.5 * P[i] + 0.25 * P[i + 1]

    # Per-residue face normal: helix -> points out from the helix axis;
    # sheet -> perpendicular to the sheet (cross of strand direction and C=O)
    N = np.zeros_like(P)
    for i in range(n):
        lo, hi = max(i - 1, 0), min(i + 1, n - 1)
        if ss[i] == "H" and 0 < i < n - 1:
            N[i] = 2 * P[i] - P[i - 1] - P[i + 1]
        elif ss[i] == "S" and seg[i]["c"] is not None and seg[i]["o"] is not None:
            N[i] = np.cross(path[hi] - path[lo], mol.xyz[seg[i]["o"]] - mol.xyz[seg[i]["c"]])
    N = _unit(N)
    for i in range(1, n):  # C=O alternates side each residue: keep sheet normals pointing one way
        if ss[i] == "S" and ss[i - 1] == "S" and N[i] @ N[i - 1] < 0:
            N[i] = -N[i]

    # Strand runs, for arrowheads
    run_end = {}
    i = 0
    while i < n:
        if ss[i] == "S":
            j = i
            while j + 1 < n and ss[j + 1] == "S":
                j += 1
            for k in range(i, j + 1):
                run_end[k] = (i, j)
            i = j + 1
        else:
            i += 1

    t, S = _spline(path, max(4, math.ceil(3.8 / out.spacing)))  # CA-CA is 3.8 A
    T = _unit(np.gradient(S, axis=0))
    # rotation-minimising frame: used for tubes and wherever no normal is known
    ref = np.empty_like(S)
    v = np.cross(T[0], [0, 0, 1] if abs(T[0][2]) < 0.9 else [1, 0, 0])
    for k in range(len(S)):
        v = v - (v @ T[k]) * T[k]
        v = v / np.linalg.norm(v)
        ref[k] = v

    for k in range(len(S)):
        tk = t[k]
        own = int(round(tk))
        r0 = min(int(tk), n - 2)
        f = tk - r0
        kind = ss[own]
        if kind == "S":
            s0, e0 = run_end[own]
            if tk > e0:  # past the arrow tip
                kind = "L"
        if kind == "L":
            u, vv, nu, nv = _section("ellipse", LOOP_R, LOOP_R, out.spacing)
            normal = ref[k]
        else:
            if ss[r0] == ss[r0 + 1] == kind and N[r0].any() and N[r0 + 1].any():
                normal = (1 - f) * N[r0] + f * N[r0 + 1]
            else:
                normal = N[own]
            normal = normal - (normal @ T[k]) * T[k]
            nn = np.linalg.norm(normal)
            normal = normal / nn if nn > 1e-6 else ref[k]
            if kind == "H":
                u, vv, nu, nv = _section("ellipse", HELIX_W, HELIX_T, out.spacing)
            else:
                start = max(e0 - 1.0, s0 - 0.5)
                if tk >= start and e0 > start:
                    a = ARROW_W * (e0 - tk) / (e0 - start) + 0.1
                else:
                    a = SHEET_W
                u, vv, nu, nv = _section("rect", a, SHEET_T, out.spacing)
        width = np.cross(normal, T[k])
        out.add(S[k] + u[:, None] * width + vv[:, None] * normal,
                nu[:, None] * width + nv[:, None] * normal, rgb[own])

    return {"H": ss.count("H"), "S": ss.count("S"), "L": ss.count("L")}


def _fibonacci_sphere(m):
    i = np.arange(m) + 0.5
    phi = np.arccos(1 - 2 * i / m)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)


def bond_geometry(mol, out):
    rep = [a[VISREP] for a in mol.atoms]
    sp = out.spacing
    ring = np.linspace(0, 2 * math.pi, max(8, math.ceil(2 * math.pi * STICK_R / sp)), endpoint=False)
    stick_atoms = set()
    for i, j in mol.bonds:
        x1, x2 = mol.xyz[i], mol.xyz[j]
        if np.isnan(x1[0]) or np.isnan(x2[0]):
            continue
        d = x2 - x1
        length = np.linalg.norm(d)
        ts = (np.arange(max(2, math.ceil(length / sp))) + 0.5) / max(2, math.ceil(length / sp))
        half = ts < 0.5
        if rep[i] & rep[j] & REP_STICKS:
            stick_atoms.update((i, j))
            ax = d / length
            e1 = _unit(np.cross(ax, [0, 0, 1] if abs(ax[2]) < 0.9 else [1, 0, 0]))
            e2 = np.cross(ax, e1)
            rad = np.cos(ring)[:, None] * e1 + np.sin(ring)[:, None] * e2  # (m, 3)
            for sel, colour in ((half, mol.rgb[i]), (~half, mol.rgb[j])):
                centres = x1 + ts[sel, None] * d
                pts = (centres[:, None, :] + STICK_R * rad[None]).reshape(-1, 3)
                out.add(pts, np.tile(rad, (len(centres), 1)), colour)
        elif rep[i] & rep[j] & REP_LINES:
            for sel, colour in ((half, mol.rgb[i]), (~half, mol.rgb[j])):
                out.add(x1 + ts[sel, None] * d, None, colour)
    sphere = _fibonacci_sphere(max(12, math.ceil(4 * math.pi * STICK_R ** 2 / sp ** 2)))
    for i in stick_atoms:  # round joints
        out.add(mol.xyz[i] + STICK_R * sphere, sphere, mol.rgb[i])


class Geometry:
    def __init__(self, spacing):
        self.spacing = spacing
        self._pts, self._nrm, self._rgb = [], [], []

    def add(self, pts, normals, colour):
        """normals=None marks points that are drawn unshaded (lines)."""
        self._pts.append(pts)
        self._nrm.append(np.zeros_like(pts) if normals is None else normals)
        self._rgb.append(np.broadcast_to(colour, pts.shape))

    def finish(self):
        self.pts = np.concatenate(self._pts) if self._pts else np.zeros((0, 3))
        self.nrm = np.concatenate(self._nrm) if self._nrm else np.zeros((0, 3))
        self.rgb = np.concatenate(self._rgb) if self._rgb else np.zeros((0, 3))
        self.flat = ~self.nrm.any(axis=1)
        return self


def build_scene(mols, spacing):
    g, counts = Geometry(spacing), {"H": 0, "S": 0, "L": 0}
    for mol in mols:
        for seg in cartoon_residues(mol):
            for k, v in cartoon_segment(mol, seg, g).items():
                counts[k] += v
        bond_geometry(mol, g)
    return g.finish(), counts


# ---------------------------------------------------------------- rendering

LIGHT = _unit(np.array([-0.45, 0.55, 0.7]))
HALFWAY = _unit(LIGHT + np.array([0.0, 0.0, 1.0]))


OUTLINE_GAP = 2.5  # Angstrom: a pixel this far behind its neighbour is drawn as a dark outline


def render(geom, R, origin, scale_x, scale_y, w, h, bg, chunk=500_000):
    """Points projected orthographically, nearest wins.

    scale_x / scale_y are pixels per Angstrom; they differ when pixels are not square.
    Returns an (h, w, 3) uint8 image and an (h, w) mask of lit pixels: covered,
    and not part of an outline.
    """
    img = np.empty((h, w, 3))
    img[:] = bg
    lit = np.zeros((h, w), bool)
    zbuf = np.full(h * w, -np.inf)
    owner = np.full(h * w, -1)
    # splat each point over kx by ky pixels so the surface has no holes between points
    kx, ky = (max(1, math.ceil(geom.spacing * s)) for s in (scale_x, scale_y))
    dx, dy = (d.ravel() for d in np.meshgrid(np.arange(kx) - kx // 2, np.arange(ky) - ky // 2))
    step = max(1, chunk // len(dx))  # bounded memory for big pictures
    for start in range(0, len(geom.pts), step):
        cam = (geom.pts[start:start + step] - origin) @ R.T
        sx = np.floor(cam[:, 0] * scale_x + w / 2).astype(np.int64)
        sy = np.floor(h / 2 - cam[:, 1] * scale_y).astype(np.int64)
        keep = np.nonzero((sx > -kx) & (sx < w + kx) & (sy > -ky) & (sy < h + ky))[0]
        ids = np.repeat(keep, len(dx))
        px = np.repeat(sx[keep], len(dx)) + np.tile(dx, len(keep))
        py = np.repeat(sy[keep], len(dx)) + np.tile(dy, len(keep))
        inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        ids, pix = ids[inside], (py * w + px)[inside]
        z = cam[ids, 2]
        order = np.lexsort((-z, pix))  # by pixel, nearest first
        ids, pix, z = ids[order], pix[order], z[order]
        first = np.r_[True, pix[1:] != pix[:-1]]
        ids, pix, z = ids[first], pix[first], z[first]
        better = z > zbuf[pix]
        zbuf[pix[better]] = z[better]
        owner[pix[better]] = ids[better] + start
    pix = np.nonzero(owner >= 0)[0]
    if not len(pix):
        return (img * 255).astype(np.uint8), lit
    sel = owner[pix]

    n = geom.nrm[sel] @ R.T
    diffuse = np.abs(n @ LIGHT)
    spec = np.abs(n @ HALFWAY) ** 40
    col = geom.rgb[sel] * (0.3 + 0.7 * diffuse)[:, None] + 0.3 * spec[:, None]
    flat = geom.flat[sel]
    col[flat] = geom.rgb[sel][flat]
    zs = zbuf[pix]  # depth cue: fade the far side toward the background, like PyMOL
    fog = 0.45 * (zs.max() - zs) / max(zs.max() - zs.min(), 1e-6)
    col = col * (1 - fog[:, None]) + np.asarray(bg) * fog[:, None]
    img.reshape(-1, 3)[pix] = col

    # Outlines: darken pixels that sit well behind a neighbouring pixel, so a ribbon
    # passing in front of another part stays separated from it at low resolution
    zimg = zbuf.reshape(h, w)
    nearest_nb = np.full((h, w), -np.inf)
    nearest_nb[1:] = np.maximum(nearest_nb[1:], zimg[:-1])
    nearest_nb[:-1] = np.maximum(nearest_nb[:-1], zimg[1:])
    nearest_nb[:, 1:] = np.maximum(nearest_nb[:, 1:], zimg[:, :-1])
    nearest_nb[:, :-1] = np.maximum(nearest_nb[:, :-1], zimg[:, 1:])
    covered = np.isfinite(zimg)
    edge = covered & (nearest_nb - np.where(covered, zimg, 0) > OUTLINE_GAP)
    img[edge] = img[edge] * 0.25 + np.asarray(bg) * 0.75
    lit = covered & ~edge
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8), lit


_CUBE = np.array([0, 95, 135, 175, 215, 255])
# Quadrant characters, indexed by which pixels of a 2x2 cell take the foreground
# colour: bit 0 top-left, bit 1 top-right, bit 2 bottom-left, bit 3 bottom-right
QUADRANTS = " ▘▝▀▖▌▞▛▗▚▐▜▄▙▟█"
# Every way to split 4 pixels into two groups (bottom-right always in the background group)
_SPLITS = np.array([[(m >> i) & 1 for i in range(4)] for m in range(8)], float)[:, None, None, :, None]


def _two_colour_cells(img):
    """For each 2x2 block, the split into two colours that matches it best.

    Returns the split (index into QUADRANTS), foreground and background colours.
    """
    h, w = img.shape[0] // 2, img.shape[1] // 2
    blk = img[:2 * h, :2 * w].astype(float).reshape(h, 2, w, 2, 3).transpose(0, 2, 1, 3, 4).reshape(h, w, 4, 3)
    n_fg = _SPLITS.sum(3)
    fg = (blk * _SPLITS).sum(3) / np.maximum(n_fg, 1)          # (8, h, w, 3)
    bg = (blk * (1 - _SPLITS)).sum(3) / (4 - n_fg)
    approx = np.where(_SPLITS > 0, fg[:, :, :, None], bg[:, :, :, None])
    best = ((blk - approx) ** 2).sum((3, 4)).argmin(0)        # ties go to split 0: one flat colour
    i, j = np.indices(best.shape)
    fg, bg = fg[best, i, j], bg[best, i, j]
    fg[best == 0] = bg[best == 0]
    return best, (fg + 0.5).astype(np.uint8), (bg + 0.5).astype(np.uint8)


def _colour_codes(rgb, truecolor):
    if truecolor:
        return [[f"{r};{g};{b}" for r, g, b in row] for row in rgb.tolist()], "38;2;", "48;2;"
    q = np.abs(rgb[..., None].astype(int) - _CUBE).argmin(-1)
    return (16 + 36 * q[..., 0] + 6 * q[..., 1] + q[..., 2]).tolist(), "38;5;", "48;5;"


# Braille dot bit for each pixel of a 2-wide, 4-tall cell, as [row][column]
_BRAILLE_BITS = np.array([[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]])


def to_braille(img, lit, bg, truecolor):
    """2x4 image pixels per terminal cell: one braille dot per lit pixel, in the cell's mean colour."""
    h, w = img.shape[0] // 4, img.shape[1] // 2
    cell_lit = lit[:4 * h, :2 * w].reshape(h, 4, w, 2).transpose(0, 2, 1, 3)       # (h, w, 4, 2)
    cell_rgb = img[:4 * h, :2 * w].astype(float).reshape(h, 4, w, 2, 3).transpose(0, 2, 1, 3, 4)
    codes = (cell_lit * _BRAILLE_BITS).sum((2, 3))
    n = cell_lit.sum((2, 3))
    mean = (cell_rgb * cell_lit[..., None]).sum((2, 3)) / np.maximum(n, 1)[..., None]
    fg = np.clip(mean * 1.3, 0, 255).astype(np.uint8)  # dots cover less of the cell than blocks: brighten
    fgs, fg_fmt, bg_fmt = _colour_codes(fg, truecolor)
    bg_code = _colour_codes((np.array([[bg]]) * 255 + 0.5).astype(np.uint8), truecolor)[0][0][0]
    lines = []
    for c_row, f_row in zip(codes.tolist(), fgs):
        parts, last = [f"\033[{bg_fmt}{bg_code}m"], None
        for c, f in zip(c_row, f_row):
            if c and f != last:
                parts.append(f"\033[{fg_fmt}{f}m")
                last = f
            parts.append(chr(0x2800 + c) if c else " ")
        lines.append("".join(parts) + "\033[0m")
    return lines


def to_ansi(img, truecolor):
    """2x2 image pixels per terminal cell, drawn as a quadrant character in two colours."""
    split, fg, bg = _two_colour_cells(img)
    fgs, fg_fmt, bg_fmt = _colour_codes(fg, truecolor)
    bgs = _colour_codes(bg, truecolor)[0]
    lines = []
    for m_row, f_row, b_row in zip(split.tolist(), fgs, bgs):
        parts, last_f, last_b = [], None, None
        for m, f, b in zip(m_row, f_row, b_row):
            if b != last_b or (m and f != last_f):
                parts.append(f"\033[{fg_fmt}{f};{bg_fmt}{b}m")
                last_f, last_b = f, b
            parts.append(QUADRANTS[m])
        lines.append("".join(parts) + "\033[0m")
    return lines


def to_ascii(img, bg):
    """Monochrome fallback for output that is not a terminal (no colour codes)."""
    ramp = ".:-=+*#%@"
    bg8 = (np.asarray(bg) * 255 + 0.5).astype(np.uint8)
    h = img.shape[0] - img.shape[0] % 2
    pair = img[:h].reshape(h // 2, 2, img.shape[1], 3)
    fg = (pair != bg8).any(-1).any(1)
    lum = (pair.astype(float) @ [0.3, 0.59, 0.11]).max(1) / 255
    idx = np.minimum((lum * len(ramp)).astype(int), len(ramp) - 1)
    return ["".join(ramp[i] if f else " " for i, f in zip(ri, rf)).rstrip() for ri, rf in zip(idx, fg)]


def write_png(path, img):
    h, w = img.shape[:2]
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


# ---------------------------------------------------------------- viewer


def _rot(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class Viewer:
    def __init__(self, session, mols, path):
        self.path = path
        self.mols = mols
        shown = [m for m in mols if m.visible] or mols[:1]
        # 'n' cycles: what the session shows, then each object on its own
        self.choices = [(" + ".join(m.name for m in shown), shown)] + [(m.name, [m]) for m in mols]
        self.choice, self._cache, self._fit_pending = 0, {}, False
        settings = {s[0]: s[2] for s in session.get("settings") or []}
        bg = settings.get(6, 0)  # bg_rgb, stored as 0xRRGGBB or as an [r, g, b] list
        self.bg = tuple(bg) if isinstance(bg, (list, tuple)) else (
            ((bg >> 16) & 255) / 255, ((bg >> 8) & 255) / 255, (bg & 255) / 255)
        view = session.get("view")
        if view and len(view) >= 22:
            self.R0 = np.array(view[:16]).reshape(4, 4).T[:3, :3]  # column-major, model -> camera
            self.origin0 = np.array(view[19:22])
            fov = math.radians(settings.get(152, 20.0))
            self.half_height0 = abs(view[18]) * math.tan(fov / 2)  # Angstrom visible above centre
            self.reset()
        else:
            self.R0, self.R = np.eye(3), np.eye(3)
            self.fit(4, 3)  # no saved camera: fit a 4:3 screen
            self.origin0, self.half_height0 = self.origin, self.half_height

    def reset(self):
        """Back to the camera saved in the session."""
        self.R, self.zoom = self.R0.copy(), 1.0
        self.origin, self.half_height = self.origin0, self.half_height0

    def fit(self, w=None, h=None, pixel_aspect=1.0):
        """Centre on the shown molecules and zoom so they fill the screen at the current angle.

        Without a screen size, the fit is deferred to the next image() call.
        """
        if w is None:
            self._fit_pending = True
            return
        self._fit_pending = False
        pts = self.geometry()[0].pts
        centre = (pts.min(0) + pts.max(0)) / 2
        with np.errstate(all="ignore"):  # spurious macOS Accelerate matmul warnings
            cam = (pts - centre) @ self.R.T
        lo, hi = cam[:, :2].min(0), cam[:, :2].max(0)
        self.origin = centre + self.R.T @ np.array([*(lo + hi) / 2, 0.0])
        ext_x, ext_y = np.maximum(hi - lo, 1.0)
        scale_y = 0.94 * min(h / ext_y, w * pixel_aspect / ext_x)
        self.zoom = 1.0
        self.half_height = min(h, w * pixel_aspect) / 2 / scale_y

    def geometry(self, spacing=SPACINGS[0]):
        key = (self.choice, spacing)
        if key not in self._cache:
            self._cache[key] = build_scene(self.choices[self.choice][1], spacing)
        return self._cache[key]

    def image(self, w, h, pixel_aspect=1.0):
        """Render w x h pixels; pixel_aspect is pixel width / height (0.5 for quadrant cells)."""
        if self._fit_pending:
            self.fit(w, h, pixel_aspect)
        # half_height is vertical; also fit the width
        scale_y = self.zoom * min(h, w * pixel_aspect) / 2 / self.half_height
        scale_x = scale_y / pixel_aspect
        # use the coarsest point spacing that still splats at most 2 pixels wide
        spacing = next((s for s in SPACINGS if s * max(scale_x, scale_y) <= 2), SPACINGS[-1])
        with np.errstate(all="ignore"):  # numpy on macOS Accelerate raises spurious matmul warnings
            return render(self.geometry(spacing)[0], self.R, self.origin, scale_x, scale_y, w, h, self.bg)

    def status(self):
        c = self.geometry()[1]
        return (f" {self.choices[self.choice][0]}   helix {c['H']} · sheet {c['S']} · loop {c['L']} residues"
                "   arrows rotate  +/- zoom  space spin  b braille/blocks  f fit  n next object  r reset  q quit")

    def run(self):
        import termios
        import tty
        fd, out = sys.stdin.fileno(), sys.stdout
        old = termios.tcgetattr(fd)
        truecolor = os.environ.get("COLORTERM") in ("truecolor", "24bit")
        bar = "\033[48;2;70;70;70;38;2;225;225;225m" if truecolor else "\033[48;5;238;38;5;253m"  # grey
        spin, dirty, size, braille = False, True, None, True
        try:
            tty.setcbreak(fd)
            out.write("\033[?1049h\033[?25l")  # alternate screen, hide cursor
            while True:
                cols, rows = shutil.get_terminal_size()
                if dirty or spin or (cols, rows) != size:
                    size = (cols, rows)
                    if braille:  # braille dots: 2x4 per cell, square pixels
                        img, lit = self.image(cols * 2, (rows - 1) * 4)
                        lines = to_braille(img, lit, self.bg, truecolor)
                    else:  # quadrant blocks: 2x2 per cell, pixels twice as tall as wide
                        lines = to_ansi(self.image(cols * 2, (rows - 1) * 2, pixel_aspect=0.5)[0], truecolor)
                    status = self.status()[:cols].ljust(cols)
                    out.write("\033[H" + "\n".join(lines) + "\n" + bar + status + "\033[0m")
                    out.flush()
                    dirty = False
                if not select.select([fd], [], [], 0.02 if spin else 0.3)[0]:
                    if spin:
                        self.R = _rot("y", 0.05) @ self.R
                    continue
                keys = os.read(fd, 64).decode(errors="ignore")
                keys = (keys.replace("\033[A", "w").replace("\033[B", "s")
                        .replace("\033[C", "d").replace("\033[D", "a"))
                for key in keys:
                    dirty = True
                    if key in "qQ":
                        return
                    elif key in "aA":
                        self.R = _rot("y", -0.15) @ self.R
                    elif key in "dD":
                        self.R = _rot("y", 0.15) @ self.R
                    elif key in "wW":
                        self.R = _rot("x", -0.15) @ self.R
                    elif key in "sS":
                        self.R = _rot("x", 0.15) @ self.R
                    elif key in "+=":
                        self.zoom *= 1.2
                    elif key in "-_":
                        self.zoom /= 1.2
                    elif key == " ":
                        spin = not spin
                    elif key in "bB":
                        braille = not braille
                    elif key in "fF":
                        self.fit()
                    elif key in "rR":
                        self.reset()
                    elif key in "nN":
                        self.choice = (self.choice + 1) % len(self.choices)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            out.write("\033[0m\033[?25h\033[?1049l")
            out.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", help="PyMOL .pse file")
    p.add_argument("--png", metavar="FILE", help="save a picture of the starting view and exit")
    p.add_argument("--png-size", default="1200x800", help="picture size in pixels for --png (WxH)")
    p.add_argument("--size", default="160x50", help="text size for non-terminal output (COLSxROWS)")
    p.add_argument("--object", help="show only this object instead of what the session shows")
    args = p.parse_args()

    session, mols = load_session(args.session)
    viewer = Viewer(session, mols, args.session)
    if args.object:
        names = [c[0] for c in viewer.choices]
        if args.object not in names:
            sys.exit(f"No object {args.object!r}. Objects: {', '.join(m.name for m in mols)}")
        viewer.choice = names.index(args.object)

    if args.png:
        w, h = (int(v) for v in args.png_size.lower().split("x"))
        write_png(args.png, viewer.image(w, h)[0])
        print(f"wrote {args.png}")
        return
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        cols, rows = (int(v) for v in args.size.lower().split("x"))
        print("\n".join(line for line in to_ascii(viewer.image(cols, rows * 2)[0], viewer.bg) if line))
        print(viewer.status())
        return
    viewer.run()


if __name__ == "__main__":
    main()
