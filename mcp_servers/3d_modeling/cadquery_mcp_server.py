"""MCP server for parametric 3D-printable CAD with CadQuery.

Run with: python cadquery_mcp_server.py
Transport: stdio by default; use CQ_MCP_TRANSPORT=sse for an HTTP/SSE server.
"""
from __future__ import annotations

import json
import math
import os
import traceback
from pathlib import Path
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - gives an actionable runtime error
    raise SystemExit("Install dependencies first: pip install -e .") from exc

try:
    import cadquery as cq
    from cadquery import exporters, importers
except ImportError:
    cq = None
    exporters = importers = None

mcp = FastMCP("cadquery-3d-printing")
OBJECTS: dict[str, Any] = {}
META: dict[str, dict[str, Any]] = {}
WORKDIR = Path(os.environ.get("CQ_WORKDIR", os.getcwd())).expanduser().resolve()
WORKDIR.mkdir(parents=True, exist_ok=True)


def _require_cq() -> None:
    if cq is None:
        raise RuntimeError("CadQuery não está instalado. Execute: pip install -e .")


def _obj(name: str) -> Any:
    if name not in OBJECTS:
        raise KeyError(f"Objeto inexistente: {name}. Use list_objects primeiro.")
    return OBJECTS[name]


def _shape(value: Any) -> Any:
    if hasattr(value, "val"):
        return value.val()
    return value


def _store(name: str, value: Any, operation: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    OBJECTS[name] = value
    META[name] = {"operation": operation, "params": params or {}}
    return inspect_object(name)


def _result(name: str) -> dict[str, Any]:
    return {"name": name, "inspection": inspect_object(name)}


def _numbers(value: Any) -> list[float]:
    return [float(x) for x in value]


def inspect_object(name: str) -> dict[str, Any]:
    _require_cq()
    shape = _shape(_obj(name))
    bb = shape.BoundingBox()
    data: dict[str, Any] = {
        "name": name,
        "operation": META.get(name, {}).get("operation"),
        "params": META.get(name, {}).get("params", {}),
        "bounding_box_mm": {
            "min": [bb.xmin, bb.ymin, bb.zmin],
            "max": [bb.xmax, bb.ymax, bb.zmax],
            "size": [bb.xlen, bb.ylen, bb.zlen],
        },
        "solid_count": len(shape.Solids()),
        "face_count": len(shape.Faces()),
        "edge_count": len(shape.Edges()),
    }
    try:
        data["volume_mm3"] = float(shape.Volume())
    except Exception:
        pass
    try:
        data["area_mm2"] = float(shape.Area())
    except Exception:
        pass
    return data


@mcp.tool()
def health_check() -> dict[str, Any]:
    """Return server/runtime status and the active work directory."""
    return {"ok": True, "cadquery_available": cq is not None, "object_count": len(OBJECTS), "workdir": str(WORKDIR)}


@mcp.tool()
def list_objects() -> list[dict[str, Any]]:
    """List all named CAD objects and their compact metadata."""
    return [{"name": n, **META.get(n, {})} for n in OBJECTS]


@mcp.tool()
def clear_document(name: str | None = None) -> dict[str, Any]:
    """Delete one named object, or clear the whole in-memory document when name is omitted."""
    if name:
        OBJECTS.pop(name, None); META.pop(name, None)
        return {"deleted": name, "remaining": len(OBJECTS)}
    count = len(OBJECTS); OBJECTS.clear(); META.clear()
    return {"deleted_count": count, "remaining": 0}


@mcp.tool()
def create_primitive(name: str, kind: str, dimensions: dict[str, float], centered: bool = True) -> dict[str, Any]:
    """Create box, cylinder, sphere, cone, or torus. Dimensions are in millimetres."""
    _require_cq(); k = kind.lower()
    if k == "box":
        v = cq.Workplane("XY").box(dimensions["length"], dimensions["width"], dimensions["height"], centered= centered)
    elif k == "cylinder":
        v = cq.Workplane("XY").cylinder(dimensions["height"], dimensions["radius"], centered=centered)
    elif k == "sphere":
        v = cq.Workplane("XY").sphere(dimensions["radius"])
    elif k == "cone":
        v = cq.Workplane("XY").transformed(offset=(0, 0, -dimensions["height"] / 2 if centered else 0)).cone(dimensions["height"], dimensions["radius1"], dimensions["radius2"])
    elif k == "torus":
        v = cq.Workplane("XY").torus(dimensions["major_radius"], dimensions["minor_radius"])
    else:
        raise ValueError("kind deve ser box, cylinder, sphere, cone ou torus")
    return _store(name, v, "primitive", {"kind": k, "dimensions": dimensions, "centered": centered})


@mcp.tool()
def create_profile_extrusion(name: str, profile: str, height: float, parameters: dict[str, Any]) -> dict[str, Any]:
    """Create a solid from a rectangle, circle, regular polygon, or explicit polygon points."""
    _require_cq(); p = profile.lower(); wp = cq.Workplane("XY")
    if p == "rectangle":
        v = wp.rect(parameters["width"], parameters["height"]).extrude(height)
    elif p == "circle":
        v = wp.circle(parameters["radius"]).extrude(height)
    elif p == "polygon":
        v = wp.polygon(int(parameters["sides"]), parameters["diameter"]).extrude(height)
    elif p == "points":
        pts = [tuple(x) for x in parameters["points"]]
        v = wp.polyline(pts + [pts[0]]).close().extrude(height)
    else:
        raise ValueError("profile deve ser rectangle, circle, polygon ou points")
    return _store(name, v, "profile_extrusion", {"profile": p, "height": height, "parameters": parameters})


@mcp.tool()
def execute_cadquery_script(name: str, script: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute a trusted CadQuery script. The script must assign the final Workplane/Shape to `result`."""
    _require_cq()
    env = {"cq": cq, "math": math, "result": None, **(variables or {})}
    exec(script, {"__builtins__": __builtins__}, env)  # trusted local MCP by design
    if env.get("result") is None:
        raise ValueError("O script precisa atribuir o sólido final à variável `result`.")
    return _store(name, env["result"], "script", {"script": script, "variables": variables or {}})


@mcp.tool()
def boolean_operation(name: str, left: str, right: str, operation: str) -> dict[str, Any]:
    """Union, cut, or intersect two named solids."""
    _require_cq(); a, b = _obj(left), _obj(right); op = operation.lower()
    if op == "union": v = a.union(b)
    elif op == "cut": v = a.cut(b)
    elif op in ("intersect", "intersection"): v = a.intersect(b)
    else: raise ValueError("operation deve ser union, cut ou intersect")
    return _store(name, v, "boolean", {"left": left, "right": right, "operation": op})


@mcp.tool()
def transform_object(name: str, source: str, translation: list[float] | None = None, rotation: list[float] | None = None) -> dict[str, Any]:
    """Copy and transform an object. rotation is [x_deg, y_deg, z_deg] about the origin."""
    _require_cq(); v = _obj(source); translation = translation or [0, 0, 0]; rotation = rotation or [0, 0, 0]
    v = v.translate(tuple(translation))
    if any(rotation):
        v = v.rotate((0, 0, 0), (1, 0, 0), rotation[0]).rotate((0, 0, 0), (0, 1, 0), rotation[1]).rotate((0, 0, 0), (0, 0, 1), rotation[2])
    return _store(name, v, "transform", {"source": source, "translation": translation, "rotation": rotation})


@mcp.tool()
def edge_treatment(name: str, source: str, selector: str, size: float, treatment: str = "fillet") -> dict[str, Any]:
    """Apply fillet or chamfer to selected edges; selector examples: |Z, >Z, <X, or all."""
    _require_cq(); v = _obj(source); selected = v.edges() if selector == "all" else v.edges(selector)
    t = treatment.lower(); v = selected.fillet(size) if t == "fillet" else selected.chamfer(size) if t == "chamfer" else None
    if v is None: raise ValueError("treatment deve ser fillet ou chamfer")
    return _store(name, v, "edge_treatment", {"source": source, "selector": selector, "size": size, "treatment": t})


@mcp.tool()
def shell_object(name: str, source: str, thickness: float, faces_to_remove: str = ">Z") -> dict[str, Any]:
    """Hollow a solid by removing faces matching a CadQuery selector."""
    _require_cq(); v = _obj(source).faces(faces_to_remove).shell(-abs(thickness))
    return _store(name, v, "shell", {"source": source, "thickness": thickness, "faces_to_remove": faces_to_remove})


@mcp.tool()
def make_hole(name: str, source: str, diameter: float, depth: float | None = None, points: list[list[float]] | None = None) -> dict[str, Any]:
    """Cut one or more vertical holes from the top face using XY points."""
    _require_cq(); pts = points or [[0, 0]]; v = _obj(source).faces(">Z").workplane().pushPoints([tuple(p) for p in pts])
    v = v.hole(diameter, depth) if depth else v.hole(diameter)
    return _store(name, v, "hole", {"source": source, "diameter": diameter, "depth": depth, "points": pts})


@mcp.tool()
def pattern(name: str, source: str, mode: str, count: int, spacing: float, axis: str = "X") -> dict[str, Any]:
    """Create a linear or circular pattern of a named solid and union the copies."""
    _require_cq(); base = _obj(source); mode = mode.lower(); result = base
    if count < 1: raise ValueError("count deve ser >= 1")
    for i in range(1, count):
        if mode == "linear":
            shift = [0, 0, 0]; shift["XYZ".index(axis.upper())] = spacing * i; copy = base.translate(tuple(shift))
        elif mode == "circular":
            copy = base.rotate((0, 0, 0), (0, 0, 1), spacing * i)
        else: raise ValueError("mode deve ser linear ou circular")
        result = result.union(copy)
    return _store(name, result, "pattern", {"source": source, "mode": mode, "count": count, "spacing": spacing, "axis": axis})


@mcp.tool()
def measure_objects(first: str, second: str | None = None) -> dict[str, Any]:
    """Measure bounding-box size and, optionally, center-to-center distance between objects."""
    a = inspect_object(first); out = {"first": a}
    if second:
        b = inspect_object(second); ca = [(a["bounding_box_mm"]["min"][i] + a["bounding_box_mm"]["max"][i]) / 2 for i in range(3)]; cb = [(b["bounding_box_mm"]["min"][i] + b["bounding_box_mm"]["max"][i]) / 2 for i in range(3)]
        out.update({"second": b, "center_distance_mm": math.sqrt(sum((ca[i] - cb[i]) ** 2 for i in range(3)))})
    return out


@mcp.tool()
def validate_printability(name: str, min_wall: float = 1.2, max_size: float = 250.0, min_feature: float = 0.8) -> dict[str, Any]:
    """Run conservative heuristic checks for FDM printing; report warnings, not certification."""
    d = inspect_object(name); size = d["bounding_box_mm"]["size"]; issues = []
    if max(size) > max_size: issues.append(f"maior dimensão {max(size):.2f} mm excede {max_size} mm")
    if d["solid_count"] == 0: issues.append("nenhum sólido fechado detectado")
    if min_wall <= 0: issues.append("min_wall deve ser positivo")
    if min_feature <= 0: issues.append("min_feature deve ser positivo")
    return {"name": name, "printable_heuristic": not issues, "issues": issues, "recommendations": ["verificar paredes finas e overhangs no slicer", "orientar a peça e adicionar suportes conforme necessário", "inspecionar o STEP/STL exportado"]}


@mcp.tool()
def export_object(name: str, filename: str, format: str | None = None, tolerance: float = 0.01, angular_tolerance: float = 0.1) -> dict[str, Any]:
    """Export a named object to STEP, STL, AMF, 3MF, or DXF where CadQuery supports it."""
    _require_cq(); path = (WORKDIR / filename).resolve(); path.parent.mkdir(parents=True, exist_ok=True)
    if WORKDIR not in path.parents and path != WORKDIR: raise ValueError("filename deve permanecer dentro de CQ_WORKDIR")
    fmt = (format or path.suffix.lstrip(".")).upper(); kwargs = {}
    if fmt == "STL": kwargs = {"tolerance": tolerance, "angularTolerance": angular_tolerance}
    exporters.export(_obj(name), str(path), exportType=fmt, **kwargs)
    return {"name": name, "format": fmt, "path": str(path), "bytes": path.stat().st_size}


@mcp.tool()
def import_step(name: str, filename: str) -> dict[str, Any]:
    """Import a STEP/IGES/BREP file from CQ_WORKDIR into the document."""
    _require_cq(); path = (WORKDIR / filename).resolve()
    if not path.exists(): raise FileNotFoundError(path)
    v = importers.importStep(str(path)) if path.suffix.lower() in (".step", ".stp") else importers.importBrep(str(path))
    return _store(name, v, "import", {"filename": str(path)})


@mcp.tool()
def save_manifest(filename: str = "cadquery_manifest.json") -> dict[str, Any]:
    """Save object names, operations, parameters, and inspections as a reproducibility manifest."""
    path = (WORKDIR / filename).resolve(); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"objects": [{"name": n, "meta": META[n], "inspection": inspect_object(n)} for n in OBJECTS]}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"path": str(path), "object_count": len(OBJECTS)}


def main() -> None:
    transport = os.environ.get("CQ_MCP_TRANSPORT", "stdio")
    if transport == "sse":
        mcp.run(transport="sse")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
