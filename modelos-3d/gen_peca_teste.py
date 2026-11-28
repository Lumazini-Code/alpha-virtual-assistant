
import cadquery as cq, os, json

OUT = os.path.abspath("modelos-3d")
os.makedirs(OUT, exist_ok=True)

L, W, T = 60.0, 40.0, 6.0
HOLE_D = 3.4          # passagem M3
INSET = 6.0
BOSS_D, BOSS_H = 14.0, 10.0

base = (
    cq.Workplane("XY")
    .box(L, W, T, centered=(True, True, False))
    .edges("|Z").fillet(4.0)
    .faces(">Z").workplane()
    .pushPoints([(-L/2+INSET, -W/2+INSET), (L/2-INSET, -W/2+INSET),
                 (-L/2+INSET,  W/2-INSET), (L/2-INSET,  W/2-INSET)])
    .hole(HOLE_D)
)

boss = (
    cq.Workplane("XY").workplane(offset=T)
    .circle(BOSS_D/2).extrude(BOSS_H)
    .faces(">Z").workplane()
    .hole(8.0, 8.0)   # rebaixo central 8mm
)

part = base.union(boss)
part = part.edges(">Z").chamfer(0.6)

solid = part.val()
bb = solid.BoundingBox()
info = {
    "solids": len(part.solids().vals()),
    "volume_mm3": round(solid.Volume(), 2),
    "bbox": [round(bb.xlen,2), round(bb.ylen,2), round(bb.zlen,2)],
    "faces": len(part.faces().vals()),
}

step = os.path.join(OUT, "peca_teste.step")
stl  = os.path.join(OUT, "peca_teste.stl")
cq.exporters.export(part, step)
cq.exporters.export(part, stl, tolerance=0.01, angularTolerance=0.1)
info["files"] = {p: os.path.getsize(p) for p in (step, stl)}
print(json.dumps(info, indent=2))
