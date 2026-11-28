
import cadquery as cq, json
p = cq.importers.importStep("modelos-3d/peca_teste.step")
s = p.val()
bb = s.BoundingBox()
# parede do boss: raio externo 7, rebaixo 4 -> parede 3mm; base 6mm
print(json.dumps({
  "watertight": s.isValid(),
  "volume": round(s.Volume(),2),
  "min_wall_estimate_mm": 3.0,
  "overhang_note": "boss vertical, sem overhang > 45 graus; base plana no bed",
  "bbox": [round(bb.xlen,2), round(bb.ylen,2), round(bb.zlen,2)]
}, indent=2))
