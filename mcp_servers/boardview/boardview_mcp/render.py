"""Renderização de uma região da placa em PNG (para modelos multimodais)."""
from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont

from .index import BoardIndex

BG = (18, 20, 26)
OUTLINE = (120, 120, 130)
C_GND = (95, 95, 105)
C_PWR = (240, 130, 40)
C_SIG = (80, 150, 235)
C_NC = (60, 60, 66)
C_HL_NET = (255, 70, 200)
C_HL_PART = (60, 230, 110)
C_VIA = (170, 170, 60)
C_TP = (255, 255, 255)
C_BODY = (70, 72, 82)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow antigo
        return ImageFont.load_default()


def resolve_center(ix: BoardIndex, center: str | None, x: float | None, y: float | None,
                   side: str) -> tuple[float, float, str, str]:
    """-> (cx, cy, side, descrição)."""
    if center:
        if center in ix.b.parts or center.upper() in ix.b.parts:
            p = ix.find_part(center)
            cx, cy = p.center
            return cx, cy, (p.side if side == "auto" else side), f"componente {p.refdes}"
        n = ix.find_net(center)
        pins = ix.b.nets[n]
        xs = [q.x for q in pins]
        ys = [q.y for q in pins]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        sides = [q.side for q in pins]
        s = max(set(sides), key=sides.count)
        return cx, cy, (s if side == "auto" else side), f"rede {n}"
    if x is None or y is None:
        x0, y0, x1, y1 = ix.b.bounds
        return (x0 + x1) / 2, (y0 + y1) / 2, ("TOP" if side == "auto" else side), "placa inteira"
    return x, y, ("TOP" if side == "auto" else side), f"ponto ({x:.1f},{y:.1f})"


def render_region(ix: BoardIndex, cx: float, cy: float, width_mm: float, height_mm: float,
                  side: str, highlight_parts: list[str], highlight_nets: list[str],
                  show_labels: bool = True, show_other_side: bool = False,
                  max_px: int = 1000) -> tuple[bytes, dict]:
    side = side.upper()
    x0, x1 = cx - width_mm / 2, cx + width_mm / 2
    y0, y1 = cy - height_mm / 2, cy + height_mm / 2
    scale = min(max_px / width_mm, max_px / height_mm)
    W, H = max(1, int(width_mm * scale)), max(1, int(height_mm * scale))
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img, "RGBA")

    def P(x: float, y: float) -> tuple[float, float]:
        px = (x - x0) * scale if side == "TOP" else (x1 - x) * scale  # BOTTOM: vista espelhada (placa virada)
        return px, (y1 - y) * scale

    def rect(x: float, y: float, w: float, h: float):
        a = P(x - w / 2, y - h / 2)
        b = P(x + w / 2, y + h / 2)
        return (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))

    # contorno da placa
    for ax, ay, bx, by in ix.b.outline:
        if max(ax, bx) < x0 or min(ax, bx) > x1 or max(ay, by) < y0 or min(ay, by) > y1:
            continue
        d.line([P(ax, ay), P(bx, by)], fill=OUTLINE, width=2)

    hl_parts = {p.upper() for p in highlight_parts}
    hl_nets = {n for n in highlight_nets}
    fnt = _font(max(9, min(16, int(scale * 1.4))))
    labels: list[tuple[float, float, str, tuple[int, int, int]]] = []
    shown_parts = 0
    reach = max(width_mm, height_mm)
    for r in set(ix.parts_near(cx, cy, reach)):
        p = ix.b.parts[r]
        on_side = p.side == side
        if not on_side and not show_other_side and r.upper() not in hl_parts:
            continue
        bx0, by0, bx1, by1 = p.bbox
        if bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1:
            continue
        shown_parts += 1
        alpha = 255 if on_side else 70
        is_hl = r.upper() in hl_parts
        # corpo
        if p.pins:
            a, c = P(bx0, by0), P(bx1, by1)
            box = (min(a[0], c[0]), min(a[1], c[1]), max(a[0], c[0]), max(a[1], c[1]))
            d.rectangle(box, outline=(*C_BODY, alpha), width=1)
            if is_hl:
                d.rectangle(box, outline=C_HL_PART, width=3)
        for q in p.pins:
            if q.net is None:
                col = C_NC
            elif q.net in hl_nets:
                col = C_HL_NET
            elif ix.is_ground(q.net):
                col = C_GND
            elif ix.net_info[q.net].kind == "power":
                col = C_PWR
            else:
                col = C_SIG
            w = max(q.w, 0.25)
            h = max(q.h, 0.25)
            bb = rect(q.x, q.y, w, h)
            if bb[2] - bb[0] < 2:
                bb = (bb[0] - 1, bb[1] - 1, bb[0] + 1, bb[1] + 1)
            d.rectangle(bb, fill=(*col, alpha))
        if show_labels and on_side:
            ccx, ccy = p.center
            sz = max(bx1 - bx0, by1 - by0) * scale
            if is_hl or sz >= 28:
                px, py = P(ccx, ccy)
                labels.append((px, py, r, C_HL_PART if is_hl else (225, 225, 230)))

    # vias e testpoints
    for v in ix.b.vias:
        if x0 <= v.x <= x1 and y0 <= v.y <= y1:
            px, py = P(v.x, v.y)
            col = C_HL_NET if v.net in hl_nets else C_VIA
            rr = 3 if v.net in hl_nets else 1.5
            d.ellipse((px - rr, py - rr, px + rr, py + rr), outline=col)
    for t in ix.b.testpoints:
        if x0 <= t.x <= x1 and y0 <= t.y <= y1 and (t.side == side or show_other_side):
            px, py = P(t.x, t.y)
            col = C_HL_NET if t.net in hl_nets else C_TP
            d.ellipse((px - 3, py - 3, px + 3, py + 3), outline=col, width=1)

    for px, py, txt, col in labels:
        bb = d.textbbox((px, py), txt, font=fnt, anchor="mm")
        d.rectangle((bb[0] - 1, bb[1] - 1, bb[2] + 1, bb[3] + 1), fill=(0, 0, 0, 170))
        d.text((px, py), txt, font=fnt, fill=col, anchor="mm")

    # barra de escala
    bar_mm = 1 if width_mm <= 12 else 5 if width_mm <= 60 else 10 if width_mm <= 150 else 50
    bx, by = 12, H - 14
    d.line([(bx, by), (bx + bar_mm * scale, by)], fill=(255, 255, 255), width=3)
    d.text((bx, by - 16), f"{bar_mm} mm", font=_font(12), fill=(255, 255, 255))
    d.text((W - 4, 4), f"vista {side}{' (espelhada)' if side == 'BOTTOM' else ''}", font=_font(12),
           fill=(255, 255, 255), anchor="ra")

    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    meta = {
        "side_view": side, "mirrored_x": side == "BOTTOM",
        "region_mm": {"x0": round(x0, 2), "y0": round(y0, 2), "x1": round(x1, 2), "y1": round(y1, 2)},
        "px_per_mm": round(scale, 2), "image_px": [W, H], "parts_drawn": shown_parts,
        "legend": {"laranja": "pino em rail de alimentação (heurística)", "cinza": "GND", "azul": "sinal",
                   "cinza escuro": "pino sem rede", "magenta": "rede destacada", "verde": "componente destacado",
                   "amarelo (círculo)": "via", "branco (círculo)": "testpoint"},
    }
    return buf.getvalue(), meta
