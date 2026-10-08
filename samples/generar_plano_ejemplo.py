"""Genera samples/plano_ejemplo.pdf (hoja A3 apaisada) con errores plantados para probar la revisión de planos:
- la cadena de cotas 100 + 200 + 250 no suma el total indicado (600)
- cajetín sin revisión ni 'revisó/aprobó'
- marca de corte A-A sin la vista de corte correspondiente
"""
import pymupdf

doc = pymupdf.open()
W, H = 1191, 842
pg = doc.new_page(width=W, height=H)
pg.draw_rect(pymupdf.Rect(20, 20, W - 20, H - 20), width=1.5)
# viga en planta
pg.draw_rect(pymupdf.Rect(150, 300, 750, 360), width=1.2)
for x in (150, 250, 450, 700, 750):
    pg.draw_line((x, 300), (x, 250), width=0.5)
for (xa, xb, label) in ((150, 250, "100"), (250, 450, "200"), (450, 700, "250")):
    pg.draw_line((xa, 260), (xb, 260), width=0.8)
    pg.insert_text(((xa + xb) / 2 - 10, 255), label, fontsize=11)
pg.draw_line((150, 225), (750, 225), width=0.8)
pg.insert_text((430, 220), "600", fontsize=11)
pg.insert_text((150, 395), "PLANTA VIGA V-12   ESC 1:50", fontsize=12)
# marca de corte A-A sin vista asociada
pg.draw_line((450, 280), (450, 380), width=1.5)
pg.insert_text((455, 285), "A", fontsize=12)
pg.insert_text((455, 385), "A", fontsize=12)
pg.insert_text((60, 120), "NOTAS: 1. ACERO A36.  2. SOLDADURA E70XX.  3. DIMENSIONES EN mm.", fontsize=10)
# cajetín (inferior derecha)
x0, y0 = 700, 700
pg.draw_rect(pymupdf.Rect(x0, y0, W - 20, H - 20), width=1.2)
rows = ["PROYECTO: EDIFICIO DEMO", "TITULO: PLANTA VIGA V-12", "PLANO N°: ST-012   HOJA 1 DE 1",
        "ESCALA: 1:50   FECHA: 2026-03-10", "DIBUJO: J. PEREZ", "REVISO:", "APROBO:", "REV: (vacío)"]
for i, t in enumerate(rows):
    pg.insert_text((x0 + 8, y0 + 18 + i * 14), t, fontsize=9)
doc.save("samples/plano_ejemplo.pdf")
print("samples/plano_ejemplo.pdf listo")
