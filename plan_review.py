"""Revisión de planos de ingeniería (PDF impresos desde DWG, o imágenes).

Cada hoja se revisa mirando su imagen: vista general + cajetín ampliado + cuadrantes ampliados, junto con
el texto vectorial del PDF (si existe). Requiere un backend con visión:
  - ClaudeLLM (API de Claude), o
  - local_llm.LocalVLM (Qwen2.5-VL, gratis en la GPU de Kaggle).
Con un backend solo-texto (LocalLLM) se revisa únicamente el texto vectorial extraído, con aviso.
"""
from __future__ import annotations

import base64
import io
import re
import tempfile
from pathlib import Path

import pymupdf

from doc_review_agent import (
    REVIEW_SCHEMA, SYSTEM_PROMPT, ReviewConfig, _text_block, is_project_file, resolve_dir,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
PLAN_HINTS = re.compile(r"plano|plan[_\- ]|lamina|lámina|sheet|drawing|dwg|layout|isometric|isom", re.I)
LARGE_SIDE_PT = 1000  # A3 apaisado mide 1191 x 842 pt; A4 mide 842 x 595

CAJETIN_SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "string"} for k in (
        "numero", "titulo", "proyecto", "revision", "fecha", "escala", "unidades",
        "hoja", "dibujo", "reviso", "aprobo")},
    "required": ["numero", "titulo", "proyecto", "revision", "fecha", "escala", "unidades",
                 "hoja", "dibujo", "reviso", "aprobo"],
    "additionalProperties": False,
}
PLAN_SCHEMA = {
    **REVIEW_SCHEMA,
    "properties": {**REVIEW_SCHEMA["properties"], "cajetin": CAJETIN_SCHEMA},
    "required": REVIEW_SCHEMA["required"] + ["cajetin"],
}

PLAN_SYSTEM = SYSTEM_PROMPT + """

Estás revisando PLANOS de ingeniería (dibujos técnicos impresos a PDF desde CAD). Lee el cajetín y las cotas \
con cuidado. En «cajetin» transcribe los campos tal como aparecen; si un campo no existe o no se lee, escribe \
"(no consta)". En «tipo_documento» indica el tipo de plano (planta, corte, isométrico, detalle, diagrama...)."""


def _png_block(png: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.standard_b64encode(png).decode("ascii")}}


def is_plan_pdf(path: Path, mode: str = "auto") -> bool:
    if mode == "always":
        return path.suffix.lower() == ".pdf"
    if mode in ("never", "skip"):  # «skip» lo resuelve review_folder (que usa «auto» para decidir qué omitir)
        return False
    with pymupdf.open(path) as doc:
        if not len(doc):
            return False
        r = doc[0].rect
    return max(r.width, r.height) > LARGE_SIDE_PT or bool(PLAN_HINTS.search(path.stem))


def render_clip(page, clip, max_side: int) -> bytes:
    """Renderiza una zona de la página como PNG con lado mayor ≈ max_side px."""
    zoom = max_side / max(clip.width, clip.height)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip, alpha=False)
    return pix.tobytes("png")


def sheet_images(page, max_side: int, tiles: bool = True) -> list[tuple[str, bytes]]:
    """[(descripción, png)]: vista general, cajetín (esquina inferior derecha) y 4 cuadrantes con solape."""
    r = page.rect
    W, H = r.width, r.height
    out = [("vista general de la hoja", render_clip(page, r, max_side))]
    tb = pymupdf.Rect(r.x0 + 0.5 * W, r.y0 + 0.7 * H, r.x1, r.y1)
    out.append(("cajetín ampliado (esquina inferior derecha)", render_clip(page, tb, max_side)))
    if tiles:
        ov = 0.08
        for name, (xa, xb, ya, yb) in {
            "cuadrante superior izquierdo": (0, 0.5 + ov, 0, 0.5 + ov),
            "cuadrante superior derecho": (0.5 - ov, 1, 0, 0.5 + ov),
            "cuadrante inferior izquierdo": (0, 0.5 + ov, 0.5 - ov, 1),
            "cuadrante inferior derecho": (0.5 - ov, 1, 0.5 - ov, 1),
        }.items():
            clip = pymupdf.Rect(r.x0 + xa * W, r.y0 + ya * H, r.x0 + xb * W, r.y0 + yb * H)
            out.append((name + " ampliado", render_clip(page, clip, max_side)))
    return out


def sheet_text(page, limit: int = 6000) -> tuple[str, str]:
    """(texto de la hoja, texto de la zona del cajetín). Vacío si el CAD imprimió el texto como líneas."""
    full = page.get_text("text", sort=True).strip()
    r = page.rect
    zone = pymupdf.Rect(r.x0 + 0.5 * r.width, r.y0 + 0.7 * r.height, r.x1, r.y1)
    tb = page.get_text("text", clip=zone, sort=True).strip()
    return (full[:limit] + ("\n[... texto recortado ...]" if len(full) > limit else "")), tb[:1500]


def _sheet_content(name: str, idx: int, total: int, page, llm, cfg: ReviewConfig, task: str | None = None) -> list[dict]:
    text, tb = sheet_text(page)
    vision = getattr(llm, "vision", False)
    parts, notes = [], []
    desc = ""
    if vision:
        imgs = sheet_images(page, getattr(llm, "image_max_side", 1500), cfg.plan_tiles)
        parts += [_png_block(png) for _, png in imgs]
        desc = "Imágenes en este orden: " + "; ".join(f"{i}) {d}" for i, (d, _) in enumerate(imgs, 1)) + ". "
    else:
        notes.append("AVISO: este modelo no ve imágenes; solo dispones del texto vectorial extraído del PDF. "
                     "No puedes evaluar geometría ni cotas dibujadas; limita la revisión a lo que está escrito.")
    if len(text) < 30:
        notes.append("El PDF no contiene texto vectorial (el CAD lo imprimió como líneas o es un escaneo): "
                     "toda la lectura depende de las imágenes." if vision else
                     "El PDF no contiene texto vectorial: no hay nada que revisar sin visión.")
    prompt = (f"Plano «{name}», hoja {idx} de {total}. {desc}\n" + "\n".join(notes) +
              (f"\n\nTEXTO VECTORIAL DE LA HOJA (puede estar desordenado o incompleto):\n{text}" if text else "") +
              (f"\n\nTEXTO EN LA ZONA DEL CAJETÍN:\n{tb}" if tb else "") +
              "\n\n" + (task or "Revisa esta hoja según el checklist de planos. Numera los hallazgos H-01, H-02, ... por severidad."))
    return parts + [_text_block(prompt)]


def review_plan_pdf(llm, path: Path, cfg: ReviewConfig) -> list[dict]:
    """Devuelve una revisión por hoja (formato de REVIEW_SCHEMA + 'cajetin')."""
    system = PLAN_SYSTEM + "\n\n# Checklist de revisión de planos\n" + Path(cfg.plan_checklist_path).read_text(encoding="utf-8")
    if cfg.extra_instructions.strip():
        system += f"\n\n# Contexto adicional del proyecto\n{cfg.extra_instructions.strip()}"
    results = []
    with pymupdf.open(path) as doc:
        total = len(doc)
        if total > cfg.plan_max_pages:
            print(f"  ! {path.name}: {total} hojas; se revisan las primeras {cfg.plan_max_pages} (ReviewConfig.plan_max_pages)")
        for i in range(min(total, cfg.plan_max_pages)):
            print(f"  hoja {i + 1}/{min(total, cfg.plan_max_pages)}")
            content = _sheet_content(path.name, i + 1, total, doc[i], llm, cfg)
            r = llm.generate_json(system, content, PLAN_SCHEMA)
            r["documento"] = path.name if total == 1 else f"{path.name} · hoja {i + 1}"
            results.append(r)
    return results


def review_plan_image(llm, path: Path, cfg: ReviewConfig) -> list[dict]:
    """Un plano entregado como imagen (png/jpg/tif): se convierte a PDF de una hoja y se revisa igual."""
    with pymupdf.open(path) as img:
        pdf_bytes = img.convert_to_pdf()
    tmp = Path(tempfile.gettempdir()) / (path.stem + ".tmp_plan.pdf")  # /kaggle/input es de solo lectura
    try:
        tmp.write_bytes(pdf_bytes)
        res = review_plan_pdf(llm, tmp, cfg)
    finally:
        tmp.unlink(missing_ok=True)
    for r in res:
        r["documento"] = path.name
    return res


# --------------------------------------------------------------------------- interpretación

def _obj(**fields):
    return {"type": "object", "properties": {k: {"type": "string"} for k in fields}, "required": list(fields),
            "additionalProperties": False}


def _arr(item):
    return {"type": "array", "items": item}


INTERPRET_SCHEMA = {
    "type": "object",
    "properties": {
        "tipo_plano": {"type": "string"},
        "descripcion_general": {"type": "string"},
        "cajetin": CAJETIN_SCHEMA,
        "elementos": _arr(_obj(nombre="", descripcion="", ubicacion="")),
        "dimensiones": _arr(_obj(elemento="", valor="", unidad="", ubicacion="")),
        "materiales_y_especificaciones": _arr(_obj(posicion="", descripcion="", cantidad="", material_norma="")),
        "notas": {"type": "array", "items": {"type": "string"}},
        "referencias": _arr(_obj(tipo="", destino="", ubicacion="")),
        "vistas_y_cortes": {"type": "array", "items": {"type": "string"}},
        "no_legible": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["tipo_plano", "descripcion_general", "cajetin", "elementos", "dimensiones",
                 "materiales_y_especificaciones", "notas", "referencias", "vistas_y_cortes", "no_legible"],
    "additionalProperties": False,
}

INTERPRET_SYSTEM = SYSTEM_PROMPT + """

Tu tarea es INTERPRETAR planos de ingeniería: extraer con fidelidad qué muestra cada hoja para que luego se \
puedan responder preguntas sin volver a mirarla. Reglas:
- Transcribe solo lo que está escrito o claramente dibujado; no deduzcas ni midas sobre la imagen.
- «descripcion_general»: qué representa la hoja y para qué sirve (3-6 frases).
- «elementos»: componentes, equipos, tramos, ejes, marcas o tags identificables, con su ubicación en la hoja.
- «dimensiones»: cada cota escrita con su elemento, valor y unidad; si la unidad no está explícita, usa la de las notas.
- «materiales_y_especificaciones»: filas de la lista de materiales y especificaciones (posición, cantidad, material, norma).
- «notas»: notas generales y específicas, textualmente. «referencias»: planos, cortes, detalles o normas citados.
- «no_legible»: lo que existe pero no puedes leer con certeza. Mejor declararlo que inventarlo.
- Campos desconocidos: "(no consta)"."""

INTERPRET_TASK = ("Interpreta esta hoja: completa el JSON con la información que contiene (tipo de plano, descripción, "
                  "cajetín, elementos, dimensiones, materiales y especificaciones, notas, referencias, vistas y cortes, "
                  "y lo no legible).")


def interpret_plan_pdf(llm, path: Path, cfg: ReviewConfig) -> list[dict]:
    """Una interpretación por hoja, con 'documento', 'archivo' y 'pagina' para poder volver a la imagen."""
    out = []
    with pymupdf.open(path) as doc:
        total = len(doc)
        n = min(total, cfg.plan_max_pages)
        if total > n:
            print(f"  ! {path.name}: {total} hojas; se interpretan las primeras {n} (ReviewConfig.plan_max_pages)")
        for i in range(n):
            print(f"  interpretando hoja {i + 1}/{n}")
            content = _sheet_content(path.name, i + 1, total, doc[i], llm, cfg, task=INTERPRET_TASK)
            r = llm.generate_json(INTERPRET_SYSTEM, content, INTERPRET_SCHEMA)
            r.update(documento=path.name if total == 1 else f"{path.name} · hoja {i + 1}",
                     archivo=str(path), pagina=i + 1)
            out.append(r)
    return out


def interpret_plan_image(llm, path: Path, cfg: ReviewConfig) -> list[dict]:
    with pymupdf.open(path) as img:
        pdf_bytes = img.convert_to_pdf()
    pdf = Path(tempfile.gettempdir()) / (path.stem + ".plano.pdf")  # persiste en la sesión para volver a mirar la hoja
    pdf.write_bytes(pdf_bytes)
    res = interpret_plan_pdf(llm, pdf, cfg)
    for r in res:
        r["documento"] = path.name
    return res


def render_interpretation_md(plans: list[dict]) -> str:
    out = ["# Interpretación de planos\n"]
    for p in plans:
        c = p["cajetin"]
        out.append(f"## {p['documento']} — {p['tipo_plano']}\n")
        out.append(f"**N° {c['numero']} · Rev. {c['revision']} · Escala {c['escala']} · {c['titulo']}**\n")
        out.append(p["descripcion_general"] + "\n")
        if p["dimensiones"]:
            out.append("**Dimensiones**\n\n| Elemento | Valor | Unidad | Ubicación |\n|---|---|---|---|")
            out += [f"| {d['elemento']} | {d['valor']} | {d['unidad']} | {d['ubicacion']} |" for d in p["dimensiones"]]
            out.append("")
        if p["materiales_y_especificaciones"]:
            out.append("**Materiales y especificaciones**\n\n| Pos. | Descripción | Cant. | Material / norma |\n|---|---|---|---|")
            out += [f"| {m['posicion']} | {m['descripcion']} | {m['cantidad']} | {m['material_norma']} |"
                    for m in p["materiales_y_especificaciones"]]
            out.append("")
        if p["elementos"]:
            out.append("**Elementos**\n")
            out += [f"- {e['nombre']}: {e['descripcion']} ({e['ubicacion']})" for e in p["elementos"]]
            out.append("")
        for title, key in (("Notas", "notas"), ("Vistas y cortes", "vistas_y_cortes"), ("No legible", "no_legible")):
            if p[key]:
                out.append(f"**{title}**\n")
                out += [f"- {x}" for x in p[key]]
                out.append("")
        if p["referencias"]:
            out.append("**Referencias**\n")
            out += [f"- {r['tipo']} → {r['destino']} ({r['ubicacion']})" for r in p["referencias"]]
            out.append("")
    return "\n".join(out)


def interpret_folder(in_dir: str | Path, out_dir: str | Path, cfg: ReviewConfig | None = None, llm=None) -> list[dict]:
    """Interpreta todos los planos (PDF de plano o imágenes) y guarda planos_interpretados.json / .md."""
    import json

    cfg = cfg or ReviewConfig()
    if llm is None:
        from doc_review_agent import ClaudeLLM
        llm = ClaudeLLM(cfg)
    if not getattr(llm, "vision", False):
        print("AVISO: el modelo no ve imágenes; la interpretación se limitará al texto vectorial del PDF.")
    plans, errors = [], {}
    for p in sorted(resolve_dir(in_dir).rglob("*")):
        ext = p.suffix.lower()
        if p.is_file() and not is_project_file(p) and not p.name.endswith(".plano.pdf") and (
                (ext in IMAGE_EXTS and cfg.plan_mode not in ("never", "skip")) or (ext == ".pdf" and is_plan_pdf(p, cfg.plan_mode))):
            print(f"Interpretando {p.name} ...")
            try:
                plans += (interpret_plan_image if ext in IMAGE_EXTS else interpret_plan_pdf)(llm, p, cfg)
            except Exception as e:
                errors[p.name] = f"{type(e).__name__}: {e}"
                print(f"  ! {errors[p.name]}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "planos_interpretados.json").write_text(
        json.dumps({"planos": plans, "errores": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "planos_interpretados.md").write_text(render_interpretation_md(plans), encoding="utf-8")
    print(f"Listo: {len(plans)} hojas interpretadas, {len(errors)} con error -> {out}")
    return plans
