"""Chat sobre documentos de ingeniería: preguntas, resúmenes y consulta de la revisión.

Funciona con cualquier backend del proyecto (LocalLLM gratis en Kaggle o ClaudeLLM).
La recuperación de pasajes es BM25 en Python puro (sin modelo de embeddings ni dependencias extra),
así el modelo recibe solo los fragmentos relevantes y cabe en contextos pequeños.

    chat = DocChat(llm)
    chat.add_folder("/kaggle/input/documentos-ingenieria")
    chat.add_review("/kaggle/working/informe/revision.json")   # opcional: hallazgos de la revisión
    chat.ask("¿Qué norma se usa y con qué factor de seguridad?")
    chat.summarize("memoria.pdf")
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from doc_review_agent import SUPPORTED, is_project_file, load_document, resolve_dir

CHAT_SYSTEM = """Eres un ingeniero revisor que conversa sobre documentos de ingeniería que ya revisaste.
Recibirás: (1) la REVISIÓN previa con sus hallazgos, (2) el contenido de los DOCUMENTOS (completo o fragmentos) \
y la pregunta del usuario.
Cómo responder:
- Razona con lógica de ingeniería a partir del contexto: relaciona hallazgos, datos, cálculos y normas entre sí \
y explica el porqué, no solo qué dice el texto.
- Distingue claramente lo que está escrito en los documentos de lo que es tu inferencia o recomendación.
- Si la información no alcanza para concluir, dilo y señala exactamente qué dato falta. No inventes valores, \
normas ni páginas.
- Cita la fuente de los datos clave entre corchetes, por ejemplo [memoria.pdf, p. 3] o [hallazgo H-02].
- En cálculos, muestra los pasos con unidades.
- Si hay INTERPRETACIÓN DE PLANOS, úsala como base para describir y explicar los planos; si además recibes \
imágenes de la hoja, úsalas para verificar y completar, pero no midas sobre la imagen: usa solo cotas escritas. \
Si la interpretación y la imagen difieren, dilo.
- Responde en español, claro y directo."""

SUMMARY_SYSTEM = ("Eres un ingeniero que resume documentos técnicos en español: objetivo, datos y criterios clave, "
                  "resultados/conclusiones, normas citadas y puntos pendientes. Sé fiel al texto; no inventes.")


@dataclass
class Passage:
    doc: str
    label: str  # "p. 3", "hoja Cálculos", "revisión"
    text: str


def _tokens(text: str) -> list[str]:
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)?", t)


_STOP = set("de la el los las un una y o en que se por con para del al es son a lo como mas su sus este esta".split())


def chunk_text(doc: str, text: str, max_chars: int = 1500, overlap_lines: int = 2) -> list[Passage]:
    """Parte el texto en pasajes con etiqueta de página/hoja, sin cortar líneas."""
    out, label, buf, size = [], "", [], 0

    def flush():
        nonlocal buf, size
        if buf and "".join(buf).strip():
            out.append(Passage(doc, label, "\n".join(buf)))
        buf, size = buf[-overlap_lines:] if overlap_lines and len(buf) > overlap_lines else [], 0
        size = sum(len(x) + 1 for x in buf)

    for line in text.splitlines():
        m = re.match(r"\[Página (\d+)\]", line) or None
        h = re.match(r"## Hoja: (.+)", line)
        if m or h:
            flush(); buf, size = [], 0
            label = f"p. {m.group(1)}" if m else f"hoja {h.group(1)}"
            if m:
                continue
        while len(line) > max_chars:
            buf.append(line[:max_chars]); size += max_chars; flush(); line = line[max_chars:]
        if size + len(line) + 1 > max_chars and buf:
            flush()
        buf.append(line); size += len(line) + 1
    flush()
    return out


class BM25:
    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.p, self.k1, self.b = passages, k1, b
        self.tf = [Counter(w for w in _tokens(x.text) if w not in _STOP) for x in passages]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 0
        df = Counter(w for c in self.tf for w in c)
        n = len(passages)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def search(self, query: str, k: int = 6) -> list[Passage]:
        q = [w for w in _tokens(query) if w not in _STOP]
        scores = []
        for i, c in enumerate(self.tf):
            s = 0.0
            for w in q:
                if w in c:
                    f = c[w]
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1)))
            scores.append(s)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.p[i] for i in order[:k] if scores[i] > 0]


class DocChat:
    def __init__(self, llm, top_k: int = 6, max_history: int = 4, max_chars: int = 1500,
                 full_context_chars: int = 30000, digest_chars: int = 7000, look_sheets: int = 2):
        """full_context_chars: si todos los documentos suman menos que esto (~9k tokens), se entregan completos
        al modelo; si no, solo los pasajes más relevantes. La revisión se entrega siempre."""
        self.llm, self.top_k, self.max_history, self.max_chars = llm, top_k, max_history, max_chars
        self.full_context_chars, self.digest_chars = full_context_chars, digest_chars
        self.look_sheets = look_sheets
        self.review_digest = ""
        self.plans: list[dict] = []  # interpretaciones de planos (una por hoja)
        self.plan_passages: list[Passage] = []
        self._plan_index: BM25 | None = None
        self.passages: list[Passage] = []
        self.texts: dict[str, str] = {}
        self.history: list[tuple[str, str]] = []
        self._index: BM25 | None = None

    # ---- carga
    def add_file(self, path: str | Path) -> None:
        path = Path(path)
        doc = load_document(path, native_pdf=False)
        text = "\n".join(b["text"] for b in doc.blocks if b["type"] == "text")
        self.texts[doc.name] = text
        self.passages += chunk_text(doc.name, text, self.max_chars)
        self._index = None

    def add_folder(self, folder: str | Path) -> None:
        for p in sorted(resolve_dir(folder).rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED and not is_project_file(p):
                try:
                    self.add_file(p)
                    print(f"Cargado: {p.name}")
                except Exception as e:
                    print(f"  ! {p.name}: {e}")
        print(f"{len(self.passages)} pasajes indexados de {len(self.texts)} documentos.")

    def add_review(self, revision_json: str | Path) -> None:
        """Incorpora la revisión (resúmenes, hallazgos, inconsistencias) como contexto permanente del chat."""
        data = json.loads(Path(revision_json).read_text(encoding="utf-8"))
        order = {"critica": 0, "mayor": 1, "menor": 2, "observacion": 3}
        lines = []
        for r in data.get("revisiones", []):
            lines.append(f"## {r['documento']} ({r['tipo_documento']}) - evaluación: {r['evaluacion_global']}")
            lines.append(f"Resumen: {r['resumen']}")
            for h in sorted(r["hallazgos"], key=lambda h: order[h["severidad"]]):
                lines.append(f"- [{h['id']}] {h['severidad'].upper()} · {h['categoria']} · {h['ubicacion']}: "
                             f"{h['problema']} (evidencia: {h['evidencia']}) → {h['recomendacion']}")
            if r["informacion_faltante"]:
                lines.append("Información faltante: " + "; ".join(r["informacion_faltante"]))
        cross = data.get("consistencia")
        if cross and cross.get("inconsistencias"):
            lines.append("## Inconsistencias entre documentos")
            lines += [f"- {c['severidad'].upper()} ({', '.join(c['documentos'])}): {c['descripcion']}"
                      for c in cross["inconsistencias"]]
        digest = "\n".join(lines)
        if len(digest) > self.digest_chars:  # ya viene ordenado por severidad dentro de cada documento
            digest = digest[:self.digest_chars] + "\n[... revisión recortada por tamaño ...]"
        self.review_digest = digest

    def add_plans(self, interpretations: str | Path | list[dict]) -> None:
        """Incorpora planos interpretados (planos_interpretados.json o la lista que devuelve interpret_folder)."""
        if not isinstance(interpretations, list):
            interpretations = json.loads(Path(interpretations).read_text(encoding="utf-8"))["planos"]
        for pl in interpretations:
            self.plans.append(pl)
            sheet = pl["documento"]
            c = pl["cajetin"]
            head = (f"PLANO {sheet}: {pl['tipo_plano']}. N° {c['numero']}, título {c['titulo']}, rev. {c['revision']}, "
                    f"escala {c['escala']}, fecha {c['fecha']}, proyecto {c['proyecto']}. {pl['descripcion_general']}")
            texts = [head]
            texts += [f"Elemento {e['nombre']}: {e['descripcion']} ({e['ubicacion']})" for e in pl["elementos"]]
            texts += [f"Cota {d['elemento']} = {d['valor']} {d['unidad']} ({d['ubicacion']})" for d in pl["dimensiones"]]
            texts += [f"Material pos. {m['posicion']}: {m['descripcion']}, cant. {m['cantidad']}, {m['material_norma']}"
                      for m in pl["materiales_y_especificaciones"]]
            texts += [f"Nota: {n}" for n in pl["notas"]]
            texts += [f"Referencia {r['tipo']} → {r['destino']} ({r['ubicacion']})" for r in pl["referencias"]]
            texts += [f"Vista/corte: {v}" for v in pl["vistas_y_cortes"]]
            texts += [f"No legible: {n}" for n in pl["no_legible"]]
            # agrupa de a ~8 líneas para que cada pasaje conserve contexto
            for i in range(0, len(texts), 8):
                self.plan_passages.append(Passage(sheet, "interpretación", "\n".join(texts[i:i + 8])))
        self._plan_index = None
        print(f"{len(interpretations)} hojas de planos incorporadas ({len(self.plan_passages)} pasajes).")

    # ---- uso
    def _search(self, query: str) -> list[Passage]:
        if self._index is None:
            self._index = BM25(self.passages)
        return self._index.search(query, self.top_k)

    def _context(self, question: str) -> tuple[str, list[str]]:
        """Devuelve (texto de contexto, fuentes). Documentos completos si caben; si no, pasajes relevantes."""
        total = sum(len(t) for t in self.texts.values())
        if total <= self.full_context_chars:
            ctx = "\n\n".join(f"=== DOCUMENTO: {n} ===\n{t}" for n, t in self.texts.items())
            return ctx, list(self.texts)
        query = question if not self.history else f"{self.history[-1][0]} {question}"
        found = self._search(query)
        ctx = "\n\n".join(f"[FRAGMENTO {i} | {p.doc}, {p.label}]\n{p.text}" for i, p in enumerate(found, 1))
        return ctx, list(dict.fromkeys(f"{p.doc} ({p.label})" for p in found))

    def _plan_context(self, question: str, k: int = 10) -> tuple[str, list[dict]]:
        """Pasajes de la interpretación de planos relevantes + hojas candidatas para mirar la imagen."""
        if not self.plan_passages:
            return "", []
        if self._plan_index is None:
            self._plan_index = BM25(self.plan_passages)
        query = question if not self.history else f"{self.history[-1][0]} {question}"
        found = self._plan_index.search(query, k)
        by_sheet = {pl["documento"]: pl for pl in self.plans}
        ctx = "\n\n".join(f"[{p.doc}]\n{p.text}" for p in found)
        # hojas a mirar: la nombrada en la pregunta ("hoja 2", n° de plano) o las mejor puntuadas
        q = question.lower()
        named = [pl for pl in self.plans
                 if (pl["cajetin"]["numero"] and pl["cajetin"]["numero"].lower() in q and pl["cajetin"]["numero"] != "(no consta)")
                 or pl["documento"].lower() in q]
        m = re.search(r"hoja\s*(\d+)", q)
        if m:
            named += [pl for pl in self.plans if pl["pagina"] == int(m.group(1))]
        ranked = [by_sheet[p.doc] for p in found if p.doc in by_sheet]
        sheets = []
        for pl in named + ranked:
            if pl not in sheets:
                sheets.append(pl)
        return ctx, sheets[:self.look_sheets]

    def ask(self, question: str, show_sources: bool = True, look: bool | str = "auto") -> str:
        """look='auto': si el modelo ve imágenes y hay planos, adjunta la(s) hoja(s) relevante(s) a la pregunta."""
        if not self.passages and not self.plan_passages:
            return "No hay documentos cargados."
        ctx, sources = self._context(question) if self.passages else ("", [])
        plan_ctx, sheets = self._plan_context(question)
        images, legend = [], []
        if look and getattr(self.llm, "vision", False) and sheets:
            import pymupdf

            from plan_review import sheet_images

            for pl in sheets:
                try:
                    with pymupdf.open(pl["archivo"]) as pdf:
                        imgs = sheet_images(pdf[pl["pagina"] - 1], getattr(self.llm, "image_max_side", 1500),
                                            getattr(self.llm, "qa_tiles", True))
                except Exception as e:  # el PDF ya no está disponible: se responde solo con la interpretación
                    print(f"  (no pude abrir {pl['archivo']}: {e})")
                    continue
                for desc, png in imgs:
                    images.append(png)
                    legend.append(f"{pl['documento']}: {desc}")
        hist = "".join(f"Usuario: {q}\nAsistente: {a}\n\n" for q, a in self.history[-self.max_history:])
        parts = []
        if self.review_digest:
            parts.append("REVISIÓN PREVIA:\n" + self.review_digest)
        if ctx or not plan_ctx:
            parts.append("DOCUMENTOS:\n" + (ctx or "(sin fragmentos relevantes para esta pregunta)"))
        if plan_ctx:
            parts.append("INTERPRETACIÓN DE PLANOS (extraída previamente de cada hoja):\n" + plan_ctx)
        if images:
            parts.append("Se adjuntan imágenes en este orden: " + "; ".join(f"{i}) {d}" for i, d in enumerate(legend, 1)) + ".")
        if hist:
            parts.append("CONVERSACIÓN PREVIA:\n" + hist.rstrip())
        parts.append("PREGUNTA: " + question)
        user = "\n\n".join(parts)
        answer = (self.llm.generate_text(CHAT_SYSTEM, user, images=images) if images
                  else self.llm.generate_text(CHAT_SYSTEM, user))
        self.history.append((question, answer))
        if show_sources:
            how = "imagen + interpretación" if images else "interpretación"
            used = sources + [f"{pl['documento']} (plano, {how})" for pl in sheets]
            if used:
                answer += "\n\nContexto usado: " + "; ".join(dict.fromkeys(used))
        return answer

    def ask_image(self, question: str, pdf_path: str | Path, page: int = 1, tiles: bool = True) -> str:
        """Pregunta sobre lo que SE VE en una hoja de plano (requiere modelo con visión: LocalVLM o ClaudeLLM)."""
        if not getattr(self.llm, "vision", False):
            return "El modelo cargado no ve imágenes. Usa LocalVLM (gratis) o ClaudeLLM para preguntar sobre planos."
        import pymupdf

        from plan_review import sheet_images, sheet_text

        with pymupdf.open(pdf_path) as doc:
            pg = doc[page - 1]
            imgs = sheet_images(pg, getattr(self.llm, "image_max_side", 1500), tiles)
            text, tb = sheet_text(pg)
        legend = "; ".join(f"{i}) {d}" for i, (d, _) in enumerate(imgs, 1))
        parts = []
        if self.review_digest:
            parts.append("REVISIÓN PREVIA:\n" + self.review_digest)
        parts.append(f"Hoja {page} de «{Path(pdf_path).name}». Imágenes en orden: {legend}.")
        if text:
            parts.append("TEXTO VECTORIAL DE LA HOJA:\n" + text)
        if self.history:
            parts.append("CONVERSACIÓN PREVIA:\n" + "".join(f"Usuario: {q}\nAsistente: {a}\n\n" for q, a in self.history[-self.max_history:]).rstrip())
        parts.append("PREGUNTA: " + question + "\nNo midas sobre la imagen: usa solo cotas escritas; si algo no se lee con certeza, dilo.")
        answer = self.llm.generate_text(CHAT_SYSTEM, "\n\n".join(parts), images=[png for _, png in imgs])
        self.history.append((question, answer))
        return answer

    def summarize(self, doc: str | None = None, window_chars: int = 14000) -> str:
        """Resumen de un documento (o de todos). Documentos largos: resumen por tramos y luego resumen final.
        Muestra el avance; con un modelo local puede tardar varios minutos en documentos largos."""
        names = [doc] if doc else list(self.texts)
        missing = [n for n in names if n not in self.texts]
        if missing:
            return f"No encuentro {missing}. Disponibles: {list(self.texts)}"
        partials = []
        for n in names:
            text = self.texts[n]
            parts = [text[i:i + window_chars] for i in range(0, len(text), window_chars)] or [""]
            sums = []
            for i, t in enumerate(parts, 1):
                print(f"  Resumiendo «{n}»: parte {i} de {len(parts)} ...", flush=True)
                sums.append(self.llm.generate_text(
                    SUMMARY_SYSTEM, f"Documento «{n}», parte {i} de {len(parts)}:\n\n{t}\n\n"
                    "Resume esta parte en un máximo de 8 viñetas concisas."))
            joined = "\n".join(sums)
            if len(parts) > 1:
                print(f"  Integrando el resumen de «{n}» ...", flush=True)
                joined = self.llm.generate_text(
                    SUMMARY_SYSTEM, f"Integra estos resúmenes parciales de «{n}» en un solo resumen estructurado "
                    "(objetivo, datos y criterios clave, resultados y conclusiones, normas citadas, puntos pendientes), "
                    "de máximo una página:\n\n" + joined)
            partials.append(f"## {n}\n{joined}")
        return "\n\n".join(partials)

    def reset(self) -> None:
        self.history.clear()

    def _handle(self, text: str) -> str:
        """Interpreta lo escrito por el usuario: pregunta, o comando (/resumen [archivo], /reset, /docs)."""
        text = text.strip()
        if text == "/reset":
            self.reset()
            return "Conversación reiniciada."
        if text == "/docs":
            return "Documentos cargados: " + (", ".join(self.texts) or "ninguno")
        if text.startswith("/resumen"):
            return self.summarize(text[8:].strip() or None)
        return self.ask(text)

    def ui(self):
        """Caja de preguntas con botones en el notebook (si no se ve, usa chat.ask('...') o chat.repl())."""
        import ipywidgets as w
        from IPython.display import display

        box = w.Textarea(placeholder="Escribe tu pregunta sobre los documentos...", layout=w.Layout(width="95%", height="80px"))
        ask_b = w.Button(description="Preguntar", button_style="primary", icon="question")
        sum_b = w.Button(description="Resumen de todos los documentos", icon="file-text")
        rst_b = w.Button(description="Nueva conversación", icon="refresh")
        out = w.Output()

        def run(cmd):
            with out:
                print("⏳ pensando...")
                try:
                    ans = self._handle(cmd)
                except Exception as e:  # mostrar el error sin romper la caja
                    ans = f"Error: {type(e).__name__}: {e}"
                out.clear_output()
                print(ans)

        ask_b.on_click(lambda _: box.value.strip() and run(box.value))
        sum_b.on_click(lambda _: run("/resumen"))
        rst_b.on_click(lambda _: run("/reset"))
        display(w.VBox([box, w.HBox([ask_b, sum_b, rst_b]), out]))

    def repl(self) -> None:
        """Chat interactivo en el notebook. Comandos: /resumen [archivo], /reset, /docs, /salir."""
        print("Pregunta lo que quieras sobre los documentos. /resumen [archivo], /reset, /docs, /salir")
        while True:
            q = input("\nTú: ").strip()
            if not q:
                continue
            if q == "/salir":
                break
            print("\nAgente:", self._handle(q))
