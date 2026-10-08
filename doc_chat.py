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

from doc_review_agent import SUPPORTED, load_document

CHAT_SYSTEM = """Eres un asistente técnico que responde preguntas sobre documentos de ingeniería.
Reglas:
- Responde SOLO con la información de los FRAGMENTOS entregados. Si no alcanza para responder, dilo \
y señala qué falta; no inventes valores, normas ni páginas.
- Cita la fuente de cada dato clave entre corchetes, por ejemplo [memoria.pdf, p. 3].
- Si te piden un cálculo, muéstralo paso a paso con unidades y advierte si los datos son insuficientes.
- Responde en español, claro y conciso."""

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
    def __init__(self, llm, top_k: int = 6, max_history: int = 4, max_chars: int = 1500):
        self.llm, self.top_k, self.max_history, self.max_chars = llm, top_k, max_history, max_chars
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
        for p in sorted(Path(folder).rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED:
                try:
                    self.add_file(p)
                    print(f"Cargado: {p.name}")
                except Exception as e:
                    print(f"  ! {p.name}: {e}")
        print(f"{len(self.passages)} pasajes indexados de {len(self.texts)} documentos.")

    def add_review(self, revision_json: str | Path) -> None:
        """Incorpora los hallazgos de la revisión para poder preguntarlos (p. ej. «¿cuáles son los críticos?»)."""
        data = json.loads(Path(revision_json).read_text(encoding="utf-8"))
        for r in data.get("revisiones", []):
            for h in r["hallazgos"]:
                t = (f"HALLAZGO {h['id']} [{h['severidad']}] ({h['categoria']}) en {r['documento']}, "
                     f"{h['ubicacion']}: {h['problema']} Evidencia: {h['evidencia']} Recomendación: {h['recomendacion']}")
                self.passages.append(Passage(r["documento"], "revisión", t))
            self.passages.append(Passage(r["documento"], "revisión",
                                         f"RESUMEN DE REVISIÓN de {r['documento']} ({r['evaluacion_global']}): {r['resumen']}"))
        self._index = None

    # ---- uso
    def _search(self, query: str) -> list[Passage]:
        if self._index is None:
            self._index = BM25(self.passages)
        return self._index.search(query, self.top_k)

    def ask(self, question: str, show_sources: bool = True) -> str:
        if not self.passages:
            return "No hay documentos cargados."
        # La pregunta de seguimiento ("¿y su valor?") se enriquece con el turno anterior para recuperar mejor.
        query = question if not self.history else f"{self.history[-1][0]} {question}"
        found = self._search(query)
        if not found:
            answer = "No encontré fragmentos relacionados con esa pregunta en los documentos cargados."
        else:
            ctx = "\n\n".join(f"[FRAGMENTO {i} | {p.doc}, {p.label}]\n{p.text}" for i, p in enumerate(found, 1))
            hist = "".join(f"Usuario: {q}\nAsistente: {a}\n\n" for q, a in self.history[-self.max_history:])
            prev = ("CONVERSACIÓN PREVIA:\n" + hist) if hist else ""
            user = f"FRAGMENTOS:\n{ctx}\n\n{prev}PREGUNTA: {question}"
            answer = self.llm.generate_text(CHAT_SYSTEM, user)
        self.history.append((question, answer))
        out = answer
        if show_sources and found:
            out += "\n\nFuentes consultadas: " + "; ".join(dict.fromkeys(f"{p.doc} ({p.label})" for p in found))
        return out

    def summarize(self, doc: str | None = None, window_chars: int = 9000) -> str:
        """Resumen de un documento (o de todos). Documentos largos: resumen por tramos y luego resumen final."""
        names = [doc] if doc else list(self.texts)
        missing = [n for n in names if n not in self.texts]
        if missing:
            return f"No encuentro {missing}. Disponibles: {list(self.texts)}"
        partials = []
        for n in names:
            text = self.texts[n]
            parts = [text[i:i + window_chars] for i in range(0, len(text), window_chars)] or [""]
            sums = [self.llm.generate_text(SUMMARY_SYSTEM, f"Documento «{n}», parte {i} de {len(parts)}:\n\n{t}\n\n"
                                           "Resume esta parte en viñetas concisas.") for i, t in enumerate(parts, 1)]
            joined = "\n".join(sums)
            if len(parts) > 1:
                joined = self.llm.generate_text(SUMMARY_SYSTEM, f"Integra estos resúmenes parciales de «{n}» en un solo "
                                                "resumen estructurado:\n\n" + joined)
            partials.append(f"## {n}\n{joined}")
        return "\n\n".join(partials)

    def reset(self) -> None:
        self.history.clear()

    def repl(self) -> None:
        """Chat interactivo en el notebook. Comandos: /resumen [archivo], /reset, /salir."""
        print("Pregunta lo que quieras sobre los documentos. /resumen [archivo], /reset, /salir")
        while True:
            q = input("\nTú: ").strip()
            if not q:
                continue
            if q == "/salir":
                break
            if q == "/reset":
                self.reset(); print("Conversación reiniciada."); continue
            if q.startswith("/resumen"):
                print(self.summarize(q[8:].strip() or None)); continue
            print("\nAgente:", self.ask(q))
