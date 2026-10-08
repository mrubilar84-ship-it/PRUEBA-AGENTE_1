"""Prueba offline con un cliente simulado (no gasta API)."""
import json, sys, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import doc_review_agent as d

FAKE_REVIEW = {"tipo_documento": "Memoria de cálculo", "resumen": "ok", "evaluacion_global": "requiere_revision",
               "hallazgos": [{"id": "H-01", "severidad": "mayor", "categoria": "Cálculos", "ubicacion": "§7",
                              "evidencia": "δ = 16 mm", "problema": "recalculo da ~20 mm", "recomendacion": "corregir"}],
               "informacion_faltante": ["revisor"]}
FAKE_CROSS = {"resumen": "sin cruces", "inconsistencias": []}

class _Stream:
    def __init__(self, payload): self.p = payload
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def get_final_message(self):
        blk = types.SimpleNamespace(type="text", text=json.dumps(self.p))
        return types.SimpleNamespace(stop_reason="end_turn", content=[blk], stop_details=None)

class FakeClient:
    def __init__(self):
        self.messages = self.beta = types.SimpleNamespace(messages=self)
        self.calls = []
    def stream(self, **kw):
        self.calls.append(kw)
        return _Stream(FAKE_CROSS if "inconsistencias" in json.dumps(kw["output_config"]) else FAKE_REVIEW)

def test_pipeline(tmp_path):
    src = tmp_path / "in"; src.mkdir()
    (src / "a.txt").write_text("hola"); (src / "b.md").write_text("mundo"); (src / "x.exe").write_text("no")
    c = FakeClient()
    c.messages = c  # client.messages.stream
    res = d.review_folder(src, tmp_path / "out", d.ReviewConfig(), client=c)
    assert len(res["revisiones"]) == 2 and res["consistencia"] == FAKE_CROSS and not res["errores"]
    assert (tmp_path / "out" / "informe.md").exists()
    assert "H-01" in (tmp_path / "out" / "hallazgos.csv").read_text(encoding="utf-8-sig")


# ---- backend local (sin GPU): parser/coerción y troceado con un LLM falso
import local_llm as L


def test_extract_json_y_coerce():
    raw = 'Claro:\n```json\n{"tipo_documento":"X","resumen":"r","evaluacion_global":"Requiere revisión",' \
          '"hallazgos":[{"id":"H-01","severidad":"Crítica","categoria":"c","ubicacion":"u","evidencia":"e",' \
          '"problema":"p","recomendacion":"r","extra":1}],"informacion_faltante":"nada"} fin'
    out = L.coerce(L.extract_json(raw), d.REVIEW_SCHEMA)
    assert out["hallazgos"][0]["severidad"] == "critica" and "extra" not in out["hallazgos"][0]
    assert out["evaluacion_global"] == "requiere_revision" and out["informacion_faltante"] == ["nada"]


def test_troceado_y_fusion(tmp_path):
    class Fake:
        native_pdf, chunk_chars = False, 50
        n = 0
        def generate_json(self, system, content, schema):
            Fake.n += 1
            return {"tipo_documento": "T", "resumen": f"p{Fake.n}", "evaluacion_global": "aprobado",
                    "hallazgos": [{"id": "H-01", "severidad": "mayor" if Fake.n == 2 else "menor", "categoria": "c",
                                   "ubicacion": "u", "evidencia": "e", "problema": "p", "recomendacion": "r"}],
                    "informacion_faltante": ["rev"]}
    doc = d.LoadedDoc("big.txt", [{"type": "text", "text": "\n".join(f"linea {i} " * 3 for i in range(20))}])
    r = d.review_document(Fake(), doc, d.ReviewConfig())
    assert Fake.n > 2 and r["evaluacion_global"] == "requiere_revision"
    assert r["hallazgos"][0]["severidad"] == "mayor" and r["hallazgos"][0]["id"] == "H-01"
    assert r["informacion_faltante"] == ["rev"]


# ---- chat sobre documentos
import doc_chat as C


def _fake_chat(tmp_path, **kw):
    (tmp_path / "mem.txt").write_text(
        "[Página 1]\nObjetivo: verificar la viga V-12.\n[Página 2]\nNorma aplicada AISC 360 con factor phi 0.9.\n"
        "[Página 3]\nDeflexión máxima 16 mm.", encoding="utf-8")
    (tmp_path / "otro.txt").write_text("Procedimiento de soldadura WPS-01 para tuberías.", encoding="utf-8")

    class Fake:
        native_pdf = False
        seen = None
        def generate_text(self, system, user):
            Fake.seen = user
            return "Se usa AISC 360 [mem.txt, p. 2]"
    chat = C.DocChat(Fake(), **kw)
    chat.add_folder(tmp_path)
    return chat, Fake


def test_chat_contexto_completo_con_revision(tmp_path):
    chat, Fake = _fake_chat(tmp_path)
    rev = tmp_path / "rev.json"
    rev.write_text(json.dumps({"revisiones": [{"documento": "mem.txt", "tipo_documento": "Memoria", "resumen": "res",
        "evaluacion_global": "requiere_revision", "informacion_faltante": ["revisor"],
        "hallazgos": [{"id": "H-01", "severidad": "mayor", "categoria": "Cálculos", "ubicacion": "§7",
                       "evidencia": "δ=16", "problema": "deflexión mal calculada", "recomendacion": "recalcular"}]}],
        "consistencia": None}), encoding="utf-8")
    chat.add_review(rev)
    out = chat.ask("¿Por qué la deflexión está mal?")
    assert "deflexión mal calculada" in Fake.seen and "AISC 360" in Fake.seen and "WPS-01" in Fake.seen
    assert "Contexto usado" in out and chat.history
    assert chat.summarize("nope").startswith("No encuentro") and "mem.txt" in chat.summarize("mem.txt")


def test_chat_recuperacion_si_no_cabe(tmp_path):
    chat, Fake = _fake_chat(tmp_path, full_context_chars=50)
    out = chat.ask("¿Qué norma se aplica y cuál es el factor phi?")
    assert "AISC 360" in Fake.seen and "soldadura" not in Fake.seen
    assert "mem.txt (p. 2)" in out


# ---- planos
import shutil


def _plan_result(cajetin_num="ST-012"):
    r = dict(FAKE_REVIEW)
    r["cajetin"] = {k: "x" for k in ("titulo", "proyecto", "revision", "fecha", "escala", "unidades", "hoja",
                                     "dibujo", "reviso", "aprobo")}
    r["cajetin"]["numero"] = cajetin_num
    return r


class _Rec:
    """Backend falso que registra lo que recibe."""
    native_pdf = False
    chunk_chars = 0

    def __init__(self, vision):
        self.vision, self.calls = vision, []

    def generate_json(self, system, content, schema):
        self.calls.append(content)
        return json.loads(json.dumps(_plan_result())) if "cajetin" in schema["properties"] else dict(FAKE_CROSS)


def test_plano_con_vision_envia_imagenes(tmp_path):
    shutil.copy(Path(__file__).resolve().parents[1] / "samples" / "plano_ejemplo.pdf", tmp_path / "plano_ejemplo.pdf")
    llm = _Rec(vision=True)
    res = d.review_folder(tmp_path, tmp_path / "out", d.ReviewConfig(), llm=llm)
    content = llm.calls[0]
    assert sum(b["type"] == "image" for b in content) == 6  # general + cajetín + 4 cuadrantes
    assert "ST-012" in content[-1]["text"] and "REV: (vacío)" in content[-1]["text"]  # texto vectorial incluido
    assert res["revisiones"][0]["cajetin"]["numero"] == "ST-012"
    assert "N° plano" in (tmp_path / "out" / "informe.md").read_text(encoding="utf-8")


def test_plano_sin_vision_solo_texto_con_aviso(tmp_path):
    shutil.copy(Path(__file__).resolve().parents[1] / "samples" / "plano_ejemplo.pdf", tmp_path / "plano_ejemplo.pdf")
    llm = _Rec(vision=False)
    d.review_folder(tmp_path, tmp_path / "out", d.ReviewConfig(), llm=llm)
    content = llm.calls[0]
    assert all(b["type"] == "text" for b in content) and "no ve imágenes" in content[-1]["text"]


def test_plan_mode_never_trata_pdf_como_documento(tmp_path):
    import plan_review as P
    f = tmp_path / "plano_ejemplo.pdf"
    shutil.copy(Path(__file__).resolve().parents[1] / "samples" / "plano_ejemplo.pdf", f)
    assert P.is_plan_pdf(f, "auto") and not P.is_plan_pdf(f, "never") and P.is_plan_pdf(f, "always")
