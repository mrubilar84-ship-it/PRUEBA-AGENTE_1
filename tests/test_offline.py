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
