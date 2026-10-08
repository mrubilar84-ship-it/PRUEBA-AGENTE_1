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
