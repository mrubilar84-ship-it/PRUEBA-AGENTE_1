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
        if "cajetin" in schema["properties"]:
            return json.loads(json.dumps(_plan_result()))
        if "inconsistencias" in schema["properties"]:
            return dict(FAKE_CROSS)
        return json.loads(json.dumps(FAKE_REVIEW))


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


# ---- interpretación de planos + chat que mira la hoja relevante
import plan_review as P


def _interp(numero, desc):
    base = {k: "(no consta)" for k in ("titulo", "proyecto", "revision", "fecha", "escala", "unidades", "hoja",
                                       "dibujo", "reviso", "aprobo")}
    return {"tipo_plano": "planta", "descripcion_general": desc, "cajetin": {**base, "numero": numero},
            "elementos": [{"nombre": "V-12", "descripcion": "viga de acero", "ubicacion": "centro"}],
            "dimensiones": [{"elemento": "luz total", "valor": "600", "unidad": "mm", "ubicacion": "arriba"}],
            "materiales_y_especificaciones": [], "notas": ["Acero A36"], "referencias": [],
            "vistas_y_cortes": [], "no_legible": []}


class _VisionFake:
    native_pdf, vision, chunk_chars = False, True, 0
    qa_tiles = False
    n = 0

    def generate_json(self, system, content, schema):
        _VisionFake.n += 1
        return _interp(f"ST-0{_VisionFake.n}", "Planta de la viga" if _VisionFake.n == 1 else "Detalle de soldadura")

    def generate_text(self, system, user, images=None):
        self.user, self.images = user, images
        return "respuesta"


def _two_sheet_pdf(path):
    doc = pymupdf_open()
    for t in ("PLANTA VIGA V-12", "DETALLE SOLDADURA D-1"):
        pg = doc.new_page(width=1191, height=842)
        pg.insert_text((100, 100), t, fontsize=14)
    doc.save(str(path))


def pymupdf_open():
    import pymupdf
    return pymupdf.open()


def test_interpretar_y_preguntar_con_imagen(tmp_path):
    _two_sheet_pdf(tmp_path / "plano_dos.pdf")
    llm = _VisionFake()
    plans = P.interpret_folder(tmp_path, tmp_path / "out", d.ReviewConfig(), llm=llm)
    assert len(plans) == 2 and plans[1]["pagina"] == 2
    md = (tmp_path / "out" / "planos_interpretados.md").read_text(encoding="utf-8")
    assert "luz total" in md and (tmp_path / "out" / "planos_interpretados.json").exists()

    chat = C.DocChat(llm)
    chat.add_plans(tmp_path / "out" / "planos_interpretados.json")
    out = chat.ask("¿Qué muestra la hoja 2?")
    assert len(llm.images) == 4  # 2 hojas candidatas × (vista general + cajetín)
    assert "plano_dos.pdf · hoja 2: vista general" in llm.user  # la hoja nombrada va primero
    assert "INTERPRETACIÓN DE PLANOS" in llm.user and "plano, imagen + interpretación" in out


def test_chat_planos_sin_vision_responde_con_interpretacion(tmp_path):
    _two_sheet_pdf(tmp_path / "plano_dos.pdf")
    v = _VisionFake()
    plans = P.interpret_folder(tmp_path, tmp_path / "out", d.ReviewConfig(), llm=v)

    class TextOnly:
        vision = False
        def generate_text(self, system, user): self.user = user; return "ok"
    t = TextOnly()
    chat = C.DocChat(t)
    chat.add_plans(plans)
    out = chat.ask("¿Cuál es la luz total de la viga?")
    assert "Cota luz total = 600 mm" in t.user and "interpretación)" in out


# ---- Gemini (cliente simulado)
import base64
import gemini_llm as G


class _Resp:
    def __init__(self, text): self.text = text


class _GErr(Exception):
    def __init__(self, code, msg=""):
        super().__init__(msg); self.code = code


class _GClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return _Resp(r)


def test_gemini_json_con_imagen_y_reintento_429(monkeypatch):
    monkeypatch.setattr(G.time, "sleep", lambda s: None)
    reply = json.dumps({**FAKE_REVIEW, "evaluacion_global": "Requiere revisión", "extra": 1})
    c = _GClient([_GErr(429), "no es json", reply])
    llm = G.GeminiLLM(model="m", client=c)
    img = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                       "data": base64.b64encode(b"\x89PNGfake").decode()}}
    out = llm.generate_json("sys", [img, {"type": "text", "text": "revisa"}], d.REVIEW_SCHEMA)
    assert out["evaluacion_global"] == "requiere_revision" and "extra" not in out
    assert len(c.calls) == 3  # 429 -> reintento; texto no JSON -> reintento de parseo; luego OK
    model, contents, cfg = c.calls[-1]
    assert model == "m" and len(contents) == 2 and cfg.response_mime_type == "application/json"


def test_gemini_texto_y_error_no_reintentable():
    llm = G.GeminiLLM(client=_GClient(["hola"]))
    assert llm.generate_text("s", "u") == "hola"
    bad = G.GeminiLLM(client=_GClient([_GErr(400)]))
    try:
        bad.generate_text("s", "u"); assert False
    except RuntimeError as e:  # 400 no se reintenta: se envuelve con una pista
        assert "400" in str(e) and bad.client.calls and len(bad.client.calls) == 1


def test_gemini_sin_clave(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False); monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    try:
        G.GeminiLLM(); assert False
    except RuntimeError as e:
        assert "GEMINI_API_KEY" in str(e)


# ---- resolver carpeta / URL de dataset de Kaggle
def test_resolve_dir_url_y_slug(tmp_path, monkeypatch):
    root = tmp_path / "kaggle_input"
    (root / "datasets" / "mauricio" / "agente-rev-doc-ing").mkdir(parents=True)
    monkeypatch.setattr(d, "KAGGLE_INPUT", root)
    hit = d.resolve_dir("https://www.kaggle.com/datasets/mauricio/agente-rev-doc-ing")
    assert hit.name == "agente-rev-doc-ing" and hit.is_dir()
    assert d.resolve_dir("agente-rev-doc-ing") == hit
    assert d.resolve_dir(hit) == hit
    try:
        d.resolve_dir("otro-dataset"); assert False
    except FileNotFoundError as e:
        assert "agente-rev-doc-ing" in str(e) or "datasets" in str(e)


def test_ignora_archivos_del_proyecto(tmp_path):
    for n in ("checklist_default.md", "README.md", "checklist_planos.md", "memoria.md"):
        (tmp_path / n).write_text("contenido", encoding="utf-8")
    c = FakeClient(); c.messages = c
    res = d.review_folder(tmp_path, tmp_path / "out", d.ReviewConfig(), client=c)
    assert [r["documento"] for r in res["revisiones"]] == ["memoria.md"]


def test_gemini_error_claro_404_y_list_models():
    llm = G.GeminiLLM(client=_GClient([_GErr(404)]), model="modelo-viejo")
    try:
        llm.generate_text("s", "u"); assert False
    except RuntimeError as e:
        assert "404" in str(e) and "modelo-viejo" in str(e) and "list_models" in str(e)

    class M:
        def __init__(self, n, a): self.name, self.supported_actions = n, a
    c = _GClient([]); c.list = lambda: [M("models/a", ["generateContent"]), M("models/emb", ["embedContent"])]
    c.models = c
    assert G.GeminiLLM(client=c).list_models() == ["a"]


def test_gemini_404_cambia_a_un_flash_disponible(monkeypatch):
    class M:
        def __init__(self, n, a=("generateContent",)): self.name, self.supported_actions = n, list(a)
    c = _GClient([_GErr(404), "listo"])
    c.list = lambda: [M("models/gemini-2.5-flash"), M("models/gemini-3.1-flash-lite"), M("models/gemini-3.5-flash"),
                      M("models/gemini-4-flash-preview"), M("models/gemini-3.5-pro"), M("models/text-embedding", ["embedContent"])]
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    llm = G.GeminiLLM(client=c)
    assert llm.generate_text("s", "u") == "listo"
    assert llm.model == "gemini-3.5-flash"  # el flash estable de mayor versión (ignora lite, pro y preview)
    assert [call[0] for call in c.calls] == ["gemini-3.8-flash", "gemini-3.5-flash"]


def test_gemini_404_con_modelo_explicito_no_se_cambia():
    llm = G.GeminiLLM(client=_GClient([_GErr(404)]), model="mi-modelo")
    try:
        llm.generate_text("s", "u"); assert False
    except RuntimeError as e:
        assert "mi-modelo" in str(e) and llm.model == "mi-modelo"


class _GModel:
    def __init__(self, n): self.name, self.supported_actions = "models/" + n, ["generateContent"]


QUOTA_MSG = "You exceeded your current quota... limit: 20, model: gemini-3.8-flash Please retry in 1h1m23.951924311s."


def test_gemini_cuota_diaria_cambia_a_otro_modelo(monkeypatch):
    monkeypatch.setattr(G.time, "sleep", lambda s: pytest_fail("no debe dormir 1 hora"))
    c = _GClient([_GErr(429, QUOTA_MSG), "listo"])
    c.list = lambda: [_GModel("gemini-3.8-flash"), _GModel("gemini-3.8-flash-lite"), _GModel("gemini-3.5-flash")]
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    llm = G.GeminiLLM(client=c)
    assert llm.generate_text("s", "u") == "listo"
    assert llm.model == "gemini-3.5-flash"  # flash normal antes que lite
    assert [x[0] for x in c.calls] == ["gemini-3.8-flash", "gemini-3.5-flash"]


def pytest_fail(msg):
    raise AssertionError(msg)


def test_gemini_cuota_sin_alternativa_o_modelo_fijo_da_error_claro():
    c = _GClient([_GErr(429, QUOTA_MSG)])
    c.list = lambda: [_GModel("gemini-3.8-flash")]  # sin otro modelo
    for llm in (G.GeminiLLM(client=c), G.GeminiLLM(client=_GClient([_GErr(429, QUOTA_MSG)]), model="fijo")):
        try:
            llm.generate_text("s", "u"); assert False
        except RuntimeError as e:
            assert "1 h 1 min" in str(e) and "facturación" in str(e)
    assert llm.model == "fijo"


def test_retry_seconds_y_humanize():
    assert abs(G._retry_seconds(Exception("Please retry in 1h1m23.9s.")) - 3683.9) < 0.01
    assert G._retry_seconds(Exception("Please retry in 7s")) == 7
    assert G._retry_seconds(Exception("{'retryDelay': '34s'}")) == 34
    assert G._retry_seconds(Exception("otro error")) is None
    assert G._humanize(3683) == "1 h 1 min" and G._humanize(125) == "2 min" and G._humanize(40) == "40 s"


# ---- LocalLLM._chat con tokenizador y modelo simulados (requiere torch; se omite si no está instalado)
def test_localllm_chat_compatible_con_transformers_4_y_5():
    torch = __import__("pytest").importorskip("torch")

    class Enc(dict):  # como BatchEncoding: dict con .to()
        def to(self, device): return self

    class Tok:
        eos_token_id = 0
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            assert tokenize is False and msgs[0]["role"] == "system"
            return "PROMPT"
        def __call__(self, text, return_tensors=None, add_special_tokens=True):
            assert text == "PROMPT" and return_tensors == "pt"
            return Enc(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.tensor([[1, 1, 1]]))
        def decode(self, ids, skip_special_tokens=True):
            return "ids:" + ",".join(str(int(i)) for i in ids)

    class Model:
        device = "cpu"
        def generate(self, input_ids=None, attention_mask=None, **kw):
            assert kw["do_sample"] is False
            return torch.cat([input_ids, torch.tensor([[7, 8]])], dim=1)

    llm = L.LocalLLM.__new__(L.LocalLLM)  # sin cargar un modelo real
    llm.tok, llm.model, llm.max_new_tokens, llm.retries = Tok(), Model(), 50, 1
    assert llm.generate_text("sys", "user") == "ids:7,8"  # solo los tokens nuevos
    llm._chat = lambda system, user, images=None: '{"tipo_documento":"X","resumen":"r","evaluacion_global":"aprobado","hallazgos":[],"informacion_faltante":[]}'
    out = llm.generate_json("s", [{"type": "text", "text": "doc"}], d.REVIEW_SCHEMA)
    assert out["evaluacion_global"] == "aprobado" and out["hallazgos"] == []


# ---- administrador de documentos
import doc_manager as DM


def test_docfolder_agregar_varios_borrar_reemplazar(tmp_path, monkeypatch):
    src = tmp_path / "ds" / "documento-1"; src.mkdir(parents=True)
    (src / "a.txt").write_text("A1"); (src / "b.md").write_text("B1"); (src / "c.exe").write_text("x")
    (src / "checklist_x.md").write_text("no"); (tmp_path / "suelto.txt").write_text("S")
    monkeypatch.setattr(d, "KAGGLE_INPUT", tmp_path / "ds")
    docs = DM.DocFolder(tmp_path / "work")

    assert sorted(docs.add("documento-1")) == ["a.txt", "b.md"]  # por nombre de dataset; ignora .exe y checklists
    assert docs.add(tmp_path / "suelto.txt") == ["suelto.txt"]
    (src / "a.txt").write_text("A2")
    assert docs.add(src / "a.txt") == []  # ya existe: se conserva
    assert (tmp_path / "work" / "a.txt").read_text() == "A1"
    assert docs.add(src / "a.txt", replace=True) == ["a.txt"] and (tmp_path / "work" / "a.txt").read_text() == "A2"

    docs.replace("b.md", tmp_path / "suelto.txt")  # b.md se borra y queda suelto.txt actualizado
    assert [p.name for p in docs.files()] == ["a.txt", "suelto.txt"]
    assert docs.remove("*.txt") == ["a.txt", "suelto.txt"] and docs.files() == []
    docs.add(src / "a.txt"); docs.clear()
    assert docs.files(), "clear() sin confirm no debe borrar"
    docs.clear(confirm=True)
    assert docs.files() == []


def test_replace_con_origen_en_la_misma_carpeta(tmp_path):
    docs = DM.DocFolder(tmp_path / "w")
    (tmp_path / "w" / "x.txt").write_text("v1")
    docs.replace("x.txt", tmp_path / "w" / "x.txt")  # no debe perder el archivo
    assert (tmp_path / "w" / "x.txt").read_text() == "v1"


def test_guardar_subidos_formatos_ipywidgets_v7_y_v8(tmp_path):
    v7 = {"a.txt": {"content": b"uno"}, "../evil.txt": {"content": b"dos"}, "x.exe": {"content": b"no"}}
    assert DM._save_uploaded(v7, tmp_path) == ["a.txt", "evil.txt"]  # sin rutas; omite formatos no soportados
    v8 = ({"name": "b.md", "content": memoryview(b"tres")},)
    assert DM._save_uploaded(v8, tmp_path) == ["b.md"] and (tmp_path / "b.md").read_bytes() == b"tres"
    assert not (tmp_path.parent / "evil.txt").exists()


# ---- modos «solo documentos» de plan_mode
def _carpeta_con_plano_y_doc(tmp_path):
    shutil.copy(Path(__file__).resolve().parents[1] / "samples" / "plano_ejemplo.pdf", tmp_path / "plano_ejemplo.pdf")
    (tmp_path / "memoria.txt").write_text("texto de la memoria", encoding="utf-8")
    import pymupdf
    im = pymupdf.open(); im.new_page(width=100, height=100).insert_text((10, 50), "x")
    pix = im[0].get_pixmap(); pix.save(str(tmp_path / "foto.png"))


def test_plan_mode_skip_revisa_solo_documentos(tmp_path):
    _carpeta_con_plano_y_doc(tmp_path)
    llm = _Rec(vision=True)
    res = d.review_folder(tmp_path, tmp_path / "out", d.ReviewConfig(plan_mode="skip"), llm=llm)
    assert [r["documento"] for r in res["revisiones"]] == ["memoria.txt"]
    assert set(res["omitidos"]) == {"plano_ejemplo.pdf", "foto.png"}
    assert all(b["type"] == "text" for call in llm.calls for b in call)  # nunca se enviaron imágenes


def test_plan_mode_never_lee_todo_pdf_como_documento_e_ignora_imagenes(tmp_path):
    _carpeta_con_plano_y_doc(tmp_path)
    llm = _Rec(vision=True)
    res = d.review_folder(tmp_path, tmp_path / "out", d.ReviewConfig(plan_mode="never"), llm=llm)
    assert sorted(r["documento"] for r in res["revisiones"]) == ["memoria.txt", "plano_ejemplo.pdf"]
    assert set(res["omitidos"]) == {"foto.png"}


def test_solo_planos_con_skip_da_error_claro(tmp_path):
    shutil.copy(Path(__file__).resolve().parents[1] / "samples" / "plano_ejemplo.pdf", tmp_path / "plano_ejemplo.pdf")
    try:
        d.review_folder(tmp_path, tmp_path / "out", d.ReviewConfig(plan_mode="skip"), llm=_Rec(vision=False))
        assert False
    except FileNotFoundError as e:
        assert "plano_ejemplo.pdf" in str(e)


def test_chat_handle_comandos(tmp_path):
    chat, Fake = _fake_chat(tmp_path)
    assert chat._handle("/docs") == "Documentos cargados: mem.txt, otro.txt"
    assert "AISC" in chat._handle("¿Qué norma se usa?") and chat.history
    assert "mem.txt" in chat._handle("/resumen mem.txt")
    assert chat._handle("/reset") == "Conversación reiniciada." and chat.history == []
