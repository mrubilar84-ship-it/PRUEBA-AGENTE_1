"""Backend Gemini (Google) para el agente: texto + visión (planos), contexto largo.

Uso en Kaggle (Internet: On; secreto GEMINI_API_KEY en Add-ons → Secrets):
    from gemini_llm import GeminiLLM
    llm = GeminiLLM()                       # modelo por defecto o variable GEMINI_MODEL
    review_folder(IN_DIR, OUT_DIR, cfg, llm=llm)

Clave gratuita: Google AI Studio (aistudio.google.com). OJO: en el nivel gratuito Google puede usar lo que envías
para mejorar sus productos y revisores humanos pueden leerlo; no subas documentos confidenciales ahí. Con
facturación activada ese uso se excluye. Los límites del nivel gratuito cambian y son bajos (peticiones por
minuto/día): el backend reintenta con espera ante 429/5xx.
"""
from __future__ import annotations

import json
import os
import re
import time

from local_llm import _schema_example, coerce, extract_json

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"  # si no existe para tu clave se elige otro "flash" disponible (ver list_models())


_HINTS = {
    400: "Pista: revisa que la clave GEMINI_API_KEY esté completa y sea válida (¿copiaste el ID del proyecto en vez de la clave?) y que la petición no sea demasiado grande.",
    401: "Pista: la clave GEMINI_API_KEY no es válida (¿copiaste el ID del proyecto en vez de la clave?).",
    403: "Pista: la clave no tiene permiso para este modelo o el proyecto no tiene acceso gratuito. "
         "Crea otra clave en AI Studio o prueba otro modelo (llm.list_models()).",
    404: "Pista: ese modelo no existe o ya no está disponible. Ejecuta llm.list_models() y usa uno de la lista: "
         "GeminiLLM('nombre').",
    429: "Pista: agotaste la cuota del plan gratuito (por minuto o por día). Espera, usa otro modelo, o activa facturación.",
}


def _retry_seconds(e) -> float | None:
    """Segundos que Gemini pide esperar («Please retry in 1h1m23.9s» / retryDelay: '3s'), si los indica."""
    txt = str(e)
    m = re.search(r"retry in\s*(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:([\d.]+)s)?", txt, re.I)
    if m and any(m.groups()):
        h, mi, sec = (float(x) if x else 0.0 for x in m.groups())
        return h * 3600 + mi * 60 + sec
    m = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", txt)
    return float(m.group(1)) if m else None


def _humanize(sec: float) -> str:
    sec = int(sec)
    h, rest = divmod(sec, 3600)
    mi = rest // 60
    return f"{h} h {mi} min" if h else (f"{mi} min" if mi else f"{sec} s")


class GeminiLLM:
    native_pdf = False  # el agente extrae el texto del PDF; las imágenes de planos van aparte
    vision = True
    image_max_side = 2000
    qa_tiles = True
    chunk_chars = 0  # contexto de ~1M tokens: sin troceado

    def __init__(self, model: str | None = None, api_key: str | None = None, client=None,
                 max_output_tokens: int = 16000, retries: int = 6, parse_retries: int = 2):
        self._auto_model = not model and not os.environ.get("GEMINI_MODEL")  # modelo por defecto: se puede sustituir
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self._tried = {self.model}
        self.max_output_tokens, self.retries, self.parse_retries = max_output_tokens, retries, parse_retries
        if client is None:
            from google import genai

            key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError("Falta la clave: define GEMINI_API_KEY (en Kaggle: Add-ons → Secrets).")
            key = key.strip().strip("\"'")  # tolera espacios, saltos de línea o comillas al pegar
            client = genai.Client(api_key=key)
        self.client = client

    def list_models(self) -> list[str]:
        """Modelos que tu clave puede usar para generar contenido."""
        out = []
        for m in self.client.models.list():
            acts = getattr(m, "supported_actions", None) or []
            if not acts or "generateContent" in acts:
                out.append(m.name.replace("models/", ""))
        return out

    def _candidates(self) -> list[str]:
        """Otros modelos «flash» disponibles, de mejor a peor opción (cada modelo tiene su propia cuota gratuita)."""
        skip = ("image", "tts", "audio", "live", "embed", "robotics", "computer", "veo", "imagen")
        try:
            names = [n for n in self.list_models() if "flash" in n and not any(x in n for x in skip)
                     and n not in self._tried]
        except Exception:
            return []

        def key(n):
            v = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
            return ("lite" in n, bool(re.search(r"preview|exp", n)), -(float(v.group(1)) if v else 0), len(n))
        return sorted(names, key=key)

    def _switch_model(self, why: str) -> bool:
        """Cambia a otro modelo disponible si el modelo no fue fijado por el usuario."""
        if not self._auto_model:
            return False
        alts = self._candidates()
        if not alts:
            return False
        print(f"  {why}; uso «{alts[0]}».")
        self.model = alts[0]
        self._tried.add(self.model)
        return True

    def check(self) -> str:
        """Prueba rápida de conexión (clave, modelo y cuota) antes de procesar documentos."""
        return self.generate_text("Responde en una palabra.", "Di: listo")

    def _call(self, system: str, text: str, images: list[bytes] | None, json_mode: bool) -> str:
        from google.genai import types

        parts = [types.Part.from_bytes(data=b, mime_type="image/png") for b in (images or [])] + [text]
        config = types.GenerateContentConfig(
            system_instruction=system, temperature=0.2, max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json" if json_mode else None,
        )
        delay, attempt = 5.0, 0
        while True:
            try:
                resp = self.client.models.generate_content(model=self.model, contents=parts, config=config)
                if not resp.text:
                    raise RuntimeError(f"Gemini no devolvió texto (bloqueo o corte): {getattr(resp, 'prompt_feedback', '')}")
                return resp.text
            except Exception as e:
                code = getattr(e, "code", None)
                wait = _retry_seconds(e) if code == 429 else None
                if code == 429 and wait and wait > 90:
                    # Cuota agotada (p. ej. 20 peticiones/día en el plan gratuito): esperar segundos no sirve.
                    if self._switch_model(f"Cuota agotada en «{self.model}» (vuelve en ≈ {_humanize(wait)})"):
                        continue
                    raise RuntimeError(
                        f"Gemini: se agotó la cuota gratuita del modelo «{self.model}»; vuelve a estar disponible en "
                        f"≈ {_humanize(wait)}.\nDetalle: {getattr(e, 'message', None) or e}\n"
                        "Opciones: esperar, probar otro modelo (llm.list_models(); GeminiLLM('nombre')), o activar "
                        "facturación en Google AI Studio (se puede fijar un límite de gasto).") from e
                if code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    attempt += 1
                    pause = (wait + 1) if wait else delay
                    print(f"  Gemini {code}: reintento en {pause:.0f}s (límite del plan gratuito o saturación)")
                    time.sleep(pause)
                    delay = min(delay * 2, 90)
                    continue
                if code == 404 and self._switch_model(f"El modelo «{self.model}» no está disponible"):
                    continue
                if code is not None:
                    raise RuntimeError(f"Gemini devolvió el error {code} con el modelo «{self.model}»: "
                                       f"{getattr(e, 'message', None) or e}\n{_HINTS.get(code, '')}") from e
                raise

    def generate_text(self, system: str, user: str, images: list[bytes] | None = None) -> str:
        return self._call(system, user, images, json_mode=False).strip()

    def generate_json(self, system: str, content: list[dict], schema: dict) -> dict:
        import base64

        document = "\n\n".join(b["text"] for b in content if b["type"] == "text")
        images = [base64.b64decode(b["source"]["data"]) for b in content if b["type"] == "image"]
        example = json.dumps(_schema_example(schema), ensure_ascii=False, indent=2)
        user = (f"{document}\n\n---\nResponde ÚNICAMENTE con un objeto JSON válido con exactamente esta estructura; "
                f"en los campos con opciones separadas por | elige UNA:\n{example}")
        last = None
        for _ in range(self.parse_retries + 1):
            raw = self._call(system, user, images, json_mode=True)
            try:
                return coerce(extract_json(raw), schema)
            except (ValueError, json.JSONDecodeError) as e:
                last = e
        raise RuntimeError(f"Gemini no devolvió JSON válido tras {self.parse_retries + 1} intentos: {last}")
