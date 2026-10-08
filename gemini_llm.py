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
import time

from local_llm import _schema_example, coerce, extract_json

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"  # verifica en AI Studio qué modelos tienes disponibles


_HINTS = {
    400: "Pista: revisa que la clave GEMINI_API_KEY sea válida y que la petición no sea demasiado grande.",
    401: "Pista: la clave GEMINI_API_KEY no es válida.",
    403: "Pista: la clave no tiene permiso para este modelo o el proyecto no tiene acceso gratuito. "
         "Crea otra clave en AI Studio o prueba otro modelo (llm.list_models()).",
    404: "Pista: ese modelo no existe o ya no está disponible. Ejecuta llm.list_models() y usa uno de la lista: "
         "GeminiLLM('nombre').",
    429: "Pista: agotaste la cuota del plan gratuito (por minuto o por día). Espera, usa otro modelo, o activa facturación.",
}


class GeminiLLM:
    native_pdf = False  # el agente extrae el texto del PDF; las imágenes de planos van aparte
    vision = True
    image_max_side = 2000
    qa_tiles = True
    chunk_chars = 0  # contexto de ~1M tokens: sin troceado

    def __init__(self, model: str | None = None, api_key: str | None = None, client=None,
                 max_output_tokens: int = 16000, retries: int = 6, parse_retries: int = 2):
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.max_output_tokens, self.retries, self.parse_retries = max_output_tokens, retries, parse_retries
        if client is None:
            from google import genai

            key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError("Falta la clave: define GEMINI_API_KEY (en Kaggle: Add-ons → Secrets).")
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
        delay = 5.0
        for attempt in range(self.retries + 1):
            try:
                resp = self.client.models.generate_content(model=self.model, contents=parts, config=config)
                if not resp.text:
                    raise RuntimeError(f"Gemini no devolvió texto (bloqueo o corte): {getattr(resp, 'prompt_feedback', '')}")
                return resp.text
            except Exception as e:
                code = getattr(e, "code", None)
                if code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    print(f"  Gemini {code}: reintento en {delay:.0f}s (límite del plan gratuito o saturación)")
                    time.sleep(delay)
                    delay = min(delay * 2, 90)
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
