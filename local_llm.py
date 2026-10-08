"""Backend LLM local y gratuito para el agente (corre en la GPU de Kaggle, sin API ni clave).

Carga un modelo abierto (por defecto Qwen2.5-Instruct) con transformers, opcionalmente en 4 bits
(bitsandbytes) para que quepa en una T4 de 16 GB.

    from local_llm import LocalLLM
    llm = LocalLLM("/kaggle/input/qwen2.5/transformers/7b-instruct/1")   # o "Qwen/Qwen2.5-7B-Instruct"
    review_folder(IN_DIR, OUT_DIR, cfg, llm=llm)
"""
from __future__ import annotations

import base64
import json
import re
import unicodedata


def extract_json(text: str) -> dict:
    """Extrae el primer objeto JSON balanceado de un texto (tolera ```json y texto alrededor)."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError("la respuesta no contiene JSON")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("JSON incompleto (posible corte por max_new_tokens)")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()
    return re.sub(r"[\s-]+", "_", s)


def coerce(value, schema: dict):
    """Ajusta la salida del modelo al esquema: tipos, claves faltantes, enums y claves extra."""
    t = schema.get("type")
    if t == "object":
        value = value if isinstance(value, dict) else {}
        props = schema["properties"]
        return {k: coerce(value.get(k), sub) for k, sub in props.items()}
    if t == "array":
        if value is None:
            value = []
        if not isinstance(value, list):
            value = [value]
        return [coerce(v, schema["items"]) for v in value]
    if "enum" in schema:
        by_norm = {_norm(e): e for e in schema["enum"]}
        n = _norm(value) if value is not None else ""
        if n in by_norm:
            return by_norm[n]
        for k, e in by_norm.items():  # coincidencias parciales: "crítica grave" -> critica
            if k and (k in n or n in k) and n:
                return e
        # Valor desconocido: el más leve para severidades; "requiere_revision" para evaluación global.
        return "requiere_revision" if "requiere_revision" in schema["enum"] else schema["enum"][-1]
    if t == "string":
        return "" if value is None else str(value)
    return value


def _schema_example(schema: dict):
    t = schema.get("type")
    if t == "object":
        return {k: _schema_example(v) for k, v in schema["properties"].items()}
    if t == "array":
        return [_schema_example(schema["items"])]
    if "enum" in schema:
        return " | ".join(schema["enum"])
    return "texto"


class LocalLLM:
    native_pdf = False  # los modelos locales leen texto: los PDF se extraen con pypdf
    vision = False
    # ~4k tokens por parte: deja espacio al checklist, al esquema y a la respuesta.
    chunk_chars = 14000

    def __init__(self, model: str = "Qwen/Qwen2.5-7B-Instruct", load_4bit: bool = True,
                 max_new_tokens: int = 3000, retries: int = 2):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.max_new_tokens, self.retries = max_new_tokens, retries
        self.tok = AutoTokenizer.from_pretrained(model)
        kwargs = {"device_map": "auto"}
        if load_4bit and torch.cuda.is_available():
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        self.model.eval()

    def _chat(self, system: str, user: str, images=None) -> str:
        import torch

        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        ids = ids.to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                repetition_penalty=1.05, pad_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    def generate_text(self, system: str, user: str, images: list[bytes] | None = None) -> str:
        return self._chat(system, user, images).strip()

    def generate_json(self, system: str, content: list[dict], schema: dict) -> dict:
        document = "\n\n".join(b["text"] for b in content if b["type"] == "text")
        images = [base64.b64decode(b["source"]["data"]) for b in content if b["type"] == "image"]
        example = json.dumps(_schema_example(schema), ensure_ascii=False, indent=2)
        user = (
            f"{document}\n\n---\nResponde ÚNICAMENTE con un objeto JSON válido (sin texto adicional ni "
            f"bloques de código) con exactamente esta estructura; en los campos con opciones separadas "
            f"por | elige UNA:\n{example}"
        )
        last_err = None
        for _ in range(self.retries + 1):
            raw = self._chat(system, user, images or None)
            try:
                return coerce(extract_json(raw), schema)
            except (ValueError, json.JSONDecodeError) as e:
                last_err = e
        raise RuntimeError(f"El modelo local no devolvió JSON válido tras {self.retries + 1} intentos: {last_err}")


class LocalVLM(LocalLLM):
    """Modelo local con visión (Qwen2.5-VL) para revisar planos, gratis en la GPU de Kaggle.

    Sirve también para los documentos de texto. Las imágenes se pasan en el orden recibido.
    """
    vision = True
    image_max_side = 1400  # px del lado mayor por imagen (≈1.2 Mpx -> ~1.5k tokens visuales)
    qa_tiles = False  # en el chat solo vista general + cajetín (rápido); la interpretación ya usó los cuadrantes

    def __init__(self, model: str = "Qwen/Qwen2.5-VL-7B-Instruct", load_4bit: bool = True,
                 max_new_tokens: int = 4500, retries: int = 2):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        self.max_new_tokens, self.retries = max_new_tokens, retries
        self.processor = AutoProcessor.from_pretrained(model, min_pixels=256 * 28 * 28, max_pixels=1700 * 28 * 28)
        self.tok = self.processor.tokenizer
        kwargs = {"device_map": "auto"}
        if load_4bit and torch.cuda.is_available():
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        else:
            kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **kwargs)
        self.model.eval()

    def _chat(self, system: str, user: str, images=None) -> str:
        import io

        import torch
        from PIL import Image

        pil = [Image.open(io.BytesIO(b)).convert("RGB") for b in (images or [])]
        content = [{"type": "image"} for _ in pil] + [{"type": "text", "text": user}]
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=pil or None, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      repetition_penalty=1.05)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
