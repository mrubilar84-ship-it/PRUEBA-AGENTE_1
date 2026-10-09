# Agente de revisión de documentos de ingeniería

Agente basado en Claude que revisa documentos técnicos (memorias de cálculo, especificaciones, listas de materiales, procedimientos, informes) contra un checklist y entrega:

- `informe.md`: informe legible con resumen ejecutivo, hallazgos por severidad e inconsistencias entre documentos.
- `hallazgos.csv`: todos los hallazgos en tabla (abre en Excel).
- `revision.json`: salida estructurada completa.

Formatos: planos PDF/PNG/JPG (visión), PDF (nativo con Claude; texto extraído con el modelo local), DOCX, XLSX/XLSM (valores y fórmulas), CSV, TXT, MD.

## Tres opciones de modelo
| | Modelo local gratis (Qwen2.5) | API de Claude | API de Gemini |
|---|---|---|---|
| Costo | Gratis (GPU de Kaggle) | De pago | Clave gratis con límites bajos, o de pago |
| Notebook | `kaggle_revision_documentos.ipynb` | `kaggle_revision_documentos_claude.ipynb` | `kaggle_revision_documentos_gemini.ipynb` |
| Secreto en Kaggle | (ninguno) | `ANTHROPIC_API_KEY` | `GEMINI_API_KEY` |
| GPU | Sí (T4 x2) | No | No |
| PDF | Solo texto (escaneados no) | Nativo, con tablas y figuras | Texto extraído + imágenes en planos |
| Planos | Con `LocalVLM` (visión local, limitada en planos densos) | Visión de alta calidad | Visión (modelo en la nube) |
| Calidad | Buena para checklist y cifras simples; puede errar en cálculos largos | Mayor | Alta (según el modelo elegido) |
| Documentos largos | Se revisan por partes y se fusionan | Una sola pasada | Una sola pasada |
| Confidencialidad | Nada sale de Kaggle | Datos a la API de Anthropic | En el plan gratuito Google puede usar los datos para mejorar sus productos: **no usar con documentos confidenciales** |

## Ejecutar en Kaggle (modo gratis)
1. *Settings → Accelerator*: GPU T4 x2; *Internet: On* (solo para bajar el modelo).
2. Dataset con tus documentos + Dataset con `doc_review_agent.py`, `plan_review.py`, `local_llm.py`, `doc_chat.py`, `checklist_default.md`, `checklist_planos.md`.
3. Importa `kaggle_revision_documentos.ipynb`, ajusta `IN_DIR` y ejecuta todo.

## Ejecutar con Claude (de pago)
1. Mismos datasets que arriba; importa `kaggle_revision_documentos_claude.ipynb`.
2. *Settings → Internet: On*; en *Add-ons → Secrets* agrega `ANTHROPIC_API_KEY`.
3. Ajusta `IN_DIR`, la norma y el contexto en `ReviewConfig`, y ejecuta todo.

## Ejecutar con Gemini (clave gratis o de pago)
1. Crea una clave en [Google AI Studio](https://aistudio.google.com/apikey).
2. *Settings → Internet: On* (no necesitas GPU); en *Add-ons → Secrets* agrega `GEMINI_API_KEY`.
3. Mismos datasets que arriba, más `gemini_llm.py`; importa `kaggle_revision_documentos_gemini.ipynb`.
4. Ajusta `IN_DIR`, la norma y el contexto en `ReviewConfig`, y ejecuta todo. El modelo por defecto es `gemini-3.8-flash`; cámbialo con `GeminiLLM("modelo")` o la variable `GEMINI_MODEL` (verifica en AI Studio los modelos disponibles).
5. **Cuota gratuita:** puede ser muy baja (p. ej. 20 peticiones por día por modelo). Cada documento, cada hoja de plano y cada pregunta del chat gasta una. Si se agota, el agente cambia solo a otro modelo «flash» disponible (cada modelo tiene su propia cuota) o, si no hay otro, dice en cuánto tiempo vuelve. Facturación en AI Studio (con límite de gasto) elimina el tope y excluye el uso de tus datos para mejorar productos.
6. En el plan gratuito los límites son bajos y cambian (el agente espera y reintenta ante errores 429), y Google puede usar los datos enviados para mejorar sus productos: no lo uses con documentos confidenciales sin activar facturación.

## Administrar documentos y revisar solo documentos
`doc_manager.py` (`DocFolder`) trabaja sobre una carpeta de trabajo (`/kaggle/working/documentos`), porque los datasets de Kaggle son de solo lectura:
```python
docs = DocFolder()
docs.add("documento-1")                 # copia los documentos de un dataset (nombre, URL o carpeta); varios a la vez
docs.add("/kaggle/input/x/a.pdf", "/kaggle/input/x/b.docx")
docs.upload()                           # botón para subir archivos desde tu computador
docs.replace("viejo.pdf", "nuevo.pdf")  # reemplazar uno por otro
docs.remove("*.xlsx")                   # borrar (admite comodines)
docs.list(); docs.clear(confirm=True)
```
Para cambiar un archivo dentro de un dataset de Kaggle, créale una nueva versión desde su página (⋮ → New version).
**Solo documentos:** `ReviewConfig(plan_mode="skip")` omite (con aviso) los PDF que parecen planos y las imágenes; `plan_mode="never"` lee todo PDF como documento de texto e ignora las imágenes. Los notebooks vienen con `"skip"`.

## Planos de ingeniería (PDF impresos desde DWG)
`plan_review.py` revisa **cada hoja mirando su imagen**: vista general, cajetín ampliado y 4 cuadrantes ampliados, más el texto vectorial del PDF (si el CAD lo imprimió como texto). Entrega por hoja: datos del cajetín (n° de plano, revisión, escala, responsables…) y hallazgos según `checklist_planos.md` (cajetín, control de revisiones, escala, cotas y cadenas que no cierran, cortes/detalles sin referencia, simbología y notas, lista de materiales, legibilidad). En multi-hoja compara numeración y revisiones entre planos.
- Detección automática: PDF de formato ≥ A3 o con «plano/lámina/dwg» en el nombre (`ReviewConfig(plan_mode="always")` para forzar; `"never"` para desactivar). También acepta PNG/JPG/TIF.
- Modo gratis: `LocalVLM("Qwen/Qwen2.5-VL-7B-Instruct")` (visión local en Kaggle). Es más lento (varios minutos por hoja con cuadrantes; `plan_tiles=False` acelera) y **lee peor los textos pequeños** en planos densos. Con un modelo solo-texto (`LocalLLM`) solo se revisa el texto vectorial y se avisa.
- Modo Claude: lee planos con mucha más fiabilidad.
- **Interpretar y preguntar:** `interpret_folder(IN_DIR, OUT, cfg, llm=llm)` extrae de cada hoja el tipo de plano, cajetín, elementos, cotas (valor, unidad, ubicación), materiales y especificaciones, notas, referencias, vistas/cortes y lo no legible; guarda `planos_interpretados.md/json`. Luego `chat.add_plans(planos)` y `chat.ask("¿Cuál es la luz total de la viga V-12?")`: el agente busca en la interpretación y, si el modelo ve imágenes, **adjunta la hoja relevante** (la que nombras: «hoja 2», n° de plano; si no, las mejor puntuadas) para verificar lo que ve. Responde citando de dónde sale cada dato y señala si interpretación e imagen difieren.
- Preguntas visuales puntuales sobre una hoja: `chat.ask_image("¿Las cotas suman el total?", "plano.pdf", page=1)`.
- Limitaciones: no mide sobre la imagen ni lee el DWG (solo el PDF); los hallazgos hay que verificarlos en el DWG. Un PDF con el texto convertido a líneas o escaneado depende solo de la imagen (lectura menos confiable).
- Prueba: `samples/plano_ejemplo.pdf` (cotas 100+200+250 ≠ 600, cajetín sin revisión/revisó/aprobó, corte A-A sin vista); se regenera con `samples/generar_plano_ejemplo.py`.

## Chat con los documentos y planos (preguntas y resúmenes)
Después de la revisión, el mismo notebook tiene un chat (`doc_chat.py`):
```python
chat = DocChat(llm); chat.add_folder(IN_DIR); chat.add_review("/kaggle/working/informe/revision.json")
chat.ask("¿Qué norma se aplica y con qué factor de seguridad?")   # responde con citas [archivo, p. N]
chat.summarize("memoria.pdf")                                      # resumen (por tramos si es largo)
chat.repl()                                                        # chat interactivo
```
Cómo entiende el contexto: en cada pregunta el modelo recibe (1) **la revisión completa** (resúmenes, hallazgos por severidad, inconsistencias), (2) **los documentos** (completos si caben en ~30.000 caracteres; si no, los pasajes más relevantes por búsqueda BM25) y (3) la conversación previa. Está instruido para razonar con lógica de ingeniería, separar lo que dice el documento de su inferencia, citar `[archivo, p. N]` / `[hallazgo H-02]` y decir qué dato falta cuando no puede concluir. Ajustable con `DocChat(llm, full_context_chars=..., top_k=...)`.

Para respuestas más razonadas en modo gratis usa un modelo mayor: `LocalLLM("Qwen/Qwen2.5-14B-Instruct")` (4 bits, ~9 GB, cabe en T4 x2; más lento).

## Personalizar
- `checklist_default.md`: criterios y escala de severidad (puedes pasar otro con `ReviewConfig(checklist_path=...)`).
- `ReviewConfig`: `model` (por defecto `claude-opus-5-5`, o variable `DOC_REVIEW_MODEL`), `effort` (`low`…`max`), `extra_instructions`, `cross_check`.

## Local / prueba
```
pip install -r requirements.txt pytest
python -m pytest tests            # prueba offline con cliente simulado
ANTHROPIC_API_KEY=... python doc_review_agent.py samples /tmp/informe --contexto "AISC 360"
```
`samples/memoria_ejemplo.txt` tiene errores plantados (deflexión mal calculada, sin revisor, norma sin edición) para validar que el agente los detecta.

## Notas
- Un PDF admite hasta 30 MB / 600 páginas; si es mayor, divídelo.
- Los documentos se envían a la API de Anthropic: no subas información confidencial sin autorización.
- La revisión es una ayuda; no reemplaza la revisión y firma de un ingeniero responsable.
- Ni el modelo local (sin GPU aquí) ni la API de Claude (sin clave) se ejecutaron en esta sesión; solo las pruebas offline.
