# Agente de revisión de documentos de ingeniería

Agente basado en Claude que revisa documentos técnicos (memorias de cálculo, especificaciones, listas de materiales, procedimientos, informes) contra un checklist y entrega:

- `informe.md`: informe legible con resumen ejecutivo, hallazgos por severidad e inconsistencias entre documentos.
- `hallazgos.csv`: todos los hallazgos en tabla (abre en Excel).
- `revision.json`: salida estructurada completa.

Formatos: PDF (se envía nativo, Claude ve tablas y figuras), DOCX, XLSX/XLSM (valores y fórmulas), CSV, TXT, MD.

## Ejecutar en Kaggle
1. Crea un Dataset con tus documentos y otro (o el mismo) con `doc_review_agent.py` y `checklist_default.md`.
2. Importa `kaggle_revision_documentos.ipynb` como notebook.
3. *Settings → Internet: On*; en *Add-ons → Secrets* agrega `ANTHROPIC_API_KEY`.
4. Ajusta `IN_DIR`, la norma y el contexto en `ReviewConfig`, y ejecuta todo.

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
- La ejecución real contra la API no se probó en esta sesión (sin clave); solo la prueba offline.
