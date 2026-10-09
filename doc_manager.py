"""Administrador de documentos para el notebook: agregar varios, borrar, reemplazar y subir desde tu computador.

Los datasets de Kaggle (/kaggle/input) son de solo lectura, así que el agente trabaja sobre una CARPETA DE TRABAJO
(/kaggle/working/documentos) a la que copias los documentos que quieres revisar:

    docs = DocFolder()
    docs.add("documento-1")                 # copia todos los documentos de un dataset (nombre, URL o carpeta)
    docs.add("/kaggle/input/otro/a.pdf", "/kaggle/input/otro/b.docx")   # o archivos sueltos
    docs.upload()                           # botón para subir archivos desde tu computador
    docs.list()                             # qué hay ahora
    docs.replace("a.pdf", "/kaggle/input/otro/a_v2.pdf")   # reemplaza un documento por otro
    docs.remove("b.docx")                   # borra uno (admite comodines: "*.xlsx")
    docs.clear(confirm=True)                # vacía la carpeta

Luego revisa con IN_DIR = docs.folder. Para cambiar un archivo DENTRO de un dataset de Kaggle, créale una nueva versión
desde la página del dataset (⋮ → New version).
"""
from __future__ import annotations

import fnmatch
import shutil
from datetime import datetime
from pathlib import Path

from doc_review_agent import SUPPORTED, is_project_file, resolve_dir

DEFAULT_FOLDER = "/kaggle/working/documentos"


def _save_uploaded(value, dest: Path) -> list[str]:
    """Guarda lo que entrega ipywidgets.FileUpload (formato v7: dict, v8: tupla de dicts). Devuelve los nombres."""
    items = value.items() if isinstance(value, dict) else [(u["name"], u) for u in value]
    names = []
    for name, info in items:
        name = Path(name).name  # sin rutas: evita escribir fuera de la carpeta
        if Path(name).suffix.lower() not in SUPPORTED:
            print(f"  ! {name}: formato no soportado, se omite")
            continue
        (dest / name).write_bytes(bytes(info["content"]))
        names.append(name)
    return names


class DocFolder:
    def __init__(self, folder: str | Path = DEFAULT_FOLDER):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    # ---- consultar
    def files(self) -> list[Path]:
        return sorted(p for p in self.folder.rglob("*") if p.is_file())

    def list(self) -> list[str]:
        files = self.files()
        if not files:
            print(f"La carpeta {self.folder} está vacía. Usa add(), upload() o replace().")
            return []
        print(f"{len(files)} documento(s) en {self.folder}:")
        for p in files:
            st = p.stat()
            print(f"  - {p.relative_to(self.folder)}  ({st.st_size / 1024:.0f} KB, "
                  f"{datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M})")
        return [str(p.relative_to(self.folder)) for p in files]

    # ---- agregar
    def add(self, *sources: str | Path, replace: bool = False) -> list[str]:
        """Copia documentos a la carpeta. Cada origen: archivo, carpeta, o nombre/URL de un dataset de Kaggle.
        Si ya existe un documento con el mismo nombre se conserva el actual, salvo replace=True."""
        added = []
        for src in sources:
            sp = Path(str(src))
            if sp.is_file():
                candidates = [sp]
            else:
                candidates = [p for p in sorted(resolve_dir(src).rglob("*")) if p.is_file()]
            for p in candidates:
                if p.suffix.lower() not in SUPPORTED or is_project_file(p):
                    continue
                dest = self.folder / p.name
                if dest.exists() and not replace:
                    print(f"  = {p.name} ya existe (usa replace=True o replace('{p.name}', origen) para reemplazarlo)")
                    continue
                verb = "reemplazado" if dest.exists() else "agregado"
                shutil.copy2(p, dest)
                print(f"  + {p.name} {verb}")
                added.append(p.name)
        if not added:
            print("No se agregó ningún documento nuevo.")
        return added

    def upload(self):
        """Muestra un botón para subir archivos desde tu computador (reemplaza si el nombre ya existe)."""
        import ipywidgets as w
        from IPython.display import display

        up = w.FileUpload(accept="", multiple=True, description="Elegir archivos")
        btn = w.Button(description="Guardar en la carpeta", button_style="success")
        out = w.Output()

        def save(_):
            with out:
                out.clear_output()
                names = _save_uploaded(up.value, self.folder)
                print(f"Guardados: {names}" if names else "No hay archivos válidos para guardar.")
                self.list()
        btn.on_click(save)
        display(w.VBox([up, btn, out]))

    # ---- reemplazar / borrar
    def replace(self, name: str, source: str | Path) -> None:
        """Reemplaza el documento `name` por el archivo `source` (queda con el nombre del nuevo archivo)."""
        sp = Path(str(source))
        if not sp.is_file():
            raise FileNotFoundError(f"No existe el archivo de reemplazo: {source}")
        data = sp.read_bytes()  # se lee antes de borrar: el origen podría ser el propio archivo a reemplazar
        self.remove(name, quiet=True)
        (self.folder / sp.name).write_bytes(data)
        print(f"  ~ {name} reemplazado por {sp.name}")

    def remove(self, *names: str, quiet: bool = False) -> list[str]:
        """Borra documentos por nombre; admite comodines (p. ej. "*.xlsx")."""
        removed = []
        for pat in names:
            hits = [p for p in self.files() if fnmatch.fnmatch(p.name, pat) or str(p.relative_to(self.folder)) == pat]
            if not hits and not quiet:
                print(f"  ? no encontré «{pat}» (usa list() para ver los nombres)")
            for p in hits:
                p.unlink()
                removed.append(p.name)
                if not quiet:
                    print(f"  - {p.name} borrado")
        return removed

    def clear(self, confirm: bool = False) -> None:
        """Vacía la carpeta. Requiere confirm=True para evitar borrados accidentales."""
        if not confirm:
            print(f"Esto borraría {len(self.files())} documento(s) de {self.folder}. Repite con clear(confirm=True).")
            return
        for p in self.files():
            p.unlink()
        print("Carpeta vaciada.")
