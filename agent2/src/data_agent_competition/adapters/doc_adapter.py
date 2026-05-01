from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree


@dataclass(frozen=True, slots=True)
class TextDocument:
    path: Path
    text: str


def load_document_text(path: Path) -> TextDocument:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return TextDocument(path=path, text=path.read_text(encoding="utf-8"))
    if suffix == ".docx":
        return TextDocument(path=path, text=_load_docx_text(path))
    if suffix == ".pdf":
        return TextDocument(path=path, text=_load_pdf_text(path))
    raise ValueError(f"Unsupported document type: {path}")


def chunk_document(text: str, *, max_chars: int = 1200) -> tuple[str, ...]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ()
    return tuple(
        normalized[index : index + max_chars].strip()
        for index in range(0, len(normalized), max_chars)
    )


def _load_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml_bytes)
    text_nodes = [node.text for node in root.iter() if node.text]
    return "\n".join(text_nodes)


def _load_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PDF support requires pypdf to be installed.") from exc
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()
