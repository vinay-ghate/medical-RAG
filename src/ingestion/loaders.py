"""Document loaders for medical PDFs, JSON files, and web/text documents."""

import json
from pathlib import Path
from typing import List, Union
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from pypdf import PdfReader
from src.config import setup_logger

logger = setup_logger("ingestion.loaders")


def load_pdf(file_path: Union[str, Path]) -> List[Document]:
    """Load a PDF file and extract pages as LangChain Documents with metadata."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    docs: List[Document] = []
    reader = PdfReader(str(path))
    stem = path.stem.replace("_", " ").title()

    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path.as_posix()),
                    "drug": stem,
                    "section": f"Page {idx + 1}",
                    "date": "2024-01-01",
                    "doc_type": "guideline_pdf",
                    "page_number": idx + 1,
                },
            )
        )
    logger.info("Loaded %d pages from PDF %s", len(docs), path.name)
    return docs


def load_json_records(file_path: Union[str, Path]) -> List[Document]:
    """Load structured medical JSON records (e.g. openFDA or DailyMed extracts)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    docs: List[Document] = []
    for item in data:
        drug_name = item.get("drug", item.get("name", "Unknown"))
        sections = item.get("sections", {})
        date = item.get("date", "2024-01-01")
        source = item.get("source", str(path.as_posix()))

        if isinstance(sections, dict):
            for section_name, content in sections.items():
                if isinstance(content, list):
                    content = "\n".join(content)
                if content and str(content).strip():
                    docs.append(
                        Document(
                            page_content=str(content).strip(),
                            metadata={
                                "source": source,
                                "drug": drug_name,
                                "section": section_name,
                                "date": date,
                                "doc_type": item.get("doc_type", "drug_label"),
                            },
                        )
                    )
        elif "content" in item:
            docs.append(
                Document(
                    page_content=str(item["content"]).strip(),
                    metadata={
                        "source": source,
                        "drug": drug_name,
                        "section": item.get("section", "General"),
                        "date": date,
                        "doc_type": item.get("doc_type", "drug_label"),
                    },
                )
            )

    logger.info("Loaded %d sections from JSON %s", len(docs), path.name)
    return docs


def load_html_or_text(file_path: Union[str, Path]) -> List[Document]:
    """Load and parse HTML or plain text/markdown medical documents."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    if path.suffix.lower() in [".html", ".htm"]:
        soup = BeautifulSoup(raw_text, "html.parser")
        text = soup.get_text(separator="\n").strip()
    else:
        text = raw_text.strip()

    stem = path.stem.replace("_", " ").title()
    doc = Document(
        page_content=text,
        metadata={
            "source": str(path.as_posix()),
            "drug": stem,
            "section": "Overview",
            "date": "2024-01-01",
            "doc_type": "article",
        },
    )
    logger.info("Loaded text document %s (%d chars)", path.name, len(text))
    return [doc]


def load_directory(dir_path: Union[str, Path]) -> List[Document]:
    """Scan and load all supported documents from a directory."""
    path = Path(dir_path)
    if not path.exists():
        return []

    all_docs: List[Document] = []
    for file in sorted(path.rglob("*")):
        if file.is_file():
            suffix = file.suffix.lower()
            if suffix == ".pdf":
                all_docs.extend(load_pdf(file))
            elif suffix == ".json":
                all_docs.extend(load_json_records(file))
            elif suffix in [".txt", ".md", ".html", ".htm"]:
                all_docs.extend(load_html_or_text(file))

    logger.info("Loaded total %d raw document sections from %s", len(all_docs), path)
    return all_docs
