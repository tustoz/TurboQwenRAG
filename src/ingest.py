import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(file_path: str) -> list:
    import fitz
    docs = []
    pdf = fitz.open(file_path)
    for page_num in range(len(pdf)):
        text = pdf[page_num].get_text("text").strip()
        if text:
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "filename": Path(file_path).name,
                    "page": page_num + 1,
                    "total_pages": len(pdf),
                    "file_type": "pdf",
                }
            ))
    pdf.close()
    return docs


def load_text(file_path: str) -> list:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip()
    if not content:
        return []
    return [Document(
        page_content=content,
        metadata={
            "source": file_path,
            "filename": Path(file_path).name,
            "page": 1,
            "file_type": Path(file_path).suffix.lstrip("."),
        }
    )]


def load_documents(data_dir: str = "./data/docs") -> list:
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Directory {data_dir} created.")
        return []

    loaders = {".pdf": load_pdf, ".txt": load_text, ".md": load_text}
    all_docs = []

    for file_path in Path(data_dir).rglob("*"):
        ext = file_path.suffix.lower()
        if ext not in loaders:
            continue
        try:
            docs = loaders[ext](str(file_path))
            all_docs.extend(docs)
            print(f"Loaded {len(docs)} page(s) from {file_path.name}")
        except Exception as e:
            print(f"Failed {file_path.name}: {e}")

    print(f"\\nTotal: {len(all_docs)} pages")
    return all_docs


def chunk_documents(documents: list, chunk_size: int = 512, chunk_overlap: int = 64) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\\n\\n", "\\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_total"] = len(chunks)
    print(f"{len(documents)} pages to {len(chunks)} chunks")
    return chunks