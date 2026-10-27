import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "markdown"
OUTPUT_DIR = PROJECT_ROOT / "data" / "chunks"
MAX_MARKDOWN_BYTES = 20 * 1024 * 1024
MAX_CHUNK_CHARACTERS = 1200
CHUNK_OVERLAP_CHARACTERS = 150


def _project_directory(path: Path) -> Path:
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"Directory must be inside the project: {path}") from error
    return resolved_path


def _azure_embeddings() -> OpenAIEmbeddings:
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "").strip()

    missing = [
        name
        for name, value in (
            ("AZURE_OPENAI_API_KEY", api_key),
            ("AZURE_OPENAI_ENDPOINT", endpoint),
            ("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", deployment),
        )
        if not value
    ]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"Missing required Azure OpenAI setting(s): {names}")

    parsed_endpoint = urlsplit(endpoint)
    if (
        parsed_endpoint.scheme != "https"
        or not parsed_endpoint.netloc
        or parsed_endpoint.username
        or parsed_endpoint.password
        or parsed_endpoint.path.rstrip("/") != "/openai/v1"
        or parsed_endpoint.query
        or parsed_endpoint.fragment
    ):
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT must be the HTTPS Azure OpenAI v1 endpoint "
            "ending in /openai/v1, without credentials, query, or fragment."
        )

    return OpenAIEmbeddings(
        model=deployment,
        base_url=f"{endpoint.rstrip('/')}/",
        api_key=api_key,
    )


def _write_jsonl_atomically(output_path: Path, records: list[dict[str, object]]) -> None:
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.",
        suffix=".tmp",
        dir=output_path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def chunk_markdown_files(embeddings: Embeddings) -> int:
    input_dir = _project_directory(INPUT_DIR)
    output_dir = _project_directory(OUTPUT_DIR)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Markdown input directory does not exist: {input_dir}")

    markdown_files = sorted(input_dir.glob("*.md"))
    if not markdown_files:
        raise FileNotFoundError(f"No Markdown files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = _project_directory(output_dir)

    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_chunk_size=150,
    )
    size_limiter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_CHARACTERS,
        chunk_overlap=CHUNK_OVERLAP_CHARACTERS,
    )

    total_chunks = 0
    for markdown_path in markdown_files:
        source_path = markdown_path.resolve()
        try:
            source_path.relative_to(input_dir)
        except ValueError as error:
            raise ValueError(f"Markdown file resolves outside the input directory: {markdown_path}") from error

        file_size = source_path.stat().st_size
        if file_size > MAX_MARKDOWN_BYTES:
            raise ValueError(
                f"{source_path.name} is {file_size} bytes; the limit is "
                f"{MAX_MARKDOWN_BYTES} bytes."
            )

        text = source_path.read_text(encoding="utf-8")
        if not text.strip():
            print(f"Skipping empty Markdown file: {source_path.name}")
            continue

        semantic_documents = semantic_splitter.create_documents([text])
        chunks = size_limiter.split_documents(semantic_documents)
        records = [
            {
                "text": chunk.page_content,
                "metadata": {
                    "source": source_path.name,
                    "chunk_index": chunk_index,
                },
            }
            for chunk_index, chunk in enumerate(chunks)
            if chunk.page_content.strip()
        ]

        output_path = output_dir / f"{source_path.stem}.jsonl"
        _write_jsonl_atomically(output_path, records)
        total_chunks += len(records)
        print(f"Created {len(records)} chunks from {source_path.name}")

    return total_chunks


def main() -> None:
    total_chunks = chunk_markdown_files(_azure_embeddings())
    print(f"Finished semantic chunking: {total_chunks} chunks saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()