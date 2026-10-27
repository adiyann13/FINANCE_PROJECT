import json
import os
import re
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from urllib.parse import urlsplit

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

if __package__:
    from .semantic_chunker import (
        OUTPUT_DIR,
        PROJECT_ROOT,
        _azure_embeddings,
        chunk_markdown_files,
    )
else:
    from semantic_chunker import (
        OUTPUT_DIR,
        PROJECT_ROOT,
        _azure_embeddings,
        chunk_markdown_files,
    )


VECTOR_FIELD = "content_vector"
VECTOR_PROFILE = "finance-vector-profile"
VECTOR_ALGORITHM = "finance-hnsw"
VECTOR_DIMENSIONS = 1536
EMBEDDING_BATCH_SIZE = 64
SEARCH_UPLOAD_BATCH_SIZE = 500
DOCUMENT_ID_NAMESPACE = uuid.UUID("e1d14a7c-8c14-4e82-8a40-5c91bb4f4221")


def _required_setting(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required setting: {name}")
    return value


def _search_endpoint() -> str:
    endpoint = _required_setting("AZURE_SEARCH_ENDPOINT")
    parsed_endpoint = urlsplit(endpoint)
    if (
        parsed_endpoint.scheme != "https"
        or not parsed_endpoint.netloc
        or parsed_endpoint.username
        or parsed_endpoint.password
        or parsed_endpoint.path not in ("", "/")
        or parsed_endpoint.query
        or parsed_endpoint.fragment
    ):
        raise ValueError(
            "AZURE_SEARCH_ENDPOINT must be an HTTPS Azure AI Search service URL "
            "without credentials, path, query, or fragment."
        )
    return endpoint.rstrip("/")


def _create_index(index_name: str) -> SearchIndex:
    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
        ),
        SimpleField(
            name="source",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SimpleField(
            name="chunk_index",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),
        SearchField(
            name=VECTOR_FIELD,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=VECTOR_DIMENSIONS,
            vector_search_profile_name=VECTOR_PROFILE,
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name=VECTOR_ALGORITHM)],
        profiles=[
            VectorSearchProfile(
                name=VECTOR_PROFILE,
                algorithm_configuration_name=VECTOR_ALGORITHM,
            )
        ],
    )
    return SearchIndex(name=index_name, fields=fields, vector_search=vector_search)


def _validate_existing_index(index: object, index_name: str) -> None:
    fields = {field.name: field for field in getattr(index, "fields", [])}
    required_fields = {"id", "content", "source", "chunk_index", VECTOR_FIELD}
    missing_fields = sorted(required_fields - fields.keys())
    if missing_fields:
        raise ValueError(
            f"Azure AI Search index '{index_name}' is missing required field(s): "
            f"{', '.join(missing_fields)}. Add these fields or use a dedicated index."
        )

    if not getattr(fields["id"], "key", False):
        raise ValueError(f"Index '{index_name}' must define 'id' as its key field.")

    vector_field = fields[VECTOR_FIELD]
    actual_dimensions = getattr(vector_field, "vector_search_dimensions", None)
    if actual_dimensions != VECTOR_DIMENSIONS:
        raise ValueError(
            f"Index '{index_name}' vector field '{VECTOR_FIELD}' has "
            f"{actual_dimensions} dimensions; this embedding model returns "
            f"{VECTOR_DIMENSIONS}. Configure a matching index and embedding model."
        )

    vector_profile = getattr(vector_field, "vector_search_profile_name", None)
    profiles = {
        profile.name
        for profile in getattr(getattr(index, "vector_search", None), "profiles", [])
    }
    if not vector_profile or vector_profile not in profiles:
        raise ValueError(
            f"Index '{index_name}' vector field '{VECTOR_FIELD}' must reference "
            "a configured vector search profile."
        )


def _ensure_index(index_client: SearchIndexClient, index_name: str) -> None:
    try:
        existing_index = index_client.get_index(index_name)
    except ResourceNotFoundError:
        index_client.create_index(_create_index(index_name))
        print(f"Created vector index: {index_name}")
        return

    _validate_existing_index(existing_index, index_name)


def _iter_jsonl_chunks(chunk_file: Path) -> Iterator[dict[str, object]]:
    with chunk_file.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {chunk_file.name} at line {line_number}."
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {chunk_file.name} at line {line_number}."
                )
            text = record.get("text")
            metadata = record.get("metadata")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Missing non-empty text in {chunk_file.name} at line {line_number}."
                )
            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Missing metadata in {chunk_file.name} at line {line_number}."
                )

            source = metadata.get("source")
            chunk_index = metadata.get("chunk_index")
            if (
                not isinstance(source, str)
                or not source
                or Path(source).name != source
                or not source.lower().endswith(".md")
            ):
                raise ValueError(
                    f"Invalid source filename in {chunk_file.name} at line {line_number}."
                )
            if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
                raise ValueError(
                    f"Invalid chunk index in {chunk_file.name} at line {line_number}."
                )

            yield {"text": text, "source": source, "chunk_index": chunk_index}


def _batched(items: Sequence[dict[str, object]], batch_size: int) -> Iterator[Sequence[dict[str, object]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _document_id(source: str, chunk_index: int) -> str:
    return str(uuid.uuid5(DOCUMENT_ID_NAMESPACE, f"{source}\0{chunk_index}"))


def _check_indexing_results(results: Sequence[object]) -> None:
    failures = [
        result
        for result in results
        if not getattr(result, "succeeded", False)
    ]
    if failures:
        details = "; ".join(
            f"{getattr(result, 'key', '<unknown key>')}: "
            f"{getattr(result, 'error_message', 'unknown indexing error')}"
            for result in failures[:5]
        )
        raise RuntimeError(
            f"Azure AI Search failed to index {len(failures)} document(s): {details}"
        )


def _remove_stale_documents(
    search_client: SearchClient,
    source: str,
    current_ids: set[str],
) -> int:
    escaped_source = source.replace("'", "''")
    existing_ids = {
        str(result["id"])
        for result in search_client.search(
            search_text="*",
            filter=f"source eq '{escaped_source}'",
            select=["id"],
        )
    }
    stale_ids = sorted(existing_ids - current_ids)
    deleted_count = 0
    for start in range(0, len(stale_ids), SEARCH_UPLOAD_BATCH_SIZE):
        batch = [{"id": document_id} for document_id in stale_ids[start : start + SEARCH_UPLOAD_BATCH_SIZE]]
        results = search_client.delete_documents(documents=batch)
        _check_indexing_results(results)
        deleted_count += len(batch)
    return deleted_count


def _ingest_chunks(
    search_client: SearchClient,
    embeddings: Embeddings,
    chunk_files: list[Path],
) -> int:
    total_uploaded = 0
    for chunk_file in chunk_files:
        records = list(_iter_jsonl_chunks(chunk_file))
        if not records:
            print(f"Skipping empty chunk file: {chunk_file.name}")
            continue

        source_names = {str(record["source"]) for record in records}
        if len(source_names) != 1:
            raise ValueError(f"{chunk_file.name} contains chunks from multiple source files.")
        source = source_names.pop()
        if chunk_file.stem != Path(source).stem:
            raise ValueError(
                f"Chunk file {chunk_file.name} does not match its source document {source}."
            )

        current_ids: set[str] = set()
        for record_batch in _batched(records, EMBEDDING_BATCH_SIZE):
            texts = [str(record["text"]) for record in record_batch]
            vectors = embeddings.embed_documents(texts)
            if len(vectors) != len(record_batch):
                raise RuntimeError("Embedding service returned an unexpected number of vectors.")

            documents: list[dict[str, object]] = []
            for record, vector in zip(record_batch, vectors, strict=True):
                if len(vector) != VECTOR_DIMENSIONS:
                    raise ValueError(
                        f"Embedding deployment returned {len(vector)} dimensions; "
                        f"the search index expects {VECTOR_DIMENSIONS}."
                    )
                source_name = str(record["source"])
                chunk_index = int(record["chunk_index"])
                document_id = _document_id(source_name, chunk_index)
                current_ids.add(document_id)
                documents.append(
                    {
                        "id": document_id,
                        "content": str(record["text"]),
                        "source": source_name,
                        "chunk_index": chunk_index,
                        VECTOR_FIELD: vector,
                    }
                )

            for document_batch in _batched(documents, SEARCH_UPLOAD_BATCH_SIZE):
                results = search_client.merge_or_upload_documents(
                    documents=list(document_batch)
                )
                _check_indexing_results(results)
                total_uploaded += len(document_batch)

        removed_count = _remove_stale_documents(search_client, source, current_ids)
        print(
            f"Indexed {len(current_ids)} chunks from {source}"
            + (f"; removed {removed_count} stale chunks." if removed_count else ".")
        )

    return total_uploaded


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    search_endpoint = _search_endpoint()
    search_api_key = _required_setting("AZURE_SEARCH_API_KEY")
    index_name = _required_setting("AZURE_SEARCH_INDEX_NAME")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,127}", index_name):
        raise ValueError(
            "AZURE_SEARCH_INDEX_NAME must be 1-128 characters using lowercase "
            "letters, digits, and hyphens, and must start with a letter or digit."
        )

    embeddings = _azure_embeddings()
    generated_chunks = chunk_markdown_files(embeddings)
    if generated_chunks == 0:
        raise ValueError("Semantic chunking produced no chunks; nothing was indexed.")

    chunk_dir = OUTPUT_DIR.resolve()
    try:
        chunk_dir.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError("Chunk output directory must stay inside the project.") from error

    chunk_files = sorted(chunk_dir.glob("*.jsonl"))
    if not chunk_files:
        raise FileNotFoundError(f"No JSONL chunk files found in {chunk_dir}")

    credential = AzureKeyCredential(search_api_key)
    index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)
    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=index_name,
        credential=credential,
    )
    try:
        _ensure_index(index_client, index_name)
        uploaded = _ingest_chunks(search_client, embeddings, chunk_files)
    finally:
        search_client.close()
        index_client.close()

    print(f"Finished Azure AI Search ingestion: {uploaded} chunks in '{index_name}'.")


if __name__ == "__main__":
    main()
