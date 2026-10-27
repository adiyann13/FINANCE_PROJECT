import argparse
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv

try:
    from .semantic_chunker import PROJECT_ROOT
except ImportError:
    from semantic_chunker import PROJECT_ROOT


DEFAULT_TOP_K = 5
MAX_TOP_K = 50


@dataclass(frozen=True)
class SearchResult:
    content: str
    source: str
    chunk_index: int
    score: float


def _required_setting(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required setting in .env: {name}")
    return value


def _search_client() -> SearchClient:
    load_dotenv(PROJECT_ROOT / ".env")

    endpoint = _required_setting("AZURE_SEARCH_ENDPOINT")
    index_name = _required_setting("AZURE_SEARCH_INDEX_NAME")
    api_key = _required_setting("AZURE_SEARCH_API_KEY")

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
            "AZURE_SEARCH_ENDPOINT must be the HTTPS Azure AI Search service URL."
        )

    return SearchClient(
        endpoint=endpoint.rstrip("/"),
        index_name=index_name,
        credential=AzureKeyCredential(api_key),
    )


class BM25Retriever:
    """Retrieve text-matched chunks ranked by Azure AI Search's BM25 scoring."""

    def __init__(self, search_client: SearchClient) -> None:
        self._search_client = search_client

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must not be empty.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be an integer from 1 to {MAX_TOP_K}.")

        response = self._search_client.search(
            search_text=query.strip(),
            query_type="simple",
            search_mode="any",
            search_fields=["content"],
            select=["content", "source", "chunk_index"],
            top=top_k,
        )

        results: list[SearchResult] = []
        for item in response:
            content = item.get("content")
            source = item.get("source")
            chunk_index = item.get("chunk_index")
            score: Any = item.get("@search.score")

            if not isinstance(content, str) or not isinstance(source, str):
                raise ValueError(
                    "Search result is missing the expected content or source field. "
                    "Check the Azure AI Search index schema."
                )
            if isinstance(chunk_index, bool) or not isinstance(chunk_index, int):
                raise ValueError(
                    "Search result has an invalid chunk_index. "
                    "Check the Azure AI Search index schema."
                )
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError("Azure AI Search returned a result without a valid BM25 score.")

            results.append(
                SearchResult(
                    content=content,
                    source=source,
                    chunk_index=chunk_index,
                    score=float(score),
                )
            )
        return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search the finance document index using BM25 keyword ranking."
    )
    parser.add_argument("query", help="Text to search for")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Maximum results to return (1-{MAX_TOP_K}; default: {DEFAULT_TOP_K})",
    )
    args = parser.parse_args()

    client = _search_client()
    try:
        results = BM25Retriever(client).retrieve(args.query, args.top_k)
    finally:
        client.close()

    if not results:
        print("No matching documents found.")
        return

    for rank, result in enumerate(results, start=1):
        print(
            f"\n[{rank}] {result.source} | chunk {result.chunk_index} "
            f"| BM25 score: {result.score:.4f}"
        )
        print(result.content)


if __name__ == "__main__":
    main()
