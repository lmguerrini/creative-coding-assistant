#!/usr/bin/env python3
"""Report read-only retrieval coverage for the canonical Capstone demo pack."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb

from creative_coding_assistant.core import load_settings
from creative_coding_assistant.eval import build_capstone_retrieval_demo_pack
from creative_coding_assistant.rag.retrieval import (
    KnowledgeBaseRetriever,
    build_query_embedder,
)
from creative_coding_assistant.vectorstore import ChromaCollection


class _ExistingCollectionsOnlyClient:
    """Adapt Chroma reads without allowing repository collection creation."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def get_or_create_collection(
        self,
        *,
        name: str,
        metadata: dict[str, object] | None = None,
    ) -> Any:
        del metadata
        return self._client.get_collection(name=name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the canonical seven-case retrieval demo pack against the configured "
            "local KB and print a JSON coverage report. The command reads the index "
            "and calls the configured query-embedding provider; it does not write "
            "benchmark or KB data."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        choices=range(1, 21),
        metavar="1..20",
        help="Retrieved chunks per canonical case (default: 5).",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact JSON instead of indented JSON.",
    )
    return parser


def main(argv: tuple[str, ...] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()
    query_embedder = build_query_embedder(settings)
    if query_embedder is None:
        raise SystemExit(
            "Canonical retrieval requires the configured query-embedding provider."
        )

    kb_path = Path(settings.chroma_persist_dir)
    if not kb_path.exists():
        raise SystemExit(f"Configured Chroma index does not exist: {kb_path}")

    client = chromadb.PersistentClient(path=str(kb_path))
    retriever = KnowledgeBaseRetriever(
        client=_ExistingCollectionsOnlyClient(client),
        embedder=query_embedder,
    )
    report = build_report(retriever=retriever, limit=args.limit)
    report["schemaVersion"] = "canonical-retrieval-report.v1"
    report["evaluatedAt"] = datetime.now(UTC).isoformat()
    report["embeddingModel"] = settings.openai_embedding_model
    report["kbSnapshot"] = build_kb_snapshot(
        client.get_collection(name=ChromaCollection.KB_OFFICIAL_DOCS.value)
    )
    report["selectionFingerprint"] = fingerprint_selection(report)
    print(
        json.dumps(
            report,
            indent=None if args.compact else 2,
            sort_keys=True,
        )
    )
    return 0


def build_report(*, retriever: KnowledgeBaseRetriever, limit: int) -> dict[str, Any]:
    pack = build_capstone_retrieval_demo_pack()
    cases: list[dict[str, Any]] = []
    total_expected_sources = 0
    total_covered_sources = 0
    total_requested_domains = 0
    total_covered_domains = 0

    for scenario in pack.scenarios:
        response = retriever.search(scenario.build_request(limit=limit))
        returned_source_ids = _unique(item.source_id for item in response.results)
        returned_domains = _unique(item.domain.value for item in response.results)
        covered_source_ids = tuple(
            source_id
            for source_id in scenario.expected_source_ids
            if source_id in returned_source_ids
        )
        requested_domains = tuple(domain.value for domain in scenario.domains)
        covered_domains = tuple(
            domain for domain in requested_domains if domain in returned_domains
        )

        total_expected_sources += len(scenario.expected_source_ids)
        total_covered_sources += len(covered_source_ids)
        total_requested_domains += len(requested_domains)
        total_covered_domains += len(covered_domains)
        cases.append(
            {
                "caseId": scenario.demo_id,
                "title": scenario.title,
                "requestedDomains": list(requested_domains),
                "returnedDomains": list(returned_domains),
                "coveredRequestedDomains": list(covered_domains),
                "domainCoverage": _coverage(
                    len(covered_domains),
                    len(requested_domains),
                ),
                "expectedSourceIds": list(scenario.expected_source_ids),
                "returnedSourceIds": list(returned_source_ids),
                "coveredExpectedSourceIds": list(covered_source_ids),
                "expectedSourceOverlap": _coverage(
                    len(covered_source_ids),
                    len(scenario.expected_source_ids),
                ),
                "resultCount": len(response.results),
                "results": [
                    {
                        "recordId": item.record_id,
                        "rank": rank,
                        "sourceId": item.source_id,
                        "domain": item.domain.value,
                        "documentTitle": item.document_title,
                        "chunkIndex": item.chunk_index,
                        "charCount": item.char_count,
                        "distance": round(item.distance, 6),
                        "score": round(item.score, 6),
                    }
                    for rank, item in enumerate(response.results, start=1)
                ],
            }
        )

    return {
        "benchmarkId": pack.pack_id,
        "benchmarkCaseCount": len(pack.scenarios),
        "retrievalLimit": limit,
        "interpretation": (
            "Expected source IDs are coverage anchors, not a requirement that every "
            "listed source appear in the top-k results."
        ),
        "summary": {
            "casesWithResults": sum(case["resultCount"] > 0 for case in cases),
            "expectedSourceOverlap": _coverage(
                total_covered_sources,
                total_expected_sources,
            ),
            "requestedDomainCoverage": _coverage(
                total_covered_domains,
                total_requested_domains,
            ),
        },
        "cases": cases,
    }


def _coverage(covered: int, expected: int) -> dict[str, int | float]:
    return {
        "covered": covered,
        "expected": expected,
        "ratio": round(covered / expected, 6) if expected else 0.0,
    }


def build_kb_snapshot(collection: Any) -> dict[str, int | str]:
    """Fingerprint non-text KB metadata so a report is bound to one index state."""

    result = collection.get(include=["metadatas"])
    ids = result.get("ids") or []
    metadatas = result.get("metadatas") or []
    rows = []
    for index, record_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) else {}
        rows.append(
            {
                "recordId": record_id,
                "sourceId": metadata.get("source_id"),
                "domain": metadata.get("domain"),
                "documentTitle": metadata.get("document_title"),
                "chunkIndex": metadata.get("chunk_index"),
                "charCount": metadata.get("char_count"),
                "contentHash": metadata.get("content_hash"),
                "chunkHash": metadata.get("chunk_hash"),
            }
        )
    return {
        "collection": ChromaCollection.KB_OFFICIAL_DOCS.value,
        "recordCount": len(rows),
        "metadataFingerprint": _fingerprint(
            sorted(rows, key=lambda row: str(row["recordId"]))
        ),
    }


def fingerprint_selection(report: dict[str, Any]) -> str:
    """Fingerprint benchmark inputs, selected lineage, and the KB snapshot."""

    return _fingerprint(
        _canonicalize_numbers(
            {
                "benchmarkId": report["benchmarkId"],
                "retrievalLimit": report["retrievalLimit"],
                "embeddingModel": report["embeddingModel"],
                "kbSnapshot": report["kbSnapshot"],
                "summary": report["summary"],
                "cases": report["cases"],
            }
        )
    )


def _canonicalize_numbers(value: object) -> object:
    """Make fingerprints stable when JSON readers collapse `1.0` into `1`."""

    if isinstance(value, dict):
        return {
            str(key): _canonicalize_numbers(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_numbers(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return format(value, ".15g")
    return value


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


if __name__ == "__main__":
    raise SystemExit(main())
