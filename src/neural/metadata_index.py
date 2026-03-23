from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from neural.vector_index import SearchResult

ENRICHED_CHUNKS_FILENAME = "enriched_chunks.json"
DOCUMENT_METADATA_FILENAME = "document_metadata.json"


def _normalize(value: str) -> str:
    return value.strip().casefold()


@dataclass(frozen=True, slots=True)
class ChunkKey:
    source_file: str
    start_timestamp: str
    end_timestamp: str


@dataclass(frozen=True, slots=True)
class ChunkMetadataRecord:
    source_file: str
    start_timestamp: str
    end_timestamp: str
    episode_type: str
    guest_names: tuple[str, ...]
    host_names: tuple[str, ...]
    episode_topic: str
    speaker: str
    topic: str
    subtopic: str
    mentioned_people: tuple[str, ...]
    mentioned_teams: tuple[str, ...]
    mentioned_leagues: tuple[str, ...]

    @property
    def key(self) -> ChunkKey:
        return ChunkKey(
            source_file=self.source_file,
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
        )


@dataclass(frozen=True, slots=True)
class DocumentMetadataRecord:
    source_file: str
    episode_type: str
    guest_names: tuple[str, ...]
    host_names: tuple[str, ...]
    topic: str
    holiday_theme: str


@dataclass(frozen=True, slots=True)
class MetadataIndex:
    chunk_records: dict[ChunkKey, ChunkMetadataRecord]
    document_records: dict[str, DocumentMetadataRecord]

    def metadata_for_result(self, result: SearchResult) -> ChunkMetadataRecord | None:
        key = ChunkKey(
            source_file=result.chunk.source_file,
            start_timestamp=result.chunk.start_timestamp,
            end_timestamp=result.chunk.end_timestamp,
        )
        return self.chunk_records.get(key)


@dataclass(frozen=True, slots=True)
class RetrievalFilters:
    episode_type: str | None = None
    guest_name: str | None = None
    speaker: str | None = None
    team: str | None = None
    topic: str | None = None
    source_file: str | None = None

    def active(self) -> bool:
        return any(
            value and value.strip()
            for value in (
                self.episode_type,
                self.guest_name,
                self.speaker,
                self.team,
                self.topic,
                self.source_file,
            )
        )


def load_metadata_index(metadata_dir: Path) -> MetadataIndex:
    chunk_path = metadata_dir / ENRICHED_CHUNKS_FILENAME
    document_path = metadata_dir / DOCUMENT_METADATA_FILENAME
    for artifact_path in (chunk_path, document_path):
        if not artifact_path.exists():
            msg = f"Missing metadata artifact: {artifact_path}"
            raise FileNotFoundError(msg)

    chunk_dicts = json.loads(chunk_path.read_text(encoding="utf-8"))
    document_dicts = json.loads(document_path.read_text(encoding="utf-8"))

    chunk_records: dict[ChunkKey, ChunkMetadataRecord] = {}
    for item in chunk_dicts:
        chunk_meta = item.get("chunk_meta") or {}
        llm_meta = item.get("llm_meta") or {}
        record = ChunkMetadataRecord(
            source_file=str(item.get("source_file", "")),
            start_timestamp=str(item.get("start_timestamp", "")),
            end_timestamp=str(item.get("end_timestamp", "")),
            episode_type=str(item.get("episode_type", "")),
            guest_names=tuple(str(v) for v in item.get("guest_names") or ()),
            host_names=tuple(str(v) for v in item.get("host_names") or ()),
            episode_topic=str(item.get("episode_topic", "")),
            speaker=str(llm_meta.get("speaker", "")),
            topic=str(llm_meta.get("topic", "")),
            subtopic=str(llm_meta.get("subtopic", "")),
            mentioned_people=tuple(str(v) for v in chunk_meta.get("mentioned_people") or ()),
            mentioned_teams=tuple(str(v) for v in chunk_meta.get("mentioned_teams") or ()),
            mentioned_leagues=tuple(str(v) for v in chunk_meta.get("mentioned_leagues") or ()),
        )
        chunk_records[record.key] = record

    document_records: dict[str, DocumentMetadataRecord] = {}
    for item in document_dicts:
        source_file = str(item.get("source_file", ""))
        document_records[source_file] = DocumentMetadataRecord(
            source_file=source_file,
            episode_type=str(item.get("episode_type", "")),
            guest_names=tuple(str(v) for v in item.get("guest_names") or ()),
            host_names=tuple(str(v) for v in item.get("host_names") or ()),
            topic=str(item.get("topic", "")),
            holiday_theme=str(item.get("holiday_theme", "")),
        )

    return MetadataIndex(chunk_records=chunk_records, document_records=document_records)


def filter_search_results(
    results: list[SearchResult],
    metadata_index: MetadataIndex,
    filters: RetrievalFilters,
) -> list[SearchResult]:
    if not filters.active():
        return results

    filtered: list[SearchResult] = []
    for result in results:
        source_file = result.chunk.source_file
        record = metadata_index.metadata_for_result(result)
        doc_record = metadata_index.document_records.get(source_file)
        if _matches_filters(result, record, doc_record, filters):
            filtered.append(result)
    return filtered


def metadata_to_citation(record: ChunkMetadataRecord | None) -> dict[str, object] | None:
    if record is None:
        return None
    return {
        "episode_type": record.episode_type,
        "guest_names": list(record.guest_names),
        "host_names": list(record.host_names),
        "episode_topic": record.episode_topic,
        "speaker": record.speaker,
        "topic": record.topic,
        "subtopic": record.subtopic,
        "mentioned_people": list(record.mentioned_people),
        "mentioned_teams": list(record.mentioned_teams),
        "mentioned_leagues": list(record.mentioned_leagues),
    }


def _contains_match(expected: str | None, values: tuple[str, ...]) -> bool:
    if not expected or not expected.strip():
        return True
    needle = _normalize(expected)
    return any(_normalize(value) == needle for value in values)


def _text_match(expected: str | None, *values: str) -> bool:
    if not expected or not expected.strip():
        return True
    needle = _normalize(expected)
    return any(needle in _normalize(value) for value in values if value)


def _matches_filters(
    result: SearchResult,
    record: ChunkMetadataRecord | None,
    doc_record: DocumentMetadataRecord | None,
    filters: RetrievalFilters,
) -> bool:
    if filters.source_file and _normalize(result.chunk.source_file) != _normalize(
        filters.source_file
    ):
        return False

    episode_type_values = tuple(
        value
        for value in (
            record.episode_type if record else "",
            doc_record.episode_type if doc_record else "",
        )
        if value
    )
    if filters.episode_type and not any(
        _normalize(value) == _normalize(filters.episode_type) for value in episode_type_values
    ):
        return False

    guest_names = tuple(record.guest_names if record else ()) + tuple(
        doc_record.guest_names if doc_record else ()
    )
    if not _contains_match(filters.guest_name, guest_names):
        return False

    if filters.speaker and record is None:
        return False
    if (
        filters.speaker
        and record is not None
        and _normalize(record.speaker) != _normalize(filters.speaker)
    ):
        return False

    team_values = record.mentioned_teams if record else ()
    if not _contains_match(filters.team, team_values):
        return False

    topic_values = (
        record.topic if record else "",
        record.subtopic if record else "",
        record.episode_topic if record else "",
        doc_record.topic if doc_record else "",
    )
    if not _text_match(filters.topic, *topic_values):
        return False

    return True
