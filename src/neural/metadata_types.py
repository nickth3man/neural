"""Controlled vocabularies and metadata dataclasses for transcript enrichment.

Defines the glossary-driven enums and tiered metadata structures from PLAN.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

# ============================================================================
# Controlled Vocabularies (Tier 1-3 enums)
# ============================================================================


class EpisodeType(StrEnum):
    """Show format classification from intro-window cues."""

    SEASON_PREMIERE = "season_premiere"
    RECAP = "recap"
    REACTION = "reaction"
    PREDICTION = "prediction"
    GUEST_INTERVIEW = "guest_interview"
    TRADE_BREAKDOWN = "trade_breakdown"
    PLAYOFF_RECAP = "playoff_recap"
    LEAGUE_NEWS = "league_news"


class ContentType(StrEnum):
    """Segment-level content classification."""

    AD_READ = "ad_read"
    COLD_OPEN_BANTER = "cold_open_banter"
    CAST_INTRO = "cast_intro"
    TOPIC_RUNDOWN = "topic_rundown"
    MAIN_DISCUSSION = "main_discussion"
    GUEST_INTERVIEW = "guest_interview"
    PROMO_READ = "promo_read"
    OUTRO = "outro"


class TopicCategory(StrEnum):
    """Constrained taxonomy for topic classification."""

    TEAM = "team"
    PLAYER = "player"
    LEAGUE = "league"
    EVENT = "event"
    BUSINESS = "business"
    SOCIAL_JUSTICE = "social_justice"
    SHOW_META = "show_meta"
    SPONSOR = "sponsor"
    HOLIDAY = "holiday"


class ConversationType(StrEnum):
    """Dialogue mode classification."""

    DEBATE = "debate"
    ANALYSIS = "analysis"
    REACTION = "reaction"
    BANTER = "banter"
    INTERVIEW = "interview"
    NEWS_ROUNDUP = "news_roundup"
    PROMOTION = "promotion"


class ClaimType(StrEnum):
    """Statement classification."""

    FACT = "fact"
    OPINION = "opinion"
    PREDICTION = "prediction"
    RUMOR = "rumor"
    ANECDOTE = "anecdote"
    PROMOTION = "promotion"


class Stance(StrEnum):
    """Speaker stance toward a topic."""

    SUPPORTIVE = "supportive"
    CRITICAL = "critical"
    MIXED = "mixed"
    SKEPTICAL = "skeptical"
    DESCRIPTIVE = "descriptive"


class SourceConfidence(StrEnum):
    """Confidence level for extracted metadata."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ============================================================================
# Anchor Types (Tier 1-3)
# ============================================================================


@dataclass(frozen=True)
class DocumentAnchor:
    """Anchor pointing to intro/setup text or manifest-derived evidence."""

    line_index: int
    timestamp: str
    excerpt: str


@dataclass(frozen=True)
class ChunkAnchor:
    """Anchor pointing to local chunk text evidence."""

    chunk_index: int
    start_timestamp: str
    excerpt: str


# Union type for anchors
Anchor = DocumentAnchor | ChunkAnchor


# ============================================================================
# Tier 1: Document-Level Metadata
# ============================================================================


@dataclass(frozen=True)
class DocumentMetadata:
    """Episode-level metadata extracted from intro windows and manifest.

    Populated primarily via intro-window parsing and manifest facts.
    Fields inherit document-level anchors for traceability.
    """

    show_title: str = "Gil's Arena"
    episode_type: EpisodeType = EpisodeType.REACTION
    host_names: tuple[str, ...] = ()
    guest_names: tuple[str, ...] = ()
    episode_date: str = ""
    episode_description: str = ""
    episode_url: str = ""
    url_slug: str = ""
    season: int | None = None
    episode_number: int | None = None
    holiday_theme: str = ""
    topic: str = ""
    subtopic: str = ""
    content_type: ContentType = ContentType.MAIN_DISCUSSION

    # Anchors for traceability
    episode_type_anchor: DocumentAnchor | None = None
    host_names_anchor: DocumentAnchor | None = None
    guest_names_anchor: DocumentAnchor | None = None
    holiday_theme_anchor: DocumentAnchor | None = None
    topic_anchor: DocumentAnchor | None = None

    # Roster built from intro for speaker constraint
    episode_roster: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""
        return asdict(self)


# ============================================================================
# Tier 2: Chunk-Level Deterministic Metadata
# ============================================================================


@dataclass(frozen=True)
class ChunkMetadata:
    """Chunk-level metadata from position, context, and controlled vocab.

    Inherits stable document context and uses controlled vocabularies.
    """

    chunk_index: int = 0
    adjacent_context_window: str = ""
    mentioned_people: tuple[str, ...] = ()
    mentioned_teams: tuple[str, ...] = ()
    mentioned_leagues: tuple[str, ...] = ()
    conversation_type: ConversationType = ConversationType.BANTER
    claim_type: ClaimType = ClaimType.OPINION
    content_type: ContentType = ContentType.MAIN_DISCUSSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""
        return asdict(self)


# ============================================================================
# Tier 3: LLM-Inferred Metadata
# ============================================================================


@dataclass(frozen=True)
class LlmMetadata:
    """LLM-inferred metadata from constrained extraction.

    Speaker must be chosen from the closed episode roster.
    """

    speaker: str = ""
    topic: str = ""
    subtopic: str = ""
    sentiment: str = ""
    stance: Stance = Stance.DESCRIPTIVE
    source_confidence: SourceConfidence = SourceConfidence.MEDIUM

    # Chunk-local anchor
    speaker_anchor: ChunkAnchor | None = None
    topic_anchor: ChunkAnchor | None = None
    sentiment_anchor: ChunkAnchor | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""
        return asdict(self)


# ============================================================================
# Combined Enriched Chunk
# ============================================================================


@dataclass(frozen=True)
class EnrichedChunk:
    """Fully enriched chunk combining base chunk data with all metadata tiers."""

    # Base chunk fields (from TranscriptChunk)
    episode_title: str
    source_file: str
    start_timestamp: str
    end_timestamp: str
    start_seconds: int
    end_seconds: int
    chunk_text: str
    line_count: int

    # Tier 2: Deterministic chunk metadata
    chunk_meta: ChunkMetadata = field(default_factory=ChunkMetadata)

    # Tier 3: LLM-inferred metadata
    llm_meta: LlmMetadata = field(default_factory=LlmMetadata)

    # Inherited document-level fields (flat for convenience)
    show_title: str = "Gil's Arena"
    episode_type: str = ""
    host_names: tuple[str, ...] = ()
    guest_names: tuple[str, ...] = ()
    episode_date: str = ""
    holiday_theme: str = ""
    episode_topic: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""
        return asdict(self)


# ============================================================================
# JSON Schema for LLM Structured Output
# ============================================================================

# Enums as lists for JSON schema constraints
EPISODE_TYPE_VALUES = [e.value for e in EpisodeType]
CONTENT_TYPE_VALUES = [e.value for e in ContentType]
TOPIC_CATEGORY_VALUES = [e.value for e in TopicCategory]
CONVERSATION_TYPE_VALUES = [e.value for e in ConversationType]
CLAIM_TYPE_VALUES = [e.value for e in ClaimType]
STANCE_VALUES = [e.value for e in Stance]
SOURCE_CONFIDENCE_VALUES = [e.value for e in SourceConfidence]


def document_extraction_schema() -> dict[str, Any]:
    """JSON schema for document-level glossary extraction prompt."""
    return {
        "type": "object",
        "properties": {
            "episode_type": {
                "type": "string",
                "enum": EPISODE_TYPE_VALUES,
                "description": "Show format classification from intro cues.",
            },
            "host_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of host names mentioned in cast intros.",
            },
            "guest_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Guest names explicitly introduced.",
            },
            "holiday_theme": {
                "type": "string",
                "description": (
                    "Holiday or theme framing (e.g., 'Halloween', 'Juneteenth') "
                    "or empty string."
                ),
            },
            "topic": {
                "type": "string",
                "description": "Primary topic seed from topic-rundown lines.",
            },
            "subtopic": {
                "type": "string",
                "description": "Subtopic seed or empty string.",
            },
            "topic_category": {
                "type": "string",
                "enum": TOPIC_CATEGORY_VALUES,
                "description": "Topic category taxonomy.",
            },
            "content_type": {
                "type": "string",
                "enum": CONTENT_TYPE_VALUES,
                "description": "Primary content type of the intro window.",
            },
            "episode_roster": {
                "type": "array",
                "items": {"type": "string"},
                "description": "All speaker names detected in intro/cast mentions.",
            },
            "sponsors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Sponsor names from ad reads in the intro window.",
            },
            "intro_excerpt": {
                "type": "string",
                "description": "Key excerpt from the intro window for anchoring.",
            },
        },
        "required": [
            "episode_type",
            "host_names",
            "guest_names",
            "holiday_theme",
            "topic",
            "subtopic",
            "topic_category",
            "content_type",
            "episode_roster",
            "sponsors",
            "intro_excerpt",
        ],
        "additionalProperties": False,
    }


def chunk_enrichment_schema(roster: list[str]) -> dict[str, Any]:
    """JSON schema for chunk-level enrichment extraction prompt."""
    return {
        "type": "object",
        "properties": {
            "speaker": {
                "type": "string",
                "enum": roster if roster else ["unknown"],
                "description": "Speaker attribution from closed episode roster.",
            },
            "conversation_type": {
                "type": "string",
                "enum": CONVERSATION_TYPE_VALUES,
                "description": "Dialogue mode classification.",
            },
            "claim_type": {
                "type": "string",
                "enum": CLAIM_TYPE_VALUES,
                "description": "Statement classification.",
            },
            "content_type": {
                "type": "string",
                "enum": CONTENT_TYPE_VALUES,
                "description": "Content type for this chunk.",
            },
            "topic": {
                "type": "string",
                "description": "Chunk-local topic if different from episode topic, else empty.",
            },
            "subtopic": {
                "type": "string",
                "description": "Chunk-local subtopic or empty.",
            },
            "mentioned_people": {
                "type": "array",
                "items": {"type": "string"},
                "description": "People mentioned in this chunk.",
            },
            "mentioned_teams": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Teams mentioned in this chunk.",
            },
            "mentioned_leagues": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Leagues mentioned in this chunk.",
            },
            "sentiment": {
                "type": "string",
                "description": "Overall sentiment of the chunk text.",
            },
            "stance": {
                "type": "string",
                "enum": STANCE_VALUES,
                "description": "Speaker stance toward the topic.",
            },
            "source_confidence": {
                "type": "string",
                "enum": SOURCE_CONFIDENCE_VALUES,
                "description": "Confidence level for the extraction.",
            },
            "speaker_evidence": {
                "type": "string",
                "description": "Key text excerpt supporting speaker attribution.",
            },
            "topic_evidence": {
                "type": "string",
                "description": "Key text excerpt supporting topic classification.",
            },
        },
        "required": [
            "speaker",
            "conversation_type",
            "claim_type",
            "content_type",
            "topic",
            "subtopic",
            "mentioned_people",
            "mentioned_teams",
            "mentioned_leagues",
            "sentiment",
            "stance",
            "source_confidence",
            "speaker_evidence",
            "topic_evidence",
        ],
        "additionalProperties": False,
    }


# ============================================================================
# Manifest metadata for episode context
# ============================================================================


@dataclass(frozen=True)
class EpisodeManifestEntry:
    """Manifest entry providing episode context for extraction."""

    title: str
    url_slug: str
    date: str
    description: str
    transcript_url: str
    file_path: str

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> EpisodeManifestEntry:
        """Create from a manifest JSON dict."""
        return cls(
            title=data.get("title", ""),
            url_slug=data.get("url_slug", ""),
            date=data.get("date", ""),
            description=data.get("description", ""),
            transcript_url=data.get("transcript_url", ""),
            file_path=data.get("file_path", ""),
        )
